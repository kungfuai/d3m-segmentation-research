import os
import random
import logging

import numpy as np
import tensorflow as tf
import segmentation_models as sm

from src.training.segmentation_session_arg_parser import SegmentationSessionArgParser
from src.dataset.segmentation_dataset import SegmentationDataset
from src.model.unet import Unet

LOGGER = logging.getLogger(__name__)


class SegmentationSession:
    """Responsible for segmentation model setup and configuration."""

    def __init__(self, args):
        self.args = args

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.compile_model()

        # freeze encoder
        epochs_elapsed = self.train(self.args.epochs_frozen) 

        # unfreeze encoder
        for layer in self.model.layers:
            layer.trainable = True
        
        self.train(
            self.args.epochs_unfrozen + epochs_elapsed, 
            initial_epoch=epochs_elapsed
        )

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.random.set_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)

    def load_data(self):
        segmentation = SegmentationDataset(
            self.args.train_records,
            one_pixel_mask=self.args.one_pixel_mask,
            one_image_label=self.args.one_image_label,
            loss_function=self.args.loss_function,
            nb_class=self.args.num_classes,
            batch_size=self.args.batch_size,
            shuffle_buffer_size=self.args.shuffle_buffer_size
        )

        self.train_dataset = segmentation.dataset
        self.class_weights = segmentation.class_weights

        if self.args.val_records:
            self.val_dataset = SegmentationDataset(
                self.args.val_records,
                one_pixel_mask=self.args.one_pixel_mask,
                one_image_label=self.args.one_image_label,
                loss_function=self.args.loss_function,
                nb_class=self.args.num_classes,
                batch_size=self.args.batch_size,
                shuffle_buffer_size=0 # don't shuffle during validation
            ).dataset
        else:
            self.val_dataset = None

    def create_model(self):
        self.model = Unet(
            input_shape=(128, 128, 10), 
            classes=self.args.num_classes,
            encoder_weights=self.args.encoder_weights,
            encoder_freeze=True,
            one_image_label=self.args.one_image_label,
        )

    def compile_model(self):
        if self.args.loss_function == 'focal':
            loss = sm.losses.categorical_focal_loss
            self.class_weights = None
        elif self.args.loss_function == 'xent':
            if self.args.one_image_label:
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            else:
                loss = sm.losses.CategoricalCELoss(class_weights=self.class_weights)
                self.class_weights = None
        else:
            raise ValueError("'loss_function' must be one of 'focal' or 'xent'")

        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=[
                'categorical_accuracy', 
            ], 
        )

    def train(self, epochs, initial_epoch=0):
        history = self.model.fit(
            self.train_dataset,
            initial_epoch=initial_epoch,
            epochs=epochs,
            class_weight=self.class_weights,
            validation_data=self.val_dataset,
            validation_steps=self.args.validation_steps,
            callbacks=[
                tf.keras.callbacks.CSVLogger(
                    os.path.join(self.args.log_dir, 'metrics.csv'),
                    append=True
                ), 
                tf.keras.callbacks.EarlyStopping(patience=self.args.patience), 
                tf.keras.callbacks.ReduceLROnPlateau(), 
                tf.keras.callbacks.TensorBoard(self.args.log_dir),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.args.log_dir, 'model.h5'),
                    save_best_only=True,
                    save_weights_only=True
                )
            ],
            workers=self.args.workers,
            use_multiprocessing=True
        )
        epochs_elapsed = len(history.history['loss'])
        return epochs_elapsed

if __name__ == "__main__":
    args = SegmentationSessionArgParser().parse_args()
    session = SegmentationSession(args)
    session.run()
