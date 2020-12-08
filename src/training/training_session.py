import os
import random
import logging
import numpy as np
import tensorflow as tf

from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.dataset.big_earth_dataset import BigEarthDataset
from src.model.resnet50 import ResNet50

LOGGER = logging.getLogger(__name__)


class TrainingSession:
    """Responsible for model training setup and configuration."""

    def __init__(self, args):
        self.args = args

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.compile_model()
        self.train()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.random.set_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)

    def load_data(self):
        big_earth = BigEarthDataset(
            self.args.train_records,
            nb_class=self.args.num_classes,
            batch_size=self.args.batch_size,
            shuffle_buffer_size=self.args.shuffle_buffer_size
        )
        self.train_dataset = big_earth.dataset
        self.class_weights = big_earth.class_weights

        if self.args.val_records:
            self.val_dataset = BigEarthDataset(
                self.args.val_records,
                nb_class=self.args.num_classes,
                batch_size=self.args.batch_size,
                shuffle_buffer_size=0 # don't shuffle during validation
            ).dataset
        else:
            self.val_dataset = None

    def create_model(self):
        self.model = ResNet50(
            input_shape = (120, 120, 10), 
            classes = self.args.num_classes
        )

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

    def train(self):
        self.model.fit(
            self.train_dataset,
            class_weight=self.class_weights,
            epochs=self.args.epochs,
            validation_data=self.val_dataset,
            callbacks=[
                tf.keras.callbacks.CSVLogger(os.path.join(self.args.log_dir, 'metrics.csv')), 
                tf.keras.callbacks.EarlyStopping(patience=self.args.patience), 
                tf.keras.callbacks.ReduceLROnPlateau(), 
                tf.keras.callbacks.TensorBoard(self.args.log_dir),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.args.log_dir, 'model.ckpt'),
                    save_best_only=True
                )
            ],
            workers=self.args.workers,
            use_multiprocessing=True
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
