import os
import random
import logging
import json

import numpy as np
import tensorflow as tf
import segmentation_models as sm
from sklearn.metrics import confusion_matrix

from src.evaluation.evaluation_session_arg_parser import EvaluationSessionArgParser
from src.dataset.segmentation_dataset import SegmentationDataset
from src.model.unet import Unet

LOGGER = logging.getLogger(__name__)


class EvaluationSession:
    """Responsible for evaluation setup and configuration."""

    def __init__(self, args):
        self.args = args

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.evaluate()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.random.set_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)

    def load_data(self):
        self.test_dataset = SegmentationDataset(
            self.args.test_records,
            nb_class=self.args.num_classes,
            batch_size=self.args.batch_size,
            shuffle_buffer_size=0
        ).dataset

    def create_model(self):
        if self.args.num_classes > 1:
            activation='softmax'
        else:
            activation='sigmoid'
        self.model = Unet(
            input_shape=(128, 128, 10), 
            classes=self.args.num_classes,
            weights=self.args.model_weights,
            one_image_label=self.args.one_image_label,
            activation=activation
        )

        if self.args.one_image_label:
            features = self.model.get_layer('decoder_stage4b_relu').output
            output = self.model.get_layer('final_fc')(features) 
            output = tf.keras.layers.Activation(activation)(output)

            self.model = tf.keras.Model(
                inputs=self.model.input,
                outputs=output
            )

    def evaluate(self):

        metrics = {
            'accuracy': [],
            #'iou_score': [],
        }

        if self.args.num_classes > 1:
            acc_metric = tf.keras.metrics.CategoricalAccuracy()
            conf_c = self.args.num_classes
        else:
            acc_metric = tf.keras.metrics.BinaryAccuracy()
            conf_c = 2

        batch_sizes = []
        confusion = np.zeros((conf_c, conf_c))
        for batch in self.test_dataset:
            batch_sizes.append(batch[1].shape[0])
            preds = self.model.predict_on_batch(batch)
                        
            acc_metric.update_state(batch[1], preds)
            acc = acc_metric.result().numpy()
            acc_metric.reset_states()
            #iou = sm.metrics.iou_score(batch[1], preds).numpy()

            metrics['accuracy'].append(acc)
            #metrics['iou_score'].append(iou)

            if self.args.num_classes > 1:
                gt = tf.math.argmax(batch[1], 3)
                p = tf.math.argmax(preds, 3)
            else:
                gt = batch[1]
                p = tf.math.round(preds)
            
            gt = tf.reshape(gt[:, 1:-1, 1:-1], [-1])
            p = tf.reshape(p[:, 1:-1, 1:-1], [-1])

            confusion += confusion_matrix(gt, p, labels=np.arange(conf_c))

        metrics = {
            metric: np.average(vals, weights=batch_sizes) 
            for metric, vals in metrics.items()
        }

        with open(os.path.join(self.args.log_dir, "metrics.json"), "w") as f:  
            json.dump(metrics, f) 

        confusion /= confusion.sum(axis=1)[:, np.newaxis]

        np.savetxt(
            os.path.join(self.args.log_dir, "confusion.csv"), 
            confusion, 
            delimiter=","
        )


if __name__ == "__main__":
    args = EvaluationSessionArgParser().parse_args()
    session = EvaluationSession(args)
    session.run()
