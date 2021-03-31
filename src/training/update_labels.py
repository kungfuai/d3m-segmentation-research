import os
import random
import logging
from functools import partial

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
import tensorflow as tf

from src.dataset.segmentation_dataset_torch import preprocess
from src.model.unet_torch import Unet, SegmentationHeadImageLabelEval
from src.model.calibration_model import CalibrationModel
from src.training.update_labels_arg_parser import UpdateLabelsArgParser

LOGGER = logging.getLogger(__name__)

class UpdateLabels:
    """Responsible for updating TFRecord files with pseudo labels from earlier training pass."""

    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.seed_generators()
        self.load_data()
        self.create_model()
        self.generate_labels()
        self.write_tfrecords()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def load_data(self):

        self.original_dataset = TFRecordDataset(
            self.args.update_records,
            index_path=None,
            shuffle_queue_size=0
        )

        self.input_dataset = TFRecordDataset(
            self.args.update_records,
            index_path=None,
            shuffle_queue_size=0,
            transform=partial(
                preprocess,
                tile_size=self.args.tile_size,
                estonia_data=self.args.estonia_data
            )
        )

        self.loader = torch.utils.data.DataLoader(
            self.input_dataset,
            batch_size=self.args.batch_size,
        )

    def create_model(self):
        
        self.model = Unet(
            encoder_freeze=False,
            one_image_label=self.args.one_image_label,
            device=self.device
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(self.args.model_weights, map_location=self.device)
        )

        if self.args.one_image_label:

            self.model.segmentation_head = SegmentationHeadImageLabelEval(
                self.model.segmentation_head
            )

        if self.args.calibrate:
            self.calibration_model = CalibrationModel()
            self.calibration_model.load_state_dict(
                torch.load(self.args.calibration_temp, map_location=self.device)
            )

    def generate_labels(self):

        all_preds = []
        for batch in self.loader:
            inputs = batch[0].to(self.device)
            logits = self.model.predict(inputs)
            if self.args.calibrate:
                logits = self.calibration_model(logits)
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu().numpy().squeeze()

            pad = (128 - self.args.tile_size) // 2
            p = preds[:, pad:-pad, pad:-pad]
            all_preds.append(p)

        self.preds = np.vstack(all_preds)

    def write_tfrecords(self):

        keys = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        keys += ['Corine_labels', 'random_pixel', 'class_label', 'sample_index']

        out_segments = self.args.update_records.split('/')
        out_file = out_segments[-1].split('.')[0] + '-pseudo.tfrecord'
        out_path = '/'.join(out_segments[:-1] + [out_file])
        writer = tf.io.TFRecordWriter(out_path, options='')

        progress_bar = tf.keras.utils.Progbar(target = len(self.preds))
        for i, (record, preds) in enumerate(zip(self.original_dataset, self.preds)):
            feature_dict = {
                key: tf.train.Feature(int64_list=tf.train.Int64List(value=record[key]))
                for key in keys
            }
            feature_dict['Corine_labels'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=np.ravel(preds))
            )
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

            writer.write(example.SerializeToString())
            progress_bar.update(i)
  

if __name__ == "__main__":
    args = UpdateLabelsArgParser().parse_args()
    UpdateLabels(args).run()