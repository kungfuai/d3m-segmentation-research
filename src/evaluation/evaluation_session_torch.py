import os
import random
import logging
import json
from functools import partial

import numpy as np
import pandas as pd
import torch
from tfrecord.torch.dataset import TFRecordDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 

from src.dataset.segmentation_dataset_torch import preprocess
from src.model.unet_torch import Unet, SegmentationHeadImageLabelEval
from src.model.calibration_model import CalibrationModel
from src.evaluation.evaluation_session_arg_parser import EvaluationSessionArgParser
from src.evaluation.metric_plotter import MetricPlotter

LOGGER = logging.getLogger(__name__)

class EvaluationSessionTorch:
    """Responsible for evaluation setup and configuration."""

    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            torch.manual_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.calibration_dir = os.path.join(
            '/'.join(self.args.log_dir.split('/')[:-2]),
            'calibration-plots'
        )
        os.makedirs(self.calibration_dir, exist_ok=True)

    def load_data(self):

        test_dataset = TFRecordDataset(
            self.args.test_records,
            index_path=None,
            shuffle_queue_size=0,
            transform=partial(
                preprocess,
                tile_size=self.args.tile_size,
                estonia_data=self.args.estonia_data
            )
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
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

    def evaluate(self):

        accs = []
        batch_sizes = []
        all_labels = []
        all_preds = []
        confusion = np.zeros((2,2))
        for batch in self.test_loader:
            inputs = batch[0].to(self.device)
            labels = batch[1].squeeze().numpy()

            logits = self.model.predict(inputs)
            if self.args.calibrate:
                logits = self.calibration_model(logits, self.device)
            preds = torch.sigmoid(logits)

            preds = preds.detach().cpu().numpy().squeeze()
            batch_sizes.append(labels.shape[0])

            pad = (128 - self.args.tile_size) // 2
            gt = labels[:, pad:-pad, pad:-pad].flatten()
            p = preds[:, pad:-pad, pad:-pad].flatten()

            acc = np.sum((np.round(p) == gt)) / gt.shape[0]
            accs.append(acc)

            all_labels.append(gt)
            all_preds.append(p)

            confusion += confusion_matrix(gt, np.round(p), labels=np.arange(2))

        max_cal_e, exp_cal_e = self.calibration(all_labels, all_preds)

        metrics = {
            "accuracy": np.average(accs, weights=batch_sizes),
            "max_calibration_error": max_cal_e,
            "expected_calibration_error": exp_cal_e
        }

        with open(os.path.join(self.args.log_dir, "metrics.json"), "w") as f:  
            json.dump(metrics, f) 

        confusion /= confusion.sum(axis=1)[:, np.newaxis]

        np.savetxt(
            os.path.join(self.args.log_dir, "confusion.csv"), 
            confusion, 
            delimiter=","
        )

    def calibration(self, all_labels, all_preds):
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        bin_width = 1 / self.args.num_bins
        bins = np.arange(0, 1, bin_width)
        bin_inds = np.digitize(preds, bins)
        
        bin_preds = [
            preds[np.where(bin_inds == i)[0]]
            for i in range(1, self.args.num_bins + 1)
        ]
        bin_labels = [
            labels[np.where(bin_inds == i)[0]]
            for i in range(1, self.args.num_bins + 1)
        ]
        bin_sizes = [len(preds) for preds in bin_preds]

        bin_confs = [np.mean(preds) for preds in bin_preds]
        bin_freqs = [
            labels.sum() / size
            for labels, size in zip(bin_labels, bin_sizes)
        ]
        diffs = [
            abs(freq - conf)
            for freq, conf in zip(bin_freqs, bin_confs)
        ]
        weighted_diffs = [
            w / np.sum(bin_sizes) * val
            for w, val in zip(bin_sizes, diffs)
        ]

        max_cal_e = np.nanmax(diffs)
        exp_cal_e = np.nansum(weighted_diffs)

        self.plot_calibration(bins, bin_confs, bin_freqs)

        return max_cal_e, exp_cal_e

    def plot_calibration(self, bins, confs, freqs):

        data_obs = pd.DataFrame({
            'confidence': confs,
            'observed frequency': freqs,
            'calibration': 'model'
        })
        data_perfect = pd.DataFrame({
            'confidence': bins,
            'observed frequency': bins,
            'calibration': 'perfect'
        })
        data = pd.concat([data_obs, data_perfect])
        data = data.append(
            {
                'confidence': 1,
                'observed frequency': 1,
                'calibration': 'perfect'
            },
            ignore_index=True
        )

        plt.clf()
        sns.lineplot(
            x="confidence", 
            y="observed frequency",
            hue='calibration',
            data=data
        )
        dir_str = self.args.log_dir.split('/')[-2]

        if len(dir_str.split('-')) == 2:
            ds_size, condition = dir_str.split('-')
            plt.title(f'Model Calibration: {condition} - {ds_size} images')
            plt.savefig(os.path.join(
                self.calibration_dir, f'calibration-plot-{ds_size}-{condition}.png'
            ))
        else:
            dir_strs = dir_str.split('-')
            ds_size = dir_strs[0]
            condition = dir_strs[1]
            k = dir_strs[2]
            plt.title(f'Model Calibration: {condition} - {ds_size} images - {k}')
            plt.savefig(os.path.join(
                self.calibration_dir, f'calibration-plot-{ds_size}-{condition}-{k}.png'
            )) 

if __name__ == "__main__":
    args = EvaluationSessionArgParser().parse_args()
    session = EvaluationSessionTorch(args)
    session.run()