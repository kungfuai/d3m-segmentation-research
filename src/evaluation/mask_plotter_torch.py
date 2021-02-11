import logging
import os 
import random
import json
from functools import partial

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset

from src.evaluation.mask_plotter_arg_parser import MaskPlotterArgParser
from src.evaluation.mask_plotter import MaskPlotter
from src.evaluation.metric_plotter import MetricPlotter
from src.dataset.segmentation_dataset_torch import preprocess
from src.model.unet_torch import Unet, SegmentationHeadImageLabelEval

LOGGER = logging.getLogger(__name__)


class MaskPlotterTorch(MaskPlotter):
    """ plots segmentation masks generated from experiment session"""

    def __init__(self, args):
        super().__init__(args)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        super().run()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def create_directories(self):
        super().create_directories()

    def load_data(self):
        test_dataset_raw = TFRecordDataset(
            self.args.test_records,
            index_path=None,
            shuffle_queue_size=0,
        )
        test_dataset_raw = iter(torch.utils.data.DataLoader(
            test_dataset_raw,
            batch_size=self.args.n_examples,
        ))

        for _ in range(self.args.batch_no):
            img_dict = next(test_dataset_raw)

        size = (self.args.n_examples, self.args.tile_size, self.args.tile_size)
        imgs = np.stack(
            [
                img_dict['B04'].reshape(size), 
                img_dict['B03'].reshape(size), 
                img_dict['B02'].reshape(size)
            ],
            axis=-1
        )
        self.imgs = (imgs / 4096) ** 0.5 * 255
        self.labels = [img_dict['Corine_labels'].reshape(size)]

        test_dataset = TFRecordDataset(
            self.args.test_records,
            index_path=None,
            shuffle_queue_size=0,
            transform=partial(
                preprocess,
                tile_size=self.args.tile_size
            )
        )
        self.test_dataset = iter(torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.n_examples,
        ))

    def create_models(self):

        self.models = {}
        for f in sorted(os.listdir(self.args.experiment_dir)):
            d = os.path.join(self.args.experiment_dir, f)
            if os.path.isdir(d) and d.split('/')[-1] not in [
                'metrics', 
                'confusion-matrices',
                'calibration-plots'
            ]:
                dataset_size, condition = MetricPlotter.parse_dir_name(d)
                one_image_label = (condition=='one_image_label')

                if self.args.num_classes > 1:
                    activation='softmax'
                else:
                    activation='sigmoid'

                model = Unet(
                    encoder_freeze=False,
                    one_image_label=one_image_label,
                    device=self.device
                ).to(self.device)

                model.load_state_dict(
                    torch.load(os.path.join(d, 'train', 'model.pth'))
                )

                if one_image_label:

                    model.segmentation_head = SegmentationHeadImageLabelEval(
                        model.segmentation_head
                    )

                if dataset_size in self.models.keys():
                    self.models[dataset_size].append(model)
                else:
                    self.models[dataset_size] = [model]

        self.model_types = len(self.models[dataset_size])
        
    def make_predictions(self):
        
        for _ in range(self.args.batch_no):
            batch = next(self.test_dataset)
            imgs = batch[0].to(self.device)

        self.masks = []
        pad = (128 - self.args.tile_size) // 2
        for _, models in self.models.items():
            preds = [torch.sigmoid(model.predict(imgs)) for model in models]
            preds = [p.detach().cpu().numpy().squeeze() for p in preds]
            preds = [np.round(p[:, pad:-pad, pad:-pad]) for p in preds]
            masks = np.stack(self.labels + preds) 
            self.masks.append(masks)
        self.masks = np.stack(self.masks)

    def plot(self):
        super().plot()


if __name__ == "__main__":
    args = MaskPlotterArgParser().parse_args()
    plotter = MaskPlotterTorch(args)
    plotter.run()