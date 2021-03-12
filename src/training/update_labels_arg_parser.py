import os
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class UpdateLabelsArgParser(ArgumentParser):
    """Parses UpdateLabels command-line options and arguments."""

    def __init__(self):
        super().__init__(
            description="Update TFRecord files with pseudo labels from earlier training pass",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--update_records",
            type=str,
            default='data/prepped/',
            help="TF Records file to be updated",
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=5,
            help="Number of classes in segmentation model",
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Number of training examples used per batch",
        )
        self.add_argument(
            "--model_weights",
            type=str,
            default='logs/',
            help="Parameters of trained segmentation model",
        )
        self.add_argument(
            "--one_image_label",
            type=bool,
            default=False,
            help="If True - create model architecture with classification head",
        )
        self.add_argument(
            "--tile_size",
            type=int,
            default=126,
            help="Size of Sentinel-2 image tiles (in pixels)",
        )
        self.add_argument(
            "--calibrate",
            type=bool,
            default=False,
            help="If True - calibrates model predictions on validation set"
        )
        self.add_argument(
            "--calibration_temp",
            type=str,
            default='logs/',
            help="Temperature of trained calibration model - only used if calibrate == True",
        )
