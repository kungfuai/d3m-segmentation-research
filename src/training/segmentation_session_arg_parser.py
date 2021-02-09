import os
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class SegmentationSessionArgParser(ArgumentParser):
    """Parses SegmentationSession command-line options and arguments."""

    def __init__(self):
        super().__init__(
            prog="train-segmentation.sh",
            description="Train segmentation model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--log_dir",
            type=str,
            default=os.path.join("logs/segmentation", datetime.now().strftime("%Y%m%d-%H%M%S")),
            help="Directory where training progress and model checkpoints are saved.",
        )
        self.add_argument(
            "--train_records",
            type=str,
            default='data/prepped/',
            help="TF Records file for train dataset",
        )
        self.add_argument(
            "--val_records",
            type=str,
            default='data/prepped/',
            help="TF Records file for validation dataset",
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
            "--shuffle_buffer_size",
            type=int,
            default=5000,
            help="Shuffle buffer size for training dataset",
        )
        self.add_argument(
            "--epochs_frozen",
            type=int,
            default=10,
            help="Number of full training passes with frozen encoder weights",
        )
        self.add_argument(
            "--epochs_unfrozen",
            type=int,
            default=100,
            help="Number of full training passes with entire model unfrozen",
        )
        self.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Number of epochs without improvement after which training stops",
        )
        self.add_argument(
            "--workers",
            type=int,
            default=8,
            help="Number of workers",
        )
        self.add_argument(
            "--validation_steps",
            type=int,
            default=10,
            help="Number of validation steps",
        )
        self.add_argument(
            "--encoder_weights",
            type=str,
            default='logs/',
            help="Parameters of trained classification model",
        )
        self.add_argument(
            "--one_pixel_mask",
            type=bool,
            default=False,
            help="If True - train with one pixel supervision per image",
        )
        self.add_argument(
            "--one_image_label",
            type=bool,
            default=False,
            help="If True - train with one image-level label supervision per image",
        )
        self.add_argument(
            "--loss_function",
            type=str,
            default='focal',
            help="Loss function to use for segmentation training - options are 'focal' and 'xent'",
        )
        self.add_argument(
            "--tile_size",
            type=int,
            default=126,
            help="Size of Sentinel-2 image tiles (in pixels)",
        )
