import os
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class ExperimentSessionArgParser(ArgumentParser):
    """Parses ExperimentSession command-line options and arguments."""

    def __init__(self):
        super().__init__(
            prog="experiment.sh",
            description="Run experiments, varying training size and conditions",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--log_dir",
            type=str,
            default=os.path.join("logs/experiments", datetime.now().strftime("%Y%m%d-%H%M%S")),
            help="Directory where training progress and model checkpoints are saved.",
        )
        self.add_argument(
            "--data_dir",
            type=str,
            default='data/prepped',
            help="Folder with tf records files for train, validation, and test",
        )
        self.add_argument(
            "--training_sizes",
            type=int,
            default=10,
            nargs='+',
            help="Training sizes for experiments"
        )
        self.add_argument(
            "--conditions",
            type=str,
            default='full_pixel_mask',
            nargs='+',
            help="Conditions for experiments"
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
            default=128,
            help="Number of training examples used per batch",
        )
        self.add_argument(
            "--epochs_frozen",
            type=int,
            default=20,
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
            default=10,
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
            default=None,
            help="Parameters of trained classification model",
        )
        self.add_argument(
            "--loss_function",
            type=str,
            default='focal',
            help="Loss function to use for segmentation training - options are 'focal' and 'xent'",
        )
        self.add_argument(
            "--framework",
            type=str,
            default='tensorflow',
            help="Whether to run experiments with tensorflow (or pytorch) framework",
        )
        self.add_argument(
            "--num_bins",
            type=int,
            default=10,
            help="Number of bins to use in approximations of calibration errors",
        )
        self.add_argument(
            "--tile_size",
            type=int,
            default=126,
            help="Size of Sentinel-2 image tiles (in pixels)",
        )
        self.add_argument(
            "--data_parameters",
            type=bool,
            default=False,
            help="Whether to implement data parameters curriculum learning method from " + 
                "https://proceedings.neurips.cc/paper/2019/file/926ffc0ca56636b9e73c565cf994ea5a-Paper.pdf" ,
        )
        self.add_argument(
            "--calibrate",
            type=bool,
            default=False,
            help="If True - runs calibration module on validation set"
        )
        self.add_argument(
            "--duplicates",
            type=int,
            default=1,
            help="Number of times to run each size / condition combination with different data"
        )
        self.add_argument(
            "--super_loss",
            type=bool,
            default=False,
            help="Whether to implement super loss curriculum learning method from " + 
                "https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf" ,
        )
