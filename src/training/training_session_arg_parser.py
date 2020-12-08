import os
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class TrainingSessionArgParser(ArgumentParser):
    """Parses TrainingSession command-line options and arguments."""

    def __init__(self):
        super().__init__(
            prog="train.sh",
            description="Train model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--log_dir",
            type=str,
            default=os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
            help="Directory where training progress and model checkpoints are saved.",
        )
        self.add_argument(
            "--train_records",
            type=str,
            default='data/prepped/dev-train.tfrecord',
            help="TF Records file for train dataset",
        )
        self.add_argument(
            "--val_records",
            type=str,
            default='data/prepped/dev-test.tfrecord',
            help="TF Records file for validation dataset",
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=5,
            help="Number of training examples used per batch.",
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=32,#1000,
            help="Number of training examples used per batch.",
        )
        self.add_argument(
            "--shuffle_buffer_size",
            type=int,
            default=50,#130000,
            help="Number of training examples used per batch.",
        )
        self.add_argument(
            "--epochs",
            type=int,
            default=5,#100,
            help="Number of full training passes over the entire dataset.",
        )
        self.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Number of epochs without improvement after which training stops.",
        )
        self.add_argument(
            "--workers",
            type=int,
            default=8,
            help="Number of epochs without improvement after which training stops.",
        )
