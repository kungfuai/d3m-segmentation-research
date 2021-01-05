import os
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class EvaluationSessionArgParser(ArgumentParser):
    """Parses EvaluationSession command-line options and arguments."""

    def __init__(self):
        super().__init__(
            prog="evaluate.sh",
            description="Evaluate segmentation model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--log_dir",
            type=str,
            default=os.path.join("logs/evaluation", datetime.now().strftime("%Y%m%d-%H%M%S")),
            help="Directory where training progress and model checkpoints are saved.",
        )
        self.add_argument(
            "--test_records",
            type=str,
            default='data/prepped/',
            help="TF Records file for test dataset",
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
