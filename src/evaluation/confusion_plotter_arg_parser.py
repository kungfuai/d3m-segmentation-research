from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class ConfusionPlotterArgParser(ArgumentParser):
    """ parses ConfusionPlotter command-line options and arguments"""

    def __init__(self):
        super().__init__(
            description="plot confusion matrices from an experiment session",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--experiment_dir", 
            type=str,
            help='directory containing subfolders pertaining to different experimental conditions' 
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=5,
            help="Number of classes in segmentation model",
        )