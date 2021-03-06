from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class ProxyCorrelationPlotterArgParser(ArgumentParser):
    """ parses ProxyCorrelationPlotter command-line options and arguments"""

    def __init__(self):
        super().__init__(
            description="plot correlation between proxy task and segmentation task for different proxy tasks",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--experiment_dirs", 
            type=str,
            help='directory containing subfolders pertaining to different experimental conditions',
            nargs='+',
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=1,
            help="Number of classes in segmentation model",
        )
        self.add_argument(
            "--framework",
            type=str,
            default='tensorflow',
            help="Whether to run experiments with tensorflow (or pytorch) framework",
        )