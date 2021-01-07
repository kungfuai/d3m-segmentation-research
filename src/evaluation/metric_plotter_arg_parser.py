from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MetricPlotterArgParser(ArgumentParser):
    """ parses MetricPlotter command-line options and arguments"""

    def __init__(self):
        super().__init__(
            description="plot metrics generated from experiment session",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--experiment_dir", 
            type=str,
            help='directory containing subfolders pertaining to different experimental conditions' 
        )