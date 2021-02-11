from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MeasureVariationArgParser(ArgumentParser):
    """ parses MeasureVariation command-line options and arguments"""

    def __init__(self):
        super().__init__(
            description="measures variation in metrics over multiple replications of experiments " + 
                        "with different datasets",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--experiment_dir", 
            type=str,
            help='directory containing subfolders pertaining to different experimental conditions' 
        )