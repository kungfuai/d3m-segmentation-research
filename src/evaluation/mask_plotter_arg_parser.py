from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MaskPlotterArgParser(ArgumentParser):
    """ parses MaskPlotter command-line options and arguments"""

    def __init__(self):
        super().__init__(
            description="plot/compare segmentation masks generated from experiment session",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        self.add_argument(
            "--seed", type=int, default=0, help="Seed for random number generators."
        )
        self.add_argument(
            "--experiment_dir", 
            type=str,
            help='directory containing subfolders pertaining to different experimental conditions' 
        )
        self.add_argument(
            "--n_examples",
            type=int,
            default=10,
            help="Number of segmentation masks to plot/compare",
        )
        self.add_argument(
            "--batch_no",
            type=int,
            default=1,
            help="Batch number to plot/compare (1-indexed)",
        )
        self.add_argument(
            "--test_records",
            type=str,
            default='data/prepped-binary/segmentation-test.tfrecord',
            help="TF Records file for test dataset",
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=1,
            help="Number of classes in segmentation model",
        )
        self.add_argument(
            "--tile_size",
            type=int,
            default=126,
            help="Size of Sentinel-2 image tiles (in pixels)",
        )
