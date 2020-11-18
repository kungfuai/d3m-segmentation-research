from argparse import ArgumentParser


class PredictionSessionArgParser(ArgumentParser):
    """Parses PredictionSession command-line options and arguments."""

    def __init__(self):
        super().__init__(prog="predict.sh", description="Run trained prediction model")

        self.add_argument(
            "--log_dir",
            type=str,
            default="logs/",
            help="Directory where log output is saved.",
        )
