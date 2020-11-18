import os
import random
import logging
import numpy as np
from src.prediction.predictor import Predictor
from src.prediction.prediction_session_arg_parser import PredictionSessionArgParser

LOGGER = logging.getLogger(__name__)


class PredictionSession:
    """Responsible for model prediction setup and configuration."""

    def __init__(self, args):
        self.args = args

    def run(self):
        self.seed_generators()
        self.configure_logging()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.create_predictor()
        self.run_predictor()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)

    def configure_logging(self):
        pass

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)

    def load_data(self):
        pass

    def create_model(self):
        pass

    def create_predictor(self):
        self.predictor = Predictor(self.model, self.args)

    def run_predictor(self):
        report = self.predictor.run()
        report.save(self.args.log_dir)


if __name__ == "__main__":
    args = PredictionSessionArgParser().parse_args()
    session = PredictionSession(args)
    session.run()
