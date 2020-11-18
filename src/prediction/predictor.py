import logging
from src.prediction.prediction_report import PredictionReport

LOGGER = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def run(self):
        report = PredictionReport()
        return report
