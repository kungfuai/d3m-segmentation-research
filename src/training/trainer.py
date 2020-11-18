import logging
from src.training.training_report import TrainingReport

LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def run(self):
        report = TrainingReport()
        return report
