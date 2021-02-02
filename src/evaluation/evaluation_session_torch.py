import os
import random
import logging
import json

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from sklearn.metrics import confusion_matrix

from src.evaluation.evaluation_session_arg_parser import EvaluationSessionArgParser
from src.dataset.segmentation_dataset_torch import preprocess
from src.model.unet_torch import Unet, SegmentationHeadImageLabelEval

LOGGER = logging.getLogger(__name__)

class EvaluationSessionTorch:
    """Responsible for evaluation setup and configuration."""

    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.evaluate()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)

    def load_data(self):

        test_dataset = TFRecordDataset(
            self.args.test_records,
            index_path=None,
            shuffle_queue_size=0,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
        )

    def create_model(self):
        
        self.model = Unet(
            encoder_freeze=False,
            one_image_label=self.args.one_image_label,
            device=self.device
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(self.args.model_weights)
        )

        if self.args.one_image_label:

            self.model.segmentation_head = SegmentationHeadImageLabelEval(
                self.model.segmentation_head
            )

    def evaluate(self):

        accs = []
        batch_sizes = []
        confusion = np.zeros((2,2))
        for batch in self.test_loader:
            inputs = batch[0].to(self.device)
            labels = batch[1].squeeze().numpy()
            preds = self.model.predict(inputs)
            preds = np.round(preds.detach().cpu().numpy().squeeze())
            batch_sizes.append(labels.shape[0])

            acc = np.sum((preds == labels)) / labels.shape[0]
            acc /= (labels.shape[2] ** 2)
            accs.append(acc)

            gt = labels[:, 1:-1, 1:-1].flatten()
            p = preds[:, 1:-1, 1:-1].flatten()

            confusion += confusion_matrix(gt, p, labels=np.arange(2))

        metrics = {"accuracy": np.average(accs, weights=batch_sizes)}

        with open(os.path.join(self.args.log_dir, "metrics.json"), "w") as f:  
            json.dump(metrics, f) 

        confusion /= confusion.sum(axis=1)[:, np.newaxis]

        np.savetxt(
            os.path.join(self.args.log_dir, "confusion.csv"), 
            confusion, 
            delimiter=","
        )


if __name__ == "__main__":
    args = EvaluationSessionArgParser().parse_args()
    session = EvaluationSessionTorch(args)
    session.run()