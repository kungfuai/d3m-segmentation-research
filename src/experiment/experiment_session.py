import os
import logging

from src.experiment.experiment_session_arg_parser import ExperimentSessionArgParser
from src.training.segmentation_session import SegmentationSession
from src.training.segmentation_session_arg_parser import SegmentationSessionArgParser
from src.evaluation.evaluation_session import EvaluationSession
from src.evaluation.evaluation_session_arg_parser import EvaluationSessionArgParser

LOGGER = logging.getLogger(__name__)


class ExperimentSession:
    """Responsible for experiment setup and configuration."""

    def __init__(self, args):
        self.args = args

    def run(self):

        if not isinstance(self.args.training_sizes, list):
            self.args.training_sizes = [self.args.training_sizes]
        if not isinstance(self.args.conditions, list):
            self.args.conditions = [self.args.conditions]

        for i, train_size in enumerate(self.args.training_sizes):
            for j, condition in enumerate(self.args.conditions):
                train_args = self.set_train_args(train_size, condition)
                eval_args = self.set_eval_args(train_size, condition)

                n = i * len(self.args.conditions) + j + 1
                print(f'\nRunning training session {n} -- TRAIN SIZE = {train_size}, CONDITION = {condition}\n')
                training_session = SegmentationSession(train_args).run()

                print(f'\nRunning evaluation session {n} -- TRAIN SIZE = {train_size}, CONDITION = {condition}\n')
                evaluation_session = EvaluationSession(eval_args).run()

    def set_train_args(self, train_size, condition):

        train_args = SegmentationSessionArgParser().parse_args([])
        setattr(train_args, "seed", self.args.seed)
        setattr(
            train_args,
            "train_records",
            os.path.join(self.args.data_dir, f'segmentation-train-{train_size}.tfrecord')
        )
        setattr(
            train_args,
            "val_records",
            os.path.join(self.args.data_dir, f'segmentation-val.tfrecord')
        )
        setattr(train_args, "num_classes", self.args.num_classes)
        setattr(train_args, "batch_size", self.args.batch_size)
        setattr(train_args, "shuffle_buffer_size", train_size)
        setattr(train_args, "epochs_frozen", self.args.epochs_frozen)
        setattr(train_args, "epochs_unfrozen", self.args.epochs_unfrozen)
        setattr(train_args, "patience", self.args.patience)
        setattr(train_args, "workers", self.args.workers)
        setattr(train_args, "validation_steps", self.args.validation_steps)
        setattr(train_args, "encoder_weights", self.args.encoder_weights)

        log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}', 'train')
        os.makedirs(log_dir, exist_ok=True)
        setattr(train_args, "log_dir", log_dir)

        if condition == 'one_pixel_mask':
            setattr(train_args, "one_pixel_mask", True)
        elif condition == 'one_image_label':
            setattr(train_args, "one_image_label", True)
        return train_args

    def set_eval_args(self, train_size, condition):

        eval_args = EvaluationSessionArgParser().parse_args([])
        setattr(eval_args, "seed", self.args.seed)
        setattr(
            eval_args,
            "test_records",
            os.path.join(self.args.data_dir, f'segmentation-test.tfrecord')
        )
        setattr(eval_args, "num_classes", self.args.num_classes)
        setattr(eval_args, "batch_size", self.args.batch_size)
        
        model_weights = os.path.join(
            self.args.log_dir, 
            f'{train_size}-{condition}',
            'train',
            'model.h5'
        )
        setattr(eval_args, "model_weights", model_weights)

        log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}', 'eval')
        os.makedirs(log_dir, exist_ok=True)
        setattr(eval_args, "log_dir", log_dir)

        if condition == 'one_image_label':
            setattr(eval_args, "one_image_label", True)
        
        return eval_args


if __name__ == "__main__":
    args = ExperimentSessionArgParser().parse_args()
    session = ExperimentSession(args)
    session.run()
