import os
import logging

from src.experiment.experiment_session_arg_parser import ExperimentSessionArgParser
from src.training.segmentation_session import SegmentationSession
from src.training.segmentation_session_torch import SegmentationSessionTorch
from src.training.segmentation_session_arg_parser import SegmentationSessionArgParser
from src.evaluation.evaluation_session import EvaluationSession
from src.evaluation.evaluation_session_torch import EvaluationSessionTorch
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
                for k in range(self.args.duplicates):
                    train_args = self.set_train_args(train_size, condition, k)
                    eval_args = self.set_eval_args(train_size, condition, k)

                    print(f'\nRunning training session {k+1}/{self.args.duplicates} -- TRAIN SIZE = {train_size}, CONDITION = {condition}\n')
                    if self.args.framework == 'tensorflow':
                        training_session = SegmentationSession(train_args).run()
                    elif self.args.framework == 'torch':
                        training_session = SegmentationSessionTorch(train_args).run()
                    else:
                        raise ValueError("framework must be either 'tensorflow' or 'torch'")

                    print(f'\nRunning evaluation session {k+1}/{self.args.duplicates} -- TRAIN SIZE = {train_size}, CONDITION = {condition}\n')
                    if self.args.framework == 'tensorflow':
                        evaluation_session = EvaluationSession(eval_args).run()
                    else:
                        evaluation_session = EvaluationSessionTorch(eval_args).run()

    def set_train_args(self, train_size, condition, k):

        train_args = SegmentationSessionArgParser().parse_args([])
        setattr(train_args, "seed", self.args.seed)
        if self.args.duplicates > 1:
            records = os.path.join(self.args.data_dir, f'segmentation-train-{train_size}-{k}.tfrecord')
            log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}-{k}', 'train')
        else:
            records = os.path.join(self.args.data_dir, f'segmentation-train-{train_size}.tfrecord')
            log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}', 'train')
            
        setattr(train_args, "train_records", records)
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
        setattr(train_args, "loss_function", self.args.loss_function)

        os.makedirs(log_dir, exist_ok=True)
        setattr(train_args, "log_dir", log_dir)

        if condition == 'one_pixel_mask':
            setattr(train_args, "one_pixel_mask", True)
        elif condition == 'one_image_label':
            setattr(train_args, "one_image_label", True)

        setattr(train_args, "tile_size", self.args.tile_size)
        setattr(train_args, "data_parameters", self.args.data_parameters)
        setattr(train_args, "calibrate", self.args.calibrate)
        setattr(train_args, "super_loss", self.args.super_loss)

        return train_args

    def set_eval_args(self, train_size, condition, k):

        eval_args = EvaluationSessionArgParser().parse_args([])

        if self.args.duplicates > 1:
            log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}-{k}')
        else:
            log_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}')

        setattr(eval_args, "seed", self.args.seed)
        setattr(
            eval_args,
            "test_records",
            os.path.join(self.args.data_dir, f'segmentation-test.tfrecord')
        )
        setattr(eval_args, "num_classes", self.args.num_classes)
        setattr(eval_args, "batch_size", self.args.batch_size)
        
        if self.args.framework == 'tensorflow':
            model_file = 'model.h5'
        else:
            model_file = 'model.pth'

        setattr(eval_args, "model_weights", os.path.join(log_dir, 'train', model_file))

        os.makedirs(os.path.join(log_dir, 'eval'), exist_ok=True)
        setattr(eval_args, "log_dir", os.path.join(log_dir, 'eval'))

        if condition == 'one_image_label':
            setattr(eval_args, "one_image_label", True)

        setattr(eval_args, "num_bins", self.args.num_bins)
        setattr(eval_args, "tile_size", self.args.tile_size)
        setattr(eval_args, "calibrate", self.args.calibrate)

        calibration_temp = os.path.join(log_dir, 'train', 'calibration-temp.pth')
        setattr(eval_args, "calibration_temp", calibration_temp)

        return eval_args


if __name__ == "__main__":
    args = ExperimentSessionArgParser().parse_args()
    session = ExperimentSession(args)
    session.run()
