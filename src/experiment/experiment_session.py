import os
import logging

from src.experiment.experiment_session_arg_parser import ExperimentSessionArgParser
from src.training.segmentation_session import SegmentationSession
from src.training.segmentation_session_torch import SegmentationSessionTorch
from src.training.segmentation_session_arg_parser import SegmentationSessionArgParser
from src.training.update_labels import UpdateLabels
from src.training.update_labels_arg_parser import UpdateLabelsArgParser
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

                    if self.args.use_pseudo_labels:
                        print(f'\nGenerating pseudo labels {k+1}/{self.args.duplicates} -- TRAIN SIZE = {train_size}, CONDITION = {condition}\n')
                        update_args = self.set_update_args(train_size, condition, k)
                        UpdateLabels(update_args).run()

                        setattr(
                            update_args,
                            "update_records",
                            os.path.join(self.args.data_dir, f'segmentation-val.tfrecord')
                        )
                        UpdateLabels(update_args).run()

                    if self.args.duplicates > 1:
                        log_dir_str = f'{train_size}-{condition}-{k}'
                    else:
                        log_dir_str = f'{train_size}-{condition}'

                    if self.args.use_pseudo_labels:
                        self.log_dir = os.path.join(self.args.log_dir, f'{log_dir_str}-pseudo')
                        run_condition = 'full_pixel_mask'                        
                    else:
                        self.log_dir = os.path.join(self.args.log_dir, log_dir_str)
                        run_condition = condition

                    train_args = self.set_train_args(train_size, run_condition, k, self.args.use_pseudo_labels)
                    eval_args = self.set_eval_args(train_size, run_condition, k, self.args.use_pseudo_labels)

                    print(f'\nRunning training session {k+1}/{self.args.duplicates} -- TRAIN SIZE = {train_size}, CONDITION = {run_condition}\n')
                    if self.args.framework == 'tensorflow':
                        SegmentationSession(train_args).run()
                    elif self.args.framework == 'torch':
                        SegmentationSessionTorch(train_args).run()
                    else:
                        raise ValueError("framework must be either 'tensorflow' or 'torch'")

                    print(f'\nRunning evaluation session {k+1}/{self.args.duplicates} -- TRAIN SIZE = {train_size}, CONDITION = {run_condition}\n')
                    if self.args.framework == 'tensorflow':
                        EvaluationSession(eval_args).run()
                    else:
                        EvaluationSessionTorch(eval_args).run()

    def set_train_args(self, train_size, condition, k, pseudo_labels):

        train_args = SegmentationSessionArgParser().parse_args([])
        setattr(train_args, "seed", self.args.seed)

        if self.args.duplicates > 1:
            records_str = f'segmentation-train-{train_size}-{k}'
            model_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}-{k}', 'train')
        else:
            records_str = f'segmentation-train-{train_size}'
            model_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}', 'train')

        if pseudo_labels:
            records_str += '-pseudo'
            setattr(train_args, "model_weights", os.path.join(model_dir, 'model.pth'))
        
        records = os.path.join(self.args.data_dir, f'{records_str}.tfrecord')
            
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

        os.makedirs(os.path.join(self.log_dir, 'train'), exist_ok=True)
        setattr(train_args, "log_dir", os.path.join(self.log_dir, 'train'))

        if condition == 'one_pixel_mask':
            setattr(train_args, "one_pixel_mask", True)
        elif condition == 'one_image_label':
            setattr(train_args, "one_image_label", True)

        setattr(train_args, "tile_size", self.args.tile_size)
        setattr(train_args, "data_parameters", self.args.data_parameters)
        setattr(train_args, "calibrate", self.args.calibrate)
        setattr(train_args, "super_loss", self.args.super_loss)
        setattr(train_args, "pseudo_label_conf_threshold", self.args.pseudo_label_conf_threshold)

        return train_args

    def set_eval_args(self, train_size, condition, k, pseudo_labels):

        eval_args = EvaluationSessionArgParser().parse_args([])

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

        setattr(eval_args, "model_weights", os.path.join(self.log_dir, 'train', model_file))

        os.makedirs(os.path.join(self.log_dir, 'eval'), exist_ok=True)
        setattr(eval_args, "log_dir", os.path.join(self.log_dir, 'eval'))

        if condition == 'one_image_label':
            setattr(eval_args, "one_image_label", True)

        setattr(eval_args, "num_bins", self.args.num_bins)
        setattr(eval_args, "tile_size", self.args.tile_size)
        setattr(eval_args, "calibrate", self.args.calibrate)

        calibration_temp = os.path.join(self.log_dir, 'train', 'calibration-temp.pth')
        setattr(eval_args, "calibration_temp", calibration_temp)

        return eval_args

    def set_update_args(self, train_size, condition, k):

        update_args = UpdateLabelsArgParser().parse_args([])
        
        setattr(update_args, "seed", self.args.seed)

        if self.args.duplicates > 1:
            records = os.path.join(self.args.data_dir, f'segmentation-train-{train_size}-{k}.tfrecord')
            model_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}-{k}', 'train')
        else:
            records = os.path.join(self.args.data_dir, f'segmentation-train-{train_size}.tfrecord')
            model_dir = os.path.join(self.args.log_dir, f'{train_size}-{condition}', 'train')

        setattr(update_args, "update_records", records)
        setattr(update_args, "num_classes", self.args.num_classes)
        setattr(update_args, "batch_size", self.args.batch_size)
        setattr(update_args, "model_weights", os.path.join(model_dir, 'model.pth'))

        if condition == 'one_image_label':
            setattr(update_args, "one_image_label", True)

        setattr(update_args, "tile_size", self.args.tile_size)
        setattr(update_args, "calibrate", self.args.calibrate)

        calibration_temp = os.path.join(model_dir, 'train', 'calibration-temp.pth')
        setattr(update_args, "calibration_temp", calibration_temp)

        return update_args

if __name__ == "__main__":
    args = ExperimentSessionArgParser().parse_args()
    session = ExperimentSession(args)
    session.run()
