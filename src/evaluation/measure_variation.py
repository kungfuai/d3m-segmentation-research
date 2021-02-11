import os
import json

import pandas as pd
import numpy as np

from src.evaluation.measure_variation_arg_parser import MeasureVariationArgParser

class MeasureVariation:
    """ measures variation in metrics over multiple replications of experiments 
        with different datasets
    """ 

    def __init__(self, args):
        self.args = args

    def measure(self):

        metrics = []
        for f in os.listdir(self.args.experiment_dir):
            d = os.path.join(self.args.experiment_dir, f)
            if os.path.isdir(d) and d.split('/')[-1] not in [
                'metrics', 
                'confusion-matrices',
                'calibration-plots'
            ]:  
                dataset_size, condition = self.parse_dir_name(d)
                metric_file = os.path.join(d, 'eval', 'metrics.json')
                if os.path.isfile(metric_file):
                    data = pd.DataFrame([pd.read_json(metric_file, typ='series')])
                    data['ds_size'] = int(dataset_size)
                    data['condition'] = condition
                    metrics.append(data)
        metrics = pd.concat(metrics).reset_index()

        log_file = os.path.join(self.args.experiment_dir, "variation.txt")
        with open(log_file, "w") as f:
            f.write('Variation Statistics\n')

        for ds_size in np.unique(metrics['ds_size'].values):
            for condition in np.unique(metrics['condition'].values):
                with open(log_file, "a") as f:
                    f.write(f'\n{condition} - {ds_size}\n')

                data = metrics[(metrics['ds_size']==ds_size) & (metrics['condition']==condition)]
                
                for m in ['accuracy', 'max_calibration_error', 'expected_calibration_error']:
                    mu = round(np.mean(data[m]), 3)
                    std = round(np.std(data[m]), 3)
                    mi = round(np.min(data[m]), 3)
                    ma = round(np.max(data[m]), 3)

                    with open(log_file, "a") as f:
                        f.write(f'{m} -- mu: {mu}, std: {std}, min: {mi}, max: {ma}\n')

    @staticmethod
    def parse_dir_name(directory):
        dir_str = directory.split('/')[-1]
        dataset_size, condition, _ = dir_str.split('-')
        return dataset_size, condition

if __name__ == "__main__":
    args = MeasureVariationArgParser().parse_args()
    meter = MeasureVariation(args)
    meter.measure()