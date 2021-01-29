import logging
import os 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

from src.evaluation.metric_plotter_arg_parser import MetricPlotterArgParser

LOGGER = logging.getLogger(__name__)


class MetricPlotter:
    """ plots metrics generated during experiment session"""

    def __init__(self, args):
        self.args = args

        os.makedirs(
            os.path.join(self.args.experiment_dir, 'metrics'), 
            exist_ok=True
        )

    def plot(self):

        metrics = []
        for f in os.listdir(self.args.experiment_dir):
            d = os.path.join(self.args.experiment_dir, f)
            if os.path.isdir(d) and d.split('/')[-1] not in ['metrics', 'confusion-matrices']:
                dataset_size, condition = self.parse_dir_name(d)
                metric_file = os.path.join(d, 'eval', 'metrics.json')
                data = pd.DataFrame([pd.read_json(metric_file, typ='series')])
                data['dataset size'] = int(dataset_size)
                data['condition'] = condition
                #data['training'] = training
                metrics.append(data)
        metrics = pd.concat(metrics).reset_index()

        for metric in ['accuracy']:#, 'iou_score']:
            plt.clf()
            sns.lineplot(
                x="dataset size", 
                y=metric,
                hue="condition",
                data=metrics,
                #style="training"
            )
            f = os.path.join(self.args.experiment_dir, 'metrics', f'metrics-{metric}.png')
            plt.xscale('log')
            plt.title(f'Change in segmentation {metric} as dataset size increases')
            plt.savefig(f)

    @staticmethod
    def parse_dir_name(directory):
        dir_str = directory.split('/')[-1]
        dataset_size, condition = dir_str.split('-')
        return dataset_size, condition
        # if len(tags) == 2:
        #     return tags[0], tags[1], 'random'
        # else:
        #     return tags[0], tags[1], tags[2]

if __name__ == "__main__":
    args = MetricPlotterArgParser().parse_args()
    plotter = MetricPlotter(args)
    plotter.plot()