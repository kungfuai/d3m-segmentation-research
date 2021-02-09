import logging
import os 
import json

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

from src.evaluation.metric_plotter import MetricPlotter
from src.evaluation.confusion_plotter_arg_parser import ConfusionPlotterArgParser

LOGGER = logging.getLogger(__name__)


class ConfusionPlotter:
    """ plot confusion matrices from an experiment session """

    def __init__(self, args):
        self.args = args

        os.makedirs(
            os.path.join(self.args.experiment_dir, 'confusion-matrices'), 
            exist_ok=True
        )

        if self.args.num_classes > 1:
            self.class_keys = [
                'Artificial',
                'Agricultural',
                'Forest',
                'Wetlands',
                'Water Bodies'
            ]
        else:
            self.class_keys = ['Not Agricultural', 'Agricultural']

    def plot(self): 

        metrics = []
        for f in os.listdir(self.args.experiment_dir):
            d = os.path.join(self.args.experiment_dir, f)
            if os.path.isdir(d) and d.split('/')[-1] not in [
                'metrics', 
                'confusion-matrices',
                'calibration-plots'
            ]:
                dataset_size, condition = MetricPlotter.parse_dir_name(d)
                confusion_file = os.path.join(d, 'eval', 'confusion.csv')
                data = pd.read_csv(
                    confusion_file, 
                    header=None,
                    names=self.class_keys
                )
                data.index = data.columns

                plt.clf()
                ax = sns.heatmap(
                    data,
                    cmap="YlGnBu",
                    cbar=False,
                    vmin=0,
                    vmax=1,
                    annot=True,
                    yticklabels=True

                )
                ax.figure.subplots_adjust(left = 0.2)
                plt.xlabel('Predictions')
                plt.ylabel('Labels')
                plt.title(f'Condition: {condition}, Dataset Size: {dataset_size}')

                f = os.path.join(
                    self.args.experiment_dir, 
                    'confusion-matrices', 
                    f'confusion-matrix-{dataset_size}-{condition}.png'
                )
                plt.savefig(f)


if __name__ == "__main__":
    args = ConfusionPlotterArgParser().parse_args()
    plotter = ConfusionPlotter(args)
    plotter.plot()