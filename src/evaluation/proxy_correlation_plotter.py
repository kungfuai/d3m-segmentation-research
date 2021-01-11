import logging
import os 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

from src.evaluation.proxy_correlation_plotter_arg_parser import ProxyCorrelationPlotterArgParser
from src.evaluation.metric_plotter import MetricPlotter

LOGGER = logging.getLogger(__name__)


class ProxyCorrelationPlotter:
    """ plot correlation between proxy task and segmentation task for different proxy tasks """

    def __init__(self, args):
        self.args = args

    def plot(self):

        if not isinstance(self.args.experiment_dirs, list):
            self.args.experiment_dirs = [self.args.experiment_dirs]

        os.makedirs(
            os.path.join(self.args.experiment_dirs[0], 'metrics'), 
            exist_ok=True
        )

        metrics = []
        for e in self.args.experiment_dirs:
            for f in os.listdir(e):
                d = os.path.join(e, f)
                if os.path.isdir(d) and d.split('/')[-1] not in ['metrics', 'confusion-matrices']:
                    dataset_size, condition = MetricPlotter.parse_dir_name(d)
                    real_metric_file = os.path.join(d, 'eval', 'metrics.json')
                    data = pd.DataFrame([pd.read_json(real_metric_file, typ='series')])
                    data = data.rename(columns={
                        'accuracy': "segmentation accuracy",
                        'iou_score': 'segmentation iou_score'
                    })

                    proxy_metric_file = os.path.join(d, 'train', 'metrics.csv')
                    proxy_data = pd.read_csv(proxy_metric_file)
                    proxy_acc = proxy_data['val_categorical_accuracy'].values[-1]
                    proxy_iou = proxy_data['val_iou_score'].values[-1]

                    data['proxy task accuracy'] = proxy_acc
                    data['proxy task iou_score'] = proxy_iou
                    data['condition'] = condition

                    metrics.append(data)

        metrics = pd.concat(metrics).reset_index()

        for metric in ['accuracy', 'iou_score']:
            for condition in ['full_pixel_mask', 'one_pixel_mask', 'one_image_label']:
                data = metrics[metrics['condition'] == condition]
                plt.clf()
                sns.scatterplot(
                    x=f"proxy task {metric}", 
                    y=f"segmentation {metric}",
                    data=data
                )
                f = os.path.join(
                    self.args.experiment_dirs[0], 
                    'metrics', 
                    f'correlation-{condition}-{metric}.png'
                )

                correlation = data[f"segmentation {metric}"].corr(data[f"proxy task {metric}"])
                plt.title(f'condition: {condition}, correlation: {round(correlation, 3)}')
                plt.savefig(f)


if __name__ == "__main__":
    args = ProxyCorrelationPlotterArgParser().parse_args()
    plotter = ProxyCorrelationPlotter(args)
    plotter.plot()