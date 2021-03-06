import logging
import os 
import random
import json

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow as tf 

from src.evaluation.mask_plotter_arg_parser import MaskPlotterArgParser
from src.evaluation.metric_plotter import MetricPlotter
from src.dataset.segmentation_dataset import SegmentationDataset
from src.model.unet import Unet

LOGGER = logging.getLogger(__name__)


class MaskPlotter:
    """ plots segmentation masks generated from experiment session"""

    def __init__(self, args):
        self.args = args

        with open('data/label_indices.json', 'rb') as f:
            label_indices = json.load(f)
        
        if self.args.num_classes > 1:
            self.class_keys = label_indices['BigEarthNet-19_labels'].keys()
            self.class_colors = [
                'grey',
                'tan',
                'green',
                'turquoise',
                'blue'
            ]
        else:
            self.class_keys = label_indices['BigEarthNet-19_labels_binary'].keys()
            self.class_colors = ['green', 'tan']

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_models()
        self.make_predictions()
        self.plot()

    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            tf.random.set_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(
            os.path.join(self.args.experiment_dir, 'metrics'),
            exist_ok=True
        )

    def load_data(self):
        test_dataset_raw = iter(SegmentationDataset(
            self.args.test_records,
            nb_class=self.args.num_classes,
            batch_size=self.args.n_examples,
            shuffle_buffer_size=0,
            normalize=False
        ).dataset)

        for _ in range(self.args.batch_no):
            img_dict = next(test_dataset_raw)

        imgs = np.stack(
            [img_dict['B04'], img_dict['B03'], img_dict['B02']],
            axis=-1
        )
        self.imgs = (imgs / 4096) ** 0.5 * 255
        self.labels = [img_dict['Corine_labels']]

        self.test_dataset = iter(SegmentationDataset(
            self.args.test_records,
            nb_class=self.args.num_classes,
            batch_size=self.args.n_examples,
            shuffle_buffer_size=0,
        ).dataset)

    def create_models(self):

        self.models = {}
        for f in sorted(os.listdir(self.args.experiment_dir)):
            d = os.path.join(self.args.experiment_dir, f)
            if os.path.isdir(d) and d.split('/')[-1] not in [
                'metrics', 
                'confusion-matrices',
                'calibration-plots'
            ]:                
                dataset_size, condition = MetricPlotter.parse_dir_name(d)
                one_image_label = (condition=='one_image_label')

                if self.args.num_classes > 1:
                    activation='softmax'
                else:
                    activation='sigmoid'

                model = Unet(
                    input_shape=(128, 128, 10), 
                    classes=self.args.num_classes,
                    weights=os.path.join(d, 'train', 'model.h5'),
                    one_image_label=one_image_label,
                    activation=activation
                )

                if one_image_label:
                    features = model.get_layer('decoder_stage4b_relu').output
                    output = model.get_layer('final_fc')(features) 

                    if self.args.num_classes == 1:
                        output = tf.keras.activations.sigmoid(output)

                    model = tf.keras.Model(
                        inputs=model.input,
                        outputs=output
                    )

                if dataset_size in self.models.keys():
                    self.models[dataset_size].append(model)
                else:
                    self.models[dataset_size] = [model]

        self.model_types = len(self.models[dataset_size])
        
    def make_predictions(self):
        
        for _ in range(self.args.batch_no):
            batch = next(self.test_dataset)

        self.masks = []
        for _, models in self.models.items():
            preds = [model.predict_on_batch(batch) for model in models]
            if self.args.num_classes > 1:
                preds = [tf.math.argmax(p, 3)[:, 1:-1, 1:-1] for p in preds]
            else:
                preds = [tf.math.round(tf.squeeze(p))[:, 1:-1, 1:-1] for p in preds]
            masks = np.stack(self.labels + preds) 
            self.masks.append(masks)
        self.masks = np.stack(self.masks)

    def plot(self):

        rows = self.args.n_examples + 1
        cols = self.model_types + 2
        col_titles = [
            'Sentinel-2 Image', 
            'Ground Truth (Corine)',
            'Full Pixel Mask',
            'One Image Label',
            'One Pixel Mask'
        ]

        for ds_size, masks in zip(self.models.keys(), self.masks):
            plt.clf()
            plt.figure(figsize=(3 * cols, 3 * rows))

            for row in range(rows): 
                for col, col_title in zip(range(cols), col_titles[:cols]):
                    i = row * cols + col + 1
                    ax = plt.subplot(rows, cols, i)

                    plt.tick_params(
                        axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False
                    )

                    if row == 0:
                        plt.axis('off')

                        if i == cols:
                            plt.legend(
                                handles=[
                                    mpatches.Patch(color=c, label=k)
                                    for c, k in zip(self.class_colors, self.class_keys)
                                ],
                                loc='upper right'
                            )

                    else:
                        if row == 1:
                            plt.title(col_title)

                        if col == 0:
                            plt.imshow(self.imgs[row-1].astype("uint8"))
                        else:
                            plt.imshow(
                                masks[col-1][row-1], 
                                vmin=0, 
                                vmax=max(1, self.args.num_classes - 1),
                                cmap=ListedColormap(self.class_colors)
                                #cmap='Oranges'
                            )

            plt.savefig(
                os.path.join(self.args.experiment_dir, 'metrics', f'masks-{ds_size}.png'),
                bbox_inches='tight', 
                pad_inches=0.5
            )


if __name__ == "__main__":
    args = MaskPlotterArgParser().parse_args()
    plotter = MaskPlotter(args)
    plotter.run()