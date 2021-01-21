import numpy as np
import tensorflow as tf

from src.dataset.big_earth_dataset import BAND_STATS

class SegmentationDataset:
    def __init__(
        self, 
        TFRecord_paths, 
        one_pixel_mask: bool = False,
        one_image_label: bool = False,
        loss_function: str = 'focal',
        nb_class: int = 5,
        batch_size: int = 64, 
        shuffle_buffer_size: int = 100,
        normalize: bool = True
    ):

        self.one_pixel_mask = one_pixel_mask
        self.one_image_label = one_image_label
        self.loss_function = loss_function

        dataset = tf.data.TFRecordDataset(TFRecord_paths)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True
            )

        dataset = dataset.map(
            lambda x: self.parse_function(x), 
            num_parallel_calls=10
        )

        if normalize:
            dataset = dataset.map(
                lambda x: self.normalize(x, nb_class), 
                num_parallel_calls=10
            )

        dataset = dataset.batch(batch_size, drop_remainder=False)
        self.dataset = dataset.prefetch(10)

        if normalize and nb_class > 1:
            self.class_weights = self.class_weights(nb_class)

    def parse_function(self, example_proto):

        parsed_features = tf.io.parse_single_example(
                example_proto, 
                {
                    'B01': tf.io.FixedLenFeature([21*21], tf.int64),
                    'B02': tf.io.FixedLenFeature([126*126], tf.int64),
                    'B03': tf.io.FixedLenFeature([126*126], tf.int64),
                    'B04': tf.io.FixedLenFeature([126*126], tf.int64),
                    'B05': tf.io.FixedLenFeature([63*63], tf.int64),
                    'B06': tf.io.FixedLenFeature([63*63], tf.int64),
                    'B07': tf.io.FixedLenFeature([63*63], tf.int64),
                    'B08': tf.io.FixedLenFeature([126*126], tf.int64),
                    'B8A': tf.io.FixedLenFeature([63*63], tf.int64),
                    'B09': tf.io.FixedLenFeature([21*21], tf.int64),
                    'B11': tf.io.FixedLenFeature([63*63], tf.int64),
                    'B12': tf.io.FixedLenFeature([63*63], tf.int64),
                    'Corine_labels': tf.io.FixedLenFeature([126*126], tf.int64),
                    'random_pixel': tf.io.FixedLenFeature([2], tf.int64),
                    'class_label': tf.io.FixedLenFeature([1], tf.int64)
                }
            )
        
        return {
            'B01': tf.reshape(tf.cast(parsed_features['B01'], tf.float32), [21, 21]),
            'B02': tf.reshape(tf.cast(parsed_features['B02'], tf.float32), [126, 126]),
            'B03': tf.reshape(tf.cast(parsed_features['B03'], tf.float32), [126, 126]),
            'B04': tf.reshape(tf.cast(parsed_features['B04'], tf.float32), [126, 126]),
            'B05': tf.reshape(tf.cast(parsed_features['B05'], tf.float32), [63, 63]),
            'B06': tf.reshape(tf.cast(parsed_features['B06'], tf.float32), [63, 63]),
            'B07': tf.reshape(tf.cast(parsed_features['B07'], tf.float32), [63, 63]),
            'B08': tf.reshape(tf.cast(parsed_features['B08'], tf.float32), [126, 126]),
            'B8A': tf.reshape(tf.cast(parsed_features['B8A'], tf.float32), [63, 63]),
            'B09': tf.reshape(tf.cast(parsed_features['B09'], tf.float32), [21, 21]),
            'B11': tf.reshape(tf.cast(parsed_features['B11'], tf.float32), [63, 63]),
            'B12': tf.reshape(tf.cast(parsed_features['B12'], tf.float32), [63, 63]),
            'Corine_labels': tf.reshape(parsed_features['Corine_labels'], [126, 126]),
            'random_pixel': tf.reshape(parsed_features['random_pixel'], [1,2]),
            'class_label': parsed_features['class_label']
        }

    def normalize(self, img_dict, nb_class):
        B01  = (img_dict['B01'] - BAND_STATS['mean']['B01']) / BAND_STATS['std']['B01']
        B02  = (img_dict['B02'] - BAND_STATS['mean']['B02']) / BAND_STATS['std']['B02']
        B03  = (img_dict['B03'] - BAND_STATS['mean']['B03']) / BAND_STATS['std']['B03']
        B04  = (img_dict['B04'] - BAND_STATS['mean']['B04']) / BAND_STATS['std']['B04']
        B05  = (img_dict['B05'] - BAND_STATS['mean']['B05']) / BAND_STATS['std']['B05']
        B06  = (img_dict['B06'] - BAND_STATS['mean']['B06']) / BAND_STATS['std']['B06']
        B07  = (img_dict['B07'] - BAND_STATS['mean']['B07']) / BAND_STATS['std']['B07']
        B08  = (img_dict['B08'] - BAND_STATS['mean']['B08']) / BAND_STATS['std']['B08']
        B8A  = (img_dict['B8A'] - BAND_STATS['mean']['B8A']) / BAND_STATS['std']['B8A']
        B09  = (img_dict['B09'] - BAND_STATS['mean']['B09']) / BAND_STATS['std']['B09']
        B11  = (img_dict['B11'] - BAND_STATS['mean']['B11']) / BAND_STATS['std']['B11']
        B12  = (img_dict['B12'] - BAND_STATS['mean']['B12']) / BAND_STATS['std']['B12']
        bands_10m = tf.stack([B04, B03, B02, B08], axis=2)
        bands_20m = tf.stack([B05, B06, B07, B8A, B11, B12], axis=2)
        bands_60m = tf.stack([B01, B09], axis=2)
        img = tf.concat(
            [
                bands_10m, 
                tf.image.resize(bands_20m, [126, 126], method='bicubic')
            ], 
            axis=2
        )

        img = tf.pad(
            img, 
            tf.constant([[1,1], [1,1], [0,0]]), 
            "CONSTANT"
        )

        if self.one_image_label:
            labels = img_dict['class_label']
        else:
            if self.one_pixel_mask:
                pixel = img_dict['random_pixel']
                value = tf.gather_nd(img_dict['Corine_labels'], pixel)
                if nb_class > 1:
                    labels = tf.zeros((126, 126), dtype=tf.int64) - 1
                else:
                    labels = tf.zeros((126, 126), dtype=tf.int64)
                labels = tf.tensor_scatter_nd_update(labels, pixel, value)
                mask = tf.tensor_scatter_nd_update(
                    tf.zeros((126, 126), dtype=tf.int64),
                    pixel,
                    tf.ones((1,), dtype=tf.int64)
                )
            else:
                labels = img_dict['Corine_labels']
                mask = tf.ones((126,126), dtype=tf.int64)
                
            labels = tf.pad(
                labels, 
                tf.constant([[1,1], [1,1]]),
                "CONSTANT",
                constant_values=-1
            )
            mask = tf.pad(
                mask, 
                tf.constant([[1,1], [1,1]]),
                "CONSTANT",
            )

        if nb_class > 1:
            one_hots = tf.one_hot(labels, nb_class)

            if self.one_pixel_mask:
                return img, one_hots, mask
            elif self.one_image_label and self.loss_function == 'xent':
                return img, labels
            else:
                return img, one_hots

        else:
            if self.one_image_label:
                return img, tf.expand_dims(labels, axis=-1)
            else:
                return img, tf.expand_dims(labels, axis=-1), mask

    def class_weights(self, nb_class):
        
        labels = []
        for batch in self.dataset:
            labels.append(tf.where(batch[1] == 1)[:,-1])
        counts = np.bincount(np.concatenate(labels))
        weights = 1 / (counts + 1)

        if weights.shape[0] < nb_class:
            extend = np.ones(nb_class - weights.shape[0])
            weights = np.concatenate((weights, extend))
        
        weights /= np.max(weights)

        if self.one_image_label:
            return {i: w for i, w in enumerate(weights)}
        else:
            return weights


