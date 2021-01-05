# -*- coding: utf-8 -*-
#
# BigEarthNet class to create tf.data.Dataset based on the TFRecord files. 
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1

import numpy as np
import tensorflow as tf

BAND_STATS = {
            'mean': {
                'B01': 340.76769064,
                'B02': 429.9430203,
                'B03': 614.21682446,
                'B04': 590.23569706,
                'B05': 950.68368468,
                'B06': 1792.46290469,
                'B07': 2075.46795189,
                'B08': 2218.94553375,
                'B8A': 2266.46036911,
                'B09': 2246.0605464,
                'B11': 1594.42694882,
                'B12': 1009.32729131
            },
            'std': {
                'B01': 554.81258967,
                'B02': 572.41639287,
                'B03': 582.87945694,
                'B04': 675.88746967,
                'B05': 729.89827633,
                'B06': 1096.01480586,
                'B07': 1273.45393088,
                'B08': 1365.45589904,
                'B8A': 1356.13789355,
                'B09': 1302.3292881,
                'B11': 1079.19066363,
                'B12': 818.86747235
            }
        }

class BigEarthDataset:
    def __init__(
        self, 
        TFRecord_paths, 
        nb_class: int = 5,
        batch_size: int = 64, 
        shuffle_buffer_size: int = 100
    ):
        dataset = tf.data.TFRecordDataset(TFRecord_paths)
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, 
                reshuffle_each_iteration=True
            )
        
        dataset = dataset.map(
            lambda x: self.parse_function(x, nb_class = nb_class), 
            num_parallel_calls=10
        )
        dataset = dataset.map(
            lambda x: self.normalize(x), 
            num_parallel_calls=10
        )

        dataset = dataset.batch(batch_size, drop_remainder=False)

        self.dataset = dataset.prefetch(10)
        self.class_weights = self.class_weights(nb_class)

    def parse_function(self, example_proto, nb_class = 5):

        parsed_features = tf.io.parse_single_example(
                example_proto, 
                {
                    'B01': tf.io.FixedLenFeature([20*20], tf.int64),
                    'B02': tf.io.FixedLenFeature([120*120], tf.int64),
                    'B03': tf.io.FixedLenFeature([120*120], tf.int64),
                    'B04': tf.io.FixedLenFeature([120*120], tf.int64),
                    'B05': tf.io.FixedLenFeature([60*60], tf.int64),
                    'B06': tf.io.FixedLenFeature([60*60], tf.int64),
                    'B07': tf.io.FixedLenFeature([60*60], tf.int64),
                    'B08': tf.io.FixedLenFeature([120*120], tf.int64),
                    'B8A': tf.io.FixedLenFeature([60*60], tf.int64),
                    'B09': tf.io.FixedLenFeature([20*20], tf.int64),
                    'B11': tf.io.FixedLenFeature([60*60], tf.int64),
                    'B12': tf.io.FixedLenFeature([60*60], tf.int64),
                    'patch_name': tf.io.VarLenFeature(dtype=tf.string),
                    'BigEarthNet-19_labels': tf.io.VarLenFeature(dtype=tf.string),
                    'BigEarthNet-19_labels_multi_hot': tf.io.FixedLenFeature([nb_class], tf.int64)
                }
            )

        return {
            'B01': tf.reshape(tf.cast(parsed_features['B01'], tf.float32), [20, 20]),
            'B02': tf.reshape(tf.cast(parsed_features['B02'], tf.float32), [120, 120]),
            'B03': tf.reshape(tf.cast(parsed_features['B03'], tf.float32), [120, 120]),
            'B04': tf.reshape(tf.cast(parsed_features['B04'], tf.float32), [120, 120]),
            'B05': tf.reshape(tf.cast(parsed_features['B05'], tf.float32), [60, 60]),
            'B06': tf.reshape(tf.cast(parsed_features['B06'], tf.float32), [60, 60]),
            'B07': tf.reshape(tf.cast(parsed_features['B07'], tf.float32), [60, 60]),
            'B08': tf.reshape(tf.cast(parsed_features['B08'], tf.float32), [120, 120]),
            'B8A': tf.reshape(tf.cast(parsed_features['B8A'], tf.float32), [60, 60]),
            'B09': tf.reshape(tf.cast(parsed_features['B09'], tf.float32), [20, 20]),
            'B11': tf.reshape(tf.cast(parsed_features['B11'], tf.float32), [60, 60]),
            'B12': tf.reshape(tf.cast(parsed_features['B12'], tf.float32), [60, 60]),
            'patch_name': parsed_features['patch_name'],
            'BigEarthNet-19_labels': parsed_features['BigEarthNet-19_labels'],
            'BigEarthNet-19_labels_multi_hot': 
                tf.cast(parsed_features['BigEarthNet-19_labels_multi_hot'], tf.float32) 
        }

    def normalize(self, img_dict):
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
                tf.image.resize(bands_20m, [120, 120], method='bicubic')
            ], 
            axis=2
        )
        return img, img_dict['BigEarthNet-19_labels_multi_hot']

    def class_weights(self, nb_class):
        
        class_counts = np.zeros(nb_class)
        for batch in self.dataset:
            labels = batch[1].numpy()
            class_counts += labels.sum(axis=0)
        
        if len(np.where(class_counts == 0)[0]):
            class_counts = class_counts + 1
        
        return {i: 1 / ct for i, ct in enumerate(class_counts)}
