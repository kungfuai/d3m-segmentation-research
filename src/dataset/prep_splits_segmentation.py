import argparse
import os
import json
import random
import shutil 

import rasterio
import numpy as np
import pickle
import tensorflow as tf
import zipfile
from geolib import geohash as glgh

from src.dataset.tensorflow_utils import band_names
from src.dataset.get_sentinel_data import sentinel_channels

def load_estonia_data(sentinel, corine):

    bands = {}
    for band_name in band_names:
        band_path = os.path.join(sentinel, 'T35VMF_20180510T094031_' + band_name + '.jp2')
        band_ds = rasterio.open(band_path)
        band_data = np.array(band_ds.read(1))
        bands[band_name] = np.array(band_data)
    
    clc_ds = rasterio.open(corine)
    clc_data = np.array(clc_ds.read(1))
    bands['clc_data'] = clc_data

    return bands    

def convert_estonia_labels(labels, binary):

    with open('data/label_indices.json', 'rb') as f:
        label_indices = json.load(f)
    
    if binary:
        label_conversion = np.array(label_indices['label_conversion_binary'])
    else:
        label_conversion = np.array(label_indices['label_conversion'])
    label_conversion = {
        label + 1: i 
        for i, label_set in enumerate(label_conversion)
        for label in label_set
    }
    label_conversion[44] = -1 # no data, mask
    labels = np.vectorize(label_conversion.get)(labels)

    return labels

def load_ethiopia_data(sentinel_folder, labels_file, tile_size, num_samples=600):
    
    labels_img = rasterio.open(labels_file)
    labels = labels_img.read(1)

    tiles = []
    sentinel_img_sample = np.random.choice(
        os.listdir(sentinel_folder), 
        num_samples, 
        replace=False
    )
    for sentinel_img in sentinel_img_sample:
        
        bands = {}
        geohash = sentinel_img.split('.')[0]

        # if '.' in sentinel_img:
            # try:
            #     with zipfile.ZipFile(os.path.join(sentinel_folder, sentinel_img)) as zip_ref:
            #         zip_ref.extractall(os.path.join(sentinel_folder, geohash))
            #         os.remove(os.path.join(sentinel_folder, sentinel_img))
            # except:
            #     os.remove(os.path.join(sentinel_folder, sentinel_img))
            #     continue

        try:
            for band_name in sentinel_channels[:-1]: 
                band_path = os.path.join(sentinel_folder, geohash, f'{geohash}.{band_name}.tif')
                band_ds = rasterio.open(band_path)
                bands[band_name] = band_ds.read(1).astype(int)
        except:
            shutil.rmtree(os.path.join(sentinel_folder, geohash))
            continue

        geo_bounds = glgh.bounds(geohash)
        max_y, min_x = rasterio.transform.rowcol(
            labels_img.transform, 
            geo_bounds[0][1], 
            geo_bounds[0][0],
        )
        min_y, max_x = rasterio.transform.rowcol(
            labels_img.transform, 
            geo_bounds[1][1], 
            geo_bounds[1][0],
        )
        bands['clc_data'] = labels[min_x:max_x+1, min_y: max_y+1] 
        bands = standardize_sizes(bands)
        bands['clc_data'] = convert_ethiopia_labels(bands['clc_data'])

        tiles = split(bands, tile_size, tiles, estonia_data=False)

    # look at class distribution of pixels
    dist = np.concatenate([tile['clc_data'].flatten() for tile in tiles])
    dist = np.bincount(dist)
    print(dist / np.sum(dist))
    print(f'{len(tiles)} total tiles')

    return tiles

def standardize_sizes(bands):

    img_x, img_y = bands["B1"].shape
    label_x, label_y = bands['clc_data'].shape

    min_x = min(img_x, label_x)
    min_y = min(img_y, label_y)

    for band_name in sentinel_channels[:-1] + ['clc_data']:
        bands[band_name] = bands[band_name][:min_x, :min_y]

    return bands

def convert_ethiopia_labels(labels):

    # From https://www.eo4idi.eu/sites/default/files/content/attachments/eo4sd_agri_ethiopia_2017.pdf
    original_labels = {
        "water": 0,
        "urban": 1,
        "bare_soil": 2,
        "agriculture": 3,
        "grassland": 4,
        "shrubs": 5,
        "forest": 6
    }

    label_conversion = np.array([
        [0,1,2,4,5,6], # not agriculture
        [3] # agriculture
    ])

    label_conversion = {
        label + 1: i 
        for i, label_set in enumerate(label_conversion)
        for label in label_set
    }
    label_conversion[0] = -1 # no data, mask
    labels = np.vectorize(label_conversion.get)(labels)

    return labels

def split(bands, tile_size, tiles=[], estonia_data=True):

    if estonia_data:
        bands_10m = ['B04', 'B03', 'B02', 'B08']
        bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        bands_60m = ['B01', 'B09']
        names = band_names
    else:
        bands_10m = ['B4', 'B3', 'B2', 'B8']
        bands_20m = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        bands_60m = ['B1', 'B9']       
        names = sentinel_channels[:-1]

    num_tiles = bands['clc_data'].shape[0] // tile_size

    for i in range(num_tiles):
        for j in range(num_tiles):
            tile_bands = {}

            si = i * tile_size
            ei = (i + 1) * tile_size
            sj = j * tile_size
            ej = (j + 1) * tile_size

            clc = bands['clc_data'][si:ei, sj:ej]

            if np.where(clc == -1)[0].shape[0]:
                continue
                
            tile_bands['clc_data'] = clc

            for band_name in names:
                if band_name in bands_10m:
                    pixels = tile_size
                elif band_name in bands_20m:
                    pixels = tile_size // 2
                elif band_name in bands_60m:
                    pixels = tile_size // 6

                si = i * pixels
                ei = (i + 1) * pixels
                sj = j * pixels
                ej = (j + 1) * pixels

                tile_bands[band_name] = bands[band_name][si:ei, sj:ej]

            tiles.append(tile_bands)

    return tiles

def shuffle(train_tiles, train_sizes):

    random.shuffle(train_tiles)
    trains = [train_tiles[:ts] for ts in train_sizes]
    return trains

def to_tf_records(data, out_folder, name, tile_size, estonia_data=True):

    if estonia_data:
        names = band_names
    else:    
        names = sentinel_channels[:-1]

    out_path = os.path.join(out_folder, 'segmentation-' + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(out_path, options='')

    progress_bar = tf.keras.utils.Progbar(target = len(data))
    for i, record in enumerate(data):

        pixel = np.random.randint(0,tile_size,size=(2))
        class_label = np.argmax(np.bincount(record['clc_data'].flatten()))

        feature_dict = {
            band_name: tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.ravel(record[name]))
            )
            for name, band_name in zip(names, band_names)
        }
        feature_dict['Corine_labels'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.ravel(record['clc_data']))
        )
        feature_dict['random_pixel'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.ravel(pixel))
        )
        feature_dict['class_label'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.ravel([class_label]))
        )
        feature_dict['sample_index'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.ravel([i]))
        )
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature_dict
            )
        )

        writer.write(example.SerializeToString())
        progress_bar.update(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script creates TFRecord files for the segmentation train,test splits'
    )
    parser.add_argument('-s', '--sentinel', dest = 'sentinel', type = str,
                        help = 'path to Sentinel2 satellite image folder')
    parser.add_argument('-l', '--labels', dest = 'labels', type = str,
                        help = 'path to land cover map (Corine or World Bank)')
    parser.add_argument('-t', '--tile_size', dest = 'tile_size', type = int,
                        help = 'tile size (in pixels squared)', default = 120)
    parser.add_argument('-v', '--val_split', dest = 'val_split', type = float,
                        help = 'validation split', default = 0.15)
    parser.add_argument('-o', '--out_folder', dest = 'out_folder', type = str,
                        help = 'folder path containing resulting TFRecord files')
    parser.add_argument('-n', '--train_sizes', dest='train_sizes', type = int, 
                        help = 'size of training sets', nargs = '+')
    parser.add_argument('-b', '--binary', dest='binary', type = bool, 
                        help = 'label images for binary classification', default = True)
    parser.add_argument('-d', '--duplicates', dest='duplicates', type = int, 
                        help = 'number of random duplicates for each training set size', default = 0)
    parser.add_argument('-e', '--estonia_data', dest='estonia_data', type = bool, 
                        help = 'whether to process Estonia land cover data (or Ethiopia)', default = False)

    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(args.out_folder, exist_ok=True)

    if args.estonia_data:
        #bands = load_estonia_data(args.sentinel, args.labels)
        #pickle.dump(bands, open("bands.pkl", "wb"))
        bands = pickle.load(open("bands.pkl", "rb" ))
        bands['clc_data'] = convert_estonia_labels(bands['clc_data'], args.binary)
        tiles = split(bands, args.tile_size) 

    else:
        tiles = load_ethiopia_data(args.sentinel, args.labels, args.tile_size)

    n = int(len(tiles) * args.val_split)
    test = tiles[-n:]
    val = tiles[-2*n:-n]
    to_tf_records(test, args.out_folder, 'test', args.tile_size, args.estonia_data)
    to_tf_records(val, args.out_folder, 'val', args.tile_size, args.estonia_data)

    train_tiles = tiles[:-2*n]

    for i in range(args.duplicates):
        trains = shuffle(train_tiles, args.train_sizes)

        for train in trains:
            to_tf_records(train, args.out_folder, f'train-{len(train)}-{i}', args.tile_size, args.estonia_data)





