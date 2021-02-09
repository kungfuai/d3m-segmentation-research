import argparse
import os
import json

import rasterio
import numpy as np
import pickle
import tensorflow as tf

from src.dataset.tensorflow_utils import band_names

def load_image_data(sentinel, corine):

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

def convert_labels(clc_data, binary):

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
    clc_data = np.vectorize(label_conversion.get)(clc_data)
    
    return clc_data

def split(bands, tile_size):

    bands_10m = ['B04', 'B03', 'B02', 'B08']
    bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    bands_60m = ['B01', 'B09']

    num_tiles = bands['clc_data'].shape[0] // tile_size

    tiles = []
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

            for band_name in band_names:
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

def shuffle(tiles, val_split, train_sizes):

    n = int(len(tiles) * val_split)
    test = tiles[-n:]
    val = tiles[-2*n:-n]
    train_tiles = tiles[:-2*n]
    trains = [train_tiles[:ts] for ts in train_sizes]
    return trains, val, test

def to_tf_records(data, out_folder, name, tile_size):

    out_path = os.path.join(out_folder, 'segmentation-' + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(out_path, options='')

    progress_bar = tf.keras.utils.Progbar(target = len(data))
    for i, record in enumerate(data):

        pixel = np.random.randint(0,tile_size,size=(2))
        class_label = np.argmax(np.bincount(record['clc_data'].flatten()))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'B01': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B01']))),
                    'B02': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B02']))),
                    'B03': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B03']))),
                    'B04': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B04']))),
                    'B05': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B05']))),
                    'B06': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B06']))),
                    'B07': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B07']))),
                    'B08': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B08']))),
                    'B8A': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B8A']))),
                    'B09': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B09']))),
                    'B11': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B11']))),
                    'B12': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['B12']))),
                    'Corine_labels': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=np.ravel(record['clc_data']))),
                    'random_pixel': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=pixel)),
                    'class_label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[class_label]))
                }
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
    parser.add_argument('-c', '--corine', dest = 'corine', type = str,
                        help = 'path to Corine Land Cover segmentation map')
    parser.add_argument('-t', '--tile_size', dest = 'tile_size', type = int,
                        help = 'tile size (in pixels squared)', default = 126)
    parser.add_argument('-v', '--val_split', dest = 'val_split', type = float,
                        help = 'validation split', default = 0.15)
    parser.add_argument('-o', '--out_folder', dest = 'out_folder', type = str,
                        help = 'folder path containing resulting TFRecord files')
    parser.add_argument('-n', '--train_sizes', dest='train_sizes', type = int, 
                        help = 'size of training sets', nargs = '+')
    parser.add_argument('-b', '--binary', dest='binary', type = bool, 
                        help = 'label images for binary classification', default = True)

    args = parser.parse_args()

    np.random.seed(0)

    #bands = load_image_data(args.sentinel, args.corine)
    #pickle.dump(bands, open("bands.pkl", "wb"))
    bands = pickle.load(open("bands.pkl", "rb" ))

    bands['clc_data'] = convert_labels(bands['clc_data'], args.binary)

    tiles = split(bands, args.tile_size) 

    trains, val, test = shuffle(tiles, args.val_split, args.train_sizes)
    
    for train in trains:
        to_tf_records(train, args.out_folder, f'train-{len(train)}', args.tile_size)

    to_tf_records(val, args.out_folder, 'val', args.tile_size)
    to_tf_records(test, args.out_folder, 'test', args.tile_size)




