"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa

Reads in a tsv file with pre-trained bottom up attention features 
of the adaptive number of boxes and stores it in HDF5 format.  
Also store {image_id: feature_idx} as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_boxes x 2048
  'image_bb': num_boxes x 4
  'spatial_features': num_boxes x 6
  'pos_boxes': num_images x 2 }
"""
from __future__ import print_function

import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import utils
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder, suffix):
    images = load_folder(folder, suffix)
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='img_dir where numpy files are stored (with / at the end of the string)')
    args = parser.parse_args()
    return args

def extract(img_dir):
    feature_length = 2048
    num_boxes = 100 * 40504

    imgids = load_imageid(img_dir, '.npz')
    cPickle.dump(imgids, open(img_dir + 'data/ids.pkl', 'wb'))

    h = h5py.File(img_dir + 'data/train.hdf5', 'w')

    img_features = h.create_dataset(
        'image_features', (num_boxes, feature_length), 'f')

    counter = 0
    num_boxes = 0
    indices = {}

    for f in tqdm(os.listdir(img_dir)):
        filename = os.fsdecode(f)
        if not filename.endswith(".npz"): 
            continue
        
        image_id = int(f.split('/')[-1].split('.')[0].split('_')[-1])

        data = np.load(img_dir + f, allow_pickle=True)['arr_0']
        feat = data

        indices[image_id] = counter
        img_features[num_boxes:num_boxes + 100, :] = feat

        num_boxes += 100

        counter += 1

    cPickle.dump(indices, open(img_dir + 'data/imgid2idx.pkl', 'wb'))
    h.close()
    print("done!")

if __name__ == '__main__':
    args = parse_args()

    extract(args.img_dir)