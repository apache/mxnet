# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


"""
This scripts downloads and prepares the Oxford 102 Category Flower Dataset for training
Dataset is from: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
Script is modified from: https://github.com/Arsey/keras-transfer-learning-for-oxford102
"""

import glob
import os
import tarfile
from shutil import copyfile

import numpy as np
from mxnet import gluon
from scipy.io import loadmat

label_names = [
    'pink primrose',
    'hard-leaved pocket orchid',
    'canterbury bells',
    'sweet pea',
    'english marigold',
    'tiger lily',
    'moon orchid',
    'bird of paradise',
    'monkshood',
    'globe thistle',
    'snapdragon',
    "colt's foot",
    'king protea',
    'spear thistle',
    'yellow iris',
    'globe-flower',
    'purple coneflower',
    'peruvian lily',
    'balloon flower',
    'giant white arum lily',
    'fire lily',
    'pincushion flower',
    'fritillary',
    'red ginger',
    'grape hyacinth',
    'corn poppy',
    'prince of wales feathers',
    'stemless gentian',
    'artichoke',
    'sweet william',
    'carnation',
    'garden phlox',
    'love in the mist',
    'mexican aster',
    'alpine sea holly',
    'ruby-lipped cattleya',
    'cape flower',
    'great masterwort',
    'siam tulip',
    'lenten rose',
    'barbeton daisy',
    'daffodil',
    'sword lily',
    'poinsettia',
    'bolero deep blue',
    'wallflower',
    'marigold',
    'buttercup',
    'oxeye daisy',
    'common dandelion',
    'petunia',
    'wild pansy',
    'primula',
    'sunflower',
    'pelargonium',
    'bishop of llandaff',
    'gaura',
    'geranium',
    'orange dahlia',
    'pink-yellow dahlia?',
    'cautleya spicata',
    'japanese anemone',
    'black-eyed susan',
    'silverbush',
    'californian poppy',
    'osteospermum',
    'spring crocus',
    'bearded iris',
    'windflower',
    'tree poppy',
    'gazania',
    'azalea',
    'water lily',
    'rose',
    'thorn apple',
    'morning glory',
    'passion flower',
    'lotus',
    'toad lily',
    'anthurium',
    'frangipani',
    'clematis',
    'hibiscus',
    'columbine',
    'desert-rose',
    'tree mallow',
    'magnolia',
    'cyclamen',
    'watercress',
    'canna lily',
    'hippeastrum ',
    'bee balm',
    'ball moss',
    'foxglove',
    'bougainvillea',
    'camellia',
    'mallow',
    'mexican petunia',
    'bromelia',
    'blanket flower',
    'trumpet creeper',
    'blackberry lily'
]

def download_data():
    data_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    image_file_name = '102flowers.tgz'
    label_file_name = 'imagelabels.mat'
    setid_file_name = 'setid.mat'

    global data_path, image_path, label_path, setid_path
    image_path = os.path.join(data_path, image_file_name)
    label_path = os.path.join(data_path, label_file_name)
    setid_path = os.path.join(data_path, setid_file_name)
    # download the dataset into current directory
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.isfile(image_path):
        gluon.utils.download(url=data_url + image_file_name, path=data_path)
    if not os.path.exists(os.path.join(data_path, 'jpg')):
        print("Extracting downloaded dataset...")
        tarfile.open(image_path).extractall(path=data_path)
    if not os.path.isfile(label_path):
        gluon.utils.download(url=data_url + label_file_name, path=data_path)
    if not os.path.isfile(setid_path):
        gluon.utils.download(url=data_url + setid_file_name, path=data_path)


def prepare_data():
    # Read .mat file containing training, testing, and validation sets.
    global data_path, image_path, label_path, setid_path, label_names
    setid = loadmat(setid_path)

    idx_train = setid['trnid'][0] - 1
    idx_test = setid['tstid'][0] - 1
    idx_valid = setid['valid'][0] - 1

    # Read .mat file containing image labels.
    image_labels = loadmat(label_path)['labels'][0]

    # Subtract one to get 0-based labels
    image_labels -= 1

    # convert label from number to flower names
    image_labels = [label_names[i] for i in image_labels]
    # extracted images are stored in folder 'jpg'
    files = sorted(glob.glob(os.path.join(data_path, 'jpg', '*.jpg')))
    file_label_pairs = np.array([i for i in zip(files, image_labels)])

    # move files from extracted folder to train, test, valid
    move_files('train', file_label_pairs[idx_test, :])
    move_files('test', file_label_pairs[idx_train, :])
    move_files('valid', file_label_pairs[idx_valid, :])


def move_files(dir_name, file_label_pairs):
    data_segment_dir = os.path.join(data_path, dir_name)
    if not os.path.exists(data_segment_dir):
        os.mkdir(data_segment_dir)

    for label in label_names:
        class_dir = os.path.join(data_segment_dir, label)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    for file, label in file_label_pairs:
        src = str(file)
        dst = os.path.join(data_path, dir_name, label, src.split(os.sep)[-1])
        copyfile(src, dst)


def generate_synset():
    with open('synset.txt', 'w') as f:
        # Gluon Dataset API will load synset in sorted order
        for label in sorted(label_names):
            f.write(label.strip() + '\n')
        f.close()


def get_data(dir_name):
    global data_path
    data_path = dir_name
    download_data()
    prepare_data()
    generate_synset()
