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

import mxnet as mx
from mxnet import gluon

import argparse
import os
import numpy as np
import cv2

import vgg
import gradcam

# Receive image path from command line
parser = argparse.ArgumentParser(description='Grad-CAM demo')
parser.add_argument('img_path', metavar='image_path', type=str, help='path to the image file')

args = parser.parse_args()

# We'll use VGG-16 for visualization
network = vgg.vgg16(pretrained=True, ctx=mx.cpu())
# We'll resize images to 224x244 as part of preprocessing
image_sz = (224, 224)

def preprocess(data):
    """Preprocess the image before running it through the network"""
    data = mx.image.imresize(data, image_sz[0], image_sz[1])
    data = data.astype(np.float32)
    data = data/255
    # These mean values were obtained from
    # https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html
    data = mx.image.color_normalize(data,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    data = mx.nd.transpose(data, (2,0,1)) # Channel first
    return data

def read_image_mxnet(path):
    with open(path, 'rb') as fp:
        img_bytes = fp.read()
    return mx.img.imdecode(img_bytes)

def read_image_cv(path):
    return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), image_sz)

# synset.txt contains the names of Imagenet categories
# Load the file to memory and create a helper method to query category_index -> category name
synset_url = "http://data.mxnet.io/models/imagenet/synset.txt"
synset_file_name = "synset.txt"
mx.test_utils.download(synset_url, fname=synset_file_name)

synset = []
with open('synset.txt', 'r') as f:
    synset = [l.rstrip().split(' ', 1)[1].split(',')[0] for l in f]
    
def get_class_name(cls_id):
    return "%s (%d)" % (synset[cls_id], cls_id)

def run_inference(net, data):
    """Run the input image through the network and return the predicted category as integer"""
    out = net(data)
    return out.argmax(axis=1).asnumpy()[0].astype(int)

def visualize(net, img_path, conv_layer_name):
    """Create Grad-CAM visualizations using the network 'net' and the image at 'img_path'
    conv_layer_name is the name of the top most layer of the feature extractor"""
    image = read_image_mxnet(img_path)
    image = preprocess(image)
    image = image.expand_dims(axis=0)
    
    pred_str = get_class_name(run_inference(net, image))
    
    orig_img = read_image_cv(img_path)
    vizs = gradcam.visualize(net, image, orig_img, conv_layer_name)
    return (pred_str, (orig_img, *vizs))

# Create Grad-CAM visualization for the user provided image
last_conv_layer_name = 'vgg0_conv2d12'
cat, vizs = visualize(network, args.img_path, last_conv_layer_name)

print("{0:20}: {1:80}".format("Predicted category", cat))

# Write the visualiations into file
img_name = os.path.split(args.img_path)[1].split('.')[0]
suffixes = ['orig', 'gradcam', 'guided_gradcam', 'saliency']
image_desc = ['Original Image', 'Grad-CAM', 'Guided Grad-CAM', 'Saliency Map']

for i, img in enumerate(vizs):
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_file_name = "%s_%s.jpg" % (img_name, suffixes[i])
    cv2.imwrite(out_file_name, img)
    print("{0:20}: {1:80}".format(image_desc[i], out_file_name))

