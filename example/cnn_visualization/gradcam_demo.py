import argparse

import mxnet as mx
from mxnet import gluon

import numpy as np
import cv2

import vgg
import gradcam

parser = argparse.ArgumentParser(description='Grad-CAM demo')
parser.add_argument('img_path', metavar='image_path', type=str, help='path to the image file')

args = parser.parse_args()

print(args.img_path)


network = vgg.vgg16(pretrained=True, ctx=mx.cpu())

image_sz = (224, 224)

def preprocess(data):
    data = mx.image.imresize(data, image_sz[0], image_sz[1])
    data = data.astype(np.float32)
    data = data/255
    data = mx.image.color_normalize(data,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    data = mx.nd.transpose(data, (2,0,1))
    return data

def read_image_mxnet(path):
    with open(path, 'rb') as fp:
        img_bytes = fp.read()
    return mx.img.imdecode(img_bytes)

def read_image_cv(path):
    return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), image_sz)

synset = []
with open('synset.txt', 'r') as f:
    synset = [l.rstrip().split(' ', 1)[1].split(',')[0] for l in f]
    
def get_class_name(cls_id):
    return "%s (%d)" % (synset[cls_id], cls_id)

def run_inference(net, data):
    out = net(data)
    return out.argmax(axis=1).asnumpy()[0].astype(int)

def visualize(net, img_path, conv_layer_name):
    image = read_image_mxnet(img_path)
    image = preprocess(image)
    image = image.expand_dims(axis=0)
    
    pred_str = get_class_name(run_inference(net, image))
    
    orig_img = read_image_cv(img_path)
    vizs = gradcam.visualize(net, image, orig_img, conv_layer_name)
    return (pred_str, (orig_img, *vizs))

last_conv_layer_name = 'vgg0_conv2d12'

cat, vizs = visualize(network, "img/hummingbird.jpg", last_conv_layer_name)

for i, img in enumerate(vizs):
    cv2.imwrite("%d.jpg" % i, img)


