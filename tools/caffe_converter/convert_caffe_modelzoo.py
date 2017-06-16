"""Convert Caffe's modelzoo
"""
import os
import argparse
from convert_model import convert_model
from convert_mean import convert_mean
import mxnet as mx

_mx_caffe_model = 'http://data.mxnet.io/models/imagenet/test/caffe/'

"""Dictionary for model meta information

For each model, it requires three attributes:

  - prototxt: URL for the deploy prototxt file
  - caffemodel: URL for the binary caffemodel
  - mean : URL for the data mean or a tuple of float

Optionly it takes

  - top-1-acc : top 1 accuracy for testing
  - top-5-acc : top 5 accuracy for testing
"""
model_meta_info = {
    # pylint: disable=line-too-long
    'bvlc_alexnet' : {
        'prototxt' : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        'mean' : 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/caffe/imagenet_mean.binaryproto',
        'top-1-acc' : 0.571,
        'top-5-acc' : 0.802
    },
    'bvlc_googlenet' : {
        'prototxt' : 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt',
        'caffemodel' : 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        'mean' : (123, 117, 104),
        'top-1-acc' : 0.687,
        'top-5-acc' : 0.889
    },
    'vgg-16' : {
        'prototxt' : 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt',
        # 'caffemodel' : 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel',
        'caffemodel' : 'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_16_layers.caffemodel',
        'mean': (123.68, 116.779, 103.939),
        'top-1-acc' : 0.734,
        'top-5-acc' : 0.914
    },
    'vgg-19' : {
        'prototxt' : 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt',
        # 'caffemodel' : 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel',
        'caffemodel' : 'http://data.mxnet.io/models/imagenet/test/caffe/VGG_ILSVRC_19_layers.caffemodel',
        'mean' : (123.68, 116.779, 103.939),
        'top-1-acc' : 0.731,
        'top-5-acc' : 0.913
    },
    'resnet-50' : {
        'prototxt' : _mx_caffe_model+'ResNet-50-deploy.prototxt',
        'caffemodel' : _mx_caffe_model+'ResNet-50-model.caffemodel',
        'mean' : _mx_caffe_model+'ResNet_mean.binaryproto',
        'top-1-acc' : 0.753,
        'top-5-acc' : 0.922
    },
    'resnet-101' : {
        'prototxt' : _mx_caffe_model+'ResNet-101-deploy.prototxt',
        'caffemodel' : _mx_caffe_model+'ResNet-101-model.caffemodel',
        'mean' : _mx_caffe_model+'ResNet_mean.binaryproto',
        'top-1-acc' : 0.764,
        'top-5-acc' : 0.929
    },
    'resnet-152' : {
        'prototxt' : _mx_caffe_model+'ResNet-152-deploy.prototxt',
        'caffemodel' : _mx_caffe_model+'ResNet-152-model.caffemodel',
        'mean' : _mx_caffe_model+'ResNet_mean.binaryproto',
        'top-1-acc' : 0.77,
        'top-5-acc' : 0.933
    },
}

def get_model_meta_info(model_name):
    """returns a dict with model information"""
    return dict(dict(model_meta_info)[model_name])

def _download_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download caffe model into disk by the given meta info """
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    model_name = os.path.join(dst_dir, model_name)
    assert 'prototxt' in meta_info, "missing prototxt url"
    prototxt = mx.test_utils.download(meta_info['prototxt'], model_name+'_deploy.prototxt')
    assert 'caffemodel' in meta_info, "mssing caffemodel url"
    caffemodel = mx.test_utils.download(meta_info['caffemodel'], model_name+'.caffemodel')
    assert 'mean' in meta_info, 'no mean info'
    mean = meta_info['mean']
    if isinstance(mean, str):
        mean = mx.test_utils.download(mean, model_name+'_mean.binaryproto')
    return (prototxt, caffemodel, mean)

def convert_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download, convert and save a caffe model"""

    (prototxt, caffemodel, mean) = _download_caffe_model(model_name, meta_info, dst_dir)
    model_name = os.path.join(dst_dir, model_name)
    convert_model(prototxt, caffemodel, model_name)
    if isinstance(mean, str):
        mx_mean = model_name + '-mean.nd'
        convert_mean(mean, mx_mean)
        mean = mx_mean
    return (model_name, mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Caffe model zoo')
    parser.add_argument('model_name', help='can be '+', '.join(model_meta_info.keys()))
    args = parser.parse_args()
    assert args.model_name in model_meta_info, 'Unknown model ' + args.model_name
    fname, _ = convert_caffe_model(args.model_name, model_meta_info[args.model_name])
    print('Model is saved into ' + fname)
