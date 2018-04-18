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

"""Convert Caffe's modelzoo
"""
import os
import argparse
from convert_model import convert_model
from convert_mean import convert_mean
import mxnet as mx

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
_mx_caffe_model_root = '{repo}caffe/models/'.format(repo=repo_url)

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
        'prototxt' : (_mx_caffe_model_root + 'bvlc_alexnet/deploy.prototxt',
                      'cb77655eb4db32c9c47699c6050926f9e0fc476a'),
        'caffemodel' : (_mx_caffe_model_root + 'bvlc_alexnet/bvlc_alexnet.caffemodel',
                        '9116a64c0fbe4459d18f4bb6b56d647b63920377'),
        'mean' : (_mx_caffe_model_root + 'bvlc_alexnet/imagenet_mean.binaryproto',
                  '63e4652e656abc1e87b7a8339a7e02fca63a2c0c'),
        'top-1-acc' : 0.571,
        'top-5-acc' : 0.802
    },
    'bvlc_googlenet' : {
        'prototxt' : (_mx_caffe_model_root + 'bvlc_googlenet/deploy.prototxt',
                      '7060345c8012294baa60eeb5901d2d3fd89d75fc'),
        'caffemodel' : (_mx_caffe_model_root + 'bvlc_googlenet/bvlc_googlenet.caffemodel',
                        '405fc5acd08a3bb12de8ee5e23a96bec22f08204'),
        'mean' : (123, 117, 104),
        'top-1-acc' : 0.687,
        'top-5-acc' : 0.889
    },
    'vgg-16' : {
        'prototxt' : (_mx_caffe_model_root + 'vgg/VGG_ILSVRC_16_layers_deploy.prototxt',
                      '2734e5500f1445bd7c9fee540c99f522485247bd'),
        'caffemodel' : (_mx_caffe_model_root + 'vgg/VGG_ILSVRC_16_layers.caffemodel',
                        '9363e1f6d65f7dba68c4f27a1e62105cdf6c4e24'),
        'mean': (123.68, 116.779, 103.939),
        'top-1-acc' : 0.734,
        'top-5-acc' : 0.914
    },
    'vgg-19' : {
        'prototxt' : (_mx_caffe_model_root + 'vgg/VGG_ILSVRC_19_layers_deploy.prototxt',
                      '132d2f60b3d3b1c2bb9d3fdb0c8931a44f89e2ae'),
        'caffemodel' : (_mx_caffe_model_root + 'vgg/VGG_ILSVRC_19_layers.caffemodel',
                        '239785e7862442717d831f682bb824055e51e9ba'),
        'mean' : (123.68, 116.779, 103.939),
        'top-1-acc' : 0.731,
        'top-5-acc' : 0.913
    },
    'resnet-50' : {
        'prototxt' : (_mx_caffe_model_root + 'resnet/ResNet-50-deploy.prototxt',
                      '5d6fd5aeadd8d4684843c5028b4e5672b9e51638'),
        'caffemodel' : (_mx_caffe_model_root + 'resnet/ResNet-50-model.caffemodel',
                        'b7c79ccc21ad0479cddc0dd78b1d20c4d722908d'),
        'mean' : (_mx_caffe_model_root + 'resnet/ResNet_mean.binaryproto',
                  '0b056fd4444f0ae1537af646ba736edf0d4cefaf'),
        'top-1-acc' : 0.753,
        'top-5-acc' : 0.922
    },
    'resnet-101' : {
        'prototxt' : (_mx_caffe_model_root + 'resnet/ResNet-101-deploy.prototxt',
                      'c165d6b6ccef7cc39ee16a66f00f927f93de198b'),
        'caffemodel' : (_mx_caffe_model_root + 'resnet/ResNet-101-model.caffemodel',
                        '1dbf5f493926bb9b6b3363b12d5133c0f8b78904'),
        'mean' : (_mx_caffe_model_root + 'resnet/ResNet_mean.binaryproto',
                  '0b056fd4444f0ae1537af646ba736edf0d4cefaf'),
        'top-1-acc' : 0.764,
        'top-5-acc' : 0.929
    },
    'resnet-152' : {
        'prototxt' : (_mx_caffe_model_root + 'resnet/ResNet-152-deploy.prototxt',
                      'ae15aade2304af8a774c5bfb1d32457f119214ef'),
        'caffemodel' : (_mx_caffe_model_root + 'resnet/ResNet-152-model.caffemodel',
                        '251edb93604ac8268c7fd2227a0f15144310e1aa'),
        'mean' : (_mx_caffe_model_root + 'resnet/ResNet_mean.binaryproto',
                  '0b056fd4444f0ae1537af646ba736edf0d4cefaf'),
        'top-1-acc' : 0.77,
        'top-5-acc' : 0.933
    },
}

def get_model_meta_info(model_name):
    """returns a dict with model information"""
    return model_meta_info[model_name].copy()

def download_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download caffe model into disk by the given meta info """
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    model_name = os.path.join(dst_dir, model_name)

    assert 'prototxt' in meta_info, "missing prototxt url"
    proto_url, proto_sha1 = meta_info['prototxt']
    prototxt = mx.gluon.utils.download(proto_url,
                                       model_name+'_deploy.prototxt',
                                       sha1_hash=proto_sha1)

    assert 'caffemodel' in meta_info, "mssing caffemodel url"
    caffemodel_url, caffemodel_sha1 = meta_info['caffemodel']
    caffemodel = mx.gluon.utils.download(caffemodel_url,
                                         model_name+'.caffemodel',
                                         sha1_hash=caffemodel_sha1)
    assert 'mean' in meta_info, 'no mean info'
    mean = meta_info['mean']
    if isinstance(mean[0], str):
        mean_url, mean_sha1 = mean
        mean = mx.gluon.utils.download(mean_url,
                                       model_name+'_mean.binaryproto',
                                       sha1_hash=mean_sha1)
    return (prototxt, caffemodel, mean)

def convert_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download, convert and save a caffe model"""

    (prototxt, caffemodel, mean) = download_caffe_model(model_name, meta_info, dst_dir)
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
