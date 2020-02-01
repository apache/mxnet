#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
test pretrained models
"""
from __future__ import print_function
import mxnet as mx
from common import find_mxnet, modelzoo
from score import score

VAL_DATA='data/val-5k-256.rec'
def download_data():
    return mx.test_utils.download(
        'http://data.mxnet.io/data/val-5k-256.rec', VAL_DATA)

def test_imagenet1k_resnet(**kwargs):
    models = ['imagenet1k-resnet-50', 'imagenet1k-resnet-152']
    accs = [.77, .78]
    for (m, g) in zip(models, accs):
        acc = mx.metric.create('acc')
        (speed,) = score(model=m, data_val=VAL_DATA,
                         rgb_mean='0,0,0', metrics=acc, **kwargs)
        r = acc.get()[1]
        print('Tested %s, acc = %f, speed = %f img/sec' % (m, r, speed))
        assert r > g and r < g + .1

def test_imagenet1k_inception_bn(**kwargs):
    acc = mx.metric.create('acc')
    m = 'imagenet1k-inception-bn'
    g = 0.75
    (speed,) = score(model=m,
                     data_val=VAL_DATA,
                     rgb_mean='123.68,116.779,103.939', metrics=acc, **kwargs)
    r = acc.get()[1]
    print('Tested %s acc = %f, speed = %f img/sec' % (m, r, speed))
    assert r > g and r < g + .1

if __name__ == '__main__':
    num_gpus = mx.context.num_gpus()
    assert num_gpus > 0
    batch_size = 16 * num_gpus
    gpus = ','.join(map(str, range(num_gpus)))
    kwargs = {'gpus':gpus, 'batch_size':batch_size, 'max_num_examples':500}
    download_data()
    test_imagenet1k_resnet(**kwargs)
    test_imagenet1k_inception_bn(**kwargs)
