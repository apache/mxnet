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

import os
import numpy as np
import mxnet as mx
from mxnet.test_utils import *
from mxnet.gluon import utils

def _get_model():
    if not os.path.exists('model/Inception-7-symbol.json'):
        download('http://data.mxnet.io/models/imagenet/inception-v3.tar.gz', dirname='model')
        os.system("cd model; tar -xf inception-v3.tar.gz --strip-components 1")

def _dump_images(shape):
    import skimage.io
    import skimage.transform
    img_list = []
    for img in sorted(os.listdir('data/test_images/')):
        img = skimage.io.imread('data/test_images/'+img)
        short_egde = min(img.shape[:2])
        yy = int((img.shape[0] - short_egde) / 2)
        xx = int((img.shape[1] - short_egde) / 2)
        img = img[yy : yy + short_egde, xx : xx + short_egde]
        img = skimage.transform.resize(img, shape)
        img_list.append(img)
    imgs = np.asarray(img_list, dtype=np.float32).transpose((0, 3, 1, 2)) - 128
    np.save('data/test_images_%d_%d.npy'%shape, imgs)

def _get_data(shape):
    hash_test_img = "355e15800642286e7fe607d87c38aeeab085b0cc"
    hash_inception_v3 = "91807dfdbd336eb3b265dd62c2408882462752b9"
    utils.download("http://data.mxnet.io/data/test_images_%d_%d.npy" % (shape),
                   path="data/test_images_%d_%d.npy" % (shape),
                   sha1_hash=hash_test_img)
    utils.download("http://data.mxnet.io/data/inception-v3-dump.npz",
                   path='data/inception-v3-dump.npz',
                   sha1_hash=hash_inception_v3)

def test_consistency(dump=False):
    shape = (299, 299)
    _get_model()
    _get_data(shape)
    if dump:
        _dump_images(shape)
        gt = None
    else:
        gt = {n: mx.nd.array(a) for n, a in np.load('data/inception-v3-dump.npz').items()}
    data = np.load('data/test_images_%d_%d.npy'%shape)
    sym, arg_params, aux_params = mx.model.load_checkpoint('model/Inception-7', 1)
    arg_params['data'] = data
    arg_params['softmax_label'] = np.random.randint(low=1, high=1000, size=(data.shape[0],))
    ctx_list = [{'ctx': mx.gpu(0), 'data': data.shape, 'type_dict': {'data': data.dtype}},
                {'ctx': mx.cpu(0), 'data': data.shape, 'type_dict': {'data': data.dtype}}]
    gt = check_consistency(sym, ctx_list, arg_params=arg_params, aux_params=aux_params,
                           tol=1e-3, grad_req='null', raise_on_err=False, ground_truth=gt)
    if dump:
        np.savez('data/inception-v3-dump.npz', **{n: a.asnumpy() for n, a in gt.items()})

if __name__ == '__main__':
    test_consistency(False)
