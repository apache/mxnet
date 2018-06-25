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

from __future__ import print_function

import os.path
import subprocess
import mxnet as mx
import numpy as np
from time import time
import sys
import urllib

def get_use_tensorrt():
    return int(os.environ.get("MXNET_USE_TENSORRT", 0))

def set_use_tensorrt(status = False):
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))

def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def download_cifar10(data_dir):
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def get_cifar10_iterator(args, kv):
    data_shape = (3, 32, 32) #28, 28) 
    data_dir = args['data_dir']
    if os.name == "nt":
        data_dir = data_dir[:-1] + "\\"
    if '://' not in args['data_dir']:
        print("Did not find data.")
        download_cifar10(data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "cifar10_train.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        data_shape  = data_shape,
        batch_size  = args['batch_size'],
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv['num_workers'],
        part_index  = kv['rank'])

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "cifar10_val.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args['batch_size'],
        num_parts   = kv['num_workers'],
        part_index  = kv['rank'])

    return (train, val)


# To support Python 2 and 3.x < 3.5
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_exec(model_prefix='resnet50', image_size=(32, 32), batch_size = 128, ctx=mx.gpu(0), epoch=1):

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)

    h, w = image_size
    data_shape=(batch_size, 3, h, w)
    sm_shape=(batch_size,)

    data = mx.sym.Variable("data")
    softmax_label = mx.sym.Variable("softmax_label")

    all_params = merge_dicts(arg_params, aux_params)

    if not get_use_tensorrt():
        all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in all_params.items()])

    executor = sym.simple_bind(ctx=ctx, data = data_shape,
        softmax_label=sm_shape, grad_req='null', shared_buffer=all_params, force_rebind=True)

    return executor, h, w

def compute(model_prefix, epoch, data_dir, batch_size=128):

    executor, h, w = get_exec(model_prefix=model_prefix,
                              image_size=(32, 32), 
                              batch_size=batch_size, 
                              ctx=mx.gpu(0),
                              epoch=epoch)
    num_ex = 10000
    all_preds = np.zeros([num_ex, 10])

    train_iter, test_iter = get_cifar10_iterator(args={'data_dir':data_dir, 'batch_size':batch_size}, kv={'num_workers':1, 'rank':0})

    train_iter2, test_iter2 = get_cifar10_iterator(args={'data_dir':data_dir, 'batch_size':num_ex}, kv={'num_workers':1, 'rank':0})

    all_label_train = train_iter2.next().label[0].asnumpy()
    all_label_test = test_iter2.next().label[0].asnumpy().astype(np.int32)

    train_iter, test_iter = get_cifar10_iterator(args={'data_dir':'./data', 'batch_size':batch_size}, kv={'num_workers':1, 'rank':0})

    start = time()

    example_ct = 0

    for idx, dbatch in enumerate(test_iter):
        data = dbatch.data[0]
        executor.arg_dict["data"][:] = data
        executor.forward(is_train=False)
        preds = executor.outputs[0].asnumpy()
        offset = idx*batch_size
        extent = batch_size if num_ex - offset > batch_size else num_ex - offset
        all_preds[offset:offset+extent, :] = preds[:extent]
        example_ct += extent

    all_preds = np.argmax(all_preds, axis=1)

    matches = (all_preds[:example_ct] == all_label_test[:example_ct]).sum()

    percentage = 100.0 * matches / example_ct

    return percentage, time() - start

if __name__ == '__main__':

    model_prefix = sys.argv[1]
    epoch = int(sys.argv[2])
    data_dir = sys.argv[3]
    batch_size = 1024

    print("\nRunning inference in MXNet\n")
    set_use_tensorrt(False)
    mxnet_pct, mxnet_time = compute(model_prefix, epoch, data_dir, batch_size)

    print("\nRunning inference in MXNet-TensorRT\n")
    set_use_tensorrt(True)
    trt_pct, trt_time = compute(model_prefix, epoch, data_dir, batch_size)

    print("MXNet time: %f" % mxnet_time)
    print("MXNet-TensorRT time: %f" % trt_time)
    print("Speedup: %fx" % (mxnet_time / trt_time))

