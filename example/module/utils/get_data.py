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
import mxnet as mx

def get_mnist(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        import urllib, zipfile
        zippath = os.path.join(os.getcwd(), "mnist.zip")
        urllib.urlretrieve("http://data.mxnet.io/mxnet/data/mnist.zip", zippath)
        zf = zipfile.ZipFile(zippath, "r")
        zf.extractall()
        zf.close()
        os.remove(zippath)
    os.chdir("..")

def get_cifar10(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    cwd = os.path.abspath(os.getcwd())
    os.chdir(data_dir)
    if (not os.path.exists('train.rec')) or \
       (not os.path.exists('test.rec')) :
        import urllib, zipfile, glob
        dirname = os.getcwd()
        zippath = os.path.join(dirname, "cifar10.zip")
        urllib.urlretrieve("http://data.mxnet.io/mxnet/data/cifar10.zip", zippath)
        zf = zipfile.ZipFile(zippath, "r")
        zf.extractall()
        zf.close()
        os.remove(zippath)
        for f in glob.glob(os.path.join(dirname, "cifar", "*")):
            name = f.split(os.path.sep)[-1]
            os.rename(f, os.path.join(dirname, name))
        os.rmdir(os.path.join(dirname, "cifar"))
    os.chdir(cwd)

# data
def get_cifar10_iterator(args, kv):
    data_shape = (3, 28, 28)
    data_dir = args.data_dir
    if os.name == "nt":
        data_dir = data_dir[:-1] + "\\"
    if '://' not in args.data_dir:
        get_cifar10(data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "train.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "test.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)
