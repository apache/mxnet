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
"""Create the helper functions to mnist dataset for Caffe operators in MXNet"""
import mxnet as mx
from mxnet.test_utils import get_mnist_ubyte


def get_iterator(data_shape, use_caffe_data):
    """Generate the iterator of mnist dataset"""
    def get_iterator_impl_mnist(args, kv):
        """return train and val iterators for mnist"""
        # download data
        get_mnist_ubyte()
        flat = False if len(data_shape) != 1 else True

        train = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            input_shape=data_shape,
            batch_size=args.batch_size,
            shuffle=True,
            flat=flat,
            num_parts=kv.num_workers,
            part_index=kv.rank)

        val = mx.io.MNISTIter(
            image="data/t10k-images-idx3-ubyte",
            label="data/t10k-labels-idx1-ubyte",
            input_shape=data_shape,
            batch_size=args.batch_size,
            flat=flat,
            num_parts=kv.num_workers,
            part_index=kv.rank)

        return (train, val)

    def get_iterator_impl_caffe(args, kv):
        flat = False if len(data_shape) != 1 else True
        train = mx.io.CaffeDataIter(
            prototxt=
            'layer { \
                name: "mnist" \
                type: "Data" \
                top: "data" \
                top: "label" \
                include { \
                    phase: TRAIN \
                } \
                transform_param { \
                    scale: 0.00390625 \
                } \
                data_param { \
                    source: "mnist_train_lmdb" \
                    batch_size: 64 \
                    backend: LMDB \
                } \
            }',
            flat=flat,
            num_examples=60000
            # float32 is the default, so left out here in order to illustrate
        )

        val = mx.io.CaffeDataIter(
            prototxt=
            'layer { \
                name: "mnist" \
                type: "Data" \
                top: "data" \
                top: "label" \
                include { \
                    phase: TEST \
                } \
                transform_param { \
                    scale: 0.00390625 \
                } \
                data_param { \
                    source: "mnist_test_lmdb" \
                    batch_size: 100 \
                    backend: LMDB \
                } \
            }',
            flat=flat,
            num_examples=10000,
            dtype="float32"  # float32 is the default
        )

        return train, val

    if use_caffe_data:
        return get_iterator_impl_caffe
    else:
        return get_iterator_impl_mnist
