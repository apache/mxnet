---
layout: page_category
title:  Convert from Caffe to MXNet
category: faq
faq_c: Deployment Environments
question: How to convert a Caffe model to MXNet?
permalink: /api/faq/caffe
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# How to | Convert from Caffe to MXNet

Key topics covered include the following:

- [Calling Caffe operators in MXNet](#calling-caffe-operators-in-mxnet)

## Calling Caffe operators in MXNet

MXNet supports calling most Caffe operators,
including network layer, data layer, and loss function, directly. It is
particularly useful if there are customized operators implemented in Caffe, then
we do not need to re-implement them in MXNet.

### How to install

This feature requires Caffe. In particular, we need to re-compile Caffe before
[PR #4527](https://github.com/BVLC/caffe/pull/4527) is merged into Caffe. There
are the steps of how to rebuild Caffe:

1. Download [Caffe](https://github.com/BVLC/caffe). E.g. `git clone
   https://github.com/BVLC/caffe`
2. Download the
   [patch for the MXNet interface](https://github.com/BVLC/caffe/pull/4527.patch)
   and apply to Caffe. E.g.
   ```bash
   cd caffe && wget https://github.com/BVLC/caffe/pull/4527.patch && git apply 4527.patch
   ```
3. Build and install Caffe by following the
   [official guide](https://caffe.berkeleyvision.org/installation.html).

Next we need to compile MXNet with Caffe supports

1. Copy `make/config.mk` (for Linux) or `make/osx.mk`
   (for Mac) into the MXNet root folder as `config.mk` if you have not done it yet
2. Open the copied `config.mk` and uncomment these two lines
   ```bash
   CAFFE_PATH = $(HOME)/caffe
   MXNET_PLUGINS += plugin/caffe/caffe.mk
   ```
   Modify `CAFFE_PATH` to your Caffe installation, if necessary.
3. Then build with 8 threads `make clean && make -j8`.

### How to use

This Caffe plugin adds three components into MXNet:

- `sym.CaffeOp` : Caffe neural network layer
- `sym.CaffeLoss` : Caffe loss functions
- `io.CaffeDataIter` : Caffe data layer

#### Use `sym.CaffeOp`
The following example shows the definition of a 10 classes multi-layer perceptron:

```Python
data = mx.sym.Variable('data')
fc1  = mx.sym.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
act1 = mx.sym.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
fc2  = mx.sym.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
act2 = mx.sym.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
fc3 = mx.sym.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
```

Let's break it down. First, `data = mx.sym.Variable('data')` defines a variable
as a placeholder for input.  Then, it's fed through Caffe operators with `fc1 =
mx.sym.CaffeOp(...)`. `CaffeOp` accepts several arguments:

- The inputs to Caffe operators are named as `data_i` for *i=0, ..., num_data-1*
- `num_data` is the number of inputs. In default it is 1, and therefore
skipped in the above example.
- `num_out` is the number of outputs. In default it is 1 and also skipped.
- `num_weight` is the number of weights (`blobs_`).  Its default value is 0. We
need to explicitly specify it for a non-zero value.
- `prototxt` is the protobuf configuration string.

#### Use `sym.CaffeLoss`

Using Caffe loss is similar.
We can replace the MXNet loss with Caffe loss.
We can replace

Replacing the last line of the above example with the following two lines we can
call Caffe loss instead of MXNet loss.

```Python
label = mx.sym.Variable('softmax_label')
mlp = mx.sym.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
```

Similar to `CaffeOp`, `CaffeLoss` has arguments `num_data` (2 in default) and
`num_out` (1 in default). But there are two differences

1. Inputs are `data` and `label`. And we need to explicitly create a variable
   placeholder for label, which is implicitly done in MXNet loss.
2. `grad_scale` is the weight of this loss.

#### Use `io.CaffeDataIter`

We can also wrap a Caffe data layer into MXNet's data iterator. Below is an
example for creating a data iterator for MNIST

```python
train = mx.io.CaffeDataIter(
    prototxt =
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
            source: "caffe/examples/mnist/mnist_test_lmdb" \
            batch_size: 100 \
            backend: LMDB \
        } \
    }',
    flat           = flat,
    num_examples   = 60000,
)
```
