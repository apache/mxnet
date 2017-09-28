# How to | Convert from Caffe to MXNet

Key topics covered include the following:

- [Converting Caffe trained models to MXNet](#converting-caffe-trained-models-to-mxnet)
- [Calling Caffe operators in MXNet](#calling-caffe-operators-in-mxnet)

## Converting Caffe trained models to MXNet

The converting tool is available at
[tools/caffe_converter](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter). On
the remaining of this section, we assume we are on the `tools/caffe_converter`
directory.

### How to build

If Caffe's python package is installed, namely we can run `import caffe` in
python, then we are ready to go.

For example, we can used
[AWS Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B06VSPXKDX) with
both Caffe and MXNet installed.

Otherwise we can install the
[Google protobuf](https://developers.google.com/protocol-buffers/?hl=en)
compiler and its python binding. It is easier to install, but may be slower
during running.

1. Install the compiler:
  - Linux: install `protobuf-compiler` e.g. `sudo apt-get install
    protobuf-compiler` for Ubuntu and `sudo yum install protobuf-compiler` for
     Redhat/Fedora.
  - Windows: Download the win32 build of
    [protobuf](https://github.com/google/protobuf/releases). Make sure to
    download the version that corresponds to the version of the python binding
    on the next step. Extract to any location then add that location to your
    `PATH`
  - Mac OS X: `brew install protobuf`

2. Install the python binding by either `conda install -c conda-forge protobuf`
   or `pip install protobuf`.

3. Compile Caffe proto definition. Run `make` in Linux or Mac OS X, or
   `make_win32.bat` in Windows

### How to use

There are three tools:

- `convert_symbol.py` : convert Caffe model definition in protobuf into MXNet's
  Symbol in JSON format.
- `convert_model.py` : convert Caffe model parameters into MXNet's NDArray format
- `convert_mean.py` : convert Caffe input mean file into MXNet's NDArray format

In addition, there are two tools:
- `convert_caffe_modelzoo.py` : download and convert models from Caffe model
  zoo.
- `test_converter.py` : test the converted models by checking the prediction
  accuracy.

## Calling Caffe operators in MXNet

Besides converting Caffe models, MXNet supports calling most Caffe operators,
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
   [official guide](http://caffe.berkeleyvision.org/installation.html).

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

### Put it all together

The complete example is available at
[example/caffe](https://github.com/dmlc/mxnet/blob/master/example/caffe/)
