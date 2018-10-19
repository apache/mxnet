# binary neural nets with mxnet // Hasso Plattner Institute

A fork of the deep learning framework [mxnet](http://mxnet.io) to study and implement quantization and binarization in neural networks.

Our current efforts are focused on binarizing the inputs and weights of convolutional layers, enabling the use of performant bit operations instead of expensive matrix multiplications as described in, however this related to an older version of BMXNet:

- [BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet](https://arxiv.org/abs/1705.09864)

## News

- **Sep 01, 2018** - MXNet v1.3.0
    - We are currently rebuilding BMXNet to utilize the new Gluon API for better maintainability
    - To build binary neural networks, you can use drop in replacements of convolution and dense layers (see [Usage](#usage)):
    - Changes are now documented in the [Changelog](CHANGELOG.md)
- **Dec 22, 2017** - MXNet v1.0.0 and cuDNN
    - We are updating the underlying MXNet to version 1.0.0, see changes and release notes [here](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0).
    - cuDNN is now supported in the training of binary networks, speeding up the training process by about 2x

# Setup

We use [CMake](https://cmake.org/download/) to build the project.
Make sure to install all the dependencies described [here](docs/install/build_from_source.md#prerequisites).
If you install CUDA 10, you will need CMake >=3.12.2

Adjust settings in cmake (build-type ``Release`` or ``Debug``, configure CUDA, OpenBLAS or Atlas, OpenCV, OpenMP etc.).

Further, we recommend [Ninja](https://ninja-build.org/) as a build system for faster builds (Ubuntu: `sudo apt-get install ninja-build`).

```bash
git clone --recursive git@gitlab.hpi.de:joseph.bethge/bmxnet.git # remember to include the --recursive
mkdir build && cd build
cmake .. -G Ninja # if any error occurs, apply ccmake or cmake-gui to adjust the cmake config.
ccmake . # or GUI cmake
ninja
```

#### Build the MXNet Python binding

Step 1 Install prerequisites - python, setup-tools, python-pip and numpy.
```bash
sudo apt-get install -y python-dev python3-dev virtualenv
wget -nv https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
python2 get-pip.py
# optionally create a virtualenv before this step
pip2 install nose cpplint==1.3.0 pylint==1.9.3 'numpy<=1.15.2,>=1.8.2' nose-timer 'requests<2.19.0,>=2.18.4' h5py==2.8.0rc1 scipy==1.0.1 boto3
pip3 install nose cpplint==1.3.0 pylint==2.1.1 'numpy<=1.15.2,>=1.8.2' nose-timer 'requests<2.19.0,>=2.18.4' h5py==2.8.0rc1 scipy==1.0.1 boto3
```

Step 2 Install the MXNet Python binding.
```bash
cd <mxnet-root>/python
pip install -e .
```

If your mxnet python binding still not works, you can add the location of the libray to your ``LD_LIBRARY_PATH`` as well as the mxnet python folder to your ``PYTHONPATH``:
```bash
$ export LD_LIBRARY_PATH=<mxnet-root>/build/Release
$ export PYTHONPATH=<mxnet-root>/python
```

## Training

Make sure that you have a new version of our example submodule [example/bmxnet-examples](example/bmxnet-examples):
```bash
cd example/bmxnet-examples
git checkout master
git pull
```

The best hyperparameters are documented in the [Wiki](https://gitlab.hpi.de/joseph.bethge/bmxnet/wikis/hyperparameters).

## Tests

To run BMXNet specific tests install `pytest`:
```bash
pip install pytest
```

Then simply run:
```bash
pytest tests/python/unittest/test_binary.py
```

## Usage

We added binary versions of the following layers of the gluon API:
- gluon.nn.Dense -> gluon.nn.QDense
- gluon.nn.Conv1D -> gluon.nn.QConv1D
- gluon.nn.Conv2D -> gluon.nn.QConv2D
- gluon.nn.Conv3D -> gluon.nn.QConv3D

## Overview of Changes

We added three functions `det_sign` ([ada4ea1d](https://gitlab.hpi.de/joseph.bethge/bmxnet/commit/ada4ea1d4418cfdd6cbc6d0159e1a716cb01cd85)), `round_ste` ([044f81f0](https://gitlab.hpi.de/joseph.bethge/bmxnet/commit/044f81f028887b9842070df28b28de394bd07516)) and `contrib.gradcancel` to MXNet (see [src/operator/contrib/gradient_cancel[-inl.h|.cc|.cu]](src/operator/contrib)).

The rest of our code resides in the following folders/files:
- Examples are in a submodule in [example/bmxnet-examples](https://gitlab.hpi.de/joseph.bethge/bmxnet-examples)
- Tests are in [tests/python/unittest/test_binary.py](tests/python/unittest/test_binary.py)
- Layers are in [python/mxnet/gluon/nn/binary_layers.py](python/mxnet/gluon/nn/binary_layers.py)

### Citing BMXNet

Please cite BMXNet in your publications if it helps your research work:

```text
@inproceedings{bmxnet,
 author = {Yang, Haojin and Fritzsche, Martin and Bartz, Christian and Meinel, Christoph},
 title = {BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet},
 booktitle = {Proceedings of the 2017 ACM on Multimedia Conference},
 series = {MM '17},
 year = {2017},
 isbn = {978-1-4503-4906-2},
 location = {Mountain View, California, USA},
 pages = {1209--1212},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3123266.3129393},
 doi = {10.1145/3123266.3129393},
 acmid = {3129393},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {binary neural networks, computer vision, machine learning, open source},
} 

```

### References

- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
