# BMXNet v2 // Hasso Plattner Institute

A fork of the deep learning framework [mxnet](http://mxnet.io) to study and implement quantization and binarization in neural networks.

This project is based on the [first version of BMXNet](https://github.com/hpi-xnor/BMXNet), but is different in that it reuses more of the original MXNet operators.
This aim was to have only minimal changes to C++ code to get better maintainability with future versions of mxnet.

## News

- **Sep 01, 2018** - MXNet v1.3.0
    - We are currently rebuilding BMXNet to utilize the new Gluon API for better maintainability
    - To build binary neural networks, you can use drop in replacements of convolution and dense layers (see [Usage](#usage)):
    - Changes are now documented in the [Changelog](CHANGELOG.md)
    - Note that this project is still in beta and changes might be frequent
    - We do not yet support deployment and inference with binary operations and models (please use the [first version of BMXNet](https://github.com/hpi-xnor/BMXNet) instead if you need this).

# Setup

We use [CMake](https://cmake.org/download/) to build the project.
Make sure to install all the dependencies described [here](docs/install/build_from_source.md#prerequisites).
If you install CUDA 10, you will need CMake >=3.12.2

Adjust settings in cmake (build-type ``Release`` or ``Debug``, configure CUDA, OpenBLAS or Atlas, OpenCV, OpenMP etc.).

Further, we recommend [Ninja](https://ninja-build.org/) as a build system for faster builds (Ubuntu: `sudo apt-get install ninja-build`).

```bash
git clone --recursive https://github.com/hpi-xnor/BMXNet-v2.git # remember to include the --recursive
cd BMXNet-v2
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
```

Step 1b (Optional) Create or activate a [virtualenv](https://virtualenv.pypa.io/).

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

Make sure that you have a new version of our example submodule [example/bmxnet-examples](https://github.com/hpi-xnor/BMXNet-v2-examples/):
```bash
cd example/bmxnet-examples
git checkout master
git pull
```

Examples for hyperparameters are documented in the [Wiki](https://github.com/hpi-xnor/BMXNet-v2-wiki/blob/master/hyperparameters.md).

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

We added three functions `det_sign` ([ada4ea1d](https://github.com/hpi-xnor/BMXNet-v2/commit/ada4ea1d4418cfdd6cbc6d0159e1a716cb01cd85)), `round_ste` ([044f81f0](https://github.com/hpi-xnor/BMXNet-v2/commit/044f81f028887b9842070df28b28de394bd07516)) and `contrib.gradcancel` to MXNet (see [src/operator/contrib/gradient_cancel[-inl.h|.cc|.cu]](src/operator/contrib)).

The rest of our code resides in the following folders/files:
- Examples are in a submodule in [example/bmxnet-examples](https://github.com/hpi-xnor/BMXNet-v2-examples)
- Tests are in [tests/python/unittest/test_binary.py](tests/python/unittest/test_binary.py)
- Layers are in [python/mxnet/gluon/nn/binary_layers.py](python/mxnet/gluon/nn/binary_layers.py)

For more details see the [Changelog](CHANGELOG.md).

### Citing BMXNet

Please cite our paper about BMXNet v2 in your publications if it helps your research work:

```text
@article{bmxnetv2,
  title = {Training Competitive Binary Neural Networks from Scratch},
  author = {Joseph Bethge and Marvin Bornstein and Adrian Loy and Haojin Yang and Christoph Meinel},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1812.01965},
  Year = {2018}
}
```

### References

- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
