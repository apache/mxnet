## MxNet-TensorRT runtime integration examples

**Note:** The TensorRT integration API documentation can be found [here](../../../docs/api/python/contrib/tensorrt.md). Please read it carefully as it describes how to build MxNet with TensorRT, and how to adapt existing pre-trained models for both the symboli API and for Gluon.

**Note:** To use this feature, MxNet needs to be built with TensorRT support. Please see the details as to how to build a Docker container with TensorRT support for the x86_64 platform [here](../../../docs/api/python/contrib/tensorrt.md).

**Note:** This example uses pre-trained models from the [Gluon model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html). In order to use it, please install the [gluoncv](https://pypi.org/project/gluoncv/) pip package as follows:
```
pip install gluoncv
```

The following example shows how to run image classification models using pure MxNet for inference, followed by MxNet with TensorRT integration, comparing performance and accuracy. The models in question are:

* cifar_resnet20_v1
* cifar_resnet56_v1
* cifar_resnet110_v1
* cifar_resnet20_v2
* cifar_resnet56_v2
* cifar_resnet110_v2
* cifar_wideresnet16_10
* cifar_wideresnet28_10
* cifar_wideresnet40_8
* cifar_resnext29_16x64d

Please run the example as follows:
```bash
python ${MXNET_HOME}/tests/python/tensorrt/test_tensorrt_resnet_resnext.py
```
