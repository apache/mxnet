# Model Quantization with Calibration Examples

This folder contains examples of quantizing a FP32 model with Intel® MKL-DNN or CUDNN.

<h2 id="0">Contents</h2>

* [1. Model Quantization with Intel® MKL-DNN](#1)
* [2. Model Quantization with CUDNN](#2)

<h2 id="1">Model Quantization with Intel® MKL-DNN</h2>

Intel® MKL-DNN supports quantization with subgraph features on Intel® CPU Platform and can bring performance improvements on the [Intel® Xeon® Scalable Platform](https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-platform.html). A new quantization script `imagenet_gen_qsym_mkldnn.py` has been designed to launch quantization for CNN models with Intel® MKL-DNN. This script integrates with [Gluon-CV modelzoo](https://gluon-cv.mxnet.io/model_zoo/classification.html), so that more pre-trained models can be downloaded from Gluon-CV and then converted for quantization. This script also supports custom models.

Calibration is used for generating a calibration table for the quantized symbol. The quantization script supports three methods:

- **none:** No calibration will be used. The thresholds for quantization will be calculated on the fly. This will result in inference speed slowdown and loss of accuracy in general.
- **naive:** Simply take min and max values of layer outputs as thresholds for quantization. In general, the inference accuracy worsens with more examples used in calibration. It is recommended to use `entropy` mode as it produces more accurate inference results.
- **entropy:** Calculate KL divergence of the fp32 output and quantized output for optimal thresholds. This mode is expected to produce the best inference accuracy of all three kinds of quantized models if the calibration dataset is representative enough of the inference dataset.

Use the following command to install [Gluon-CV](https://gluon-cv.mxnet.io/):

```
pip install gluoncv
```

The following models have been tested on Linux systems.

| Model | Source | Dataset | FP32 Accuracy (top-1/top-5)| INT8 Accuracy (top-1/top-5)|
|:---|:---|---|:---:|:---:|
| [ResNet50-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 75.87%/92.72%  |  75.71%/92.65% |
| [ResNet101-V1](#4)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 77.3%/93.58%  | 77.09%/93.41%  |
|[Squeezenet 1.0](#5)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|57.01%/79.71%|56.62%/79.55%|
|[MobileNet 1.0](#6)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|69.76%/89.32%|69.61%/89.09%|
|[Inception V3](#7)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|76.49%/93.10% |76.38%/93% |
|[ResNet152-V2](#8)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|76.76%/93.03%|76.48%/92.96%|
|[Inception-BN](#9)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/inception-bn/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|72.09%/90.60%|72.00%/90.53%|
| [SSD-VGG](#10) | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | VOC2007/2012  | 0.83 mAP  | 0.82 mAP  |

<h3 id='3'>ResNet50-V1</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='4'>ResNet101-V1</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=resnet101_v1 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --param-file=./model/resnet101_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet101_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
python imagenet_inference.py --symbol-file=./model/resnet101_v1-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='5'>SqueezeNet 1.0</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=squeezenet1.0 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --param-file=./model/squeezenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/squeezenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='6'>MobileNet 1.0</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=mobilenet1.0 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --param-file=./model/mobilenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/mobilenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='7'>Inception-V3</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=inceptionv3 --image-shape=3,299,299 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --param-file=./model/inceptionv3-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/inceptionv3-quantized-5batches-naive-symbol.json --param-file=./model/inceptionv3-quantized-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
python imagenet_inference.py --symbol-file=./model/inceptionv3-quantized-5batches-naive-symbol.json --image-shape=3,299,299 --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='8'>ResNet152-V2</h3>

The following command is to download the pre-trained model from the [MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/) which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-resnet-152 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='9'>Inception-BN</h3>

The following command is to download the pre-trained model from the [MXNet ModelZoo](http://data.mxnet.io/models/imagenet/inception-bn/) which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-inception-bn --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h3 id='10'>SSD-VGG</h3>

Follow the [SSD example's instructions](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#train-the-model) in [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd) to train a FP32 `SSD-VGG16_reduced_300x300` model based on Pascal VOC dataset. You can also download our [SSD-VGG16 pre-trained model](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_vgg16_reduced_300-dd479559.zip) and [packed binary data](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/ssd-val-fc19a535.zip). Extract the zip files, then rename the directories to `model` and `data` respectively. Then, rename the files in directories as follows.

```
data/
|---val.rec
|---val.lxt
|---val.idx
model/
|---ssd_vgg16_reduced_300.params
|---ssd_vgg16_reduced_300-symbol.json
```

Then, use the following command for quantization. By default, this script uses 5 batches (32 samples per batch) for naive calibration:

```
python quantization.py
```

After quantization, INT8 models will be saved in `model/` dictionary.  Use the following command to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/ssd_

# Launch INT8 Inference
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/cqssd_

# Launch dummy data Inference
python benchmark_score.py --deploy --prefix=./model/ssd_
python benchmark_score.py --deploy --prefix=./model/cqssd_
```

<h3 id='11'>Custom Model</h3>

This script also supports custom symbolic models. You can easily add some quantization layer configs in `imagenet_gen_qsym_mkldnn.py` like below:

```
elif args.model == 'custom':
    # add rgb mean/std of your model.
    rgb_mean = '0,0,0'
    rgb_std = '0,0,0'
    calib_layer = lambda name: name.endswith('_output')
    # add layer names you donnot want to quantize.
    # add conv/pool layer names that has negative inputs
    # since Intel® MKL-DNN only support uint8 quantization temporary.
    # add all fc layer names since Intel® MKL-DNN does not support temporary.
    excluded_sym_names += ['layers']
    # add your first conv layer names since Intel® MKL-DNN only support uint8 quantization temporary.
    if exclude_first_conv:
        excluded_sym_names += ['layers']
```

Some tips on quantization configs:

1. First, you should prepare your data, symbol file (custom-symbol.json) and parameter file (custom-0000.params) of your fp32 symbolic model.
2. Then, you should run the following command and verify that your fp32 symbolic model runs inference as expected.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/custom-symbol.json --param-file=./model/custom-0000.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu --data-nthreads=1
```

3. Then, you should add `rgb_mean`, `rgb_std` and `excluded_sym_names` in this script. Notice that you should exclude conv/pool layers that have negative data since Intel® MKL-DNN only supports `uint8` quantization temporarily. You should also exclude all fc layers in your model.

4. Then, you can run the following command for quantization:

```
python imagenet_gen_qsym_mkldnn.py --model=custom --num-calib-batches=5 --calib-mode=naive
```

5. After quantization, the quantized symbol and parameter files will be saved in the `model/` directory.

6. Finally, you can run INT8 inference:

```
# Launch INT8 Inference 
python imagenet_inference.py --symbol-file=./model/*.json --param-file=./model/*.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu --data-nthreads=1

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/*.json --batch-size=* --num-inference-batches=500 --ctx=cpu --benchmark=True
```

<h2 id="2">Model Quantization with CUDNN</h2>

This folder contains examples of quantizing a FP32 model with or without calibration and using the calibrated
quantized for inference. Two pre-trained imagenet models are taken as examples for quantization. One is
[Resnet-152](http://data.mxnet.io/models/imagenet/resnet/152-layers/), and the other one is
[Inception with BatchNorm](http://data.mxnet.io/models/imagenet/inception-bn/). The calibration dataset
is the [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) for testing the pre-trained models.

Here are the details of the four files in this folder.
- `imagenet_gen_qsym.py` This script provides an example of taking FP32 models and calibration dataset to generate
calibrated quantized models. When launched for the first time, the script would download the user-specified model,
either Resnet-152 or Inception,
and calibration dataset into `model` and `data` folders, respectively. The generated quantized models can be found in
the `model` folder.
- `imagenet_inference.py` This script is used for calculating the accuracy of FP32 models or quantized models on the
validation dataset which was downloaded for calibration in `imagenet_gen_qsym.py`.
- `launch_quantize.sh` This is a shell script that generates various quantized models for Resnet-152 and
Inception with BatchNorm with different configurations. Users can copy and paste the command from the script to
the console to run model quantization for a specific configuration.
- `launch_inference.sh` This is a shell script that calculate the accuracies of all the quantized models generated
by invoking `launch_quantize.sh`.

**NOTE**: 
- This example has only been tested on Linux systems.
- Performance is expected to decrease with GPU, however the memory footprint of a quantized model is smaller. The purpose of the quantization implementation is to minimize accuracy loss when converting FP32 models to INT8. MXNet community is working on improving the performance. 