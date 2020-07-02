<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Model Quantization with Calibration Examples

This folder contains examples of quantizing a FP32 model with Intel® MKL-DNN or CUDNN.

<h2 id="0">Contents</h2>

* [1. Model Quantization with Intel® MKL-DNN](#1)
* [2. Model Quantization with CUDNN](#2)

<h2 id="1">Model Quantization with Intel® MKL-DNN</h2>

Intel® MKL-DNN supports quantization with subgraph features on Intel® CPU Platform and can bring performance improvements on the [Intel® Xeon® Scalable Platform](https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-platform.html). A new quantization script `imagenet_gen_qsym_mkldnn.py` has been designed to launch quantization for image-classification models with Intel® MKL-DNN. This script integrates with [Gluon-CV modelzoo](https://gluon-cv.mxnet.io/model_zoo/classification.html), so that more pre-trained models can be downloaded from Gluon-CV and then converted for quantization. To apply quantization flow to your project directly, please refer [Quantize custom models with MKL-DNN backend](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_quantization.html).

```
usage: imagenet_gen_qsym_mkldnn.py [-h] [--model MODEL] [--epoch EPOCH]
                                   [--no-pretrained] [--batch-size BATCH_SIZE]
                                   [--label-name LABEL_NAME]
                                   [--calib-dataset CALIB_DATASET]
                                   [--image-shape IMAGE_SHAPE]
                                   [--data-nthreads DATA_NTHREADS]
                                   [--num-calib-batches NUM_CALIB_BATCHES]
                                   [--exclude-first-conv] [--shuffle-dataset]
                                   [--shuffle-chunk-seed SHUFFLE_CHUNK_SEED]
                                   [--shuffle-seed SHUFFLE_SEED]
                                   [--calib-mode CALIB_MODE]
                                   [--quantized-dtype {auto,int8,uint8}]
                                   [--enable-calib-quantize ENABLE_CALIB_QUANTIZE]

Generate a calibrated quantized model from a FP32 model with Intel MKL-DNN
support

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to be quantized.
  --epoch EPOCH         number of epochs, default is 0
  --no-pretrained       If enabled, will not download pretrained model from
                        MXNet or Gluon-CV modelzoo.
  --batch-size BATCH_SIZE
  --label-name LABEL_NAME
  --calib-dataset CALIB_DATASET
                        path of the calibration dataset
  --image-shape IMAGE_SHAPE
  --data-nthreads DATA_NTHREADS
                        number of threads for data decoding
  --num-calib-batches NUM_CALIB_BATCHES
                        number of batches for calibration
  --exclude-first-conv  excluding quantizing the first conv layer since the
                        input data may have negative value which doesn't
                        support at moment
  --shuffle-dataset     shuffle the calibration dataset
  --shuffle-chunk-seed SHUFFLE_CHUNK_SEED
                        shuffling chunk seed, see https://mxnet.incubator.apac
                        he.org/api/python/io/io.html?highlight=imager#mxnet.io
                        .ImageRecordIter for more details
  --shuffle-seed SHUFFLE_SEED
                        shuffling seed, see https://mxnet.incubator.apache.org
                        /api/python/io/io.html?highlight=imager#mxnet.io.Image
                        RecordIter for more details
  --calib-mode CALIB_MODE
                        calibration mode used for generating calibration table
                        for the quantized symbol; supports 1. none: no
                        calibration will be used. The thresholds for
                        quantization will be calculated on the fly. This will
                        result in inference speed slowdown and loss of
                        accuracy in general. 2. naive: simply take min and max
                        values of layer outputs as thresholds for
                        quantization. In general, the inference accuracy
                        worsens with more examples used in calibration. It is
                        recommended to use `entropy` mode as it produces more
                        accurate inference results. 3. entropy: calculate KL
                        divergence of the fp32 output and quantized output for
                        optimal thresholds. This mode is expected to produce
                        the best inference accuracy of all three kinds of
                        quantized models if the calibration dataset is
                        representative enough of the inference dataset.
  --quantized-dtype {auto,int8,uint8}
                        quantization destination data type for input data
  --enable-calib-quantize ENABLE_CALIB_QUANTIZE
                        If enabled, the quantize op will be calibrated offline
                        if calibration mode is enabled
```

A new benchmark script `launch_inference_mkldnn.sh` has been designed to launch performance benchmark for float32 or int8 image-classification models with Intel® MKL-DNN.
```
usage: bash ./launch_inference_mkldnn.sh [[[-s symbol_file ] [-b batch_size] [-iter iteraton] [-ins instance] [-c cores/instance]] | [-h]]

optional arguments:
  -h, --help                show this help message and exit
  -s, --symbol_file         symbol file for benchmark
  -b, --batch_size          inference batch size
                            default: 64
  -iter, --iteration        inference iteration
                            default: 500
  -ins, --instance          launch multi-instance inference
                            default: one instance per socket
  -c, --core                number of cores per instance
                            default: divide full physical cores

example: resnet int8 performance benchmark on c5.24xlarge(duo sockets, 24 physical cores per socket).

    bash ./launch_inference_mkldnn.sh -s ./model/resnet50_v1-quantized-5batches-naive-symbol.json

will launch two instances for throughput benchmark and each instance will use 24 physical cores.
```

Use the following command to install [Gluon-CV](https://gluon-cv.mxnet.io/):

```
pip install gluoncv
```

The following models have been tested on Linux systems. Accuracy is collected on Intel XEON Cascade Lake CPU. For CPU with Skylake Lake or eariler architecture, the accuracy may not be the same.

| Model | Source | Dataset | FP32 Accuracy (top-1/top-5)| INT8 Accuracy (top-1/top-5)|
|:---|:---|---|:---:|:---:|
| [ResNet18-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |70.15%/89.38%|69.92%/89.30%|
| [ResNet50-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 76.34%/93.13%  |  76.06%/92.99% |
| [ResNet101-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 77.33%/93.59%  | 77.07%/93.47%  |
|[Squeezenet 1.0](#4)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|56.98%/79.20%|56.79%/79.47%|
|[MobileNet 1.0](#5)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|72.23%/90.64%|72.06%/90.53%|
|[MobileNetV2 1.0](#6)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|70.27%/89.62%|69.82%/89.35%|
|[Inception V3](#7)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|77.76%/93.83% |78.05%/93.91% |
|[ResNet152-V2](#8)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|76.65%/93.07%|76.25%/92.89%|
|[Inception-BN](#9)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/inception-bn/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|72.28%/90.63%|72.02%/90.53%|
| [SSD-VGG16](#10) | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | VOC2007/2012  | 0.8366 mAP  | 0.8357 mAP  |
| [SSD-VGG16](#10) | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | COCO2014  | 0.2552 mAP  | 0.253 mAP  |

<h3 id='3'>ResNetV1</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. Set `--model` to `resnet18_v1/resnet50_v1b/resnet101_v1` to quantize other models. The following command is to launch inference.

```
# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/resnet50_v1-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/resnet50_v1-quantized-5batches-naive-symbol.json
```

<h3 id='4'>SqueezeNet 1.0</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=squeezenet1.0 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --param-file=./model/squeezenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/squeezenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/squeezenet1.0-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/squeezenet1.0-quantized-5batches-naive-symbol.json
```

<h3 id='5'>MobileNet 1.0</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=mobilenet1.0 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --param-file=./model/mobilenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/mobilenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/mobilenet1.0-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/mobilenet1.0-quantized-5batches-naive-symbol.json
```

<h3 id='6'>MobileNetV2 1.0</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=mobilenetv2_1.0 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --param-file=./model/mobilenetv2_1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=./model/mobilenetv2_1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/mobilenetv2_1.0-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/mobilenetv2_1.0-quantized-5batches-naive-symbol.json
```

<h3 id='7'>Inception-V3</h3>

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=inceptionv3 --image-shape=3,299,299 --num-calib-batches=5 --calib-mode=naive
```
The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --param-file=./model/inceptionv3-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/inceptionv3-quantized-5batches-naive-symbol.json --param-file=./model/inceptionv3-quantized-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/inceptionv3-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/inceptionv3-quantized-5batches-naive-symbol.json
```

<h3 id='8'>ResNet152-V2</h3>

The following command is to download the pre-trained model from the [MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/) which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-resnet-152 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/imagenet1k-resnet-152-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json
```

<h3 id='9'>Inception-BN</h3>

The following command is to download the pre-trained model from the [MXNet ModelZoo](http://data.mxnet.io/models/imagenet/inception-bn/) which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-inception-bn --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/imagenet1k-inception-bn-symbol.json
bash ./launch_inference_mkldnn.sh -s ./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json
```

<h3 id='10'>SSD-VGG16</h3>

SSD model is located in [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd), follow [the insturctions](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#quantize-model) to run quantized SSD model.

<h3 id='11'>Custom Model</h3>

This script also supports custom symbolic models. You can easily add some quantization layer configs in `imagenet_gen_qsym_mkldnn.py` like below:

```
else:
    logger.info('Please set proper RGB configs for model %s' % args.model)
    # add rgb mean/std of your model.
    rgb_mean = '0,0,0'
    rgb_std = '0,0,0'
    # add layer names you donnot want to quantize.
    logger.info('Please set proper excluded_sym_names for model %s' % args.model)
    excluded_sym_names += ['layers']
    if exclude_first_conv:
        excluded_sym_names += ['layers']
```

Some tips on quantization configs:

1. First, you should prepare your data, symbol file (custom-symbol.json) and parameter file (custom-0000.params) of your fp32 symbolic model.
2. Then, you should run the following command and verify that your fp32 symbolic model runs inference as expected.

```

# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/custom-symbol.json --param-file=./model/custom-0000.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu
```

3. Then, you should add `rgb_mean`, `rgb_std` and `excluded_sym_names` in this script.

4. Then, you can run the following command for quantization:

```
python imagenet_gen_qsym_mkldnn.py --model=custom --num-calib-batches=5 --calib-mode=naive
```

5. After quantization, the quantized symbol and parameter files will be saved in the `model/` directory.

6. Finally, you can run INT8 inference:

```
# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/*.json --param-file=./model/*.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu

# Launch dummy data Inference
bash ./launch_inference_mkldnn.sh -s ./model/*.json
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
