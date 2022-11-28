<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# Model Quantization with Calibration Examples

This folder contains examples of quantizing a FP32 model with oneAPI Deep Neural Network Library (oneDNN) to (U)INT8 model.

<h2 id="1">Model Quantization with oneDNN</h2>

oneDNN supports quantization with subgraph features on Intel® CPU Platform and can bring performance improvements on the [Intel® Xeon® Scalable Platform](https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-platform.html).

```
usage: python imagenet_gen_qsym_onednn.py [-h] [--model MODEL] [--epoch EPOCH]
                                          [--no-pretrained] [--batch-size BATCH_SIZE]
                                          [--calib-dataset CALIB_DATASET]
                                          [--image-shape IMAGE_SHAPE]
                                          [--data-nthreads DATA_NTHREADS]
                                          [--num-calib-batches NUM_CALIB_BATCHES]
                                          [--exclude-first-conv] [--shuffle-dataset]
                                          [--calib-mode CALIB_MODE]
                                          [--quantized-dtype {auto,int8,uint8}]
                                          [--quiet]

Generate a calibrated quantized model from a FP32 model with oneDNN support

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to be quantized. If no-pretrained is set then
                        model must be provided to `model` directory in the same path
                        as this python script, default is `resnet50_v1`
  --epoch EPOCH         number of epochs, default is `0`
  --no-pretrained       If enabled, will not download pretrained model from
                        MXNet or Gluon-CV modelzoo, default is `False`
  --batch-size BATCH_SIZE
                        batch size to be used when calibrating model, default is `32`
  --calib-dataset CALIB_DATASET
                        path of the calibration dataset, default is `data/val_256_q90.rec`
  --image-shape IMAGE_SHAPE
                        number of channels, height and width of input image separated by comma,
                        default is `3,224,224`
  --data-nthreads DATA_NTHREADS
                        number of threads for data loading, default is `0`
  --num-calib-batches NUM_CALIB_BATCHES
                        number of batches for calibration, default is `10`
  --exclude-first-conv  excluding quantizing the first conv layer since the
                        input data may have negative value which doesn't
                        support at moment
  --shuffle-dataset     shuffle the calibration dataset
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
                        divergence of the FP32 output and quantized output for
                        optimal thresholds. This mode is expected to produce
                        the best inference accuracy of all three kinds of
                        quantized models if the calibration dataset is
                        representative enough of the inference dataset.
                        default is `entropy`
  --quantized-dtype {auto,int8,uint8}
                        quantization destination data type for input data,
                        default is `auto`
  --quiet               suppress most of log
```

A new benchmark script `launch_inference_onednn.sh` has been designed to launch performance benchmark for FP32 or INT8 image-classification models with oneDNN.
```
usage: bash ./launch_inference_onednn.sh -s symbol_file [-b batch_size] [-iter iteraton] [-ins instance] [-c cores/instance] [-h]

arguments:
  -h, --help                show this help message and exit
  -s, --symbol_file         symbol file for benchmark, required
  -b, --batch_size          inference batch size
                            default: 64
  -iter, --iteration        inference iteration
                            default: 500
  -ins, --instance          launch multi-instance inference
                            default: one instance per socket
  -c, --core                number of cores per instance
                            default: divide full physical cores

example: resnet INT8 performance benchmark on c5.24xlarge(duo sockets, 24 physical cores per socket).

    bash ./launch_inference_onednn.sh -s ./model/resnet50_v1-quantized-5batches-naive-symbol.json

will launch two instances for throughput benchmark and each instance will use 24 physical cores.
```

The following models have been tested on Linux systems. Accuracy is collected on Intel XEON Cascade Lake CPU. For CPU with Skylake Lake or eariler architecture, the accuracy may not be the same.
| Model | Source | Dataset | FP32 Accuracy (top-1/top-5)| INT8 Accuracy (top-1/top-5)|
|:---|:---|---|:---:|:---:|
| ResNet18-V1  | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |70.45%/89.55%|70.22%/89.38%|
| ResNet50-V1  | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |76.36%/93.49%|76.04%/93.30%|
| ResNet101-V1  | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |78.23%/93.99%|77.85%/93.69%|
| MobileNet v2 1.0  | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |71.72%/90.28%|71.22%/89.92%|
| VGG16 | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |72.83%/91.11%|72.81%/91.10%|
| VGG19  | [MXNet ModelZoo](https://github.com/apache/mxnet/tree/master/python/mxnet/gluon/model_zoo)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |73.67%/91.63%|73.67%/91.67%|
*Measured on validation ImageNet (ILSVRC2012) with batch-size=64,  num-calib-batches=10 and calib-mode=entropy*

<h3>Pre-trained Model</h3>

The following command is to download the pre-trained model from [MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/) and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_onednn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. Set `--model` to one of above listed verified models to quantize them. The following command is to launch inference.

```
# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=0.485,0.456,0.406 --rgb-std=0.229,0.224,0.225 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=0.485,0.456,0.406 --rgb-std=0.229,0.224,0.225 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

# Launch dummy data Inference
bash ./launch_inference_onednn.sh -s ./model/resnet50_v1-symbol.json
bash ./launch_inference_onednn.sh -s ./model/resnet50_v1-quantized-5batches-naive-symbol.json
```

<h3 id='4'>Custom Model</h3>

This script also supports custom symbolic models. Quantization layer configs can easily be added in `imagenet_gen_qsym_onednn.py` like below:

```
if logger:
    frameinfo = getframeinfo(currentframe())
    logger.info(F'Please set proper RGB configs inside this script below {frameinfo.filename}:{frameinfo.lineno} for model {args.model}!')
# add rgb mean/std of your model.
rgb_mean = '0,0,0'
rgb_std = '0,0,0'
# add layer names that shouldn't be quantized.
if logger:
    frameinfo = getframeinfo(currentframe())
    logger.info(F'Please set proper excluded_sym_names inside this script below {frameinfo.filename}:{frameinfo.lineno} for model {args.model} if required!')
excluded_sym_names += []
if exclude_first_conv:
    excluded_sym_names += []
```

Some tips on quantization configs:

1. First, data, symbol file (custom-symbol.json) and parameter file (custom-0000.params) of FP32 symbolic model should be prepared.
2. Then, following command should be run to verify that FP32 symbolic model runs inference as expected.

```
# Launch FP32 Inference
python imagenet_inference.py --symbol-file=./model/custom-symbol.json --param-file=./model/custom-0000.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/*
```

3. Proper `rgb_mean`, `rgb_std` and `excluded_sym_names` should be added in `imagenet_gen_qsym_onednn.py` script.

4. Run following command for quantization:

```
python imagenet_gen_qsym_onednn.py --model=custom --num-calib-batches=5 --calib-mode=naive
```

5. After quantization, the quantized symbol and parameter files will be saved in the `model/` directory.

6. Finally, INT8 inference can be run:

```
# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-10batches-entropy-symbol.json --param-file=./model/resnet50_v1-quantized-10batches-entropy-0000.params --benchmark

# Launch dummy data Inference
bash ./launch_inference_onednn.sh -s ./model/*.json
```
