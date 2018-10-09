# Model Quantization with Calibration Examples
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

**NOTE**: This example has only been tested on Linux systems.

# Model Quantization with Intel MKL-DNN

MKL-DNN supports quantization well with subgraph feature on Intel® CPU Platform and can bring huge performance improvement on Intel® Xeon® Scalable Platform. A new quantization script `imagenet_gen_qsym_mkldnn.py` has been designed to launch quantization for image-classification models with MKL-DNN. This script intergrates with Gluon-CV modelzoo so that more pre-trained models can be get from Gluon-CV and can be converted for quantization. This script also supports custom models.

The following models have been tested on Linux systems.

| Model | Source | Dataset | FP32 Accuracy | INT8 Accuracy |
|:---|:---|---|:---:|:---:|
| ResNet50-V1  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 75.87%/92.72%  |  75.71%/92.65% |
| SSD-VGG | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | VOC2007/2012  | 0.83 mAP  | 0.82 mAP  |

## ResNet50-V1

Use below command to convert pre-trained model from Gluon-CV and quantization. Use calib mode can get better accuracy and performance and the calibration dataset
is the [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
```

After quantization, you will get a quantized symbol and parameter in `./model` dictionary. Use below command to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=128 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=128 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
```

## SSD-VGG

Go to [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd) dictionary. Following the [instruction](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#train-the-model) in [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd) to train a FP32 `SSD-VGG16_reduced_300x300` model based on Pascal VOC dataset. You can also download our [pre-trained model](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_vgg16_reduced_300-dd479559.zip) and [packed binary data](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/ssd-val-fc19a535.zip) then rename them and extract to `model/` and `data/` dictionary as below.

```
data/
|---val.rec
|---val.lxt
|---val.idx
model/
|---ssd_vgg16_reduced_300.params
|---ssd_vgg16_reduced_300-symbol.json
```

Then, use the following command for quantization. By default, this script use 5 batches(32 samples per batch) for naive calib:

```
python quantization.py
```

After quantization, INT8 models will be saved in `model/` dictionary.  Use below command to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/ssd_

# Launch INT8 Inference
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/cqssd_
```

## Custom Model

This script also supports custom symbolic models. You can easily add some quantization layer configs in `imagenet_gen_qsym_mkldnn.py` like below:

```
elif args.model == 'custom':
    # add rgb mean/std of your model.
    rgb_mean = '0,0,0'
    rgb_std = '0,0,0'
    calib_layer = lambda name: name.endswith('_output')
    # add layer names you donnot want to quantize.
    # add conv/pool layer names that has negative inputs
    # since MKLDNN only support uint8 quantization temporary.
    # add all fc layer names since MKLDNN does not support temporary.
    excluded_sym_names += ['layers']
    # add your first conv layer names since MKLDNN only support uint8 quantization temporary.
    if exclude_first_conv:
        excluded_sym_names += ['layers']
```

Some tips on quantization configs:

1. First, you should prepare your data, symbol file and parameter file of your fp32 symbolic model.
2. Then, you should run the below command and keep sure that your fp32 symbolic model runs inference well.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/*.json --param-file=./model/*.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu --data-nthreads=1
```

3. Then, you should add `rgb_mean`, `rgb_std`and `excluded_sym_names` in this script. Notice that you should exxclude conv/pool layers that has negative data since MKLDNN only support uint8 quantization temporary. You should also exclude all fc layers in your mdoel.

4. Then, you can run below command for quantization:

```
python imagenet_gen_qsym_mkldnn.py --model=custom --num-calib-batches=5 --calib-mode=naive
```

5. After quantization, INT8 symbol and parameter will be saved in `model/` dictionary.

6. Finally, you can run INT8 inference:

```
# Launch INT8 Inference 
python imagenet_inference.py --symbol-file=./model/*.json --param-file=./model/*.params --rgb-mean=* --rgb-std=* --num-skipped-batches=* --batch-size=* --num-inference-batches=*--dataset=./data/* --ctx=cpu --data-nthreads=1
```
