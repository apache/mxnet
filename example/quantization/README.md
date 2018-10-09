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

Following the [instruction](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#train-the-model) in [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd) to train a FP32 `SSD-VGG16_reduced_300x300` model based on Pascal VOC dataset. You can also download our pre-trained model and packed binary data from [here](http://data.mxnet.io/data/) and extract to `model/` and `data/`dictionary.

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