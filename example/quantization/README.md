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