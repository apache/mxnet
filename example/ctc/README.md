# Connectionist Temporal Classification

[Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (CTC) is a cost function that is used to train Recurrent Neural Networks (RNNs) to label unsegmented input sequence data in supervised learning. For example in a speech recognition application, using a typical cross-entropy loss the input signal needs to be segmented into words or sub-words. However, using CTC-loss, a single unaligned label sequence per input sequence is sufficient for the network to learn both the alignment and labeling. Baidu's warp-ctc page contains a more detailed [introduction to CTC-loss](https://github.com/baidu-research/warp-ctc#introduction).

## LSTM OCR Example
In this example, we use CTC loss to train a network on the problem of Optical Character Recognition (OCR) of CAPTCHA images. This example uses the `captcha` python package to generate a random dataset for training. Training the network requires a CTC-loss layer and MXNet provides two options for such layer. The OCR example is constructed as follows:

1. 80x30 CAPTCHA images containing 3 to 4 random digits are generated using python captcha library.
2. Each image is used as a data sequence with sequence-length of 80 and vector length of 30.
3. The output layer uses CTC loss in training and softmax in inference.

Note: When using CTC-loss, one prediction label is reserved for blank label. In this example, when predicting digits between 0 to 9, softmax output has 11 labels, with label 0 used for blank and 1 to 10 used for digit 0 to digit 9 respectively.

### Description of the files
LSTM-OCR example contains the following files:
* `captcha_generator.py`: Module for generating random 3 or 4 digit CAPTCHA images for training. It also contains a script for generating sample CAPTCHA images into an output file for inference testing.
* `ctc_metrics.py`: Module for calculating the prediction accuracy during training. Two accuracy measures are implemented: A simple accuracy measure that calculates number of correct predictions divided by total number of predictions and a second accuracy measure based on sum of Longest Common Sequence (LCS) ratio of all predictions divided by total number of predictions.
* `hyperparameters.py`: Contains all hyperparameters for the network structure and training.
* `lstm.py`: Contains LSTM network implementations. Options for adding mxnet-ctc and warp-ctc loss for training as well as adding softmax for inference are available.
* `lstm_ocr_infer.py`: Script for running inference after training.
* `lstm_ocr_train.py`: Script for training with ctc or warp-ctc loss.
* `multiproc_data.py`: A module for multiprocess data generation.
* `oct_iter.py`: A DataIter module for iterating through training data.

## CTC-loss in MXNet
MXNet supports two CTC-loss layers in Symbol API:

* `mxnet.symbol.contrib.ctc_loss` is implemented in MXNet and included as part of the standard package.
* `mxnet.symbol.WarpCTC` uses Baidu's warp-ctc library and requires building warp-ctc library and mxnet library both from source.

### Building MXNet with warp-ctc
In order to use `mxnet.symbol.WarpCTC` layer, you need to first build Baidu's [warp-ctc](https://github.com/baidu-research/warp-ctc) library from source and then build MXNet from source with warp-ctc config flags enabled.

#### Building warp-ctc
You need to first build warp-ctc from source and then install it in your system. Please follow [instructions here](https://github.com/baidu-research/warp-ctc#compilation) to build warp-ctc from source. Once compiled, you need to install the library by running the following command from `warp-ctc/build` directory:
```
$ sudo make install
```

#### Building MXNet from source with warp-ctc integration
In order to build MXNet from source, you need to follow [instructions here](http://mxnet.incubator.apache.org/install/index.html). After choosing your system configuration, Python environment, and "Build from Source" options, before running `make` in step 4, you need to enable warp-ctc integration by uncommenting the following lines in `make/config.mk` in `incubator-mxnet` directory:
```
WARPCTC_PATH = $(HOME)/warp-ctc
MXNET_PLUGINS += plugin/warpctc/warpctc.mk
```

## Run LSTM OCR Example
Running this example requires the following pre-requisites:
* `captcha` and `opencv` python packages are installed:
```
$ pip install captcha
$ pip install opencv-python
```
* You have access to one (or more) `ttf` font files. You can download a collection of font files from [Ubuntu's website](https://design.ubuntu.com/font/). The instructions in this section assume that a `./font/Ubuntu-M.ttf` file exists under the `example/ctc/` directory.

### Training
The training script demonstrates how to construct a network with both CTC loss options and train using `mxnet.Module` API. Training is done by generating random CAPTCHA images using the font(s) provided. This example uses 80x30 captcha images that contain 3 to 4 digits each.

When using a GPU for training, the training bottleneck will be data generation. To remedy this bottleneck, this example implements a multiprocess data generation. Number of processes for image generation as well as training on CPU or GPU can be configured using command line arguments.

To see the list of all arguments:
```
$ python lstm_ocr_train.py --help
```
Using command line, you can also select between ctc or warp-ctc loss options. For example, the following command initiates a training session on a single GPU with 4 CAPTCHA generating processes using ctc loss and `font/Ubuntu-M.ttf` font file:
```
$ python lstm_ocr_train.py --gpu 1 --num_proc 4 --loss ctc font/Ubuntu-M.ttf
```

You can train with multiple fonts by specifying a folder that contains multiple `ttf` font files instead. The training saves a checkpoint after each epoch. The prefix used for checkpoint is 'ocr' by default, but can be changed with `--prefix` argument.

When testing this example, the following system configuration was used:
* p2.xlarge AWS EC2 instance (4 x CPU and 1 x K80 GPU)
* Deep Learning Amazon Machine Image (with mxnet 1.0.0)

This training example finishes after 100 epochs with ~87% accuracy. If you continue training further, the network achieves over 95% accuracy. Similar accuracy is achieved with both ctc (`--loss ctc`) and warp-ctc (`--loss warpctc`) options. Logs of the last training epoch:

```
05:58:36,128 Epoch[99] Batch [50]	Speed: 1067.63 samples/sec	accuracy=0.877757
05:58:42,119 Epoch[99] Batch [100]	Speed: 1068.14 samples/sec	accuracy=0.859688
05:58:48,114 Epoch[99] Batch [150]	Speed: 1067.73 samples/sec	accuracy=0.870469
05:58:54,107 Epoch[99] Batch [200]	Speed: 1067.91 samples/sec	accuracy=0.864219
05:58:58,004 Epoch[99] Train-accuracy=0.877367
05:58:58,005 Epoch[99] Time cost=28.068
05:58:58,047 Saved checkpoint to "ocr-0100.params"
05:59:00,721 Epoch[99] Validation-accuracy=0.868886
```

### Inference
The inference script demonstrates how to load a network from a checkpoint, modify its final layer, and predict a label for a CAPTCHA image using `mxnet.Module` API. You can choose the prefix as well as the epoch number of the checkpoint using command line arguments. To see the full list of arguments:
```
$ python lstm_ocr_infer.py --help
```
For example, to predict label for 'sample.jpg' file using 'ocr' prefix and checkpoint at epoch 100:
```
$ python lstm_ocr_infer.py --prefix ocr --epoch 100 sample.jpg

Digits: [0, 0, 8, 9]
```

Note: The above command expects the following files, generated by the training script, to exist in the current directory:
* ocr-symbol.json
* ocr-0100.params

#### Generate CAPTCHA samples
CAPTCHA images can be generated using the `captcha_generator.py` script. To see the list of all arguments:
```
$ python captcha_generator.py --help
```
For example, to generate a CAPTCHA image with random digits from 'font/Ubuntu-M.ttf' and save to 'sample.jpg' file:
```
$ python captcha_generator.py font/Ubuntu-M.ttf sample.jpg
```
