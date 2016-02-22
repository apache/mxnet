# End-to-End Deep Learning Tutorial for Kaggle NDSB-II

In this example, we will demo how to use MXNet to build an end-to-end deep learning system to help Diagnose Heart Disease.  The demo network is able to achieve 0.039222 CRPS on validation set, which is good enough to get Top-10 (on Dec 22nd, 2015).

Notice this is a very simple model with no attempt to optimize the structure or hyper parameters, you can build fantastic network based on it. While this tutorial is written in python, mxnet comes with support for other popular languages such as R and Julia which can also be used. You are more than welcomed to try and contribute back to this example.

This example requires GPU to train. If you are working with AWS,
A simple guide to build MXNet on AWS and existing AMI can be found at [This document](https://mxnet.readthedocs.org/en/latest/aws.html).
you can also choose to put your data on S3, and having all the machine directly load data from S3, without having to copy data over when you are starting new instances.


## General Overview of model
### Input Data
We notice for in most of data, there are 30 frames for a sequence. A simple idea is pack this sequence into a multi-channel input, then let neural network learn from it. This tutorial is based on this idea: We first find accumulate all suitable data with 30 frames, then feed to the neural network to learn the target directly.

Another idea is use difference to measure change in time-series. By using MXNet symbolic interface, we can build a dynamic difference channels to transform input inside of the network. It helps a little in the final result.

### Network Objective
For the network, we use a 20 years old LeNet style convolution network with batch normalization and dropout. We did not finetune the configuration and hyper parameters as this is mainly for demonstration purposes. We are sure better solutions can be found.

One important idea of the model is to predict what the problem is asking for. In this problem, we are asked to predict a CDF value on 600 data-point. So we formulate the problem as a regression problem. We ask the neural-net to output 600 values, which corresponds to the CDF value to be predicted. The label is transformed into the 0-1 function as used in the evaluation target.


## Preprocessing
We first run a preprocessing step, to pack the data into a csv file. Each line of the csv file corresponds to a 30 x 64 x 64 tensor, which gives 30 frames of resized images. We can also use other inputs besides csv. We choose the csv because this format is quite common for all language and it is easy to manipulate.
The input dataset is quite big. While they can fit into memory of a big machine, we want to be safe for all desktop settings, so we will use a CSVIter from mxnet to load data from disk on the fly during training, without loading all the data into memory. You are also more than welcomed to try the in-memory setting.



## Step by step

Prepare raw data in ```data``` folder. The tree of ```data``` folder is like

```
-data
 |
 ---- sample_submission_validate.csv
 |
 ---- train.csv
 |
 ---- train
 |    |
 |    ---- 0
 |    |
 |    ---- …
 |
 ---- validate
      |
      ---- 501
      |
      ---- …
```

2. Run ```python3 Preprocessing.py``` to do preprocessing of data.
3. After we have the processed data, run ```python3 Train.py``` to generate ```submission.csv```
4. We also provide the R code with the same network structure and parameters in ```Train.R```. Right now it used the pre-processed csv files by ```Preprocessing.py```. We will add the pre-processing R code later.

Note:
- To run with python2, you need to change ```Train.py, line #145 #199``` to the python2 syntax.
- To modify network, change ```get_lenet``` function in ```Train.py``` or ```get.lenet``` function in ```Train.R```.
- We also provide ```local_train```, ```local_test``` file for local parameter tuning.
- To run on multiple GPU with huge network, or questions about saving network parameters etc, please refer [MXNet docs](https://mxnet.readthedocs.org/en/latest/)


## About MXNet
MXNet is a deep learning framework designed for both efficiency and flexibility by DMLC group. Like all other packages in DMLC, it will fully utilize all the resources to solve the problem under limited resource constraint, with a flexible programming interface. You can use it for all purposes of data science and deep learning tasks with R, Julia, python and more.
