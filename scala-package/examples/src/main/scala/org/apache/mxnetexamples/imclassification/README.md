# Image Classification Models

This examples contains a number of image classification models that can be run on various datasets.

## Models

Currently, the following models are supported:
- MultiLayerPerceptron
- Lenet
- Resnet

## Datasets

Currently, the following datasets are supported:
- MNIST

#### Synthetic Benchmark Data

Additionally, the datasets can be replaced by randomly generated data for benchmarking.
Data is produced to match the shapes of the supported datasets above.

The following additional dataset image shapes are also defined for use with the benchmark synthetic data:
- imagenet



## Setup

### MNIST

For this dataset, the data must be downloaded and extracted from the source or 
```$xslt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/mnist/mnist.zip
```

Afterwards, the location of the data folder must be passed in through the `--data-dir` argument.
