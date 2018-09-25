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
- imagenet (only supported with synthetic data for benchmark)

Additionally, the datasets can be replaced by synthetic randomly data generated at the appropriate image size for the dataset for benchmarking


## Setup

### MNIST

For this dataset, the data must be downloaded and extracted from the source or 
```$xslt
https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/mnist/mnist.zip
```

Afterwards, the location of the data folder must be passed in through the `--data-dir` argument.
