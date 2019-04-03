<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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
