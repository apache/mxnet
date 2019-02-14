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

# Example of a Convolutional Autoencoder

Autoencoder architecture is often used for unsupervised feature learning. This [link](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/) contains an introduction tutorial to autoencoders. This example illustrates a simple autoencoder using stack of fully-connected layers for both encoder and decoder. The number of hidden layers and size of each hidden layer can be customized using command line arguments.

## Training Stages
This example uses a two-stage training. In the first stage, each layer of encoder and its corresponding decoder are trained separately in a layer-wise training loop. In the second stage the entire autoencoder network is fine-tuned end to end.

## Dataset
The dataset used in this example is [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. This example uses scikit-learn module to download this dataset.

## Simple autoencoder example
mnist_sae.py: this example uses a simple auto-encoder architecture to encode and decode MNIST images with size of 28x28 pixels. It contains several command line arguments. Pass -h (or --help) to view all available options. To start the training on CPU (use --gpu option for training on GPU) using default options:

```
python mnist_sae.py
```
