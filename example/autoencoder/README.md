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

Autoencoder architectures are often used for unsupervised feature learning. This [link](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/) contains an introduction tutorial to autoencoders. This example illustrates a simple autoencoder using a stack of convolutional layers for both the encoder and the decoder. 


![](https://cdn-images-1.medium.com/max/800/1*LSYNW5m3TN7xRX61BZhoZA.png)

([Diagram source](https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85))


The idea of an autoencoder is to learn to use bottleneck architecture to encode the input and then try to decode it to reproduce the original. By doing so, the network learns to effectively compress the information of the input, the resulting embedding representation can then be used in several domains. For example as featurized representation for visual search, or in anomaly detection.

## Dataset

The dataset used in this example is [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. 

## Variational Autoencoder

You can check an example of variational autoencoder [here](https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html)

