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

Variational Auto Encoder(VAE)
=============================

This folder contains a tutorial which implements the Variational Auto Encoder in MXNet using the MNIST handwritten digit
recognition dataset. Model built is referred from [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114/)
paper. This paper introduces a stochastic variational inference and learning algorithm that scales to large datasets.

Prerequisites:
To run this example, you need:
- [Jupyter Notebook](http://jupyter.org/index.html)
- Matplotlib

Files in this folder:
- **VAE_example.ipynb** : Jupyter notebook which explains concept of VAE step by step and also shows how to use
MXNet-based VAE class(from VAE.py) to do the training directly.

- **VAE.py** : Contains class which implements the Variational Auto Encoder. This is used in the above tutorial.

In VAE, the encoder becomes a variational inference network that maps the data to a distribution
for the hidden variables, and the decoder becomes a generative network that maps the latent variables back to the data.
The network architecture shown in the tutorial uses Gaussian MLP as an encoder and Bernoulli MLP as a decoder.
