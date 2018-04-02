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
