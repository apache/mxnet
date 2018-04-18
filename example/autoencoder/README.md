# Example of Autencoder

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
