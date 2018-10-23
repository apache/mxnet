# Example of a Convolutionnal Autencoder

Autoencoder architecture is often used for unsupervised feature learning. This [link](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/) contains an introduction tutorial to autoencoders. This example illustrates a simple autoencoder using stack of convolutionnal layers for both encoder and decoder. 

![](https://cdn-images-1.medium.com/max/800/1*LSYNW5m3TN7xRX61BZhoZA.png)

([Diagram source](https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85))

The idea of an autoencoder is to learn to use bottleneck architecture to encode the input and then try to decode it to reproduce the original. By doing so, the network learns to effectively compress the information of the input, the resulting embedding representation can then be used in several domains. For example as featurized representation for visual search, or in anomaly detection.

## Dataset

The dataset used in this example is [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. 

## Variationnal Autoencoder

You can check an example of variational autoencoder [here](https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html)

