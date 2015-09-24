# Training Neural Networks on MNIST

The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits
has a training set of 60,000 examples, and a test set of 10,000 examples. Each
example is a 28 × 28 gray image. They are provided by Yann LeCun, Corinna
Cortes, and Christopher J.C. Burges.


## Neural Networks

- [mlp.py](mlp.py) : multilayer perceptron with 3 fully connected layers
- [lenet.py](lenet.py) : LeNet with 2 convolution layers followed by 2 fully
  connected layers

## Results


Using 100 minibatch size and 20 data passes (not fine tuned.)

Machine: Dual Xeon E5-2680 2.8GHz, Dual GTX 980, Ubuntu 14.0, GCC 4.8.

| | val accuracy | 2 x E5-2680 | 1 x GTX 980 | 2 x GTX 980 |
| --- | ---: | ---: | ---: | ---: |
| `mlp.py` MKL + CUDA 7 | 97.8% | 40K img/sec | 103K img/sec | 60K img/sec |
| `lenet.py` MKL + CUDA 7 | 99% | 368 img/sec | 22.5K img/sec  | 33K img/sec |
| `lenet.py` MKL + CUDA 7 + CUDNN v3 | - | - | 19K img/sec | 29 K img/sec |
