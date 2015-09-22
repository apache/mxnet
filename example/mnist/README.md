# Training Neural Networks on MNIST

The [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits,
available from this page, has a training set of 60,000 examples, and a test set
of 10,000 examples. Each example is a 28 × 28 gray image. They are provided by
Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.


## Neural Networks

- [mlp.py](mlp.py) multilayer perceptron with 3 fully connected layers
- [lenet.py](lenet.py) LeNet with 2 convolution layers followed by 2 fully
  connected layers

## Performance

Machine: Dual Xeon E5-2680 2.8GHz, Dual GTX 980, CUDA 7.0

| | 2 x E5-2680 | 1 x GTX 980 | 2 x GTX 980 |
| --- | --- | --- | --- |
| `mlp.py` | 40K img/sec | 103K img/sec | 60K img/sec |

Dual GPUs slow down the performance due to the tiny size of workload.

## Results

sample output using single GTX 980

```bash
~/mxnet/example/mnist $ python mlp.py
[20:52:47] src/io/iter_mnist.cc:84: MNISTIter: load 60000 images, shuffle=1, shape=(100,784)
[20:52:47] src/io/iter_mnist.cc:84: MNISTIter: load 10000 images, shuffle=1, shape=(100,784)
INFO:root:Start training with 1 devices
INFO:root:Iteration[0] Train-accuracy=0.920833
INFO:root:Iteration[0] Time cost=0.656
INFO:root:Iteration[0] Validation-accuracy=0.961100
INFO:root:Iteration[1] Train-accuracy=0.965317
INFO:root:Iteration[1] Time cost=0.576
INFO:root:Iteration[1] Validation-accuracy=0.963000
INFO:root:Iteration[2] Train-accuracy=0.974817
INFO:root:Iteration[2] Time cost=0.567
INFO:root:Iteration[2] Validation-accuracy=0.965800
INFO:root:Iteration[3] Train-accuracy=0.978433
INFO:root:Iteration[3] Time cost=0.590
INFO:root:Iteration[3] Validation-accuracy=0.970900
INFO:root:Iteration[4] Train-accuracy=0.982583
INFO:root:Iteration[4] Time cost=0.593
INFO:root:Iteration[4] Validation-accuracy=0.973100
INFO:root:Iteration[5] Train-accuracy=0.982217
INFO:root:Iteration[5] Time cost=0.592
INFO:root:Iteration[5] Validation-accuracy=0.971300
INFO:root:Iteration[6] Train-accuracy=0.985817
INFO:root:Iteration[6] Time cost=0.555
INFO:root:Iteration[6] Validation-accuracy=0.969400
INFO:root:Iteration[7] Train-accuracy=0.987033
INFO:root:Iteration[7] Time cost=0.546
INFO:root:Iteration[7] Validation-accuracy=0.974800
INFO:root:Iteration[8] Train-accuracy=0.988333
INFO:root:Iteration[8] Time cost=0.535
INFO:root:Iteration[8] Validation-accuracy=0.975900
INFO:root:Iteration[9] Train-accuracy=0.987983
INFO:root:Iteration[9] Time cost=0.531
INFO:root:Iteration[9] Validation-accuracy=0.968900
```
