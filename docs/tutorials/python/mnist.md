# Handwritten Digit Recognition

In this tutorial, we’ll give you a step by step walk-through of how to build a hand written digit classifier using the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. For someone new to deep learning, this exercise is arguably the “Hello World” equivalent.

MNIST is a widely used dataset for the hand written digit classification task. It consists of 70,000 28×28 pixel grayscale images of hand written digits. The dataset is split into 60,000 training images and 10,000 test images.  There are 10 classes (for each of the 10 digits). The task at hand is to train our model using the 60,000 training images and subsequently test the classification accuracy using the 10,000 test images.

Here are some sample images from the dataset.

![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/mnist.png)

## Loading Data

We first fetch the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

The following code downloads and loads the images and the corresponding labels into memory.

```python
import mxnet as mx
mnist = mx.test_utils.get_mnist()
```

Next, we create data iterators for MXNet. Data iterator is the mechanism by which we feed input data into our training algorithm. MXNet data iterators are designed with speed and efficiency in mind. In our case, we'll configure the data iterator to feed examples in small batches. For MNIST dataset, each example consists of a 28x28 grayscale image and the corresponding label.

Images are commonly represented by a 4-D matrix with shape `(batch_size, num_channels, width, height)`. For MNIST dataset, since the images are grayscale, there is only one color channel. Also, the images are 28x28 pixels, so each image has width and height equal to 28. Therefore, the shape of input is `(batch_size, 1, 28, 28)`. Another important consideration is the order of input samples. When feeding training examples it is critical that we don't feed samples with the same label in succession. Doing so can slow down training.
Data iterators take care of this by randomly shuffling the inputs. Note that we only need to shuffle training data. The order does not matter for test data.

```python
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
```

## Multilayer Perceptron

We first use [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron), MLP for short, to solve this problem. We'll define the MLP using MXNet's symbolic interface. We being by creating a place holder variable for the input data. When working with an MLP, we need to flatten our 28x28 images into a flat 1-D structure of 784 (28 * 28) dimensions. The order of raw pixel values in the flattened vector does not matter as long as we are being consistent about how we do this across all input images. One might wonder if we are discarding valuable information by flattening. That is indeed true and we'll cover this more when we talk about convolutional neural networks where we preserve the input shape. For now, we'll go ahead and work with flattened images.

```python
data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)
```

MLP contains several fully-connected layers. A fully-connected layer, with an *n x m* input matrix *X* outputs a matrix *Y* with size *n x k*, where *k* is often called as the hidden size. This layer has two learnable parameters, the *m x k* weight matrix *W* and the *m x 1* bias vector *b*. It computes the outputs with *Y = W X + b*.

The output of a fully-connected layer is often fed into an activation function,
which applies an element-wise non-linearity. Common activation functions include sigmoid, tanh, and [rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29) ("relu" for short). In this example, we'll use the relu activation function which has several desirable properties and is typically considered a default choice.

```python
# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
```

The last fully-connected layer often has the hidden size equal to the number of
output classes in the dataset. Then we stack a softmax layer, which maps its input to a probability score for each class of output type. During the training stage, a loss function computes the cross entropy between the probability distribution (softmax output) predicted by the network and true probability distribution given by the label.

![png](https://raw.githubusercontent.com/madjam/web-data/master/mxnet/image/mlp_mnist.png)

```python
# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
```

Now both the neural network definition and data iterators are ready. We can
start training. The following commands train the MLP on the
MNIST dataset by minibatch stochastic gradient descent (SGD). We'll select a mini-batch size of 100 and learning rate of 0.1. Settings such as batch size and learning rate are usually referred to as hyper-parameters. We'll run the training for 10 epochs and stop. An epoch is one pass over all input data.

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,  # training data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train at most 10 data passes
```

## Convolutional Neural Networks

Earlier we briefly touched on the drawback with MLP where the first fully-connected layer simply reshapes the image into a 784-dimensional vector during training. This discards the fact that pixels in the image have a strong spatial correlation along both horizontal and vertical dimensions. A convolutional neural network (CNN) aims to address this drawback by using a more structured weight *W* representation. Instead of flattening the image and doing a simple matrix-matrix multiplication, it employs one or more convolutional layers that each perform a 2-D convolution on the input image to obtain the output.

Besides the convolutional layer, another major change of the convolutional
neural network is the addition of pooling layers. A pooling layer reduces a
*n x m* patch into a single value to make the network less sensitive to the spatial location.

![png](https://raw.githubusercontent.com/madjam/web-data/master/mxnet/image/conv_mnist.png)

The following code defines a convolutional neural network architecture called LeNet:

```python
data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
```

Now we train LeNet with the same hyper-parameters as before. Note that, if a GPU is available, we recommend using it. This greatly speeds up computation given that LeNet is more complex and compute-intensive than the previous multilayer perceptron. To do so, we only need to change `mx.cpu()` to `mx.gpu()`.

```python
# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
```

## Predict

After training is done, we can test the model we trained by running predictions on test data. The following code computes the prediction probability scores for each test image. *prob[i][j]* is the probability that the *i*-th image contains the *j*-th object in the label set.

```python
test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = mlp_model.predict(test_iter)
assert prob.shape == (10000, 10)
```

Since we also have labels for test images, we can compute the accuracy metric.

```python
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96

# predict accuracy for lenet
acc.reset()
lenet_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
