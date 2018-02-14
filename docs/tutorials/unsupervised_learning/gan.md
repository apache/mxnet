# Generative Adversarial Network (GAN)

Generative Adversarial Networks (GANs) are a class of algorithms used in unsupervised learning - you don't need labels for your dataset in order to train a GAN.

The GAN framework is composed of two neural networks: a Generator network and a Discriminator network.

The Generator's job is to take a set of random numbers and produce the data (such as images or text).

The Discriminator then takes in that data as well as samples of that data from a dataset and tries to determine if it is "fake" (created by the Generator network) or "real" (from the original dataset).

During training, the two networks play a game against each other. The Generator tries to create realistic data, so that it can fool the Discriminator into thinking that the data it generated is from the original dataset. At the same time, the Discriminator tries to not be fooled - it learns to become better at determining if data is real or fake.

Since the two networks are fighting in this game, they can be seen as as adversaries, which is where the term "Generative Adversarial Network" comes from.

## Deep Convolutional Generative Adversarial Networks

This tutorial takes a look at Deep Convolutional Generative Adversarial Networks (DCGAN), which combines Convolutional Neural Networks (CNNs) and GANs.

We will create a DCGAN that is able to create images of handwritten digits from random numbers. The tutorial uses the neural net architecture and guidelines outlined in [this paper](https://arxiv.org/abs/1511.06434), and the MNIST dataset.

## How to Use This Tutorial
You can use this tutorial by executing each snippet of python code in order as it appears in the tutorial.


1. The first net is the "Generator" and creates images of handwritten digits from random numbers.
2. The second net is the "Discriminator" and determines if the image created by the Generator is real (a realistic looking image of handwritten digits) or fake (an image that does not look like it is from the original dataset).

Apart from creating a DCGAN, you'll also learn:

- How to manipulate and iterate through batches of image data that you can feed into your neural network.

- How to create a custom MXNet data iterator that generates random numbers from a normal distribution.

- How to create a custom training process in MXNet, using lower level functions from the MXNet Module API such as .bind() .forward() and .backward(). The training process for a DCGAN is more complex than many other neural networks, so we need to use these functions instead of using the higher level .fit() function.

- How to visualize images as they are going through the training process

## Prerequisites

This tutorial assumes you are familiar with the concepts of CNNs and have implemented one in MXNet. You should also be familiar with the concept of logistic regression. Having a basic understanding of MXNet data iterators helps, since we will create a custom data iterator to iterate though random numbers as inputs to the Generator network.

This example is designed to be trained on a single GPU. Training this network on CPU can be slow, so it's recommended that you use a GPU for training.

To complete this tutorial, you need:

- MXNet
- Python 2.7, and the following libraries for Python:
    - Numpy - for matrix math
    - OpenCV - for image manipulation
    - Scikit-learn - to easily get the MNIST dataset
    - Matplotlib - to visualize the output

## The Data
We need two pieces of data to train the DCGAN:
    1. Images of handwritten digits from the MNIST dataset
    2. Random numbers from a normal distribution

The Generator network will use the random numbers as the input to produce the images of handwritten digits, and the Discriminator network will use images of handwritten digits from the MNIST dataset to determine if images produced by the Generator are realistic.

We are going to use the python library, scikit-learn, to get the MNIST dataset. Scikit-learn comes with a function that gets the dataset for us, which we will then manipulate to create the training and testing inputs.

The MNIST dataset contains 70,000 images of handwritten digits. Each image is 28x28 pixels in size. To create random numbers, we're going to create a custom MXNet data iterator, which will returns random numbers from a normal distribution as we need then.

## Prepare the Data

### 1. Preparing the MNSIT dataset

Let us start by preparing the handwritten digits from the MNIST dataset. We import the fetch_mldata function from scikit-learn, and use it to get the MNSIT dataset. Notice that it's shape is 70000x784. This contains 70000 images, one per row and 784 pixels of each image in the columns of each row. Each image is 28x28 pixels, but has been flattened so that all 784 pixels are represented in a single list.

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
```

Next, we will randomize the handwritten digits by using numpy to create random permutations on the dataset on the rows (images). We will then reshape the dataset from 70000x786 to 70000x28x28, so that every image in the dataset is arranged into a 28x28 grid, where each cell in the grid represents 1 pixel of the image.

```python
import numpy as np
#Use a seed so that we get the same random permutation each time
np.random.seed(1)
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
X = X.reshape((70000, 28, 28))
```
Since the DCGAN that we're creating takes in a 64x64 image as the input, we will use OpenCV to resize the each 28x28 image to 64x64 images:
```python
import cv2
X = np.asarray([cv2.resize(x, (64,64)) for x in X])
```
Each pixel in the 64x64 image is represented by a number between 0-255, that represents the intensity of the pixel. However, we want to input numbers between -1 and 1 into the DCGAN, as suggested by the [research paper](https://arxiv.org/abs/1511.06434). To rescale the pixel values, we will divide it by (255/2). This changes the scale to 0-2. We then subtract by 1 to get them in the range of -1 to 1.

```python
X = X.astype(np.float32)/(255.0/2) - 1.0
```
Ultimately, images are fed into the neural net through a 70000x3x64x64 array but they are currently in a 70000x64x64 array. We need to add 3 channels to the images. Typically, when we are working with the images, the 3 channels represent the red, green, and blue (RGB) components of each image. Since the MNIST dataset is grayscale, we only need 1 channel to represent the dataset. We will pad the other channels with 0's:

```python
X = X.reshape((70000, 1, 64, 64))
X = np.tile(X, (1, 3, 1, 1))
```
Finally, we will put the images into MXNet's NDArrayIter, which will allow MXNet to easily iterate through the images during training. We will also split them up into batches of 64 images each. Every time we iterate, we will get a 4 dimensional array with size (64, 3, 64, 64), representing a batch of 64 images.
```python
import mxnet as mx
batch_size = 64
image_iter = mx.io.NDArrayIter(X, batch_size=batch_size)
```
### 2. Preparing Random Numbers

We need to input random numbers from a normal distribution to the Generator network, so we will create an MXNet DataIter that produces random numbers for each training batch. The DataIter is the base class of MXNet's Data Loading API. Below, we create a class called RandIter which is a subclass of DataIter. We use MXNet's built-in mx.random.normal function to return the random numbers from a normal distribution during the iteration.

```python
class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        #Returns random numbers from a gaussian (normal) distribution
        #with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]
```
When we initialize the RandIter, we need to provide two numbers: the batch size and how many random numbers we want in order to produce a single image from. This number is referred to as Z, and we will set this to 100. This value comes from the research paper on the topic. Every time we iterate and get a batch of random numbers, we will get a 4 dimensional array with shape: (batch_size, Z, 1, 1), which in the example is (64, 100, 1, 1).
```python
Z = 100
rand_iter = RandIter(batch_size, Z)
```
## Create the Model

The model has two networks that we will train together - the Generator network and the Discriminator network.

### The Generator

Let us start off by defining the Generator network, which uses Deconvolution layers (also called as fractionally strided layers) to generate an image form random numbers :
```python
no_bias = True
fix_gamma = True
epsilon = 1e-5 + 1e-12

rand = mx.sym.Variable('rand')

g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=1024, no_bias=no_bias)
gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=epsilon)
gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=epsilon)
gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=epsilon)
gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=epsilon)
gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, no_bias=no_bias)
generatorSymbol = mx.sym.Activation(g5, name='gact5', act_type='tanh')
```

The Generator image starts with random numbers that will be obtained from the RandIter we created earlier, so we created the rand variable for this input.
We then start creating the model starting with a Deconvolution layer (sometimes called 'fractionally strided layer'). We apply batch normalization and ReLU activation after the Deconvolution layer.

We repeat this process 4 times, applying a (2,2) stride and (1,1) pad at each Deconvolution layer, which doubles the size of the image at each layer. By creating these layers, the Generator network will have to learn to upsample the input vector of random numbers, Z at each layer, so that network output a final image. We also reduce by half the number of filters at each layer, reducing dimensionality at each layer. Ultimately, the output layer is a 64x64x3 layer, representing the size and channels of the image. We use tanh activation instead of relu on the last layer, as recommended by the research on DCGANs. The output of neurons in the final gout layer represent the pixels of generated image.

Notice we used 3 parameters to help us create the model: no_bias, fixed_gamma, and epsilon. Neurons in the network won't have a bias added to them, this seems to work better in practice for the DCGAN. In the batch norm layer, we set fixed_gamma=True, which means gamma=1 for all of the batch norm layers. epsilon is a small number that gets added to the batch norm so that we don't end up dividing by zero. By default, CuDNN requires that this number is greater than 1e-5, so we add a small number to this value, ensuring this values stays small.

### The Discriminator

Let us now create the Discriminator network, which will take in images of handwritten digits from the MNIST dataset and images created by the Generator network:
```python
data = mx.sym.Variable('data')

d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=epsilon)
dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=epsilon)
dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=1024, no_bias=no_bias)
dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=epsilon)
dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
d5 = mx.sym.Flatten(d5)

label = mx.sym.Variable('label')
discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')
```

We start off by creating the data variable, which is used to hold the input images to the Discriminator.

The Discriminator then goes through a series of 5 convolutional layers, each with a 4x4 kernel, 2x2 stride, and 1x1 pad. These layers half the size of the image (which starts at 64x64) at each convolutional layer. The model also increases dimensionality at each layer by doubling the number of filters per convolutional layer, starting at 128 filters and ending at 1024 filters before we flatten the output.

At the final convolution, we flatten the neural net to get one number as the final output of Discriminator network. This number is the probability that the image is real, as determined by the Discriminator. We use logistic regression to determine this probability. When we pass in "real" images from the MNIST dataset, we can label these as 1 and we can label the "fake" images from the Generator net as 0 to perform logistic regression on the Discriminator network.

### Prepare the models using the Module API

So far we have defined a MXNet Symbol for both the Generator and the Discriminator network. Before we can train the model, we need to bind these symbols using the Module API, which creates the computation graph for the models. It also allows us to decide how we want to initialize the model and what type of optimizer we want to use. Let us set up the Module for both the networks:
```python
#Hyper-parameters
sigma = 0.02
lr = 0.0002
beta1 = 0.5
# If you do not have a GPU. Use the below outlined
# ctx = mx.cpu()
ctx = mx.gpu(0)

#=============Generator Module=============
generator = mx.mod.Module(symbol=generatorSymbol, data_names=('rand',), label_names=None, context=ctx)
generator.bind(data_shapes=rand_iter.provide_data)
generator.init_params(initializer=mx.init.Normal(sigma))
generator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods = [generator]

# =============Discriminator Module=============
discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=('data',), label_names=('label',), context=ctx)
discriminator.bind(data_shapes=image_iter.provide_data,
          label_shapes=[('label', (batch_size,))],
          inputs_need_grad=True)
discriminator.init_params(initializer=mx.init.Normal(sigma))
discriminator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods.append(discriminator)
```
First, we create Modules for the networks and then bind the symbols that we've created in the previous steps to the modules.
We use rand_iter.provide_data as the  data_shape to bind the Generator network. This means that as we iterate though batches of the data on the Generator Module, the RandIter will provide us with random numbers to feed the Module using it's provide_data function.

Similarly, we bind the Discriminator Module to image_iter.provide_data, which gives us images from MNIST from the NDArrayIter we had set up earlier, called image_iter.

Notice that we are using the Normal Initialization, with the hyperparameter sigma=0.02. This means the weight initializations for the neurons in the networks will be random numbers from a Gaussian (normal) distribution with a mean of 0 and a standard deviation of 0.02.

We also use the Adam optimizer for gradient decent. We've set up two hyperparameters, lr and beta1 based on the values used in the DCGAN paper. We're using a single gpu, gpu(0) for training. Set the context to cpu() if you do not have a GPU on your machine.

### Visualizing The Training
Before we train the model, let us set up some helper functions that will help visualize what the Generator is producing, compared to what the real image is:
```python
from matplotlib import pyplot as plt

#Takes the images in the batch and arranges them in an array so that they can be
#Plotted using matplotlib
def fill_buf(buf, num_images, img, shape):
    width = buf.shape[0]/shape[1]
    height = buf.shape[1]/shape[0]
    img_width = int(num_images%width)*shape[0]
    img_hight = int(num_images/height)*shape[1]
    buf[img_hight:img_hight+shape[1], img_width:img_width+shape[0], :] = img

#Plots two images side by side using matplotlib
def visualize(fake, real):
    #64x3x64x64 to 64x64x64x3
    fake = fake.transpose((0, 2, 3, 1))
    #Pixel values from 0-255
    fake = np.clip((fake+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    #Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)

    #Create buffer array that will hold all the images in the batch
    #Fill the buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n*fake.shape[1]), int(n*fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n*real.shape[1]), int(n*real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])

    #Create a matplotlib figure with two subplots: one for the real and the other for the fake
    #fill each plot with the buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(rbuff)
    plt.show()
```

## Fit the Model
Training the DCGAN is a complex process that requires multiple steps.
To fit the model, for every batch of data in the MNIST dataset:

1. Use the Z vector, which contains the random numbers to do a forward pass through the Generator network. This outputs the "fake" image, since it is created from the Generator.

2. Use the fake image as the input to do a forward and backward pass through the Discriminator network. We set the labels for logistic regression to 0 to represent that this is a fake image. This trains the Discriminator to learn what a fake image looks like. We save the gradient produced in backpropagation for the next step.

3. Do a forward and backward pass through the Discriminator using a real image from the MNIST dataset. The label for logistic regression will now be 1 to represent the real images, so the Discriminator can learn to recognize a real image.

4. Update the Discriminator by adding the result of the gradient generated during backpropagation on the fake image with the gradient from backpropagation on the real image.

5. Now that the Discriminator has been updated for the this data batch, we still need to update the Generator. First, do a forward and backwards pass with the same data batch on the updated Discriminator, to produce a new gradient. Use the new gradient to do a backwards pass

Here is the main training loop for the DCGAN:

```python
# =============train===============
print('Training...')
for epoch in range(1):
    image_iter.reset()
    for i, batch in enumerate(image_iter):
        #Get a batch of random numbers to generate an image from the generator
        rbatch = rand_iter.next()
        #Forward pass on training batch
        generator.forward(rbatch, is_train=True)
        #Output of training batch is the 64x64x3 image
        outG = generator.get_outputs()

        #Pass the generated (fake) image through the discriminator, and save the gradient
        #Label (for logistic regression) is an array of 0's since this image is fake
        label = mx.nd.zeros((batch_size,), ctx=ctx)
        #Forward pass on the output of the discriminator network
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        #Do the backward pass and save the gradient
        discriminator.backward()
        gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

        #Pass a batch of real images from MNIST through the discriminator
        #Set the label to be an array of 1's because these are the real images
        label[:] = 1
        batch.label = [label]
        #Forward pass on a batch of MNIST images
        discriminator.forward(batch, is_train=True)
        #Do the backward pass and add the saved gradient from the fake images to the gradient
        #generated by this backwards pass on the real images
        discriminator.backward()
        for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        #Update gradient on the discriminator
        discriminator.update()

        #Now that we've updated the discriminator, let's update the generator
        #First do a forward pass and backwards pass on the newly updated discriminator
        #With the current batch
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        #Get the input gradient from the backwards pass on the discriminator,
        #and use it to do the backwards pass on the generator
        diffD = discriminator.get_input_grads()
        generator.backward(diffD)
        #Update the gradients on the generator
        generator.update()

        #Increment to the next batch, printing every 50 batches
        i += 1
        if i % 50 == 0:
            print('epoch:', epoch, 'iter:', i)
            print
            print("   From generator:        From MNIST:")

            visualize(outG[0].asnumpy(), batch.data[0].asnumpy())
```

This will train the GAN network and visualize the progress that we are making as the networks are trained. After every 25 iterations, we are calling the visualize function that we created earlier, which plots the intermediate results.

The plot on the left will represent what the Generator created (the fake image) in the most recent iteration. The plot on the right will represent the Original (real) image from the MNIST dataset that was inputted to the Discriminator on the same iteration.

As the training goes on, the Generator becomes better at generating realistic images. You can see this happening since the images on the left becomes closer to the original dataset with each iteration.

## Summary

We have now successfully used Apache MXNet to train a Deep Convolutional Generative Adversarial Neural Networks (DCGAN) using the MNIST dataset.

As a result, we have created two neural nets: a Generator, which is able to create images of handwritten digits from random numbers, and a Discriminator, which is able to take an image and determine if it is an image of handwritten digits.

Along the way, we have learned how to do the image manipulation and visualization that is associated with the training of deep neural nets. We have also learned how to use MXNet's Module APIs to perform advanced model training functionality to fit the model.

## Acknowledgements
This tutorial is based on [MXNet DCGAN codebase](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dcgan.py),
[The original paper on GANs](https://arxiv.org/abs/1406.2661), as well as [this paper on deep convolutional GANs](https://arxiv.org/abs/1511.06434).
