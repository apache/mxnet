# Handwritten Digits Image Classification

[MNIST](http://yann.lecun.com/exdb/mnist/) is a popular data set containing simple images of handwritten digits 0-9. 
Every digit is represented by a 28 x 28 pixel image. 
There is a long-term hosted competition on [Kaggle](https://www.kaggle.com/c/digit-recognizer) using MNIST.
This tutorial shows how to use MXNet in R to develop neural network models for competing in this multi-class classification challenge.


## Loading the Data

First, let's download the data from [Kaggle](http://www.kaggle.com/c/digit-recognizer/data) and put it in a ``data/`` sub-folder inside the current working directory.

```{.python .input  n=1}
if (!dir.exists('data')) {
    dir.create('data')
}
if (!file.exists('data/train.csv')) {
    download.file(url='https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/train.csv',
                  destfile='data/train.csv', method='wget')
}
if (!file.exists('data/test.csv')) {
    download.file(url='https://raw.githubusercontent.com/wehrley/Kaggle-Digit-Recognizer/master/test.csv',
                  destfile='data/test.csv', method='wget')
}
```

The above commands rely on ``wget`` being installed on your machine.
If they fail, you can instead manually get the data yourself via the following steps:

1) Create a folder named ``data/`` in the current working directory (to see which directory this is, enter: ``getwd()`` in your current R console). 

2) Navigate to the [Kaggle website](https://www.kaggle.com/c/digit-recognizer/data), log into (or create) your Kaggle account and accept the terms for this competition.

3) Finally, click on **Download All** to download the data to your computer, and copy the files ``train.csv`` and ``test.csv`` into the previously-created ``data/`` folder.

Once the downloads have completed, we can read the data into R and convert it to matrices:

```{.python .input  n=2}
require(mxnet)
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test_features <- data.matrix(test) # Labels are not provided for test data.
train_features <- train[,-1]
train_y <- train[,1]
train_x <- t(train_features/255)
test_x <- t(test_features/255)
```

Every image is represented as a single row in ``train.features``/``test.features``. The greyscale of each image falls in the range [0, 255]. Above, we linearly transform the pixel values into [0,1]. We have also transposed the input matrix to *npixel* x *nexamples*, which is the major format for columns accepted by MXNet (and the convention of R).

Let's view an example image:

```{.python .input  n=3}
i = 10 # change this value to view different examples from the training data

pixels = matrix(train_x[,i],nrow=28, byrow=TRUE)
image(t(apply(pixels, 2, rev)) , col=gray((0:255)/255), 
      xlab="", ylab="", main=paste("Label for this image:", train_y[i]))
```

The proportion of each label within the training data is known to significantly affect the results of multi-class classification models.
We can see that in our MNIST training set, the number of images from each digit is fairly evenly distributed:

```{.python .input  n=4}
table(train_y)
```

## Configuring a neural network

Now that we have the data, let’s create a neural network model. Here, we can use the ``Symbol`` framework in MXNet to declare our desired network architecture:

```{.python .input  n=5}
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
```

1)  ``data`` above represents the input data, i.e. the inputs to our neural network model.

2)  We define the first hidden layer with ``fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)``. 
This is a standard fully-connected layer that takes in ``data`` as its input, can be referenced by the name "fc1", and consists of 128 hidden neurons.

3)  The rectified linear unit ("relu") activation function is chosen for this first layer "fc1". 

4)  The second layer "fc2" is another fully-connected layer that takes the (post-relu) activations of the first hidden layer as its input, consists of 64 hidden neurons, and also employs relu activations. Note that when specifying which activation function to use for which layer, you must refer to the approporiate name of the layer. 

5)  Fully-connected layer "fc3" produces the outputs of our model (it is the ouput layer). Note that this layer employs 10 neurons (corresponding to 10 output values), one for each of the 10 possible classes in our classification task.  To ensure the output values represent valid class-probabilities (i.e. they are nonnegative and sum to 1), the network finally applies the softmax function to the outputs of "fc3".

## Training

We are almost ready to train the neural network we have defined. 
Before we start the computation, let’s decide which device to use:

```{.python .input  n=6}
devices <- mx.cpu()
```

This command tells **mxnet** to use the CPU for all neural network computations. 

Now, you can run the following command to train the neural network!

```{.python .input  n=7}
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train_x, y=train_y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                        epoch.end.callback=mx.callback.log.train.metric(100))
```

Note that ``mx.set.seed`` is the function that controls all randomness in **mxnet** and is critical to ensure reproducible results 
(R's ``set.seed`` function does not govern randomness within **mxnet**).

By declaring we are interested in ``mx.metric.accuracy`` in the above command, the loss function used during training is automatically chosen as the cross-entropy loss -- the de facto choice for multiclass classification tasks.

## Making Predictions

We can easily use our trained network to make predictions on the test data:

```{.python .input  n=8}
preds <- predict(model, test_x)
predicted_labels <- max.col(t(preds)) - 1
table(predicted_labels)
```

There are 28,000 test examples, and our model produces 10 numbers for each example, which represent the estimated probabilities of each class for a given image. ``preds`` is a 28000 x 10 matrix, where each column contains the predicted class-probabilities for a particular test image.  To determine which label our model estimates to be most likely for a test image (``predicted_labels``), we used ``max.col``.

 
Let's view a particular prediction:

```{.python .input  n=9}
i = 2  # change this to view the predictions for different test examples

class_probs_i = preds[,i]
names(class_probs_i) = as.character(0:9)
print("Predicted class probabilities for this test image:"); print(class_probs_i)
image(t(apply(matrix(test_x[,i],nrow=28, byrow=TRUE), 2, rev)) , col=gray((0:255)/255), 
      xlab="", ylab="", main=paste0("Predicted Label: ",predicted_labels[i], ".  Confidence: ", floor(max(class_probs_i)*100),"%"))
```

With a little extra effort, we can create a CSV-formatted file that contains our predictions for all of the test data.
After running the below command, you can submit the ``submission.csv`` file to the [Kaggle competition](https://www.kaggle.com/c/digit-recognizer/submit)!

```{.python .input  n=10}
submission <- data.frame(ImageId=1:ncol(test_x), Label=predicted_labels)
write.csv(submission, file='submission.csv', row.names=FALSE,  quote=FALSE)
```

## Convolutional Neural Network (LeNet)

Previously, we used a standard feedforward neural network (with only fully-connected layers) as our classification model.
For the same task, we can instead use a convolutional neural network, which employs alternative types of layers better-suited for handling the spatial structure present in image data. 
The specific convolutional network we use is the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture, which was previously proposed by Yann LeCun for recognizing handwritten digits.

Here's how we can construct the LeNet network:

```{.python .input  n=10}
# declare input data
data <- mx.symbol.Variable('data')
# first convolutional layer: 20 filters
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# second convolutional layer: 50 filters
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# flatten resulting outputs into a vector
flatten <- mx.symbol.Flatten(data=pool2)
# first fully-connected layer
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fully-connected layer (output layer)
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
lenet <- mx.symbol.SoftmaxOutput(data=fc2)
```

This model first passes the input through two convolutional layers with max-pooling and tanh activations, before subsequently applying two standard fully-connected layers (again with tanh activation). 
The number of filters employed by each convolutional layer can be thought of as the number of distinct patterns that the layer searches for in its input.
In convolutional neural networks, it is important to *flatten* the spatial output of convolutional layers into a vector before passing these values to subsequent fully-connected layers.

We also reshape our data matrices into spatially-arranged arrays, which is important since convolutional layers are highly sensitive to the spatial layout of their inputs.

```{.python .input  n=11}
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))
```

Before training our convolutional network, we once again specify what devices to run computations on.

```{.python .input  n=18}
n.gpu <- 1 # you can set this to the number of GPUs available on your machine

device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
```

We can pass a list of devices to ask MXNet to train on multiple GPUs (you can do this for CPUs, but because internal computation of CPUs is already multi-threaded, there is less gain than with using GPUs).

Start by training our convolutional neural network on the CPU first. Because this takes a bit time, we run it for just one training epoch:

```{.python .input  n=13}
mx.set.seed(0)
model <- mx.model.FeedForward.create(lenet, X=train_array, y=train_y,
            ctx=device.cpu, num.round=1, array.batch.size=100,
            learning.rate=0.05, momentum=0.9, wd=0.00001,
            eval.metric=mx.metric.accuracy,
            epoch.end.callback=mx.callback.log.train.metric(100))
```

Here, ``wd`` specifies a small amount of weight-decay (i.e. l2 regularization) should be employed while training our network parameters.

We could also train the same model using the GPU instead, which can significantly speed up training. 

**Note:** The below command that specifies GPU training will only work if the GPU-version of MXNet has been properly installed. To avoid issues, we set the Boolean flag ``use_gpu`` based on whether or not a GPU is detected in the current environment.

```{.python .input  n=20}
use_gpu <- !inherits(try(mx.nd.zeros(1,mx.gpu()), silent = TRUE), 'try-error') # TRUE if GPU is detected.
if (use_gpu) {
    mx.set.seed(0)
    model <- mx.model.FeedForward.create(lenet, X=train_array, y=train_y,
                ctx=device.gpu, num.round=5, array.batch.size=100,
                learning.rate=0.05, momentum=0.9, wd=0.00001,
                eval.metric=mx.metric.accuracy,
                epoch.end.callback=mx.callback.log.train.metric(100))
}
```

Finally, we can submit the convolutional neural network predictions to Kaggle to see if our ranking in the competition has improved:

```{.python .input  n=15}
preds <- predict(model, test_array)
predicted_labels <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test_x), Label=predicted_labels)
write.csv(submission, file='lenetsubmission.csv', row.names=FALSE, quote=FALSE)
```

## User Exercise 

Try to further improve MNIST classification performance by playing with factors such as:

- the neural network architecture (# of convolutional/fully-connected layers, # of neurons in each layer, the activation functions and pooling-strategy, etc.)

- the type of optimizer (c.f. ``mx.opt.adam``) used and its hyperparameters (e.g. learning rate, momentum)

- how the neural network parameters are initialized (c.f. ``mx.init.normal``)

- different regularization strategies (e.g. altering the value of ``wd`` or introducing dropout: ``mx.symbol.Dropout``)

- augmenting the training data with additional examples created through simple transformations such as rotation/cropping
