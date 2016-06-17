Handwritten Digits Classification Competition
=============================================

[MNIST](http://yann.lecun.com/exdb/mnist/) is a handwritten digits image data set created by Yann LeCun. Every digit is represented by a 28x28 image. It has become a standard data set to test classifiers on simple image input. Neural network is no doubt a strong model for image classification tasks. There's a [long-term hosted competition](https://www.kaggle.com/c/digit-recognizer) on Kaggle using this data set.
We will present the basic usage of [mxnet](https://github.com/dmlc/mxnet/tree/master/R-package) to compete in this challenge.

This tutorial is written in Rmarkdown. You can download the source [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/mnistCompetition.Rmd) and view a
hosted version of tutorial [here](http://mxnet.readthedocs.io/en/latest/packages/r/mnistCompetition.html).

## Data Loading

First, let us download the data from [here](https://www.kaggle.com/c/digit-recognizer/data), and put them under the `data/` folder in your working directory.

Then we can read them in R and convert to matrices.


```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

```r
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]
```

Here every image is represented as a single row in train/test. The greyscale of each image falls in the range [0, 255], we can linearly transform it into [0,1] by


```r
train.x <- t(train.x/255)
test <- t(test/255)
```
We also transpose the input matrix to npixel x nexamples, which is the column major format accepted by mxnet (and the convention of R).

In the label part, we see the number of each digit is fairly even:


```r
table(train.y)
```

```
## train.y
##    0    1    2    3    4    5    6    7    8    9
## 4132 4684 4177 4351 4072 3795 4137 4401 4063 4188
```

## Network Configuration

Now we have the data. The next step is to configure the structure of our network.


```r
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
```

1. In `mxnet`, we use its own data type `symbol` to configure the network. `data <- mx.symbol.Variable("data")` use `data` to represent the input data, i.e. the input layer.
2. Then we set the first hidden layer by `fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)`. This layer has `data` as the input, its name and the number of hidden neurons.
3. The activation is set by `act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")`. The activation function takes the output from the first hidden layer `fc1`.
4. The second hidden layer takes the result from `act1` as the input, with its name as "fc2" and the number of hidden neurons as 64.
5. the second activation is almost the same as `act1`, except we have a different input source and name.
6. Here comes the output layer. Since there's only 10 digits, we set the number of neurons to 10.
7. Finally we set the activation to softmax to get a probabilistic prediction.

## Training

We are almost ready for the training process. Before we start the computation, let's decide what device should we use.


```r
devices <- mx.cpu()
```

Here we assign CPU to `mxnet`. After all these preparation, you can run the following command to train the neural network! Note that `mx.set.seed` is the correct function to control the random process in `mxnet`.


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Start training with 1 devices
## Batch [100] Train-accuracy=0.6563
## Batch [200] Train-accuracy=0.777999999999999
## Batch [300] Train-accuracy=0.827466666666665
## Batch [400] Train-accuracy=0.855499999999999
## [1] Train-accuracy=0.859832935560859
## Batch [100] Train-accuracy=0.9529
## Batch [200] Train-accuracy=0.953049999999999
## Batch [300] Train-accuracy=0.955866666666666
## Batch [400] Train-accuracy=0.957525000000001
## [2] Train-accuracy=0.958309523809525
## Batch [100] Train-accuracy=0.968
## Batch [200] Train-accuracy=0.9677
## Batch [300] Train-accuracy=0.9696
## Batch [400] Train-accuracy=0.970650000000002
## [3] Train-accuracy=0.970809523809526
## Batch [100] Train-accuracy=0.973
## Batch [200] Train-accuracy=0.974249999999999
## Batch [300] Train-accuracy=0.976
## Batch [400] Train-accuracy=0.977100000000003
## [4] Train-accuracy=0.977452380952384
## Batch [100] Train-accuracy=0.9834
## Batch [200] Train-accuracy=0.981949999999999
## Batch [300] Train-accuracy=0.981900000000001
## Batch [400] Train-accuracy=0.982600000000003
## [5] Train-accuracy=0.983000000000003
## Batch [100] Train-accuracy=0.983399999999999
## Batch [200] Train-accuracy=0.98405
## Batch [300] Train-accuracy=0.985000000000001
## Batch [400] Train-accuracy=0.985725000000003
## [6] Train-accuracy=0.985952380952384
## Batch [100] Train-accuracy=0.988999999999999
## Batch [200] Train-accuracy=0.9876
## Batch [300] Train-accuracy=0.988100000000001
## Batch [400] Train-accuracy=0.988750000000003
## [7] Train-accuracy=0.988880952380955
## Batch [100] Train-accuracy=0.991999999999999
## Batch [200] Train-accuracy=0.9912
## Batch [300] Train-accuracy=0.990066666666668
## Batch [400] Train-accuracy=0.990275000000003
## [8] Train-accuracy=0.990452380952384
## Batch [100] Train-accuracy=0.9937
## Batch [200] Train-accuracy=0.99235
## Batch [300] Train-accuracy=0.991966666666668
## Batch [400] Train-accuracy=0.991425000000003
## [9] Train-accuracy=0.991500000000003
## Batch [100] Train-accuracy=0.9942
## Batch [200] Train-accuracy=0.99245
## Batch [300] Train-accuracy=0.992433333333334
## Batch [400] Train-accuracy=0.992275000000002
## [10] Train-accuracy=0.992380952380955
```

## Prediction and Submission

To make prediction, we can simply write


```r
preds <- predict(model, test)
dim(preds)
```

```
## [1]    10 28000
```

It is a matrix with 28000 rows and 10 cols, containing the desired classification probabilities from the output layer. To extract the maximum label for each row, we can use the `max.col` in R:


```r
pred.label <- max.col(t(preds)) - 1
table(pred.label)
```

```
## pred.label
##    0    1    2    3    4    5    6    7    8    9
## 2818 3195 2744 2767 2683 2596 2798 2790 2784 2825
```

With a little extra effort in the csv format, we can have our submission to the competition!


```r
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)
```

## LeNet

Next we are going to introduce a new network structure: [LeNet](http://yann.lecun.com/exdb/lenet/). It is proposed by Yann LeCun to recognize handwritten digits. Now we are going to demonstrate how to construct and train an LeNet in `mxnet`.

First we construct the network:


```r
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)
```

Then let us reshape the matrices into arrays:


```r
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))
```

Next we are going to compare the training speed on different devices, so the definition of the devices goes first:


```r
n.gpu <- 1
device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
```

As you can see, we can pass a list of devices, to ask mxnet to train on multiple GPUs (you can do similar thing for cpu,
but since internal computation of cpu is already multi-threaded, there is less gain than using GPUs).

We start by training on CPU first. Because it takes a bit time to do so, we will only run it for one iteration.


```r
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=1, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Start training with 1 devices
## Batch [100] Train-accuracy=0.1066
## Batch [200] Train-accuracy=0.16495
## Batch [300] Train-accuracy=0.401766666666667
## Batch [400] Train-accuracy=0.537675
## [1] Train-accuracy=0.557136038186157
```

```r
print(proc.time() - tic)
```

```
##    user  system elapsed
## 130.030 204.976  83.821
```

Training on GPU:


```r
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.gpu, num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Start training with 1 devices
## Batch [100] Train-accuracy=0.1066
## Batch [200] Train-accuracy=0.1596
## Batch [300] Train-accuracy=0.3983
## Batch [400] Train-accuracy=0.533975
## [1] Train-accuracy=0.553532219570405
## Batch [100] Train-accuracy=0.958
## Batch [200] Train-accuracy=0.96155
## Batch [300] Train-accuracy=0.966100000000001
## Batch [400] Train-accuracy=0.968550000000003
## [2] Train-accuracy=0.969071428571432
## Batch [100] Train-accuracy=0.977
## Batch [200] Train-accuracy=0.97715
## Batch [300] Train-accuracy=0.979566666666668
## Batch [400] Train-accuracy=0.980900000000003
## [3] Train-accuracy=0.981309523809527
## Batch [100] Train-accuracy=0.9853
## Batch [200] Train-accuracy=0.985899999999999
## Batch [300] Train-accuracy=0.986966666666668
## Batch [400] Train-accuracy=0.988150000000002
## [4] Train-accuracy=0.988452380952384
## Batch [100] Train-accuracy=0.990199999999999
## Batch [200] Train-accuracy=0.98995
## Batch [300] Train-accuracy=0.990600000000001
## Batch [400] Train-accuracy=0.991325000000002
## [5] Train-accuracy=0.991523809523812
```

```r
print(proc.time() - tic)
```

```
##    user  system elapsed
##   9.288   1.680   6.889
```

As you can see by using GPU, we can get a much faster speedup in training!
Finally we can submit the result to Kaggle again to see the improvement of our ranking!


```r
preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)
```

![](../../web-data/mxnet/knitr/mnistCompetition-kaggle-submission.png)
