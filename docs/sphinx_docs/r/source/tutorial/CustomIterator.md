# Custom Iterator Tutorial

This tutorial provides a guideline on how to use and write custom iterators, which can help handle a dataset that does not fit into memory.

## Getting the data

The data we are going to use is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in CSV format. 
The data can be found at this [website](https://pjreddie.com/projects/mnist-in-csv/).

To download the data from R:

```{.python .input .R  n=4}
if (!file.exists('mnist_train.csv')) {
    download.file(url='http://pjreddie.com/media/files/mnist_train.csv',
                  destfile='mnist_train.csv', method='wget')
}
if (!file.exists('mnist_test.csv')) {
    download.file(url='http://pjreddie.com/media/files/mnist_test.csv',
                  destfile='mnist_test.csv', method='wget')
}
```

You’ll get two files, **mnist_train.csv** that contains 60,000 images of handwritten numbers and **mnist_test.csv** that contains 10,000 such images, all of which are formatted in a comma-separated value (CSV) style.
The first element of each line in the CSV is the label, which is a number between 0 and 9. The rest of the line are 784 numbers between 0 and 255, corresponding to the levels of grey of a matrix of 28x28 pixels which together comprise the image. Thus, each line of the file contains an image of 28x28 pixels of a hand written number and its true label.

**Note:** The above command relies on ``wget``.  If the command fails, you can manually download these data files: 
first navigate to the links above in your browser, and then place the downloaded files **mnist_train.csv** & **mnist_test.csv**  into the curent working directory of our R session (use ``getwd()`` command to print this directory from the current R notebook).

## Custom CSV Iterator

We will create a custom CSV Iterator based on the [C++ CSVIterator class](https://github.com/apache/incubator-mxnet/blob/master/src/io/iter_csv.cc).

For that we are going to use the R function ``mx.io.CSVIter`` as a base class. This class has as parameters ``data.csv``, ``data.shape``, ``batch.size`` and two main functions, ``iter.next()`` that calls the iterator in the next batch of data and ``value()`` that returns the train data and the label.

The R Custom Iterator needs to inherit from the C++ data iterator class, for that we used the class ``Rcpp_MXArrayDataIter`` extracted with RCPP. Also, it needs to have the same parameters: ``data.csv``, ``data.shape``, ``batch.size``. Apart from that, we can also add the field ``iter``, which is the CSV Iterator that we are going to expand.

```{.python .input .R  n=9}
require(mxnet)
CustomCSVIter <- setRefClass("CustomCSVIter",
                             fields=c("iter", "data.csv", "data.shape", "batch.size"),
                             contains = "Rcpp_MXArrayDataIter",
                             # , ... This is just an incomplete example, we will add more arguments later.
                            )
```

The next step is to initialize the class. For that we call the base ``mx.io.CSVIter`` and fill the rest of the fields.

```{.python .input .R  n=11}
CustomCSVIter <- setRefClass("CustomCSVIter",
                                fields=c("iter", "data.csv", "data.shape", "batch.size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods=list(
                                    initialize=function(iter, data.csv, data.shape, batch.size){
                                        feature_len <- data.shape*data.shape + 1
                                        csv_iter <- mx.io.CSVIter(data.csv=data.csv, data.shape=c(feature_len), batch.size=batch.size)
                                        .self$iter <- csv_iter
                                        .self$data.csv <- data.csv
                                        .self$data.shape <- data.shape
                                        .self$batch.size <- batch.size
                                        .self
                                    })
                                # , ... # This is just an incomplete example, we will add more arguments later.
                                )
```

So far there is no difference between the original class and our custom class. Let’s implement the function ``value()``. In this case, what we are going to do is transform the data that comes from the original class as an array of 785 numbers into a matrix of 28x28 and a label. We will also normalize the training data to be between 0 and 1.

```{.python .input .R}
CustomCSVIter <- setRefClass("CustomCSVIter",
                                fields=c("iter", "data.csv", "data.shape", "batch.size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods=list(
                                    initialize=function(iter, data.csv, data.shape, batch.size){
                                        feature_len <- data.shape*data.shape + 1
                                        csv_iter <- mx.io.CSVIter(data.csv=data.csv, data.shape=c(feature_len), batch.size=batch.size)
                                        .self$iter <- csv_iter
                                        .self$data.csv <- data.csv
                                        .self$data.shape <- data.shape
                                        .self$batch.size <- batch.size
                                        .self
                                    },
                                    value=function(){
                                        val <- as.array(.self$iter$value()$data)
                                        val.x <- val[-1,]
                                        val.y <- val[1,]
                                        val.x <- val.x/255
                                        dim(val.x) <- c(data.shape, data.shape, 1, ncol(val.x))
                                        val.x <- mx.nd.array(val.x)
                                        val.y <- mx.nd.array(val.y)
                                        list(data=val.x, label=val.y)
                                    }
                                # , ... This is just an incomplete example, we will add more arguments later.
                                )
                            )
```

Finally we are going to add the rest of the functions needed for the training to work correctly. The final ``CustomCSVIter`` looks like this:

```{.python .input .R  n=12}
CustomCSVIter <- setRefClass("CustomCSVIter",
                                fields=c("iter", "data.csv", "data.shape", "batch.size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods=list(
                                    initialize=function(iter, data.csv, data.shape, batch.size){
                                        feature_len <- data.shape*data.shape + 1
                                        csv_iter <- mx.io.CSVIter(data.csv=data.csv, data.shape=c(feature_len), batch.size=batch.size)
                                        .self$iter <- csv_iter
                                        .self$data.csv <- data.csv
                                        .self$data.shape <- data.shape
                                        .self$batch.size <- batch.size
                                        .self
                                    },
                                    value=function(){
                                        val <- as.array(.self$iter$value()$data)
                                        val.x <- val[-1,]
                                        val.y <- val[1,]
                                        val.x <- val.x/255
                                        dim(val.x) <- c(data.shape, data.shape, 1, ncol(val.x))
                                        val.x <- mx.nd.array(val.x)
                                        val.y <- mx.nd.array(val.y)
                                        list(data=val.x, label=val.y)
                                    },
                                    iter.next=function(){
                                        .self$iter$iter.next()
                                    },
                                    reset=function(){
                                        .self$iter$reset()
                                    },
                                    num.pad=function(){
                                        .self$iter$num.pad()
                                    },
                                    finalize=function(){
                                        .self$iter$finalize()
                                    }
                                )
                            )
```

To call the class we can just do:

```{.python .input .R  n=13}
batch.size <- 100
train.iter <- CustomCSVIter$new(iter = NULL, data.csv = "mnist_train.csv", data.shape = 28, batch.size = batch.size)
```

## CNN Model

For the rest of this tutorial, we are going to use the known LeNet architecture.
This is a convolutional neural network classification model with two convolution layers that use max-pooling, two fully-connected layers, and tanh-activation functions.

```{.python .input .R  n=16}
lenet.model <- function(){
  data <- mx.symbol.Variable('data')
  conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20) #first conv
  tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
  pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
  conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)# second conv
  tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
  pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  flatten <- mx.symbol.Flatten(data=pool2)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=100) # first fullc
  tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
  fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10) # second fullc
  network <- mx.symbol.SoftmaxOutput(data=fc2) # loss
  network
}
network <- lenet.model()
```

## Training with the Custom Iterator

Finally, we can directly add the custom iterator as the training data source.

In order to speed up the code below, you can switch ``ctx=mx.gpu(0)`` to perform training on a GPU instead of the CPU (assuming you have already properly installed the GPU-version of MXNet).

```{.python .input .R  n=17}
model <- mx.model.FeedForward.create(symbol=network,
                                     X=train.iter,
                                     ctx=mx.cpu(0), # To train on GPU instead, use: ctx=mx.gpu(0),
                                     num.round=2,
                                     array.batch.size=batch.size,
                                     learning.rate=0.1,
                                     momentum=0.9,  
                                     eval.metric=mx.metric.accuracy,
                                     wd=0.00001,
                                     batch.end.callback=mx.callback.log.speedometer(batch.size, frequency = 100)
                                     )
```

## Conclusion

We have shown how to create a custom CSV Iterator by extending the class ``mx.io.CSVIter``. In our class, we iteratively read from a CSV file a batch of data that will be transformed and then processed in the stochastic gradient descent optimization. That way, we are able to manage CSV files that are bigger than the memory of the machine we are using.

Based on this custom iterator, we can also create data loaders that internally transform or expand the data, allowing us to handle training data files of any size/format.
