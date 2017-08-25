Custom Iterator Tutorial
======================================

This tutorial provides a guideline on how to use and write custom iterators, which can very useful when having a dataset that does not fit into memory.

Getting the data
----------
The data we are going to use is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in CSV format, the data can be found in this [web](http://pjreddie.com/projects/mnist-in-csv/).

To download the data:

```bash
wget http://pjreddie.com/media/files/mnist_train.csv
wget http://pjreddie.com/media/files/mnist_test.csv
```

You'll get two files, `mnist_train.csv` that contains 60.000 examples of hand written numbers and `mxnist_test.csv` that contains 10.000 examples. The first element of each line in the CSV is the label, which is a number between 0 and 9. The rest of the line are 784 numbers between 0 and 255, corresponding to the levels of grey of a matrix of 28x28. Therefore, each line contains an image of 28x28 pixels of a hand written number and its true label.

Custom CSV Iterator
----------
Next we are going to create a custom CSV Iterator based on the [C++ CSVIterator class](https://github.com/dmlc/mxnet/blob/master/src/io/iter_csv.cc).

For that we are going to use the R function `mx.io.CSVIter` as a base class. This class has as parameters `data.csv, data.shape, batch.size` and two main functions, `iter.next()` that calls the iterator in the next batch of data and `value()` that returns the train data and the label.

The R Custom Iterator needs to inherit from the C++ data iterator class, for that we used the class `Rcpp_MXArrayDataIter` extracted with RCPP. Also, it needs to have the same parameters: `data.csv, data.shape, batch.size`. Apart from that, we can also add the field `iter`, which is the CSV Iterator that we are going to expand.

```r
CustomCSVIter <- setRefClass("CustomCSVIter",
								fields=c("iter", "data.csv", "data.shape", "batch.size"),
								contains = "Rcpp_MXArrayDataIter",
								#...
                            )
```

The next step is to initialize the class. For that we call the base `mx.io.CSVIter` and fill the rest of the fields.

```r
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
                             	#...
                             	)
                            )
```

So far there is no difference between the original class and the custom class. Let's implement the function `value()`. In this case what we are going to do is transform the data that comes from the original class as an array of 785 numbers into a matrix of 28x28 and a label. We will also normalize the training data to be between 0 and 1.

```r
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
                             	#...
                             	)
                            )
```
Finally we are going to add the rest of the functions needed for the training to work correctly. The final `CustomCSVIter` looks like this:

```r
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

```r
batch.size <- 100
train.iter <- CustomCSVIter$new(iter = NULL, data.csv = "mnist_train.csv", data.shape = 28, batch.size = batch.size)
```

CNN Model
----------

For this tutorial we are going to use the known LeNet architecture:

```r
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

Training with the Custom Iterator
----------
Finally, we can directly add the custom iterator as the training data source.

```r
model <- mx.model.FeedForward.create(symbol=network,
                                     X=train.iter,
                                     ctx=mx.gpu(0),
                                     num.round=10,
                                     array.batch.size=batch.size,
                                     learning.rate=0.1,
                                     momentum=0.9,  
                                     eval.metric=mx.metric.accuracy,
                                     wd=0.00001,
                                     batch.end.callback=mx.callback.log.speedometer(batch.size, frequency = 100)
                                     )
```

The last 2 iterations with a K80 GPU looks like this:

```bash
[8] Train-accuracy=0.998866666666667
Batch [100] Speed: 15413.0104454713 samples/sec Train-accuracy=0.999
Batch [200] Speed: 16629.3412459049 samples/sec Train-accuracy=0.99935
Batch [300] Speed: 18412.6900509319 samples/sec Train-accuracy=0.9995
Batch [400] Speed: 16757.2882328335 samples/sec Train-accuracy=0.999425
Batch [500] Speed: 17116.6529207406 samples/sec Train-accuracy=0.99946
Batch [600] Speed: 19627.589505195 samples/sec Train-accuracy=0.99945
[9] Train-accuracy=0.9991
Batch [100] Speed: 18971.5745536982 samples/sec Train-accuracy=0.9992
Batch [200] Speed: 15554.8822435383 samples/sec Train-accuracy=0.99955
Batch [300] Speed: 18327.6950115053 samples/sec Train-accuracy=0.9997
Batch [400] Speed: 17103.0705411788 samples/sec Train-accuracy=0.9997
Batch [500] Speed: 15104.8656902394 samples/sec Train-accuracy=0.99974
Batch [600] Speed: 13818.7899518255 samples/sec Train-accuracy=0.99975
[10] Train-accuracy=0.99975
```

Conclusion
----------

We have shown how to create a custom CSV Iterator by extending the class `mx.io.CSVIter`. In our class, we iteratively read from a CSV file a batch of data that will be transformed and then processed in the stochastic gradient descent optimization. That way, we are able to manage CSV files that are bigger than the memory of the machine we are using.

Based of this custom iterator, we can also create data loaders that internally transform or expand the data, allowing to manage files of any size.
