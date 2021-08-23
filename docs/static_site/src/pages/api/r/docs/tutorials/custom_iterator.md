---
layout: page_api
title: Custom Iterator Tutorial
is_tutorial: true
tag: r
permalink: /api/r/docs/tutorials/custom_iterator
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


Custom Iterator Tutorial
========================

This tutorial provides a guideline on how to use and write custom iterators, which can very useful when having a dataset that does not fit into memory.

Getting the data
----------
The data we are going to use is the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) in CSV format, the data can be found in this [web](https://pjreddie.com/projects/mnist-in-csv/).

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


Conclusion
----------

We have shown how to create a custom CSV Iterator by extending the class `mx.io.CSVIter`. In our class, we iteratively read from a CSV file a batch of data that will be transformed and then processed in the stochastic gradient descent optimization. That way, we are able to manage CSV files that are bigger than the memory of the machine we are using.

Based of this custom iterator, we can also create data loaders that internally transform or expand the data, allowing to manage files of any size.
