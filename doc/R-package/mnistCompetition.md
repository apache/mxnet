---
title: "Handwritten Digits Classification Competition"
author: "Tong He"
date: "October 17, 2015"
output: html_document
---

[MNIST](http://yann.lecun.com/exdb/mnist/) is a handwritten digits image data set created by Yann LeCun. Every digit is represented by a 28x28 image. It has become a standard data set to test classifiers on simple image input. Neural network is no doubt a strong model for image classification tasks. There's a [long-term hosted competition](https://www.kaggle.com/c/digit-recognizer) on Kaggle using this data set. We will present the basic usage of `mxnet` to compete in this challenge.

## Data Loading

First, let us download the data from [here](https://www.kaggle.com/c/digit-recognizer/data), and put them under the `data/` folder in your working directory.

Then we can read them in R and convert to matrices.


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
train.x <- train.x/255
test <- test/255
```

In the label part, we see the number of each digit is fairly even:


```r
table(train.y)
```

## Network Configuration

Now we have the data. The next step is to configure the structure of our network.


```r
data <- mx.symbol.Variable("data")
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.Variable"
```

```r
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.FullyConnected"
```

```r
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.Activation"
```

```r
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.FullyConnected"
```

```r
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.Activation"
```

```r
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.FullyConnected"
```

```r
softmax <- mx.symbol.Softmax(fc3, name = "sm")
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.symbol.Softmax"
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
devices <- lapply(1:2, function(i) {
  mx.cpu(i)
})
```

```
## Error in FUN(1:2[[1L]], ...): could not find function "mx.cpu"
```

Here we assign two threads of our CPU to `mxnet`. After all these preparation, you can run the following command to train the neural network!


```r
set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Error in eval(expr, envir, enclos): could not find function "mx.model.FeedForward.create"
```

## Prediction and Submission

To make prediction, we can simply write


```r
preds <- predict(model, test)
```

```
## Error in predict(model, test): object 'model' not found
```

```r
dim(preds)
```

```
## Error in eval(expr, envir, enclos): object 'preds' not found
```

It is a matrix with 28000 rows and 10 cols, containing the desired classification probabilities from the output layer. To extract the maximum label for each row, we can use the `max.col` in R:


```r
pred.label <- max.col(preds) - 1
```

```
## Error in as.matrix(m): object 'preds' not found
```

```r
table(pred.label)
```

```
## Error in table(pred.label): object 'pred.label' not found
```

With a little extra effort in the csv format, we can have our submission to the competition!


```r
submission <- data.frame(ImageId=1:nrow(test), Label=pred.label)
```

```
## Error in nrow(test): object 'test' not found
```

```r
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)
```

```
## Error in is.data.frame(x): object 'submission' not found
```










