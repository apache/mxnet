Customized loss function
======================================

This tutorial provides guidelines for using customized loss function in network construction.


Model Training Example
----------

Let's begin with a small regression example. We can build and train a regression model with the following code:


 ```r
    library(mxnet)
    data(BostonHousing, package="mlbench")
    train.ind = seq(1, 506, 3)
    train.x = data.matrix(BostonHousing[train.ind, -14])
    train.y = BostonHousing[train.ind, 14]
    test.x = data.matrix(BostonHousing[-train.ind, -14])
    test.y = BostonHousing[-train.ind, 14]
    data <- mx.symbol.Variable("data")
    fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
    lro <- mx.symbol.LinearRegressionOutput(fc1)
    mx.set.seed(0)
    model <- mx.model.FeedForward.create(
      lro, X=train.x, y=train.y,
      eval.data=list(data=test.x, label=test.y),
      ctx=mx.cpu(), num.round=10, array.batch.size=20,
      learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
 ```

Besides the `LinearRegressionOutput`, we also provide `LogisticRegressionOutput` and `MAERegressionOutput`.
However, this might not be enough for real-world models. You can provide your own loss function
by using `mx.symbol.MakeLoss` when constructing the network.


How to Use Your Own Loss Function
---------

We still use our previous example.

 ```r
    library(mxnet)
    data <- mx.symbol.Variable("data")
    fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
    lro <- mx.symbol.MakeLoss(mx.symbol.square(mx.symbol.Reshape(fc1, shape = 0) - label))
 ```

In the last line of network definition, we do not use the predefined loss function. We define the loss
by ourselves, which is `(pred-label)^2`.

We have provided many operations on the symbols, so you can also define `|pred-label|` using the line below.

 ```r
    lro <- mx.symbol.MakeLoss(mx.symbol.abs(mx.symbol.Reshape(fc1, shape = 0) - label))
 ```

## Next Steps
* [Neural Networks with MXNet in Five Minutes](http://mxnet.io/tutorials/r/fiveMinutesNeuralNetwork.html)
* [Classify Real-World Images with a PreTrained Model](http://mxnet.io/tutorials/r/classifyRealImageWithPretrainedModel.html)
* [Handwritten Digits Classification Competition](http://mxnet.io/tutorials/r/mnistCompetition.html)
* [Character Language Model Using RNN](http://mxnet.io/tutorials/r/charRnnModel.html)
