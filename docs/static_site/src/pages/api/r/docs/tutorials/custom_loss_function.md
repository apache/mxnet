---
layout: page_api
title: Custom Loss Function
is_tutorial: true
tag: r
permalink: /api/r/docs/tutorials/custom_loss_function
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


Customized loss function
======================================

This tutorial provides guidelines for using customized loss function in network construction.

Model Training Example
----------------------

Let's begin with a small regression example. We can build and train a regression model with the following code:

``` r
data(BostonHousing, package = "mlbench")
BostonHousing[, sapply(BostonHousing, is.factor)] <-
  as.numeric(as.character(BostonHousing[, sapply(BostonHousing, is.factor)]))
BostonHousing <- data.frame(scale(BostonHousing))

test.ind = seq(1, 506, 5)    # 1 pt in 5 used for testing
train.x = data.matrix(BostonHousing[-test.ind,-14])
train.y = BostonHousing[-test.ind, 14]
test.x = data.matrix(BostonHousing[--test.ind,-14])
test.y = BostonHousing[--test.ind, 14]

require(mxnet)
```

    ## Loading required package: mxnet

``` r
data <- mx.symbol.Variable("data")
label <- mx.symbol.Variable("label")
fc1 <- mx.symbol.FullyConnected(data, num_hidden = 14, name = "fc1")
tanh1 <- mx.symbol.Activation(fc1, act_type = "tanh", name = "tanh1")
fc2 <- mx.symbol.FullyConnected(tanh1, num_hidden = 1, name = "fc2")
lro <- mx.symbol.LinearRegressionOutput(fc2, name = "lro")

mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X = train.x, y = train.y,
                                     ctx = mx.cpu(),
                                     num.round = 5,
                                     array.batch.size = 60,
                                     optimizer = "rmsprop",
                                     verbose = TRUE,
                                     array.layout = "rowmajor",
                                     batch.end.callback = NULL,
                                     epoch.end.callback = NULL)
```

    ## Start training with 1 devices

``` r
pred <- predict(model, test.x)
```

    ## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..

``` r
sum((test.y - pred[1,])^2) / length(test.y)
```

    ## [1] 0.2485236

Besides the `LinearRegressionOutput`, we also provide `LogisticRegressionOutput` and `MAERegressionOutput`. However, this might not be enough for real-world models. You can provide your own loss function by using `mx.symbol.MakeLoss` when constructing the network.

How to Use Your Own Loss Function
---------------------------------

We still use our previous example, but this time we use `mx.symbol.MakeLoss` to minimize the `(pred-label)^2`

``` r
data <- mx.symbol.Variable("data")
label <- mx.symbol.Variable("label")
fc1 <- mx.symbol.FullyConnected(data, num_hidden = 14, name = "fc1")
tanh1 <- mx.symbol.Activation(fc1, act_type = "tanh", name = "tanh1")
fc2 <- mx.symbol.FullyConnected(tanh1, num_hidden = 1, name = "fc2")
lro2 <- mx.symbol.MakeLoss(mx.symbol.square(mx.symbol.Reshape(fc2, shape = 0) - label), name="lro2")
```

Then we can train the network just as usual.

``` r
mx.set.seed(0)
model2 <- mx.model.FeedForward.create(lro2, X = train.x, y = train.y,
                                      ctx = mx.cpu(),
                                      num.round = 5,
                                      array.batch.size = 60,
                                      optimizer = "rmsprop",
                                      verbose = TRUE,
                                      array.layout = "rowmajor",
                                      batch.end.callback = NULL,
                                      epoch.end.callback = NULL)
```

    ## Start training with 1 devices

We should get very similar results because we are actually minimizing the same loss function. However, the result is quite different.

``` r
pred2 <- predict(model2, test.x)
```

    ## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..

``` r
sum((test.y - pred2)^2) / length(test.y)
```

    ## [1] 1.234584

This is because output of `mx.symbol.MakeLoss` is the gradient of loss with respect to the input data. We can get the real prediction as below.

``` r
internals = internals(model2$symbol)
fc_symbol = internals[[match("fc2_output", outputs(internals))]]

model3 <- list(symbol = fc_symbol,
               arg.params = model2$arg.params,
               aux.params = model2$aux.params)

class(model3) <- "MXFeedForwardModel"

pred3 <- predict(model3, test.x)
```

    ## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..

``` r
sum((test.y - pred3[1,])^2) / length(test.y)
```

    ## [1] 0.248294

We have provided many operations on the symbols. An example of `|pred-label|` can be found below.

``` r
lro_abs <- mx.symbol.MakeLoss(mx.symbol.abs(mx.symbol.Reshape(fc2, shape = 0) - label))
mx.set.seed(0)
model4 <- mx.model.FeedForward.create(lro_abs, X = train.x, y = train.y,
                                      ctx = mx.cpu(),
                                      num.round = 20,
                                      array.batch.size = 60,
                                      optimizer = "sgd",
                                      learning.rate = 0.001,
                                      verbose = TRUE,
                                      array.layout = "rowmajor",
                                      batch.end.callback = NULL,
                                      epoch.end.callback = NULL)
```

    ## Start training with 1 devices

``` r
internals = internals(model4$symbol)
fc_symbol = internals[[match("fc2_output", outputs(internals))]]

model5 <- list(symbol = fc_symbol,
               arg.params = model4$arg.params,
               aux.params = model4$aux.params)

class(model5) <- "MXFeedForwardModel"

pred5 <- predict(model5, test.x)
```

    ## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..

``` r
sum(abs(test.y - pred5[1,])) / length(test.y)
```

    ## [1] 0.7056902

``` r
lro_mae <- mx.symbol.MAERegressionOutput(fc2, name = "lro")
mx.set.seed(0)
model6 <- mx.model.FeedForward.create(lro_mae, X = train.x, y = train.y,
                                      ctx = mx.cpu(),
                                      num.round = 20,
                                      array.batch.size = 60,
                                      optimizer = "sgd",
                                      learning.rate = 0.001,
                                      verbose = TRUE,
                                      array.layout = "rowmajor",
                                      batch.end.callback = NULL,
                                      epoch.end.callback = NULL)
```

    ## Start training with 1 devices

``` r
pred6 <- predict(model6, test.x)
```

    ## Warning in mx.model.select.layout.predict(X, model): Auto detect layout of input matrix, use rowmajor..

``` r
sum(abs(test.y - pred6[1,])) / length(test.y)
```

    ## [1] 0.7056902


## Next Steps
* [Neural Networks with MXNet in Five Minutes](/api/r/docs/tutorials/five_minutes_neural_network)
* [Classify Real-World Images with a PreTrained Model](/api/r/docs/tutorials/classify_real_image_with_pretrained_model)
* [Handwritten Digits Classification Competition](/api/r/docs/tutorials/mnist_competition)
* [Character Language Model Using RNN](/api/r/docs/tutorials/char_rnn_model)
