# Custom Loss Function

This tutorial shows how to define and use a customized loss function to guide training of your neural network.

Let’s begin with a small regression example. We can build, train, and evaluate a regression model on the Boston Housing dataset with the following code:

```{.python .input  n=1}
data(BostonHousing, package = "mlbench")
BostonHousing[, sapply(BostonHousing, is.factor)] <-
  as.numeric(as.character(BostonHousing[, sapply(BostonHousing, is.factor)]))
BostonHousing <- data.frame(scale(BostonHousing))

test.ind = seq(1, 506, 5) # every 5th point is held-out for testing
train.x = data.matrix(BostonHousing[-test.ind, -14])
train.y = BostonHousing[-test.ind, 14]
test.x = data.matrix(BostonHousing[test.ind, -14])
test.y = BostonHousing[test.ind, 14]
```

```{.python .input  n=2}
require(mxnet)
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
pred <- predict(model, test.x)
mse.test <- mean((test.y - pred[1,])^2)
sprintf("MSE on test data: %f",mse.test)
```

**mxnet** provides the following built-in loss functions (with corresponding appropriate output transformations):

- ``LinearRegressionOutput``: seeks to estimate conditional expectations via the mean-squared-error loss.

- ``MAERegressionOutput``: seeks to estimate conditional medians via the mean-absolute-error loss. Useful when our training data is contaminated by outliers.

- ``LogisticRegressionOutput``: seeks to estimate conditional class-probabilities via the logistic loss (first applies sigmoid transformation to output and then computes binary cross-entropy loss). Useful when we are predicting binary class labels rather than numerical target values. 

However, we may wish to use a different loss function in our own application. 
You can provide your own loss function by using ``mx.symbol.MakeLoss`` when constructing the network.

## Using Your Own Loss Function

We still use the same regression task and neural network architecture from the previous example. However, this time we use ``mx.symbol.MakeLoss`` to minimize a custom loss function that is applied to the ``fc2`` output from our neural network.  Below, we provide an example showing how to use the [pseudo-Huber loss function](https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function), a popular choice for robust regression.

First, we construct the pseudo-Huber loss function below. Note that all operations in our loss function must be defined in terms of **mxnet** ``Symbol`` objects rather than R arrays, in order to allow for subsequent automatic differentiation of the loss during network training.  The **mnxnet** package contains a variety of ``Symbol`` operations you can combine to form pretty much any loss function.  The loss should take in model predictions and the corresponding ground truth labels (as well as other auxiliary parameters, such as the ``delta`` value used in the Huber loss).

```{.python .input  n=4}
pseudoHuberLoss <- function(pred, label, delta=1) {
    diff <- mx.symbol.Reshape(fc2, shape = 0) - label
    (mx.symbol.sqrt(1 + mx.symbol.square(diff/delta)) - 1) * delta^2
}
```

Define our neural network archictecture which makes use of this custom loss function:

```{.python .input  n=5}
data <- mx.symbol.Variable("data")
label <- mx.symbol.Variable("label")
fc1 <- mx.symbol.FullyConnected(data, num_hidden = 14, name = "fc1")
tanh1 <- mx.symbol.Activation(fc1, act_type = "tanh", name = "tanh1")
fc2 <- mx.symbol.FullyConnected(tanh1, num_hidden = 1, name = "fc2")
lro2 <- mx.symbol.MakeLoss(pseudoHuberLoss(fc2,label), name="psuedohuber")
```

Now we can train the network just as usual:

```{.python .input  n=6}
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

Finally, we can evaluate the pseudo-Huber loss of our trained network on the test data.

**Caution:** the output of ``mx.symbol.MakeLoss`` is the gradient of the loss with respect to the input data. 
Thus, we cannot simply call ``predict`` on ``model2``.
Instead, here's how to get predictions from this trained model:

```{.python .input  n=8}
# pred.wrong <- predict(model2, test.x) # This would produce INVALID predictions.
internals = internals(model2$symbol)
fc_symbol = internals[[match("fc2_output", outputs(internals))]]

model.huber <- list(symbol = fc_symbol,
               arg.params = model2$arg.params,
               aux.params = model2$aux.params)
class(model.huber) <- "MXFeedForwardModel"

pred.huber <- predict(model.huber, test.x)
losses.test <- apply(matrix(c(pred.huber[1,],test.y),nrow=2,byrow=T), MARGIN=2, 
                     function(x,delta=1) (sqrt(1 + ((x[1]-x[2])/delta)^2) - 1) * delta^2
                    ) # since we don't need gradients, losses are computed via R arrays rather than Symbol objects 
sprintf("Pseudo-Huber Loss on test data: %f", mean(losses.test))
```
