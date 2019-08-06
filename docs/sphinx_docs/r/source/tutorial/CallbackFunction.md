# Callback Functions

This tutorial provides guidelines for using and writing callback functions, which can very useful in model training.

Let’s begin with a small example. We can build and train a model with the following code:

```{.python .input .R}
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

``mx.model.FeedForward.create`` can also take in two optional parameters: ``batch.end.callback`` and ``epoch.end.callback``, which provide a flexible way to specify what/when we wish to log as model-training progresses. We demonstrate the usage of these options below.


## How to Use Callback Functions

The **mxnet** package provides two built-in callback functions: ``mx.callback.save.checkpoint`` and ``mx.callback.log.train.metric``

``mx.callback.save.checkpoint`` saves a checkpoint to files during each period iteration.

```{.python .input .R}
model <- mx.model.FeedForward.create(
   lro, X=train.x, y=train.y,
   eval.data=list(data=test.x, label=test.y),
   ctx=mx.cpu(), num.round=10, array.batch.size=20,
   learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
   epoch.end.callback = mx.callback.save.checkpoint("boston"))
```

``mx.callback.log.train.metric`` logs a training metric each period. You can use it either as a ``batch.end.callback`` or an ``epoch.end.callback``

```{.python .input .R}
model <- mx.model.FeedForward.create(
           lro, X=train.x, y=train.y,
           eval.data=list(data=test.x, label=test.y),
           ctx=mx.cpu(), num.round=10, array.batch.size=20,
           learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
           batch.end.callback = mx.callback.log.train.metric(5))
```

You also can save the training and evaluation errors for later use by passing a reference class:

```{.python .input .R}
logger <- mx.metric.logger$new()
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
```

```{.python .input .R}
head(logger$train)
```

```{.python .input .R}
head(logger$eval)
```

## How to Write Your Own Callback Functions

You can find the source code for the two callback functions on [GitHub](https://github.com/dmlc/mxnet/blob/master/R-package/R/callback.R) and use it as a template:

Basically, all callback functions follow the following structure:

```{.python .input .R}
mx.callback.fun <- function() {
  function(iteration, nbatch, env, verbose = FALSE) {
  }
}
```

The following ``mx.callback.save.checkpoint`` function is stateless. It gets the model from the environment and saves it:

```{.python .input .R}
 mx.callback.save.checkpoint <- function(prefix, period=1) {
      function(iteration, nbatch, env, verbose = FALSE) {
      if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      cat(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
      }
    }
```

``mx.callback.log.train.metric`` is a little more complex. It holds a reference class and updates it during the training process:

```{.python .input .R}
mx.callback.log.train.metric <- function(period, logger=NULL) {
  function(iteration, nbatch, env, verbose = FALSE) {
if (nbatch %% period == 0 && !is.null(env$metric)) {
  result <- env$metric$get(env$train.metric)
  if (nbatch != 0)
    cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
  if (!is.null(logger)) {
    if (class(logger) != "mx.metric.logger") {
      stop("Invalid mx.metric.logger.")
    }
    logger$train <- c(logger$train, result$value)
    if (!is.null(env$eval.metric)) {
      result <- env$metric$get(env$eval.metric)
      if (nbatch != 0)
        cat(paste0("Batch [", nbatch, "] Validation-", result$name, "=", result$value, "\n"))
      logger$eval <- c(logger$eval, result$value)
    }
  }
}
return(TRUE)
  }
}
```

Now you might be curious why both callback functions **return(TRUE)**.

Can we **return(FALSE)**?

Yes! You can stop the training early with **return(FALSE)**.
See the following examples:

```{.python .input .R}
mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose = FALSE) {
if (!is.null(env$metric)) {
  if (!is.null(eval.metric)) {
    result <- env$metric$get(env$eval.metric)
    if (result$value < eval.metric) {
      return(FALSE)
    }
  }
}
return(TRUE)
  }
}
model <- mx.model.FeedForward.create(
      lro, X=train.x, y=train.y,
      eval.data=list(data=test.x, label=test.y),
      ctx=mx.cpu(), num.round=10, array.batch.size=20,
      learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
      epoch.end.callback = mx.callback.early.stop(10))
```

Here when the validation metric dips below the threshold we set, the training process stops.
