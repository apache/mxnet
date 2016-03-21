MXNet R Tutorial on Callback Function
=====================================

This vignette gives users a guideline for using and writing callback functions, which can very useful in model training.

This tutorial is written in Rmarkdown.

-   You can directly view the hosted version of the tutorial from [MXNet R Document](http://mxnet.readthedocs.org/en/latest/R-package/CallbackFunctionTutorial.html)

-   You can find the Rmarkdown source from [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/CallbackFunctionTutorial.Rmd)

Model training example
----------------------

Let's begin from a small example. We can build and train a model using the following code:

``` r
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

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..

## Start training with 1 devices
## [1] Train-rmse=16.0632823518301
## [1] Validation-rmse=10.1766422920373
## [2] Train-rmse=12.279237506027
## [2] Validation-rmse=12.4331790778543
## [3] Train-rmse=11.1984636779466
## [3] Validation-rmse=10.3303056488224
## [4] Train-rmse=10.2645248908194
## [4] Validation-rmse=8.4276043147402
## [5] Train-rmse=9.49711028435789
## [5] Validation-rmse=8.44557796721793
## [6] Train-rmse=9.07733782360216
## [6] Validation-rmse=8.33225491992671
## [7] Train-rmse=9.07884473775881
## [7] Validation-rmse=8.38827850773013
## [8] Train-rmse=9.10463854132266
## [8] Validation-rmse=8.37394439717837
## [9] Train-rmse=9.0397702815595
## [9] Validation-rmse=8.25927956738835
## [10] Train-rmse=8.96870645997285
## [10] Validation-rmse=8.1950925862604
```

Besides, we provide two optional parameters, `batch.end.callback` and `epoch.end.callback`, which can provide great flexibility in model training.

How to use callback functions
-----------------------------

Two callback functions are provided in this package:

-   `mx.callback.save.checkpoint` is used to save checkpoint to files each period iteration.

``` r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.save.checkpoint("boston"))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..

## Start training with 1 devices
## [1] Train-rmse=19.1621417275317
## [1] Validation-rmse=20.7215198590237
## Model checkpoint saved to boston-0001.params
## [2] Train-rmse=13.5127400747248
## [2] Validation-rmse=14.1822108465029
## Model checkpoint saved to boston-0002.params
## [3] Train-rmse=10.4242999715995
    ............
```

-   `mx.callback.log.train.metric` is used to log training metric each period. You can use it either as a `batch.end.callback` or a `epoch.end.callback`.

``` r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  batch.end.callback = mx.callback.log.train.metric(5))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..

## Start training with 1 devices
## Batch [5] Train-rmse=17.651456218277
## [1] Train-rmse=15.287961510075
## [1] Validation-rmse=12.3332071982351
## Batch [5] Train-rmse=11.9393931754201
## [2] Train-rmse=11.4382247928496
## [2] Validation-rmse=9.91176567967788
............
```

You can also save the training and evaluation errors for later usage by passing a reference class.

``` r
logger <- mx.metric.logger$new()
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
```

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..

## Start training with 1 devices
## [1] Train-rmse=19.1083223736224
## [1] Validation-rmse=12.7150678057308
## [2] Train-rmse=15.7684369967015
## [2] Validation-rmse=14.8105301816037
............
```

``` r
head(logger$train)
```

```
## [1] 19.108322 15.768437 13.531469 11.386050  9.555476  9.351324
```

``` r
head(logger$eval)
```

```
## [1] 12.715068 14.810530 15.840362 10.898733  9.349705  9.363086
```

How to write your own callback functions
----------------------------------------

You can find the source code for two callback functions from [here](https://github.com/dmlc/mxnet/blob/master/R-package/R/callback.R) and they can be used as your template:

Basically, all callback functions follow the structure below:

``` r
mx.callback.fun <- function() {
  function(iteration, nbatch, env, verbose) {
  }
}
```

The `mx.callback.save.checkpoint` function below is stateless. It just get the model from environment and save it.

``` r
mx.callback.save.checkpoint <- function(prefix, period=1) {
  function(iteration, nbatch, env, verbose) {
    if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      if(verbose) cat(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
  }
}
```

The `mx.callback.log.train.metric` is a little more complex. It will hold a reference class and update it during the training process.

``` r
mx.callback.log.train.metric <- function(period, logger=NULL) {
  function(iteration, nbatch, env, verbose) {
    if (nbatch %% period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0)
        if(verbose) cat(paste0("Batch [", nbatch, "] Train-", result$name, "=", result$value, "\n"))
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

Now you might be curious why both callback functions `return(TRUE)`. Can we `return(FALSE)`?

Yes! You can stop the training early by `return(FALSE)`. See the examples below.

``` r
mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose) {
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

```
## Warning in mx.model.select.layout.train(X, y): Auto detect layout of input matrix, use rowmajor..

## Start training with 1 devices
## [1] Train-rmse=18.5897976897187
## [1] Validation-rmse=13.5555215885869
## [2] Train-rmse=12.5867552109686
## [2] Validation-rmse=9.76305022663742
```

You can see once the validation metric goes below the threshold we set, the training process will stop early.
