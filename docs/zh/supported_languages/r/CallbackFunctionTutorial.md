# MXNet R教程之回调函数

本文将概述我们如何在模型训练的时候使用或者自定义一些回调函数。本教程使用 Rmarkdown 编写。

- 你可以直接看我们在主站上的教程： [MXNet R Document](http://mxnet.readthedocs.org/en/latest/R-package/CallbackFunctionTutorial.html)

- 你可以在这里找到Rmarkdown的源码： [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/CallbackFunctionTutorial.Rmd)

## 模型训练示例

让我们一起小试牛刀吧。你可以用下面的代码开始训练一个模型：

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

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=16.063282524034
## [1] Validation-rmse=10.1766446093622
## [2] Train-rmse=12.2792375712573
## [2] Validation-rmse=12.4331776190813
## [3] Train-rmse=11.1984634005885
## [3] Validation-rmse=10.3303041888193
## [4] Train-rmse=10.2645236892904
## [4] Validation-rmse=8.42760407903415
## [5] Train-rmse=9.49711005504284
## [5] Validation-rmse=8.44557808483234
## [6] Train-rmse=9.07733734175182
## [6] Validation-rmse=8.33225500266177
## [7] Train-rmse=9.07884450847991
## [7] Validation-rmse=8.38827833418459
## [8] Train-rmse=9.10463850277417
## [8] Validation-rmse=8.37394452365264
## [9] Train-rmse=9.03977049028532
## [9] Validation-rmse=8.25927979725672
## [10] Train-rmse=8.96870685004475
## [10] Validation-rmse=8.19509291481822
```

此外, 我们提供了两个可选参数： `batch.end.callback` 和 `epoch.end.callback`。这两个参数给模型训练提供了很大的自由度。

## 如何使用回调函数

这个包提供了两个回调函数:

- `mx.callback.save.checkpoint` 在每次迭代中将检查点保存到文件。

```r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.save.checkpoint("boston"))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=19.1621424021617
## [1] Validation-rmse=20.721515592165
## Model checkpoint saved to boston-0001.params
## [2] Train-rmse=13.5127391952367
## [2] Validation-rmse=14.1822123675007
## Model checkpoint saved to boston-0002.params
............
```

- `mx.callback.log.train.metric` 用于记录每个时期的训练指标. 您可以使用它作为一个 `batch.end.callback` 或者 `epoch.end.callback`.

```r
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  batch.end.callback = mx.callback.log.train.metric(5))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## Batch [5] Train-rmse=17.6514558545416
## [1] Train-rmse=15.2879610219001
## [1] Validation-rmse=12.3332062820921
## Batch [5] Train-rmse=11.939392828565
## [2] Train-rmse=11.4382242547217
## [2] Validation-rmse=9.91176550103181
............
```

你也可以通过引用类为以后使用保存训练和评估误差。

```r
logger <- mx.metric.logger$new()
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
  epoch.end.callback = mx.callback.log.train.metric(5, logger))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=19.1083228733256
## [1] Validation-rmse=12.7150687428974
## [2] Train-rmse=15.7684378116157
## [2] Validation-rmse=14.8105319420491
............
```

```r
head(logger$train)
```

```
## [1] 19.108323 15.768438 13.531470 11.386050  9.555477  9.351324
```

```r
head(logger$eval)
```

```
## [1] 12.715069 14.810532 15.840361 10.898733  9.349706  9.363087
```

## 如何编写自己的回调函数

你可以从[这里](https://github.com/dmlc/mxnet/blob/master/R-package/R/callback.R)找到两个回调函数的源代码并以此用作模板:

基本上,所有的回调函数遵循以下结构:

```r
mx.callback.fun <- function() {
  function(iteration, nbatch, env) {
  }
}
```

`mx.callback.save.checkpoint` 函数是无状态的。它只是从环境中获得模型并将其保存。

```r
mx.callback.save.checkpoint <- function(prefix, period=1) {
  function(iteration, nbatch, env) {
    if (iteration %% period == 0) {
      mx.model.save(env$model, prefix, iteration)
      cat(sprintf("Model checkpoint saved to %s-%04d.params\n", prefix, iteration))
    }
    return(TRUE)
  }
}
```

`mx.callback.log.train.metric` 就稍微有点复杂。它引入一个引用类并且在训练模型的时候更新。

```r
mx.callback.log.train.metric <- function(period, logger=NULL) {
  function(iteration, nbatch, env) {
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

现在你可能会好奇为什么两个回调函数都有 `return(TRUE)`。

我们可以 `return(FALSE)`?

当然! 你可以通过 `return(FALSE)` 来停止模型的训练，让我们看看下面的例子：

```r
mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env) {
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
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=18.5897984387033
## [1] Validation-rmse=13.5555213820571
## [2] Train-rmse=12.5867564040256
## [2] Validation-rmse=9.76304967080928
```

你可以看到一旦验证指标低于阈值,训练过程就会停止。
