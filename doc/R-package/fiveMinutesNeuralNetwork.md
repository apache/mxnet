Neural Network with MXNet in Five Minutes
=============================================

This is the first tutorial for new users of the R package `mxnet`. You will learn to construct a neural network to do regression in 5 minutes.

We will show you how to do classification and regression tasks respectively. The data we use comes from the package `mlbench`.

Preface
-------
This tutorial is written in Rmarkdown.
- You can directly view the hosted version of the tutorial from [MXNet R Document](http://mxnet.readthedocs.org/en/latest/R-package/fiveMinutesNeuralNetwork.html)
- You can find the download the Rmarkdown source from [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/fiveMinutesNeuralNetwork.Rmd)

## Classification

First of all, let us load in the data and preprocess it:


```r
require(mlbench)
```

```
## Loading required package: mlbench
```

```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

```r
data(Sonar, package="mlbench")

Sonar[,61] = as.numeric(Sonar[,61])-1
train.ind = c(1:50, 100:150)
train.x = data.matrix(Sonar[train.ind, 1:60])
train.y = Sonar[train.ind, 61]
test.x = data.matrix(Sonar[-train.ind, 1:60])
test.y = Sonar[-train.ind, 61]
```

The next step is to define the structure of the neural network.


```r
# Define the input data
data <- mx.symbol.Variable("data")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=20)

# An activation function
# fc1: input source
# act_type: type for the activation function
act1 <- mx.symbol.Activation(fc1, act_type="tanh")
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=2)

# SoftmaxOutput means multi-class probability prediction.
softmax <- mx.symbol.SoftmaxOutput(fc2)
```

According to the comments in the code, you can see the meaning of each function and its arguments. They can be easily modified according to your need.

Before we start to train the model, we can specify where to run our program:


```r
device.cpu = mx.cpu()
```

Here we choose to run it on CPU.

After the network configuration, we can start the training process:


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=device.cpu, num.round=20, array.batch.size=15,
                                     learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-accuracy=0.5
## [2] Train-accuracy=0.514285714285714
## [3] Train-accuracy=0.514285714285714
## [4] Train-accuracy=0.514285714285714
## [5] Train-accuracy=0.514285714285714
## [6] Train-accuracy=0.609523809523809
## [7] Train-accuracy=0.676190476190476
## [8] Train-accuracy=0.695238095238095
## [9] Train-accuracy=0.723809523809524
## [10] Train-accuracy=0.780952380952381
## [11] Train-accuracy=0.8
## [12] Train-accuracy=0.761904761904762
## [13] Train-accuracy=0.742857142857143
## [14] Train-accuracy=0.761904761904762
## [15] Train-accuracy=0.847619047619047
## [16] Train-accuracy=0.857142857142857
## [17] Train-accuracy=0.857142857142857
## [18] Train-accuracy=0.828571428571429
## [19] Train-accuracy=0.838095238095238
## [20] Train-accuracy=0.857142857142857
```

Note that `mx.set.seed` is the correct function to control the random process in `mxnet`. You can see the accuracy in each round during training. It is also easy to make prediction and evaluate.


```r
preds = predict(model, test.x)
```

```
## Auto detect layout of input matrix, use rowmajor..
```

```r
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
```

```
##           test.y
## pred.label  0  1
##          0 24 14
##          1 36 33
```

Note for multi-class prediction, mxnet outputs nclass x nexamples, each each row corresponding to probability of that class.

## Regression

Again, let us preprocess the data first.


```r
data(BostonHousing, package="mlbench")

train.ind = seq(1, 506, 3)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]
```

We can configure another network as what we have done above. The main difference is in the output activation:


```r
# Define the input data
data <- mx.symbol.Variable("data")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc1)
```

What we changed is mainly the last function, this enables the new network to optimize for squared loss. We can now train on this simple data set.


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=device.cpu, num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=16.0632825223292
## [2] Train-rmse=12.2792375527391
## [3] Train-rmse=11.1984634148088
## [4] Train-rmse=10.2645236892904
## [5] Train-rmse=9.49711003902655
## [6] Train-rmse=9.07733735504537
## [7] Train-rmse=9.07884447337348
## [8] Train-rmse=9.10463849901276
## [9] Train-rmse=9.03977048081203
## [10] Train-rmse=8.96870681959898
## [11] Train-rmse=8.93113268945833
## [12] Train-rmse=8.89937250031474
## [13] Train-rmse=8.87182124831547
## [14] Train-rmse=8.84476111567396
## [15] Train-rmse=8.81464687265692
## [16] Train-rmse=8.78672579209995
## [17] Train-rmse=8.76265895056591
## [18] Train-rmse=8.73946101364483
## [19] Train-rmse=8.7165194446551
## [20] Train-rmse=8.69457580107095
## [21] Train-rmse=8.67354933875898
## [22] Train-rmse=8.65328764760528
## [23] Train-rmse=8.63378016812285
## [24] Train-rmse=8.61488175856399
## [25] Train-rmse=8.59651041652324
## [26] Train-rmse=8.57868122898644
## [27] Train-rmse=8.56135865255391
## [28] Train-rmse=8.54448212525355
## [29] Train-rmse=8.52802110389574
## [30] Train-rmse=8.51195043845808
## [31] Train-rmse=8.49624250344235
## [32] Train-rmse=8.48087452797975
## [33] Train-rmse=8.46582681750595
## [34] Train-rmse=8.45107900842757
## [35] Train-rmse=8.43661347614512
## [36] Train-rmse=8.42241598595198
## [37] Train-rmse=8.40847223745159
## [38] Train-rmse=8.39476934189048
## [39] Train-rmse=8.38129658669852
## [40] Train-rmse=8.36804245552321
## [41] Train-rmse=8.35499814305568
## [42] Train-rmse=8.34215500774088
## [43] Train-rmse=8.3295045517182
## [44] Train-rmse=8.31703965839842
## [45] Train-rmse=8.30475372106883
## [46] Train-rmse=8.2926402584762
## [47] Train-rmse=8.2806936364631
## [48] Train-rmse=8.26890890119326
## [49] Train-rmse=8.25728092677924
## [50] Train-rmse=8.24580513680541
```

It is also easy to make prediction and evaluate


```r
preds = predict(model, test.x)
```

```
## Auto detect layout of input matrix, use rowmajor..
```

```r
sqrt(mean((preds-test.y)^2))
```

```
## [1] 7.800502
```

Currently we have two pre-defined metrics "accuracy" and "rmse". One might wonder how to customize the evaluation metric. `mxnet` provides the interface for users to define their own metric of interests:


```r
demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  res <- mean(abs(label-pred))
  return(res)
})
```

This is an example for mean absolute error. We can simply plug it in the training function:


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=device.cpu, num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=demo.metric.mae,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-mae=13.1889538090676
## [2] Train-mae=9.81431958410475
## [3] Train-mae=9.21576420929697
## [4] Train-mae=8.38071537613869
## [5] Train-mae=7.45462434962392
## [6] Train-mae=6.93423304392232
## [7] Train-mae=6.91432355824444
## [8] Train-mae=7.02742730538464
## [9] Train-mae=7.00618193757513
## [10] Train-mae=6.92541587183045
## [11] Train-mae=6.87530209053722
## [12] Train-mae=6.847573687012
## [13] Train-mae=6.82966502538572
## [14] Train-mae=6.81151769575146
## [15] Train-mae=6.78394197610517
## [16] Train-mae=6.75914737499422
## [17] Train-mae=6.74180429437094
## [18] Train-mae=6.72585320373376
## [19] Train-mae=6.70932160268227
## [20] Train-mae=6.69288677523534
## [21] Train-mae=6.67695207827621
## [22] Train-mae=6.66184799075127
## [23] Train-mae=6.64754500372542
## [24] Train-mae=6.63358518299129
## [25] Train-mae=6.62027624067333
## [26] Train-mae=6.60738218476375
## [27] Train-mae=6.59505565381712
## [28] Train-mae=6.58346203284131
## [29] Train-mae=6.57285475134849
## [30] Train-mae=6.56259016940991
## [31] Train-mae=6.55277890273266
## [32] Train-mae=6.54353418886248
## [33] Train-mae=6.53441721167829
## [34] Train-mae=6.52557678090202
## [35] Train-mae=6.51697915651732
## [36] Train-mae=6.50847910601232
## [37] Train-mae=6.50014858543873
## [38] Train-mae=6.49207666102383
## [39] Train-mae=6.48412067078882
## [40] Train-mae=6.47650481263797
## [41] Train-mae=6.46893873314063
## [42] Train-mae=6.46142139865292
## [43] Train-mae=6.45395037829876
## [44] Train-mae=6.44652904189295
## [45] Train-mae=6.43916221575605
## [46] Train-mae=6.43183771024148
## [47] Train-mae=6.42455528063907
## [48] Train-mae=6.41731397675143
## [49] Train-mae=6.41011299813787
## [50] Train-mae=6.40312501904037
```

Congratulations! Now you have learnt the basic for using `mxnet`.


