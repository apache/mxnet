# Develop a Neural Network with MXNet in Five Minutes

This 5 minute tutorial is designed for new users of the mxnet package for R. It shows how to construct a neural network for classification and regression tasks. The data we use is in the **mlbench** R package. 

First, we need to load the required R packages:

```{.python .input .R  n=1}
if (!require(mlbench)) {
    install.packages('mlbench')
}
require(mxnet)
```

## Classification

We load some data for a simple classification problem with two classes:

```{.python .input .R  n=2}
data(Sonar, package="mlbench")
Sonar[,61] = as.numeric(Sonar[,61])-1  # the target labels
set.seed(0)
train.ind = sample(1:nrow(Sonar), size=ceiling(0.7*nrow(Sonar)))
train.x = data.matrix(Sonar[train.ind, 1:60])
train.y = Sonar[train.ind, 61]
test.x = data.matrix(Sonar[-train.ind, 1:60])
test.y = Sonar[-train.ind, 61]
table(train.y)  # distribution of classes in training data
```

We are going to use a multi-layer perceptron as our classifier (i.e. what is commonly thought of as the standard feedforward neural network). 
In the **mxnet** package, we have a function called ``mx.mlp`` for building a general multi-layer neural network to do classification or regression.

``mx.mlp`` requires the following parameters:

- Training data and label

- Number of hidden nodes in each hidden layer

- Number of nodes in the output layer

- Type of the activation

- Type of the output loss

- The device to train (GPU or CPU)

- Other extra parameters for ``mx.model.FeedForward.create``

The following code shows an example usage of ``mx.mlp``:

```{.python .input .R  n=4}
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(200,100), out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.1, momentum=0.9,
                eval.metric=mx.metric.accuracy)
```

Specifying ``mx.set.seed`` controls all randomness within **mxnet** (as opposed to ``set.seed`` within R), and ensures our results are reproducible.

Above, we have created a three-layer neural network with 200 hidden units in the first layer, 100 hidden units in the second layer, and 2 units in the output layer (one for each possible class in our classification task).
Our network has been trained for 20 epochs (``num.round``) over the provided data via the SGD-with-momentum optimization algorithm using mini-batches of size 15 (``array.batch.size``), a learning rate of 0.1, and a momentum factor of 0.9.
You can see the training accuracy in each epoch as training progresses. 

To summarize what operations are performed by our network and its overall architecture, view the computation graph:

```{.python .input .R  n=5}
graph.viz(model$symbol)
```

It is easy to use our trained model to make predictions regarding the probability of each class for our test examples:

```{.python .input  n=9}
preds = predict(model, test.x)
```

Note for that for multi-class predictions, **mxnet** outputs nclass x nexamples, with each row corresponding to the probability of the class.
We can easily evaluate the quality of these predictions:

```{.python .input  n=10}
p = max.col(t(preds))-1
    table(p, test.y)
```

## Regression

Next, we load data for a simple regression task:

```{.python .input  n=48}
data(BostonHousing, package="mlbench")
set.seed(0)
train.ind = sample(1:nrow(BostonHousing), size=ceiling(0.7*nrow(BostonHousing)))
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]
summary(train.y)  # distribution of target values in training data
```

We can simply invoke ``mx.mlp`` again to train a feedforward neural network for regression. We just need to  appropriately change ``out_activation`` and ``eval.metric`` to tell our model to employ the root mean-square error objective, which is more appropriate for predicting continuous values (we also set ``out_node`` = 1 to reflect the fact that model should now only output a single value as its prediction):

```{.python .input  n=69}
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(20,10), out_node=1, out_activation="rmse",
                num.round=20, array.batch.size=15, learning.rate=1e-4, momentum=0.9,
                eval.metric=mx.metric.rmse)
```

However, this time we are also going to introduce a flexible way to configure neural networks in **mxnet** via the ``Symbol`` system.  The ``Symbol`` system takes care of the links among nodes, activation, dropout ratio, etc. 
We can configure a multi-layer neural network as follows:

```{.python .input  n=71}
# Define the input data
    data <- mx.symbol.Variable("data")
    # A fully connected hidden layer
    # data: input source
    # num_hidden: number of neurons in this hidden layer
    fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

    # Use linear regression for the output layer
    lro <- mx.symbol.LinearRegressionOutput(fc1)
```

What matters for a regression task is mainly the last function. It enables the new network to optimize for squared loss. Now let’s train on this simple data set. In this configuration, we omitted all hidden layers so that the input layer is directly connected to the output layer (i.e. we are simply using a linear model).

Using ``mx.model.FeedForward.create``, we can instantiate the parameters of the network structure defined above and learn good values for them based on our training dataset:

```{.python .input  n=141}
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=20, array.batch.size=15,
                                     learning.rate=1e-6, momentum=0.9,  eval.metric=mx.metric.rmse)
```

It is again easy to use this learned model to make predictions on new data points and evaluate the quality of these predictions:

```{.python .input  n=147}
preds = predict(model, test.x)
summary(as.vector(preds))
sprintf("test RMSE = %f", sqrt(mean((preds-test.y)^2)))
```

Currently, **mxnet** has four predefined evaluation metrics: "accuracy", "rmse", "mae", and "rmsle". 
You can also define your own metrics via the provided interface:

```{.python .input  n=148}
demo.metric.mae <- mx.metric.custom("demo_mae", function(label, pred) {
      pred <- mx.nd.reshape(pred, shape = 0)
      res <- mx.nd.mean(mx.nd.abs(label-pred))
      return(as.array(res))
    })
```

As an example, we have defined the mean absolute error metric ourselves above. Now, we can simply plug it into the training function:

```{.python .input  n=149}
mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=20, array.batch.size=15,
                                     learning.rate=1e-6, momentum=0.9, eval.metric=demo.metric.mae)
```

```{.python .input  n=150}
preds = predict(model, test.x)
sprintf("test MAE = %f", mean(abs(preds-test.y)))
```

Congratulations! You’ve learned the basics for using MXNet to train neural networks in R. To learn how to use MXNet’s more advanced features, see the other tutorials provided on this website.
