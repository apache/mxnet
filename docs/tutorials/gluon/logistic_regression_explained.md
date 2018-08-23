
# Logistic regression using Gluon API explained

Logistic Regression is one of the first models newcomers to Deep Learning are implementing. The focus of this tutorial is to show how to do logistic regression using Gluon API.

Before anything else, let's import required packages for this tutorial.


```python
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Trainer
from mxnet.gluon.data import DataLoader, ArrayDataset

mx.random.seed(12345)  # Added for reproducibility
```

In this tutorial we will use fake dataset, which contains 10 features drawn from a normal distribution with mean equals to 0 and standard deviation equals to 1, and a class label, which can be either 0 or 1. The size of the dataset is an arbitrary value. The function below helps us to generate a dataset. Class label `y` is generated via a non-random logic, so the network would have a pattern to look for. Boundary of 3 is selected to make sure that number of positive examples smaller than negative, but not too small


```python
def get_random_data(size, ctx):
    x = nd.normal(0, 1, shape=(size, 10), ctx=ctx)
    y = x.sum(axis=1) > 3
    return x, y
```

Also, let's define a set of hyperparameters, that we are going to use later. Since our model is simple and dataset is small, we are going to use CPU for calculations. Feel free to change it to GPU for a more advanced scenario.


```python
ctx = mx.cpu()
train_data_size = 1000
val_data_size = 100
batch_size = 10
```

## Working with data

To work with data, Apache MXNet provides [Dataset](https://mxnet.incubator.apache.org/api/python/gluon/data.html#mxnet.gluon.data.Dataset) and [DataLoader](https://mxnet.incubator.apache.org/api/python/gluon/data.html#mxnet.gluon.data.DataLoader) classes. The former is used to provide an indexed access to the data, the latter is used to shuffle and batchify the data. To learn more about working with data in Gluon, please refer to [Gluon Datasets and Dataloaders](https://mxnet.incubator.apache.org/tutorials/gluon/datasets.html) tutorial.

Below we define training and validation datasets, which we are going to use in the tutorial.


```python
train_x, train_ground_truth_class = get_random_data(train_data_size, ctx)
train_dataset = ArrayDataset(train_x, train_ground_truth_class)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_x, val_ground_truth_class = get_random_data(val_data_size, ctx)
val_dataset = ArrayDataset(val_x, val_ground_truth_class)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
```

## Defining and training the model

The only requirement for the logistic regression is that the last layer of the network must be a single neuron. Apache MXNet allows us to do so by using [Dense](https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Dense) layer and specifying the number of units to 1. The rest of the network can be arbitrarily complex.

Below, we define a model which has an input layer of 10 neurons, a couple of inner layers of 10 neurons each, and output layer of 1 neuron. We stack the layers using [HybridSequential](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.HybridSequential) block and initialize parameters of the network using [Xavier](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.initializer.Xavier) initialization.


```python
net = nn.HybridSequential()

with net.name_scope():
    net.add(nn.Dense(units=10, activation='relu'))  # input layer
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 1
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 2
    net.add(nn.Dense(units=1))   # output layer: notice, it must have only 1 neuron

net.initialize(mx.init.Xavier())
```

After defining the model, we need to define a few more things: our loss, our trainer and our metric.

Loss function is used to calculate how the output of the network differs from the ground truth. Because classes  of the logistic regression are either 0 or 1, we are using [SigmoidBinaryCrossEntropyLoss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss). Notice that we do not specify `from_sigmoid` attribute in the code, which means that the output of the neuron doesn't need to go through sigmoid, but at inference we'd have to pass it through sigmoid. You can learn more about cross entropy on [wikipedia](https://en.wikipedia.org/wiki/Cross_entropy).

Trainer object allows to specify the method of training to be used. For our tutorial we use [Stochastic Gradient Descent (SGD)](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#mxnet.optimizer.SGD). For more information on SGD refer to [the following tutorial](https://gluon.mxnet.io/chapter06_optimization/gd-sgd-scratch.html). We also need to parametrize it with learning rate value, which defines the weight updates, and weight decay, which is used for regularization.

Metric helps us to estimate how good our model is in terms of a problem we are trying to solve. Where loss function has more importance for the training process, a metric is usually the thing we are trying to improve and reach maximum value. We also can use more than one metric, to measure various aspects of our model. In our example, we are using [Accuracy](https://mxnet.incubator.apache.org/api/python/metric/metric.html?highlight=metric.acc#mxnet.metric.Accuracy) and [F1 score](http://mxnet.incubator.apache.org/api/python/metric/metric.html?highlight=metric.f1#mxnet.metric.F1) as measurements of success of our model.

Below we define these objects.


```python
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = Trainer(params=net.collect_params(), optimizer='sgd',
                  optimizer_params={'learning_rate': 0.1})
accuracy = mx.metric.Accuracy()
f1 = mx.metric.F1()
```

The next step is to define the training function in which we iterate over all batches of training data, execute the forward pass on each batch and calculate training loss. On line 19, we sum losses of every batch per epoch into a single variable, because we calculate loss per single batch, but want to display it per epoch.


```python
def train_model():
    cumulative_train_loss = 0

    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            # Do forward pass on a batch of training data
            output = net(data)

            # Calculate loss for the training data batch
            loss_result = loss(output, label)

        # Calculate gradients
        loss_result.backward()

        # Update parameters of the network
        trainer.step(batch_size)

        # sum losses of every batch
        cumulative_train_loss += nd.sum(loss_result).asscalar()

    return cumulative_train_loss
```

## Validating the model

Our validation function is very similar to the training one. The main difference is that we want to calculate accuracy of the model. We use [Accuracy metric](https://mxnet.incubator.apache.org/api/python/metric/metric.html?highlight=metric.acc#mxnet.metric.Accuracy) to do so. 

`Accuracy` metric requires 2 arguments: 1) a vector of ground-truth classes and 2) A vector or matrix of predictions. When predictions are of the same shape as the vector of ground-truth classes, `Accuracy` class assumes that prediction vector contains predicted classes. So, it converts the vector to `Int32` and compare each item of ground-truth classes to prediction vector.

Because of the behaviour above, you will get an unexpected result if you just apply [Sigmoid](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.sigmoid) function to the network result and pass it to `Accuracy` metric. As mentioned before, we need to apply `Sigmoid` function to the output of the neuron to get a probability of belonging to the class 1. But `Sigmoid` function produces output in range [0; 1], and all numbers in that range are going to be casted to 0, even if it is as high as 0.99. To avoid this we write a custom bit of code on line 12, that:

1. Calculates sigmoid using `Sigmoid` function

2. Subtracts a threshold from the original sigmoid output. Usually, the threshold is equal to 0.5, but it can be higher, if you want to increase certainty of an item to belong to class 1.

3. Uses [mx.nd.ceil](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.ceil) function, which converts all negative values to 0 and all positive values to 1

After these transformations we can pass the result to `Accuracy.update()` method and expect it to behave in a proper way.

For `F1` metric to work, instead of one number per class, we must pass probabilities of belonging to both classes. Because of that, on lines 21-22 we:

1. Reshape predictions to a single vector

2. We stack together two vectors: probabilities of belonging to class 0 (1 - `prediction`) and probabilities of belonging to class 1.

Then we pass this stacked matrix to `F1` score.


```python
def validate_model(threshold):
    cumulative_val_loss = 0

    for i, (val_data, val_ground_truth_class) in enumerate(val_dataloader):
        # Do forward pass on a batch of validation data
        output = net(val_data)

        # Similar to cumulative training loss, calculate cumulative validation loss
        cumulative_val_loss += nd.sum(loss(output, val_ground_truth_class)).asscalar()

        # getting prediction as a sigmoid
        prediction = net(val_data).sigmoid()

        # Converting neuron outputs to classes
        predicted_classes = mx.nd.ceil(prediction - threshold)

        # Update validation accuracy
        accuracy.update(val_ground_truth_class, predicted_classes.reshape(-1))

        # calculate probabilities of belonging to different classes. F1 metric works only with this notation
        prediction = prediction.reshape(-1)
        probabilities = mx.nd.stack(1 - prediction, prediction, axis=1)

        f1.update(val_ground_truth_class, probabilities)

    return cumulative_val_loss
```

## Putting it all together

By using the defined above functions, we can finally write our main training loop.


```python
epochs = 10
threshold = 0.5

for e in range(epochs):
    avg_train_loss = train_model() / train_data_size
    avg_val_loss = validate_model(threshold) / val_data_size

    print("Epoch: %s, Training loss: %.2f, Validation loss: %.2f, Validation accuracy: %.2f, F1 score: %.2f" %
          (e, avg_train_loss, avg_val_loss, accuracy.get()[1], f1.get()[1]))

    # we reset accuracy, so the new epoch's accuracy would be calculated from the blank state
    accuracy.reset()
```

    Epoch: 0, Training loss: 0.43, Validation loss: 0.36, Validation accuracy: 0.85, F1 score: 0.00 <!--notebook-skip-line-->

    Epoch: 1, Training loss: 0.22, Validation loss: 0.14, Validation accuracy: 0.96, F1 score: 0.35 <!--notebook-skip-line-->

    Epoch: 2, Training loss: 0.09, Validation loss: 0.11, Validation accuracy: 0.97, F1 score: 0.48 <!--notebook-skip-line-->

    Epoch: 3, Training loss: 0.07, Validation loss: 0.09, Validation accuracy: 0.96, F1 score: 0.53 <!--notebook-skip-line-->

    Epoch: 4, Training loss: 0.06, Validation loss: 0.09, Validation accuracy: 0.97, F1 score: 0.58 <!--notebook-skip-line-->

    Epoch: 5, Training loss: 0.04, Validation loss: 0.12, Validation accuracy: 0.97, F1 score: 0.59 <!--notebook-skip-line-->

    Epoch: 6, Training loss: 0.05, Validation loss: 0.09, Validation accuracy: 0.99, F1 score: 0.62 <!--notebook-skip-line-->

    Epoch: 7, Training loss: 0.05, Validation loss: 0.10, Validation accuracy: 0.97, F1 score: 0.62 <!--notebook-skip-line-->

    Epoch: 8, Training loss: 0.05, Validation loss: 0.12, Validation accuracy: 0.95, F1 score: 0.63 <!--notebook-skip-line-->

    Epoch: 9, Training loss: 0.04, Validation loss: 0.09, Validation accuracy: 0.98, F1 score: 0.65 <!--notebook-skip-line-->


In our case we hit the accuracy of 0.98 and F1 score of 0.65.

## Tip 1: Use only one neuron in the output layer

Despite that there are 2 classes, there should be only one output neuron, because `SigmoidBinaryCrossEntropyLoss` accepts only one feature as an input.

## Tip 2: Encode classes as 0 and 1

For `SigmoidBinaryCrossEntropyLoss` to work it is required that classes were encoded as 0 and 1. In some datasets the class encoding might be different, like -1 and 1 or 1 and 2. If this is how your dataset looks like, then you need to re-encode the data before using `SigmoidBinaryCrossEntropyLoss`.

## Tip 3: Use SigmoidBinaryCrossEntropyLoss instead of LogisticRegressionOutput

NDArray API has two options to calculate logistic regression loss: [SigmoidBinaryCrossEntropyLoss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss) and [LogisticRegressionOutput](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.LogisticRegressionOutput). `LogisticRegressionOutput` is designed to be an output layer when using the Module API, and is not supposed to be used when using Gluon API.

## Conclusion

In this tutorial I explained some potential pitfalls to be aware of. When doing logistic regression using Gluon API remember to:
1. Use only one neuron in the output layer
1. Encode class labels as 0 or 1
1. Use `SigmoidBinaryCrossEntropyLoss`
1. Convert probabilities to classes before calculating Accuracy

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
