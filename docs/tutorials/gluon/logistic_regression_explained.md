
# Logistic regression using Gluon API explained

Logistic Regression is one of the first models newcomers to Deep Learning are implementing. In this tutorial I am going to focus on how to do logistic regression using Gluon API and provide some high level tips.

Before anything else, let's import required packages for this tutorial.


```python
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Trainer
from mxnet.gluon.data import DataLoader, ArrayDataset

mx.random.seed(12345)  # Added for reproducibility
```

In this tutorial we will use fake dataset, which contains 10 features drawn from a normal distribution with mean equals to 0 and standard deviation equals to 1, and a class label, which can be either 0 or 1. The length of the dataset is an arbitrary value. The function below helps us to generate a dataset.


```python
def get_random_data(size, ctx):
    x = nd.normal(0, 1, shape=(size, 10), ctx=ctx)
    # Class label is generated via non-random logic so the network would have a pattern to look for
    # Number 3 is selected to make sure that number of positive examples smaller than negative, but not too small
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

To work with data, Apache MXNet provides Dataset and DataLoader classes. The former is used to provide an indexed access to the data, the latter is used to shuffle and batchify the data. 

This separation is done because a source of Dataset can vary from a simple array of numbers to complex data structures like text and images. DataLoader doesn't need to be aware of the source of data as long as Dataset provides a way to get the number of records and to load a record by index. As an outcome, Dataset doesn't need to hold in memory all data at once. Needless to say, that one can implement its own versions of Dataset and DataLoader, but we are going to use existing implementation.

Below we define 2 datasets: training dataset and validation dataset. It is a good practice to measure performance of a trained model on a data that the network hasn't seen before. That is why we are going to use training set for training the model and validation set to calculate model's accuracy.


```python
train_x, train_ground_truth_class = get_random_data(train_data_size, ctx)
train_dataset = ArrayDataset(train_x, train_ground_truth_class)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_x, val_ground_truth_class = get_random_data(val_data_size, ctx)
val_dataset = ArrayDataset(val_x, val_ground_truth_class)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
```

## Defining and training the model

In real application, model can be arbitrary complex. The only requirement for the logistic regression is that the last layer of the network must be a single neuron. Apache MXNet allows us to do so by using `Dense` layer and specifying the number of units to 1.

Below, we define a model which has an input layer of 10 neurons, a couple inner layers of 10 neurons each, and output layer of 1 neuron, as it is required by logistic regression. We stack the layers using `HybridSequential` block and initialize parameters of the network using `Xavier` initialization. 


```python
net = nn.HybridSequential()

with net.name_scope():
    net.add(nn.Dense(units=10, activation='relu'))  # input layer
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 1
    net.add(nn.Dense(units=10, activation='relu'))   # inner layer 2
    net.add(nn.Dense(units=1))   # output layer: notice, it must have only 1 neuron

net.initialize(mx.init.Xavier(magnitude=2.34))
```

After defining the model, we need to define a few more thing: our loss, our trainer and our metric.

Loss function is used to calculate how the output of the network different from the ground truth. In case of the logistic regression the ground truth are class labels, which can be either 0 or 1. Because of that, we are using `SigmoidBinaryCrossEntropyLoss`, which suites well for that scenario.

Trainer object allows to specify the method of training to be used. There are various methods available, and for our tutorial we use a widely accepted method Stochastic Gradient Descent. We also need to parametrize it with learning rate value, which defines how fast training happens, and weight decay which is used for regularization.

Metric helps us to estimate how good our model is in terms of a problem we are trying to solve. Where loss function has more importance for the training process, a metric is usually the thing we are trying to improve and reach maximum value. In our example, we are using `Accuracy` as a measurement of success of our model. 

Below we define these objects.


```python
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = Trainer(params=net.collect_params(), optimizer='sgd', 
                  optimizer_params={'learning_rate': 0.1, "wd": 0.01})
accuracy = mx.metric.Accuracy()
```

Usually, it is not enough to pass the training data through a network only once to achieve high Accuracy. It helps when the network sees each example multiple times. The number of displaying every example to the network is called `epoch`. How big this number should be is unknown in advance, and usually it is estimated using trial and error approach. 

Below we are defining the main training loop, which go through each example in batches specified number of times (epochs). After each epoch we display training loss, validation loss and calculate accuracy of the model using validation set. For now, let's take a look into the code, and I am going to explain the details later.


```python
epochs = 10
threshold = 0.5

# iterate over training data epochs number of times
for e in range(epochs):
    cumulative_train_loss = 0
    cumulative_val_loss = 0
    
    # iterate over all batches of training data and calculate training loss
    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            # Do forward pass on a batch of training data
            output = net(data)
        
            # Calculate loss for the training data batch
            loss_result = loss(output, label)

        # Calculate gradients 
        loss_result.backward()

        # Change parameters of the network
        trainer.step(batch_size)

        # Since we calculate loss per single batch, but want to display it per epoch
        # we sum losses of every batch per an epoch into a single variable
        cumulative_train_loss += nd.sum(loss_result).asscalar()

    # iterate over all batches of validation data and calculate validation loss
    for i, (val_data, val_ground_truth_class) in enumerate(val_dataloader):
        # Do forward pass on a batch of validation data
        output = net(val_data)
        
        # Similar to cumulative training loss, calculate cumulative validation loss
        cumulative_val_loss += nd.sum(loss(output, val_ground_truth_class)).asscalar()
        
        # Applying sigmoid function, to get data in range [0, 1] and then
        # subtracting threshold, to make 0 serve as a class boundary: below 0 - class 0, above 0 - class 1
        # Apply mx.nd.ceil to get classes: convert negative values to 0 and positive to 1.
        prediction = mx.nd.ceil(net(val_data).sigmoid() - threshold)
        
        # Reshape predictions to match dimension of val_ground_truth_class
        # and update accuracy with the results for that batch.
        accuracy.update(val_ground_truth_class, prediction.reshape(-1)) 
    
    # in the end of epoch, we print out current values for epoch, training and validation losses, and accuracy
    print("Epoch: %s, Training loss: %.2f, Validation loss: %.2f, Validation accuracy: %s" % 
          (e, cumulative_train_loss, cumulative_val_loss, accuracy.get()[1]))

    # we reset accuracy, so the new epoch's accuracy would be calculate from the blank state
    accuracy.reset()

```

    Epoch: 0, Training loss: 446.68, Validation loss: 40.19, Validation accuracy: 0.85 <!--notebook-skip-line-->

    Epoch: 1, Training loss: 343.15, Validation loss: 30.82, Validation accuracy: 0.85 <!--notebook-skip-line-->
    
    Epoch: 2, Training loss: 187.40, Validation loss: 11.76, Validation accuracy: 0.96 <!--notebook-skip-line-->
    
    Epoch: 3, Training loss: 90.18, Validation loss: 10.13, Validation accuracy: 0.98 <!--notebook-skip-line-->
    
    Epoch: 4, Training loss: 68.51, Validation loss: 8.69, Validation accuracy: 0.97 <!--notebook-skip-line-->
    
    Epoch: 5, Training loss: 67.43, Validation loss: 6.71, Validation accuracy: 0.99 <!--notebook-skip-line-->
    
    Epoch: 6, Training loss: 54.76, Validation loss: 7.45, Validation accuracy: 0.98 <!--notebook-skip-line-->
    
    Epoch: 7, Training loss: 48.29, Validation loss: 8.56, Validation accuracy: 0.97 <!--notebook-skip-line-->
    
    Epoch: 8, Training loss: 50.50, Validation loss: 7.24, Validation accuracy: 0.98 <!--notebook-skip-line-->
    
    Epoch: 9, Training loss: 49.42, Validation loss: 7.46, Validation accuracy: 0.97 <!--notebook-skip-line-->


## Tip 1: Use only one neuron in the output layer

Despite that there are 2 classes, there should be only one output neuron, because `SigmoidBinaryCrossEntropyLoss` accepts only one feature as an input. 

In case when there are 3 or more classes, one cannot use a single Logistic regression, but should do multiclass regression. The solution would be to increase the number of output neurons to the number of classes and use `SoftmaxCrossEntropyLoss`. 

## Tip 2: Encode classes as 0 and 1

`Sigmoid` function produces values from 0 to 1. `SigmoidBinaryCrossEntropyLoss` uses these values to calculate the   loss by essentially subtracting values and class labels from 1. [Here is the formula](https://mxnet.incubator.apache.org/api/python/gluon/loss.html?highlight=sigmoidbinarycrossentropyloss#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss) used for that calculation (we use default version with `from_sigmoid` is False). That's why it is numerically better to have classes encoded in the same range as a `Sigmoid` output with 0 and 1.

If your data comes with a label encoded in a different format, such as -1 and 1, then you can either recode it to 0 and 1 by comparing the initial class to 0, or use another function instead of `Sigmoid`, like [`Tanh`](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=tanh#mxnet.ndarray.tanh), to produce output in range [-1; 1].

## Tip 3: Use SigmoidBinaryCrossEntropyLoss instead of LogisticRegressionOutput

NDArray API has two options to calculate logistic regression loss. One is `SigmoidBinaryCrossEntropyLoss`, which I used in the example. This class inherits from the `Loss` class and is intended to be used as a loss function for logistic regression. But there is also a function called `LogisticRegressionOutput`, which can be applied to any `NDArray`. Mathematically speaking, this function does the same thing as `SigmoidBinaryCrossEntropyLoss`. 

My recommendation would be to use `SigmoidBinaryCrossEntropyLoss`, because this class properly inherits from `Loss` class, while `LogisticRegressionOutput` is just a regular function. `LogisticRegressionOutput` is a   function to go when implementing logistic regression using Symbol API, but in case of using Gluon API, there are no benefits using it. The only case when you may want to consider using `LogisticRegressionOutput` as your loss, is when you need to have a support for sparse matrices.

## Tip 4: Convert probabilities to classes before calculating Accuracy

`Accuracy` metric requires 2 arguments: 1) a vector of ground-truth classes and 2) A tensor of predictions. When tensor of predictions is of the same shape as the vector of ground-truth classes, `Accuracy` class assumes that it should contain predicted classes. So, it converts the vector to `Int32` and compare each item of ground-truth classes to prediction vector. 

Because of the behaviour above, you will get an unexpected result if you just pass the output of `Sigmoid` function as is. `Sigmoid` function produces output in range [0; 1], and all numbers in that range are going to be casted to 0, even if it is as high as 0.99. To avoid this we write a custom bit of code, that:

1. Subtracts a threshold from the original prediction. Usually, the threshold is equal to 0.5, but it can be higher, if you want to increase certainty of an item to belong to class 1.

2. Use `mx.nd.ceil` function, which converts all negative values to 0 and all positive values to 1

After these transformations we can pass the result to `SigmoidBinaryCrossEntropyLoss` and expect it to behave in the defined way.

The same is not true, if the output shape of your function is different from the shape of ground-truth classes vector. For example, when doing multiclass regression with `Softmax` as an output, the shape of the output is going to be *number_of_examples* x *number_of_classes*. In that case we don't need to do the transformation above, because `Accuracy` metric would understand that shapes are different and will assume that the prediction contains probabilities of an example to belong to these classes - exactly what we want it to be. 

This makes things a little bit easier, and that's why I have seen examples where `Softmax` is used as an output of prediction. If you want to do that, make sure to change the output layer size to 2 neurons, where each neuron will provide a value of an example to belong to class 0 and 1 respectively.

## Conclusion

In this tutorial I explained some potential pitfalls to be aware about when doing logistic regression in Apache MXNet Gluon API. There might be some other challenging scenarios, which are not covered in this tutorial, like dealing with imbalanced classes, but I hope this tutorial will serve as a guidance and all other potential pitfalls would be covered in future tutorials.

 <!-- INSERT SOURCE DOWNLOAD BUTTONS -->