!pip install --pre mxnet -f https://dist.mxnet.io/python
!pip install contextvars

```{.python .input  n=1}
import mxnet 
```

```{.python .input  n=2}
mxnet.__version__
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "'2.0.0'"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

# Necessary components that are not in the network

The data and algorithms are not the only components that you need to create your
trained deep learning model. In this notebook, we will talk about some of the
common components that you will use for training your own machine learning
models. Here is a list of components that we will talk about in this notebook

1. Loss function
    1. Built-in
    2. Custom
2. Optimizers
3. Gradients and backpropagation
    1. Autograd

```{.python .input  n=1}
from mxnet import np, npx,gluon
npx.set_np()
```

Till now you have seen how to create an algorithm using mxnet API and the basics
of using mxnet. When you start training the ML algorithm, how does it actually
learn or train?

There are three main components of what happens during training an algorithm. In
this notebook, we will talk about these components, namely,

1. Loss function which is used to calculate how far the model is from the true
distribution
2. Autograd, the mxnet auto differentiation tool to calculate the gradients to
optimize the parameters
3. Optimizer which is used to update the parameters based on an optimization
algorithm


Lets look at each of them a little closer.

## Loss function

Loss functions are used to train neural networks and help the algorithm learn
the data distribution. In a loss function, we compute the difference between
output that we get from the neural network and ground truth value. This score is
used to update the neural network weights during training. Let's look at a
simple example first.

Suppose you have a neural network `net` and the data is stored in variable
`data`. Let's take 5 total records and the output from the neural network after
the first epoch is given by the following variable. The values are in millions.

```{.python .input  n=2}
net = gluon.nn.Dense(1)
net.initialize()

nn_input = np.array([1.2,0.56,3.0,0.72,0.89]).reshape(5,1)

nn_output = net(nn_input)
nn_output
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "array([[0.00820068],\n       [0.00382698],\n       [0.02050169],\n       [0.00492041],\n       [0.00608217]])"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

But the ground truth value of the data is stored in `groundtruth_label` is

```{.python .input  n=3}
groundtruth_label = np.array([[0.0083],
                             [0.00382],
                             [0.02061],
                             [0.00495],
                             [0.00639]]).reshape(5,1)
```

For this problem, we will use the L2 Loss. L2Loss, also called Mean Squared
Error, is a regression loss function that computes the squared distances between
the target values and the output of the neural network. It is defined as:

$$L = \frac{1}{2N}\sum_i{|label_i − pred_i|)^2}$$

Compared to L1, L2 loss it is a smooth function and it creates larger gradients
for large loss values. However due to the squaring it puts high weight on
outliers.

```{.python .input  n=4}
def L2Loss(output_values, true_values):
    return np.mean((output_values - true_values)**2,axis=1)/2

L2Loss(nn_output,groundtruth_label)
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "array([4.9326903e-09, 2.4373411e-11, 5.8656311e-09, 4.3792128e-10,\n       4.7380531e-08])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now we do the same thing using the mxnet API

```{.python .input  n=5}
from mxnet.gluon import nn, loss as gloss
loss = gloss.L2Loss()

loss(nn_output,groundtruth_label)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "array([4.9326903e-09, 2.4373411e-11, 5.8656311e-09, 4.3792128e-10,\n       4.7380531e-08])"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

A network can improve by iteratively updating its weights to minimise this loss.
Some tasks use a combination of multiple loss functions, but often you’ll just
use one. MXNet Gluon provides a number of the most commonly used loss functions,
and you’ll choose certain loss functions depending on your network and task.
Some common task and loss function pairs include:

- regression: L1Loss, L2Loss

- classification: SigmoidBinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss

- embeddings: HingeLoss

#### Customizing your Loss functions

You can also create custom loss functions using **Loss Blocks**. For more
information see []()


You can inherit the base `Loss` class and write your own `forward` method. The
backward propagation will be automatically computed by autograd. However that
only holds true if you can build your loss from existing operators.


```{.python .input  n=6}
from mxnet.gluon.loss import Loss

class custom_L1_loss(Loss):
    def __init__(self,weight=None, batch_axis=0, **kwargs):
        super(custom_L1_loss,self).__init__(weight, batch_axis, **kwargs)

    def forward(self, pred, label):
        l = np.abs(label - pred)
        l = l.reshape(len(l),)
        return l
    
L1 = custom_L1_loss()
L1(nn_output,groundtruth_label)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "array([9.9324621e-05, 6.9818925e-06, 1.0831095e-04, 2.9594637e-05,\n       3.0783284e-04])"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=7}
l1=gloss.L1Loss()
l1(nn_output,groundtruth_label)
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "array([9.9324621e-05, 6.9818925e-06, 1.0831095e-04, 2.9594637e-05,\n       3.0783284e-04])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Backpropagation

The second step is the backward step which computes the gradient of the loss
with respect to the parameters. In Gluon, this step is achieved by doing the
first step in an `autograd.record()` scope to record the computations needed to
calculate the loss, and then calling `l.backward()` to compute the gradient of
the loss with respect to the parameters.

In this step, you learn how to use the MXNet autograd package to perform
gradient calculations by automatically calculating derivatives.

This is helpful because it will help you save time and effort. You train models
to get better as a function of experience. Usually, getting better means
minimizing a loss function. To achieve this goal, you often iteratively compute
the gradient of the loss with respect to weights and then update the weights
accordingly. Gradient calculations are straightforward through a chain rule.
However, for complex models, working this out manually is challenging.

The autograd package helps you by automatically calculating derivatives.

```{.python .input  n=8}
from mxnet import autograd
```

### Basic use

As an example, you could differentiate a function $f(x) = 2 x^2$ with respect to
parameter $x$. You can start by assigning an initial value of $x$, as follows:

```{.python .input  n=9}
x = np.array([[1, 2], [3, 4]])
x
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "array([[1., 2.],\n       [3., 4.]])"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

After you compute the gradient of $f(x)$ with respect to $x$, you need a place
to store it. In MXNet, you can tell an nparray that you plan to store a gradient
by invoking its `attach_grad()` method, shown in the following example.

```{.python .input  n=10}
x.attach_grad()
```

Next, define the function $y=f(x)$. To let MXNet store $y$, so that you can
compute gradients later, use the following code to put the definition inside an
`autograd.record()` scope.

```{.python .input  n=11}
with autograd.record():
    y = 2 * x * x
```

You can invoke back propagation (backprop) by calling `y.backward()`. When $y$
has more than one entry, `y.backward()` is equivalent to `y.sum().backward()`.

```{.python .input  n=12}
y.backward()
```

Next, verify whether this is the expected output. Note that $y=2x^2$ and
$\frac{dy}{dx} = 4x$, which should be `[[4, 8],[12, 16]]`. Check the
automatically computed results.

```{.python .input  n=13}
x.grad
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "array([[ 4.,  8.],\n       [12., 16.]])"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Taking the same example from above, lets look at what happens when you use
`backward()` method

```{.python .input  n=14}
# Attaching gradients to the input 
nn_input.attach_grad()

# Computing the gradients 
with autograd.record():
    nn_output = net(nn_input)
    L2_loss = loss(nn_output,groundtruth_label)

L2_loss.backward()

print(nn_input.grad)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[-6.7877414e-07]\n [ 4.7713527e-08]\n [-7.4018578e-07]\n [-2.0224668e-07]\n [-2.1036976e-06]]\n"
 }
]
```

## Optimizer

The loss function is how much the parameters are changing based on how far the
model is. Optimizer is how the model weights or parameters are updated based on
the loss function. In Gluon, this optimization step is performed by the
`gluon.Trainer`.

Lets look at a basic example of how to call the `gluon.Trainer` method.

```{.python .input  n=15}
from mxnet import optimizer
```

```{.python .input  n=16}
trainer = gluon.Trainer(net.collect_params(),
                       optimizer='Adam',
                       optimizer_params={
                           'learning_rate':0.1,
                           'wd':0.001
                       })
```

When creating a **Gluon Trainer** you must provide the trainer object with
1. A collection of parameters that need to be learnt. The collection of
parameters will the weights and biases of your network that you are training.
2. An Optimization algorithm (optimizer) that you want to use for training. This
algorithm will be used to update the parameters every training iteration when
`trainer.step` is called. For more information, see [optimizers in v1.6 TODO:
CHANGE](https://mxnet.apache.org/versions/1.6/api/python/docs/api/optimizer/index.html)

```{.python .input  n=17}
curr_weight = net.weight.data()
print(curr_weight)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[0.0068339]]\n"
 }
]
```

```{.python .input  n=18}
batch_size = len(nn_input)
trainer.step(batch_size)
print(net.weight.data())
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[0.10660961]]\n"
 }
]
```

```{.python .input  n=19}
print(curr_weight - net.weight.grad() * 1 / 5)

```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[0.00698099]]\n"
 }
]
```

```{.python .input}

```

```{.python .input}

```
