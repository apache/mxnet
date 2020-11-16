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

# Step 3: Automatic differentiation with autograd

In this step, you learn how to use the MXNet `autograd` package to perform
gradient calculations.

## Basic use

To get started, import the `autograd` package with the following code.

```python
from mxnet import np, npx
from mxnet import autograd
npx.set_np()
```

As an example, you could differentiate a function $f(x) = 2 x^2$ with respect to
parameter $x$. For Autograd, you can start by assigning an initial value of $x$,
as follows:

```python
x = np.array([[1, 2], [3, 4]])
x
```

After you compute the gradient of $f(x)$ with respect to $x$, you need a place
to store it. In MXNet, you can tell a ndarray that you plan to store a gradient
by invoking its `attach_grad` method, as shown in the following example.

```python
x.attach_grad()
```

Next, define the function $y=f(x)$. To let MXNet store $y$, so that you can
compute gradients later, use the following code to put the definition inside an
`autograd.record()` scope.

```python
with autograd.record():
    y = 2 * x * x
```

You can invoke back propagation (backprop) by calling `y.backward()`. When $y$
has more than one entry, `y.backward()` is equivalent to `y.sum().backward()`.

```python
y.backward()
```

Next, verify whether this is the expected output. Note that $y=2x^2$ and
$\frac{dy}{dx} = 4x$, which should be `[[4, 8],[12, 16]]`. Check the
automatically computed results.

```python
x.grad
```

Now you get to dive into `y.backward()` by first discussing a bit on gradients. As
alluded to earlier `y.backward()` is equivalent to `y.sum().backward()`.

```python
with autograd.record():
    y = np.sum(2 * x * x)
y.backward()
x.grad
```

Additionally, you can only run backward once. Unless you use the flag
`retain_graph` to be `True`.

```python
with autograd.record():
    y = np.sum(2 * x * x)
y.backward(retain_graph=True)
print(x.grad)
print("Since you have retained your previous graph you can run backward again")
y.backward()
print(x.grad)

try:
    y.backward()
except:
    print("However, you can't do backward twice unless you retain the graph.")
```

## Custom MXNet ndarray operations

In order to understand the `backward()` method it is beneficial to first
understand how you can create custom operations. MXNet operators are classes
with a forward and backward method. Where the number of args in `backward()`
must equal the number of items returned in the `forward()` method. Additionally,
the number of arguments in the `forward()` method must match the number of
output arguments from `backward()`. You can modify the gradients in backward to
return custom gradients. For instance, below you can return a different gradient then
the actual derivative.

```python
class My_First_Custom_Operation(autograd.Function):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        return 2 * x, 2 * x * y, 2 * y
    def backward(self, dx, dxy, dy):
        """
        The input number of arguments must match the number of outputs from forward.
        Furthermore, the number of output arguments must match the number of inputs from forward.
        """
        return x, y
```

Now you can use the first custom operation you have built.

```python
x = np.random.uniform(-1, 1, (2, 3)) 
y = np.random.uniform(-1, 1, (2, 3))
x.attach_grad()
y.attach_grad()
with autograd.record():
    z = My_First_Custom_Operation()
    z1, z2, z3 = z(x, y)
    out = z1 + z2 + z3 
out.backward()
print(np.array_equiv(x.asnumpy(), x.asnumpy()))
print(np.array_equiv(y.asnumpy(), y.asnumpy()))
```

Alternatively, you may want to have a function which is different depending on
if you are training or not.

```python
def my_first_function(x):
    if autograd.is_training(): # Return something else when training
        return(4 * x)
    else:
        return(x)
```

```python
y = my_first_function(x)
print(np.array_equiv(y.asnumpy(), x.asnumpy()))
with autograd.record(train_mode=False):
    y = my_first_function(x)
y.backward()
print(x.grad)
with autograd.record(train_mode=True): # train_mode = True by default
    y = my_first_function(x)
y.backward()
print(x.grad)
```

You could create functions with `autograd.record()`.

```python
def my_second_function(x):
    with autograd.record():
        return(2 * x)
```

```python
y = my_second_function(x)
y.backward()
print(x.grad)
```

You can also combine multiple functions.

```python
y = my_second_function(x)
with autograd.record():
    z = my_second_function(y) + 2
z.backward()
print(x.grad)
```

Additionally, MXNet records the execution trace and computes the gradient
accordingly. The following function `f` doubles the inputs until its `norm`
reaches 1000. Then it selects one element depending on the sum of its elements.

```python
def f(a):
    b = a * 2
    while np.abs(b).sum() < 1000:
        b = b * 2
    if b.sum() >= 0:
        c = b[0]
    else:
        c = b[1]
    return c
```

In this example, you record the trace and feed in a random value.

```python
a = np.random.uniform(size=2)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

You can see that `b` is a linear function of `a`, and `c` is chosen from `b`.
The gradient with respect to `a` be will be either `[c/a[0], 0]` or `[0,
c/a[1]]`, depending on which element from `b` is picked. You see the results of
this example with this code:

```python
a.grad == c / a
```

As you can notice there are 3 values along the dimension 0, so taking a `mean`
along this axis is the same as summing that axis and multiplying by `1/3`.

## Advanced MXNet ndarray operations with Autograd

You can control gradients for different ndarray operations. For instance,
perhaps you want to check that the gradients are propagating properly?
the `attach_grad()` method automatically detaches itself from the gradient.
Therefore, the input up until y will no longer look like it has `x`. To
illustrate this notice that `x.grad` and `y.grad` is not the same in the second
example.

```python
with autograd.record():
    y = 3 * x
    y.attach_grad()
    z = 4 * y + 2 * x
z.backward()
print(x.grad)
print(y.grad)
```

Is not the same as:

```python
with autograd.record():
    y = 3 * x
    z = 4 * y + 2 * x
z.backward()
print(x.grad)
print(y.grad)
```

## Next steps

Learn how to initialize weights, choose loss function, metrics and optimizers for training your neural network [Step 4: Necessary components
to train the neural network](4-components.md).
