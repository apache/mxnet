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

In this step, you learn how to use the MXNet `autograd` package to perform gradient calculations by automatically calculating derivatives.

This is helpful because it will help you save time and effort. You train models to get better as a function of experience. Usually, getting better means minimizing a loss function. To achieve this goal, you often iteratively compute the gradient of the loss with respect to weights and then update the weights accordingly. Gradient calculations are straightforward through a chain rule. However, for complex models, working this out manually is challenging.

The `autograd` package helps you by automatically calculating derivatives.

## Basic use

To get started, import the `autograd` package as in the following code.

```{.python .input}
from mxnet import np, npx
from mxnet import autograd
npx.set_np()
```

As an example, you could differentiate a function $f(x) = 2 x^2$ with respect to parameter $x$. You can start by assigning an initial value of $x$, as follows:

```{.python .input  n=3}
x = np.array([[1, 2], [3, 4]])
x
```

After you compute the gradient of $f(x)$ with respect to $x$, you need a place to store it. In MXNet, you can tell an ndarray that you plan to store a gradient by invoking its `attach_grad` method, shown in the following example.

```{.python .input  n=6}
x.attach_grad()
```

Next, define the function $y=f(x)$. To let MXNet store $y$, so that you can compute gradients later, use the following code to put the definition inside an `autograd.record()` scope. 

```{.python .input  n=7}
with autograd.record():
    y = 2 * x * x
```

You can invoke back propagation (backprop) by calling `y.backward()`. When $y$ has more than one entry, `y.backward()` is equivalent to `y.sum().backward()`.
<!-- I'm not sure what this second part really means. I don't have enough context. TMI?-->

```{.python .input  n=8}
y.backward()
```

Next, verify whether this is the expected output. Note that $y=2x^2$ and $\frac{dy}{dx} = 4x$, which should be `[[4, 8],[12, 16]]`. Check the automatically computed results.

```{.python .input  n=9}
x.grad
```

## Using Python control flows

Sometimes you want to write dynamic programs where the execution depends on real-time values. MXNet records the execution trace and computes the gradient as well.

Consider the following function `f` in the following example code. The function doubles the inputs until its `norm` reaches 1000. Then it selects one element depending on the sum of its elements. 
<!-- I wonder if there could be another less "mathy" demo of this -->

```{.python .input}
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

```{.python .input}
a = np.random.uniform(size=2)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

You can see that `b` is a linear function of `a`, and `c` is chosen from `b`. The gradient with respect to `a` be will be either `[c/a[0], 0]` or `[0, c/a[1]]`, depending on which element from `b` is picked. You see the results of this example with this code:

```{.python .input}
a.grad == c/a
```

## Next Steps

After you have used `autograd`, learn about training a neural network. See [Step 4: Train the neural network](4-train.md).
