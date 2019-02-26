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

# Automatic differentiation

MXNet supports automatic differentiation with the `autograd` package.
`autograd` allows you to differentiate a graph of NDArray operations
with the chain rule.
This is called define-by-run, i.e., the network is defined on-the-fly by
running forward computation. You can define exotic network structures
and differentiate them, and each iteration can have a totally different
network structure.

```python
import mxnet as mx
from mxnet import autograd
```

To use `autograd`, we must first mark variables that require gradient and
attach gradient buffers to them:

```python
x = mx.nd.array([[1, 2], [3, 4]])
x.attach_grad()
```

Now we can define the network while running forward computation by wrapping
it inside a `record` (operations out of `record` does not define
a graph and cannot be differentiated):

```python
with autograd.record():
  y = x * 2
  z = y * x
```

Let's backprop with `z.backward()`, which is equivalent to
`z.backward(mx.nd.ones_like(z))`. When z has more than one entry, `z.backward()`
is equivalent to `mx.nd.sum(z).backward()`:

```python
z.backward()
print(x.grad)
```

Now, let's see if this is the expected output.

Here, y = f(x), z = f(y) = f(g(x))
which means y = 2 * x and z = 2 * x * x.

After, doing backprop with `z.backward()`, we will get gradient dz/dx as follows:

dy/dx = 2,
dz/dx = 4 * x

So, we should get x.grad as an array of [[4, 8],[12, 16]].

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
