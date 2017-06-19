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
dx = mx.nd.zeros_like(x)
x.attach_grad(dx)
```

Now we can define the network while running forward computation by wrapping
it inside a `train_section` (operations out of `train_section` does not define
a graph and cannot be differentiated):

```python
with autograd.train_section():
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
