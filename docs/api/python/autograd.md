# Autograd Package


```eval_rst
.. currentmodule:: mxnet.autograd
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

## Overview

The ``autograd`` package consists of functions that enable automatic
differentiation of scalar values with respect to NDArrays.
In machine learning applications,
``autograd`` is often used to calculate the gradients
of loss functions with respect to parameters.



### Record vs Pause

The enable differentiation without first declaring a computation graph,
``autograd`` builds the graph on the fly.
Building the graph incurs some overhead,
so it only takes place inside an ``with autograd.record():`` block.
We can return at any point to the default scope (not recording)
using a ``with auto_grad.pause()`` block.

For any variable ``x`` with respect to which we want a gradient,
we must first call ``x.attach_grad`` to allocate space for the gradient.
Then once we calculate the value of a function ``y``
inside a ``with autograd.record()`` block, we can call ``y.backward()``.

```python
>>> x = mx.nd.array([1,2,3,4])
>>> x.attach_grad()
>>> with mx.autograd.record():
    y = x * x
>>> print(x.grad)
>>>[ 2.  4.  6.  8.]
<NDArray 4 @cpu(0)>
```


## ``Train_mode`` and ``predict_mode``

Some functions are intended to behave differently
during model training models vs. making predictions.
We can dictate this behavior by setting the ``train_mode`` or ``predict_mode`` scope.
By default, MXNet assumes we are in ``predict_mode``.
When we turn on autograd, this by default turns on train_mode
(``with autograd.record()`` is equivalent to
``with autograd.record(train_mode=True):``).
To change this default behavior
(as when generating adversarial examples),
we can optionally call record via
(``with autograd.record(train_mode=False):``).

Although training usually coincides with recording,
this isn't always the case.
So MXNet lets us decouple *training* vs *predict_mode* from
*recording* vs *not recording*.
At any point, we can override the current scope
for some lines of code by placing them
in a ``with autograd.train_mode():``
or ``with autograd.predict_mode():`` block, respectively.

Detailed tutorials are available in Part 1 of
[the MXNet gluon book](http://gluon.mxnet.io/).






<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

## Autograd

```eval_rst
.. autosummary::
    :nosignatures:

    record
    pause
    train_mode
    predict_mode
    backward
    set_training
    is_training
    set_recording
    is_recording
    mark_variables
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.autograd
    :members:
```

<script>auto_index("api-reference");</script>
