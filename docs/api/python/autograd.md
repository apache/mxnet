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
For example, in machine learning applications,
``autograd`` is often used to calculate the gradient
of a loss functions with respect to parameters.

While automatic differentiation was previously available
through the symbolic API, ``autograd`` works in the fully imperative context.
In other words, we do not need to specify a computational graph a priori
in order to take gradients.
Of course, in order to differentiate a value ``y`` with respect
to an NDArray ``x``, we need to know how ``y`` was calculated from ``x``.
You might wonder, how does ``autograd`` do this
without a pre-specified computation graph?


The trick here is that ``autograd`` builds a computation graph on the fly.
When we calculate ``y = f(x)``, MXNet can remember
how the value of ``y`` relates to ``x``.
It's as if MXNet turned on a tape recorder to keep track of
how each value was generated.
To indicate to MXNet that we want to turn on the metaphorical tape recorder,
all we have to do is place the code in a ``with autograd.record():`` block.
For any variable ``x`` where we might later want to access its gradient,
we can call ``x.attach_grad`` to allocate its space.
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

### ``Train_mode`` and ``predict_mode``

Often, we want to define functions that behave differently
when we are training models vs making predictions.
By default, MXNet assumes we are in predict mode.
However, usually when we take gradients, we are in the process of training.
MXNet let's us decouple *training* vs *predict_mode* from
*recording* vs *not recording*.

When we turn on autograd, this by default turns on train_mode
(``with autograd.record(train_mode=True):``).
To over ride this behavior (as when generating adversarial examples),
we can optionally call record by (``with autograd.record(train_mode=False):``).

Additionally we can override the current score and elicit train or predict
behaviors for some lines of code by placing them
in a ``with aurograd.train_mode():`` or ``with aurograd.predict_mode():``
block, respectively.


A detailed tutorials is available in Part 1 of
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
