# Autograd Package

```eval_rst
.. currentmodule:: mxnet.autograd
```

## Overview

The `autograd` package enables automatic
differentiation of NDArray operations.
In machine learning applications,
`autograd` is often used to calculate the gradients
of loss functions with respect to parameters.


### Record vs Pause

`autograd` records computation history on the fly to calculate gradients later.
This is only enabled inside a `with autograd.record():` block.
A `with auto_grad.pause()` block can be used inside a `record()` block
to temporarily disable recording.

To compute gradient with respect to an `NDArray` `x`, first call `x.attach_grad()`
to allocate space for the gradient. Then, start a `with autograd.record()` block,
and do some computation. Finally, call `backward()` on the result:

```python
>>> x = mx.nd.array([1,2,3,4])
>>> x.attach_grad()
>>> with mx.autograd.record():
...     y = x * x + 1
>>> y.backward()
>>> print(x.grad)
[ 2.  4.  6.  8.]
<NDArray 4 @cpu(0)>
```


## Train mode and Predict Mode

Some operators (Dropout, BatchNorm, etc) behave differently in
when training and when making predictions.
This can be controlled with `train_mode` and `predict_mode` scope.

By default, MXNet is in `predict_mode`.
A `with autograd.record()` block by default turns on `train_mode`
(equivalent to ``with autograd.record(train_mode=True)``).
To compute a gradient in prediction mode (as when generating adversarial examples),
call record with `train_mode=False` and then call `backward(train_mode=False)`

Although training usually coincides with recording,
this isn't always the case.
To control *training* vs *predict_mode* without changing
*recording* vs *not recording*,
Use a `with autograd.train_mode():`
or `with autograd.predict_mode():` block.

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
    Function
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.autograd
    :members:
```

<script>auto_index("api-reference");</script>
