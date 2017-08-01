# Experimental APIs

```eval_rst
.. currentmodule:: mxnet.contrib
```

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

## Overview

The `contrib` package contains many useful experimental APIs for new features. This is a place for the community to try out the new features, so that feature contributors can receive feedback.

## `contrib.ndarray` and `contrib.symbol` module

```eval_rst
.. autosummary::
    :nosignatures:

    ndarray.CTCLoss
    ndarray.DeformableConvolution
    ndarray.DeformablePSROIPooling
    ndarray.MultiBoxDetection
    ndarray.MultiBoxPrior
    ndarray.MultiBoxTarget
    ndarray.MultiProposal
    ndarray.PSROIPooling
    ndarray.Proposal
    ndarray.count_sketch
    ndarray.ctc_loss
    ndarray.dequantize
    ndarray.fft
    ndarray.ifft
    ndarray.quantize
```

```eval_rst
.. autosummary::
    :nosignatures:

    symbol.CTCLoss
    symbol.DeformableConvolution
    symbol.DeformablePSROIPooling
    symbol.MultiBoxDetection
    symbol.MultiBoxPrior
    symbol.MultiBoxTarget
    symbol.MultiProposal
    symbol.PSROIPooling
    symbol.Proposal
    symbol.count_sketch
    symbol.ctc_loss
    symbol.dequantize
    symbol.fft
    symbol.ifft
    symbol.quantize
```

## `contrib.autograd` module
```eval_rst
.. autosummary::
    :nosignatures:

    autograd.TrainingStateScope
    autograd.backward
    autograd.compute_gradient
    autograd.grad
    autograd.grad_and_loss
    autograd.mark_variables
    autograd.set_is_training
    autograd.test_section
    autograd.train_section
```

## `contrib.tensorboard` module
```eval_rst
.. autosummary::
    :nosignatures:

    tensorboard.LogMetricsCallback
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.contrib.ndarray
    :members:

.. automodule:: mxnet.contrib.symbol
    :members:

.. automodule:: mxnet.contrib.autograd
    :members:

.. automodule:: mxnet.contrib.tensorboard
    :members:
```

<script>auto_index("api-reference");</script>
