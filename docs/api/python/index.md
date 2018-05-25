# MXNet - Python API

MXNet provides a comprehensive and flexible Python API to serve a broad community of developers with different levels of experience and wide ranging requirements. In this section, we provide an in-depth discussion of the functionality provided by various MXNet Python packages.

MXNet's Python API has two primary high-level packages*: the Gluon API and Module API. We recommend that new users start with the Gluon API as it's more flexible and easier to debug. Underlying these high-level packages are the core packages of NDArray and Symbol.

NDArray works with arrays in an imperative fashion, i.e. you define how arrays will be transformed to get to an end result. Symbol works with arrays in a declarative fashion, i.e. you define the end result that is required (via a symbolic graph) and the MXNet engine will use various optimizations to determine the steps required to obtain this. With NDArray you have a great deal of flexibility when composing operations (as you can use Python control flow), and you can easily step through your code and inspect the values of arrays, which helps with debugging. Unfortunately, this comes at a performance cost when compared to Symbol, which can perform optimizations on the symbolic graph.

Module API is backed by Symbol, so, although it's very performant, it's also a little more restrictive. With the Gluon API, you can get the best of both worlds. You can develop and test your model imperatively using NDArray, a then switch to Symbol for faster model training and inference (if Symbol equivalents exist for your operations).

Code examples are placed throughout the API documentation and these can be run after importing MXNet as follows:

```python
>>> import mxnet as mx
```

```eval_rst

.. note:: A convenient way to execute code examples is using the ``%doctest_mode`` mode of
   Jupyter notebook, which allows for pasting multi-line examples containing
   ``>>>`` while preserving indentation. Run ``%doctest_mode?`` in Jupyter notebook
   for more details.

```

\* Some old references to Model API may exist, but this API has been deprecated.

## Autograd API

```eval_rst
.. toctree::
   :maxdepth: 1

   autograd/autograd.md
```

## Callback API

```eval_rst
.. toctree::
   :maxdepth: 1

   callback/callback.md
```

## Contrib Package

```eval_rst
.. toctree::
   :maxdepth: 1

   contrib/contrib.md
   contrib/text.md
   contrib/onnx.md
   contrib/svrg_optimization.md
```

## Gluon API

```eval_rst
.. toctree::
   :maxdepth: 1

   gluon/gluon.md
   gluon/nn.md
   gluon/rnn.md
   gluon/loss.md
   gluon/data.md
   gluon/model_zoo.md
   gluon/contrib.md
```

## Image API

```eval_rst
.. toctree::
   :maxdepth: 1

   image/image.md
```

## IO API

```eval_rst
.. toctree::
   :maxdepth: 1

   io/io.md
```

## KV Store API

```eval_rst
.. toctree::
   :maxdepth: 1

   kvstore/kvstore.md
```

## Metric API

```eval_rst
.. toctree::
   :maxdepth: 1

   metric/metric.md
```

## Module API

```eval_rst
.. toctree::
   :maxdepth: 1

   module/module.md
   executor/executor.md
```

## NDArray API

```eval_rst
.. toctree::
   :maxdepth: 1

   ndarray/ndarray.md
   ndarray/random.md
   ndarray/linalg.md
   ndarray/sparse.md
   ndarray/contrib.md
```

## Optimization API

```eval_rst
.. toctree::
   :maxdepth: 1

   optimization/optimization.md
   optimization/contrib.md
```

## Profiler API

```eval_rst
.. toctree::
   :maxdepth: 1

   profiler/profiler.md
```

## Run-Time Compilation API

```eval_rst
.. toctree::
   :maxdepth: 1

   rtc/rtc.md
```

## Symbol API

```eval_rst
.. toctree::
   :maxdepth: 1

   symbol/symbol.md
   symbol/random.md
   symbol/linalg.md
   symbol/sparse.md
   symbol/contrib.md
   symbol/rnn.md
```

## Symbol in Pictures API

```eval_rst
.. toctree::
   :maxdepth: 1

   symbol_in_pictures/symbol_in_pictures.md
```

## Tools

```eval_rst
.. toctree::
    :maxdepth: 1

    tools/test_utils.md
    tools/visualization.md
```
