# MXNet 0.9 (NNVM) Release Note

Version 0.9 brings a number of important features and changes, including a back-end refactor to adopt the [NNVM](https://github.com/dmlc/nnvm) framework, a profiler for analyzing performance, a fast image IO and augmentation module that bypasses GIL, and various other changes.

## NNVM Refactor

NNVM is a library for neural network graph construction, optimization, and operator registration. It serves as an intermediary layer between the front-end (MXNet user API) and the back-end (computation on the device). After version 0.9, MXNet fully adopts the NNVM framework. Now it's easier to create operators. You can also register "pass"es that process and optimizes the graph when `bind` is called on the symbol. For more discussion on how to create operators with NNVM, please refer to [How to Create New Operators](../how_to/new_op.md)

Other changes brought by NNVM include:
- Backward shape inference is now supported
- All operators can now be used with both symbolic and ndarray API. For example, `mx.nd.Activation(x, act_type='relu')` works now.
- Optional cython API for mx.symbol and mx.ndarray is now available. Use `make cython` to activate it for accelerated communication with the back-end.

## Profiler

![MLP Profile](https://cloud.githubusercontent.com/assets/17693755/18035938/0a43484a-6d93-11e6-80d4-241c6ca552ea.png)

MXNet now provides a native profiler for analyzing the performance of operators. This feature compliments general profiling tools like nvprof and gprof by summarizing at the operator level, instead of function, kernel, or instruction level.

To use this feature, first set `USE_PROFILER = 1` in `config.mk` and rebuild mxnet. Then add three lines at the beginning and end of the section of your program you want to profile:
```python
mx.profiler.profiler_set_config(mode=scope, filename=fname)
profiler.profiler_set_state('run')

# do computation ...

profiler.profiler_set_state('stop')
```
`scope` can be 'symbolic' (to only include symbolic operations) or 'all' (to include all operations), and `fname` is the path to save profiler output.

After program finishes, navigate to [chrome://tracing](chrome://tracing) in a Chrome browser and load profiler output to see the results.

## Image IO

MXNet already has `mx.io.ImageRecordIter` for loading and preprocessing images. However, some tasks need more flexible image processing API. Detection, for example, requires transforming labels together with images. Usually, people write custom data iterators in python to handle this. But due to the infamous Global Interpreter Lock (GIL), python scripts cannot use multithreading to speed up processing.

`mx.image` provides a set of fast image processing API that leverage MXNet Engine to automatically parallelize processing. You can write
```python
imgs = [mx.image.imdecode(open(f).read()) for f in img_paths]
```
and decoding will be automatically run in parallel.

## Miscellaneous

- sgd and adam optimizer are now implemented with a single imperative call. They should be as fast and memory efficient as cc optimizers. ccsgd is now deprecated and redirects to sgd.
- Layout support is added. Use `mx.io.DataDesc(..., layout='NHWC')` in provide_data to specify data layout. use `mx.sym.YourSymbol(..., __layout__='NHWC')` to specify output layout. `layout` option is now available for Convolution layer.
- element_mask is removed. Please use src*mask.reshape((mask.size, 1, 1, ..., 1)) directly as binary ops now support broadcasting.
- sum_axis, max_axis, and min_axis are deprecated. Please use mx.nd.max(src, axis=n) instead.
- symbol attributes are now limited to ctx_group, lr_mult, wd_mult, force_mirroring. All other custom attributes need to be in __xxx__ format (start and end with double underscore) or an error will be triggered during attribute parsing.
