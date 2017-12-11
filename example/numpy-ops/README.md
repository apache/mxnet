# Training MNIST With NumpyOp

Uses the same setup as example/mnist/mlp.py. Except the loss symbol is
custom defined with NumpyOp. mxnet.operator.NumpyOp help move computation
in a symbol's forward/backward operation to python frontend. This is for
fast implementation/experimentation of non-performance-critical symbols.
If it is becoming a bottleneck, please consider write a C++/CUDA version.

# Example operator with CustomOp

You can find the example of a custom operator which performs elementwise
square for sparse ndarray: `custom_sparse_sqr.py`. The example contains
implementations for `infer_storage_type` and `infer_storage_type_backward`
interfaces which can be used to infer sparse storage types `csr`
and `row_sparse` in the forward and backward pass respectively.

To run the example :
```
python custom_sparse_sqr.py
```
OR
```
python3 custom_sparse_sqr.py
```
