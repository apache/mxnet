# Training MNIST With NumpyOp

Uses the same setup as example/mnist/mlp.py. Except the loss symbol is
custom defined with NumpyOp. mxnet.operator.NumpyOp help move computation
in a symbol's forward/backward operation to python frontend. This is for
fast implementation/experimentation of non-performance-critical symbols.
If it is becoming a bottleneck, please consider write a C++/CUDA version.