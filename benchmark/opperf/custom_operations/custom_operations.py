import mxnet as mx
import mxnet.ndarray as nd

"""
MXNet's Custom Operator Benchmark Tests.

It does a simple element wise addition to make sure computation
is not too much and we can observe custom operator logistics overhead.
"""


# 1. Define Custom Operator - Element wise Addition Multiplication
class CustomAddOne(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0] + 1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("CustomAddOne")
class CustomAddOneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CustomAddOneProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['in']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        # inputs, outputs, aux
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return CustomAddOne()


"""Helps to benchmark MXNet's Custom Op for Element wise addition on a (1000, 1) tensor.
    Performs both forward and backward operation.

    This test mainly uncovers core custom op overhead in MXNet.

    Benchmark will be done on the following operation:
    native_add -> native_add -> native_add -> CUSTOM_ADD -> native_add -> native_add -> native_add

    By default run on 'float32' precision.
"""

# TODO
