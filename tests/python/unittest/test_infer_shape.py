# pylint: skip-file
import mxnet as mx
from common import models
from nose.tools import *

def test_mlp2_infer_shape():
    # Build MLP
    out = models.mlp2()
    # infer shape
    data_shape = (100, 100)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(out.list_arguments(), arg_shapes))
    assert len(out_shapes) == 1
    assert out_shapes[0] == (100, 10)
    true_shapes = {'fc2_bias': (10,),
                   'fc2_weight' : (10, 1000),
                   'fc1_bias' : (1000,),
                   'fc1_weight' : (1000,100) }
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v

@raises(mx.MXNetError)
def test_mlp2_infer_error():
    # Test shape inconsistent case
    out = models.mlp2()
    weight_shape= (1, 100)
    data_shape = (100, 100)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape, fc1_weight=weight_shape)


def test_backward_infer():
    w = mx.sym.Variable("weight")
    wshift = mx.sym.Variable("wshift", shape=(1,))
    data = mx.sym.Variable("data")
    # broadcast add here, not being able to deduce shape correctly
    wt = mx.sym.broadcast_add(w, wshift)
    # shape constraint, this is what enables backward shape inference
    wt = mx._symbol_internal._identity_with_attr_like_rhs(wt, w)
    net = mx.sym.FullyConnected(data=data, weight=wt, num_hidden=11, no_bias=True)
    data_shape = (7, 100)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(net.list_arguments(), arg_shapes))
    true_shapes = {'weight': (11, 100)}
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v

if __name__ == "__main__":
    test_mlp2_infer_shape()
    test_mlp2_infer_error()
    test_backward_infer()
