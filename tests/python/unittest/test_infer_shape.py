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
    print(len(aux_shapes))
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

if __name__ == "__main__":
    test_mlp2_infer_shape()
    test_mlp2_infer_error()
