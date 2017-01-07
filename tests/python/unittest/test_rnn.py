import mxnet as mx
import numpy as np

def test_rnn_build():
    cell = mx.rnn.rnn_cell.RNNCell(100)
    params = mx.rnn.rnn_cell.RNNParams()
    states = cell.begin_state()
    outputs = []
    for i in range(3):
        output, states = cell(mx.sym.Variable('data_%d'%i), states, params, prefix='rnn_')
        outputs.append(output)
    print outputs
    outputs = mx.sym.Group(outputs)
    print outputs.list_arguments()

    print outputs.infer_shape(data_0=(10,50), data_1=(10,50), data_2=(10,50))

    a = mx.sym.Variable('a', shape=(100,0,0))
    b = mx.sym.Variable('b', shape=(0,1000,0))
    d = mx.sym.Variable('d', shape=(0,0,10))

    c = a + b + d
    print c.get_internals().infer_shape_partial()


if __name__ == '__main__':
    test_rnn_build()