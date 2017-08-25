import numpy as np
import mxnet as mx

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    if diff == 0:
        return 0
    reldiff = diff  / norm
    return reldiff

def test_chain():
    n = 2
    data1 = mx.sym.Variable('data1')
    data2 = mx.sym.Variable('data2')
    with mx.AttrScope(ctx_group='dev1'):
        net = data1 + data2
        net = net * 3

    with mx.AttrScope(ctx_group='dev2'):
        net = net + data1

    with mx.Context(mx.cpu(0)):
        shape = (4, 5)
        arr = [mx.nd.empty(shape) for i in range(n)]
        arr_grad = [mx.nd.empty(shape) for i in range(n)]

    exec1 = net.bind(mx.cpu(),
                     args=arr,
                     args_grad=arr_grad,
                     group2ctx={'dev1': mx.cpu(0), 'dev2': mx.cpu(1)})
    arr[0][:] = 1.0
    arr[1][:] = 2.0
    arr2 = [a.copyto(mx.cpu()) for a in arr]
    arr_grad2 = [a.copyto(mx.cpu()) for a in arr_grad]
    exec2 = net.bind(mx.cpu(),
                     args=arr2,
                     args_grad=arr_grad2)

    # Show the execution plan that involves copynode
    print(exec1.debug_str())
    exec1.forward()
    exec2.forward()
    assert reldiff(exec1.outputs[0].asnumpy(), exec2.outputs[0].asnumpy()) < 1e-6
    out_grad = mx.nd.empty(shape, mx.cpu(1))
    out_grad[:] = 1.0
    exec1.backward([out_grad])
    exec2.backward([out_grad.copyto(mx.cpu())])
    for a, b in zip(arr_grad, arr_grad2):
        assert reldiff(a.asnumpy(), b.asnumpy()) < 1e-6


if __name__ == '__main__':
    test_chain()
