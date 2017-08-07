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
    ctx1 = mx.cpu(0)
    ctx2 = mx.cpu(1)
    n = 2
    data1 = mx.sym.Variable('data1')
    data2 = mx.sym.Variable('data2')
    data3 = mx.sym.Variable('data3')
    with mx.AttrScope(ctx_group='dev1'):
        net = data1 + data2
        net = net * 3

    with mx.AttrScope(ctx_group='dev2'):
        net = net + data3

    arr = []
    arr_grad = []
    shape = (4, 5)
    with mx.Context(ctx1):
        for i in range(n):
            arr.append(mx.nd.empty(shape))
            arr_grad.append(mx.nd.empty(shape))
    with mx.Context(ctx2):
        arr.append(mx.nd.empty(shape))
        arr_grad.append(mx.nd.empty(shape))

    exec1 = net.bind(ctx1,
                     args=arr,
                     args_grad=arr_grad,
                     group2ctx={'dev1': ctx1, 'dev2': ctx2})
    arr[0][:] = 1.0
    arr[1][:] = 2.0
    arr[2][:] = 3.0
    arr2 = [a.copyto(ctx1) for a in arr]
    arr_grad2 = [a.copyto(ctx1) for a in arr_grad]
    exec2 = net.bind(ctx1,
                     args=arr2,
                     args_grad=arr_grad2)

    # Show the execution plan that involves copynode
    print(exec1.debug_str())
    exec1.forward(is_train=True)
    exec2.forward(is_train=True)
    assert reldiff(exec1.outputs[0].asnumpy(), exec2.outputs[0].asnumpy()) < 1e-6
    out_grad = mx.nd.empty(shape, ctx1)
    out_grad[:] = 1.0
    exec1.backward([out_grad])
    exec2.backward([out_grad.copyto(ctx1)])
    for a, b in zip(arr_grad, arr_grad2):
        assert reldiff(a.asnumpy(), b.asnumpy()) < 1e-6


if __name__ == '__main__':
    test_chain()
