import mxnet as mx
from mxnet.operator import NumpyOp

import numpy as np

def _np_reduce(dat, axis, keepdims, numpy_reduce_func):
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis) if axis is not None else range(len(dat.shape))
    ret = dat
    for i in reversed(sorted(axis)):
        ret = numpy_reduce_func(ret, axis=i)
    if keepdims:
        keepdims_shape = list(dat.shape)
        for i in axis:
            keepdims_shape[i] = 1
        ret = ret.reshape(tuple(keepdims_shape))
    return ret

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    if diff == 0:
        return 0
    reldiff = diff  / norm
    return reldiff

class SumAllLoss(NumpyOp):
    """
    Operator to sum all elements in a tensor.
    """
    def __init__(self):
        super(SumAllLoss, self).__init__(False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        return in_shape, [(1,)]
    def forward(self, in_data, out_data):
        out_data[0][:] = np.sum(in_data[0])
    def backward(self, out_grad, in_data, out_data, in_grad):
        in_grad[0][:] = 1

def numeric_grad(executor, location, eps=1e-4):
    """ Class based on Theano's `theano.gradient.numeric_grad` [1]
    Calculates a numeric gradient via finite difference method.

    Parameters:
    -----------
    executor: `mxnet.executor.Executor`
        exectutor that computes the forward pass

    location: list np.ndarray
        location in which to compute gradient. list should be the same size
        as executor.arg_arrays

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    args = executor.arg_arrays
    for a, l in zip(args, location):
        a[:] = np.asarray(l)
    approx_grads = [np.zeros_like(l) for l in location]

    executor.forward(is_train=True)
    f_x = executor.outputs[0].asnumpy()

    x_copy = [np.copy(x) for x in location]
    for ap_grad, loc, reset in zip(approx_grads, location, x_copy):
        for i in range(np.prod(loc.shape)):
            # inplace update of memory
            loc.ravel()[i] += eps

            # set initial states. Need to set all due to inplace operations
            for inp, val in zip(args, location):
                inp[:] = val
            executor.forward(is_train=True)
            f_eps = executor.outputs[0].asnumpy()
            ap_grad.ravel()[i] = (f_eps - f_x) / eps
            loc.ravel()[i] = reset.ravel()[i]

    return approx_grads


rng = np.random.RandomState(1234)

def check_numeric_gradient(sym, location, aux_states=[], numeric_eps=1e-4, check_eps=1e-2):
    """
    Verify an operation by checking backwards pass via
    finite difference method.

    Based on Theano's `theano.gradient.numeric_grad` [1]

    Parameters:
    -----------
    sym: `mxnet.symbol.Symbol`
        Symbol containing op to test
    location: list of numpy.ndarray
        list of numpy.ndarray used as location to compute gradient
    numeric_eps: float, optional
        delta for location to compute numeric gradient
    check_eps: float, optional
        relative error eps used when comparing numeric grad to symbolic grad

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """

    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection(shape):
        plain = rng.rand(*shape) + 0.1
        #plain = np.ones(shape)
        return plain

    kwargs = {name:array.shape for name, array in zip(sym.list_arguments(), location)}
    arg_shape, out_shape, aux_shape = sym.infer_shape(**kwargs)

    proj = mx.sym.Variable("__random_proj")
    out = SumAllLoss()(sym*proj)

    args = out.list_arguments()

    kwargs = {a:loc.shape for a,loc in zip(args, location)}

    arr_data = [mx.nd.array(l) for l in location] + [mx.nd.empty(out_shape[0])]
    arr_grad = [mx.nd.empty(l.shape) for l in location] + [mx.nd.empty(out_shape[0])]
    arr_aux = [mx.nd.array(l) for l in aux_states]

    executor = out.bind(mx.cpu(), args=arr_data, args_grad=arr_grad, aux_states=arr_aux)

    location = location + [random_projection(out_shape[0])]
    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location))
        )
    for inp, source in zip(location, executor.arg_arrays):
        source[:] = inp

    for g in executor.grad_arrays:
        if g:
            g[:] = 0

    assert len(executor.outputs) == 1

    executor.forward(is_train=True)
    executor.backward()
    # remove the proj from grads
    symbolic_grad = [g.asnumpy() for g in executor.grad_arrays[0:-1]]

    # refactor forward out of here as this no longer computes correct forward pass
    numeric_gradients = numeric_grad(executor, location, eps=numeric_eps)

    for name, numeric, symbolic in zip(out.list_arguments(), numeric_gradients, symbolic_grad):
        rel = reldiff(numeric, symbolic)
        if rel > check_eps:
            raise Exception("Numeric check failed for %s. relative error of %f expected <= %f"%(name, rel, check_eps))

def check_symbolic_forward(sym, location, expected, check_eps=1e-5):
    """ Compare foward call to expected value.

    Parameters
    ---------
    sym: mxnet.symbol.Symbol
        output symbol
    location: list np.ndarray
        list of numpy arrays corresponding to sym.list_arguments
    expected: list np.ndarray
        list of arrays corresponding to sym.outputs
    check_eps: float
        relative error to check to
    """
    kwargs = {name:array.shape for name, array in zip(sym.list_arguments(), location)}
    arg_shape, out_shape, aux_shape = sym.infer_shape(**kwargs)

    args = sym.list_arguments()

    kwargs = {a:loc.shape for a,loc in zip(args, location)}

    arr_data = [mx.nd.array(l) for l in location]
    arr_grad = [mx.nd.empty(l.shape) for l in location]

    executor = sym.bind(mx.cpu(), args=arr_data, args_grad=arr_grad)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location))
        )
    for inp, source in zip(location, executor.arg_arrays):
        source[:] = inp

    for g in executor.grad_arrays:
        if g:
            g[:] = 0

    assert len(executor.outputs) == 1

    executor.forward()

    outputs = [x.asnumpy() for x in executor.outputs]
    for expect, output in zip(expected, outputs):
        assert reldiff(expect, output) <= check_eps

def check_symbolic_backward(sym, location, out_grad, expected, check_eps=1e-5):
    """ Compare backwards call to expected value.

    Parameters
    ---------
    sym: mxnet.symbol.Symbol
        output symbol
    location: list np.ndarray
        list of numpy arrays corresponding to sym.list_arguments
    location: list np.ndarray
        list of numpy arrays corresponding to sym.outputs for incomming gradient
    expected: list np.ndarray
        list of arrays corresponding to sym.outputs
    check_eps: float
        relative error to check to
    """

    kwargs = {name:array.shape for name, array in zip(sym.list_arguments(), location)}
    arg_shape, out_shape, aux_shape = sym.infer_shape(**kwargs)

    args = sym.list_arguments()

    kwargs = {a:loc.shape for a,loc in zip(args, location)}

    arr_data = [mx.nd.array(l) for l in location]
    arr_grad = [mx.nd.empty(l.shape) for l in location]
    out_grad = [mx.nd.array(j) for j in out_grad]

    executor = sym.bind(mx.cpu(), args=arr_data, args_grad=arr_grad)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location))
        )
    for inp, source in zip(location, executor.arg_arrays):
        source[:] = inp

    for g in executor.grad_arrays:
        if g:
            g[:] = 0

    executor.forward()
    executor.backward(out_grad)

    grads = [x.asnumpy() for x in executor.grad_arrays]
    for expect, grad in zip(expected, grads):
        assert reldiff(expect, grad) <= check_eps
