# coding: utf-8
"""Tools for testing."""
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals, too-many-branches, too-many-statements, broad-except, line-too-long, unused-import
from __future__ import absolute_import, print_function, division
import time
import traceback
import numbers
import numpy as np
import numpy.testing as npt
import mxnet as mx

from .context import cpu, gpu, Context
from .ndarray import array
from .symbol import Symbol

_rng = np.random.RandomState(1234)


def default_context():
    """Get default context for regression test."""
    # _TODO: get context from environment variable to support
    # testing with GPUs
    return Context.default_ctx


def set_default_context(ctx):
    """Set default ctx"""
    Context.default_ctx = ctx


def default_dtype():
    """Get default data type for regression test."""
    # _TODO: get default dtype from environment variable
    return np.float32


def get_atol(atol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 1e-20 if atol is None else atol


def get_rtol(rtol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 1e-5 if rtol is None else rtol


def random_arrays(*shapes):
    """Generate some random numpy arrays."""
    arrays = [np.random.randn(*s).astype(default_dtype())
              for s in shapes]
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def np_reduce(dat, axis, keepdims, numpy_reduce_func):
    """Compatible reduce for old version numpy

    Parameters
    ----------
    dat : np.ndarray
        Same as Numpy

    axis : None or int or list-like
        Same as Numpy

    keepdims : bool
        Same as Numpy

    numpy_reduce_func : function
        Numpy reducing function like `np.sum` or `np.max`
    """
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


def find_max_violation(a, b, rtol=None, atol=None):
    """find location of maximum violation"""
    rtol = get_rtol(rtol)
    atol = get_atol(atol)
    diff = np.abs(a-b)
    tol = atol + rtol*np.abs(b)
    violation = diff/(tol+1e-20)
    loc = np.argmax(violation)
    idx = np.unravel_index(loc, violation.shape)
    return idx, np.max(violation)


def same(a, b):
    """Test if two numpy arrays are the same

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    """
    return np.array_equal(a, b)


def almost_equal(a, b, rtol=None, atol=None):
    """Test if two numpy arrays are almost equal."""
    return np.allclose(a, b, rtol=get_rtol(rtol), atol=get_atol(atol))


def assert_almost_equal(a, b, rtol=None, atol=None, names=('a', 'b')):
    """Test that two numpy arrays are almost equal. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    threshold : None or float
        The checking threshold. Default threshold will be used if set to None
    """
    rtol = get_rtol(rtol)
    atol = get_atol(atol)

    if almost_equal(a, b, rtol, atol):
        return

    index, rel = find_max_violation(a, b, rtol, atol)
    np.set_printoptions(threshold=4, suppress=True)
    msg = npt.build_err_msg([a, b],
                            err_msg="Error %f exceeds tolerance rtol=%f, atol=%f. "
                                    " Location of maximum error:%s, a=%f, b=%f"
                            % (rel, rtol, atol, str(index), a[index], b[index]),
                            names=names)
    raise AssertionError(msg)


def almost_equal_ignore_nan(a, b, rtol=None, atol=None):
    """Test that two numpy arrays are almost equal (ignoring NaN in either array).
    Combines a relative and absolute measure of approximate eqality.
    If either the relative or absolute check passes, the arrays are considered equal.
    Including an absolute check resolves issues with the relative check where all
    array values are close to zero.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float
        The relative threshold. Default threshold will be used if set to None
    atol : None or float
        The absolute threshold. Default threshold will be used if set to None
    """
    a = np.copy(a)
    b = np.copy(b)
    nan_mask = np.logical_or(np.isnan(a), np.isnan(b))
    a[nan_mask] = 0
    b[nan_mask] = 0

    return almost_equal(a, b, rtol, atol)

def assert_almost_equal_ignore_nan(a, b, rtol=None, atol=None, names=('a', 'b')):
    """Test that two numpy arrays are almost equal (ignoring NaN in either array).
    Combines a relative and absolute measure of approximate eqality.
    If either the relative or absolute check passes, the arrays are considered equal.
    Including an absolute check resolves issues with the relative check where all
    array values are close to zero.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float
        The relative threshold. Default threshold will be used if set to None
    atol : None or float
        The absolute threshold. Default threshold will be used if set to None
    """
    a = np.copy(a)
    b = np.copy(b)
    nan_mask = np.logical_or(np.isnan(a), np.isnan(b))
    a[nan_mask] = 0
    b[nan_mask] = 0

    assert_almost_equal(a, b, rtol, atol, names)


def retry(n):
    """Retry n times before failing for stochastic test cases"""
    assert n > 0
    def decorate(f):
        """Decorate a test case"""
        def wrapper(*args, **kwargs):
            """Wrapper for tests function"""
            for _ in range(n):
                try:
                    f(*args, **kwargs)
                except AssertionError as e:
                    err = e
                    continue
                return
            raise err
        return wrapper
    return decorate


def simple_forward(sym, ctx=None, is_train=False, **inputs):
    """A simple forward function for a symbol.

    Primarily used in doctest to conveniently test the function
    of a symbol. Takes numpy array as inputs and outputs are
    also converted to numpy arrays.

    Parameters
    ----------
    ctx : Context
        If None, will take the default context.
    inputs : keyword arguments
        Mapping each input name to a numpy array.

    Returns
    -------
    The result as a numpy array. Multiple results will
    be returned as a list of numpy arrays.
    """
    ctx = ctx or default_context()
    inputs = {k: array(v) for k, v in inputs.iteritems()}
    exe = sym.bind(ctx, args=inputs)
    exe.forward(is_train=is_train)
    outputs = [x.asnumpy() for x in exe.outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def _parse_location(sym, location, ctx):
    """Parse the given location to a dictionary

    Parameters
    ----------
    sym : Symbol
    location : None or list of np.ndarray or dict of str to np.ndarray

    Returns
    -------
    dict of str to np.ndarray
    """
    assert isinstance(location, (dict, list, tuple))
    if isinstance(location, dict):
        if set(location.keys()) != set(sym.list_arguments()):
            raise ValueError("Symbol arguments and keys of the given location do not match."
                             "symbol args:%s, location.keys():%s"
                             % (str(set(sym.list_arguments())), str(set(location.keys()))))
    else:
        location = {k: v for k, v in zip(sym.list_arguments(), location)}
    location = {k: mx.nd.array(v, ctx=ctx) for k, v in location.items()}
    return location


def _parse_aux_states(sym, aux_states, ctx):
    """

    Parameters
    ----------
    sym : Symbol
    aux_states : None or list of np.ndarray or dict of str to np.ndarray

    Returns
    -------
    dict of str to np.ndarray
    """
    if aux_states is not None:
        if isinstance(aux_states, dict):
            if set(aux_states.keys()) != set(sym.list_auxiliary_states()):
                raise ValueError("Symbol aux_states names and given aux_states do not match."
                                 "symbol aux_names:%s, aux_states.keys:%s"
                                 % (str(set(sym.list_auxiliary_states())),
                                    str(set(aux_states.keys()))))
        elif isinstance(aux_states, (list, tuple)):
            aux_names = sym.list_auxiliary_states()
            aux_states = {k:v for k, v in zip(aux_names, aux_states)}
        aux_states = {k: mx.nd.array(v, ctx=ctx) for k, v in aux_states.items()}
    return aux_states


def numeric_grad(executor, location, aux_states=None, eps=1e-4, use_forward_train=True):
    """Calculates a numeric gradient via finite difference method.

    Class based on Theano's `theano.gradient.numeric_grad` [1]

    Parameters
    ----------
    executor : Executor
        exectutor that computes the forward pass
    location : list of numpy.ndarray or dict of str to numpy.ndarray
        Argument values used as location to compute gradient
        Maps the name of arguments to the corresponding numpy.ndarray.
        Value of all the arguments must be provided.
    aux_states : None or list of numpy.ndarray or dict of str to numpy.ndarray, optional
        Auxiliary states values used as location to compute gradient
        Maps the name of aux_states to the corresponding numpy.ndarray.
        Value of all the auxiliary arguments must be provided.
    eps : float, optional
        epsilon for the finite-difference method
    use_forward_train : bool, optional
        Whether to use `is_train=True` in testing.
    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    approx_grads = {k: np.zeros(v.shape, dtype=np.float32)
                    for k, v in location.items()}
    for k, v in location.items():
        executor.arg_dict[k][:] = v
    for k in location:
        location[k] = np.ascontiguousarray(location[k])
    for k, v in location.items():
        old_value = v.copy()
        for i in range(np.prod(v.shape)):
            # inplace update
            v.ravel()[i] += eps/2.0
            executor.arg_dict[k][:] = v
            if aux_states is not None:
                for key, val in aux_states.items():
                    executor.aux_dict[key][:] = val
            executor.forward(is_train=use_forward_train)
            f_peps = executor.outputs[0].asnumpy()

            v.ravel()[i] -= eps
            executor.arg_dict[k][:] = v
            if aux_states is not None:
                for key, val in aux_states.items():
                    executor.aux_dict[key][:] = val
            executor.forward(is_train=use_forward_train)
            f_neps = executor.outputs[0].asnumpy()

            approx_grads[k].ravel()[i] = (f_peps - f_neps).sum() / eps
            v.ravel()[i] = old_value.ravel()[i]
        # copy back the original value
        executor.arg_dict[k][:] = old_value
    return approx_grads


def check_numeric_gradient(sym, location, aux_states=None, numeric_eps=1e-3, rtol=1e-2,
                           atol=None, grad_nodes=None, use_forward_train=True, ctx=None):
    """Verify an operation by checking backward pass via finite difference method.

    Based on Theano's `theano.gradient.verify_grad` [1]

    Parameters
    ----------
    sym : Symbol
        Symbol containing op to test
    location : list or tuple or dict
        Argument values used as location to compute gradient

        - if type is list of numpy.ndarray
            inner elements should have the same the same order as mxnet.sym.list_arguments().
        - if type is dict of str -> numpy.ndarray
            maps the name of arguments to the corresponding numpy.ndarray.
        *In either case, value of all the arguments must be provided.*
    aux_states : ist or tuple or dict, optional
        The auxiliary states required when generating the executor for the symbol
    numeric_eps : float, optional
        Delta for the finite difference method that approximates the gradient
    check_eps : float, optional
        relative error eps used when comparing numeric grad to symbolic grad
    grad_nodes : None or list or tuple or dict, optional
        Names of the nodes to check gradient on
    use_forward_train : bool
        Whether to use is_train=True when computing the finite-difference
    ctx : Context, optional
        Check the gradient computation on the specified device
    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    if ctx is None:
        ctx = default_context()

    def random_projection(shape):
        """Get a random weight matrix with not too small elements

        Parameters
        ----------
        shape : list or tuple
        """
        # random_projection should not have elements too small,
        # otherwise too much precision is lost in numerical gradient
        plain = _rng.rand(*shape) + 0.1
        return plain

    location = _parse_location(sym=sym, location=location, ctx=ctx)
    location_npy = {k:v.asnumpy() for k, v in location.items()}
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx)
    if aux_states is not None:
        aux_states_npy = {k:v.asnumpy() for k, v in aux_states.items()}
    else:
        aux_states_npy = None
    if grad_nodes is None:
        grad_nodes = sym.list_arguments()
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, (list, tuple)):
        grad_nodes = list(grad_nodes)
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, dict):
        grad_req = grad_nodes.copy()
        grad_nodes = grad_nodes.keys()
    else:
        raise ValueError

    input_shape = {k: v.shape for k, v in location.items()}
    _, out_shape, _ = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable("__random_proj")
    out = sym * proj
    out = mx.sym.MakeLoss(out)

    location = dict(list(location.items()) +
                    [("__random_proj", mx.nd.array(random_projection(out_shape[0]), ctx=ctx))])
    args_grad_npy = dict([(k, _rng.normal(0, 0.01, size=location[k].shape)) for k in grad_nodes]
                         + [("__random_proj", _rng.normal(0, 0.01, size=out_shape[0]))])

    args_grad = {k: mx.nd.array(v, ctx=ctx) for k, v in args_grad_npy.items()}

    executor = out.bind(ctx, grad_req=grad_req,
                        args=location, args_grad=args_grad, aux_states=aux_states)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location)))
    assert len(executor.outputs) == 1

    executor.forward(is_train=True)
    executor.backward()
    symbolic_grads = {k:executor.grad_dict[k].asnumpy() for k in grad_nodes}

    numeric_gradients = numeric_grad(executor, location_npy, aux_states_npy,
                                     eps=numeric_eps, use_forward_train=use_forward_train)
    for name in grad_nodes:
        fd_grad = numeric_gradients[name]
        orig_grad = args_grad_npy[name]
        sym_grad = symbolic_grads[name]
        if grad_req[name] == 'write':
            assert_almost_equal(fd_grad, sym_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'add':
            assert_almost_equal(fd_grad, sym_grad - orig_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'null':
            assert_almost_equal(orig_grad, sym_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        else:
            raise ValueError("Invalid grad_req %s for argument %s"%(grad_req[name], name))


def check_symbolic_forward(sym, location, expected, rtol=1E-4, atol=None,
                           aux_states=None, ctx=None):
    """Compare foward call to expected value.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            contain all the numpy arrays corresponding to `sym.list_arguments()`
        - if type is dict of str to np.ndarray
            contain the mapping between argument names and their values
    expected : list of np.ndarray or dict of str to np.ndarray
        The expected output value

        - if type is list of np.ndarray
            contain arrays corresponding to exe.outputs
        - if type is dict of str to np.ndarray
            contain mapping between sym.list_output() and exe.outputs
    check_eps : float, optional
        relative error to check to
    aux_states : list of np.ndarray of dict, optional
        - if type is list of np.ndarray
            contain all the numpy arrays corresponding to sym.list_auxiliary_states
        - if type is dict of str to np.ndarray
            contain the mapping between names of auxiliary states and their values
    ctx : Context, optional
        running context
    """
    if ctx is None:
        ctx = default_context()

    location = _parse_location(sym=sym, location=location, ctx=ctx)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx)
    if isinstance(expected, dict):
        expected = [expected[k] for k in sym.list_outputs()]
    args_grad_data = {k:mx.nd.empty(v.shape, ctx=ctx) for k, v in location.items()}

    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states)
    for g in executor.grad_arrays:
        if g:
            g[:] = 0

    executor.forward(is_train=False)
    outputs = [x.asnumpy() for x in executor.outputs]

    for output_name, expect, output in zip(sym.list_outputs(), expected, outputs):
        assert_almost_equal(expect, output, rtol, atol,
                            ("EXPECTED_%s"%output_name, "FORWARD_%s"%output_name))


def check_symbolic_backward(sym, location, out_grads, expected, rtol=1e-5, atol=None,
                            aux_states=None, grad_req='write', ctx=None):
    """Compare backward call to expected value.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            contain all the numpy arrays corresponding to mxnet.sym.list_arguments
        - if type is dict of str to np.ndarray
            contain the mapping between argument names and their values
    out_grads : None or list of np.ndarray or dict of str to np.ndarray
        numpy arrays corresponding to sym.outputs for incomming gradient

        - if type is list of np.ndarray
            contains arrays corresponding to exe.outputs
        - if type is dict of str to np.ndarray
            contains mapping between mxnet.sym.list_output() and Executor.outputs
    expected : list of np.ndarray or dict of str to np.ndarray
        expected gradient values

        - if type is list of np.ndarray
            contains arrays corresponding to exe.grad_arrays
        - if type is dict of str to np.ndarray
            contains mapping between sym.list_arguments() and exe.outputs
    check_eps: float, optional
        relative error to check to
    aux_states : list of np.ndarray or dict of str to np.ndarray
    grad_req : str or list of str or dict of str to str, optional
        gradient requirements. 'write', 'add' or 'null'
    ctx : Context, optional
        running context
    """
    if ctx is None:
        ctx = default_context()

    location = _parse_location(sym=sym, location=location, ctx=ctx)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx)
    if isinstance(expected, (list, tuple)):
        expected = {k:v for k, v in zip(sym.list_arguments(), expected)}
    args_grad_npy = {k:_rng.normal(size=v.shape) for k, v in expected.items()}
    args_grad_data = {k: mx.nd.array(v, ctx=ctx) for k, v in args_grad_npy.items()}
    if isinstance(grad_req, str):
        grad_req = {k:grad_req for k in sym.list_arguments()}
    elif isinstance(grad_req, (list, tuple)):
        grad_req = {k:v for k, v in zip(sym.list_arguments(), grad_req)}

    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states)
    executor.forward(is_train=True)
    if isinstance(out_grads, (tuple, list)):
        out_grads = [mx.nd.array(v, ctx=ctx) for v in out_grads]
    elif isinstance(out_grads, (dict)):
        out_grads = {k:mx.nd.array(v, ctx=ctx) for k, v in out_grads.items()}
    else:
        assert out_grads is None
    executor.backward(out_grads)

    grads = {k: v.asnumpy() for k, v in args_grad_data.items()}
    for name in expected:
        if grad_req[name] == 'write':
            assert_almost_equal(expected[name], grads[name], rtol, atol,
                                ("EXPECTED_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'add':
            assert_almost_equal(expected[name], grads[name] - args_grad_npy[name],
                                rtol, atol, ("EXPECTED_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'null':
            assert_almost_equal(args_grad_npy[name], grads[name],
                                rtol, atol, ("EXPECTED_%s"%name, "BACKWARD_%s"%name))
        else:
            raise ValueError("Invalid grad_req %s for argument %s"%(grad_req[name], name))


def check_speed(sym, location=None, ctx=None, N=20, grad_req=None, typ="whole",
                **kwargs):
    """Check the running speed of a symbol

    Parameters
    ----------
    sym : Symbol
        symbol to run the speed test
    location : none or dict of str to np.ndarray
        location to evaluate the inner executor
    ctx : Context
        running context
    N : int, optional
        repeat times
    grad_req : None or str or list of str or dict of str to str, optional
        gradient requirements
    typ : str, optional
        "whole" or "forward"

        - "whole"
            test the forward_backward speed
        - "forward"
            only test the forward speed
    """
    if ctx is None:
        ctx = default_context()

    if grad_req is None:
        grad_req = 'write'
    if location is None:
        exe = sym.simple_bind(grad_req=grad_req, ctx=ctx, **kwargs)
        location = {k: _rng.normal(size=arr.shape, scale=1.0) for k, arr in
                    exe.arg_dict.items()}
    else:
        assert isinstance(location, dict), "Expect dict, get \"location\"=%s" %str(location)
        exe = sym.simple_bind(grad_req=grad_req, ctx=ctx,
                              **{k: v.shape for k, v in location.items()})

    for name, iarr in location.items():
        exe.arg_dict[name][:] = iarr.astype(exe.arg_dict[name].dtype)

    if typ == "whole":
        # Warm up
        exe.forward(is_train=True)
        exe.backward(out_grads=exe.outputs)
        for output in exe.outputs:
            output.wait_to_read()
        # Test forward + backward
        tic = time.time()
        for _ in range(N):
            exe.forward(is_train=True)
            exe.backward(out_grads=exe.outputs)
        mx.nd.waitall()
        toc = time.time()
        forward_backward_time = (toc - tic) * 1.0 / N
        return forward_backward_time
    elif typ == "forward":
        # Warm up
        exe.forward(is_train=False)
        for output in exe.outputs:
            output.wait_to_read()

        # Test forward only
        tic = time.time()
        for _ in range(N):
            exe.forward(is_train=False)
        mx.nd.waitall()
        toc = time.time()
        forward_time = (toc - tic) * 1.0 / N
        return forward_time
    else:
        raise ValueError('typ can only be "whole" or "forward".')


def check_consistency(sym, ctx_list, scale=1.0, grad_req='write',
                      arg_params=None, aux_params=None, tol=None,
                      raise_on_err=True, ground_truth=None):
    """Check symbol gives the same output for different running context

    Parameters
    ----------
    sym : Symbol or list of Symbols
        symbol(s) to run the consistency test
    ctx_list : list
        running context. See example for more detail.
    scale : float, optional
        standard deviation of the inner normal distribution. Used in initialization
    grad_req : str or list of str or dict of str to str
        gradient requirement.

    Examples
    --------
    >>> # create the symbol
    >>> sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')
    >>> # initialize the running context
    >>> ctx_list =\
[{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},\
 {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},\
 {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},\
 {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},\
 {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}}]
    >>> check_consistency(sym, ctx_list)
    >>> sym = mx.sym.Concat(name='concat', num_args=2)
    >>> ctx_list = \
[{'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},\
 {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}},\
 {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float16, 'concat_arg1': np.float16}},\
 {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},\
 {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}}]
    >>> check_consistency(sym, ctx_list)
    """
    if tol is None:
        tol = {np.dtype(np.float16): 1e-1,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0}
    elif isinstance(tol, numbers.Number):
        tol = {np.dtype(np.float16): tol,
               np.dtype(np.float32): tol,
               np.dtype(np.float64): tol,
               np.dtype(np.uint8): tol,
               np.dtype(np.int32): tol}

    assert len(ctx_list) > 1
    if isinstance(sym, Symbol):
        sym = [sym]*len(ctx_list)
    else:
        assert len(sym) == len(ctx_list)

    output_names = sym[0].list_outputs()
    arg_names = sym[0].list_arguments()
    exe_list = []
    for s, ctx in zip(sym, ctx_list):
        assert s.list_arguments() == arg_names
        assert s.list_outputs() == output_names
        exe_list.append(s.simple_bind(grad_req=grad_req, **ctx))

    arg_params = {} if arg_params is None else arg_params
    aux_params = {} if aux_params is None else aux_params
    for n, arr in exe_list[0].arg_dict.items():
        if n not in arg_params:
            arg_params[n] = np.random.normal(size=arr.shape, scale=scale)
    for n, arr in exe_list[0].aux_dict.items():
        if n not in aux_params:
            aux_params[n] = 0
    for exe in exe_list:
        for name, arr in exe.arg_dict.items():
            arr[:] = arg_params[name]
        for name, arr in exe.aux_dict.items():
            arr[:] = aux_params[name]

    dtypes = [np.dtype(exe.outputs[0].dtype) for exe in exe_list]
    max_idx = np.argmax(dtypes)
    gt = ground_truth
    if gt is None:
        gt = exe_list[max_idx].output_dict.copy()
        if grad_req != 'null':
            gt.update(exe_list[max_idx].grad_dict)

    # test
    for exe in exe_list:
        exe.forward(is_train=False)

    for i, exe in enumerate(exe_list):
        if i == max_idx:
            continue
        for name, arr in zip(output_names, exe.outputs):
            gtarr = gt[name].astype(dtypes[i]).asnumpy()
            arr = arr.asnumpy()
            try:
                assert_almost_equal(arr, gtarr, rtol=tol[dtypes[i]], atol=tol[dtypes[i]])
            except Exception as e:
                print('Predict Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                traceback.print_exc()
                if raise_on_err:
                    raise e
                else:
                    print(str(e))

    # train
    if grad_req != 'null':
        for exe in exe_list:
            exe.forward(is_train=True)
            exe.backward(exe.outputs)

        for i, exe in enumerate(exe_list):
            if i == max_idx:
                continue
            curr = zip(output_names + arg_names, exe.outputs + exe.grad_arrays)
            for name, arr in curr:
                if gt[name] is None:
                    assert arr is None
                    continue
                gtarr = gt[name].astype(dtypes[i]).asnumpy()
                arr = arr.asnumpy()
                try:
                    assert_almost_equal(arr, gtarr, rtol=tol[dtypes[i]], atol=tol[dtypes[i]])
                except Exception as e:
                    print('Train Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                    traceback.print_exc()
                    if raise_on_err:
                        raise e
                    else:
                        print(str(e))

    return gt
