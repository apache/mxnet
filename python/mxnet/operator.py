# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, no-self-use, too-many-locals, broad-except
"""numpy interface for operators."""
from __future__ import absolute_import

from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, cast, c_int, c_char, c_char_p, cast, c_bool
c_int_p = POINTER(c_int)
from .base import c_array, c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle
from . import symbol
from .ndarray import NDArray

class PythonOp(object):
    """Base class for operators implemented in python

    Parameters
    ----------
    need_top_grad : bool
        the default need_top_grad() function returns this value
    """
    _ref_holder = []

    def __init__(self, need_top_grad=True):
        self.info_ = None
        self.need_top_grad_ = need_top_grad

    def __call__(self, *args, **kwargs):
        return self.get_symbol(*args, **kwargs)

    def get_symbol(self, *args, **kwargs):
        """Create a symbol from numpy operator.
        This Should only be called once per instance if operator contains
        internal states.

        Parameters
        ----------
        args : list
            a list of input arguments (symbols)

        Returns
        -------
        sym : mxnet.symbol.Symbol
        """
        raise NotImplementedError("Must override this")

    def forward(self, in_data, out_data):
        """forward interface. override to create new operators

        Parameters
        ----------
        in_data, out_data: list
            input and output for forward. See document for
            corresponding arguments of Operator::Forward
        """
        out_data[0][:] = in_data[0]

    def backward(self, out_grad, in_data, out_data, in_grad):
        """backward interface. override to create new operators

        Parameters
        ----------
        out_grad, in_data, out_data, in_grad : list
            input and output for backward. See document for
            corresponding arguments of Operator::Backward
        """
        # pylint: disable=W0613
        in_grad[0][:] = 1.0

    def infer_shape(self, in_shape):
        """infer_shape interface. override to create new operators

        Parameters
        ----------
        in_shape : list
            list of argument shapes in the same order as
            declared in list_arguments.

        Returns
        -------
        in_shape : list
            list of argument shapes. Can be modified from in_shape.
        out_shape : list
            list of output shapes calculated from in_shape,
            in the same order as declared in list_arguments.
        """
        return in_shape, [in_shape[0]]

    def list_outputs(self):
        """list_outputs interface. override to create new operators

        Returns
        -------
        outputs : list
            list of output blob names.
        """
        return ['output']

    def list_arguments(self):
        """list_arguments interface. override to create new operators

        Returns
        -------
        in_shape : list
            list of argument shapes in the same order as
            declared in list_arguments.
        """
        return ['data']

    def need_top_grad(self):
        """Whether this operator needs out_grad for backward.

        Returns
        -------
        need_top_grad : bool
            Whether this operator needs out_grad for backward.
            Should be set to False for loss layers.
        """
        return self.need_top_grad_

class NumpyOp(PythonOp):
    """Base class for numpy operators. numpy operators allow parts
    of computation in symbolic graph to be writen in numpy. This feature
    is intended for quickly hacking out a solution for non performance
    critical parts. Please consider write a c++ implementation if it becomes
    a bottleneck.
    Note that if your operator contains internal states (like arrays),
    it cannot be used for multi-gpu training.
    """
    def __init__(self, need_top_grad=True):
        super(NumpyOp, self).__init__(need_top_grad)

    def get_symbol(self, *args, **kwargs):
        fb_functype = CFUNCTYPE(None, c_int, POINTER(POINTER(mx_float)), POINTER(c_int),
                                POINTER(POINTER(mx_uint)), POINTER(c_int), c_void_p)
        infer_functype = CFUNCTYPE(None, c_int, POINTER(c_int),
                                   POINTER(POINTER(mx_uint)), c_void_p)
        list_functype = CFUNCTYPE(None, POINTER(POINTER(POINTER(c_char))), c_void_p)
        class NumpyOpInfo(Structure):
            """Structure that holds Callback information. Passed to NumpyOpProp"""
            _fields_ = [
                ('forward', fb_functype),
                ('backward', fb_functype),
                ('infer_shape', infer_functype),
                ('list_outputs', list_functype),
                ('list_arguments', list_functype),
                ('p_forward', c_void_p),
                ('p_backward', c_void_p),
                ('p_infer_shape', c_void_p),
                ('p_list_outputs', c_void_p),
                ('p_list_arguments', c_void_p),
                ]
        def forward_entry(num_tensor, tensor_ptrs, tensor_dims,
                          tensor_shapes, tensor_tags, _):
            """C Callback for NumpyOp::Forward"""
            tensors = [[] for i in range(4)]
            for i in range(num_tensor):
                shape = [tensor_shapes[i][j] for j in range(tensor_dims[i])]
                buff = ctypes2numpy_shared(tensor_ptrs[i], shape)
                tensors[tensor_tags[i]].append(buff)
            self.forward(in_data=tensors[0], out_data=tensors[1])

        def backward_entry(num_tensor, tensor_ptrs, tensor_dims,
                           tensor_shapes, tensor_tags, _):
            """C Callback for NumpyOp::Backward"""
            tensors = [[] for i in range(4)]
            for i in range(num_tensor):
                shape = [tensor_shapes[i][j] for j in range(tensor_dims[i])]
                buff = ctypes2numpy_shared(tensor_ptrs[i], shape)
                tensors[tensor_tags[i]].append(buff)
            self.backward(in_data=tensors[0], out_data=tensors[1],
                          in_grad=tensors[2], out_grad=tensors[3])

        def infer_shape_entry(num_tensor, tensor_dims,
                              tensor_shapes, _):
            """C Callback for NumpyOpProp::InferShape"""
            n_in = len(self.list_arguments())
            n_out = len(self.list_outputs())
            assert num_tensor == n_in + n_out

            shapes = [[tensor_shapes[i][j] for j in range(tensor_dims[i])] for i in range(n_in)]
            ishape, oshape = self.infer_shape(shapes)
            assert len(oshape) == n_out
            assert len(ishape) == n_in
            rshape = list(ishape) + list(oshape)
            for i in range(n_in+n_out):
                tensor_shapes[i] = cast(c_array(mx_uint, rshape[i]), POINTER(mx_uint))
                tensor_dims[i] = len(rshape[i])

        def list_outputs_entry(out, _):
            """C Callback for NumpyOpProp::ListOutputs"""
            ret = self.list_outputs()
            ret = [c_str(i) for i in ret] + [c_char_p(0)]
            ret = c_array(c_char_p, ret)
            out[0] = cast(ret, POINTER(POINTER(c_char)))

        def list_arguments_entry(out, _):
            """C Callback for NumpyOpProp::ListArguments"""
            ret = self.list_arguments()
            ret = [c_str(i) for i in ret] + [c_char_p(0)]
            ret = c_array(c_char_p, ret)
            out[0] = cast(ret, POINTER(POINTER(c_char)))


        self.info_ = NumpyOpInfo(fb_functype(forward_entry),
                                 fb_functype(backward_entry),
                                 infer_functype(infer_shape_entry),
                                 list_functype(list_outputs_entry),
                                 list_functype(list_arguments_entry),
                                 None, None, None, None, None)
        cb_ptr = format(cast(pointer(self.info_), c_void_p).value, 'x')
        # pylint: disable=E1101
        sym = symbol.Symbol._Native(*args,
                                    info=cb_ptr,
                                    need_top_grad=self.need_top_grad(),
                                    **kwargs)
        # keep a reference of ourself in PythonOp so we don't get garbage collected.
        PythonOp._ref_holder.append(self)
        return sym

class NDArrayOp(PythonOp):
    """Base class for numpy operators. numpy operators allow parts
    of computation in symbolic graph to be writen in numpy. This feature
    is intended for quickly hacking out a solution for non performance
    critical parts. Please consider write a c++ implementation if it becomes
    a bottleneck.
    Note that if your operator contains internal states (like arrays),
    it cannot be used for multi-gpu training.
    """
    def __init__(self, need_top_grad=True):
        super(NDArrayOp, self).__init__(need_top_grad)

    def get_symbol(self, *args, **kwargs):
        fb_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_void_p), POINTER(c_int), c_void_p)
        infer_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_int),
                                   POINTER(POINTER(mx_uint)), c_void_p)
        list_functype = CFUNCTYPE(c_bool, POINTER(POINTER(POINTER(c_char))), c_void_p)
        deps_functype = CFUNCTYPE(c_bool, c_int_p, c_int_p, c_int_p,
                                  c_int_p, POINTER(c_int_p), c_void_p)
        class NDArrayOpInfo(Structure):
            """Structure that holds Callback information. Passed to NDArrayOpProp"""
            _fields_ = [
                ('forward', fb_functype),
                ('backward', fb_functype),
                ('infer_shape', infer_functype),
                ('list_outputs', list_functype),
                ('list_arguments', list_functype),
                ('declare_backward_dependency', deps_functype),
                ('p_forward', c_void_p),
                ('p_backward', c_void_p),
                ('p_infer_shape', c_void_p),
                ('p_list_outputs', c_void_p),
                ('p_list_arguments', c_void_p),
                ('p_declare_backward_dependency', c_void_p)
                ]
        def forward_entry(num_ndarray, ndarraies, tags, _):
            """C Callback for NDArrayOp::Forward"""
            try:
                tensors = [[] for i in range(4)]
                for i in range(num_ndarray):
                    if tags[i] == 1:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle),
                                                        writable=True))
                    else:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle),
                                                        writable=False))
                self.forward(in_data=tensors[0], out_data=tensors[1])
            except Exception as e:
                print('Error in NDArrayOp.forward: ', str(e))
                return False
            return True

        def backward_entry(num_ndarray, ndarraies, tags, _):
            """C Callback for NDArrayOp::Backward"""
            try:
                tensors = [[] for i in range(4)]
                for i in range(num_ndarray):
                    if tags[i] == 2:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle),
                                                        writable=True))
                    else:
                        tensors[tags[i]].append(NDArray(cast(ndarraies[i], NDArrayHandle),
                                                        writable=False))
                self.backward(in_data=tensors[0], out_data=tensors[1],
                              in_grad=tensors[2], out_grad=tensors[3])
            except Exception as e:
                print('Error in NDArrayOp.backward: ', str(e))
                return False
            return True

        def infer_shape_entry(num_tensor, tensor_dims,
                              tensor_shapes, _):
            """C Callback for NDArrayOpProp::InferShape"""
            try:
                n_in = len(self.list_arguments())
                n_out = len(self.list_outputs())
                assert num_tensor == n_in + n_out

                shapes = [[tensor_shapes[i][j] for j in range(tensor_dims[i])] for i in range(n_in)]
                ishape, oshape = self.infer_shape(shapes)
                assert len(oshape) == n_out
                assert len(ishape) == n_in
                rshape = list(ishape) + list(oshape)
                for i in range(n_in+n_out):
                    tensor_shapes[i] = cast(c_array(mx_uint, rshape[i]), POINTER(mx_uint))
                    tensor_dims[i] = len(rshape[i])
            except Exception as e:
                print('Error in NDArrayOp.infer_shape: ', str(e))
                return False
            return True

        def list_outputs_entry(out, _):
            """C Callback for NDArrayOpProp::ListOutputs"""
            try:
                ret = self.list_outputs()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception as e:
                print('Error in NDArrayOp.list_outputs: ', str(e))
                return False
            return True

        def list_arguments_entry(out, _):
            """C Callback for NDArrayOpProp::ListArguments"""
            try:
                ret = self.list_arguments()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception as e:
                print('Error in NDArrayOp.list_arguments: ', str(e))
                return False
            return True

        def declare_backward_dependency(out_grad, in_data, out_data, num_dep, deps, _):
            """C Callback for NDArrayOpProp::DeclareBacwardDependency"""
            try:
                out_grad = [out_grad[i] for i in xrange(len(self.list_outputs()))]
                in_data = [in_data[i] for i in xrange(len(self.list_arguments()))]
                out_data = [out_data[i] for i in xrange(len(self.list_outputs()))]
                rdeps = self.declare_backward_dependency(out_grad, in_data, out_data)
                num_dep[0] = len(rdeps)
                rdeps = cast(c_array(c_int, rdeps), c_int_p)
                deps[0] = rdeps
            except Exception as e:
                print('Error in NDArrayOp.declare_backward_dependency: ', str(e))
                return False
            return True

        self.info_ = NDArrayOpInfo(fb_functype(forward_entry),
                                   fb_functype(backward_entry),
                                   infer_functype(infer_shape_entry),
                                   list_functype(list_outputs_entry),
                                   list_functype(list_arguments_entry),
                                   deps_functype(declare_backward_dependency),
                                   None, None, None, None, None, None)
        cb_ptr = format(cast(pointer(self.info_), c_void_p).value, 'x')
        # pylint: disable=E1101
        sym = symbol.Symbol._NDArray(*args,
                                     info=cb_ptr,
                                     **kwargs)
        # keep a reference of ourself in PythonOp so we don't get garbage collected.
        PythonOp._ref_holder.append(self)
        return sym

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        """Declare dependencies of this operator for backward pass.

        Parameters
        ----------
        out_grad : list of int
            ids of out_grad blobs.
        in_data : list of int
            ids of in_data blobs.
        out_data: list of int
            ids of out_data blobs.

        Returns
        -------
        deps : list of int
            ids of the needed blobs.
        """
        deps = []
        if self.need_top_grad():
            deps.extend(out_grad)
        deps.extend(in_data)
        deps.extend(out_data)
        return deps
