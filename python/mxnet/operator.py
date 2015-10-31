# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, no-self-use
"""numpy interface for operators."""
from __future__ import absolute_import

from ctypes import CFUNCTYPE, POINTER, Structure, pointer, c_void_p, cast, c_int, c_char, c_char_p
from .base import c_array, c_str, mx_uint, mx_float, ctypes2numpy_shared
from . import symbol

class NumpyOp(object):
    """Base class for numpy operators. numpy operators allow parts
    of computation in symbolic graph to be writen in numpy. This feature
    is intended for quickly hacking out a solution for non performance
    critical parts. Please consider write a c++ implementation if it becomes
    a bottleneck.
    """
    def __init__(self):
        self.info_ = None

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
        fb_functype = CFUNCTYPE(None, c_int, POINTER(POINTER(mx_float)), POINTER(c_int),
                                POINTER(POINTER(mx_uint)), POINTER(c_int))
        infer_functype = CFUNCTYPE(None, c_int, POINTER(c_int), POINTER(POINTER(mx_uint)))
        list_functype = CFUNCTYPE(None, POINTER(POINTER(POINTER(c_char))))
        class NumpyOpInfo(Structure):
            """Structure that holds Callback information. Passed to NumpyOpProp"""
            _fields_ = [
                ('forward', fb_functype),
                ('backward', fb_functype),
                ('infer_shape', infer_functype),
                ('list_outputs', list_functype),
                ('list_arguments', list_functype)
                ]
        def forward_entry(num_tensor, tensor_ptrs, tensor_dims,
                          tensor_shapes, tensor_tags):
            """C Callback for NumpyOp::Forward"""
            tensors = [[] for i in range(4)]
            for i in range(num_tensor):
                shape = [tensor_shapes[i][j] for j in range(tensor_dims[i])]
                buff = ctypes2numpy_shared(tensor_ptrs[i], shape)
                tensors[tensor_tags[i]].append(buff)
            self.forward(in_data=tensors[0], out_data=tensors[1])

        def backward_entry(num_tensor, tensor_ptrs, tensor_dims,
                           tensor_shapes, tensor_tags):
            """C Callback for NumpyOp::Backward"""
            tensors = [[] for i in range(4)]
            for i in range(num_tensor):
                shape = [tensor_shapes[i][j] for j in range(tensor_dims[i])]
                buff = ctypes2numpy_shared(tensor_ptrs[i], shape)
                tensors[tensor_tags[i]].append(buff)
            self.backward(in_data=tensors[0], out_data=tensors[1],
                          in_grad=tensors[2], out_grad=tensors[3])

        def infer_shape_entry(num_tensor, tensor_dims,
                              tensor_shapes):
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

        def list_outputs_entry(out):
            """C Callback for NumpyOpProp::ListOutputs"""
            ret = self.list_outputs()
            ret = [c_str(i) for i in ret] + [c_char_p(0)]
            ret = c_array(c_char_p, ret)
            out[0] = cast(ret, POINTER(POINTER(c_char)))

        def list_arguments_entry(out):
            """C Callback for NumpyOpProp::ListArguments"""
            ret = self.list_arguments()
            ret = [c_str(i) for i in ret] + [c_char_p(0)]
            ret = c_array(c_char_p, ret)
            out[0] = cast(ret, POINTER(POINTER(c_char)))


        self.info_ = NumpyOpInfo(fb_functype(forward_entry),
                                 fb_functype(backward_entry),
                                 infer_functype(infer_shape_entry),
                                 list_functype(list_outputs_entry),
                                 list_functype(list_arguments_entry))
        cb_ptr = hex(cast(pointer(self.info_), c_void_p).value)
        # pylint: disable=E1101
        return symbol.Symbol._Native(*args,
                                     info=cb_ptr,
                                     need_top_grad=self.need_top_grad(),
                                     **kwargs)

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
        return True
