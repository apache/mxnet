# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, no-self-use, too-many-locals, broad-except, too-many-lines, unnecessary-pass
"""numpy interface for operators."""
from __future__ import absolute_import

import traceback
import warnings

from array import array
from threading import Lock
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool

from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls

c_int_p = POINTER(c_int)

class PythonOp(object):
    """Base class for operators implemented in Python.

    Parameters
    ----------
    need_top_grad : bool
        the default need_top_grad() function returns this value.
    """
    _ref_holder = []

    def __init__(self, need_top_grad=True):
        self.info_ = None
        self.need_top_grad_ = need_top_grad
        warnings.warn('PythonOp has been deprecated. Please use CustomOp')

    def __call__(self, *args, **kwargs):
        return self.get_symbol(*args, **kwargs)

    def get_symbol(self, *args, **kwargs):
        """Create a symbol from numpy operator.
        This should only be called once per instance if the operator contains
        internal states.

        Parameters
        ----------
        args : list
            a list of input arguments (symbols).

        Returns
        -------
        sym : mxnet.symbol.Symbol
        """
        raise NotImplementedError("Must override this")

    def forward(self, in_data, out_data):
        """Forward interface. Override to create new operators.

        Parameters
        ----------
        in_data, out_data: list
            input and output for forward. See document for
            corresponding arguments of Operator::Forward
        """
        out_data[0][:] = in_data[0]

    def backward(self, out_grad, in_data, out_data, in_grad):
        """Backward interface. Can override when creating new operators.

        Parameters
        ----------
        out_grad, in_data, out_data, in_grad : list
            input and output for backward. See document for
            corresponding arguments of Operator::Backward
        """
        # pylint: disable=W0613
        in_grad[0][:] = 1.0

    def infer_shape(self, in_shape):
        """Interface for ``infer_shape``. Can override when creating new operators.

        Parameters
        ----------
        in_shape : list
            List of argument shapes in the same order as
            declared in list_arguments.

        Returns
        -------
        in_shape : list
            List of argument shapes. Can be modified from in_shape.
        out_shape : list
            List of output shapes calculated from in_shape,
            in the same order as declared in list_arguments.
        """
        return in_shape, [in_shape[0]]

    def list_outputs(self):
        """Interface for ``list_outputs``. Can override when creating new operators.

        Returns
        -------
        outputs : list
            List of output blob names.
        """
        return ['output']

    def list_arguments(self):
        """Interface for ``list_arguments``. Can override when creating new operators.

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
        warnings.warn('NumpyOp has been deprecated. Please use CustomOp')

    def get_symbol(self, *args, **kwargs):
        fb_functype = CFUNCTYPE(None, c_int, POINTER(POINTER(mx_float)), POINTER(c_int),
                                POINTER(POINTER(mx_uint)), POINTER(c_int), c_void_p)
        infer_functype = CFUNCTYPE(None, c_int, POINTER(c_int),
                                   POINTER(POINTER(mx_int)), c_void_p)
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
                tensor_shapes[i] = cast(c_array_buf(mx_int,
                                                    array('i', rshape[i])),
                                        POINTER(mx_int))
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
        sym = symbol._internal._Native(*args,
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
        warnings.warn('NDArrayOp has been deprecated. Please use CustomOp')

    def get_symbol(self, *args, **kwargs):
        fb_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_void_p), POINTER(c_int), c_void_p)
        infer_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_int),
                                   POINTER(POINTER(mx_int)), c_void_p)
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
            except Exception:
                print('Error in NDArrayOp.forward: %s' % traceback.format_exc())
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
            except Exception:
                print('Error in NDArrayOp.backward: %s' % traceback.format_exc())
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
                    tensor_shapes[i] = cast(c_array_buf(mx_int,
                                                        array('i', rshape[i])),
                                            POINTER(mx_int))
                    tensor_dims[i] = len(rshape[i])
            except Exception:
                print('Error in NDArrayOp.infer_shape: %s' % traceback.format_exc())
                return False
            return True

        def list_outputs_entry(out, _):
            """C Callback for NDArrayOpProp::ListOutputs"""
            try:
                ret = self.list_outputs()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception:
                print('Error in NDArrayOp.list_outputs: %s' % traceback.format_exc())
                return False
            return True

        def list_arguments_entry(out, _):
            """C Callback for NDArrayOpProp::ListArguments"""
            try:
                ret = self.list_arguments()
                ret = [c_str(i) for i in ret] + [c_char_p(0)]
                ret = c_array(c_char_p, ret)
                out[0] = cast(ret, POINTER(POINTER(c_char)))
            except Exception:
                print('Error in NDArrayOp.list_arguments: %s' % traceback.format_exc())
                return False
            return True

        def declare_backward_dependency(out_grad, in_data, out_data, num_dep, deps, _):
            """C Callback for NDArrayOpProp::DeclareBacwardDependency"""
            try:
                out_grad = [out_grad[i] for i in range(len(self.list_outputs()))]
                in_data = [in_data[i] for i in range(len(self.list_arguments()))]
                out_data = [out_data[i] for i in range(len(self.list_outputs()))]
                rdeps = self.declare_backward_dependency(out_grad, in_data, out_data)
                num_dep[0] = len(rdeps)
                rdeps = cast(c_array_buf(c_int, array('i', rdeps)), c_int_p)
                deps[0] = rdeps
            except Exception:
                print('Error in NDArrayOp.declare_backward_dependency: %s' % traceback.format_exc())
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
        sym = symbol._internal._NDArray(*args,
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

class CustomOp(object):
    """Base class for operators implemented in python"""
    def __init__(self):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        """Forward interface. Can override when creating new operators.

        Parameters
        ----------
        is_train : bool
            whether this is for training
        req : list of str
            how to assign to out_data. can be 'null', 'write', or 'add'.
            You can optionally use self.assign(dst, req, src) to handle this.
        in_data, out_data, aux: list of NDArrays
            input, output, and auxiliary states for forward. See document for
            corresponding arguments of Operator::Forward
        """
        # pylint: disable=W0613
        pass

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Backward interface. Can override when creating new operators.

        Parameters
        ----------
        req : list of str
            how to assign to in_grad. can be 'null', 'write', or 'add'.
            You can optionally use self.assign(dst, req, src) to handle this.
        out_grad, in_data, out_data, in_grad, aux : list of NDArrays
            input and output for backward. See document for
            corresponding arguments of Operator::Backward
        """
        # pylint: disable=W0613
        pass

    def assign(self, dst, req, src):
        """Helper function for assigning into dst depending on requirements."""
        if req == 'null':
            return
        elif req in ('write', 'inplace'):
            dst[:] = src
        elif req == 'add':
            dst[:] += src

class CustomOpProp(object):
    """Base class for operator property class implemented in python.

    Parameters
    ----------
    need_top_grad : bool
        The default declare_backward_dependency function. Use this value
        to determine whether this operator needs gradient input.
    """
    def __init__(self, need_top_grad=True):
        self.need_top_grad_ = need_top_grad

    def infer_shape(self, in_shape):
        """infer_shape interface. Can override when creating new operators.

        Parameters
        ----------
        in_shape : list
            List of argument shapes in the same order as
            declared in list_arguments.

        Returns
        -------
        in_shape : list
            List of argument shapes. Can be modified from in_shape.
        out_shape : list
            List of output shapes calculated from in_shape,
            in the same order as declared in list_outputs.
        aux_shape : Optional, list
            List of aux shapes calculated from in_shape,
            in the same order as declared in list_auxiliary_states.
        """
        return in_shape, (in_shape[0],)*len(self.list_outputs()), ()

    def infer_type(self, in_type):
        """infer_type interface. override to create new operators

        Parameters
        ----------
        in_type : list of np.dtype
            list of argument types in the same order as
            declared in list_arguments.

        Returns
        -------
        in_type : list
            list of argument types. Can be modified from in_type.
        out_type : list
            list of output types calculated from in_type,
            in the same order as declared in list_outputs.
        aux_type : Optional, list
            list of aux types calculated from in_type,
            in the same order as declared in list_auxiliary_states.
        """
        return in_type, [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def infer_storage_type(self, in_stype):
        """infer_storage_type interface. Used to infer storage type of
        inputs and outputs in the forward pass. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        in_stype : list of stypes, valid stypes are default, row_sparse and
            csr

        Returns
        -------
        in_stype : list
            list of argument stypes.
        out_stype : list
            list of output types calculated from in_stype,
            in the same order as declared in list_outputs.
        aux_type : Optional, list
            list of aux types calculated from in_stype,
            in the same order as declared in list_auxiliary_states.
        """
        for i, stype in enumerate(in_stype):
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], \
            "Default infer_storage_type implementation doesnt allow non default stypes: " \
            "found non default stype '%s' for in_stype[%d]. Please implement " \
            "infer_storage_type and infer_storage_type_backward interface " \
            "in your custom operator if you have non-default input/output stypes" % (stype, i)
        return in_stype, \
               [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]]*len(self.list_outputs()), \
               [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]]*len(self.list_auxiliary_states())

    def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, igrad_stype, aux_stype):
        """infer_storage_type_backward interface. Used to infer storage
        type of inputs and outputs in the backward pass.

        Will raise an error if undefined storage type is returned.
        Returned lists have to be the same size as the input lists to infer_storage_type_backward,
        otherwise an exception will be thrown. When this interface is not implemented,
        all stypes will be inferred as default.

        Parameters
        ----------
        ograd_stype : list
            list of output gradient storage types
        in_stype : list
            list of input storage types
        out_stype : list
            list of output storage types
        igrad_stype : list
            list of input gradient storage types
        aux_stype : list
            list of auxiliary storage types

        Returns
        -------
        ograd_stype : list
            list of inferred output gradient storage types
        in_stype : list
            list of inferred input storage types
        out_stype : list
            list of inferred output storage types
        igrad_stype : list
            list of inferred input gradient storage types
        aux_stype : list
            list of inferred storage types for auxiliary states
        """
        for i, stype in enumerate(ograd_stype):
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], \
            "Default infer_storage_type_backward implementation doesnt allow non default stypes: " \
             "found non default stype '%s' for ograd_stype[%d]. Please implement " \
             "infer_storage_type and infer_storage_type_backward interface " \
             "in your custom operator if you have non-default output gradient stypes" % (stype, i)
        for i, stype in enumerate(igrad_stype):
            if stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_UNDEFINED]:
                stype = _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]
            assert stype == _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], \
            "Default infer_storage_type_backward implementation doesnt allow non default stypes: " \
            "found non default stype '%s' for igrad_stype[%d]. Please implement " \
            "infer_storage_type and infer_storage_type_backward interface " \
            "in your custom operator if you have non-default input gradient stypes" % (stype, i)
        stype_lists = [ograd_stype, in_stype, out_stype, igrad_stype, aux_stype]
        for stype_list in stype_lists:
            stype_list[:] = len(stype_list) * [_STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT]]
        return stype_lists[0], stype_lists[1], stype_lists[2], stype_lists[3], stype_lists[4]

    def list_outputs(self):
        """list_outputs interface. Can override when creating new operators.

        Returns
        -------
        outputs : list
            List of output blob names.
        """
        return ['output']

    def list_arguments(self):
        """list_arguments interface. Can override when creating new operators.

        Returns
        -------
        arguments : list
            List of argument blob names.
        """
        return ['data']

    def list_auxiliary_states(self):
        """list_auxiliary_states interface. Can override when creating new operators.

        Returns
        -------
        auxs : list
            list of auxiliary state blob names.
        """
        return []

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
        if self.need_top_grad_:
            deps.extend(out_grad)
        deps.extend(in_data)
        deps.extend(out_data)
        return deps

    def create_operator(self, ctx, in_shapes, in_dtypes):
        """Create an operator that carries out the real computation
        given the context, input shapes, and input data types."""
        # pylint: disable=W0613
        return CustomOp()

class _Registry(object):
    """CustomOp registry."""
    def __init__(self):
        self.ref_holder = {}
        self.counter = 0
        self.result_deps = set()
        self.lock = Lock()

    def inc(self):
        """Get index for new entry."""
        self.lock.acquire()
        cur = self.counter
        self.counter += 1
        self.lock.release()
        return cur

_registry = _Registry()

def register(reg_name):
    """Register a subclass of CustomOpProp to the registry with name reg_name."""
    def do_register(prop_cls):
        """Register a subclass of CustomOpProp to the registry."""
        fb_functype = CFUNCTYPE(c_int, c_int, POINTER(c_void_p), POINTER(c_int),
                                POINTER(c_int), c_int, c_void_p)
        del_functype = CFUNCTYPE(c_int, c_void_p)

        infershape_functype = CFUNCTYPE(c_int, c_int, POINTER(c_int),
                                        POINTER(POINTER(mx_int)), c_void_p)
        infertype_functype = CFUNCTYPE(c_int, c_int, POINTER(c_int), c_void_p)
        inferstorage_functype = CFUNCTYPE(c_int, c_int, POINTER(c_int), c_void_p)
        inferstorage_backward_functype = CFUNCTYPE(c_int, c_int, POINTER(c_int), \
                                                   POINTER(c_int), c_void_p)
        list_functype = CFUNCTYPE(c_int, POINTER(POINTER(POINTER(c_char))), c_void_p)
        deps_functype = CFUNCTYPE(c_int, c_int_p, c_int_p, c_int_p,
                                  c_int_p, POINTER(c_int_p), c_void_p)
        createop_functype = CFUNCTYPE(c_int, c_char_p, c_int, POINTER(POINTER(mx_uint)),
                                      POINTER(c_int), POINTER(c_int),
                                      POINTER(MXCallbackList), c_void_p)
        req_enum = ('null', 'write', 'inplace', 'add')

        def creator(op_type, argc, keys, vals, ret):
            """internal function"""
            assert py_str(op_type) == reg_name
            kwargs = dict([(py_str(keys[i]), py_str(vals[i])) for i in range(argc)])
            op_prop = prop_cls(**kwargs)

            def infer_shape_entry(num_tensor, tensor_dims,
                                  tensor_shapes, _):
                """C Callback for ``CustomOpProp::InferShape``."""
                try:
                    n_in = len(op_prop.list_arguments())
                    n_out = len(op_prop.list_outputs())
                    n_aux = len(op_prop.list_auxiliary_states())
                    assert num_tensor == n_in + n_out + n_aux

                    shapes = [[tensor_shapes[i][j] for j in range(tensor_dims[i])]
                              for i in range(n_in)]
                    ret = op_prop.infer_shape(shapes)
                    if len(ret) == 2:
                        ishape, oshape = ret
                        ashape = []
                    elif len(ret) == 3:
                        ishape, oshape, ashape = ret
                    else:
                        raise AssertionError("infer_shape must return 2 or 3 lists")
                    assert len(oshape) == n_out, \
                        "InferShape Error: expecting %d entries in returned output " \
                        "shapes, got %d."%(n_out, len(oshape))
                    assert len(ishape) == n_in, \
                        "InferShape Error: expecting %d entries in returned input " \
                        "shapes, got %d."%(n_in, len(ishape))
                    assert len(ashape) == n_aux, \
                        "InferShape Error: expecting %d entries in returned aux state " \
                        "shapes, got %d."%(n_aux, len(ashape))
                    rshape = list(ishape) + list(oshape) + list(ashape)
                    for i in range(n_in+n_out+n_aux):
                        tensor_shapes[i] = cast(c_array_buf(mx_int,
                                                            array('i', rshape[i])),
                                                POINTER(mx_int))
                        tensor_dims[i] = len(rshape[i])

                    infer_shape_entry._ref_holder = [tensor_shapes]
                except Exception:
                    print('Error in %s.infer_shape: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True


            def infer_storage_type_backward_entry(num_tensor, tensor_stypes, tags, _):
                # pylint: disable=C0301
                """C Callback for CustomOpProp::InferStorageTypeBackward"""
                try:
                    tensors = [[] for i in range(5)]
                    for i in range(num_tensor):
                        tensors[tags[i]].append(_STORAGE_TYPE_ID_TO_STR[tensor_stypes[i]])
                    # Ordering of stypes: ograd, input, output, igrad, aux
                    tensors = [tensors[3], tensors[0], tensors[1], tensors[2], tensors[4]]
                    ret = op_prop.infer_storage_type_backward(tensors[0],
                                                              tensors[1],
                                                              tensors[2],
                                                              tensors[3],
                                                              tensors[4])
                    if len(ret) == 4:
                        ret += []
                    elif len(ret) == 5:
                        pass
                    else:
                        raise AssertionError("infer_storage_type_backward must return 4 or 5 lists")
                    assert len(ret[0]) == len(tensors[0]), \
                        "InferStorageTypeBackward Error: expecting == %d " \
                        "entries in returned output gradient " \
                        "stypes, got %d."%(len(tensors[0]), len(ret[0]))
                    assert len(ret[1]) == len(tensors[1]), \
                        "InferStorageTypeBackward Error: expecting == %d " \
                        "entries in returned input stypes, " \
                        "got %d."%(len(tensors[1]), len(ret[1]))
                    assert len(ret[2]) == len(tensors[2]), \
                        "InferStorageTypeBackward Error: expecting == %d " \
                        "entries in returned output stypes, " \
                        "got %d."%(len(tensors[2]), len(ret[2]))
                    assert len(ret[3]) == len(tensors[3]), \
                        "InferStorageTypeBackward Error: expecting == %d " \
                        "entries in returned input gradient stypes, " \
                        "got %d."%(len(tensors[3]), len(ret[3]))
                    assert len(ret[4]) == len(tensors[4]), \
                        "InferStorageTypeBackward Error: expecting == %d " \
                        "entries in returned aux stypes, " \
                        "got %d."%(len(tensors[4]), len(ret[4]))
                    rstype = []
                    for i, ret_list in enumerate(ret):
                        rstype.extend(ret_list)

                    for i, stype in enumerate(rstype):
                        assert stype != _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_UNDEFINED], \
                            "stype should not be undefined"
                        assert stype in _STORAGE_TYPE_STR_TO_ID, \
                            "Provided stype: %s is not valid " \
                            "valid stypes are %s, %s, %s"%(stype,
                                                           _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT],
                                                           _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_ROW_SPARSE],
                                                           _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_CSR])
                        tensor_stypes[i] = _STORAGE_TYPE_STR_TO_ID[stype]

                    infer_storage_type_backward_entry._ref_holder = [tensor_stypes]
                except Exception:
                    print('Error in %s.infer_type: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            def infer_storage_type_entry(num_tensor, tensor_stypes, _):
                """C Callback for CustomOpProp::InferStorageType"""
                try:
                    n_in = len(op_prop.list_arguments())
                    n_out = len(op_prop.list_outputs())
                    n_aux = len(op_prop.list_auxiliary_states())
                    assert num_tensor == n_in + n_out + n_aux

                    stypes = [_STORAGE_TYPE_ID_TO_STR[tensor_stypes[i]] for i in range(n_in)]
                    ret = op_prop.infer_storage_type(stypes)
                    if len(ret) == 2:
                        istype, ostype = ret
                        astype = []
                    elif len(ret) == 3:
                        istype, ostype, astype = ret
                    else:
                        raise AssertionError("infer_storage_type must return 2 or 3 lists")

                    assert len(ostype) == n_out, \
                        "InferStorageType Error: expecting %d entries in returned output " \
                        "stypes, got %d."%(n_out, len(ostype))
                    assert len(istype) == n_in, \
                        "InferStorageType Error: expecting %d entries in returned input " \
                        "stypes, got %d."%(n_in, len(istype))
                    assert len(astype) == n_aux, \
                        "InferStorageType Error: expecting %d entries in returned aux state " \
                        "stypes, got %d."%(n_aux, len(astype))
                    rtype = list(istype) + list(ostype) + list(astype)
                    for i, dtype in enumerate(rtype):
                        tensor_stypes[i] = _STORAGE_TYPE_STR_TO_ID[dtype]
                    infer_storage_type_entry._ref_holder = [tensor_stypes]
                except Exception:
                    print('Error in %s.infer_type: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            def infer_type_entry(num_tensor, tensor_types, _):
                """C Callback for CustomOpProp::InferType"""
                try:
                    n_in = len(op_prop.list_arguments())
                    n_out = len(op_prop.list_outputs())
                    n_aux = len(op_prop.list_auxiliary_states())
                    assert num_tensor == n_in + n_out + n_aux

                    types = [_DTYPE_MX_TO_NP[tensor_types[i]] for i in range(n_in)]
                    ret = op_prop.infer_type(types)
                    if len(ret) == 2:
                        itype, otype = ret
                        atype = []
                    elif len(ret) == 3:
                        itype, otype, atype = ret
                    else:
                        raise AssertionError("infer_type must return 2 or 3 lists")
                    assert len(otype) == n_out, \
                        "InferType Error: expecting %d entries in returned output " \
                        "types, got %d."%(n_out, len(otype))
                    assert len(itype) == n_in, \
                        "InferType Error: expecting %d entries in returned input " \
                        "types, got %d."%(n_in, len(itype))
                    assert len(atype) == n_aux, \
                        "InferType Error: expecting %d entries in returned aux state " \
                        "types, got %d."%(n_aux, len(atype))
                    rtype = list(itype) + list(otype) + list(atype)
                    for i, dtype in enumerate(rtype):
                        tensor_types[i] = _DTYPE_NP_TO_MX[dtype]

                    infer_type_entry._ref_holder = [tensor_types]
                except Exception:
                    print('Error in %s.infer_type: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            def list_outputs_entry(out, _):
                """C Callback for CustomOpProp::ListOutputs"""
                try:
                    ret = op_prop.list_outputs()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_outputs_entry._ref_holder = [out]
                except Exception:
                    print('Error in %s.list_outputs: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            def list_arguments_entry(out, _):
                """C Callback for CustomOpProp::ListArguments"""
                try:
                    ret = op_prop.list_arguments()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_arguments_entry._ref_holder = [out]
                except Exception:
                    print('Error in %s.list_arguments: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            def list_auxiliary_states_entry(out, _):
                """C Callback for CustomOpProp::ListAuxiliaryStates"""
                try:
                    ret = op_prop.list_auxiliary_states()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_auxiliary_states_entry._ref_holder = [out]
                except Exception:
                    tb = traceback.format_exc()
                    print('Error in %s.list_auxiliary_states: %s' % (reg_name, tb))
                    return False
                return True

            def declare_backward_dependency_entry(out_grad, in_data, out_data, num_dep, deps, _):
                """C Callback for CustomOpProp::DeclareBacwardDependency"""
                try:
                    out_grad = [out_grad[i] for i in range(len(op_prop.list_outputs()))]
                    in_data = [in_data[i] for i in range(len(op_prop.list_arguments()))]
                    out_data = [out_data[i] for i in range(len(op_prop.list_outputs()))]
                    rdeps = op_prop.declare_backward_dependency(out_grad, in_data, out_data)
                    num_dep[0] = len(rdeps)
                    _registry.result_deps = set()
                    for dep in rdeps:
                        _registry.result_deps.add(dep)
                    rdeps = cast(c_array_buf(c_int, array('i', rdeps)), c_int_p)
                    deps[0] = rdeps

                    declare_backward_dependency_entry._ref_holder = [deps]
                except Exception:
                    tb = traceback.format_exc()
                    print('Error in %s.declare_backward_dependency: %s' % (reg_name, tb))
                    return False
                return True

            def create_operator_entry(ctx, num_inputs, shapes, ndims, dtypes, ret, _):
                """C Callback for CustomOpProp::CreateOperator"""
                try:
                    ctx = py_str(ctx)
                    sep = ctx.find('(')
                    ctx = context.Context(ctx[:sep], int(ctx[sep+1:-1]))
                    ndims = [ndims[i] for i in range(num_inputs)]
                    shapes = [[shapes[i][j] for j in range(ndims[i])] for i in range(num_inputs)]
                    dtypes = [dtypes[i] for i in range(num_inputs)]
                    op = op_prop.create_operator(ctx, shapes, dtypes)

                    def forward_entry(num_ndarray, ndarraies, tags, reqs, is_train, _):
                        """C Callback for CustomOp::Forward"""
                        try:
                            tensors = [[] for i in range(5)]
                            for i in range(num_ndarray):
                                if tags[i] == 1 or tags[i] == 4:
                                    tensors[tags[i]].append(_ndarray_cls(cast(ndarraies[i],
                                                                              NDArrayHandle),
                                                                         writable=True))
                                else:
                                    tensors[tags[i]].append(_ndarray_cls(cast(ndarraies[i],
                                                                              NDArrayHandle),
                                                                         writable=False))
                            reqs = [req_enum[reqs[i]] for i in range(len(tensors[1]))]
                            with ctx:
                                op.forward(is_train=is_train, req=reqs,
                                           in_data=tensors[0], out_data=tensors[1],
                                           aux=tensors[4])
                        except Exception:
                            print('Error in CustomOp.forward: %s' % traceback.format_exc())
                            return False
                        return True

                    def backward_entry(num_ndarray, ndarraies, tags, reqs, is_train, _):
                        """C Callback for CustomOp::Backward"""
                        # pylint: disable=W0613
                        try:
                            tensors = [[] for i in range(5)]
                            num_outputs = len(op_prop.list_outputs())
                            num_args = len(op_prop.list_arguments())
                            for i in range(num_ndarray):
                                if i in _registry.result_deps or i >= (num_outputs * 2 + num_args):
                                    # If it is a backward dependency or output or aux:
                                    # Set stype as undefined so that it returns
                                    # ndarray based on existing stype
                                    stype = _STORAGE_TYPE_UNDEFINED
                                else:
                                    # If it is some input, output or out grad ndarray not part of
                                    # backward dependency it is empty and thus the ndarray should
                                    # be set to default
                                    stype = _STORAGE_TYPE_DEFAULT
                                if tags[i] == 2 or tags[i] == 4:
                                    tensors[tags[i]].append(_ndarray_cls(cast(ndarraies[i],
                                                                              NDArrayHandle),
                                                                         writable=True,
                                                                         stype=stype))
                                else:
                                    tensors[tags[i]].append(_ndarray_cls(cast(ndarraies[i],
                                                                              NDArrayHandle),
                                                                         writable=False,
                                                                         stype=stype))
                            reqs = [req_enum[reqs[i]] for i in range(len(tensors[2]))]
                            with ctx:
                                op.backward(req=reqs,
                                            in_data=tensors[0], out_data=tensors[1],
                                            in_grad=tensors[2], out_grad=tensors[3],
                                            aux=tensors[4])
                        except Exception:
                            print('Error in CustomOp.backward: %s' % traceback.format_exc())
                            return False
                        return True

                    cur = _registry.inc()

                    def delete_entry(_):
                        """C Callback for CustomOp::del"""
                        try:
                            del _registry.ref_holder[cur]
                        except Exception:
                            print('Error in CustomOp.delete: %s' % traceback.format_exc())
                            return False
                        return True

                    callbacks = [del_functype(delete_entry),
                                 fb_functype(forward_entry),
                                 fb_functype(backward_entry)]
                    callbacks = [cast(i, CFUNCTYPE(c_int)) for i in callbacks]
                    contexts = [None, None, None]
                    ret[0] = MXCallbackList(c_int(len(callbacks)),
                                            cast(c_array(CFUNCTYPE(c_int), callbacks),
                                                 POINTER(CFUNCTYPE(c_int))),
                                            cast(c_array(c_void_p, contexts),
                                                 POINTER(c_void_p)))
                    op._ref_holder = [ret]
                    _registry.ref_holder[cur] = op
                except Exception:
                    print('Error in %s.create_operator: %s' % (reg_name, traceback.format_exc()))
                    return False
                return True

            cur = _registry.inc()

            def delete_entry(_):
                """C Callback for CustomOpProp::del"""
                try:
                    del _registry.ref_holder[cur]
                except Exception:
                    print('Error in CustomOpProp.delete: %s' % traceback.format_exc())
                    return False
                return True

            callbacks = [del_functype(delete_entry),
                         list_functype(list_arguments_entry),
                         list_functype(list_outputs_entry),
                         list_functype(list_auxiliary_states_entry),
                         infershape_functype(infer_shape_entry),
                         deps_functype(declare_backward_dependency_entry),
                         createop_functype(create_operator_entry),
                         infertype_functype(infer_type_entry),
                         inferstorage_functype(infer_storage_type_entry),
                         inferstorage_backward_functype(infer_storage_type_backward_entry)]
            callbacks = [cast(i, CFUNCTYPE(c_int)) for i in callbacks]
            contexts = [None]*len(callbacks)
            ret[0] = MXCallbackList(c_int(len(callbacks)),
                                    cast(c_array(CFUNCTYPE(c_int), callbacks),
                                         POINTER(CFUNCTYPE(c_int))),
                                    cast(c_array(c_void_p, contexts),
                                         POINTER(c_void_p)))
            op_prop._ref_holder = [ret]
            _registry.ref_holder[cur] = op_prop
            return True

        creator_functype = CFUNCTYPE(c_int, c_char_p, c_int, POINTER(c_char_p),
                                     POINTER(c_char_p), POINTER(MXCallbackList))
        creator_func = creator_functype(creator)
        check_call(_LIB.MXCustomOpRegister(c_str(reg_name), creator_func))
        cur = _registry.inc()
        _registry.ref_holder[cur] = creator_func
        return prop_cls
    return do_register

register("custom_op")(CustomOpProp)
