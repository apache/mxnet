# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, no-self-use, too-many-locals, broad-except
"""numpy interface for operators."""
from __future__ import absolute_import

from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, cast, c_int, c_char, c_char_p, cast, c_bool
c_int_p = POINTER(c_int)
from .base import _LIB, check_call
from .base import c_array, c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
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
                out_grad = [out_grad[i] for i in range(len(self.list_outputs()))]
                in_data = [in_data[i] for i in range(len(self.list_arguments()))]
                out_data = [out_data[i] for i in range(len(self.list_outputs()))]
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

class CustomOp(object):
    """Base class for operators implemented in python"""
    def __init__(self):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        """forward interface. override to create new operators

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
        """backward interface. override to create new operators

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
        elif req == 'write' or req == 'inplace':
            dst[:] = src
        elif req == 'add':
            dst[:] += src

class CustomOpProp(object):
    """Base class for operator property class implemented in python.
    MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU

    Parameters
    ----------
    need_top_grad : bool
        The default declare_backward_dependency function use this value
        to determine whether this operator needs gradient input for above.
    """
    def __init__(self, need_top_grad=False):
        self.need_top_grad_ = need_top_grad

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
            in the same order as declared in list_outputs.
        aux_shape : Optional, list
            list of aux shapes calculated from in_shape,
            in the same order as declared in list_auxiliary_states.
        """
        return in_shape, [in_shape[0]], []

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
        arguments : list
            list of argument blob names.
        """
        return ['data']

    def list_auxiliary_states(self):
        """list_auxiliary_states interface. override to create new operators

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

_registry_ref_holder = []

def register(reg_name):
    """Register a subclass of CustomOpProp to the registry with name reg_name."""
    def do_register(prop_cls):
        """Register a subclass of CustomOpProp to the registry."""
        fb_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_void_p), POINTER(c_int),
                                POINTER(c_int), c_bool)
        class CustomOpInfo(Structure):
            """Structure that holds Callback information. Passed to CustomOpProp"""
            _fields_ = [
                ('forward', fb_functype),
                ('backward', fb_functype),
                ]

        infer_functype = CFUNCTYPE(c_bool, c_int, POINTER(c_int),
                                   POINTER(POINTER(mx_uint)))
        list_functype = CFUNCTYPE(c_bool, POINTER(POINTER(POINTER(c_char))))
        deps_functype = CFUNCTYPE(c_bool, c_int_p, c_int_p, c_int_p,
                                  c_int_p, POINTER(c_int_p))
        createop_functype = CFUNCTYPE(c_bool, c_char_p, c_int, POINTER(POINTER(mx_uint)),
                                      POINTER(c_int), POINTER(c_int), POINTER(CustomOpInfo))
        class CustomOpPropInfo(Structure):
            """Structure that holds Callback information. Passed to CustomOpProp"""
            _fields_ = [
                ('list_arguments', list_functype),
                ('list_outputs', list_functype),
                ('infer_shape', infer_functype),
                ('declare_backward_dependency', deps_functype),
                ('create_operator', createop_functype),
                ('list_auxiliary_states', list_functype)
                ]
        req_enum = ['null', 'write', 'inplace', 'add']

        def creator(op_type, argc, keys, vals, ret):
            """internal function"""
            assert py_str(op_type) == reg_name
            kwargs = dict([(keys[i], vals[i]) for i in range(argc)])
            op_prop = prop_cls(**kwargs)

            def infer_shape_entry(num_tensor, tensor_dims,
                                  tensor_shapes):
                """C Callback for CustomOpProp::InferShape"""
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
                    assert len(oshape) == n_out
                    assert len(ishape) == n_in
                    assert len(ashape) == n_aux
                    rshape = list(ishape) + list(oshape) + list(ashape)
                    for i in range(n_in+n_out+n_aux):
                        tensor_shapes[i] = cast(c_array(mx_uint, rshape[i]), POINTER(mx_uint))
                        tensor_dims[i] = len(rshape[i])

                    infer_shape_entry._ref_holder = [tensor_shapes]
                except Exception as e:
                    print('Error in %s.infer_shape: '%reg_name, str(e))
                    return False
                return True

            def list_outputs_entry(out):
                """C Callback for CustomOpProp::ListOutputs"""
                try:
                    ret = op_prop.list_outputs()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_outputs_entry._ref_holder = [out]
                except Exception as e:
                    print('Error in %s.list_outputs: '%reg_name, str(e))
                    return False
                return True

            def list_arguments_entry(out):
                """C Callback for CustomOpProp::ListArguments"""
                try:
                    ret = op_prop.list_arguments()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_arguments_entry._ref_holder = [out]
                except Exception as e:
                    print('Error in %s.list_arguments: '%reg_name, str(e))
                    return False
                return True

            def list_auxiliary_states_entry(out):
                """C Callback for CustomOpProp::ListAuxiliaryStates"""
                try:
                    ret = op_prop.list_auxiliary_states()
                    ret = [c_str(i) for i in ret] + [c_char_p(0)]
                    ret = c_array(c_char_p, ret)
                    out[0] = cast(ret, POINTER(POINTER(c_char)))

                    list_auxiliary_states_entry._ref_holder = [out]
                except Exception as e:
                    print('Error in %s.list_auxiliary_states: '%reg_name, str(e))
                    return False
                return True

            def declare_backward_dependency_entry(out_grad, in_data, out_data, num_dep, deps):
                """C Callback for CustomOpProp::DeclareBacwardDependency"""
                try:
                    out_grad = [out_grad[i] for i in range(len(op_prop.list_outputs()))]
                    in_data = [in_data[i] for i in range(len(op_prop.list_arguments()))]
                    out_data = [out_data[i] for i in range(len(op_prop.list_outputs()))]
                    rdeps = op_prop.declare_backward_dependency(out_grad, in_data, out_data)
                    num_dep[0] = len(rdeps)
                    rdeps = cast(c_array(c_int, rdeps), c_int_p)
                    deps[0] = rdeps

                    declare_backward_dependency_entry._ref_holder = [deps]
                except Exception as e:
                    print('Error in %s.declare_backward_dependency: '%reg_name, str(e))
                    return False
                return True

            def create_operator_entry(ctx, num_inputs, shapes, ndims, dtypes, ret):
                """C Callback for CustomOpProp::CreateOperator"""
                try:
                    ndims = [ndims[i] for i in range(num_inputs)]
                    shapes = [[shapes[i][j] for j in range(ndims[i])] for i in range(num_inputs)]
                    dtypes = [dtypes[i] for i in range(num_inputs)]
                    op = op_prop.create_operator(ctx, shapes, dtypes)

                    def forward_entry(num_ndarray, ndarraies, tags, reqs, is_train):
                        """C Callback for CustomOp::Forward"""
                        try:
                            tensors = [[] for i in range(5)]
                            for i in range(num_ndarray):
                                if tags[i] == 1 or tags[i] == 4:
                                    tensors[tags[i]].append(NDArray(cast(ndarraies[i],
                                                                         NDArrayHandle),
                                                                    writable=True))
                                else:
                                    tensors[tags[i]].append(NDArray(cast(ndarraies[i],
                                                                         NDArrayHandle),
                                                                    writable=False))
                            reqs = [req_enum[reqs[i]] for i in range(len(tensors[1]))]
                            op.forward(is_train=is_train, req=reqs,
                                       in_data=tensors[0], out_data=tensors[1],
                                       aux=tensors[4])
                        except Exception as e:
                            print('Error in CustomOp.forward: ', str(e))
                            return False
                        return True

                    def backward_entry(num_ndarray, ndarraies, tags, reqs, is_train):
                        """C Callback for CustomOp::Backward"""
                        # pylint: disable=W0613
                        try:
                            tensors = [[] for i in range(5)]
                            for i in range(num_ndarray):
                                if tags[i] == 2 or tags[i] == 4:
                                    tensors[tags[i]].append(NDArray(cast(ndarraies[i],
                                                                         NDArrayHandle),
                                                                    writable=True))
                                else:
                                    tensors[tags[i]].append(NDArray(cast(ndarraies[i],
                                                                         NDArrayHandle),
                                                                    writable=False))
                            reqs = [req_enum[reqs[i]] for i in range(len(tensors[2]))]
                            op.backward(req=reqs,
                                        in_data=tensors[0], out_data=tensors[1],
                                        in_grad=tensors[2], out_grad=tensors[3],
                                        aux=tensors[4])
                        except Exception as e:
                            print('Error in CustomOp.backward: ', str(e))
                            return False
                        return True

                    ret[0] = CustomOpInfo(fb_functype(forward_entry), fb_functype(backward_entry))
                    op._ref_holder = [ret]
                    op_prop._ref_holder.append(op)
                except Exception as e:
                    print('Error in %s.create_operator: '%reg_name, str(e))
                    return False
                return True

            ret[0] = CustomOpPropInfo(list_functype(list_arguments_entry),
                                      list_functype(list_outputs_entry),
                                      infer_functype(infer_shape_entry),
                                      deps_functype(declare_backward_dependency_entry),
                                      createop_functype(create_operator_entry),
                                      list_functype(list_auxiliary_states_entry))
            op_prop._ref_holder = [ret]
            _registry_ref_holder.append(op_prop)
            return True

        creator_functype = CFUNCTYPE(c_bool, c_char_p, c_int, POINTER(c_char_p),
                                     POINTER(c_char_p), POINTER(CustomOpPropInfo))
        creator_func = creator_functype(creator)
        check_call(_LIB.MXCustomOpRegister(c_str(reg_name), creator_func))
        _registry_ref_holder.append(creator_func)
        return prop_cls
    return do_register

register("custom_op")(CustomOpProp)
