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
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments
"""Symbolic Executor component of MXNet."""

import numpy as np
from . import ndarray

class Executor:
    """Executor is the object providing efficient symbolic and imperative graph
    execution and optimization.

    Examples
    --------
    >>> # typical approach to create an executor is to bind symbol
    >>> a = mx.sym.var('a')
    >>> b = mx.sym.var('b')
    >>> c = 2 * a + b
    >>> texec = c._bind(mx.cpu(), {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])})
    """
    def __init__(self, sym, device, args, args_grad, grad_req, aux_states, static_alloc=False):
        self.outputs = None
        self._input_names = sym.list_inputs()
        self._aux_names = sym.list_auxiliary_states()
        self._arg_names = sym.list_arguments()
        self._output_names = sym.list_outputs()
        self._device = device
        self._grad_req = grad_req
        self.static_alloc = static_alloc
        # grad_req
        self._requires_grad = False
        if isinstance(grad_req, dict):
            for k, v in grad_req.items():
                if k in self._input_names and v != 'null':
                    self._requires_grad = True
        else:
            assert isinstance(grad_req, str)
            self._requires_grad = grad_req != 'null'

        # args grad
        self._args_grad = args_grad
        if not self._args_grad:
            self._args_grad = None

        # args
        self._args = [None] * len(self._input_names)
        if isinstance(args, dict):
            for k, v in args.items():
                try:
                    i = self._input_names.index(k)
                    self._args[i] = v.copyto(device)
                # ignore provided arg which is not present in
                # input_names
                except ValueError:
                    pass
        else:
            assert isinstance(args, (list, tuple))
            for i, arg in enumerate(args):
                name = self._arg_names[i]
                index = self._input_names.index(name)
                self._args[index] = arg.copyto(device)

        # aux states
        if aux_states:
            if isinstance(aux_states, dict):
                for k, v in aux_states.items():
                    if k in self._aux_names:
                        i = self._input_names.index(k)
                        self._args[i] = v.copyto(device)
            else:
                assert isinstance(aux_states, (list, tuple))
                for i, v in enumerate(aux_states):
                    index = self._input_names.index(self._aux_names[i])
                    self._args[index] = v.copyto(device)

        # arg grad
        if self._args_grad:
            if isinstance(self._args_grad, dict):
                for k, g in self._args_grad.items():
                    try:
                        i = self._input_names.index(k)
                        # get req
                        if isinstance(grad_req, str):
                            req = grad_req
                        else:
                            assert isinstance(grad_req, dict)
                            req = grad_req[k]
                        if req != 'null':
                            with self._device:
                                self._args[i].attach_grad(req, stype=g.stype)
                                self._args[i].grad[:] = g
                    # ignore provided arg which is not present in
                    # input_names
                    except ValueError:
                        pass
            else:
                assert isinstance(self._args_grad, (list, tuple))
                for i, g in enumerate(self._args_grad):
                    # get req
                    if isinstance(grad_req, str):
                        req = grad_req
                    else:
                        assert isinstance(grad_req, dict)
                        req = grad_req[self._input_names[i]]
                    if req != 'null':
                        with self._device:
                            self._args[i].attach_grad(req, stype=g.stype)
                            self._args[i].grad[:] = g
        self._cached_op = ndarray.CachedOp(sym, flags=[("static_alloc", self.static_alloc)])

    def get_optimized_symbol(self):
        """Get an optimized version of the symbol from the executor.

        Returns
        -------
        symbol : Symbol
            Optimized symbol from the executor.
        """
        return self._cached_op.get_optimized_symbol()


    def forward(self, is_train=False, **kwargs):
        """Calculate the outputs specified by the bound symbol.

        Parameters
        ----------
        is_train: bool, optional
            Whether this forward is for evaluation purpose. If True,
            a backward call is expected to follow.

        **kwargs
            Additional specification of input arguments.

        Examples
        --------
        >>> # doing forward by specifying data
        >>> texec.forward(is_train=True, data=mydata)
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> mydata.copyto(texec.arg_dict['data'])
        >>> texec.forward(is_train=True)
        >>> # doing forward by specifying data and get outputs
        >>> outputs = texec.forward(is_train=True, data=mydata)
        >>> print(outputs[0].asnumpy())
        """
        if kwargs:
            for name, array in kwargs.items():
                if name in self._input_names:
                    index = self._input_names.index(name)
                    with self._device:
                        arr = ndarray.array(array, dtype=array.dtype)
                        if self._args[index] is None:
                            self._args[index] = arr
                            # get req
                            if isinstance(self._grad_req, str):
                                req = self._grad_req
                            else:
                                assert isinstance(self._grad_req, dict)
                                req = self._grad_req[name]
                            if req != 'null':
                                with self._device:
                                    self._args[index].attach_grad(req)
                        else:
                            self._args[index][:] = arr

        from . import autograd
        default_device = None if self._input_names else self._device
        with autograd.record(train_mode=is_train):
            self.outputs = self._cached_op(*self._args,
                                           default_device=default_device)
        if not isinstance(self.outputs, (list, tuple)):
            self.outputs = [self.outputs]
        return self.outputs

    def backward(self, out_grads=None):
        """Do backward pass to get the gradient of arguments.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray or dict of str to NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        is_train : bool, default True
            Whether this backward is for training or inference. Note that in rare
            cases you want to call backward with is_train=False to get gradient
            during inference.

        """
        from . import autograd
        if out_grads is not None:
            if not isinstance(out_grads, (list, tuple)):
                out_grads = [out_grads]
            out_grads = [o.copyto(self._device) for o in out_grads]

        if self._requires_grad:
            if self.outputs is None:
                self.forward()
            autograd.backward(self.outputs, head_grads=out_grads)

            if isinstance(self._args_grad, dict):
                for k, v in self._args_grad.items():
                    try:
                        i = self._input_names.index(k)
                        if self._args[i].grad is not None:
                            v[:] = self._args[i].grad
                    # ignore provided arg grad which is not present in
                    # input_names
                    except ValueError:
                        pass
            else:
                assert isinstance(self._args_grad, (list, tuple))
                for arg, out in zip(self._args, self._args_grad):
                    if arg.grad is not None:
                        out[:] = arg.grad

    @property
    def aux_arrays(self):
        """the auxilary argument array"""
        assert isinstance(self._args, list)
        aux_array = []
        for name in self._aux_names:
            index = self._input_names.index(name)
            aux_array.append(self._args[index])
        return aux_array

    @property
    def arg_arrays(self):
        """the argument array"""
        assert isinstance(self._args, list)
        arg_array = []
        for name in self._arg_names:
            index = self._input_names.index(name)
            arg_array.append(self._args[index])
        return arg_array

    @property
    def grad_arrays(self):
        """the gradient array"""
        if isinstance(self._args_grad, (list, tuple)):
            return list(self._args_grad)

        arr = [None] * len(self._arg_names)
        if self._args_grad:
            assert isinstance(self._args_grad, dict)
            for k, _ in self._args_grad.items():
                try:
                    i = self._input_names.index(k)
                    j = self._arg_names.index(k)
                    arr[j] = self._args[i].grad
                # ignore provided arg grad which is not present in
                # input_names
                except ValueError:
                    pass
        return arr

    @property
    def arg_dict(self):
        """Get dictionary representation of argument arrrays.

        Returns
        -------
        arg_dict : dict of str to NDArray
            The dictionary that maps the names of arguments to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
        """
        ret = {}
        for k, v in zip(self._input_names, self._args):
            if k in self._arg_names:
                ret[k] = v
        return ret

    @property
    def aux_dict(self):
        """Get dictionary representation of auxiliary states arrays.

        Returns
        -------
        aux_dict : dict of str to NDArray
            The dictionary that maps name of auxiliary states to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the auxiliary states.
        """
        ret = {}
        for k, v in zip(self._input_names, self._args):
            if k in self._aux_names:
                ret[k] = v
        return ret

    @property
    def grad_dict(self):
        """Get dictionary representation of gradient arrays.

        Returns
        -------
        grad_dict : dict of str to NDArray
            The dictionary that maps name of arguments to gradient arrays.
        """
        ret = {}
        for k, v in zip(self._input_names, self._args):
            if k in self._arg_names:
                ret[k] = v.grad
        return ret

    @property
    def output_dict(self):
        """Get dictionary representation of output arrays.

        Returns
        -------
        output_dict : dict of str to NDArray
            The dictionary that maps name of output names to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the outputs.
        """
        ret = {}
        for k, v in zip(self._output_names, self.outputs):
            ret[k] = v
        return ret

    def copy_params_from(self, arg_params, aux_params=None, allow_extra_params=False):
        """Copy parameters from arg_params, aux_params into executor's internal array.

        Parameters
        ----------
        arg_params : dict of str to NDArray
            Parameters, dict of name to NDArray of arguments.

        aux_params : dict of str to NDArray, optional
            Parameters, dict of name to NDArray of auxiliary states.

        allow_extra_params : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Raises
        ------
        ValueError
            If there is additional parameters in the dict but ``allow_extra_params=False``.

        Examples
        --------
        >>> # set parameters with existing model checkpoint
        >>> model_prefix = 'mx_mlp'
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
        >>> texec.copy_params_from(arg_params, aux_params)
        """
        for name, array in arg_params.items():
            if name in self.arg_dict:
                dst = self.arg_dict[name]
                array.astype(dst.dtype).copyto(dst)
            elif not allow_extra_params:
                raise ValueError(f'Find name \"{name}\" that is not in the arguments')

        if aux_params is None:
            return

        for name, array in aux_params.items():
            if name in self.aux_dict:
                dst = self.aux_dict[name]
                array.astype(dst.dtype).copyto(dst)
            elif not allow_extra_params:
                raise ValueError(f'Find name {name} that is not in the auxiliary states')
