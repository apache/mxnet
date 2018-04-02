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

# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-arguments
"""Executor group is a convenient tool for managing a group of executors."""

import logging
from collections import OrderedDict

from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice


def _load_general(data, targets, major_axis):
    """Load a list of arrays into a list of arrays specified by slices."""
    for d_src, d_targets, axis in zip(data, targets, major_axis): # pylint: disable=too-many-nested-blocks
        if isinstance(d_targets, nd.NDArray):
            d_src.copyto(d_targets)
        elif isinstance(d_src, (list, tuple)):
            for src, dst in zip(d_src, d_targets):
                src.copyto(dst)
        else:
            for slice_idx, d_dst in d_targets:
                if axis >= 0:
                    # copy slice
                    shape = d_src.shape
                    do_crop = (slice_idx.start != 0 or shape[axis] != slice_idx.stop)
                    # pylint: disable=no-member,protected-access
                    if do_crop:
                        if axis == 0:
                            d_src[slice_idx.start:slice_idx.stop].copyto(d_dst)
                        else:
                            if d_src.context == d_dst.context:
                                nd.slice_axis(d_src, axis=axis, begin=slice_idx.start,
                                              end=slice_idx.stop, out=d_dst)
                            else:
                                # on different device, crop and then do cross device copy
                                d_dst_copy = nd.slice_axis(d_src, axis=axis, begin=slice_idx.start,
                                                           end=slice_idx.stop)
                                d_dst_copy.copyto(d_dst)
                    else:
                        d_src.copyto(d_dst)
                    # pylint: enable=no-member,protected-access
                else:
                    d_src.copyto(d_dst)


def _load_data(batch, targets, major_axis):
    """Load data into sliced arrays."""
    _load_general(batch.data, targets, major_axis)


def _load_label(batch, targets, major_axis):
    """Load label into sliced arrays."""
    _load_general(batch.label, targets, major_axis)


def _merge_multi_context(outputs, major_axis):
    """Merge outputs that lives on multiple context into one, so that they look
    like living on one context.
    """
    rets = []
    for tensors, axis in zip(outputs, major_axis):
        if axis >= 0:
            # pylint: disable=no-member,protected-access
            if len(tensors) == 1:
                rets.append(tensors[0])
            else:
                # Concatenate if necessary
                rets.append(nd.concat(*[tensor.as_in_context(tensors[0].context)
                                        for tensor in tensors],
                                      dim=axis))
            # pylint: enable=no-member,protected-access
        else:
            # negative axis means the there is no batch_size axis, and all the
            # results should be the same on each device. We simply take the
            # first one, without checking they are actually the same
            rets.append(tensors[0])
    return rets

def _prepare_group2ctxs(group2ctxs, ctx_len):
    """Prepare the group2contexts, will duplicate the context
    if some ctx_group map to only one context.
    """
    if group2ctxs is None:
        return [None] * ctx_len
    elif isinstance(group2ctxs, list):
        assert(len(group2ctxs) == ctx_len), "length of group2ctxs\
            should be %d" % ctx_len
        return group2ctxs
    elif isinstance(group2ctxs, dict):
        ret = [{} for i in range(ctx_len)]
        for k, v in group2ctxs.items():
            ctxs = None
            if isinstance(v, ctx.Context):
                ctxs = [v] * ctx_len
            else:
                if len(v) == 1:
                    ctxs = v * ctx_len
                else:
                    assert(len(v) == ctx_len), "length of group2ctxs[%s]\
                        should be %d or 1" % (k, ctx_len)
                    ctxs = v
            for i in range(ctx_len):
                ret[i][k] = ctxs[i]
        return ret
    else:
        assert(False), "group2ctxs should be list of dict of str to context,\
            or dict of str to context or list of context"
        return False

class DataParallelExecutorGroup(object):
    """A group of executors that lives on a group of devices.
    This is a helper class used to implement data parallelization. Each mini-batch will
    be split and run on the devices.

    Parameters
    ----------
    symbol : Symbol
        The common symbolic computation graph for all executors.
    contexts : list
        A list of contexts.
    workload : list
        If not ``None``, could be a list of numbers that specify the workload to be assigned
        to different context. Larger number indicate heavier workload.
    data_shapes : list
        Should be a list of (name, shape) tuples, for the shapes of data. Note the order is
        important and should be the same as the order that the `DataIter` provide the data.
    label_shapes : list
        Should be a list of (name, shape) tuples, for the shapes of label. Note the order is
        important and should be the same as the order that the `DataIter` provide the label.
    param_names : list
        A list of strings, indicating the names of parameters (e.g. weights, filters, etc.)
        in the computation graph.
    for_training : bool
        Indicate whether the executors should be bind for training. When not doing training,
        the memory for gradients will not be allocated.
    inputs_need_grad : bool
        Indicate whether the gradients for the input data should be computed. This is currently
        not used. It will be useful for implementing composition of modules.
    shared_group : DataParallelExecutorGroup
        Defaults to ``None``. This is used in bucketing. When not ``None``, it should be a executor
        group corresponding to a different bucket. In other words, it will correspond to a different
        symbol with the same set of parameters (e.g. unrolled RNNs with different lengths).
        In this case the memory regions of the parameters will be shared.
    logger : Logger
        Default is `logging`.
    fixed_param_names: list of str
        Parameters to be fixed during training. For these parameters, not gradients
        will be calculated and thus no space will be allocated for the gradient.
    grad_req : str, list of str, dict of str to str
        Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
        (default to 'write').
        Can be specified globally (str) or for each argument (list, dict).
    group2ctxs : dict of str to context or list of context,
                 or list of dict of str to context
        Default is `None`. Mapping the `ctx_group` attribute to the context assignment.
    """
    def __init__(self, symbol, contexts, workload, data_shapes, label_shapes, param_names,
                 for_training, inputs_need_grad, shared_group=None, logger=logging,
                 fixed_param_names=None, grad_req='write', state_names=None, group2ctxs=None):
        self.param_names = param_names
        self.arg_names = symbol.list_arguments()
        self.aux_names = symbol.list_auxiliary_states()

        self.symbol = symbol
        self.contexts = contexts
        self.workload = workload
        self.group2ctxs = _prepare_group2ctxs(group2ctxs, len(contexts))

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad

        self.logger = logger
        #In the future we should have a better way to profile memory per device (haibin)
        self._total_exec_bytes = 0
        self.fixed_param_names = fixed_param_names
        if self.fixed_param_names is None:
            self.fixed_param_names = []

        self.state_names = state_names
        if self.state_names is None:
            self.state_names = []

        if not for_training:
            grad_req = 'null'

        data_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
        if label_shapes is not None:
            label_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]

        data_names = [x.name for x in data_shapes]

        if isinstance(grad_req, str):
            self.grad_req = {}
            for k in self.arg_names:
                if k in self.param_names:
                    self.grad_req[k] = 'null' if k in self.fixed_param_names else grad_req
                elif k in data_names:
                    self.grad_req[k] = grad_req if self.inputs_need_grad else 'null'
                else:
                    self.grad_req[k] = 'null'
        elif isinstance(grad_req, (list, tuple)):
            assert len(grad_req) == len(self.arg_names)
            self.grad_req = dict(zip(self.arg_names, grad_req))
        elif isinstance(grad_req, dict):
            self.grad_req = {}
            for k in self.arg_names:
                if k in self.param_names:
                    self.grad_req[k] = 'null' if k in self.fixed_param_names else 'write'
                elif k in data_names:
                    self.grad_req[k] = 'write' if self.inputs_need_grad else 'null'
                else:
                    self.grad_req[k] = 'null'
            self.grad_req.update(grad_req)
        else:
            raise ValueError("grad_req must be one of str, list, tuple, or dict.")

        if shared_group is not None:
            self.shared_data_arrays = shared_group.shared_data_arrays
        else:
            self.shared_data_arrays = [{} for _ in contexts]

        # initialize some instance variables
        self.batch_size = None
        self.slices = None
        self.execs = []
        self._default_execs = None
        self.data_arrays = None
        self.label_arrays = None
        self.param_arrays = None
        self.state_arrays = None
        self.grad_arrays = None
        self.aux_arrays = None
        self.input_grad_arrays = None

        self.data_shapes = None
        self.label_shapes = None
        self.data_names = None
        self.label_names = None
        self.data_layouts = None
        self.label_layouts = None
        self.output_names = self.symbol.list_outputs()
        self.output_layouts = [DataDesc.get_batch_axis(self.symbol[name].attr('__layout__'))
                               for name in self.output_names]
        self.num_outputs = len(self.symbol.list_outputs())

        self.bind_exec(data_shapes, label_shapes, shared_group)

    def decide_slices(self, data_shapes):
        """Decide the slices for each context according to the workload.

        Parameters
        ----------
        data_shapes : list
            list of (name, shape) specifying the shapes for the input data or label.
        """
        assert len(data_shapes) > 0
        major_axis = [DataDesc.get_batch_axis(x.layout) for x in data_shapes]

        for (name, shape), axis in zip(data_shapes, major_axis):
            if axis == -1:
                continue

            batch_size = shape[axis]
            if self.batch_size is not None:
                assert batch_size == self.batch_size, ("all data must have the same batch size: "
                                                       + ("batch_size = %d, but " % self.batch_size)
                                                       + ("%s has shape %s" % (name, shape)))
            else:
                self.batch_size = batch_size
                self.slices = _split_input_slice(self.batch_size, self.workload)

        return major_axis

    def _collect_arrays(self):
        """Collect internal arrays from executors."""
        # convenient data structures
        self.data_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)]
                            for name, _ in self.data_shapes]

        self.state_arrays = [[e.arg_dict[name] for e in self.execs]
                             for name in self.state_names]

        if self.label_shapes is not None:
            self.label_arrays = [[(self.slices[i], e.arg_dict[name])
                                  for i, e in enumerate(self.execs)]
                                 for name, _ in self.label_shapes]
        else:
            self.label_arrays = None

        self.param_arrays = [[exec_.arg_arrays[i] for exec_ in self.execs]
                             for i, name in enumerate(self.arg_names)
                             if name in self.param_names]
        if self.for_training:
            self.grad_arrays = [[exec_.grad_arrays[i] for exec_ in self.execs]
                                for i, name in enumerate(self.arg_names)
                                if name in self.param_names]
        else:
            self.grad_arrays = None

        data_names = [x[0] for x in self.data_shapes]
        if self.inputs_need_grad:
            self.input_grad_arrays = [[exec_.grad_arrays[self.arg_names.index(name)]
                                       for exec_ in self.execs]
                                      for name in data_names if name in self.arg_names]
        else:
            self.input_grad_arrays = None

        self.aux_arrays = [[exec_.aux_arrays[i] for exec_ in self.execs]
                           for i in range(len(self.aux_names))]

    def bind_exec(self, data_shapes, label_shapes, shared_group=None, reshape=False):
        """Bind executors on their respective devices.

        Parameters
        ----------
        data_shapes : list
        label_shapes : list
        shared_group : DataParallelExecutorGroup
        reshape : bool
        """
        assert reshape or not self.execs
        self.batch_size = None

        # calculate workload and bind executors
        self.data_layouts = self.decide_slices(data_shapes)
        if label_shapes is not None:
            # call it to make sure labels has the same batch size as data
            self.label_layouts = self.decide_slices(label_shapes)

        for i in range(len(self.contexts)):
            data_shapes_i = self._sliced_shape(data_shapes, i, self.data_layouts)
            if label_shapes is not None:
                label_shapes_i = self._sliced_shape(label_shapes, i, self.label_layouts)
            else:
                label_shapes_i = []

            if reshape:
                self.execs[i] = self._default_execs[i].reshape(
                    allow_up_sizing=True, **dict(data_shapes_i + label_shapes_i))
            else:
                self.execs.append(self._bind_ith_exec(i, data_shapes_i, label_shapes_i,
                                                      shared_group))

        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.data_names = [i.name for i in self.data_shapes]
        if label_shapes is not None:
            self.label_names = [i.name for i in self.label_shapes]
        self._collect_arrays()

    def reshape(self, data_shapes, label_shapes):
        """Reshape executors.

        Parameters
        ----------
        data_shapes : list
        label_shapes : list
        """
        if data_shapes == self.data_shapes and label_shapes == self.label_shapes:
            return
        if self._default_execs is None:
            self._default_execs = [i for i in self.execs]
        self.bind_exec(data_shapes, label_shapes, reshape=True)

    def set_params(self, arg_params, aux_params, allow_extra=False):
        """Assign, i.e. copy parameters to all the executors.

        Parameters
        ----------
        arg_params : dict
            A dictionary of name to `NDArray` parameter mapping.
        aux_params : dict
            A dictionary of name to `NDArray` auxiliary variable mapping.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        for exec_ in self.execs:
            exec_.copy_params_from(arg_params, aux_params, allow_extra_params=allow_extra)

    def get_params(self, arg_params, aux_params):
        """ Copy data from each executor to `arg_params` and `aux_params`.

        Parameters
        ----------
        arg_params : list of NDArray
            Target parameter arrays.
        aux_params : list of NDArray
            Target aux arrays.

        Notes
        -----
        - This function will inplace update the NDArrays in arg_params and aux_params.
        """
        for name, block in zip(self.param_names, self.param_arrays):
            weight = sum(w.copyto(ctx.cpu()) for w in block) / len(block)
            weight.astype(arg_params[name].dtype).copyto(arg_params[name])
        for name, block in zip(self.aux_names, self.aux_arrays):
            weight = sum(w.copyto(ctx.cpu()) for w in block) / len(block)
            weight.astype(aux_params[name].dtype).copyto(aux_params[name])

    def forward(self, data_batch, is_train=None):
        """Split `data_batch` according to workload and run forward on each devices.

        Parameters
        ----------
        data_batch : DataBatch
            Or could be any object implementing similar interface.
        is_train : bool
            The hint for the backend, indicating whether we are during training phase.
            Default is `None`, then the value `self.for_training` will be used.
        Returns
        -------

        """
        _load_data(data_batch, self.data_arrays, self.data_layouts)
        if is_train is None:
            is_train = self.for_training

        if self.label_arrays is not None and data_batch.label:
            _load_label(data_batch, self.label_arrays, self.label_layouts)

        for exec_ in self.execs:
            exec_.forward(is_train=is_train)

    def get_output_shapes(self):
        """Get the shapes of the outputs."""
        outputs = self.execs[0].outputs
        shapes = [out.shape for out in outputs]

        concat_shapes = []
        for key, the_shape, axis in zip(self.symbol.list_outputs(), shapes, self.output_layouts):
            the_shape = list(the_shape)
            if axis >= 0:
                the_shape[axis] = self.batch_size
            concat_shapes.append((key, tuple(the_shape)))
        return concat_shapes

    def get_outputs(self, merge_multi_context=True, begin=0, end=None):
        """Get outputs of the previous forward computation.
        If begin or end is specified, return [begin, end)-th outputs,
        otherwise return all outputs.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.
        begin : int
            starting index of returned outputs in all outputs
        end : int or None
            ending index (excluded) of returned outputs.

        Returns
        -------
        If `merge_multi_context` is ``True``, it is like ``[out1, out2]``. Otherwise, it
        is like ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``. All the output
        elements are `NDArray`.
        """
        if end is None:
            end = self.num_outputs
        outputs = [[exec_.outputs[i] for exec_ in self.execs]
                   for i in range(begin, end)]
        if merge_multi_context:
            outputs = _merge_multi_context(outputs, self.output_layouts)
        return outputs

    def get_states(self, merge_multi_context=True):
        """Get states from all devices.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is ``True``, it is like ``[out1, out2]``. Otherwise, it
        is like ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``. All the output
        elements are `NDArray`.
        """
        assert not merge_multi_context, \
            "merge_multi_context=True is not supported for get_states yet."
        return self.state_arrays

    def set_states(self, states=None, value=None):
        """Set value for states. Only one of states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArrays
            source states arrays formatted like [[state1_dev1, state1_dev2],
            [state2_dev1, state2_dev2]].
        value : number
            a single scalar value for all state arrays.
        """
        if states is not None:
            assert value is None, "Only one of states & value can be specified."
            _load_general(states, self.state_arrays, (0,)*len(states))
        else:
            assert value is not None, "At least one of states & value must be specified."
            assert states is None, "Only one of states & value can be specified."
            for d_dst in self.state_arrays:
                for dst in d_dst:
                    dst[:] = value

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Defaults to ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
        is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
        elements are `NDArray`.
        """
        assert self.inputs_need_grad
        if merge_multi_context:
            return _merge_multi_context(self.input_grad_arrays, self.data_layouts)
        return self.input_grad_arrays

    def backward(self, out_grads=None):
        """Run backward on all devices. A backward should be called after
        a call to the forward function. Backward cannot be called unless
        ``self.for_training`` is ``True``.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.for_training, 're-bind with for_training=True to run backward'
        if out_grads is None:
            out_grads = []

        for i, (exec_, islice) in enumerate(zip(self.execs, self.slices)):
            out_grads_slice = []
            for grad, axis in zip(out_grads, self.output_layouts):
                if axis >= 0:
                    # pylint: disable=no-member
                    og_my_slice = nd.slice_axis(grad, axis=axis, begin=islice.start,
                                                end=islice.stop)
                    # pylint: enable=no-member
                    out_grads_slice.append(og_my_slice.as_in_context(self.contexts[i]))
                else:
                    out_grads_slice.append(grad.copyto(self.contexts[i]))
            exec_.backward(out_grads=out_grads_slice)

    def update_metric(self, eval_metric, labels):
        """Accumulate the performance according to `eval_metric` on all devices
        by comparing outputs from [begin, end) to labels. By default use all
        outputs.

        Parameters
        ----------
        eval_metric : EvalMetric
            The metric used for evaluation.
        labels : list of NDArray
            Typically comes from `label` of a `DataBatch`.
        begin : int
            Starting index of used outputs.
        end : int or None
            Ending index of used outputs.
        """
        for texec, islice in zip(self.execs, self.slices):
            labels_slice = []
            for label, axis in zip(labels, self.label_layouts):
                if axis == 0:
                    # slicing NDArray along axis 0 can avoid copying
                    labels_slice.append(label[islice])
                elif axis > 0:
                    # pylint: disable=no-member
                    label_my_slice = nd.slice_axis(label, axis=axis, begin=islice.start,
                                                   end=islice.stop).as_in_context(label.context)
                    # pylint: enable=no-member
                    labels_slice.append(label_my_slice)
                else:
                    labels_slice.append(label)

            labels_ = OrderedDict(zip(self.label_names, labels_slice))
            preds = OrderedDict(zip(self.output_names, texec.outputs))
            eval_metric.update_dict(labels_, preds)

    def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
        """Internal utility function to bind the i-th executor.
        This function utilizes simple_bind python interface.
        """
        shared_exec = None if shared_group is None else shared_group.execs[i]
        context = self.contexts[i]
        shared_data_arrays = self.shared_data_arrays[i]

        input_shapes = dict(data_shapes)
        if label_shapes is not None:
            input_shapes.update(dict(label_shapes))

        input_types = {x.name: x.dtype for x in data_shapes}
        if label_shapes is not None:
            input_types.update({x.name: x.dtype for x in label_shapes})

        group2ctx = self.group2ctxs[i]

        executor = self.symbol.simple_bind(ctx=context, grad_req=self.grad_req,
                                           type_dict=input_types, shared_arg_names=self.param_names,
                                           shared_exec=shared_exec, group2ctx=group2ctx,
                                           shared_buffer=shared_data_arrays, **input_shapes)
        self._total_exec_bytes += int(executor.debug_str().split('\n')[-3].split()[1])
        return executor

    def _sliced_shape(self, shapes, i, major_axis):
        """Get the sliced shapes for the i-th executor.

        Parameters
        ----------
        shapes : list of (str, tuple)
            The original (name, shape) pairs.
        i : int
            Which executor we are dealing with.
        """
        sliced_shapes = []
        for desc, axis in zip(shapes, major_axis):
            shape = list(desc.shape)
            if axis >= 0:
                shape[axis] = self.slices[i].stop - self.slices[i].start
            sliced_shapes.append(DataDesc(desc.name, tuple(shape), desc.dtype, desc.layout))
        return sliced_shapes

    def install_monitor(self, mon):
        """Install monitor on all executors"""
        for exe in self.execs:
            mon.install(exe)
