# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes
"""`SequentialModule` is a container module that chains a number of modules together."""

import logging
import copy

from ..initializer import Uniform

from .base_module import BaseModule

class SequentialModule(BaseModule):
    """A SequentialModule is a container module that can chain multiple modules together.

    .. note::

        Building a computation graph with this kind of imperative container is less
        flexible and less efficient than the symbolic graph. So, this should be only used as a
        handy utility.
    """

    META_TAKE_LABELS = 'take_labels'
    META_AUTO_WIRING = 'auto_wiring'

    def __init__(self, logger=logging):
        super(SequentialModule, self).__init__(logger=logger)
        self._modules = []
        self._metas = []

        self._label_shapes = None
        self._data_shapes = None
        self._meta_keys = set([getattr(SequentialModule, x)
                               for x in dir(SequentialModule)
                               if x.startswith('META_')])

    def add(self, module, **kwargs):
        """Adds a module to the chain.

        Parameters
        ----------
        module : BaseModule
            The new module to add.
        kwargs : **keywords
            All the keyword arguments are saved as meta information
            for the added module. The currently known meta includes

            - `take_labels`: indicating whether the module expect to
              take labels when doing computation. Note any module in
              the chain can take labels (not necessarily only the top
              most one), and they all take the same labels passed
              from the original data batch for the `SequentialModule`.

        Returns
        -------
        self
            This function returns `self` to allow us to easily chain a
            series of `add` calls.

        Examples
        --------
        >>> # An example of addinging two modules to a chain.
        >>> seq_mod = mx.mod.SequentialModule()
        >>> seq_mod.add(mod1)
        >>> seq_mod.add(mod2)
        """
        self._modules.append(module)

        # a sanity check to avoid typo
        for key in kwargs:
            assert key in self._meta_keys, ('Unknown meta "%s", a typo?' % key)

        self._metas.append(kwargs)

        # after adding new modules, we are reset back to raw states, needs
        # to bind, init_params, etc.
        self.binded = False
        self.params_initialized = False
        self.optimizer_initialized = False

        return self # for easier chaining

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        if len(self._modules) > 0:
            return self._modules[0].data_names
        return []

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        if len(self._modules) > 0:
            return self._modules[-1].output_names
        return []

    @property
    def data_shapes(self):
        """Gets data shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The data shapes of the first module
            is the data shape of a `SequentialModule`.
        """
        assert self.binded
        return self._modules[0].data_shapes

    @property
    def label_shapes(self):
        """Gets label shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The return value could be `None` if
            the module does not need labels, or if the module is not bound for
            training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Gets output shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The output shapes of the last
            module is the output shape of a `SequentialModule`.
        """
        assert self.binded
        return self._modules[-1].output_shapes

    def get_params(self):
        """Gets current parameters.

        Returns
        -------
        (arg_params, aux_params)
            A pair of dictionaries each mapping parameter names to NDArray values. This
            is a merged dictionary of all the parameters in the modules.
        """
        assert self.binded and self.params_initialized

        arg_params = dict()
        aux_params = dict()

        for module in self._modules:
            arg, aux = module.get_params()
            arg_params.update(arg)
            aux_params.update(aux)

        return (arg_params, aux_params)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initializes parameters.

        Parameters
        ----------
        initializer : Initializer
        arg_params : dict
            Default ``None``. Existing parameters. This has higher priority
            than `initializer`.
        aux_params : dict
            Default ``None``. Existing auxiliary states. This has higher priority
            than `initializer`.
        allow_missing : bool
            Allow missing values in `arg_params` and `aux_params` (if not ``None``).
            In this case, missing values will be filled with `initializer`.
        force_init : bool
            Default ``False``.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        for module in self._modules:
            module.init_params(initializer=initializer, arg_params=arg_params,
                               aux_params=aux_params, allow_missing=allow_missing,
                               force_init=force_init, allow_extra=allow_extra)

        # make sure we do not have duplicated parameter names
        def _check_name(known_names, new_names, modules, i):
            """Internal function to help checking duplicated names."""
            for name in new_names:
                assert not name in known_names, "Duplicated parameter names: " + \
                    ('name "%s" in layer %d (%s) is already ' % (name, i, type(modules[i]))) + \
                    ('used in layer %d (%s).' % (known_names[name],
                                                 type(modules[known_names[name]])))
                known_names[name] = i

        arg_names = dict()
        aux_names = dict()
        for i_layer, module in enumerate(self._modules):
            arg_params, aux_params = module.get_params()
            _check_name(arg_names, arg_params.keys(), self._modules, i_layer)
            _check_name(aux_names, aux_params.keys(), self._modules, i_layer)

        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is ``True``. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. Currently shared module is not supported for `SequentialModule`.
        grad_req : str, list of str, dict of str to str
            Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
            (default to 'write').
            Can be specified globally (str) or for each argument (list, dict).
        """
        if self.binded and not force_rebind:
            self.logger.warning('Already bound, ignoring bind()')
            return

        if inputs_need_grad:
            assert for_training is True
        assert shared_module is None, 'Shared module is not supported'
        assert len(self._modules) > 0, 'Attempting to bind an empty SequentialModule'

        self.binded = True

        # the same label shapes are used for all chained modules
        self._label_shapes = label_shapes

        my_data_shapes = data_shapes
        anybody_ever_needs_label = False
        for i_layer, module in enumerate(self._modules):
            meta = self._metas[i_layer]
            if SequentialModule.META_TAKE_LABELS in meta and \
                    meta[SequentialModule.META_TAKE_LABELS]:
                my_label_shapes = label_shapes
                anybody_ever_needs_label = True
            else:
                my_label_shapes = None

            my_inputs_need_grad = bool(inputs_need_grad or
                                       (for_training and i_layer > 0))

            if meta.get(SequentialModule.META_AUTO_WIRING, False):
                data_names = module.data_names
                assert len(data_names) == len(my_data_shapes)
                my_data_shapes = [(new_name, shape) for (new_name, (_, shape))
                                  in zip(data_names, my_data_shapes)]

            module.bind(data_shapes=my_data_shapes, label_shapes=my_label_shapes,
                        for_training=for_training, inputs_need_grad=my_inputs_need_grad,
                        force_rebind=force_rebind, shared_module=None, grad_req=grad_req)

            # the output of the previous module is the data of the next module
            my_data_shapes = module.output_shapes

        if not anybody_ever_needs_label:
            # then I do not need label either
            self._label_shapes = None

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),),
                       force_init=False):
        """Installs and initializes optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default ``(('learning_rate', 0.01),)``. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        for module in self._modules:
            module.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                                  optimizer_params=optimizer_params, force_init=force_init)

        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
        is_train : bool
            Default is ``None``, in which case `is_train` is take as ``self.for_training``.
        """
        assert self.binded and self.params_initialized

        # make a shallow copy, just to maintain necessary properties (if any) like
        # bucket_key, pad, etc.
        data_batch = copy.copy(data_batch)

        for i_layer, module in enumerate(self._modules):
            module.forward(data_batch, is_train=is_train)

            if i_layer+1 == len(self._modules):
                # the last layer, do not need to do the followings
                break

            data_batch.data = module.get_outputs()
            if hasattr(data_batch, 'provide_data'):
                # need to update this, in case the internal module is using bucketing
                # or whatever
                data_names = [x[0] for x in module.output_shapes]
                assert len(data_names) == len(data_batch.data)
                data_batch.provide_data = [(name, x.shape) for name, x in
                                           zip(data_names, data_batch.data)]

    def backward(self, out_grads=None):
        """Backward computation."""
        assert self.binded and self.params_initialized

        for i_layer, module in reversed(list(zip(range(len(self._modules)), self._modules))):
            module.backward(out_grads=out_grads)
            if i_layer == 0:
                break

            out_grads = module.get_input_grads()

    def update(self):
        """Updates parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        for module in self._modules:
            module.update()

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs from a previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
            If `merge_multi_context` is ``True``, it is like ``[out1,
            out2]``. Otherwise, it is like ``[[out1_dev1, out1_dev2], [out2_dev1,
            out2_dev2]]``. All the output elements are numpy arrays.
        """
        assert self.binded and self.params_initialized
        return self._modules[-1].get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArrays or list of list of NDArrays
            If `merge_multi_context` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
            is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
            elements are `NDArray`.
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._modules[0].get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        """Evaluates and accumulates evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically ``data_batch.label``.
        """
        assert self.binded and self.params_initialized

        for meta, module in zip(self._metas, self._modules):
            if SequentialModule.META_TAKE_LABELS in meta and \
                    meta[SequentialModule.META_TAKE_LABELS]:
                module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """Installs monitor on all executors."""
        assert self.binded
        for module in self._modules:
            module.install_monitor(mon)
