class BucketingModule(BaseModule):
    """A bucketing module is a module that support bucketing.

    Parameters
    ----------
    sym_gen : function
        A function when called with a bucket key, returns a triple
        `(symbol, data_names, label_names)`.
    default_bucket_key : str (or any python object)
        The key for the default bucket.
    logger : Logger
    context : Context or list of Context
        Default `cpu()`
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    """
    def __init__(self, sym_gen, default_bucket_key=None,
                 logger=logging, context=ctx.cpu(), work_load_list=None):
        super(BucketingModule, self).__init__(logger=logger)

        assert default_bucket_key is not None
        self._default_bucket_key = default_bucket_key

        self._sym_gen = sym_gen
        self._context = context
        self._work_load_list = work_load_list

        self._buckets = {}
        self._curr_module = None

    def _reset_bind(self):
        """Internal utility function to reset binding."""
        self.binded = False
        self._buckets = {}
        self._curr_module = None

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False):
        """Binding for a `BucketingModule` means setting up the buckets and bind the
        executor for the default bucket key. Executors corresponding to other keys are
        binded afterwards with `switch_bucket`.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            This should correspond to the symbol for the default bucket.
        label_shapes : list of (str, tuple)
            This should correspond to the symbol for the default bucket.
        for_training : bool
            Default is `True`.
        inputs_need_grad : bool
            Default is `False`.
        force_rebind : bool
            Default is `False`.
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        (symbol, data_names, label_names) = self._sym_gen(self._default_bucket_key)
        module = Module(symbol, data_names, label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list)
        module.bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        self._curr_module = module
        self._buckets[self._default_bucket_key] = module

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        """Initialize parameters.

        Parameters
        ----------
        initializer : Initializer
        arg_params : dict
            Default `None`. Existing parameters. This has higher priority than `initializer`.
        aux_params : dict
            Default `None`. Existing auxiliary states. This has higher priority than `initializer`.
        allow_missing : bool
            Allow missing values in `arg_params` and `aux_params` (if not `None`). In this case,
            missing values will be filled with `initializer`.
        force_init : bool
            Default `False`.
        """
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self.curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                     aux_params=aux_params, allow_missing=allow_missing,
                                     force_init=force_init)
        self.params_initialized = True

    def switch_bucket(self, bucket_key, data_shapes, label_shapes=None):
        """Switch to a different bucket. This will change `self.curr_module`.

        Parameters
        ----------
        bucket_key : str (or any python object)
            The key of the target bucket.
        data_shapes : list of (str, tuple)
            Typically `data_batch.provide_data`.
        label_shapes : list of (str, tuple)
            Typically `data_batch.provide_label`.
        """
        assert self.binded, 'call bind before switching bucket'
        if not self.buckets.has_key(bucket_key):
            symbol, input_names = self._gen_symbol(bucket_key)
            module = Module(symbol, input_names, logger=self.logger, context=self.context,
                            work_load_list=self.work_load_list)
            module.bind(data_shapes, label_shapes, self.curr_module.for_training,
                        self.curr_module.inputs_need_grad,
                        force_rebind=False, shared_module=self.curr_module)
            self.buckets[bucket_key] = module

        self.curr_module = self.buckets[bucket_key]


    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
        is_train : bool
            Default is `None`, in which case `is_train` is take as `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self.switch_bucket(data_batch.bucket_key, data_batch.provide_data,
                           data_batch.provide_label)
        self.curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        """Backward computation."""
        self.curr_module.backward(out_grads=out_grads)

    def get_outputs(self, merge_multi_context=True):
        """Get outputs from a previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are numpy arrays.
        """
        return self.curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def init_optimizer(self, kvstore='local', optimizer='sgd', optimizer_params=dict(),
                       force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `{}`
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self.curr_module.init_optimizer(kvstore, optimizer, optimizer_params, force_init=force_init)
        for mod in self.buckets.itervalues():
            if mod is not self.curr_module:
                mod._borrow_optimizer(self.curr_module)

        self.optimizer_initialized = True

    def update(self):
        """Update parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self.curr_module.update()

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self.curr_module.update_metric(eval_metric, labels)

    def sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read
        the latest parameters from `self.arg_params` and `self.aux_params`.
        """
        self.curr_module.sync_params_from_devices()

    @property
    def arg_params(self):
        """Parameters for the current module."""
        assert self.binded and self.params_initialized
        return self.curr_module.arg_params

    @property
    def aux_params(self):
        """Auxiliary states of the current module."""
        assert self.binded and self.params_initialized
        return self.curr_module.aux_params

    @property
    def symbol(self):
        """The symbol of the current bucket being used."""
        assert self.binded
        return self.curr_module.symbol
