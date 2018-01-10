import logging
import warnings

import mxnet as mx
from mxnet.module import Module
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from mxnet.model import load_checkpoint

class SparseModule(Module):

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=mx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None, group2ctxs=None,
                 compression_params=None, sparse_params=None):

        super(SparseModule, self).__init__(symbol, data_names=data_names, label_names=label_names,
                                           logger=logger, context=context, work_load_list=work_load_list,
                                           fixed_param_names=fixed_param_names, state_names=state_names,
                                           group2ctxs=group2ctxs, compression_params=compression_params)
        self._sparse_params = sparse_params

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        import mxnet.optimizer as opt

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        if self._params_dirty:
            self._sync_params_from_devices()

        (kvstore, update_on_kvstore) = \
                _create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0/batch_size

        if isinstance(optimizer, str):
            idx2name = {}
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                     for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, mx.optimizer.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                #pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "%(
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore,
                                skip_pull=self._sparse_params)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        else:
            self._updater = opt.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None
        # TODO(haibin) refactor init kvstore

    def sync_sparse_params(self, param_rowids):
        '''Prepares the module for processing a data batch.
        Usually involves switching bucket and reshaping.
        Parameters
        ----------
        '''
        if not self._kvstore:
            return
        assert(isinstance(param_rowids, dict))
        for param_name, rowid in param_rowids.items():
            param_idx = self._exec_group.param_names.index(param_name)
            param_val = self._exec_group.param_arrays[param_idx]
            self._kvstore.row_sparse_pull(param_name, param_val, row_ids=rowid,
                                          priority=-param_idx)

    def update(self):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        See Also
        ----------
        :meth:`BaseModule.update`.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        self._params_dirty = True
        if self._update_on_kvstore:
            _update_params_on_kvstore(self._exec_group.param_arrays,
                                      self._exec_group.grad_arrays,
                                      self._kvstore, self._exec_group.param_names,
                                      skip_pull=self._sparse_params)
        else:
            _update_params(self._exec_group.param_arrays,
                           self._exec_group.grad_arrays,
                           updater=self._updater,
                           num_device=len(self._context),
                           kvstore=self._kvstore,
                           param_names=self._exec_group.param_names)
    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Creates a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is ``cpu()``.
        work_load_list : list of number
            Default ``None``, indicating uniform workload.
        fixed_param_names: list of str
            Default ``None``, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = SparseModule(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_params(self, fname):
        """Saves model parameters to file.
        Parameters
        ----------
        fname : str
            Path to output param file.
        Examples
        --------
        >>> # An example of saving module parameters.
        >>> mod.save_params('myfile')
        """
        arg_params, aux_params = self.get_params_from_kv(self._arg_params, self._aux_params)
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.nd.save(fname, save_dict)

    def get_params_from_kv(self, arg_params, aux_params):
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
        assert(self._kvstore is not None)
        for name, block in zip(self._exec_group.param_names, self._exec_group.param_arrays):
            assert(isinstance(block, list))
            if block[0].stype == 'row_sparse':
                row_ids = mx.nd.arange(start=0, stop=block[0].shape[0])
                self._kvstore.row_sparse_pull(name, arg_params[name], row_ids=row_ids)
            elif block[0].stype == 'default':
                self._kvstore.pull(name, out=arg_params[name])
            else:
                raise NotImplementedError()
        # TODO handle aux names
        print(self._exec_group.aux_names)
        #assert(self._exec_group.aux_names is None or self._exec_group.aux_arrays is None)
        #for name, block in zip(self._exec_group.aux_names, self._exec_group.aux_arrays):
        #    if block[0].stype == 'row_sparse':
        #        row_ids = mx.nd.arange(start=0, stop=block[0].shape[0])
        #        self._kvstore.row_sparse_pull(name, aux_params[name], row_ids=row_ids)
        #    elif block[0].stype == 'default':
        #        self._kvstore.pull(name, out=aux_params[name])
        #    else:
        #        raise NotImplementedError()
        return arg_params, aux_params
