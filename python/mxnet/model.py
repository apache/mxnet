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

# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""

import os
import logging
from collections import namedtuple
import numpy as np

from . import ndarray as nd
from . import symbol as sym
from . import kvstore as kvs
from .device import cpu

BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])

def _create_sparse_kvstore(kvstore):
    """Create kvstore assuming some parameters' storage types are row_sparse.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.

    Returns
    -------
    kvstore : KVStore
    update_on_kvstore : bool. Always True.
    """
    # always update on kvstore
    if isinstance(kvstore, kvs.KVStore):
        kv = kvstore
    elif isinstance(kvstore, str):
        kv = kvs.create(kvstore)
    else:
        raise TypeError(f"Cannot create '{kvstore}' KVStore with row_sparse parameters. "
                        "The type must be KVStore or str.")
    assert kv.is_capable(kvs.KVStoreBase.OPTIMIZER), \
        "KVStore with sparse weight requires optimizer support. " \
        "However, type(kv) does not support optimizer. " \
        "Please consider other kvstore backends (e.g. dist_device) instead."
    return (kv, True)

def _create_kvstore(kvstore, num_device, arg_params):
    """Create kvstore
    This function select and create a proper kvstore if given the kvstore type.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.
    num_device : int
        The number of devices
    arg_params : dict of str to `NDArray`.
        Model parameter, dict of name to `NDArray` of net's weights.
    """
    update_on_kvstore = bool(int(os.getenv('MXNET_UPDATE_ON_KVSTORE', "1")))
    if kvstore is None:
        kv = None
    elif isinstance(kvstore, kvs.KVStoreBase):
        kv = kvstore
    elif isinstance(kvstore, str):
        # create kvstore using the string type
        if num_device == 1 and 'dist' not in kvstore:
            # no need to use kv for single device and single machine
            kv = None
        else:
            kv = kvs.create(kvstore)
            if kvstore == 'local':
                # automatically select a proper local
                max_size = max(np.prod(param.shape) for param in
                               arg_params.values())
                if max_size > 1024 * 1024 * 16:
                    update_on_kvstore = False
    else:
        raise TypeError('kvstore must be KVStore, str or None')

    if kv is None:
        update_on_kvstore = False
    else:
        update_on_kvstore &= kv.is_capable(kvs.KVStoreBase.OPTIMIZER)

    return (kv, update_on_kvstore)

def _initialize_kvstore(kvstore, param_arrays, arg_params, param_names, update_on_kvstore):
    """Initialize kvstore"""
    for idx, param_on_devs in enumerate(param_arrays):
        name = param_names[idx]
        if not update_on_kvstore or arg_params[name].stype != 'default':
            kvstore.init(name, arg_params[name])
        else:
            kvstore.broadcast(name, arg_params[name], out=param_on_devs)

def _update_params_on_kvstore_nccl(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on NCCL kvstore."""
    valid_indices = [index for index, grad_list in
                     enumerate(grad_arrays) if grad_list[0] is not None]
    valid_grad_arrays = [grad_arrays[i] for i in valid_indices]
    valid_param_arrays = [param_arrays[i] for i in valid_indices]
    valid_param_names = [param_names[i] for i in valid_indices]
    size = len(valid_grad_arrays)
    start = 0
    # Use aggregation by default only with NCCL
    default_batch = '16'
    batch = int(os.getenv('MXNET_UPDATE_AGGREGATION_SIZE', default_batch))
    while start < size:
        end = start + batch if start + batch < size else size
        # push gradient, priority is negative index
        # pull back the weights
        kvstore.pushpull(valid_param_names[start:end], valid_grad_arrays[start:end],
                         out=valid_param_arrays[start:end], priority=-start)
        start = end

def _update_params_on_kvstore(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        name = param_names[index]
        # push gradient, priority is negative index
        # pull back the weights
        if grad_list[0].stype == 'default' and arg_list[0].stype == 'default':
            kvstore.pushpull(name, grad_list, out=arg_list, priority=-index)
        else:
            kvstore.push(name, grad_list, priority=-index)
            kvstore.pull(name, out=arg_list, priority=-index)

def _update_params(param_arrays, grad_arrays, updater, num_device,
                   kvstore=None, param_names=None):
    """Perform update of param_arrays from grad_arrays not on kvstore."""
    updates = [[] for _ in range(num_device)]
    for i, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        index = i
        if kvstore:
            name = param_names[index]
            # push gradient, priority is negative index
            if grad_list[0].stype == 'default' and arg_list[0].stype == 'default':
                kvstore.pushpull(name, grad_list, priority=-index)
            else:
                kvstore.push(name, grad_list, priority=-index)
                kvstore.pull(name, out=grad_list, priority=-index)
        for k, p in enumerate(zip(arg_list, grad_list)):
            # faked an index here, to make optimizer create diff
            # state for the same index but on diff devs, TODO(mli)
            # use a better solution later
            w, g = p
            updates[k].append((index*num_device+k, g, w))
    for dev_updates in updates:
        # update params if param_arrays and grad_arrays are not empty
        if dev_updates:
            i, w, g = zip(*dev_updates)
            updater(i, w, g)


def save_checkpoint(prefix, epoch, symbol, arg_params, aux_params, remove_amp_cast=True):
    """Checkpoint the model data into file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input Symbol.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    remove_amp_cast : bool, optional
        Whether to remove the amp_cast and amp_multicast operators, before saving the model.
    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-epoch.params`` will be saved for parameters.
    """
    if symbol is not None:
        symbol.save(f'{prefix}-symbol.json', remove_amp_cast=remove_amp_cast)

    save_dict = {(f'arg:{k}') : v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({(f'aux:{k}') : v.as_in_context(cpu()) for k, v in aux_params.items()})
    param_name = f'{prefix}-{epoch:04}.params'
    nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to "{}"'.format(param_name))


def load_params(prefix, epoch):
    """Load params from a file
    """
    save_dict = nd.load(f'{prefix}-{epoch:04}.params')
    arg_params = {}
    aux_params = {}
    if not save_dict:
        logging.warning("Params file '%s' is empty", f'{prefix}-{epoch:04}.params')
        return (arg_params, aux_params)
    for k, v in save_dict.items():
        tp, name = k.split(":", 1)
        if tp == "arg":
            arg_params[name] = v
        if tp == "aux":
            aux_params[name] = v
    return (arg_params, aux_params)

def load_checkpoint(prefix, epoch):
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - Symbol will be loaded from ``prefix-symbol.json``.
    - Parameters will be loaded from ``prefix-epoch.params``.
    """
    symbol = sym.load(f'{prefix}-symbol.json')
    arg_params, aux_params = load_params(prefix, epoch)
    return (symbol, arg_params, aux_params)
