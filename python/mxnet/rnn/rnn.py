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
# pylint: disable=too-many-arguments, no-member
"""Functions for constructing recurrent neural networks."""
import warnings

from ..model import save_checkpoint, load_checkpoint
from .rnn_cell import BaseRNNCell

def rnn_unroll(cell, length, inputs=None, begin_state=None, input_prefix='', layout='NTC'):
    """Deprecated. Please use cell.unroll instead"""
    warnings.warn('rnn_unroll is deprecated. Please call cell.unroll directly.')
    return cell.unroll(length=length, inputs=inputs, begin_state=begin_state,
                       input_prefix=input_prefix, layout=layout)

def save_rnn_checkpoint(cells, prefix, epoch, symbol, arg_params, aux_params):
    """Save checkpoint for model using RNN cells.
    Unpacks weight before saving.

    Parameters
    ----------
    cells : RNNCell or list of RNNCells
        The RNN cells used by this symbol.
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input symbol
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-epoch.params`` will be saved for parameters.
    """
    if isinstance(cells, BaseRNNCell):
        cells = [cells]
    for cell in cells:
        arg_params = cell.unpack_weights(arg_params)
    save_checkpoint(prefix, epoch, symbol, arg_params, aux_params)

def load_rnn_checkpoint(cells, prefix, epoch):
    """Load model checkpoint from file.
    Pack weights after loading.

    Parameters
    ----------
    cells : RNNCell or list of RNNCells
        The RNN cells used by this symbol.
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
    - symbol will be loaded from ``prefix-symbol.json``.
    - parameters will be loaded from ``prefix-epoch.params``.
    """
    sym, arg, aux = load_checkpoint(prefix, epoch)
    if isinstance(cells, BaseRNNCell):
        cells = [cells]
    for cell in cells:
        arg = cell.pack_weights(arg)

    return sym, arg, aux

def do_rnn_checkpoint(cells, prefix, period=1):
    """Make a callback to checkpoint Module to prefix every epoch.
    unpacks weights used by cells before saving.

    Parameters
    ----------
    cells : RNNCell or list of RNNCells
        The RNN cells used by this symbol.
    prefix : str
        The file prefix to checkpoint to
    period : int
        How many epochs to wait before checkpointing. Default is 1.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))
    # pylint: disable=unused-argument
    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            save_rnn_checkpoint(cells, prefix, iter_no+1, sym, arg, aux)
    return _callback
