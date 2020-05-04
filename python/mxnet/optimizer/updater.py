# coding: utf-8
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
"""Updater class."""
from __future__ import absolute_import
import pickle
import numpy
from ..base import py_str
from ..ndarray import NDArray
from ..profiler import Scope
from ..util import is_np_array
from .utils import _as_classic

__all__ = ['Updater', 'get_updater']


class Updater(object):
    """Updater for kvstore."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.states = {}
        self.states_synced = {}
        self.aggregate_updates = optimizer.aggregate_num > 1

    def __call__(self, index, grad, weight):
        """Updates weight given gradient and index."""
        allow_np = self.optimizer.allow_np_array if hasattr(self.optimizer, "allow_np_array") else is_np_array()
        if not isinstance(index, (list, tuple)):
            indices = [index]
            grads = [_as_classic(grad, allow_np)]
            weights = [_as_classic(weight, allow_np)]
        else:
            indices = index
            grads = _as_classic(grad, allow_np)
            weights = _as_classic(weight, allow_np)
        if weights:
            self.optimizer._set_current_context(weights[0].context.device_id)
        for i, idx in enumerate(indices):
            # convert ctypes.char_p.value back to python str if needed
            if isinstance(idx, bytes):
                indices[i] = py_str(idx)
                idx = indices[i]
            if idx not in self.states:
                with Scope("updater:optimizer_state"):
                    self.states[idx] = self.optimizer.create_state_multi_precision(idx, weights[i])
                self.states_synced[idx] = True
            elif not self.states_synced[idx]:
                self.states[idx] = \
                    self.sync_state_context(self.states[idx], weights[i].context)
                self.states_synced[idx] = True
        if self.aggregate_updates:
            # segregate values based on type
            if self.optimizer.aggregate_num is not numpy.inf:
                type_map = {}
                for i, w, g in zip(indices, weights, grads):
                    if w.dtype in type_map:
                        type_map[w.dtype].append((i, w, g))
                    else:
                        type_map[w.dtype] = [(i, w, g)]
                for idx in type_map:
                    current_index = 0
                    indices, weights, grads = zip(*type_map[idx])
                    while current_index < len(indices):
                        states = []
                        step = min(self.optimizer.aggregate_num, len(indices) - current_index)
                        for j in range(step):
                            states.append(self.states[indices[current_index + j]])
                        self.optimizer.update_multi_precision(
                            indices[current_index:current_index + self.optimizer.aggregate_num],
                            weights[current_index:current_index + self.optimizer.aggregate_num],
                            grads[current_index:current_index + self.optimizer.aggregate_num],
                            states)
                        current_index += self.optimizer.aggregate_num
            else:
                states = [self.states[i] for i in indices]
                self.optimizer.update_multi_precision(indices, weights, grads, states)
        else:
            for i, w, g in zip(indices, weights, grads):
                self.optimizer.update_multi_precision([i], [w], [g], [self.states[i]])

    def sync_state_context(self, state, context):
        """sync state context."""
        if isinstance(state, NDArray):
            return state.as_in_context(context)
        elif isinstance(state, (tuple, list)):
            synced_state = (self.sync_state_context(i, context) for i in state)
            if isinstance(state, tuple):
                return tuple(synced_state)
            else:
                return list(synced_state)
        else:
            return state

    def set_states(self, states):
        """Sets updater states."""
        states = pickle.loads(states)
        if isinstance(states, tuple) and len(states) == 2:
            self.states, self.optimizer = states
        else:
            self.states = states
        self.states_synced = dict.fromkeys(self.states.keys(), False)

    def get_states(self, dump_optimizer=False):
        """Gets updater states.

        Parameters
        ----------
        dump_optimizer : bool, default False
            Whether to also save the optimizer itself. This would also save optimizer
            information such as learning rate and weight decay schedules.
        """
        return pickle.dumps((self.states, self.optimizer) if dump_optimizer else self.states)


def get_updater(optimizer):
    """Returns a closure of the updater needed for kvstore.

    Parameters
    ----------
    optimizer: Optimizer
         The optimizer.

    Returns
    -------
    updater: function
         The closure of the updater.
    """
    return Updater(optimizer)
