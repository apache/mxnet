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

from __future__ import absolute_import, division, print_function

import mxnet as mx
import mxnet.ndarray as nd
import numpy
import copy
from utils import *


class ReplayMemory(object):
    def __init__(self, history_length, memory_size=1000000, replay_start_size=100,
                 state_dim=(), action_dim=(), state_dtype='uint8', action_dtype='uint8',
                 ctx=mx.gpu()):
        self.rng = get_numpy_rng()
        self.ctx = ctx
        assert type(action_dim) is tuple and type(state_dim) is tuple, \
            "Must set the dimensions of state and action for replay memory"
        self.state_dim = state_dim
        if action_dim == (1,):
            self.action_dim = ()
        else:
            self.action_dim = action_dim
        self.states = numpy.zeros((memory_size,) + state_dim, dtype=state_dtype)
        self.actions = numpy.zeros((memory_size,) + action_dim, dtype=action_dtype)
        self.rewards = numpy.zeros(memory_size, dtype='float32')
        self.terminate_flags = numpy.zeros(memory_size, dtype='bool')
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.history_length = history_length
        self.top = 0
        self.size = 0

    def latest_slice(self):
        if self.size >= self.history_length:
            return self.states.take(numpy.arange(self.top - self.history_length, self.top),
                                    axis=0, mode="wrap")
        else:
            assert False, "We can only slice from the replay memory if the " \
                          "replay size is larger than the length of frames we want to take" \
                          "as the input."

    @property
    def sample_enabled(self):
        return self.size > self.replay_start_size

    def clear(self):
        """
        Clear all contents in the relay memory
        """
        self.states[:] = 0
        self.actions[:] = 0
        self.rewards[:] = 0
        self.terminate_flags[:] = 0
        self.top = 0
        self.size = 0

    def reset(self):
        """
        Reset all the flags stored in the replay memory.
        It will not clear the inner-content and is a light/quick version of clear()
        """
        self.top = 0
        self.size = 0

    def copy(self):
        # TODO Test the copy function
        replay_memory = copy.copy(self)
        replay_memory.states = numpy.zeros(self.states.shape, dtype=self.states.dtype)
        replay_memory.actions = numpy.zeros(self.actions.shape, dtype=self.actions.dtype)
        replay_memory.rewards = numpy.zeros(self.rewards.shape, dtype='float32')
        replay_memory.terminate_flags = numpy.zeros(self.terminate_flags.shape, dtype='bool')
        replay_memory.states[numpy.arange(self.top-self.size, self.top), ::] = \
            self.states[numpy.arange(self.top-self.size, self.top)]
        replay_memory.actions[numpy.arange(self.top-self.size, self.top)] = \
            self.actions[numpy.arange(self.top-self.size, self.top)]
        replay_memory.rewards[numpy.arange(self.top-self.size, self.top)] = \
            self.rewards[numpy.arange(self.top-self.size, self.top)]
        replay_memory.terminate_flags[numpy.arange(self.top-self.size, self.top)] = \
            self.terminate_flags[numpy.arange(self.top-self.size, self.top)]
        return replay_memory

    def append(self, obs, action, reward, terminate_flag):
        self.states[self.top] = obs
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminate_flags[self.top] = terminate_flag
        self.top = (self.top + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1

    def sample_last(self, batch_size, states, offset):
        assert self.size >= batch_size and self.replay_start_size >= self.history_length
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be "
                             "bigger than start_size! Currently, size=%d, start_size=%d"
                             %(self.size, self.replay_start_size))
        actions = numpy.empty((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.empty(batch_size, dtype='float32')
        terminate_flags = numpy.empty(batch_size, dtype='bool')
        counter = 0
        first_index = self.top - self.history_length - 1
        while counter < batch_size:
            full_indices = numpy.arange(first_index, first_index + self.history_length+1)
            end_index = first_index + self.history_length
            if numpy.any(self.terminate_flags.take(full_indices[0:self.history_length], mode='wrap')):
                # Check if terminates in the middle of the sample!
                first_index -= 1
                continue
            states[counter + offset] = self.states.take(full_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, axis=0, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            counter += 1
            first_index -= 1
        return actions, rewards, terminate_flags

    def sample_mix(self, batch_size, states, offset, current_index):
        assert self.size >= batch_size and self.replay_start_size >= self.history_length
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be bigger than "
                             "start_size! Currently, size=%d, start_size=%d"
                             %(self.size, self.replay_start_size))
        actions = numpy.empty((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.empty(batch_size, dtype='float32')
        terminate_flags = numpy.empty(batch_size, dtype='bool')
        counter = 0
        first_index = self.top - self.history_length + current_index
        thisid = first_index
        while counter < batch_size:
            full_indices = numpy.arange(thisid, thisid + self.history_length+1)
            end_index = thisid + self.history_length
            if numpy.any(self.terminate_flags.take(full_indices[0:self.history_length], mode='wrap')):
                # Check if terminates in the middle of the sample!
                thisid -= 1
                continue
            states[counter+offset] = self.states.take(full_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, axis=0, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            counter += 1
            thisid = self.rng.randint(low=self.top - self.size, high=self.top - self.history_length-1)
        return actions, rewards, terminate_flags

    def sample_inplace(self, batch_size, states, offset):
        assert self.size >= batch_size and self.replay_start_size >= self.history_length
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be "
                             "bigger than start_size! Currently, size=%d, start_size=%d"
                             %(self.size, self.replay_start_size))
        actions = numpy.zeros((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.zeros(batch_size, dtype='float32')
        terminate_flags = numpy.zeros(batch_size, dtype='bool')

        counter = 0
        while counter < batch_size:
            index = self.rng.randint(low=self.top - self.size + 1, high=self.top - self.history_length )
            transition_indices = numpy.arange(index, index + self.history_length+1)
            initial_indices = transition_indices - 1
            end_index = index + self.history_length - 1
            if numpy.any(self.terminate_flags.take(initial_indices[0:self.history_length], mode='wrap')):
                # Check if terminates in the middle of the sample!
                continue
            states[counter + offset] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, axis=0, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            # next_states[counter] = self.states.take(transition_indices, axis=0, mode='wrap')
            counter += 1
        return actions, rewards, terminate_flags

    def sample(self, batch_size):
        assert self.size >= batch_size and self.replay_start_size >= self.history_length
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be bigger than "
                             "start_size! Currently, size=%d, start_size=%d"
                             %(self.size, self.replay_start_size))
        #TODO Possibly states + inds for less memory access
        states = numpy.zeros((batch_size, self.history_length) + self.state_dim,
                             dtype=self.states.dtype)
        actions = numpy.zeros((batch_size,) + self.action_dim, dtype=self.actions.dtype)
        rewards = numpy.zeros(batch_size, dtype='float32')
        terminate_flags = numpy.zeros(batch_size, dtype='bool')
        next_states = numpy.zeros((batch_size, self.history_length) + self.state_dim,
                                  dtype=self.states.dtype)
        counter = 0
        while counter < batch_size:
            index = self.rng.randint(low=self.top - self.size + 1, high=self.top - self.history_length)
            transition_indices = numpy.arange(index, index + self.history_length)
            initial_indices = transition_indices - 1
            end_index = index + self.history_length - 1
            while numpy.any(self.terminate_flags.take(initial_indices, mode='wrap')):
                # Check if terminates in the middle of the sample!
                index -= 1
                transition_indices = numpy.arange(index, index + self.history_length)
                initial_indices = transition_indices - 1
                end_index = index + self.history_length - 1
            states[counter] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, axis=0, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            next_states[counter] = self.states.take(transition_indices, axis=0, mode='wrap')
            counter += 1
        return states, actions, rewards, next_states, terminate_flags
