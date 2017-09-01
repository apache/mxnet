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

import mxnet as mx
import numpy as np

# random seed for reproduction
SEED = 12345


def define_qfunc(obs, act):

    net = mx.symbol.FullyConnected(
        data=obs,
        name="qfunc_fc1",
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net,
        name="qfunc_relu1",
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net,
        name="qfunc_fc2",
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net,
        name="qfunc_relu2",
        act_type="relu")
    net = mx.symbol.Concat(net, act, name="qunfc_concat")
    net = mx.symbol.FullyConnected(
        data=net,
        name="qfunc_fc3",
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net,
        name="qfunc_relu3",
        act_type="relu")
    qval = mx.symbol.FullyConnected(
        data=net,
        name="qfunc_qval",
        num_hidden=1)

    return qval


def define_policy(obs, action_dim):

    net = mx.symbol.FullyConnected(
        data=obs,
        name="policy_fc1",
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net,
        name="policy_relu1",
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net,
        name="policy_fc2",
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net,
        name="policy_relu2",
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net,
        name='policy_fc3',
        num_hidden=action_dim)
    action = mx.symbol.Activation(
        data=net,
        name="act",
        act_type="tanh")

    return action


def discount_return(x, discount):

    return np.sum(x * (discount ** np.arange(len(x))))


def rollout(env, agent, max_path_length=np.inf):

    reward = []
    o = env.reset()
    # agent.reset()
    path_length = 0
    while path_length < max_path_length:
        o = o.reshape((1, -1))
        a = agent.get_action(o)
        next_o, r, d, _ = env.step(a)
        reward.append(r)
        path_length += 1
        if d:
            break
        o = next_o

    return reward


def sample_rewards(env, policy, eval_samples, max_path_length=np.inf):

    rewards = []
    for _ in range(eval_samples):
        rewards.append(rollout(env, policy, max_path_length))

    return rewards




