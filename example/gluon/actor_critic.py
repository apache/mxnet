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

from __future__ import print_function

import argparse
import gym
from itertools import count
import numpy as np

import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd


parser = argparse.ArgumentParser(description='MXNet actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)


class Policy(gluon.Block):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(16, in_units=4, activation='relu')
            self.action_pred = nn.Dense(2, in_units=16)
            self.value_pred = nn.Dense(1, in_units=16)

    def forward(self, x):
        x = self.dense(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return F.softmax(probs), values

net = Policy()
net.initialize(mx.init.Uniform(0.02))
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 3e-2})
loss = gluon.loss.L1Loss()

running_reward = 10
for epoch in count(1):
    state = env.reset()
    rewards = []
    values = []
    heads = []
    actions = []
    with autograd.record():
        # Sample a sequence of actions
        for t in range(10000):
            state = mx.nd.array(np.expand_dims(state, 0))
            prob, value = net(state)
            action, logp = mx.nd.sample_multinomial(prob, get_prob=True)
            state, reward, done, _ = env.step(action.asnumpy()[0])
            if args.render:
                env.render()
            rewards.append(reward)
            values.append(value)
            actions.append(action.asnumpy()[0])
            heads.append(logp)
            if done:
                break

        # reverse accumulate and normalize rewards
        running_reward = running_reward * 0.99 + t * 0.01
        R = 0
        for i in range(len(rewards)-1, -1, -1):
            R = rewards[i] + args.gamma * R
            rewards[i] = R
        rewards = np.array(rewards)
        rewards -= rewards.mean()
        rewards /= rewards.std() + np.finfo(rewards.dtype).eps

        # compute loss and gradient
        L = sum([loss(value, mx.nd.array([r])) for r, value in zip(rewards, values)])
        final_nodes = [L]
        for logp, r, v in zip(heads, rewards, values):
            reward = r - v.asnumpy()[0,0]
            # Here we differentiate the stochastic graph, corresponds to the
            # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
            # Optimizer minimizes the loss but we want to maximizing the reward,
            # so use we use -reward here.
            final_nodes.append(logp*(-reward))
        autograd.backward(final_nodes)

    trainer.step(t)

    if epoch % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            epoch, t, running_reward))
    if running_reward > 200:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
