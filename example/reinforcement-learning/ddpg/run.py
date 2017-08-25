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

from ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from policies import DeterministicMLPPolicy
from qfuncs import ContinuousMLPQ
from strategies import OUStrategy
from utils import SEED
import mxnet as mx

# set environment, policy, qfunc, strategy

env = normalize(CartpoleEnv())

policy = DeterministicMLPPolicy(env.spec)
qfunc = ContinuousMLPQ(env.spec)
strategy = OUStrategy(env.spec)

# set the training algorithm and train

algo = DDPG(
    env=env,
    policy=policy,
    qfunc=qfunc,
    strategy=strategy,
    ctx=mx.gpu(0),
    max_path_length=100,
    epoch_length=1000,
    memory_start_size=10000,
    n_epochs=1000,
    discount=0.99,
    qfunc_lr=1e-3,
    policy_lr=1e-4,
    seed=SEED)

algo.train()
