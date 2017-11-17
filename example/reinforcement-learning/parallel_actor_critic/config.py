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


class Config(object):
    def __init__(self, args):
        # Default training settings
        self.ctx = mx.gpu(0) if args.gpu else mx.cpu()
        self.init_func = mx.init.Xavier(rnd_type='uniform', factor_type="in",
                                        magnitude=1)
        self.learning_rate = 1e-3
        self.update_rule = "adam"
        self.grad_clip = True
        self.clip_magnitude = 40

        # Default model settings
        self.hidden_size = 200
        self.gamma = 0.99
        self.lambda_ = 1.0
        self.vf_wt = 0.5        # Weight of value function term in the loss
        self.entropy_wt = 0.01  # Weight of entropy term in the loss

        self.num_envs = 16
        self.t_max = 50

        # Override defaults with values from `args`.
        for arg in self.__dict__:
            if arg in args.__dict__:
                self.__setattr__(arg, args.__dict__[arg])
