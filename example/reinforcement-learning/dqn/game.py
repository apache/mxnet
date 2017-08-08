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


DEFAULT_MAX_EPISODE_STEP = 1000000

class Game(object):
    def __init__(self):
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.max_episode_step = DEFAULT_MAX_EPISODE_STEP

    def start(self):
        raise NotImplementedError("Must Implement!")

    def begin_episode(self, max_episode_step):
        raise NotImplementedError("Must Implement!")

    @property
    def episode_terminate(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    @property
    def state_enabled(self):
        raise NotImplementedError

    def current_state(self):
        return self.replay_memory.latest_slice()

    def play(self, a):
        raise NotImplementedError
