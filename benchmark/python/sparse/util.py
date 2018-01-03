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

import os
import random

def estimate_density(DATA_PATH, feature_size):
    """sample 10 times of a size of 1000 for estimating the density of the sparse dataset"""
    if not os.path.exists(DATA_PATH):
        raise Exception("Data is not there!")
    density = []
    P = 0.01
    for _ in xrange(10):
        num_non_zero = 0
        num_sample = 0
        with open(DATA_PATH) as f:
            for line in f:
                if (random.random() < P):
                    num_non_zero += len(line.split(" ")) - 1
                    num_sample += 1
        density.append(num_non_zero * 1.0 / (feature_size * num_sample))
    return sum(density) / len(density)

