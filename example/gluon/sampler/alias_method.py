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
# pylint: skip-file

import numpy as np
import numpy.random as npr

import mxnet.gluon as gluon

K = 5
N = 1000

# Get a random probability vector.
probs = npr.dirichlet(np.ones(K), 1).ravel()

# Construct the table.
alias_method_sampler = gluon.data.AliasMethodSampler(K, probs)

# Generate variates.
X = alias_method_sampler.draw(N)

# check sampled probabilities
sampled_probs = [float(x)/N for x in X]

print(probs)
print(sampled_probs)


