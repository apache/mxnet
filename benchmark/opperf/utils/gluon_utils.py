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
import mxnet.ndarray as nd

from .profiler_utils import profile


@profile
def block_forward_backward_and_profile(*args, block, runs, **kwargs):
    """Helper function to run a given Block (block) for 'runs' number of times with
    given args and kwargs. Executes both forward and backward pass.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    :param block: Gluon block to execute. Example: an instance of gluon.nn.Dense(...)
    :param runs: Number of times to execute the block operation
    :param args: Arguments for the block being executed.
    :param kwargs: Key value arguments for the block being executed.
    :return: any results from block execution
    """
    for _ in range(runs):
        with mx.autograd.record():
            res = block.forward(*args, **kwargs)
        res.backward()
        nd.waitall()
    return res


@profile
def block_forward_and_profile(*args, block, runs, **kwargs):
    """Helper function to run a given Block (block) for 'runs' number of times with
    given args and kwargs. Executes forward pass only.

    NOTE: This is a sync call and waits for all the operations execution to complete.

    :param block: Gluon block to execute. Example: an instance of gluon.nn.Dense(...)
    :param runs: Number of times to execute the block operation
    :param args: Arguments for the block being executed.
    :param kwargs: Key value arguments for the block being executed.
    :return: any results from block execution
    """

    for _ in range(runs):
        # Imperative Mode. This is block forward function
        res = block.hybrid_forward(F=nd, *args, **kwargs)
        nd.waitall()
    return res
