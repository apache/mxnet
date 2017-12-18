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
import mxnet as mx
from mxnet import profiler
import time

def test_profiler():
    profile_filename = "test_profile.json"
    iter_num = 5
    begin_profiling_iter = 2
    end_profiling_iter = 4

    profiler.profiler_set_config(mode='symbolic', filename=profile_filename)
    print('profile file save to {0}'.format(profile_filename))

    A = mx.sym.Variable('A')
    B = mx.sym.Variable('B')
    C = mx.symbol.dot(A, B)

    executor = C.simple_bind(mx.cpu(1), 'write', A=(4096, 4096), B=(4096, 4096))

    a = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))
    b = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    print("execution begin")
    for i in range(iter_num):
        print("Iteration {}/{}".format(i + 1, iter_num))
        if i == begin_profiling_iter:
            t0 = time.clock()
            profiler.profiler_set_state('run')
        if i == end_profiling_iter:
            t1 = time.clock()
            profiler.profiler_set_state('stop')
        executor.forward()
        c = executor.outputs[0]
        c.wait_to_read()
    print("execution end")
    duration = t1 - t0
    print('duration: {0}s'.format(duration))
    print('          {0}ms/operator'.format(duration*1000/iter_num))
    profiler.dump_profile()

if __name__ == '__main__':
    test_profiler()
