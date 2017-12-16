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
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Set network parameters for benchmark test.')
    parser.add_argument('--profile_filename', type=str, default='profile_matmul_20iter.json')
    parser.add_argument('--iter_num', type=int, default=100)
    parser.add_argument('--begin_profiling_iter', type=int, default=50)
    parser.add_argument('--end_profiling_iter', type=int, default=70)
    return parser.parse_args()


args = parse_args()

if __name__ == '__main__':
    mx.profiler.profiler_set_config(mode='symbolic', filename=args.profile_filename)
    print('profile file save to {0}'.format(args.profile_filename))

    A = mx.sym.Variable('A')
    B = mx.sym.Variable('B')
    C = mx.symbol.dot(A, B)

    executor = C.simple_bind(mx.gpu(0), 'write', A=(4096, 4096), B=(4096, 4096))

    a = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))
    b = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    flag = False
    print("execution begin")
    for i in range(args.iter_num):
        if i == args.begin_profiling_iter:
            t0 = time.clock()
            mx.profiler.profiler_set_state('run')
        if i == args.end_profiling_iter:
            t1 = time.clock()
            mx.profiler.profiler_set_state('stop')
        executor.forward()
        c = executor.outputs[0]
        c.wait_to_read()
    print("execution end")
    duration = t1 - t0
    print('duration: {0}s'.format(duration))
    print('          {0}ms/operator'.format(duration*1000/args.iter_num))
