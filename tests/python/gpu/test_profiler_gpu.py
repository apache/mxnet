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

import csv
import os
import sys

import numpy as np
import mxnet as mx
mx.test_utils.set_default_device(mx.gpu(0))

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
# We import all tests from ../unittest/test_profiler.py
# They will be detected by test framework, as long as the current file has a different filename
from test_profiler import *

@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18564')
def test_gpu_memory_profiler_symbolic():
    enable_profiler('test_profiler.json')
    profiler.set_state('run')

    with profiler.scope("tensordot"):
        A = mx.sym.Variable('A')
        B = mx.sym.Variable('B')
        C = mx.symbol.dot(A, B, name="dot")

    executor = C._simple_bind(mx.gpu(), 'write', A=(1024, 2048), B=(2048, 4096))

    with profiler.scope("init"):
        a = mx.random.uniform(-1.0, 1.0, shape=(1024, 2048))
        b = mx.random.uniform(-1.0, 1.0, shape=(2048, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    executor.forward()
    executor.backward()
    c = executor.outputs[0]
    mx.nd.waitall()
    profiler.set_state('stop')
    profiler.dump(True)

    expected_alloc_entries = [
            {'Attribute Name' : 'tensordot:in_arg:A',
             'Requested Size' : str(4 * a.size)},
            {'Attribute Name' : 'tensordot:in_arg:B',
             'Requested Size' : str(4 * b.size)},
            {'Attribute Name' : 'tensordot:dot',
             'Requested Size' : str(4 * c.size)},
            {'Attribute Name' : 'tensordot:dot_backward',
             'Requested Size' : str(4 * a.size)},
            {'Attribute Name' : 'tensordot:dot_backward',
             'Requested Size' : str(4 * b.size)},
            {'Attribute Name' : 'init:_random_uniform',
             'Requested Size' : str(4 * a.size)},
            {'Attribute Name' : 'init:_random_uniform',
             'Requested Size' : str(4 * b.size)}]

    # Sample gpu_memory_profile.csv:
    # "Attribute Name","Requested Size","Device","Actual Size","Reuse?"
    # <unk>:_head_grad_0,16777216,0,16777216,0
    # init:_random_uniform,33554432,0,33554432,1
    # init:_random_uniform,8388608,0,8388608,1
    # resource:temp_space (sample_op.h +365),8,0,4096,0
    # symbol:arg_grad:unknown,8388608,0,8388608,0
    # symbol:arg_grad:unknown,33554432,0,33554432,0
    # tensordot:dot,16777216,0,16777216,0
    # tensordot:dot_backward,8388608,0,8388608,0
    # tensordot:dot_backward,33554432,0,33554432,0
    # tensordot:in_arg:A,8388608,0,8388608,0
    # tensordot:in_arg:B,33554432,0,33554432,0

    with open(f'gpu_memory_profile-pid_{os.getpid()}.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(",".join(list(row.values())))
        for expected_alloc_entry in expected_alloc_entries:
            csv_file.seek(0)
            entry_found = False
            for row in csv_reader:
                if row['Attribute Name'] == expected_alloc_entry['Attribute Name'] and \
                   row['Requested Size'] == expected_alloc_entry['Requested Size']:
                    entry_found = True
                    break
            assert entry_found, \
                    "Entry for (attr_name={}, alloc_size={}) has not been found" \
                    .format(expected_alloc_entry['Attribute Name'],
                            expected_alloc_entry['Requested Size'])
        # Make sure that there is no unknown allocation entry.
        csv_file.seek(0)
        for row in csv_reader:
            if row['Attribute Name'] == "<unk>:unknown" or \
               row['Attribute Name'] == "<unk>:":
                assert False, "Unknown allocation entry has been encountered"


@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18564')
def test_gpu_memory_profiler_gluon():
    enable_profiler(profile_filename='test_profiler.json')
    profiler.set_state('run')

    model = nn.HybridSequential()
    model.add(nn.Dense(128, activation='tanh'))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh'),
              nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))
    model.initialize(device=mx.gpu())
    model.hybridize()

    with mx.autograd.record():
        out = model(mx.np.zeros((16, 10), device=mx.gpu()))
    out.backward()
    mx.npx.waitall()
    profiler.set_state('stop')
    profiler.dump(True)

    # Sample gpu_memory_profile.csv:
    # "Attribute Name","Requested Size","Device","Actual Size","Reuse?"
    # <unk>:in_arg:data,640,0,4096,0
    # hybridsequential:activation0:hybridsequential_activation0_fwd,2048,0,4096,0
    # hybridsequential:activation0:hybridsequential_activation0_fwd_backward,8192,0,8192,0
    # hybridsequential:activation0:hybridsequential_activation0_fwd_head_grad,2048,0,4096,0
    # hybridsequential:dense0:activation0:hybridsequential_dense0_activation0_fwd,8192,0,8192,0
    # hybridsequential:dense0:arg_grad:bias,512,0,4096,0
    # hybridsequential:dense0:arg_grad:weight,5120,0,8192,0
    # hybridsequential:dense0:hybridsequential_dense0_fwd,8192,0,8192,0
    # hybridsequential:dense0:in_arg:bias,512,0,4096,0
    # hybridsequential:dense0:in_arg:weight,5120,0,8192,0
    # hybridsequential:dense1:activation0:hybridsequential_dense1_activation0_fwd,4096,0,4096,0
    # hybridsequential:dense1:arg_grad:bias,256,0,4096,0
    # hybridsequential:dense1:arg_grad:weight,32768,0,32768,0
    # hybridsequential:dense1:hybridsequential_dense1_fwd,4096,0,4096,0
    # hybridsequential:dense1:in_arg:bias,256,0,4096,0
    # hybridsequential:dense1:in_arg:weight,32768,0,32768,0
    # hybridsequential:dense2:arg_grad:bias,128,0,4096,0
    # hybridsequential:dense2:arg_grad:weight,8192,0,8192,0
    # hybridsequential:dense2:hybridsequential_dense2_fwd_backward,4096,0,4096,1
    # hybridsequential:dense2:in_arg:bias,128,0,4096,0
    # hybridsequential:dense2:in_arg:weight,8192,0,8192,0
    # hybridsequential:dropout0:hybridsequential_dropout0_fwd,8192,0,8192,0
    # hybridsequential:dropout0:hybridsequential_dropout0_fwd,8192,0,8192,0
    # resource:cudnn_dropout_state (dropout-inl.h +256),1474560,0,1474560,0
    # resource:temp_space (fully_connected-inl.h +316),15360,0,16384,0

    # We are only checking for weight parameters here, also making sure that
    # there is no unknown entries in the memory profile.
    with open(f'gpu_memory_profile-pid_{os.getpid()}.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(",".join(list(row.values())))
        for param in model.collect_params().values():
            expected_arg_name = f"{param.var().attr('__profiler_scope__')}in_arg:" + \
                                param.name
            expected_arg_size = str(4 * np.prod(param.shape))
            csv_file.seek(0)
            entry_found = False
            for row in csv_reader:
                if row['Attribute Name'] == expected_arg_name and \
                   row['Requested Size'] == expected_arg_size:
                    entry_found = True
                    break
            assert entry_found, \
                    "Entry for (attr_name={}, alloc_size={}) has not been found" \
                        .format(expected_arg_name,
                                expected_arg_size)
        # Make sure that there is no unknown allocation entry.
        csv_file.seek(0)
        for row in csv_reader:
            if row['Attribute Name'] == "<unk>:unknown" or \
               row['Attribute Name'] == "<unk>:":
                assert False, "Unknown allocation entry has been encountered"
