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
import sys

import mxnet as mx
mx.test_utils.set_default_context(mx.gpu(0))

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
# We import all tests from ../unittest/test_profiler.py
# They will be detected by test framework, as long as the current file has a different filename
from test_profiler import *


def test_gpu_memory_profiler_symbolic():
    iter_num = 5

    enable_profiler('test_profiler.json', False, False)
    profiler.set_state('run')

    with profiler.scope("tensordot"):
        A = mx.sym.Variable('A')
        B = mx.sym.Variable('B')
        C = mx.symbol.dot(A, B, name='dot')

    executor = C._simple_bind(mx.gpu(), 'write', A=(4096, 4096), B=(4096, 4096))

    a = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))
    b = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    for i in range(iter_num):
        executor.forward()
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
             'Requested Size' : str(4 * c.size)}]

    # Sample gpu_memory_profile.csv:
    # "Attribute Name","Requested Size","Device","Actual Size","Reuse?"
    # "<unk>:_zeros","67108864","0","67108864","0"
    # "<unk>:_zeros","67108864","0","67108864","0"
    # "tensordot:dot","67108864","0","67108864","1"
    # "tensordot:dot","67108864","0","67108864","1"
    # "tensordot:in_arg:A","67108864","0","67108864","0"
    # "tensordot:in_arg:B","67108864","0","67108864","0"
    # "nvml_amend","1074790400","0","1074790400","0"

    with open('gpu_memory_profile-pid_%d.csv' % (os.getpid()), mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for expected_alloc_entry in expected_alloc_entries:
            csv_file.seek(0)
            entry_found = False
            for row in csv_reader:
                if row['Attribute Name'] == expected_alloc_entry['Attribute Name']:
                    assert row['Requested Size'] == expected_alloc_entry['Requested Size'], \
                           "requested size={} is not equal to the expected size={}" \
                           .format(row['Requested Size'],
                                   expected_alloc_entry['Requested Size'])
                    entry_found = True
                    break
            assert entry_found, \
                   "Entry for attr_name={} has not been found" \
                   .format(expected_alloc_entry['Attribute Name'])


@pytest.mark.skipif(is_cd_run(), reason="flaky test - open issue #18564")
@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18564')
def test_gpu_memory_profiler_gluon():
    enable_profiler(profile_filename='test_profiler.json',
                    run=True, continuous_dump=True)
    profiler.set_state('run')

    model = nn.HybridSequential()
    model.add(nn.Dense(128, activation='tanh'))
    model.add(nn.Dropout(0.5))
    model.add(nn.Dense(64, activation='tanh'),
              nn.Dense(32, in_units=64))
    model.add(nn.Activation('relu'))
    model.initialize(ctx=mx.gpu())
    model.hybridize()

    inputs = mx.sym.var('data')

    with mx.autograd.record():
        out = model(mx.nd.zeros((16, 10), ctx=mx.gpu()))
    out.backward()
    mx.nd.waitall()
    profiler.set_state('stop')
    profiler.dump(True)

    # We are only checking for weight parameters here, also making sure that
    # there is no unknown entries in the memory profile.
    with open('gpu_memory_profile-pid_%d.csv' % (os.getpid()), mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(",".join(list(row.values())))
        for scope in ['in_arg', 'arg_grad']:
            for key, nd in model.collect_params().items():
                expected_arg_name = "%s:%s:" % (model.name, scope) + nd.name
                expected_arg_size = str(4 * np.prod(nd.shape))
                csv_file.seek(0)
                entry_found = False
                for row in csv_reader:
                    if row['Attribute Name'] == expected_arg_name:
                        assert row['Requested Size'] == expected_arg_size, \
                            "requested size={} is not equal to the expected size={}" \
                            .format(row['Requested Size'], expected_arg_size)
                        entry_found = True
                        break
                assert entry_found, \
                    "Entry for attr_name={} has not been found" \
                    .format(expected_arg_name)
        # Make sure that there is no unknown allocation entry.
        csv_file.seek(0)
        for row in csv_reader:
            if row['Attribute Name'] == "<unk>:unknown" or \
               row['Attribute Name'] == "<unk>:":
                assert False, "Unknown allocation entry has been encountered"
