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
import time
import os
import csv
import json
import unittest
import numpy as np
from collections import OrderedDict

import mxnet as mx
from mxnet import profiler
from mxnet.gluon import nn
from common import run_in_spawned_process


def enable_profiler(profile_filename, run=True, continuous_dump=False, aggregate_stats=False):
    profiler.set_config(profile_symbolic=True,
                        profile_imperative=True,
                        profile_memory=True,
                        profile_api=True,
                        filename=profile_filename,
                        continuous_dump=continuous_dump,
                        aggregate_stats=aggregate_stats)
    if run is True:
        profiler.set_state('run')


def test_profiler():
    iter_num = 5
    begin_profiling_iter = 2
    end_profiling_iter = 4

    enable_profiler('test_profiler.json', False, False)

    A = mx.sym.Variable('A')
    B = mx.sym.Variable('B')
    C = mx.symbol.dot(A, B)

    executor = C.simple_bind(mx.cpu(1), 'write', A=(4096, 4096), B=(4096, 4096))

    a = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))
    b = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    for i in range(iter_num):
        if i == begin_profiling_iter:
            t0 = time.clock()
            profiler.set_state('run')
        if i == end_profiling_iter:
            t1 = time.clock()
            profiler.set_state('stop')
        executor.forward()
        c = executor.outputs[0]
        c.wait_to_read()

    duration = t1 - t0
    profiler.dump(True)
    profiler.set_state('stop')


def test_profile_create_domain():
    enable_profiler('test_profile_create_domain.json')
    domain = profiler.Domain(name='PythonDomain')
    profiler.set_state('stop')


def test_profile_create_domain_dept():
    profiler.set_config(profile_symbolic=True, filename='test_profile_create_domain_dept.json')
    profiler.set_state('run')
    domain = profiler.Domain(name='PythonDomain')
    profiler.dump()
    profiler.set_state('stop')


def test_profile_task():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def get_log():
        template, objects = makeParams()
        logs = []
        for _ in range(10):
            logs.append(template.format(*objects))
        return logs

    enable_profiler('test_profile_task.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_task')
    task = profiler.Task(python_domain, "test_profile_task")
    task.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    assert len(get_log()) == 10
    var.asnumpy()
    stop = time.time()
    task.stop()
    profiler.set_state('stop')


def test_profile_frame():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def get_log():
        template, objects = makeParams()
        logs = []
        for _ in range(100000):
            logs.append(template.format(*objects))
        return logs

    enable_profiler('test_profile_frame.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_frame')
    frame = profiler.Frame(python_domain, "test_profile_frame")
    frame.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    assert len(get_log()) == 100000
    var.asnumpy()
    stop = time.time()
    frame.stop()
    assert stop > start
    profiler.set_state('stop')


def test_profile_event(do_enable_profiler=True):
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def get_log():
        template, objects = makeParams()
        logs = []
        for _ in range(100000):
            logs.append(template.format(*objects))
        return logs

    if do_enable_profiler is True:
        enable_profiler('test_profile_event.json')
    event = profiler.Event("test_profile_event")
    event.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    assert len(get_log()) == 100000
    var.asnumpy()
    stop = time.time()
    event.stop()
    assert stop > start
    if do_enable_profiler is True:
        profiler.set_state('stop')


def test_profile_tune_pause_resume():
    enable_profiler('test_profile_tune_pause_resume.json')
    profiler.pause()
    # "test_profile_task" should *not* show up in tuning analysis
    test_profile_task()
    profiler.resume()
    # "test_profile_event" should show up in tuning analysis
    test_profile_event()
    profiler.pause()
    profiler.set_state('stop')


def test_profile_counter(do_enable_profiler=True):
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def get_log(counter):
        template, objects = makeParams()
        range_size = 100000
        logs = []
        for i in range(range_size):
            if i <= range_size / 2:
                counter += 1
            else:
                counter -= 1
            logs.append(template.format(*objects))
        return logs

    if do_enable_profiler is True:
        enable_profiler('test_profile_counter.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_counter')
    counter = profiler.Counter(python_domain, "PythonCounter::test_profile_counter")
    counter.set_value(5)
    counter += 1
    start = time.time()
    log = get_log(counter)
    assert len(log) == 100000
    assert log[0].count('foo') == 50
    stop = time.time()
    assert stop > start
    if do_enable_profiler is True:
        profiler.set_state('stop')


def test_continuous_profile_and_instant_marker():
    file_name = 'test_continuous_profile_and_instant_marker.json'
    enable_profiler(file_name, True, True, True)
    python_domain = profiler.Domain('PythonDomain::test_continuous_profile')
    last_file_size = 0
    for i in range(5):
        profiler.Marker(python_domain, "StartIteration-" + str(i)).mark('process')
        test_profile_event(False)
        test_profile_counter(False)
        profiler.dump(False)
        # File size should keep increasing
        new_file_size = os.path.getsize(file_name)
        assert new_file_size >= last_file_size
        last_file_size = new_file_size
    profiler.dump(False)
    debug_str = profiler.dumps()
    assert(len(debug_str) > 0)
    profiler.set_state('stop')


def test_aggregate_stats_valid_json_return():
    file_name = 'test_aggregate_stats_json_return.json'
    enable_profiler(file_name, True, True, True)
    test_profile_event(False)
    debug_str = profiler.dumps(format = 'json')
    assert(len(debug_str) > 0)
    target_dict = json.loads(debug_str)
    assert 'Memory' in target_dict and 'Time' in target_dict and 'Unit' in target_dict
    profiler.set_state('stop')


def test_aggregate_stats_sorting():
    sort_by_options = {'total': 'Total', 'avg': 'Avg', 'min': 'Min',\
        'max': 'Max', 'count': 'Count'}
    ascending_options = [False, True]

    def check_ascending(lst, asc):
        assert(lst == sorted(lst, reverse = not asc))

    def check_sorting(debug_str, sort_by, ascending):
        target_dict = json.loads(debug_str, object_pairs_hook=OrderedDict)
        lst = []
        for domain_name, domain in target_dict['Time'].items():
            lst = [item[sort_by_options[sort_by]] for item_name, item in domain.items()]
            check_ascending(lst, ascending)
        # Memory items do not have stat 'Total'
        if sort_by != 'total':
            for domain_name, domain in target_dict['Memory'].items():
                lst = [item[sort_by_options[sort_by]] for item_name, item in domain.items()]
                check_ascending(lst, ascending)

    file_name = 'test_aggregate_stats_sorting.json'
    enable_profiler(file_name, True, True, True)
    test_profile_event(False)
    for sb in sort_by_options:
        for asc in ascending_options:
            debug_str = profiler.dumps(format='json', sort_by=sb, ascending=asc)
            check_sorting(debug_str, sb, asc)
    profiler.set_state('stop')


def test_aggregate_duplication():
    file_name = 'test_aggregate_duplication.json'
    enable_profiler(profile_filename=file_name, run=True, continuous_dump=True, \
                    aggregate_stats=True)
    # clear aggregate stats
    profiler.dumps(reset=True)
    inp = mx.nd.zeros(shape=(100, 100))
    y = mx.nd.sqrt(inp)
    inp = inp + 1
    inp = inp + 1
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format='json')
    target_dict = json.loads(debug_str)
    assert 'Time' in target_dict and 'operator' in target_dict['Time'] \
        and 'sqrt' in target_dict['Time']['operator'] \
        and 'Count' in target_dict['Time']['operator']['sqrt'] \
        and '_plus_scalar' in target_dict['Time']['operator'] \
        and 'Count' in target_dict['Time']['operator']['_plus_scalar']
    # they are called once and twice respectively
    assert target_dict['Time']['operator']['sqrt']['Count'] == 1
    assert target_dict['Time']['operator']['_plus_scalar']['Count'] == 2
    profiler.set_state('stop')


def test_custom_operator_profiling(seed=None, file_name=None):
    class Sigmoid(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            x = in_data[0].asnumpy()
            import numpy as np
            y = 1.0 / (1.0 + np.exp(-x))
            self.assign(out_data[0], req[0], mx.nd.array(y))
            # Create a dummy matrix using nd.zeros. Test if 'MySigmoid::_zeros' is in dump file
            dummy = mx.nd.zeros(shape=(100, 100))

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            y = out_data[0].asnumpy()
            dy = out_grad[0].asnumpy()
            dx = dy*(1.0 - y)*y
            self.assign(in_grad[0], req[0], mx.nd.array(dx))

    @mx.operator.register('MySigmoid')
    class SigmoidProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(SigmoidProp, self).__init__(True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shapes):
            data_shape = in_shapes[0]
            output_shape = data_shape
            return (data_shape,), (output_shape,), ()

        def create_operator(self, ctx, in_shapes, in_dtypes):
            return Sigmoid()

    if file_name is None:
        file_name = 'test_custom_operator_profiling.json'
    enable_profiler(profile_filename=file_name, run=True, continuous_dump=True,\
                    aggregate_stats=True)
    # clear aggregate stats
    profiler.dumps(reset=True)
    x = mx.nd.array([0, 1, 2, 3])
    x.attach_grad()
    with mx.autograd.record():
        y = mx.nd.Custom(x, op_type='MySigmoid')
    y.backward()
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format='json')
    target_dict = json.loads(debug_str)
    assert 'Time' in target_dict and 'Custom Operator' in target_dict['Time'] \
        and 'MySigmoid::pure_python' in target_dict['Time']['Custom Operator'] \
        and '_backward_MySigmoid::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MySigmoid::_zeros' in target_dict['Time']['Custom Operator']
    profiler.set_state('stop')


def check_custom_operator_profiling_multiple_custom_ops_output(debug_str):
    target_dict = json.loads(debug_str)
    assert 'Time' in target_dict and 'Custom Operator' in target_dict['Time'] \
        and 'MyAdd1::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd2::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd1::_plus_scalar' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd2::_plus_scalar' in target_dict['Time']['Custom Operator']


def custom_operator_profiling_multiple_custom_ops(seed, mode, file_name):
    class MyAdd(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], in_data[0] + 1)

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], out_grad[0])

    @mx.operator.register('MyAdd1')
    class MyAdd1Prop(mx.operator.CustomOpProp):
        def __init__(self):
            super(MyAdd1Prop, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            # inputs, outputs, aux
            return [in_shape[0]], [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return MyAdd()

    @mx.operator.register('MyAdd2')
    class MyAdd2Prop(mx.operator.CustomOpProp):
        def __init__(self):
            super(MyAdd2Prop, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            # inputs, outputs, aux
            return [in_shape[0]], [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return MyAdd()

    enable_profiler(profile_filename=file_name, run=True, continuous_dump=True,\
                    aggregate_stats=True)
    # clear aggregate stats
    profiler.dumps(reset=True)
    inp = mx.nd.zeros(shape=(100, 100))
    if mode == 'imperative':
        y = mx.nd.Custom(inp, op_type='MyAdd1')
        z = mx.nd.Custom(inp, op_type='MyAdd2')
    elif mode == 'symbolic':
        a = mx.symbol.Variable('a')
        b = mx.symbol.Custom(data=a, op_type='MyAdd1')
        c = mx.symbol.Custom(data=a, op_type='MyAdd2')
        y = b.bind(mx.cpu(), {'a': inp})
        z = c.bind(mx.cpu(), {'a': inp})
        yy = y.forward()
        zz = z.forward()
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format='json')
    check_custom_operator_profiling_multiple_custom_ops_output(debug_str)
    profiler.set_state('stop')


def test_custom_operator_profiling_multiple_custom_ops_symbolic():
    custom_operator_profiling_multiple_custom_ops(None, 'symbolic', \
            'test_custom_operator_profiling_multiple_custom_ops_symbolic.json')


def test_custom_operator_profiling_multiple_custom_ops_imperative():
    custom_operator_profiling_multiple_custom_ops(None, 'imperative', \
            'test_custom_operator_profiling_multiple_custom_ops_imperative.json')


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/15406")
def test_custom_operator_profiling_naive_engine():
    # run the three tests above using Naive Engine
    run_in_spawned_process(test_custom_operator_profiling, \
            {'MXNET_ENGINE_TYPE' : "NaiveEngine"}, \
            'test_custom_operator_profiling_naive.json')
    run_in_spawned_process(custom_operator_profiling_multiple_custom_ops, \
            {'MXNET_ENGINE_TYPE' : "NaiveEngine"}, 'imperative', \
            'test_custom_operator_profiling_multiple_custom_ops_imperative_naive.json')
    run_in_spawned_process(custom_operator_profiling_multiple_custom_ops, \
            {'MXNET_ENGINE_TYPE' : "NaiveEngine"}, 'symbolic', \
            'test_custom_operator_profiling_multiple_custom_ops_symbolic_naive.json')


@unittest.skipIf(mx.context.num_gpus() == 0, "GPU memory profiler records allocation on GPUs only")
def test_gpu_memory_profiler_symbolic():
    iter_num = 5

    enable_profiler('test_profiler.json', False, False)
    profiler.set_state('run')

    with profiler.Scope("tensordot"):
        A = mx.sym.Variable('A')
        B = mx.sym.Variable('B')
        C = mx.symbol.dot(A, B, name='dot')

    executor = C.simple_bind(mx.gpu(), 'write', A=(4096, 4096), B=(4096, 4096))

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
            {'Attribute Name' : 'tensordot:arg_grad:A',
             'Requested Size' : str(4 * a.size)},
            {'Attribute Name' : 'tensordot:arg_grad:B',
             'Requested Size' : str(4 * b.size)},
            {'Attribute Name' : 'tensordot:dot',
             'Requested Size' : str(4 * c.size)},
            {'Attribute Name' : 'tensordot:dot_head_grad',
             'Requested Size' : str(4 * c.size)}]

    # Sample gpu_memory_profile.csv:
    # "Attribute Name","Requested Size","Device","Actual Size","Reuse?"
    # "tensordot:arg_grad:A","67108864","0","67108864","0"
    # "tensordot:arg_grad:B","67108864","0","67108864","0"
    # "tensordot:dot","67108864","0","67108864","0"
    # "tensordot:dot_head_grad","67108864","0","67108864","0"
    # "tensordot:in_arg:A","67108864","0","67108864","0"
    # "tensordot:in_arg:B","67108864","0","67108864","0"

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


@unittest.skipIf(mx.context.num_gpus() == 0, "GPU memory profiler records allocation on GPUs only")
def test_gpu_memory_profiler_gluon():
    enable_profiler(profile_filename='test_profiler.json',
                    run=True, continuous_dump=True)
    profiler.set_state('run')

    model = nn.HybridSequential(prefix='net_')
    with model.name_scope():
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

    # Sample gpu_memory_profiler.csv
    # "Attribute Name","Requested Size","Device","Actual Size","Reuse?"
    # "<unk>:in_arg:data","640","0","4096","0"
    # "net:arg_grad:net_dense0_bias","512","0","4096","0"
    # "net:arg_grad:net_dense0_weight","5120","0","8192","0"
    # "net:arg_grad:net_dense1_bias","256","0","4096","0"
    # "net:arg_grad:net_dense1_weight","32768","0","32768","0"
    # "net:arg_grad:net_dense2_bias","128","0","4096","0"
    # "net:arg_grad:net_dense2_weight","8192","0","8192","0"
    # "net:dense0:net_dense0_fwd","8192","0","8192","0"
    # "net:dense0:tanh:net_dense0_tanh_fwd","8192","0","8192","0"
    # "net:dense1:net_dense1_fwd","4096","0","4096","0"
    # "net:dense1:tanh:net_dense1_tanh_fwd","4096","0","4096","0"
    # "net:dense2:net_dense2_fwd","2048","0","4096","0"
    # "net:dense2:net_dense2_fwd_backward","4096","0","4096","0"
    # "net:dropout0:net_dropout0_fwd","8192","0","8192","0"
    # "net:dropout0:net_dropout0_fwd","8192","0","8192","0"
    # "net:in_arg:net_dense0_bias","512","0","4096","0"
    # "net:in_arg:net_dense0_weight","5120","0","8192","0"
    # "net:in_arg:net_dense1_bias","256","0","4096","0"
    # "net:in_arg:net_dense1_weight","32768","0","32768","0"
    # "net:in_arg:net_dense2_bias","128","0","4096","0"
    # "net:in_arg:net_dense2_weight","8192","0","8192","0"
    # "net:relu0:net_relu0_fwd","2048","0","4096","0"
    # "net:relu0:net_relu0_fwd_backward","8192","0","8192","0"
    # "net:relu0:net_relu0_fwd_head_grad","2048","0","4096","0"
    # "resource:cudnn_dropout_state (dropout-inl.h +258)","1671168","0","1671168","0"
    # "resource:temp_space (fully_connected-inl.h +316)","34816","0","36864","0"

    # We are only checking for weight parameters here, also making sure that
    # there is no unknown entries in the memory profile.
    with open('gpu_memory_profile-pid_%d.csv' % (os.getpid()), mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for scope in ['in_arg', 'arg_grad']:
            for key, nd in model.collect_params().items():
                expected_arg_name = "net:%s:" % scope + key
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

