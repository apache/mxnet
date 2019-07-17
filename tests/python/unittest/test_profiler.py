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
import logging
import mxnet as mx
from mxnet import profiler
import time
import os
import json
from collections import OrderedDict
from common import run_in_spawned_process
import unittest

def enable_profiler(profile_filename, run=True, continuous_dump=False, aggregate_stats=False):
    profiler.set_config(profile_symbolic=True,
                        profile_imperative=True,
                        profile_memory=True,
                        profile_api=True,
                        filename=profile_filename,
                        continuous_dump=continuous_dump,
                        aggregate_stats=aggregate_stats
                        )
    print('profile file save to {}'.format(profile_filename))
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

    print("execution begin")
    for i in range(iter_num):
        print("Iteration {}/{}".format(i + 1, iter_num))
        if i == begin_profiling_iter:
            t0 = time.clock()
            profiler.set_state('run')
        if i == end_profiling_iter:
            t1 = time.clock()
            profiler.set_state('stop')
        executor.forward()
        c = executor.outputs[0]
        c.wait_to_read()
    print("execution end")
    duration = t1 - t0
    print('duration: {0}s'.format(duration))
    print('          {0}ms/operator'.format(duration*1000/iter_num))
    profiler.dump(True)
    profiler.set_state('stop')


def test_profile_create_domain():
    enable_profiler('test_profile_create_domain.json')
    domain = profiler.Domain(name='PythonDomain')
    print("Domain created: {}".format(str(domain)))
    profiler.set_state('stop')


def test_profile_create_domain_dept():
    profiler.set_config(profile_symbolic=True, filename='test_profile_create_domain_dept.json')
    profiler.set_state('run')
    domain = profiler.Domain(name='PythonDomain')
    print("Domain created: {}".format(str(domain)))
    profiler.dump_profile()
    profiler.set_state('stop')

def test_profile_task():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    enable_profiler('test_profile_task.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_task')
    task = profiler.Task(python_domain, "test_profile_task")
    task.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    task.stop()
    print('run took: %.3f' % (stop - start))
    profiler.set_state('stop')


def test_profile_frame():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    enable_profiler('test_profile_frame.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_frame')
    frame = profiler.Frame(python_domain, "test_profile_frame")
    frame.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    frame.stop()
    print('run took: %.3f' % (stop - start))
    profiler.set_state('stop')


def test_profile_event(do_enable_profiler=True):
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    if do_enable_profiler is True:
        enable_profiler('test_profile_event.json')
    event = profiler.Event("test_profile_event")
    event.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    event.stop()
    print('run took: %.3f' % (stop - start))
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

    def doLog(counter):
        template, objects = makeParams()
        range_size = 100000
        for i in range(range_size):
            if i <= range_size / 2:
                counter += 1
            else:
                counter -= 1
            logging.info(template.format(*objects))

    if do_enable_profiler is True:
        enable_profiler('test_profile_counter.json')
    python_domain = profiler.Domain('PythonDomain::test_profile_counter')
    counter = profiler.Counter(python_domain, "PythonCounter::test_profile_counter")
    counter.set_value(5)
    counter += 1
    start = time.time()
    doLog(counter)
    stop = time.time()
    print('run took: %.3f' % (stop - start))
    if do_enable_profiler is True:
        profiler.set_state('stop')


def test_continuous_profile_and_instant_marker():
    file_name = 'test_continuous_profile_and_instant_marker.json'
    enable_profiler(file_name, True, True, True)
    python_domain = profiler.Domain('PythonDomain::test_continuous_profile')
    last_file_size = 0
    for i in range(5):
        profiler.Marker(python_domain, "StartIteration-" + str(i)).mark('process')
        print("{}...".format(i))
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
    print(debug_str)
    profiler.set_state('stop')

def test_aggregate_stats_valid_json_return():
    file_name = 'test_aggregate_stats_json_return.json'
    enable_profiler(file_name, True, True, True)
    test_profile_event(False)
    debug_str = profiler.dumps(format = 'json')
    assert(len(debug_str) > 0)
    target_dict = json.loads(debug_str)
    assert "Memory" in target_dict and "Time" in target_dict and "Unit" in target_dict
    profiler.set_state('stop')

def test_aggregate_stats_sorting():
    sort_by_options = {'avg': "Avg", 'min': "Min", 'max': "Max", 'count': "Count"}
    ascending_options = [False, True]
    def check_ascending(lst, asc):
        assert(lst == sorted(lst, reverse = not asc))

    def check_sorting(debug_str, sort_by, ascending):
        target_dict = json.loads(debug_str, object_pairs_hook=OrderedDict)
        lst = []
        for domain_name, domain in target_dict['Time'].items():
            lst = [item[sort_by_options[sort_by]] for item_name, item in domain.items()]
            check_ascending(lst, ascending)
        for domain_name, domain in target_dict['Memory'].items():
            lst = [item[sort_by_options[sort_by]] for item_name, item in domain.items()]
            check_ascending(lst, ascending)

    file_name = 'test_aggregate_stats_sorting.json'
    enable_profiler(file_name, True, True, True)
    test_profile_event(False)
    for sb in sort_by_options:
        for asc in ascending_options:
            debug_str = profiler.dumps(format = 'json', sort_by = sb, ascending = asc)
            check_sorting(debug_str, sb, asc)
    profiler.set_state('stop')

def test_aggregate_duplication():
    file_name = 'test_aggregate_duplication.json'
    enable_profiler(profile_filename = file_name, run=True, continuous_dump=True, \
                    aggregate_stats=True)
    inp = mx.nd.zeros(shape=(100, 100))
    y = mx.nd.sqrt(inp)
    inp = inp + 1
    inp = inp + 1
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format = 'json')
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
    
def test_custom_operator_profiling(seed = None, file_name = None):
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
    enable_profiler(profile_filename = file_name, run=True, continuous_dump=True,\
                    aggregate_stats=True)
    x = mx.nd.array([0, 1, 2, 3])
    x.attach_grad()
    with mx.autograd.record():
        y = mx.nd.Custom(x, op_type='MySigmoid')
    y.backward()
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format = 'json')
    target_dict = json.loads(debug_str)
    assert 'Time' in target_dict and 'Custom Operator' in target_dict['Time'] \
        and 'MySigmoid::pure_python' in target_dict['Time']['Custom Operator'] \
        and '_backward_MySigmoid::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MySigmoid::_zeros' in target_dict['Time']['Custom Operator']
    profiler.set_state('stop')

def check_custom_operator_profiling_multiple_custom_ops_output(debug_str):
    target_dict = json.loads(debug_str)
    '''
    We are calling _plus_scalar within MyAdd1 and MyAdd2 and outside both the custom 
    operators, so in aggregate stats we should have three different kinds of 
    _plus_scalar under domains "Custom Operator" and "operator"
    '''
    assert 'Time' in target_dict and 'Custom Operator' in target_dict['Time'] \
        and 'MyAdd1::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd2::pure_python' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd1::_plus_scalar' in target_dict['Time']['Custom Operator'] \
        and 'MyAdd2::_plus_scalar' in target_dict['Time']['Custom Operator'] \
        and '_plus_scalar' not in target_dict['Time']['Custom Operator'] \
        and '[_plus_scalar]' not in target_dict['Time']['Custom Operator'] \
        and 'operator' in target_dict['Time'] \
        and ('_plus_scalar' in target_dict['Time']['operator'] or \
            '[_plus_scalar]' in target_dict['Time']['operator'])

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

    enable_profiler(profile_filename = file_name, run=True, continuous_dump=True,\
                    aggregate_stats=True)
    inp = mx.nd.zeros(shape=(100, 100))
    if mode == 'imperative':
        x = inp + 1
        y = mx.nd.Custom(inp, op_type='MyAdd1')
        z = mx.nd.Custom(inp, op_type='MyAdd2')
    elif mode == 'symbolic':
        a = mx.symbol.Variable('a')
        b = a + 1
        c = mx.symbol.Custom(data=a, op_type='MyAdd1')
        d = mx.symbol.Custom(data=a, op_type='MyAdd2')
        b.bind(mx.cpu(), {'a': inp}).forward()
        c.bind(mx.cpu(), {'a': inp}).forward()
        d.bind(mx.cpu(), {'a': inp}).forward()
    mx.nd.waitall()
    profiler.dump(False)
    debug_str = profiler.dumps(format = 'json')
    check_custom_operator_profiling_multiple_custom_ops_output(debug_str)
    profiler.set_state('stop')

def test_custom_operator_profiling_multiple_custom_ops_imperative():
    custom_operator_profiling_multiple_custom_ops(None, 'imperative', \
            'test_custom_operator_profiling_multiple_custom_ops_imperative.json')

def test_custom_operator_profiling_multiple_custom_ops_symbolic():
    custom_operator_profiling_multiple_custom_ops(None, 'symbolic', \
            'test_custom_operator_profiling_multiple_custom_ops_symbolic.json')

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

if __name__ == '__main__':
    import nose
    nose.runmodule()
