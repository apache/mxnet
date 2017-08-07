from __future__ import print_function
import mxnet as mx
from mxnet import profiler
import time
import numpy as np

def test_profiler():
    profile_filename = "test_profile.json"
    iter_num = 100
    begin_profiling_iter = 50
    end_profiling_iter = 50


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

    flag = False
    print("execution begin")
    for i in range(iter_num):
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

if __name__ == '__main__':
    test_profiler()
