# pylint: skip-file
import mxnet as mx
import numpy as np

shape = (4, 4)
keys = [5, 7, 11]
def init_kvstore():
    """init kvstore """
    mx.kvstore.start()
    # single
    mx.kvstore.init(3, mx.narray.zeros(shape))
    # list
    mx.kvstore.init(keys, [mx.narray.zeros(shape)] * len(keys))

def stop_kvstore():
    """stop kvstore """
    mx.kvstore.stop()

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)

def test_single_kv_pair():
    """single key-value pair push & pull"""

    init_kvstore()

    mx.kvstore.push(3, mx.narray.ones(shape))
    val = mx.narray.empty(shape)
    mx.kvstore.pull(3, out = val)
    check_diff_to_scalar(val, 1)

    stop_kvstore()

def test_list_kv_pair():
    """list key-value pair push & pull"""

    init_kvstore()

    mx.kvstore.push(keys, [mx.narray.ones(shape)*4] * len(keys))
    val = [mx.narray.empty(shape)] * len(keys)
    mx.kvstore.pull(keys, out = val)
    for v in val:
        check_diff_to_scalar(v, 4)

    stop_kvstore()

def test_aggregator():
    """aggregate value on muliple devices"""

    init_kvstore()

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [mx.narray.ones(shape, d) for d in devs]

    mx.kvstore.push(3, vals)
    mx.kvstore.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.narray.ones(shape, d)*2.0 for d in devs]] * len(keys)
    mx.kvstore.push(keys, vals)
    mx.kvstore.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * 2.0)

    stop_kvstore()

def updater(key, recv, local):
    """use updater: +="""
    local += recv

def test_updater():
    """updater"""

    init_kvstore()
    mx.kvstore.set_updater(updater)

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [mx.narray.ones(shape, d) for d in devs]

    mx.kvstore.push(3, vals)
    mx.kvstore.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.narray.ones(shape, d) for d in devs]] * len(keys)

    num_push = 4
    for i in range(num_push):
        mx.kvstore.push(keys, vals)

    mx.kvstore.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * num_push)

    stop_kvstore()

if __name__ == '__main__':
    test_single_kv_pair()
    test_list_kv_pair()
    test_aggregator()
    test_updater()
