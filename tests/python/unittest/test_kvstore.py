# pylint: skip-file
import mxnet as mx
import numpy as np

shape = (4, 4)
keys = [5, 7, 11]
def init_kv():
    """init kv """
    mx.kv.start()
    # single
    mx.kv.init(3, mx.nd.zeros(shape))
    # list
    mx.kv.init(keys, [mx.nd.zeros(shape)] * len(keys))

def stop_kv():
    """stop kv """
    mx.kv.stop()

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)

def test_single_kv_pair():
    """single key-value pair push & pull"""

    init_kv()

    mx.kv.push(3, mx.nd.ones(shape))
    val = mx.nd.empty(shape)
    mx.kv.pull(3, out = val)
    check_diff_to_scalar(val, 1)

    stop_kv()

def test_list_kv_pair():
    """list key-value pair push & pull"""

    init_kv()

    mx.kv.push(keys, [mx.nd.ones(shape)*4] * len(keys))
    val = [mx.nd.empty(shape)] * len(keys)
    mx.kv.pull(keys, out = val)
    for v in val:
        check_diff_to_scalar(v, 4)

    stop_kv()

def test_aggregator():
    """aggregate value on muliple devices"""

    init_kv()

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    mx.kv.push(3, vals)
    mx.kv.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * len(keys)
    mx.kv.push(keys, vals)
    mx.kv.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * 2.0)

    stop_kv()

def updater(key, recv, local):
    """use updater: +="""
    local += recv

def test_updater(dev = 'cpu'):
    """updater"""

    init_kv()
    mx.kv.set_updater(updater)

    # devices
    num_devs = 4
    devs = [mx.Context(dev, i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    mx.kv.push(3, vals)
    mx.kv.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.nd.ones(shape, d) for d in devs]] * len(keys)

    num_push = 4
    for i in range(num_push):
        mx.kv.push(keys, vals)

    mx.kv.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * num_push)

    stop_kv()

if __name__ == '__main__':
    test_single_kv_pair()
    test_list_kv_pair()
    test_aggregator()
    test_updater()
    # test_updater('gpu')
