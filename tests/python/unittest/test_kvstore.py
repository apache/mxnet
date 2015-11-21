# pylint: skip-file
import mxnet as mx
import numpy as np

shape = (4, 4)
keys = [5, 7, 11]
def init_kv():
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init(3, mx.nd.zeros(shape))
    # list
    kv.init(keys, [mx.nd.zeros(shape)] * len(keys))
    return kv


def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)

def test_single_kv_pair():
    """single key-value pair push & pull"""

    kv = init_kv()
    kv.push(3, mx.nd.ones(shape))
    val = mx.nd.empty(shape)
    kv.pull(3, out = val)
    check_diff_to_scalar(val, 1)

def test_init():
    """test init"""
    kv = mx.kv.create()
    kv.init(3, mx.nd.ones(shape)*4)
    a = mx.nd.zeros(shape)
    kv.pull(3, out=a)
    check_diff_to_scalar(a, 4)

def test_list_kv_pair():
    """list key-value pair push & pull"""

    kv = init_kv()

    kv.push(keys, [mx.nd.ones(shape)*4] * len(keys))
    val = [mx.nd.empty(shape)] * len(keys)
    kv.pull(keys, out = val)
    for v in val:
        check_diff_to_scalar(v, 4)


def test_aggregator():
    """aggregate value on muliple devices"""

    kv = init_kv()

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    kv.push(3, vals)
    kv.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * len(keys)
    kv.push(keys, vals)
    kv.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * 2.0)


def updater(key, recv, local):
    """use updater: +="""
    local += recv

def test_updater(dev = 'cpu'):
    """updater"""

    kv = init_kv()
    kv._set_updater(updater)

    # devices
    num_devs = 4
    devs = [mx.Context(dev, i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    kv.push(3, vals)
    kv.pull(3, out = vals)

    for v in vals:
        check_diff_to_scalar(v, num_devs)

    # list
    vals = [[mx.nd.ones(shape, d) for d in devs]] * len(keys)

    num_push = 4
    for i in range(num_push):
        kv.push(keys, vals)

    kv.pull(keys, out = vals)

    for vv in vals:
        for v in vv:
            check_diff_to_scalar(v, num_devs * num_push)

def test_get_type():
    kvtype = 'local_allreduce_cpu'
    kv = mx.kv.create(kvtype)
    assert kv.type == kvtype

if __name__ == '__main__':
    test_init()
    test_get_type()
    test_single_kv_pair()
    test_list_kv_pair()
    test_aggregator()
    test_updater()
