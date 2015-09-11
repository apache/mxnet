# pylint: skip-file
import mxnet as mx
import numpy as np

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)

def test_aggregator():

    mx.kvstore.start()

    num_devs = 2
    devs = [mx.Context('cpu', i) for i in range(num_devs)]
    # mx.kvstore.init_devices(devs)

    shape = (4, 4)
    keys = (5, 9)

    # init all key-value pairs
    mx.kvstore.init(keys, [mx.narray.zeros(shape) for k in keys])


    # first push and then pull on one key
    vals = [mx.narray.ones(shape, d) for d in devs]
    mx.kvstore.push(keys[0], vals)

    # mx.kvstore.pull(keys[0], ctx=)
    # out = mx.narray.empty(shape, devs[1])
    # mx.kvstore.pull(keys[0], out)

    # check_diff_to_scalar(out, num_devs)

    # interleave push and pull for each device
    # vals = []
    # for d in devs:
    #     vals.append([mx.narray.ones(shape, d) for k in keys])
    #     mx.kvstore.push(keys, vals[-1])
    #     mx.kvstore.pull(keys, vals[-1])

    # for v in vals:
    #     for d in v:
    #         check_diff_to_scalar(d, num_devs)

    mx.kvstore.pull(keys[0], output=None, shape=shape, ctx=devs)

    mx.kvstore.stop()

def updater(recv, local):
    local += recv

def test_updater():
    num_devs = 2
    devs = [mx.Context('cpu', i) for i in range(num_devs)]
    mx.kvstore.init_devices(devs)

    shape = (4, 4)
    key = 7
    mx.kvstore.init(key, mx.narray.zeros(shape))
    mx.kvstore.set_updater(updater)

    vals = [mx.narray.ones(shape, d) for d in devs]
    mx.kvstore.push(key, vals)
    mx.kvstore.push(key, vals)
    mx.kvstore.push(key, vals)

    mx.kvstore.pull(key, vals)
    for v in vals:
        check_diff_to_scalar(v, num_devs*3)

    mx.kvstore.stop()

if __name__ == '__main__':
    test_aggregator()
    # test_updater()
