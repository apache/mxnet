# pylint: skip-file
import mxnet as mx
import numpy as np

shapes = [(10), (100), (1000), (10000), (100000), (2,2), (2,3,4,5,6,7,8)]
keys = [1,2,3,4,5,6,7]
gpus = range(1,1+len(mx.test_utils.list_gpus()))

@unittest.skip("Test requires NCCL library installed and enabled during build")
def test_nccl_pushpull():
    for shape, key in zip(shapes, keys):
        for n_gpus in gpus:
            kv_nccl = mx.kv.create('nccl')
            a = mx.nd.ones(shape, mx.gpu(0))
            cur_key = str(key*max(gpus)+n_gpus)
            kv_nccl.init(cur_key, a)
            arr_list = [mx.nd.ones(shape, mx.gpu(x)) for x in xrange(n_gpus)]
            res = [mx.nd.zeros(shape, mx.gpu(x)) for x in xrange(n_gpus)]
            kv_nccl.push(cur_key, arr_list)
            kv_nccl.pull(cur_key, res)
            for x in xrange(n_gpus):
                assert(np.sum(np.abs((res[x]-n_gpus).asnumpy()))==0)

if __name__ == '__main__':
    test_nccl_pushpull()
