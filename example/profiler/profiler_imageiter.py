import os
# uncomment to set the number of worker threads.
# os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
from __future__ import print_function
import time
import mxnet as mx
import numpy as np


def run_imageiter(path_rec, n, batch_size = 32):
    
    data = mx.img.ImageIter(batch_size=batch_size,
                            data_shape=(3, 224, 224),
                            path_imgrec=path_rec,
                            rand_crop=True,
                            rand_resize=True,
                            rand_mirror=True)
    data.reset()
    tic = time.time()
    for i in range(n):
        data.next()
    mx.nd.waitall()
    print(batch_size*n/(time.time() - tic))

if __name__ == '__main__':
    mx.profiler.profiler_set_config(mode='all', filename='profile_imageiter.json')
    mx.profiler.profiler_set_state('run')
    run_imageiter('test.rec', 20)  # See http://mxnet.io/tutorials/python/image_io.html for how to create .rec files.
    mx.profiler.profiler_set_state('stop')