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
import os
# uncomment to set the number of worker threads.
# os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import time
import mxnet as mx


def run_imageiter(path_rec, n, batch_size=32):

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
    mx.profiler.set_config(profile_all=True, filename='profile_imageiter.json')
    mx.profiler.set_state('run')
    run_imageiter('test.rec', 20)  # See http://mxnet.io/tutorials/python/image_io.html for how to create .rec files.
    mx.profiler.set_state('stop')
