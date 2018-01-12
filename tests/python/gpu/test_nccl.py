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

import mxnet as mx
import numpy as np
import unittest

shapes = [(10), (100), (1000), (10000), (100000), (2,2), (2,3,4,5,6,7,8)]
keys = [1,2,3,4,5,6,7]
num_gpus = len(mx.test_utils.list_gpus())

#TODO The test is capped to run with 8 GPUs only. This is due to a bug in NCCL 2.1
#It is addressed in github issue https://github.com/apache/incubator-mxnet/issues/9004
#Remove this constraint when the github issue is fixed.

if num_gpus > 8 :
    print("The machine has {} gpus. We will run the test on 8 gpus only due to a bug in NCCL 2.1.".format(num_gpus))
    print("Please remove this limitation when issue #9004 is fixed.")
    num_gpus = 8;

gpus = range(1,1+num_gpus)

@unittest.skip("Test requires NCCL library installed and enabled during build")
def test_nccl_pushpull():
    for shape, key in zip(shapes, keys):
        for n_gpus in gpus:
            kv_nccl = mx.kv.create('nccl')
            a = mx.nd.ones(shape, mx.gpu(0))
            cur_key = str(key*max(gpus)+n_gpus)
            kv_nccl.init(cur_key, a)
            arr_list = [mx.nd.ones(shape, mx.gpu(x)) for x in range(n_gpus)]
            res = [mx.nd.zeros(shape, mx.gpu(x)) for x in range(n_gpus)]
            kv_nccl.push(cur_key, arr_list)
            kv_nccl.pull(cur_key, res)
            for x in range(n_gpus):
                assert(np.sum(np.abs((res[x]-n_gpus).asnumpy()))==0)

    print ("Passed")

if __name__ == '__main__':
    test_nccl_pushpull()
