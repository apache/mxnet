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


class TestNCCL(unittest.TestCase):
    num_gpus = min(8, mx.context.num_gpus())
    shapes: [1, 10, 100, 1000, 10000, 100000, (2, 2), (2, 3, 4, 5, 6, 7, 8)]
    tensors: {}

    @classmethod
    def setUpClass(cls):
        num_gpus = mx.context.num_gpus()
        if num_gpus == 0:
            raise unittest.SkipTest("No GPUs available")
        if num_gpus < 2:
            raise unittest.SkipTest("It makes sense to test NCCL functionality on more than 1 GPU only")
        if num_gpus > 8:
            print("The machine has {} GPUs. We will run the test on not more than 8 GPUs.".format(cls.num_gpus))
            print("There is a limit of 8 maximum P2P peers for all PCI-E hardware created.")

    def setUp(self):
        self.kv_nccl = mx.kv.create('nccl')

        for gpu_index in range(self.num_gpus):
            shapes = np.random.shuffle(self.shapes)
            self.tensors[gpu_index] = [np.random.random_sample(shape) for shape in shapes]

    def push_shapes(self):
        for gpu_index in range(self.num_gpus):
            tensors = [mx.nd.array(array, mx.gpu(gpu_index)) for array in self.tensors[gpu_index]]
            self.kv_nccl.push(gpu_index, tensors)

    def test_push_pull(self):
        self.push_shapes()

        for gpu_index in range(self.num_gpus):
            for gpu_index2 in range(self.num_gpus):
                if gpu_index == gpu_index2:
                    continue
                pulled_tensors = [mx.nd.zeros(array.shape, mx.gpu(gpu_index)) for array in self.tensors[gpu_index2]]
                self.kv_nccl.pull(gpu_index2, pulled_tensors)
                assert np.allclose(pulled_tensors, self.tensors[gpu_index2])


if __name__ == '__main__':
    unittest.main()
