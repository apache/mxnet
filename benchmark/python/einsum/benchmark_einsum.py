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

import time
import mxnet as mx
from mxnet import np, npx

def measure_cost(repeat, func_name, *args, **kwargs):
    """Measure time cost of running a function
    """
    mx.nd.waitall()
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs)
    mx.nd.waitall()
    end = time.time()
    diff = end - start
    return diff / repeat


def test_np_einsum():
    print("Path optimization test:")
    # Basic einsum
    a = np.ones(64).reshape(2,4,8)
    args = ['ijk,ilm,njm,nlk,abc->', a, a, a, a, a]
    cost = measure_cost(500, np.einsum, *args)
    print(f"Basic einsum: {cost * 1000} ms")

    # Sub-optimal einsum
    # cost = measure_cost(500, np.einsum, *args, optimize='optimal')
    # print("Optimal einsum: {} ms".format(cost * 1000))

    # Greedy einsum
    cost = measure_cost(500, np.einsum, *args, optimize=True)
    print(f"Greedy einsum: {cost * 1000} ms")

    print("RNN Use Case:")
    a = np.random.uniform(0, 1, size=(64, 128, 512))
    b = np.random.uniform(0, 1, size=(128, 512, 2, 2))
    args = ['bij, ijkl->bkl', a, b]
    cost = measure_cost(2, np.einsum, *args, optimize=True)
    print(f'Greedy einsum: {cost * 1000} ms')
    cost = measure_cost(2, np.einsum, *args)
    print(f'Basic einsum: {cost * 1000} ms')

    print('Inner Product:')
    a = np.ones(6000000)
    b = np.ones(6000000)
    args = [a, b]
    cost = measure_cost(50, np.tensordot, *args, axes=([0],[0]))
    print(f'Tensordot: {cost * 1000} ms')
    args = ['i, i', a, b]
    cost = measure_cost(50, np.einsum, *args, optimize=True)
    print(f'Greedy einsum: {cost * 1000} ms')
    cost = measure_cost(50, np.einsum, *args)
    print(f'Basic einsum: {cost * 1000} ms')

    print('Matrix Product:')
    a = np.ones(600000).reshape(200, 3000)
    b = np.ones(600000).reshape(3000, 200)
    args = [a, b]
    cost = measure_cost(50, np.tensordot, *args, axes=([1],[0]))
    print(f'Tensordot: {cost * 1000} ms')
    args = ['ij, jk', a, b]
    cost = measure_cost(50, np.einsum, *args, optimize=True)
    print(f'Greedy einsum: {cost * 1000} ms')
    cost = measure_cost(50, np.einsum, *args)
    print(f'Basic einsum: {cost * 1000} ms')


if __name__ == "__main__":
    npx.set_np(dtype=False)
    test_np_einsum()
