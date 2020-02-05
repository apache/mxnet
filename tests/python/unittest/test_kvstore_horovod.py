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

# pylint: skip-file
import numpy as np
import mxnet as mx

shape = (2, 3)


def test_horovod_basic():
    kv = mx.kv.create('horovod')
    assert kv.type == 'horovod'
    print('TEST num_worker: {}'.format(kv.num_workers))
    print('TEST rank: {}'.format(kv.rank))
    print('TEST local_rank: {}'.format(kv.local_rank))
    # assert kv.num_workers == 1
    # assert kv.rank == 0
    # assert kv.local_rank == 0


def test_horovod_broadcast():
    # broadcast a single key-value pair
    kv = mx.kv.create('horovod')
    a = mx.nd.ones(shape) * kv.rank
    expected = np.zeros(shape)
    kv.broadcast('1', value=a)
    if kv.rank != 0:
        print('TEST broadcast value: \n{}'.format(a.asnumpy()))
    # assert a.asnumpy().all() == expected.all()


def test_horovod_broadcast_inplace():
    kv = mx.kv.create('horovod')
    a = mx.nd.ones(shape) * kv.rank
    b = mx.nd.zeros(shape)
    kv.broadcast('1', value=a, out=b)
    if kv.rank != 0:
        print('TEST broadcast inplace value: \n{}'.format(b.asnumpy()))
    # assert a.asnumpy().all() == b.asnumpy().all()


def test_horovod_allreduce():
    kv = mx.kv.create('horovod')
    nworker = kv.num_workers
    a = mx.nd.ones(shape)
    kv.pushpull('1', a)
    print('TEST allreduce: \n{}'.format(a.asnumpy()))


test_horovod_basic()
test_horovod_broadcast()
test_horovod_allreduce()
# if __name__ == '__main__':
#    import nose
#    nose.runmodule()
