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

import nose
import mxnet as mx
import os
import unittest
from mxnet.test_utils import EnvManager

def test_bulk():
    with mx.engine.bulk(10):
        x = mx.nd.ones((10,))
        x *= 2
        x += 1
        x.wait_to_read()
        x += 1
        assert (x.asnumpy() == 4).all()
        for i in range(100):
            x += 1
    assert (x.asnumpy() == 104).all()

@unittest.skip("OMP platform dependent")
def test_engine_openmp_after_fork():
    """
    Test that the number of max threads in the child is 1. After forking we should not use a bigger
    OMP thread pool.

    With GOMP the child always has the same number when calling omp_get_max_threads, with LLVM OMP
    the child respects the number of max threads set in the parent.
    """
    with EnvManager('OMP_NUM_THREADS', '42'):
        r, w = os.pipe()
        pid = os.fork()
        if pid:
            os.close(r)
            wfd = os.fdopen(w, 'w')
            wfd.write('a')
            omp_max_threads = mx.base._LIB.omp_get_max_threads()
            print("Parent omp max threads: {}".format(omp_max_threads))
            try:
                wfd.close()
            except:
                pass
            try:
                (cpid, status) = os.waitpid(pid, 0)
                assert cpid == pid
                exit_status = status >> 8
                assert exit_status == 0
            except:
                pass
        else:
            os.close(w)
            rfd = os.fdopen(r, 'r')
            rfd.read(1)
            omp_max_threads = mx.base._LIB.omp_get_max_threads()
            print("Child omp max threads: {}".format(omp_max_threads))
            assert omp_max_threads == 1



if __name__ == '__main__':
    import nose
    nose.runmodule()
