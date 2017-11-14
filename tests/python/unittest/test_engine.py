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


if __name__ == '__main__':
    import nose
    nose.runmodule()
