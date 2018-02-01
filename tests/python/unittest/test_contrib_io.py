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

import mxnet.ndarray as nd
from mxnet.gluon.data.vision.datasets import *
from mxnet.gluon.data.dataloader import *
from mxnet.contrib.io import *
from mxnet.test_utils import *

def test_contrib_DataLoaderIter():
    dataset = MNIST()
    batch_size = 50
    dataloader = DataLoader(dataset, batch_size)
    test_iter = DataLoaderIter(dataloader)
    batch = next(test_iter)
    assert batch.data[0].shape == (batch_size, 28, 28, 1)
    assert batch.label[0].shape == (batch_size,)
    count = 0
    test_iter.reset()
    for batch in test_iter:
        count += 1
    expected = 60000 / batch_size
    assert count == expected, "expected {} batches, given {}".format(expected, count)

if __name__ == "__main__":
    test_contrib_DataLoaderIter()
