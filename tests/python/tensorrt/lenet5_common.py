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

import numpy as np
import mxnet as mx
from common import *

def get_iters(mnist, batch_size):
    """Get MNIST iterators."""
    train_iter = mx.io.NDArrayIter(mnist['train_data'],
                                   mnist['train_label'],
                                   batch_size,
                                   shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    all_test_labels = np.array(mnist['test_label'])
    return train_iter, val_iter, test_iter, all_test_labels
