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
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet import gluon
from mxnet.gluon.data.vision.transforms import AdjustLighting
from mxnet.test_utils import assert_almost_equal

def test_adjust_lighting():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    alpha_rgb = [0.05, 0.06, 0.07]
    eigval = np.array([55.46, 4.794, 1.148])
    eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5808, -0.0045, -0.8140]])
    f = AdjustLighting(alpha_rgb=alpha_rgb, eigval=eigval.ravel().tolist(), eigvec=eigvec.ravel().tolist())
    out_nd = f(nd.array(data_in, dtype=np.uint8))
    out_gt = np.clip(data_in.astype(np.float32)
                     + np.dot(eigvec * alpha_rgb, eigval.reshape((3, 1))).reshape((1, 1, 3)), 0, 255).astype(np.uint8)
    assert_almost_equal(out_nd.asnumpy(), out_gt)

if __name__ == '__main__':
    import nose
    nose.runmodule()
