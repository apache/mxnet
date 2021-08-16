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

#!/usr/bin/python
import mxnet as mx
import time
import numpy as np


def isSupported(x, y):
    for i in range(2, len(x)):
        s1 = x[i]
        s2 = y[i]
        if s2 == 0:
            return False
        if s1 % s2 != 0:
            return False

    IH = x[2]
    IW = x[3]
    OH = y[2]
    OW = y[3]

    strides_H = np.floor((IH << 1) / OH) - np.floor(IH / OH)
    strides_W = np.floor((IW << 1) / OW) - np.floor(IW / OW)
    kernel_H = np.ceil((IH << 1) / OH) - np.floor(IH / OH)
    kernel_W = np.ceil((IW << 1) / OW) - np.floor(IW / OW)
    pad_l_top = (strides_H * (OH - 1) + kernel_H - IH) / 2
    pad_l_left = (strides_W * (OW - 1) + kernel_W - IW) / 2

    return pad_l_top == 0 and pad_l_left == 0


def time_procedure(shape, output_height, count):
    data = mx.nd.random_uniform(shape=shape, low=-1.0, high=1.0)
    mx.nd.waitall()
    begin = time.time()
    for i in range(0, count):
        out = mx.nd.contrib.AdaptiveAvgPooling2D(data, output_size=output_height)
        mx.nd.waitall()
    return (time.time() - begin) / count


count = 200
for x in [1, 2, 4, 8, 16, 32]:
    for y in [1, 2, 4, 8, 16, 32, 128, 256, 512, 1024, 2048]:
        shape = (x, x, y, y)
        for i in [1, 2, 4, 8, 16, 32]:
            timing = time_procedure(shape, i, count)
            print("{}x{:5d}:{:5d} | {:.7f}".format(shape, i, isSupported([x, x, y, y], [x, x, i, i]), timing))
