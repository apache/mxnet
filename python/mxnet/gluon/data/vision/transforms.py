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

# coding: utf-8

from .. import dataset
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import ndarray, initializer


class Compose(Sequential):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.register_child(hybrid[0])
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                self.register_child(hblock)
            if i is not None:
                self.register_child(i)


class Cast(HybridBlock):
    def __init__(self, dtype='float32'):
        super(Cast, self).__init__()
        self._dtype = dtype

    def hybrid_forward(self, F, x):
        return F.cast(x, self._dtype)


class Normalize(HybridBlock):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.params.get('mean', shape=(len(mean),), dtype=None,
                        grad_req='null', differentiable=False,
                        init=initializer.Constant(ndarray.array(mean)))
        self.params.get('std', shape=(len(std),), dtype=None,
                        grad_req='null', differentiable=False,
                        init=initializer.Constant(ndarray.array(std)))
        self.initialize()

    def hybrid_forward(self, F, x, mean, std):
        x = F.broadcast_sub(x, mean)
        return F.broadcast_div(x, std)


class RandomSizedCrop(Block):
    def __init__(self, )
