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
# pylint: disable= arguments-differ, wildcard-import
"Vision transforms."

import warnings
import random

from ....block import Block, HybridBlock
from ....nn import Sequential, HybridSequential
from .....util import is_np_array

from . image import *
from .image import _append_return


class Compose(Sequential):
    """Sequentially composes multiple transforms.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.


    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.

    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.

    Examples
    --------
    >>> transformer = transforms.Compose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x256x256 @cpu(0)>
    """
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.add(hybrid[0])
                hybrid = []
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                hblock.hybridize()
                self.add(hblock)
                hybrid = []

            if i is not None:
                self.add(i)


class HybridCompose(HybridSequential):
    """Sequentially composes multiple transforms. This is the Hybrid version of Compose.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.


    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.

    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.

    Examples
    --------
    >>> transformer = transforms.HybridCompose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x256x256 @cpu(0)>
    """
    def __init__(self, transforms):
        super(HybridCompose, self).__init__()
        for i in transforms:
            if not isinstance(i, HybridBlock):
                raise ValueError("{} is not a HybridBlock, try use `Compose` instead".format(i))
            self.add(i)
        self.hybridize()


class Cast(HybridBlock):
    """Cast inputs to a specific data type

    Parameters
    ----------
    dtype : str, default 'float32'
        The target data type, in string or `numpy.dtype`.


    Inputs:
        - **data**: input tensor with arbitrary shape and dtype.

    Outputs:
        - **out**: output tensor with the same shape as `data` and data type as dtype.
    """
    def __init__(self, dtype='float32'):
        super(Cast, self).__init__()
        self._dtype = dtype

    def hybrid_forward(self, F, *args):
        if is_np_array():
            F = F.npx
        return tuple([F.cast(x, self._dtype) for x in args])


class RandomApply(Sequential):
    """Apply a list of transformations randomly given probability

    Parameters
    ----------
    transforms
        List of transformations.
    p : float
        Probability of applying the transformations.


    Inputs:
        - **data**: input tensor.

    Outputs:
        - **out**: transformed image.
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x, *args):
        if self.p < random.random():
            return x
        x = self.transforms(x)
        return _append_return(x, *args)


class HybridRandomApply(HybridSequential):
    """Apply a list of transformations randomly given probability

    Parameters
    ----------
    transforms
        List of transformations which must be HybridBlocks.
    p : float
        Probability of applying the transformations.


    Inputs:
        - **data**: input tensor.

    Outputs:
        - **out**: transformed image.
    """

    def __init__(self, transforms, p=0.5):
        super(HybridRandomApply, self).__init__()
        assert isinstance(transforms, HybridBlock)
        self.transforms = transforms
        self.p = p

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            cond = self.p < F.random.uniform(low=0, high=1, size=1)
            return F.npx.cond(cond, x, self.transforms(x))
        cond = self.p < F.random.uniform(low=0, high=1, shape=1)
        return _append_return(F.contrib.cond(cond, x, self.transforms(x)), *args)
