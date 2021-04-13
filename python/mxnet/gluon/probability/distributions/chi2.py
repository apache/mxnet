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
# pylint: disable=wildcard-import
"""Chi-sqaure distribution"""
__all__ = ['Chi2']

from .gamma import Gamma
from .constraint import Positive


class Chi2(Gamma):
    r"""Create a Chi2 distribution object.
    Chi2(df) is equivalent to Gamma(shape=df / 2, scale=2)

    Parameters
    ----------
    df : Tensor or scalar, default 0
        Shape parameter of the distribution.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    arg_constraints = {'df': Positive()}

    def __init__(self, df, F=None, validate_args=None):
        super(Chi2, self).__init__(df / 2, 2, F, validate_args)

    @property
    def df(self):
        return self.shape * 2
