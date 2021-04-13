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

# pylint: disable=undefined-all-variable, not-callable, cell-var-from-loop
"""Operators that fallback to official NumPy implementation."""

import sys
from functools import wraps
import numpy as onp

fallbacks = [
    '__version__',
    '_NoValue',
    'allclose',
    'alltrue',
    'apply_along_axis',
    'apply_over_axes',
    'argpartition',
    'argwhere',
    'array_equal',
    'array_equiv',
    'choose',
    'compress',
    'corrcoef',
    'correlate',
    'count_nonzero',
    'cov',
    'cumprod',
    'digitize',
    'divmod',
    'dtype',
    'extract',
    'float_power',
    'frexp',
    'heaviside',
    'histogram2d',
    'histogram_bin_edges',
    'histogramdd',
    'i0',
    'in1d',
    'intersect1d',
    'isclose',
    'isin',
    'ix_',
    'lexsort',
    'min_scalar_type',
    'mirr',
    'modf',
    'msort',
    'nanargmax',
    'nanargmin',
    'nancumprod',
    'nancumsum',
    'nanmax',
    'nanmedian',
    'nanmin',
    'nanpercentile',
    'nanprod',
    'nanquantile',
    'nanstd',
    'nansum',
    'nanvar',
    'ndim',
    'npv',
    'packbits',
    'partition',
    'piecewise',
    'pmt',
    'poly',
    'polyadd',
    'polydiv',
    'polyfit',
    'polyint',
    'polymul',
    'polysub',
    'positive',
    'ppmt',
    'promote_types',
    'ptp',
    'pv',
    'rate',
    'real',
    'result_type',
    'roots',
    'searchsorted',
    'select',
    'setdiff1d',
    'setxor1d',
    'signbit',
    'size',
    'spacing',
    'take_along_axis',
    'trapz',
    'tril_indices_from',
    'trim_zeros',
    'union1d',
    'unpackbits',
    'unwrap',
    'vander',
]

fallback_mod = sys.modules[__name__]

def get_func(obj, doc):
    """Get new numpy function with object and doc"""
    @wraps(obj)
    def wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    wrapper.__doc__ = doc
    return wrapper

for obj_name in fallbacks:
    onp_obj = getattr(onp, obj_name)
    if callable(onp_obj):
        new_fn_doc = onp_obj.__doc__
        if obj_name in {'divmod', 'float_power', 'frexp', 'heaviside', 'modf', 'signbit', 'spacing'}:
            # remove reference of kwargs doc and the reference to ufuncs
            new_fn_doc = new_fn_doc.replace("**kwargs\n    For other keyword-only arguments, see the"
                                            + "\n    :ref:`ufunc docs <ufuncs.kwargs>`.", '')
        elif obj_name == 'trapz':
            # remove unused reference
            new_fn_doc = new_fn_doc.replace(
                '.. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule', '')
        setattr(fallback_mod, obj_name, get_func(onp_obj, new_fn_doc))
    else:
        setattr(fallback_mod, obj_name, onp_obj)

__all__ = fallbacks
