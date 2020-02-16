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

"""Operators that fallback to official NumPy implementation."""


import numpy as onp

__all__ = [
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
    'digitize',
    'divmod',
    'extract',
    'flatnonzero',
    'float_power',
    'frexp',
    'heaviside',
    'histogram2d',
    'histogram_bin_edges',
    'histogramdd',
    'i0',
    'in1d',
    'interp',
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
    'ndim',
    'npv',
    'partition',
    'piecewise',
    'packbits',
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
    'rollaxis',
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
    'triu_indices_from',
    'union1d',
    'unpackbits',
    'unwrap',
    'vander',
]

allclose = onp.allclose
alltrue = onp.alltrue
apply_along_axis = onp.apply_along_axis
apply_over_axes = onp.apply_over_axes
argpartition = onp.argpartition
argwhere = onp.argwhere
array_equal = onp.array_equal
array_equiv = onp.array_equiv
choose = onp.choose
compress = onp.compress
corrcoef = onp.corrcoef
correlate = onp.correlate
count_nonzero = onp.count_nonzero
cov = onp.cov
digitize = onp.digitize
divmod = onp.divmod
extract = onp.extract
flatnonzero = onp.flatnonzero
float_power = onp.float_power
frexp = onp.frexp
heaviside = onp.heaviside
histogram2d = onp.histogram2d
histogram_bin_edges = onp.histogram_bin_edges
histogramdd = onp.histogramdd
i0 = onp.i0
in1d = onp.in1d
interp = onp.interp
intersect1d = onp.intersect1d
isclose = onp.isclose
isin = onp.isin
ix_ = onp.ix_
lexsort = onp.lexsort
min_scalar_type = onp.min_scalar_type
mirr = onp.mirr
modf = onp.modf
msort = onp.msort
nanargmax = onp.nanargmax
nanargmin = onp.nanargmin
nancumprod = onp.nancumprod
nancumsum = onp.nancumsum
nanmax = onp.nanmax
nanmedian = onp.nanmedian
nanmin = onp.nanmin
nanpercentile = onp.nanpercentile
nanprod = onp.nanprod
nanquantile = onp.nanquantile
nanstd = onp.nanstd
nansum = onp.nansum
nanvar = onp.nanvar
ndim = onp.ndim
npv = onp.npv
partition = onp.partition
packbits = onp.packbits
piecewise = onp.piecewise
pmt = onp.pmt
poly = onp.poly
polyadd = onp.polyadd
polydiv = onp.polydiv
polyfit = onp.polyfit
polyint = onp.polyint
polymul = onp.polymul
polysub = onp.polysub
positive = onp.positive
ppmt = onp.ppmt
promote_types = onp.promote_types
ptp = onp.ptp
pv = onp.pv
rate = onp.rate
real = onp.real
result_type = onp.result_type
rollaxis = onp.rollaxis
roots = onp.roots
searchsorted = onp.searchsorted
select = onp.select
setdiff1d = onp.setdiff1d
setxor1d = onp.setxor1d
signbit = onp.signbit
size = onp.size
spacing = onp.spacing
take_along_axis = onp.take_along_axis
trapz = onp.trapz
tril_indices_from = onp.tril_indices_from
trim_zeros = onp.trim_zeros
triu_indices_from = onp.triu_indices_from
union1d = onp.union1d
unpackbits = onp.unpackbits
unwrap = onp.unwrap
vander = onp.vander
