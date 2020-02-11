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
    'angle',
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
    'cross',
    'diag_indices_from',
    'digitize',
    'divmod',
    'ediff1d',
    'extract',
    'fabs',
    'fill_diagonal',
    'flatnonzero',
    'float_power',
    'fmax',
    'fmin',
    'fmod',
    'frexp',
    'heaviside',
    'histogram2d',
    'histogram_bin_edges',
    'histogramdd',
    'i0',
    'in1d',
    'interp',
    'isneginf',
    'intersect1d',
    'isclose',
    'isfinite',
    'isin',
    'isposinf',
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
    'nper',
    'npv',
    'partition',
    'piecewise',
    'packbits',
    'place',
    'poly',
    'polyadd',
    'polyder',
    'polydiv',
    'polyfit',
    'polyint',
    'polymul',
    'polysub',
    'positive',
    'ppmt',
    'product',
    'promote_types',
    'ptp',
    'put',
    'put_along_axis',
    'putmask',
    'pv',
    'rate',
    'ravel_multi_index',
    'real',
    'real_if_close',
    'result_type',
    'rollaxis',
    'roots',
    'round',
    'round_',
    'safe_eval',
    'searchsorted',
    'select',
    'setdiff1d',
    'setxor1d',
    'signbit',
    'size',
    'sometrue',
    'spacing',
    'take_along_axis',
    'trapz',
    'tril_indices_from',
    'trim_zeros',
    'triu',
    'triu_indices_from',
    'union1d',
    'unpackbits',
    'unwrap',
    'vander',
    'matmul',
]

allclose = onp.allclose
alltrue = onp.alltrue
angle = onp.angle
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
cross = onp.cross
diag_indices_from = onp.diag_indices_from
digitize = onp.digitize
divmod = onp.divmod
ediff1d = onp.ediff1d
extract = onp.extract
fabs = onp.fabs
fill_diagonal = onp.fill_diagonal
flatnonzero = onp.flatnonzero
float_power = onp.float_power
fmax = onp.fmax
fmin = onp.fmin
fmod = onp.fmod
frexp = onp.frexp
geomspace = onp.geomspace
heaviside = onp.heaviside
histogram2d = onp.histogram2d
histogram_bin_edges = onp.histogram_bin_edges
histogramdd = onp.histogramdd
i0 = onp.i0
in1d = onp.in1d
interp = onp.interp
intersect1d = onp.intersect1d
isclose = onp.isclose
isfinite = onp.isfinite
isin = onp.isin
isneginf = onp.isneginf
isposinf = onp.isposinf
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
nper = onp.nper
npv = onp.npv
partition = onp.partition
packbits = onp.packbits
piecewise = onp.piecewise
place = onp.place
pmt = onp.pmt
poly = onp.poly
polyadd = onp.polyadd
polyder = onp.polyder
polydiv = onp.polydiv
polyfit = onp.polyfit
polyint = onp.polyint
polymul = onp.polymul
polysub = onp.polysub
positive = onp.positive
ppmt = onp.ppmt
promote_types = onp.promote_types
ptp = onp.ptp
put = onp.put
put_along_axis = onp.put_along_axis
putmask = onp.putmask
pv = onp.pv
rate = onp.rate
ravel_multi_index = onp.ravel_multi_index
real = onp.real
real_if_close = onp.real_if_close
result_type = onp.result_type
rollaxis = onp.rollaxis
roots = onp.roots
round = onp.round
round_ = onp.round_
safe_eval = onp.safe_eval
searchsorted = onp.searchsorted
select = onp.select
setdiff1d = onp.setdiff1d
setxor1d = onp.setxor1d
signbit = onp.signbit
size = onp.size
sometrue = onp.sometrue
spacing = onp.spacing
take_along_axis = onp.take_along_axis
trapz = onp.trapz
tril_indices_from = onp.tril_indices_from
trim_zeros = onp.trim_zeros
triu = onp.triu
triu_indices_from = onp.triu_indices_from
union1d = onp.union1d
unpackbits = onp.unpackbits
unwrap = onp.unwrap
vander = onp.vander

# TODO(junwu): delete this after https://github.com/apache/incubator-mxnet/pull/16990 is merged
matmul = onp.matmul
# TODO(junwu): In pr review, delete this after the pr is merged.
product = onp.product
