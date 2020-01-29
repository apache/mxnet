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
    'interp',
    'isneginf',
    'intersect1d',
    'isclose',
    'isfinite',
    'spacing',
    'nanargmax',
    'nanargmin',
    'nancumprod',
    'nancumsum',
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
    'promote_types',
    'put_along_axis',
    'putmask',
    'pv',
    'rate',
    'ravel_multi_index',
    'real',
    'real_if_close',
    'roots',
    'round_',
    'safe_eval',
    'signbit',
    'take_along_axis',
    'tril_indices_from',
    'trim_zeros',
    'triu',
    'triu_indices_from',
    'unwrap',
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
interp = onp.interp
intersect1d = onp.intersect1d
isclose = onp.isclose
isfinite = onp.isfinite
isneginf = onp.isneginf
nanargmax = onp.nanargmax
nanargmin = onp.nanargmin
nancumprod = onp.nancumprod
nancumsum = onp.nancumsum
nanprod = onp.nanprod
nanquantile = onp.nanquantile
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
put_along_axis = onp.put_along_axis
putmask = onp.putmask
pv = onp.pv
rate = onp.rate
ravel_multi_index = onp.ravel_multi_index
real = onp.real
real_if_close = onp.real_if_close
roots = onp.roots
round_ = onp.round_
safe_eval = onp.safe_eval
signbit = onp.signbit
spacing = onp.spacing
take_along_axis = onp.take_along_axis
tril_indices_from = onp.tril_indices_from
trim_zeros = onp.trim_zeros
triu = onp.triu
triu_indices_from = onp.triu_indices_from
unwrap = onp.unwrap
