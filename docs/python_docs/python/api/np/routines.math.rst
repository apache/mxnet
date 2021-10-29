.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

Mathematical functions
**********************

.. currentmodule:: mxnet.np

.. note::

   Currently, most of the math functions only support inputs and outputs of the same dtype.
   This limitation usually results in imprecise outputs for ndarrays with integral dtype
   while floating-point values are expected in the output.
   Appropriate handling of ndarrays integral dtypes is in active development.


Trigonometric functions
-----------------------
.. autosummary::
   :toctree: generated/

   sin
   cos
   tan
   arcsin
   arccos
   arctan
   degrees
   radians
   hypot
   arctan2
   deg2rad
   rad2deg
   unwrap


Hyperbolic functions
--------------------
.. autosummary::
   :toctree: generated/

   sinh
   cosh
   tanh
   arcsinh
   arccosh
   arctanh


Rounding
--------
.. autosummary::
   :toctree: generated/

   rint
   fix
   floor
   ceil
   trunc
   around
   round_


Sums, products, differences
---------------------------
.. autosummary::
   :toctree: generated/

   sum
   prod
   cumsum
   nanprod
   nansum
   cumprod
   nancumprod
   nancumsum
   diff
   ediff1d
   cross
   trapz


Exponents and logarithms
------------------------
.. autosummary::
   :toctree: generated/

   exp
   expm1
   log
   log10
   log2
   log1p
   logaddexp


Other special functions
-----------------------
.. autosummary::
   :toctree: generated/

   i0


Floating point routines
-----------------------
.. autosummary::
   :toctree: generated/

   ldexp
   signbit
   copysign
   frexp
   spacing


Rational routines
-----------------
.. autosummary::
   :toctree: generated/

   lcm
   gcd


Arithmetic operations
---------------------
.. autosummary::
   :toctree: generated/

   add
   reciprocal
   negative
   divide
   power
   subtract
   mod
   multiply
   true_divide
   remainder
   positive
   float_power
   fmod
   modf
   divmod
   floor_divide


Miscellaneous
-------------
.. autosummary::
   :toctree: generated/

   clip
   sqrt
   cbrt
   square
   absolute
   sign
   maximum
   minimum
   fabs
   heaviside
   fmax
   fmin
   nan_to_num
   interp

