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

::

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

::

   round_


Sums, products, differences
---------------------------
.. autosummary::
   :toctree: generated/

   sum
   prod
   cumsum

::

   nanprod
   nansum
   cumprod
   nancumprod
   nancumsum
   diff
   ediff1d
   gradient
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

::

   exp2
   logaddexp
   logaddexp2

Other special functions
-----------------------
.. autosummary::
   :toctree: generated/


::

   i0
   sinc

Floating point routines
-----------------------
.. autosummary::
   :toctree: generated/

   ldexp

::

   signbit
   copysign
   frexp
   nextafter
   spacing

Rational routines
-----------------
.. autosummary::
   :toctree: generated/

   lcm

::

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

::

   positive
   floor_divide
   float_power

   fmod
   modf
   divmod

Handling complex numbers
------------------------
.. autosummary::
   :toctree: generated/


::

   angle
   real
   imag
   conj
   conjugate


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

::

   convolve

   fabs

   heaviside

   fmax
   fmin

   nan_to_num
   real_if_close

   interp
