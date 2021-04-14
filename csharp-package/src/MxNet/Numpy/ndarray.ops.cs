using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public partial class ndarray
    {
        public ndarray broadcast_to(Shape shape)
        {
            return nd_np_ops.broadcast_to(this, shape);
        }

        public ndarray zero_like(DType dtype = null, string order = "C")
        {
            return nd_np_ops.zeros_like(this, dtype, order);
        }

        public bool all()
        {
            return nd_np_ops.all(this);
        }

        public ndarray all(int axis, bool keepdims = false)
        {
            return nd_np_ops.all(this, axis, keepdims: keepdims);
        }

        public bool any()
        {
            return nd_np_ops.any(this);
        }

        public ndarray any(int axis, bool keepdims = false)
        {
            return nd_np_ops.any(this, axis, keepdims: keepdims);
        }

        public ndarray take(ndarray indices, int? axis = null, string mode = "raise")
        {
            return nd_np_ops.take(this, indices, axis, mode);
        }

        public ndarray unique(int? axis = null)
        {
            return nd_np_ops.unique(this, axis);
        }

        public (ndarray, ndarray, ndarray, ndarray) unique(bool return_index = false, bool return_inverse = false, bool return_counts = false, int? axis = null)
        {
            return nd_np_ops.unique(this, return_index, return_inverse, return_counts, axis);
        }

        public ndarray mod(ndarray x2)
        {
            return nd_np_ops.mod(this, x2);
        }

        public ndarray mod(float x2)
        {
            return nd_np_ops.mod(this, x2);
        }

        public ndarray fmod(ndarray x2)
        {
            return nd_np_ops.fmod(this, x2);
        }

        public ndarray matmul(ndarray x)
        {
            return nd_np_ops.matmul(this, x);
        }

        public ndarray remainder(ndarray x)
        {
            return nd_np_ops.remainder(this, x);
        }

        public ndarray power(ndarray x)
        {
            return nd_np_ops.power(this, x);
        }

        public ndarray power(float x)
        {
            return nd_np_ops.power(this, x);
        }

        public ndarray gcd(ndarray x)
        {
            return nd_np_ops.gcd(this, x);
        }

        public ndarray lcm(ndarray x)
        {
            return nd_np_ops.lcm(this, x);
        }

        public ndarray sin()
        {
            return nd_np_ops.sin(this);
        }

        public ndarray cos()
        {
            return nd_np_ops.cos(this);
        }

        public ndarray sinh()
        {
            return nd_np_ops.sinh(this);
        }

        public ndarray cosh()
        {
            return nd_np_ops.cosh(this);
        }

        public ndarray tanh()
        {
            return nd_np_ops.tanh(this);
        }

        public ndarray log10()
        {
            return nd_np_ops.log10(this);
        }

        public ndarray sqrt()
        {
            return nd_np_ops.sqrt(this);
        }

        public ndarray cbrt()
        {
            return nd_np_ops.cbrt(this);
        }

        public ndarray abs()
        {
            return nd_np_ops.abs(this);
        }

        public ndarray fabs()
        {
            return nd_np_ops.fabs(this);
        }

        public ndarray absolute()
        {
            return nd_np_ops.absolute(this);
        }

        public ndarray exp()
        {
            return nd_np_ops.exp(this);
        }

        public ndarray expm1()
        {
            return nd_np_ops.expm1(this);
        }

        public ndarray arcsin()
        {
            return nd_np_ops.arcsin(this);
        }

        public ndarray arccos()
        {
            return nd_np_ops.arccos(this);
        }

        public ndarray arctan()
        {
            return nd_np_ops.arctan(this);
        }

        public ndarray sign()
        {
            return nd_np_ops.sign(this);
        }

        public ndarray log()
        {
            return nd_np_ops.log(this);
        }

        public ndarray rint()
        {
            return nd_np_ops.rint(this);
        }

        public ndarray log2()
        {
            return nd_np_ops.log2(this);
        }

        public ndarray log1p()
        {
            return nd_np_ops.log1p(this);
        }

        public ndarray degrees()
        {
            return nd_np_ops.degrees(this);
        }

        public ndarray rad2deg()
        {
            return nd_np_ops.rad2deg(this);
        }

        public ndarray radians()
        {
            return nd_np_ops.radians(this);
        }

        public ndarray deg2rad()
        {
            return nd_np_ops.deg2rad(this);
        }

        public ndarray reciprocal()
        {
            return nd_np_ops.reciprocal(this);
        }

        public ndarray square()
        {
            return nd_np_ops.square(this);
        }

        public ndarray negative()
        {
            return nd_np_ops.negative(this);
        }

        public ndarray fix()
        {
            return nd_np_ops.fix(this);
        }

        public ndarray tan()
        {
            return nd_np_ops.tan(this);
        }

        public ndarray ceil()
        {
            return nd_np_ops.ceil(this);
        }

        public ndarray floor()
        {
            return nd_np_ops.floor(this);
        }

        public ndarray invert()
        {
            return nd_np_ops.invert(this);
        }

        public ndarray bitwise_not()
        {
            return nd_np_ops.bitwise_not(this);
        }

        public ndarray trunc()
        {
            return nd_np_ops.trunc(this);
        }

        public ndarray logical_not()
        {
            return nd_np_ops.logical_not(this);
        }

        public ndarray arcsinh()
        {
            return nd_np_ops.arcsinh(this);
        }

        public ndarray arccosh()
        {
            return nd_np_ops.arccosh(this);
        }

        public ndarray arctanh()
        {
            return nd_np_ops.arctanh(this);
        }

        public ndarray argsort(int axis = -1, string kind = null, string order = null)
        {
            return nd_np_ops.argsort(this, axis, kind, order);
        }

        public ndarray sort(int axis = -1, string kind = null, string order = null)
        {
            return nd_np_ops.sort(this, axis, kind, order);
        }

        public ndarray tensordot(ndarray b, int axes = 2)
        {
            return nd_np_ops.tensordot(this, b, axes);
        }

        public ndarray histogram(int bins = 10, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            return nd_np_ops.histogram(this, bins, range, normed, weights, density);
        }

        public ndarray histogram(ndarray bins, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            return nd_np_ops.histogram(this, bins, range, normed, weights, density);
        }

        public ndarray expand_dims(int axis)
        {
            return nd_np_ops.expand_dims(this, axis);
        }

        public ndarray tile(params int[] reps)
        {
            return nd_np_ops.tile(this, reps);
        }

        public ndarray trace(int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return nd_np_ops.trace(this, offset, axis1, axis2);
        }

        public ndarray transpose(params int[] axes)
        {
            return nd_np_ops.transpose(this, axes);
        }
        
        public ndarray transpose(Shape axes)
        {
            return nd_np_ops.transpose(this, axes.Data.ToArray());
        }

        public ndarray repeat(int repeats, int? axis = null)
        {
            return nd_np_ops.repeat(this, repeats, axis);
        }

        public ndarray tril(int k = 0)
        {
            return nd_np_ops.tril(k);
        }

        public ndarray[] split(int[] indices_or_sections, int axis = 0)
        {
            return nd_np_ops.split(this, indices_or_sections, axis);
        }

        public ndarray[] array_split(int[] indices_or_sections, int axis = 0)
        {
            return nd_np_ops.array_split(this, indices_or_sections, axis);
        }

        public ndarray vsplit(int[] indices_or_sections)
        {
            return nd_np_ops.vsplit(this, indices_or_sections);
        }

        public ndarray dsplit(int[] indices_or_sections)
        {
            return nd_np_ops.dsplit(this, indices_or_sections);
        }

        public ndarray append(ndarray values, int? axis = null)
        {
            return nd_np_ops.append(this, values, axis);
        }

        public ndarray maximum(ndarray x)
        {
            return nd_np_ops.maximum(this, x);
        }

        public ndarray fmax(ndarray x)
        {
            return nd_np_ops.fmax(this, x);
        }

        public ndarray minimum(ndarray x)
        {
            return nd_np_ops.minimum(this, x);
        }

        public ndarray fmin(ndarray x)
        {
            return nd_np_ops.fmin(this, x);
        }

        public ndarray max(int? axis = null, bool keepdims = false)
        {
            return nd_np_ops.max(this, axis, keepdims: keepdims);
        }

        public ndarray min(int? axis = null, bool keepdims = false)
        {
            return nd_np_ops.min(this, axis, keepdims: keepdims);
        }

        public ndarray swapaxes(int axis1, int axis2)
        {
            return nd_np_ops.swapaxes(this, axis1, axis2);
        }

        public ndarray clip(float a_min, float a_max)
        {
            return nd_np_ops.clip(this, a_min, a_max);
        }

        public ndarray argmax(int? axis = null)
        {
            return nd_np_ops.argmax(this, axis);
        }

        public ndarray argmin(int? axis = null)
        {
            return nd_np_ops.argmin(this, axis);
        }

        public ndarray amax(int? axis = null)
        {
            return nd_np_ops.amax(this, axis);
        }

        public ndarray amin(int? axis = null)
        {
            return nd_np_ops.amin(this, axis);
        }

        public ndarray average(int? axis = null, ndarray weights = null, bool returned = false)
        {
            return nd_np_ops.average(this, axis, weights, returned);
        }

        public ndarray mean(int? axis = null, DType dtype = null, bool keepdims = false)
        {
            return nd_np_ops.mean(this, axis, dtype, keepdims: keepdims);
        }

        public ndarray std(int? axis = null, DType dtype = null, bool keepdims = false)
        {
            return nd_np_ops.std(this, axis, dtype, keepdims: keepdims);
        }
        public ndarray var(int? axis = null, DType dtype = null, bool keepdims = false)
        {
            return nd_np_ops.var(this, axis, dtype, keepdims: keepdims);
        }

        public ndarray copysign(ndarray x)
        {
            return nd_np_ops.copysign(this, x);
        }

        public ndarray ravel(string order = "x")
        {
            return nd_np_ops.ravel(this, order);
        }

        public ndarray flatnonzero()
        {
            return nd_np_ops.flatnonzero(this);
        }

        public ndarray diag_indices_from()
        {
            return nd_np_ops.diag_indices_from(this);
        }

        public ndarray flip(int? axis = null)
        {
            return nd_np_ops.flip(this, axis);
        }

        public ndarray flipud()
        {
            return nd_np_ops.flipud(this);
        }

        public ndarray fliplr()
        {
            return nd_np_ops.fliplr(this);
        }

        public ndarray around(int decimals = 0)
        {
            return nd_np_ops.around(this, decimals);
        }

        public ndarray round(int decimals = 0)
        {
            return nd_np_ops.round(this, decimals);
        }

        public ndarray round_(int decimals = 0)
        {
            return nd_np_ops.round_(this, decimals);
        }

        public ndarray arctan2(ndarray x)
        {
            return nd_np_ops.arctan2(this, x);
        }

        public ndarray hypot(ndarray x)
        {
            return nd_np_ops.hypot(this, x);
        }

        public ndarray bitwise_and(ndarray x)
        {
            return nd_np_ops.bitwise_and(this, x);
        }

        public ndarray bitwise_xor(ndarray x)
        {
            return nd_np_ops.bitwise_xor(this, x);
        }

        public ndarray bitwise_or(ndarray x)
        {
            return nd_np_ops.bitwise_or(this, x);
        }

        public ndarray ldexp(ndarray x)
        {
            return nd_np_ops.ldexp(this, x);
        }

        public ndarray vdot(ndarray b)
        {
            return nd_np_ops.vdot(this, b);
        }

        public ndarray inner(ndarray b)
        {
            return nd_np_ops.inner(this, b);
        }

        public ndarray outer(ndarray b)
        {
            return nd_np_ops.outer(this, b);
        }

        public ndarray cross(ndarray b, int axisa = -1, int axisb = -1, int axisc = -1, int? axis = null)
        {
            return nd_np_ops.cross(this, b, axisa, axisb, axisc, axis);
        }

        public ndarray kron(ndarray b)
        {
            return nd_np_ops.kron(this, b);
        }

        public ndarray equal(ndarray x)
        {
            return nd_np_ops.equal(this, x);
        }

        public ndarray not_equal(ndarray x)
        {
            return nd_np_ops.not_equal(this, x);
        }

        public ndarray greater(ndarray x)
        {
            return nd_np_ops.greater(this, x);
        }

        public ndarray less(ndarray x)
        {
            return nd_np_ops.less(this, x);
        }

        public ndarray logical_and(ndarray x)
        {
            return nd_np_ops.logical_and(this, x);
        }

        public ndarray logical_or(ndarray x)
        {
            return nd_np_ops.logical_or(this, x);
        }

        public ndarray logical_xor(ndarray x)
        {
            return nd_np_ops.logical_xor(this, x);
        }

        public ndarray greater_equal(ndarray x)
        {
            return nd_np_ops.greater_equal(this, x);
        }

        public ndarray less_equal(ndarray x)
        {
            return nd_np_ops.less_equal(this, x);
        }

        public ndarray roll(int shift, int? axis = null)
        {
            return nd_np_ops.roll(this, shift, axis);
        }

        public ndarray roll(int[] shift, int? axis = null)
        {
            return nd_np_ops.roll(this, shift, axis);
        }

        public ndarray rot90(int k = 1, params int[] axes)
        {
            return nd_np_ops.rot90(this, k, axes);
        }

        public ndarray hsplit(params int[] indices_or_sections)
        {
            return nd_np_ops.hsplit(this, indices_or_sections);
        }

        public ndarray nonzero()
        {
            return nd_np_ops.nonzero(this);
        }

        public ndarray median(int? axis = null, bool? overwrite_input = null, bool keepdims = false)
        {
            return nd_np_ops.median(this, axis, overwrite_input: overwrite_input, keepdims: keepdims);
        }

        public bool shares_memory(ndarray b, int? max_work = null)
        {
            return nd_np_ops.shares_memory(this, b, max_work);
        }

        public bool may_share_memory(ndarray b, int? max_work = null)
        {
            return nd_np_ops.may_share_memory(this, b, max_work);
        }

        public ndarray diff(int n = 1, int axis = -1, ndarray prepend = null, ndarray append = null)
        {
            return nd_np_ops.diff(this, n, axis, prepend, append);
        }

        public ndarray ediff1d(ndarray to_end = null, ndarray to_begin = null)
        {
            return nd_np_ops.ediff1d(this, to_end, to_begin);
        }

        public ndarray resize(Shape new_shape)
        {
            return nd_np_ops.resize(this, new_shape);
        }

        public ndarray interp(float[] xp, float[] fp, float? left = null, float? right = null, float? period = null)
        {
            return nd_np_ops.interp(this, xp, fp, left, right, period);
        }

        public ndarray full_like(float fill_value, DType dtype = null, string order = "C", Context ctx = null)
        {
            return nd_np_ops.full_like(this, fill_value, dtype, order, ctx);
        }

        public ndarray zeros_like(DType dtype = null, string order = "C", Context ctx = null)
        {
            return nd_np_ops.zeros_like(this, dtype, order, ctx);
        }

        public ndarray ones_like(DType dtype = null, string order = "C", Context ctx = null)
        {
            return nd_np_ops.ones_like(this, dtype, order, ctx);
        }

        public ndarray fill_diagonal(float val, bool wrap = false)
        {
            return nd_np_ops.fill_diagonal(this, val, wrap);
        }

        public ndarray nan_to_num(bool copy = true, float nan = 0, float? posinf = null, float? neginf = null)
        {
            return nd_np_ops.nan_to_num(this, copy, nan, posinf, neginf);
        }

        public ndarray squeeze(int? axis = null)
        {
            return nd_np_ops.squeeze(this, axis);
        }

        public ndarray isnan()
        {
            return nd_np_ops.isnan(this);
        }

        public ndarray isinf()
        {
            return nd_np_ops.isinf(this);
        }

        public ndarray isposinf()
        {
            return nd_np_ops.isposinf(this);
        }

        public ndarray isneginf()
        {
            return nd_np_ops.isneginf(this);
        }

        public ndarray isfinite()
        {
            return nd_np_ops.isfinite(this);
        }

        public ndarray where(ndarray x = null, ndarray y = null)
        {
            return nd_np_ops.where(this, x, y);
        }

        public ndarray polyval(ndarray p)
        {
            return nd_np_ops.polyval(this, p);
        }

        public ndarray bincount(ndarray weights = null, int minlength = 0)
        {
            return nd_np_ops.bincount(this, weights, minlength);
        }

        public ndarray pad(int[] pad_width = null, string mode = "constant")
        {
            return nd_np_ops.pad(this, pad_width, mode);
        }

        public ndarray prod(int? axis = null, DType dtype = null, bool keepdims = false, float? initial = null)
        {
            return nd_np_ops.prod(this, axis, dtype, keepdims: keepdims, initial: initial);
        }

        public ndarray dot(ndarray b)
        {
            return nd_np_ops.dot(this, b);
        }

        public ndarray cumsum(int? axis = null, DType dtype = null)
        {
            return nd_np_ops.cumsum(this, axis, dtype);
        }

        public ndarray moveaxis(int source, int destination)
        {
            return nd_np_ops.moveaxis(this, source, destination);
        }

        public ndarray moveaxis(int[] source, int[] destination)
        {
            return nd_np_ops.moveaxis(this, source, destination);
        }

        public ndarray copy()
        {
            return nd_np_ops.copy(this);
        }

        public ndarray rollaxis(int axis, int start = 0)
        {
            return nd_np_ops.rollaxis(this, axis, start);
        }

        public ndarray diag(int k = 0)
        {
            return nd_np_ops.diag(this, k);
        }

        public ndarray diagflat(int k = 0)
        {
            return nd_np_ops.diagflat(this, k);
        }

        public ndarray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return nd_np_ops.diagonal(this, offset, axis1, axis2);
        }

        public ndarray sum(int? axis = null, DType dtype = null, bool keepdims = false, float? initial = null)
        {
            return nd_np_ops.sum(this, axis, dtype, keepdims: keepdims, initial: initial);
        }
    }
}
