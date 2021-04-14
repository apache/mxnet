using System;
using System.Collections.Generic;
using System.Text;
using MxNet.ND.Numpy;
using static NumpyDotNet.np;

namespace MxNet.Numpy
{
    public partial class np
    {
        public static DType Float16 = DType.Float16;
        public static DType Float32 = DType.Float32;
        public static DType Float64 = DType.Float64;
        public static DType Int8 = DType.Int8;
        public static DType UInt8 = DType.UInt8;
        public static DType Int16 = DType.Int16;
        public static DType Int32 = DType.Int32;
        public static DType Int64 = DType.Int64;
        public static DType Bool = DType.Bool;

        public static double nan => double.NaN;
        public static double NAN => double.NaN;
        public static double NaN => double.NaN;
        public static double pi => Math.PI;
        public static double e => Math.E;
        public static double euler_gamma => 0.57721566490153286060651209008240243d;
        public static double inf => double.PositiveInfinity;
        public static double Inf => double.PositiveInfinity;
        public static double NINF => double.NegativeInfinity;
        public static double PINF => double.PositiveInfinity;

        public static Random random = new Random();

        public static Linalg linalg = new Linalg();

        internal static Dictionary<string, DType> _STR_2_DTYPE_ = new Dictionary<string, DType> {
            {
                "float32",
                Float32},
            {
                "float64",
                Float64},
            {
                "float",
                Float64},
            {
                "uint8",
                UInt8},
            {
                "int8",
                Int8},
            {
                "int32",
                Int32},
            {
                "int64",
                Int64},
            {
                "int",
                Int64},
            {
                "bool",
                Bool},
            {
                "None",
                null
            }
        };

        public static ndarray empty(Shape shape, DType dtype = null, string order= "C", Context ctx= null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return new ndarray(shape, ctx: ctx, dtype: dtype);
        }

        public static ndarray array(Array obj, DType dtype= null, Context ctx= null)
        {
            return new ndarray(obj, ctx: ctx, dtype: dtype);
        }

        public static Shape shape(ndarray a)
        {
            return nd_np_ops.shape(a);
        }

        public static ndarray zeros(Shape shape, DType dtype= null, string order= "C", Context ctx= null)
        {
            return nd_np_ops.zeros(shape, dtype, order, ctx);
        }

        public static ndarray ones(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            return nd_np_ops.ones(shape, dtype, order, ctx);
        }

        public static ndarray broadcast_to(ndarray array, Shape shape)
        {
            return nd_np_ops.broadcast_to(array, shape);
        }

        public static ndarray full(Shape shape, double fill_value, DType dtype= null, string order= "C", Context ctx= null, ndarray @out= null)
        {
            return nd_np_ops.full(shape, fill_value, dtype, order, ctx, @out);
        }

        public static ndarray full_like(ndarray a, double fill_value, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.full_like(a, fill_value, dtype, order, ctx, @out);
        }

        public static bool all(ndarray a)
        {
            return nd_np_ops.all(a);
        }

        public static ndarray all(ndarray a, int axis, ndarray @out= null, bool keepdims= false)
        {
            return nd_np_ops.all(a, axis, @out, keepdims);
        }

        public static bool any(ndarray a)
        {
            return nd_np_ops.any(a);
        }

        public static ndarray any(ndarray a, int axis, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.any(a, axis, @out, keepdims);
        }

        public static ndarray identity(int n, DType dtype= null, Context ctx= null)
        {
            return nd_np_ops.identity(n, dtype, ctx);
        }

        public static ndarray take(ndarray a, ndarray indices, int? axis= null, string mode= "raise", ndarray @out= null)
        {
            return nd_np_ops.take(a, indices, axis, mode, @out);
        }

        public static ndarray unique(ndarray ar, int? axis= null)
        {
            return nd_np_ops.unique(ar, axis);
        }

        public static (ndarray, ndarray, ndarray, ndarray) unique(ndarray ar, bool return_index, bool return_inverse = false, bool return_counts = false, int? axis = null)
        {
            return nd_np_ops.unique(ar, return_index, return_inverse, return_counts, axis);
        }

        public static ndarray add(ndarray x1, ndarray x2, ndarray @out= null)
        {
            return nd_np_ops.add(x1, x2, @out);
        }

        public static ndarray add(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.add(x1, x2, @out);
        }

        public static ndarray add(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.add(x1, x2, @out);
        }

        public static ndarray subtract(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.subtract(x1, x2, @out);
        }

        public static ndarray subtract(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.subtract(x1, x2, @out);
        }

        public static ndarray subtract(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.subtract(x1, x2, @out);
        }

        public static ndarray mutiply(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.multiply(x1, x2, @out);
        }

        public static ndarray mutiply(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.multiply(x1, x2, @out);
        }

        public static ndarray mutiply(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.multiply(x1, x2, @out);
        }

        public static ndarray divide(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.divide(x1, x2, @out);
        }

        public static ndarray divide(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.divide(x1, x2, @out);
        }

        public static ndarray divide(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.divide(x1, x2, @out);
        }

        public static ndarray true_divide(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.true_divide(x1, x2, @out);
        }

        public static ndarray true_divide(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.true_divide(x1, x2, @out);
        }

        public static ndarray true_divide(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.true_divide(x1, x2, @out);
        }

        public static ndarray mod(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.mod(x1, x2, @out);
        }

        public static ndarray mod(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.mod(x1, x2, @out);
        }

        public static ndarray mod(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.mod(x1, x2, @out);
        }

        public static ndarray fmod(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.fmod(x1, x2, @out);
        }

        public static ndarray matmul(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.matmul(x1, x2, @out);
        }

        public static ndarray remainder(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.remainder(x1, x2, @out);
        }

        public static ndarray power(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.power(x1, x2, @out);
        }

        public static ndarray gcd(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.gcd(x1, x2, @out);
        }

        public static ndarray lcm(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.lcm(x1, x2, @out);
        }

        public static ndarray sin(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.sin(x, @out);
        }

        public static ndarray cos(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.cos(x, @out);
        }

        public static ndarray sinh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.sinh(x, @out);
        }

        public static ndarray cosh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.cosh(x, @out);
        }

        public static ndarray tanh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.tanh(x, @out);
        }

        public static ndarray log10(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.log10(x, @out);
        }

        public static ndarray sqrt(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.sqrt(x, @out);
        }

        public static ndarray cbrt(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.cbrt(x, @out);
        }

        public static ndarray abs(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.abs(x, @out);
        }

        public static ndarray fabs(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.fabs(x, @out);
        }

        public static ndarray absolute(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.absolute(x, @out);
        }

        public static ndarray exp(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.exp(x, @out);
        }

        public static ndarray expm1(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.expm1(x, @out);
        }

        public static ndarray arcsin(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arcsin(x, @out);
        }

        public static ndarray arccos(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arccos(x, @out);
        }

        public static ndarray arctan(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arctan(x, @out);
        }

        public static ndarray sign(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.sign(x, @out);
        }

        public static ndarray log(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.log(x, @out);
        }

        public static ndarray rint(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.rint(x, @out);
        }

        public static ndarray log2(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.log2(x, @out);
        }

        public static ndarray log1p(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.log1p(x, @out);
        }

        public static ndarray degrees(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.degrees(x, @out);
        }

        public static ndarray rad2deg(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.rad2deg(x, @out);
        }

        public static ndarray radians(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.radians(x, @out);
        }

        public static ndarray deg2rad(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.deg2rad(x, @out);
        }

        public static ndarray reciprocal(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.reciprocal(x, @out);
        }

        public static ndarray square(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.square(x, @out);
        }

        public static ndarray negative(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.negative(x, @out);
        }

        public static ndarray fix(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.fix(x, @out);
        }

        public static ndarray tan(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.tan(x, @out);
        }

        public static ndarray ceil(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.ceil(x, @out);
        }

        public static ndarray floor(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.floor(x, @out);
        }

        public static ndarray invert(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.invert(x, @out);
        }

        public static ndarray bitwise_not(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.bitwise_not(x, @out);
        }

        public static ndarray trunc(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.trunc(x, @out);
        }

        public static ndarray logical_not(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.logical_not(x, @out);
        }

        public static ndarray arcsinh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arcsinh(x, @out);
        }

        public static ndarray arccosh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arccosh(x, @out);
        }

        public static ndarray arctanh(ndarray x, ndarray @out = null)
        {
            return nd_np_ops.arctanh(x, @out);
        }

        public static ndarray argsort(ndarray x, int axis= -1, string kind= null, string order= null)
        {
            return nd_np_ops.argsort(x, axis, kind, order);
        }

        public static ndarray sort(ndarray x, int axis = -1, string kind = null, string order = null)
        {
            return nd_np_ops.sort(x, axis, kind, order);
        }

        public static ndarray tensordot(ndarray a, ndarray b, int axes= 2)
        {
            return nd_np_ops.tensordot(a, b, axes);
        }

        public static ndarray histogram(ndarray a, int bins = 10, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            return nd_np_ops.histogram(a, bins, range, normed, weights, density);
        }

        public static ndarray histogram(ndarray a, ndarray bins, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            return nd_np_ops.histogram(a, bins, range, normed, weights, density);
        }

        public static ndarray eye(int N, int? M= null, int k= 0, Context ctx = null, DType dtype= null)
        {
            return nd_np_ops.eye(N, M, k, ctx, dtype);
        }

        public static (ndarray, float?) linspace(float start, float stop, int num= 50, bool endpoint= true, bool retstep= false, DType dtype= null, int axis= 0, Context ctx= null)
        {
            return nd_np_ops.linspace(start, stop, num, endpoint, retstep, dtype, axis, ctx);
        }

        public static ndarray logspace(float start, float stop, int num = 50, bool endpoint = true, bool retstep = false, DType dtype = null, int axis = 0, Context ctx = null)
        {
            return nd_np_ops.logspace(start, stop, num, endpoint, retstep, dtype, axis, ctx);
        }

        public static ndarray expand_dims(ndarray a, int axis)
        {
            return nd_np_ops.expand_dims(a, axis);
        }

        public static ndarray tile(ndarray a, params int[] reps)
        {
            return nd_np_ops.tile(a, reps);
        }

        public static ndarray trace(ndarray a, int offset= 0, int axis1= 0, int axis2= 1, ndarray @out= null)
        {
            return nd_np_ops.trace(a, offset, axis1, axis2, @out);
        }

        public static ndarray transpose(ndarray a, params int[] axes)
        {
            return nd_np_ops.transpose(a, axes);
        }

        public static ndarray repeat(ndarray a, int repeats, int? axis= null)
        {
            return nd_np_ops.repeat(a, repeats, axis);
        }

        public static ndarray tril(ndarray m, int k = 0)
        {
            return nd_np_ops.tril(m, k);
        }

        public static ndarray tri(int N, int? M= null, int k= 0, DType dtype= null, Context ctx= null)
        {
            return nd_np_ops.tri(N, M, k, dtype, ctx);
        }

        public static ndarray triu_indices(int n, int k = 0, int? m = null, Context ctx = null)
        {
            return nd_np_ops.triu_indices(n, k, m, ctx);
        }

        public static ndarray triu_indices_from(ndarray ndarray, int k = 0)
        {
            return nd_np_ops.triu_indices_from(ndarray, k);
        }

        public static ndarray tril_indices(int n, int k = 0, int? m = null)
        {
            return nd_np_ops.tril_indices(n, k, m);
        }

        public static ndarray triu(int n, int k = 0)
        {
            return nd_np_ops.triu(n, k);
        }

        public static ndarray arange(int start, int? stop = null, int step = 1, DType dtype = null, Context ctx = null)
        {
            return nd_np_ops.arange(start, stop, step, dtype, ctx);
        }

        public static ndarray[] split(ndarray ary, int[] indices_or_sections, int axis= 0)
        {
            return nd_np_ops.split(ary, indices_or_sections, axis);
        }

        public static ndarray[] array_split(ndarray ary, int[] indices_or_sections, int axis = 0)
        {
            return nd_np_ops.array_split(ary, indices_or_sections, axis);
        }

        public static ndarray vsplit(ndarray ary, int[] indices_or_sections)
        {
            return nd_np_ops.vsplit(ary, indices_or_sections);
        }

        public static ndarray dsplit(ndarray ary, int[] indices_or_sections)
        {
            return nd_np_ops.dsplit(ary, indices_or_sections);
        }

        public static ndarray concatenate(ndarray[] seq, int axis = 0, ndarray @out = null)
        {
            return nd_np_ops.concatenate(seq, axis, @out);
        }

        public static ndarray append(ndarray arr, ndarray values, int? axis= null)
        {
            return nd_np_ops.append(arr, values, axis);
        }

        public static ndarray stack(ndarray[] arrays, int axis = 0, ndarray @out = null)
        {
            return nd_np_ops.stack(arrays, axis, @out);
        }

        public static ndarray vstack(ndarray[] arrays, ndarray @out = null)
        {
            return nd_np_ops.vstack(arrays, @out);
        }

        public static ndarray row_stack(ndarray[] arrays)
        {
            return nd_np_ops.row_stack(arrays);
        }

        public static ndarray column_stack(ndarray[] arrays)
        {
            return nd_np_ops.column_stack(arrays);
        }

        public static ndarray hstack(ndarray[] arrays)
        {
            return nd_np_ops.hstack(arrays);
        }

        public static ndarray dstack(ndarray[] arrays)
        {
            return nd_np_ops.dstack(arrays);
        }

        public static ndarray maximum(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.maximum(x1, x2, @out);
        }

        public static ndarray fmax(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.fmax(x1, x2, @out);
        }

        public static ndarray minimum(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.minimum(x1, x2, @out);
        }

        public static ndarray fmin(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.fmin(x1, x2, @out);
        }

        public static ndarray max(ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.max(a, axis, @out, keepdims);
        }

        public static ndarray min(ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.min(a, axis, @out, keepdims);
        }

        public static ndarray swapaxes(ndarray a, int axis1, int axis2)
        {
            return nd_np_ops.swapaxes(a, axis1, axis2);
        }

        public static ndarray clip(ndarray a, float a_min, float a_max, ndarray @out = null)
        {
            return nd_np_ops.clip(a, a_min, a_max, @out);
        }

        public static ndarray argmax(ndarray a, int? axis = null, ndarray @out = null)
        {
            return nd_np_ops.argmax(a, axis, @out);
        }

        public static ndarray argmin(ndarray a, int? axis = null, ndarray @out = null)
        {
            return nd_np_ops.argmin(a, axis, @out);
        }

        public static ndarray amax(ndarray a, int? axis = null, bool keepdims = false, ndarray @out = null)
        {
            return nd_np_ops.amax(a, axis, keepdims,@out);
        }

        public static ndarray amin(ndarray a, int? axis = null, bool keepdims = false, ndarray @out = null)
        {
            return nd_np_ops.amin(a, axis, keepdims, @out);
        }

        public static ndarray average(ndarray a, int? axis= null, ndarray weights= null, bool returned= false, ndarray @out = null)
        {
            return nd_np_ops.average(a, axis, weights, returned, @out);
        }

        public static ndarray mean(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.mean(a, axis, dtype, @out, keepdims);
        }

        public static ndarray std(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.std(a, axis, dtype, @out, keepdims);
        }

        public static ndarray delete(ndarray arr, int obj, int? axis= null)
        {
            return nd_np_ops.delete(arr, obj, axis);
        }

        public static ndarray delete(ndarray arr, Slice obj, int? axis = null)
        {
            return nd_np_ops.delete(arr, obj, axis);
        }

        public static ndarray delete(ndarray arr, ndarray obj, int? axis = null)
        {
            return nd_np_ops.delete(arr, obj, axis);
        }

        public static ndarray var(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            return nd_np_ops.var(a, axis, dtype, @out, keepdims);
        }

        public static ndarray indices(Shape dimensions, DType dtype= null, Context ctx= null)
        {
            return nd_np_ops.indices(dimensions, dtype, ctx);
        }

        public static ndarray copysign(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.copysign(x1, x2, @out);
        }

        public static ndarray ravel(ndarray x, string order = "x")
        {
            return nd_np_ops.ravel(x, order);
        }

        public static ndarray unravel_index(ndarray indices, Shape shape, string order = "x")
        {
            return nd_np_ops.unravel_index(indices, shape, order);
        }

        public static ndarray flatnonzero(ndarray x)
        {
            return nd_np_ops.flatnonzero(x);
        }

        public static ndarray diag_indices_from(ndarray x)
        {
            return nd_np_ops.diag_indices_from(x);
        }

        public static ndarray hanning(int M, DType dtype= null, Context ctx= null)
        {
            return nd_np_ops.hanning(M, dtype, ctx);
        }

        public static ndarray hamming(int M, DType dtype = null, Context ctx = null)
        {
            return nd_np_ops.hamming(M, dtype, ctx);
        }

        public static ndarray blackman(int M, DType dtype = null, Context ctx = null)
        {
            return nd_np_ops.blackman(M, dtype, ctx);
        }

        public static ndarray flip(ndarray m, int? axis = null, ndarray @out = null)
        {
            return nd_np_ops.flip(m, axis, @out);
        }

        public static ndarray flipud(ndarray x)
        {
            return nd_np_ops.flipud(x);
        }

        public static ndarray fliplr(ndarray x)
        {
            return nd_np_ops.fliplr(x);
        }

        public static ndarray around(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return nd_np_ops.around(x, decimals, @out);
        }

        public static ndarray round(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return nd_np_ops.round(x, decimals, @out);
        }

        public static ndarray round_(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return nd_np_ops.round_(x, decimals, @out);
        }

        public static ndarray arctan2(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.arctan2(x1, x2, @out);
        }

        public static ndarray hypot(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.hypot(x1, x2, @out);
        }

        public static ndarray bitwise_and(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.bitwise_and(x1, x2, @out);
        }

        public static ndarray bitwise_xor(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.bitwise_xor(x1, x2, @out);
        }

        public static ndarray bitwise_or(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.bitwise_or(x1, x2, @out);
        }

        public static ndarray ldexp(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.ldexp(x1, x2, @out);
        }

        public static ndarray vdot(ndarray a, ndarray b)
        {
            return nd_np_ops.vdot(a, b);
        }

        public static ndarray inner(ndarray a, ndarray b)
        {
            return nd_np_ops.inner(a, b);
        }

        public static ndarray outer(ndarray a, ndarray b)
        {
            return nd_np_ops.outer(a, b);
        }

        public static ndarray cross(ndarray a, ndarray b, int axisa= -1, int axisb = -1, int axisc = -1, int? axis = null)
        {
            return nd_np_ops.cross(a, b, axisa, axisb, axisc, axis);
        }

        public static ndarray kron(ndarray a, ndarray b)
        {
            return nd_np_ops.kron(a, b);
        }

        public static ndarray equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.equal(x1, x2, @out);
        }

        public static ndarray equal(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.equal(x1, x2, @out);
        }

        public static ndarray equal(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.equal(x1, x2, @out);
        }

        public static ndarray not_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.not_equal(x1, x2, @out);
        }

        public static ndarray not_equal(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.not_equal(x1, x2, @out);
        }

        public static ndarray not_equal(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.not_equal(x1, x2, @out);
        }

        public static ndarray greater(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.greater(x1, x2, @out);
        }

        public static ndarray greater(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.greater(x1, x2, @out);
        }

        public static ndarray greater(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.greater(x1, x2, @out);
        }

        public static ndarray less(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.less(x1, x2, @out);
        }

        public static ndarray less(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.less(x1, x2, @out);
        }

        public static ndarray less(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.less(x1, x2, @out);
        }

        public static ndarray logical_and(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.logical_and(x1, x2, @out);
        }

        public static ndarray logical_or(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.logical_or(x1, x2, @out);
        }

        public static ndarray logical_xor(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.logical_xor(x1, x2, @out);
        }

        public static ndarray greater_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.greater_equal(x1, x2, @out);
        }

        public static ndarray greater_equal(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.greater_equal(x1, x2, @out);
        }

        public static ndarray greater_equal(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.greater_equal(x1, x2, @out);
        }

        public static ndarray less_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.less_equal(x1, x2, @out);
        }

        public static ndarray less_equal(float x1, ndarray x2, ndarray @out = null)
        {
            return nd_np_ops.less_equal(x1, x2, @out);
        }

        public static ndarray less_equal(ndarray x1, float x2, ndarray @out = null)
        {
            return nd_np_ops.less_equal(x1, x2, @out);
        }

        public static ndarray roll(ndarray a, int shift, int? axis = null)
        {
            return nd_np_ops.roll(a, shift, axis);
        }

        public static ndarray roll(ndarray a, int[] shift, int? axis = null)
        {
            return nd_np_ops.roll(a, shift, axis);
        }

        public static ndarray rot90(ndarray m, int k= 1, params int[] axes)
        {
            return nd_np_ops.rot90(m, k, axes);
        }

        public static ndarray hsplit(ndarray ary, params int[] indices_or_sections)
        {
            return nd_np_ops.hsplit(ary, indices_or_sections);
        }

        public static ndarray einsum(string subscripts, ndarray[] operands, ndarray @out = null, bool optimize = false)
        {
            return nd_np_ops.einsum(subscripts, operands, @out, optimize);
        }

        public static ndarray insert(ndarray arr, int obj, ndarray values, int? axis= null)
        {
            return nd_np_ops.insert(arr, obj, values, axis);
        }

        public static ndarray insert(ndarray arr, ndarray obj, ndarray values, int? axis = null)
        {
            return nd_np_ops.insert(arr, obj, values, axis);
        }

        public static ndarray nonzero(ndarray a)
        {
            return nd_np_ops.nonzero(a);
        }

        public static ndarray percentile(ndarray a, ndarray q, int? axis= null, ndarray @out= null, bool? overwrite_input= null, string interpolation= "linear", bool keepdims= false)
        {
            return nd_np_ops.percentile(a, q, axis, @out, overwrite_input, interpolation, keepdims);
        }

        public static ndarray median(ndarray a, int? axis = null, ndarray @out = null, bool? overwrite_input = null, bool keepdims = false)
        {
            return nd_np_ops.median(a, axis, @out, overwrite_input, keepdims);
        }

        public static ndarray quantile(ndarray a, ndarray q, int? axis = null, ndarray @out = null, bool? overwrite_input = null, string interpolation = "linear", bool keepdims = false)
        {
            return nd_np_ops.quantile(a, q, axis, @out, overwrite_input, interpolation, keepdims);
        }

        public static bool shares_memory(ndarray a, ndarray b, int? max_work = null)
        {
            return nd_np_ops.shares_memory(a, b, max_work);
        }

        public static bool may_share_memory(ndarray a, ndarray b, int? max_work = null)
        {
            return nd_np_ops.may_share_memory(a, b, max_work);
        }

        public static ndarray diff(ndarray a, int n= 1, int axis= -1, ndarray prepend= null, ndarray  append = null)
        {
            return nd_np_ops.diff(a, n, axis, prepend, append);
        }

        public static ndarray ediff1d(ndarray ary, ndarray to_end = null, ndarray to_begin = null)
        {
            return nd_np_ops.ediff1d(ary, to_end, to_begin);
        }

        public static ndarray resize(ndarray a, Shape new_shape)
        {
            return nd_np_ops.resize(a, new_shape);
        }

        public static ndarray interp(ndarray x, float[] xp, float[] fp, float? left= null, float? right= null, float? period= null)
        {
            return nd_np_ops.interp(x, xp, fp, left, right, period);
        }

        public static ndarray full_like(ndarray a, float fill_value, DType dtype= null, string order= "C", Context ctx= null, ndarray @out= null)
        {
            return nd_np_ops.full_like(a, fill_value, dtype, order, ctx, @out);
        }

        public static ndarray zeros_like(ndarray a, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.zeros_like(a, dtype, order, ctx, @out);
        }

        public static ndarray ones_like(ndarray a, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.ones_like(a, dtype, order, ctx, @out);
        }

        public static ndarray fill_diagonal(ndarray a, float val, bool wrap =false)
        {
            return nd_np_ops.fill_diagonal(a, val, wrap);
        }

        public static ndarray nan_to_num(ndarray x, bool copy= true, float nan= 0, float? posinf= null, float? neginf= null)
        {
            return nd_np_ops.nan_to_num(x, copy, nan, posinf, neginf);
        }

        public static ndarray squeeze(ndarray a, int? axis = null)
        {
            return nd_np_ops.squeeze(a, axis);
        }

        public static ndarray isnan(ndarray a, ndarray @out = null)
        {
            return nd_np_ops.isnan(a, @out);
        }

        public static ndarray isinf(ndarray a, ndarray @out = null)
        {
            return nd_np_ops.isinf(a, @out);
        }

        public static ndarray isposinf(ndarray a, ndarray @out = null)
        {
            return nd_np_ops.isposinf(a, @out);
        }

        public static ndarray isneginf(ndarray a, ndarray @out = null)
        {
            return nd_np_ops.isneginf(a, @out);
        }

        public static ndarray isfinite(ndarray a, ndarray @out = null)
        {
            return nd_np_ops.isfinite(a, @out);
        }

        public static ndarray where(ndarray condition, ndarray x = null, ndarray y = null)
        {
            return nd_np_ops.where(condition, x, y);
        }

        public static ndarray polyval(ndarray p, ndarray x)
        {
            return nd_np_ops.polyval(p, x);
        }

        public static ndarray bincount(ndarray x, ndarray weights= null, int minlength= 0)
        {
            return nd_np_ops.bincount(x, weights, minlength);
        }

        public static ndarray atleast_1d(params ndarray[] arys)
        {
            return nd_np_ops.atleast_1d(arys);
        }

        public static ndarray atleast_2d(params ndarray[] arys)
        {
            return nd_np_ops.atleast_2d(arys);
        }

        public static ndarray atleast_3d(params ndarray[] arys)
        {
            return nd_np_ops.atleast_3d(arys);
        }

        public static ndarray pad(ndarray x, int[] pad_width= null, string mode= "constant")
        {
            return nd_np_ops.pad(x, pad_width, mode);
        }

        public static ndarray prod(ndarray a, int? axis= null, DType dtype= null, ndarray @out= null, bool keepdims= false, float? initial= null)
        {
            return nd_np_ops.prod(a, axis, dtype, @out, keepdims, initial);
        }

        public static ndarray dot(ndarray a, ndarray b, ndarray @out = null)
        {
            return nd_np_ops.dot(a, b, @out);
        }

        public static ndarray cumsum(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null)
        {
            return nd_np_ops.cumsum(a, axis, dtype, @out);
        }

        public static ndarray reshape(ndarray a, Shape newshape, bool reverse = false, string order= "C")
        {
            return nd_np_ops.reshape(a, newshape, reverse, order);
        }

        public static ndarray moveaxis(ndarray a, int source, int destination)
        {
            return nd_np_ops.moveaxis(a, source, destination);
        }

        public static ndarray moveaxis(ndarray a, int[] source, int[] destination)
        {
            return nd_np_ops.moveaxis(a, source, destination);
        }

        public static ndarray copy(ndarray a)
        {
            return nd_np_ops.copy(a);
        }

        public static ndarray rollaxis(ndarray a, int axis, int start=0)
        {
            return nd_np_ops.rollaxis(a, axis, start);
        }

        public static ndarray diag(ndarray v, int k = 0)
        {
            return nd_np_ops.diag(v, k);
        }

        public static ndarray diagflat(ndarray v, int k = 0)
        {
            return nd_np_ops.diagflat(v, k);
        }

        public static ndarray diagonal(ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return nd_np_ops.diagonal(a, offset, axis1, axis2);
        }

        public static ndarray sum(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false, float? initial = null)
        {
            return nd_np_ops.sum(a, axis, dtype, @out, keepdims, initial);
        }

        public static ndarray meshgrid(ndarray[] xi, string indexing = "xy", bool sparse = false, bool copy = true)
        {
            throw new NotImplementedException();
        }
    }
}
