using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MxNet.ND.Numpy
{
    internal class nd_np_ops
    {
        private static bool? _INT64_TENSOR_SIZE_ENABLED = null;

        public static Random random = new Random();
        public static Linalg linalg = new Linalg();

        private static dynamic _api_internal = new _api_internals();

        internal static bool Int64Enabled()
        {
            //if (_INT64_TENSOR_SIZE_ENABLED == null)
            //{
            //    _INT64_TENSOR_SIZE_ENABLED = Runtime.FeatureList().IsEnabled("INT64_TENSOR_SIZE");
            //}

            //return _INT64_TENSOR_SIZE_ENABLED.Value;
            return true;
        }

        public static ndarray empty(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return new ndarray(shape, ctx: ctx, dtype: dtype);
        }

        public static Shape shape(ndarray a)
        {
            return a.shape;
        }

        public static ndarray zeros(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return _api_internal.zeros(shape: shape, ctx: ctx, dtype: dtype);
        }

        public static ndarray ones(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return _api_internal.ones(shape: shape, ctx: ctx, dtype: dtype);
        }

        public static ndarray broadcast_to(ndarray array, Shape shape)
        {
            return _api_internal.broadcast_to(array: array, shape: shape);
        }

        public static ndarray full(Shape shape, double fill_value, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            if (order != "C") throw new NotSupportedException("order");

            @out = _api_internal.full(shape: shape, value: fill_value, dtype: dtype, ctx: ctx);
            return @out;
        }

        public static ndarray zero_like(ndarray prototype, DType dtype = null, string order = "C", ndarray @out = null)
        {
            @out =  full_like(prototype, 0, dtype, order, @out: @out);
            return @out;
        }

        public static bool all(ndarray a)
        {
            return ((ndarray)_api_internal.all(a: a)).AsScalar<bool>();
        }

        public static ndarray all(ndarray a, int axis, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.all(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static bool any(ndarray a)
        {
            return ((ndarray)_api_internal.any(a: a)).AsScalar<bool>();
        }

        public static ndarray any(ndarray a, int axis, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.any(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static ndarray identity(int n, DType dtype = null, Context ctx = null)
        {
            return _api_internal.identity(shape: new Shape(n, n), dtype: dtype, ctx: ctx);
        }

        public static ndarray take(ndarray a, ndarray indices, int? axis = null, string mode = "raise", ndarray @out = null)
        {
            if (!new string[] { "wrap", "clip", "raise" }.Contains(mode))
            {
                throw new NotSupportedException($"function take does not support mode '{mode}'");
            }
            if (axis == null)
            {
                return _api_internal.take(a: reshape(a, -1), indices: indices, axis: 0, mode: mode);
            }
            else
            {
                return _api_internal.take(a: a, indices: indices, axis: axis, mode: mode);
            }
        }

        public static ndarray unique(ndarray ar, int? axis = null)
        {
            var list = (NDArrayList)_api_internal.unique(ar: ar, axis: axis, multi: true);
            return list[0];
        }

        public static (ndarray, ndarray, ndarray, ndarray) unique(ndarray ar, bool return_index = false, bool return_inverse = false, bool return_counts = false, int? axis = null)
        {
            var ret = (NDArrayList)_api_internal.unique(ar: ar, return_index: return_index, return_inverse: return_inverse, return_counts: return_counts, axis: axis,  multi: true);
            return (ret[0], ret[1], ret[2], ret[3]);
        }

        public static ndarray add(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.add(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray add(ndarray x1, float x2, ndarray @out = null)
        {
            return add(x1, full_like(x1, x2));
        }

        public static ndarray add(float x1, ndarray x2, ndarray @out = null)
        {
            return add(full_like(x2, x1), x2);
        }

        public static ndarray subtract(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.subtract(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray subtract(ndarray x1, float x2, ndarray @out = null)
        {
            return subtract(x1, full_like(x1, x2));
        }

        public static ndarray subtract(float x1, ndarray x2, ndarray @out = null)
        {
            return subtract(full_like(x2, x1), x2);
        }

        public static ndarray multiply(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.multiply(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray multiply(ndarray x1, float x2, ndarray @out = null)
        {
            return multiply(x1, full_like(x1, x2));
        }

        public static ndarray multiply(float x1, ndarray x2, ndarray @out = null)
        {
            return multiply(full_like(x2, x1), x2);
        }

        public static ndarray divide(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.divide(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray divide(ndarray x1, float x2, ndarray @out = null)
        {
            return divide(x1, full_like(x1, x2));
        }

        public static ndarray divide(float x1, ndarray x2, ndarray @out = null)
        {
            return divide(full_like(x2, x1), x2);
        }

        public static ndarray true_divide(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.true_divide(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray true_divide(ndarray x1, float x2, ndarray @out = null)
        {
            return true_divide(x1, full_like(x1, x2));
        }

        public static ndarray true_divide(float x1, ndarray x2, ndarray @out = null)
        {
            return true_divide(full_like(x2, x1), x2);
        }

        public static ndarray mod(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.mod(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray mod(ndarray x1, float x2, ndarray @out = null)
        {
            return mod(x1, full_like(x1, x2));
        }

        public static ndarray mod(float x1, ndarray x2, ndarray @out = null)
        {
            return mod(full_like(x2, x1), x2);
        }

        public static ndarray fmod(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.fmod(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray matmul(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.matmul(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray remainder(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.remainder(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray power(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.power(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray power(ndarray x1, float x2, ndarray @out = null)
        {
            return power(x1, full_like(x1, x2));
        }

        public static ndarray gcd(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.gcd(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray gcd(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.gcd(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray gcd(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.gcd(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray lcm(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.lcm(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray lcm(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.lcm(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray lcm(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.lcm(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray sin(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.sin(x: x);
            return @out;
        }

        public static ndarray cos(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.cos(x: x);
            return @out;
        }

        public static ndarray sinh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.sinh(x: x);
            return @out;
        }

        public static ndarray cosh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.cosh(x: x);
            return @out;
        }

        public static ndarray tanh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.tanh(x: x);
            return @out;
        }

        public static ndarray log10(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.log10(x: x);
            return @out;
        }

        public static ndarray sqrt(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.sqrt(x: x);
            return @out;
        }

        public static ndarray cbrt(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.cbrt(x: x);
            return @out;
        }

        public static ndarray abs(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.abs(x: x);
            return @out;
        }

        public static ndarray fabs(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.fabs(x: x);
            return @out;
        }

        public static ndarray absolute(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.absolute(x: x);
            return @out;
        }

        public static ndarray exp(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.exp(x: x);
            return @out;
        }

        public static ndarray expm1(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.expm1(x: x);
            return @out;
        }

        public static ndarray arcsin(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arcsin(x: x);
            return @out;
        }

        public static ndarray arccos(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arccos(x: x);
            return @out;
        }

        public static ndarray arctan(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arctan(x: x);
            return @out;
        }

        public static ndarray sign(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.sign(x: x);
            return @out;
        }

        public static ndarray log(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.log(x: x);
            return @out;
        }

        public static ndarray rint(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.rint(x: x);
            return @out;
        }

        public static ndarray log2(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.log2(x: x);
            return @out;
        }

        public static ndarray log1p(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.log1p(x: x);
            return @out;
        }

        public static ndarray degrees(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.degrees(x: x);
            return @out;
        }

        public static ndarray rad2deg(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.rad2deg(x: x);
            return @out;
        }

        public static ndarray radians(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.radians(x: x);
            return @out;
        }

        public static ndarray deg2rad(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.deg2rad(x: x);
            return @out;
        }

        public static ndarray reciprocal(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.reciprocal(x: x);
            return @out;
        }

        public static ndarray square(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.square(x: x);
            return @out;
        }

        public static ndarray negative(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.negative(x: x);
            return @out;
        }

        public static ndarray fix(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.fix(x: x);
            return @out;
        }

        public static ndarray tan(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.tan(x: x);
            return @out;
        }

        public static ndarray ceil(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.ceil(x: x);
            return @out;
        }

        public static ndarray floor(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.floor(x: x);
            return @out;
        }

        public static ndarray invert(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.invert(x: x);
            return @out;
        }

        public static ndarray bitwise_not(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.bitwise_not(x: x);
            return @out;
        }

        public static ndarray trunc(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.trunc(x: x);
            return @out;
        }

        public static ndarray logical_not(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.logical_not(x: x);
            return @out;
        }

        public static ndarray arcsinh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arcsinh(x: x);
            return @out;
        }

        public static ndarray arccosh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arccosh(x: x);
            return @out;
        }

        public static ndarray arctanh(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.arctanh(x: x);
            return @out;
        }

        public static ndarray argsort(ndarray x, int axis = -1, string kind = null, string order = null)
        {
            if (order != null)
                throw new NotSupportedException("order is not supported");

            return _api_internal.argsort(a: x, axis: axis, is_ascend: true, dtype: np.Int64);
        }

        public static ndarray sort(ndarray x, int axis = -1, string kind = null, string order = null)
        {
            if (order != null)
                throw new NotSupportedException("order is not supported");

            return _api_internal.sort(a: x, axis: axis, is_ascend: true);
        }

        public static ndarray tensordot(ndarray a, ndarray b, Shape axes = null)
        {
            if (axes == null) axes = new Shape(2);

            return _api_internal.tensordot(a: a, b: b, axes: axes);
        }

        public static ndarray histogram(ndarray a, int bins = 10, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            if (normed.HasValue && normed.Value)
            {
                throw new NotSupportedException("normed is not supported yet...");
            }

            if (weights != null)
            {
                throw new NotSupportedException("weights is not supported yet...");
            }

            if (density.HasValue && density.Value)
            {
                throw new NotSupportedException("density is not supported yet...");
            }

            if (!range.HasValue)
                throw new Exception("range is required.");

            return _api_internal.histogram(a: a, bin_cnt: bins, range: new Tuple<float>(range.Value.Item1, range.Value.Item2));
        }

        public static ndarray histogram(ndarray a, ndarray bins, (float, float)? range = null, bool? normed = null, ndarray weights = null, bool? density = null)
        {
            if (normed.HasValue && normed.Value)
            {
                throw new NotSupportedException("normed is not supported yet...");
            }

            if (weights != null)
            {
                throw new NotSupportedException("weights is not supported yet...");
            }

            if (density.HasValue && density.Value)
            {
                throw new NotSupportedException("density is not supported yet...");
            }

            return _api_internal.histogram(a: a, bins: bins);
        }

        public static ndarray eye(int N, int? M = null, int k = 0, Context ctx = null, DType dtype = null)
        {
            if (ctx == null) ctx = Context.CurrentContext;
            if (dtype == null) dtype = np.Float32;

            return _api_internal.eye(N, M, k, ctx, dtype);
        }

        public static (ndarray, float?) linspace(float start, float stop, int num = 50, bool endpoint = true, bool retstep = false, DType dtype = null, int axis = 0, Context ctx = null)
        {
            var step = (stop - start) / (num - 1);
            if(retstep)
                return (_api_internal.linspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype), step);
            else
                return (_api_internal.linspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype), null);
        }

        public static ndarray logspace(float start, float stop, int num = 50, bool endpoint = true, bool retstep = false, DType dtype = null, int axis = 0, Context ctx = null)
        {
            return _api_internal.logspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype);
        }

        public static ndarray expand_dims(ndarray a, int axis)
        {
            return _api_internal.expand_dims(a: a, axis: axis);
        }

        public static ndarray tile(ndarray a, params int[] reps)
        {
            return _api_internal.tile(A: a, reps: new Shape(reps));
        }

        public static ndarray trace(ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1, ndarray @out = null)
        {
            @out = _api_internal.trace(a: a, offset: offset, axis1: axis1, axis2: axis2);
            return @out;
        }

        public static ndarray transpose(ndarray a, params int[] axes)
        {
            return _api_internal.transpose(a: a, axes: new Shape(axes));
        }

        public static ndarray repeat(ndarray a, int repeats, int? axis = null)
        {
            if (axis != null)
            {
                var tmp = swapaxes(a, 0, axis.Value);
                var res = _api_internal.repeats(a: tmp, repeats: repeats, axis: 0);
                return swapaxes(res, 0, axis.Value);
            }

            return _api_internal.repeats(a: a, repeats: repeats, axis: axis);
        }

        public static ndarray tril(ndarray m, int k = 0)
        {
            return _api_internal.tril(m: m, k: k);
        }

        public static ndarray tri(int N, int? M = null, int k = 0, DType dtype = null, Context ctx = null)
        {
            return _api_internal.tri(N: N, M: M, k: k, dtype: dtype, ctx: ctx);
        }

        public static ndarray triu_indices(int n, int k = 0, int? m = null, Context ctx = null)
        {
            return nonzero(negative(tri(N: n, M: m, k: k - 1, dtype: np.Bool, ctx: ctx)));
        }

        public static ndarray triu_indices_from(ndarray arr, int k = 0)
        {
            if (arr.ndim != 2)
            {
                throw new Exception("input array must be 2-d");
            }

            return triu_indices(arr.shape[-2], k: k, m: arr.shape[-1]);
        }

        public static NDArrayList tril_indices(int n, int k = 0, int? m = null)
        {
            if (m == null)
            {
                m = n;
            }

            return _api_internal.tril_indices(n: n, k: k, m: m, multi: true);
        }

        public static ndarray triu(int m, int k = 0)
        {
            return _api_internal.triu(m: m, k: k);
        }

        public static ndarray arange(int start, int? stop = null, int step = 1, DType dtype = null, Context ctx = null)
        {
            if (stop == null)
            {
                stop = start;
                start = 0;
            }
            
            if (step == 0)
            {
                throw new Exception("step cannot be 0");
            }

            return _api_internal.arange(start: start, stop: stop, step: step, dtype: dtype, ctx: ctx);
        }

        public static NDArrayList split(ndarray ary, int[] indices_or_sections, int axis = 0)
        {
            return (NDArrayList)_api_internal.split(ary: ary, indices_or_sections: indices_or_sections, axis: axis, multi: true);
        }

        public static NDArrayList array_split(ndarray ary, int[] indices_or_sections, int axis = 0)
        {
            return (NDArrayList)_api_internal.array_split(ary: ary, indices_or_sections: indices_or_sections, axis: axis, multi: true);
        }

        public static NDArrayList vsplit(ndarray ary, int[] indices_or_sections)
        {
            return (NDArrayList)_api_internal.vsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static NDArrayList hsplit(ndarray ary, int[] indices_or_sections)
        {
            return (NDArrayList)_api_internal.hsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static NDArrayList dsplit(ndarray ary, int[] indices_or_sections)
        {
            return (NDArrayList)_api_internal.dsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static ndarray concatenate(NDArrayList seq, int axis = 0, ndarray @out = null)
        {
            @out = _api_internal.concatenate(axis: axis, seq: seq);
            return @out;
        }

        public static ndarray append(ndarray arr, ndarray values, int? axis = null)
        {
            return _api_internal.concatenate(arr: arr, values: values, axis: axis);
        }

        public static ndarray stack(NDArrayList arrays, int axis = 0, ndarray @out = null)
        {
            @out = _api_internal.stack(axis: axis, arrays: arrays);
            return @out;
        }

        public static ndarray vstack(NDArrayList arrays, ndarray @out = null)
        {
            @out = _api_internal.vstack(arrays: arrays);
            return @out;
        }

        public static ndarray row_stack(NDArrayList arrays)
        {
            var @out = _api_internal.vstack(arrays: arrays);
            return @out;
        }

        public static ndarray column_stack(NDArrayList tup)
        {
            return _api_internal.column_stack(tup: tup);
        }

        public static ndarray hstack(NDArrayList arrays)
        {
            return _api_internal.hstack(arrays: arrays);
        }

        public static ndarray dstack(NDArrayList arrays)
        {
            return _api_internal.dstack(arrays: arrays);
        }

        public static ndarray maximum(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.maximum(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray maximum(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.maximum(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray maximum(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.maximum(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray fmax(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.fmax(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray fmax(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.fmax(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray fmax(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.fmax(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray minimum(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.minimum(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray minimum(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.minimum(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray minimum(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.minimum(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray fmin(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.fmin(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray fmin(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.fmin(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray fmin(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.fmin(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray max(ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            @out =_api_internal.max(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static ndarray min(ndarray a, int? axis = null, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.min(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static ndarray swapaxes(ndarray a, int axis1, int axis2)
        {
            return _api_internal.swapaxes(a: a, dim1: axis1, dim2: axis2);
        }

        public static ndarray clip(ndarray a, float? a_min, float? a_max, ndarray @out = null)
        {
            if (a_min == null && a_max == null)
            {
                throw new Exception("array_clip: must set either max or min");
            }

            @out =_api_internal.clip(a: a, a_min: a_min, a_max: a_max);
            return @out;
        }

        public static ndarray argmax(ndarray a, int? axis = null, ndarray @out = null)
        {
            @out = _api_internal.argmax(a: a, axis: axis);
            return @out;
        }

        public static ndarray argmin(ndarray a, int? axis = null, ndarray @out = null)
        {
            @out = _api_internal.argmin(a: a, axis: axis);
            return @out;
        }

        public static ndarray amax(ndarray a, int? axis = null, bool keepdims = false, ndarray @out = null)
        {
            @out = _api_internal.amax(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static ndarray amin(ndarray a, int? axis = null, bool keepdims = false, ndarray @out = null)
        {
            @out = _api_internal.amin(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static ndarray average(ndarray a, int? axis = null, ndarray weights = null, bool returned = false, ndarray @out = null)
        {
            @out =_api_internal.average(a: a, weights: weights, axis: axis, returned: returned);
            return @out;
        }

        public static ndarray mean(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.mean(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static ndarray std(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.std(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static ndarray delete(ndarray arr, int obj, int? axis = null)
        {
            return _api_internal.delete(arr: arr, obj: obj, axis: axis);
        }

        public static ndarray delete(ndarray arr, Slice obj, int? axis = null)
        {
            return _api_internal.delete(arr, start: obj.Begin, stop: obj.End, step: obj.Step, axis: axis);
        }

        public static ndarray delete(ndarray arr, ndarray obj, int? axis = null)
        {
            return _api_internal.delete(arr: arr, obj: obj, axis: axis);
        }

        public static ndarray var(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false)
        {
            @out = _api_internal.var(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static ndarray indices(Shape dimensions, DType dtype = null, Context ctx = null)
        {
            return _api_internal.indices(dimensions: dimensions, dtype: dtype, ctx: ctx);
        }

        public static ndarray copysign(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.copysign(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray copysign(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.copysign(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray copysign(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.copysign(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray ravel(ndarray x, string order = "C")
        {
            if (order == "F")
            {
                throw new NotSupportedException($"order {order} is not supported");
            }

            return reshape(x, -1);
        }

        public static NDArrayList unravel_index(ndarray indices, Shape shape, string order = "x")
        {
            if (order == "F")
            {
                throw new NotSupportedException($"order {order} is not supported");
            }

            return _api_internal.unravel_index_fallback(indices: indices, shape: shape, multi: true);
        }

        public static NDArrayList flatnonzero(ndarray a)
        {
            return nonzero(ravel(a));
        }

        public static NDArrayList diag_indices_from(ndarray arr)
        {
            return _api_internal.diag_indices_from(arr: arr, multi: true);
        }

        public static ndarray hanning(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.hanning(M: M, dtype: dtype, ctx: ctx);
        }

        public static ndarray hamming(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.hamming(M: M, dtype: dtype, ctx: ctx);
        }

        public static ndarray blackman(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.blackman(M: M, dtype: dtype, ctx: ctx);
        }

        public static ndarray flip(ndarray m, int? axis = null, ndarray @out = null)
        {
            @out = _api_internal.flip(m: m, axis: axis);
            return @out;
        }

        public static ndarray flipud(ndarray m)
        {
            return flip(m, 0);
        }

        public static ndarray fliplr(ndarray m)
        {
            return flip(m, 1);
        }

        public static ndarray around(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static ndarray round(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static ndarray round_(ndarray x, int decimals = 0, ndarray @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static ndarray arctan2(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: x2);
        }

        public static ndarray arctan2(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray arctan2(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.arctan2(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray hypot(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.hypot(x1: x1, x2: x2);
        }

        public static ndarray hypot(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray hypot(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.hypot(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray bitwise_and(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_and(x1: x1, x2: x2);
        }

        public static ndarray bitwise_and(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.bitwise_and(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray bitwise_and(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_and(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray bitwise_xor(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_xor(x1: x1, x2: x2);
        }

        public static ndarray bitwise_xor(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.bitwise_xor(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray bitwise_xor(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_xor(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray bitwise_or(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_or(x1: x1, x2: x2);
        }

        public static ndarray bitwise_or(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.bitwise_or(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray bitwise_or(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.bitwise_or(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray ldexp(ndarray x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.ldexp(x1: x1, x2: x2);
        }

        public static ndarray ldexp(ndarray x1, float x2, ndarray @out = null)
        {
            return _api_internal.ldexp(x1: x1, x2: full_like(x1, x2));
        }

        public static ndarray ldexp(float x1, ndarray x2, ndarray @out = null)
        {
            return _api_internal.ldexp(x1: full_like(x2, x1), x2: x2);
        }

        public static ndarray vdot(ndarray a, ndarray b)
        {
            return tensordot(a.ravel(), b.ravel(), 1);
        }

        public static ndarray inner(ndarray a, ndarray b)
        {
            return tensordot(a, b, new Shape(-1, -1));
        }

        public static ndarray outer(ndarray a, ndarray b)
        {
            return tensordot(a.reshape(-1), b.reshape(-1), new Shape(0));
        }

        public static ndarray cross(ndarray a, ndarray b, int axisa = -1, int axisb = -1, int axisc = -1, int? axis = null)
        {
            if (axis != null)
            {
                axisa = axis.Value;
                axisb = axis.Value;
                axisc = axis.Value;
            }

            return _api_internal.cross(a: a, b: b, axisa: axisa, axisb: axisb, axisc: axisc);
        }

        public static ndarray kron(ndarray a, ndarray b)
        {
            return _api_internal.kron(a: a, b: b);
        }

        public static ndarray equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.equal(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray equal(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray equal(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray not_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.not_equal(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray not_equal(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.not_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray not_equal(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.not_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray greater(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.greater(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray greater(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.greater(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray greater(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.greater(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray less(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.less(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray less(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.less(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray less(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.less(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray logical_and(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_and(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray logical_and(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.logical_and(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray logical_and(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_and(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray logical_or(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_or(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray logical_or(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.logical_or(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray logical_or(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_or(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray logical_xor(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_xor(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray logical_xor(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.logical_xor(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray logical_xor(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.logical_xor(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray greater_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.greater_equal(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray greater_equal(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.greater_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray greater_equal(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.greater_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray less_equal(ndarray x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.less_equal(x1: x1, x2: x2);
            return @out;
        }

        public static ndarray less_equal(ndarray x1, float x2, ndarray @out = null)
        {
            @out = _api_internal.less_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static ndarray less_equal(float x1, ndarray x2, ndarray @out = null)
        {
            @out = _api_internal.less_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static ndarray roll(ndarray a, int shift, int? axis = null)
        {
            return _api_internal.roll(a: a, shift: shift, axis: axis);
        }

        public static ndarray roll(ndarray a, int[] shift, int? axis = null)
        {
            return _api_internal.roll(a: a, shift: new Shape(shift), axis: axis);
        }

        public static ndarray rot90(ndarray m, int k = 1, params int[] axes)
        {
            return _api_internal.rot90(m: m, k: k, axes: new Shape(axes));
        }

        public static ndarray einsum(string subscripts, NDArrayList operands, ndarray @out = null, bool optimize = false)
        {
            return _api_internal.einsum(subscripts: subscripts, optimize_arg: Convert.ToInt32(optimize), operands: operands);
        }

        public static ndarray insert(ndarray arr, int obj, ndarray values, int? axis = null)
        {
            return _api_internal.insert_scalar(arr: arr, values: values, obj: obj, axis: axis);
        }

        public static ndarray insert(ndarray arr, Slice obj, ndarray values, int? axis = null)
        {
            return _api_internal.insert_slice(arr: arr, values: values, start: obj.Begin, stop: obj.End, step: obj.Step, axis: axis);
        }

        public static ndarray insert(ndarray arr, ndarray obj, ndarray values, int? axis = null)
        {
            return _api_internal.insert_tensor(arr: arr, obj: obj, values: values, axis: axis);
        }

        public static NDArrayList nonzero(ndarray a)
        {
            NDArrayList @out = _api_internal.nonzero(a: a, multi: true);
            @out = @out.Select(x => x.transpose()).ToList();
            return @out;
        }

        public static ndarray percentile(ndarray a, ndarray q, int? axis = null, ndarray @out = null, bool? overwrite_input = null, string interpolation = "linear", bool keepdims = false)
        {
            if (overwrite_input != null)
            {
                throw new NotSupportedException("overwrite_input is not supported yet");
            }

            @out = _api_internal.percentile(a: a, q: q, axis: axis, interpolation: interpolation, keepdims: keepdims);
            return @out;
        }

        public static ndarray median(ndarray a, int? axis = null, ndarray @out = null, bool? overwrite_input = null, bool keepdims = false)
        {
            return quantile(a: a, q: 0.5, axis: axis, @out: @out, overwrite_input: overwrite_input, interpolation: "midpoint", keepdims: keepdims);
        }

        public static ndarray quantile(ndarray a, ndarray q, int? axis = null, ndarray @out = null, bool? overwrite_input = null, string interpolation = "linear", bool keepdims = false)
        {
            if (overwrite_input != null)
            {
                throw new NotSupportedException("overwrite_input is not supported yet");
            }

            @out = _api_internal.percentile(a: a, q: q * 100, axis: axis, interpolation: interpolation, keepdims: keepdims);
            return @out;
        }

        public static bool shares_memory(ndarray a, ndarray b, int? max_work = null)
        {
            ndarray ret = _api_internal.share_memory(a: a, b: b);
            return ret.AsScalar<bool>();
        }

        public static bool may_share_memory(ndarray a, ndarray b, int? max_work = null)
        {
            ndarray ret = _api_internal.share_memory(a: a, b: b);
            return ret.AsScalar<bool>();
        }

        public static ndarray diff(ndarray a, int n = 1, int axis = -1, ndarray prepend = null, ndarray append = null)
        {
            if (prepend != null || append != null)
            {
                throw new NotSupportedException("prepend and append options are not supported yet");
            }

            return _api_internal.diff(a: a, n: n, axis: axis);
        }

        public static ndarray ediff1d(ndarray ary, ndarray to_end = null, ndarray to_begin = null)
        {
            return _api_internal.ediff1d(ary: ary, to_end: to_end, to_begin: to_begin);
        }

        public static ndarray resize(ndarray a, Shape new_shape)
        {
            return _api_internal.resize_fallback(a: a, new_shape: new_shape);
        }

        public static ndarray interp(ndarray x, float[] xp, float[] fp, float? left = null, float? right = null, float? period = null)
        {
            return _api_internal.interp(xp: xp, fp: fp, x: x, left: left, right: right, period: period);
        }

        public static ndarray full_like(ndarray a, double fill_value, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.full_like(a: a, fill_value: fill_value, dtype: dtype, ctx: ctx);
            return @out;
        }

        public static ndarray zeros_like(ndarray a, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            return full_like(a, 0, dtype, order, ctx, @out);
        }

        public static ndarray ones_like(ndarray a, DType dtype = null, string order = "C", Context ctx = null, ndarray @out = null)
        {
            return full_like(a, 1, dtype, order, ctx, @out);
        }

        public static ndarray fill_diagonal(ndarray a, float[] val, bool wrap = false)
        {
            return _api_internal.fill_diagonal(a, val, wrap, a);
        }

        public static ndarray fill_diagonal(ndarray a, float val, bool wrap = false)
        {
            return fill_diagonal(a, new float[] { val }, wrap);
        }

        public static ndarray nan_to_num(ndarray x, bool copy = true, float nan = 0, float? posinf = null, float? neginf = null)
        {
            if (new List<string> {
                    "int8",
                    "uint8",
                    "int32",
                    "int64"
                }.Contains(x.dtype.Name))
            {
                return x;
            }

            if (!copy)
            {
                return _api_internal.nan_to_num(x, copy, nan, posinf, neginf, x);
            }

            return _api_internal.nan_to_num(x, copy, nan, posinf, neginf, null);
        }

        public static ndarray squeeze(ndarray a, int? axis = null)
        {
            return _api_internal.squeeze(x: a, axis: axis);
        }

        public static ndarray isnan(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.isnan(x: x);
            return @out;
        }

        public static ndarray isinf(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.isinf(x: x);
            return @out;
        }

        public static ndarray isposinf(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.isposinf(x: x);
            return @out;
        }

        public static ndarray isneginf(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.isneginf(x: x);
            return @out;
        }

        public static ndarray isfinite(ndarray x, ndarray @out = null)
        {
            @out = _api_internal.isfinite(x: x);
            return @out;
        }

        public static ndarray where(ndarray condition, ndarray x = null, ndarray y = null)
        {
            if (x == null && y == null)
            {
                return nonzero(condition);
            }
            else
            {
                return _api_internal.where(condition: condition, x: x, y: y);
            }
        }

        public static ndarray polyval(ndarray p, ndarray x)
        {
            return _api_internal.polyval(p: p, x: x);
        }

        public static ndarray bincount(ndarray x, ndarray weights = null, int minlength = 0)
        {
            if (minlength < 0)
            {
                throw new Exception("Minlength value should greater than 0");
            }

            return _api_internal.bincount(x: x, weights: weights, minlength: minlength);
        }

        public static ndarray atleast_1d(NDArrayList arys)
        {
            return _api_internal.atleast_1d(arys: arys, multi: true);
        }

        public static ndarray atleast_2d(NDArrayList arys)
        {
            return _api_internal.atleast_2d(arys: arys, multi: true);
        }

        public static ndarray atleast_3d(NDArrayList arys)
        {
            return _api_internal.atleast_3d(arys: arys, multi: true);
        }

        public static ndarray pad(ndarray x, int[] pad_width = null, string mode = "constant", float constant_values = 0, string reflect_type = "even")
        {
            if (mode == "linear_ramp")
            {
                throw new Exception("mode {'linear_ramp'} is not supported.");
            }

            if (mode == "wrap")
            {
                throw new Exception("mode {'wrap'} is not supported.");
            }

            if (mode == "median")
            {
                throw new Exception("mode {'median'} is not supported.");
            }

            if (mode == "mean")
            {
                throw new Exception("mode {'mean'} is not supported.");
            }

            if (mode == "empty")
            {
                throw new Exception("mode {'empty'} is not supported.");
            }
            
            if (mode == "constant")
            {
                return _api_internal.pad(x, pad_width, "constant", constant_values, "even");
            }
            else if (mode == "symmetric")
            {
                if (reflect_type != "even" && reflect_type != null)
                {
                    throw new Exception($"unsupported reflect_type '{reflect_type}'");
                }

                return _api_internal.pad(x, pad_width, "symmetric", 0, "even");
            }
            else if (mode == "edge")
            {
                return _api_internal.pad(x, pad_width, "edge", 0, "even");
            }
            else if (mode == "reflect")
            {
                if (reflect_type != "even" && reflect_type != null)
                {
                    throw new Exception($"unsupported reflect_type '{reflect_type}'");
                }

                return _api_internal.pad(x, pad_width, "reflect", 0, "even");
            }
            else if (mode == "maximum")
            {
                return _api_internal.pad(x, pad_width, "maximum", 0, "even");
            }
            else if (mode == "minimum")
            {
                return _api_internal.pad(x, pad_width, "minimum", 0, "even");
            }

            return _api_internal.pad(x, pad_width, "constant", 0, "even");
        }

        public static ndarray prod(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false, float? initial = null)
        {
            @out = _api_internal.prod(a: a, axis: axis, dtype: dtype, keepdims: keepdims, initial: initial);
            return @out;
        }

        public static ndarray dot(ndarray a, ndarray b, ndarray @out = null)
        {
            @out = _api_internal.dot(a: a, b: b);
            return @out;
        }

        public static ndarray cumsum(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null)
        {
            return _api_internal.cumsum(a: a, axis: axis, dtype: dtype, @out: @out);
        }

        public static ndarray reshape(ndarray a, Shape newshape, bool reverse = false, string order = "C")
        {
            return _api_internal.reshape(a: a, newshape: newshape, reverse: reverse, order: order);
        }

        public static ndarray moveaxis(ndarray a, int source, int destination)
        {
            return _api_internal.moveaxis(a: a, source: source, destination: destination);
        }

        public static ndarray moveaxis(ndarray a, int[] source, int[] destination)
        {
            return _api_internal.moveaxis(a: a, source: source, destination: destination);
        }

        public static ndarray copy(ndarray a)
        {
            return _api_internal.copy(a: a);
        }

        public static ndarray rollaxis(ndarray a, int axis, int start = 0)
        {
            return _api_internal.rollaxis(a: a, axis: axis, start: start);
        }

        public static ndarray diag(ndarray v, int k = 0)
        {
            return _api_internal.diag(v: v, k: k);
        }

        public static ndarray diagflat(ndarray v, int k = 0)
        {
            return _api_internal.diagflat(v: v, k: k);
        }

        public static ndarray diagonal(ndarray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return _api_internal.diagonal(a: a, offset: offset, axis1: axis1, axis2: axis2);
        }

        public static ndarray sum(ndarray a, int? axis = null, DType dtype = null, ndarray @out = null, bool keepdims = false, float? initial = null)
        {
            @out = _api_internal.sum(a: a, axis: axis, dtype: dtype, keepdims: keepdims, initial: initial);
            return @out;
        }

        public static ndarray meshgrid(NDArrayList xi, string indexing = "xy", bool sparse = false, bool copy = true)
        {
            throw new NotImplementedException();
        }
    }
}
