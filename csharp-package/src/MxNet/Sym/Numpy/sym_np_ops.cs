using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MxNet.Sym.Numpy
{
    internal class sym_np_ops
    {
        public static Random random = new Random();
        public static Linalg linalg = new Linalg();

        private static dynamic _api_internal = new _api_internals();

        public static _Symbol zeros(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return _api_internal.zeros(shape: shape, ctx: ctx, dtype: dtype);
        }

        public static _Symbol ones(Shape shape, DType dtype = null, string order = "C", Context ctx = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = np.Float32;
            if (ctx == null) ctx = Context.CurrentContext;

            return _api_internal.ones(shape: shape, ctx: ctx, dtype: dtype);
        }

        public static _Symbol broadcast_to(_Symbol array, Shape shape)
        {
            return _api_internal.broadcast_to(array: array, shape: shape);
        }

        public static _Symbol full(Shape shape, double fill_value, DType dtype = null, string order = "C", Context ctx = null, _Symbol @out = null)
        {
            if (order != "C") throw new NotSupportedException("order");

            @out = _api_internal.full(shape: shape, value: fill_value, dtype: dtype, ctx: ctx);
            return @out;
        }

        public static _Symbol zero_like(_Symbol prototype, DType dtype = null, string order = "C", _Symbol @out = null)
        {
            @out = full_like(prototype, 0, dtype, order, @out: @out);
            return @out;
        }

        public static _Symbol all(_Symbol a)
        {
            return ((_Symbol)_api_internal.all(a: a));
        }

        public static _Symbol all(_Symbol a, int axis, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.all(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol any(_Symbol a)
        {
            return ((_Symbol)_api_internal.any(a: a));
        }

        public static _Symbol any(_Symbol a, int axis, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.any(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol identity(int n, DType dtype = null, Context ctx = null)
        {
            return _api_internal.identity(shape: new Shape(n, n), dtype: dtype, ctx: ctx);
        }

        public static _Symbol take(_Symbol a, _Symbol indices, int? axis = null, string mode = "raise", _Symbol @out = null)
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

        public static _Symbol unique(_Symbol ar, int? axis = null)
        {
            var list = (SymbolList)_api_internal.unique(ar: ar, axis: axis, multi: true);
            return list[0];
        }

        public static (_Symbol, _Symbol, _Symbol, _Symbol) unique(_Symbol ar, bool return_index = false, bool return_inverse = false, bool return_counts = false, int? axis = null)
        {
            var ret = (SymbolList)_api_internal.unique(ar: ar, return_index: return_index, return_inverse: return_inverse, return_counts: return_counts, axis: axis, multi: true);
            return (ret[0], ret[1], ret[2], ret[3]);
        }

        public static _Symbol add(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.add(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol add(_Symbol x1, float x2, _Symbol @out = null)
        {
            return add(x1, full_like(x1, x2));
        }

        public static _Symbol add(float x1, _Symbol x2, _Symbol @out = null)
        {
            return add(full_like(x2, x1), x2);
        }

        public static _Symbol subtract(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.subtract(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol subtract(_Symbol x1, float x2, _Symbol @out = null)
        {
            return subtract(x1, full_like(x1, x2));
        }

        public static _Symbol subtract(float x1, _Symbol x2, _Symbol @out = null)
        {
            return subtract(full_like(x2, x1), x2);
        }

        public static _Symbol multiply(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.multiply(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol multiply(_Symbol x1, float x2, _Symbol @out = null)
        {
            return multiply(x1, full_like(x1, x2));
        }

        public static _Symbol multiply(float x1, _Symbol x2, _Symbol @out = null)
        {
            return multiply(full_like(x2, x1), x2);
        }

        public static _Symbol divide(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.divide(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol divide(_Symbol x1, float x2, _Symbol @out = null)
        {
            return divide(x1, full_like(x1, x2));
        }

        public static _Symbol divide(float x1, _Symbol x2, _Symbol @out = null)
        {
            return divide(full_like(x2, x1), x2);
        }

        public static _Symbol true_divide(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.true_divide(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol true_divide(_Symbol x1, float x2, _Symbol @out = null)
        {
            return true_divide(x1, full_like(x1, x2));
        }

        public static _Symbol true_divide(float x1, _Symbol x2, _Symbol @out = null)
        {
            return true_divide(full_like(x2, x1), x2);
        }

        public static _Symbol mod(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.mod(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol mod(_Symbol x1, float x2, _Symbol @out = null)
        {
            return mod(x1, full_like(x1, x2));
        }

        public static _Symbol mod(float x1, _Symbol x2, _Symbol @out = null)
        {
            return mod(full_like(x2, x1), x2);
        }

        public static _Symbol fmod(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.fmod(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol matmul(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.matmul(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol remainder(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.remainder(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol power(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.power(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol power(_Symbol x1, float x2, _Symbol @out = null)
        {
            return power(x1, full_like(x1, x2));
        }

        public static _Symbol gcd(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.gcd(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol gcd(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.gcd(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol gcd(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.gcd(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol lcm(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.lcm(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol lcm(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.lcm(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol lcm(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.lcm(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol sin(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.sin(x: x);
            return @out;
        }

        public static _Symbol cos(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.cos(x: x);
            return @out;
        }

        public static _Symbol sinh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.sinh(x: x);
            return @out;
        }

        public static _Symbol cosh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.cosh(x: x);
            return @out;
        }

        public static _Symbol tanh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.tanh(x: x);
            return @out;
        }

        public static _Symbol log10(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.log10(x: x);
            return @out;
        }

        public static _Symbol sqrt(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.sqrt(x: x);
            return @out;
        }

        public static _Symbol cbrt(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.cbrt(x: x);
            return @out;
        }

        public static _Symbol abs(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.abs(x: x);
            return @out;
        }

        public static _Symbol fabs(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.fabs(x: x);
            return @out;
        }

        public static _Symbol absolute(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.absolute(x: x);
            return @out;
        }

        public static _Symbol exp(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.exp(x: x);
            return @out;
        }

        public static _Symbol expm1(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.expm1(x: x);
            return @out;
        }

        public static _Symbol arcsin(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arcsin(x: x);
            return @out;
        }

        public static _Symbol arccos(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arccos(x: x);
            return @out;
        }

        public static _Symbol arctan(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arctan(x: x);
            return @out;
        }

        public static _Symbol sign(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.sign(x: x);
            return @out;
        }

        public static _Symbol log(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.log(x: x);
            return @out;
        }

        public static _Symbol rint(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.rint(x: x);
            return @out;
        }

        public static _Symbol log2(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.log2(x: x);
            return @out;
        }

        public static _Symbol log1p(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.log1p(x: x);
            return @out;
        }

        public static _Symbol degrees(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.degrees(x: x);
            return @out;
        }

        public static _Symbol rad2deg(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.rad2deg(x: x);
            return @out;
        }

        public static _Symbol radians(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.radians(x: x);
            return @out;
        }

        public static _Symbol deg2rad(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.deg2rad(x: x);
            return @out;
        }

        public static _Symbol reciprocal(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.reciprocal(x: x);
            return @out;
        }

        public static _Symbol square(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.square(x: x);
            return @out;
        }

        public static _Symbol negative(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.negative(x: x);
            return @out;
        }

        public static _Symbol fix(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.fix(x: x);
            return @out;
        }

        public static _Symbol tan(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.tan(x: x);
            return @out;
        }

        public static _Symbol ceil(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.ceil(x: x);
            return @out;
        }

        public static _Symbol floor(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.floor(x: x);
            return @out;
        }

        public static _Symbol invert(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.invert(x: x);
            return @out;
        }

        public static _Symbol bitwise_not(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.bitwise_not(x: x);
            return @out;
        }

        public static _Symbol trunc(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.trunc(x: x);
            return @out;
        }

        public static _Symbol logical_not(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.logical_not(x: x);
            return @out;
        }

        public static _Symbol arcsinh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arcsinh(x: x);
            return @out;
        }

        public static _Symbol arccosh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arccosh(x: x);
            return @out;
        }

        public static _Symbol arctanh(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.arctanh(x: x);
            return @out;
        }

        public static _Symbol argsort(_Symbol x, int axis = -1, string kind = null, string order = null)
        {
            if (order != null)
                throw new NotSupportedException("order is not supported");

            return _api_internal.argsort(a: x, axis: axis, is_ascend: true, dtype: np.Int64);
        }

        public static _Symbol sort(_Symbol x, int axis = -1, string kind = null, string order = null)
        {
            if (order != null)
                throw new NotSupportedException("order is not supported");

            return _api_internal.sort(a: x, axis: axis, is_ascend: true);
        }

        public static _Symbol tensordot(_Symbol a, _Symbol b, Shape axes = null)
        {
            if (axes == null) axes = new Shape(2);

            return _api_internal.tensordot(a: a, b: b, axes: axes);
        }

        public static _Symbol histogram(_Symbol a, int bins = 10, (float, float)? range = null, bool? normed = null, _Symbol weights = null, bool? density = null)
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

        public static _Symbol histogram(_Symbol a, _Symbol bins, (float, float)? range = null, bool? normed = null, _Symbol weights = null, bool? density = null)
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

        public static _Symbol eye(int N, int? M = null, int k = 0, Context ctx = null, DType dtype = null)
        {
            if (ctx == null) ctx = Context.CurrentContext;
            if (dtype == null) dtype = np.Float32;

            return _api_internal.eye(N, M, k, ctx, dtype);
        }

        public static (_Symbol, float?) linspace(float start, float stop, int num = 50, bool endpoint = true, bool retstep = false, DType dtype = null, int axis = 0, Context ctx = null)
        {
            var step = (stop - start) / (num - 1);
            if (retstep)
                return (_api_internal.linspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype), step);
            else
                return (_api_internal.linspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype), null);
        }

        public static _Symbol logspace(float start, float stop, int num = 50, bool endpoint = true, bool retstep = false, DType dtype = null, int axis = 0, Context ctx = null)
        {
            return _api_internal.logspace(start: start, stop: stop, num: num, endpoint: endpoint, ctx: ctx, dtype: dtype);
        }

        public static _Symbol expand_dims(_Symbol a, int axis)
        {
            return _api_internal.expand_dims(a: a, axis: axis);
        }

        public static _Symbol tile(_Symbol a, params int[] reps)
        {
            return _api_internal.tile(A: a, reps: new Shape(reps));
        }

        public static _Symbol trace(_Symbol a, int offset = 0, int axis1 = 0, int axis2 = 1, _Symbol @out = null)
        {
            @out = _api_internal.trace(a: a, offset: offset, axis1: axis1, axis2: axis2);
            return @out;
        }

        public static _Symbol transpose(_Symbol a, params int[] axes)
        {
            return _api_internal.transpose(a: a, axes: new Shape(axes));
        }

        public static _Symbol repeat(_Symbol a, int repeats, int? axis = null)
        {
            if (axis != null)
            {
                var tmp = swapaxes(a, 0, axis.Value);
                var res = _api_internal.repeats(a: tmp, repeats: repeats, axis: 0);
                return swapaxes(res, 0, axis.Value);
            }

            return _api_internal.repeats(a: a, repeats: repeats, axis: axis);
        }

        public static _Symbol tril(_Symbol m, int k = 0)
        {
            return _api_internal.tril(m: m, k: k);
        }

        public static _Symbol tri(int N, int? M = null, int k = 0, DType dtype = null, Context ctx = null)
        {
            return _api_internal.tri(N: N, M: M, k: k, dtype: dtype, ctx: ctx);
        }

        public static _Symbol triu_indices(int n, int k = 0, int? m = null, Context ctx = null)
        {
            return nonzero(negative(tri(N: n, M: m, k: k - 1, dtype: np.Bool, ctx: ctx)));
        }

        public static SymbolList tril_indices(int n, int k = 0, int? m = null)
        {
            if (m == null)
            {
                m = n;
            }

            return _api_internal.tril_indices(n: n, k: k, m: m, multi: true);
        }

        public static _Symbol triu(int m, int k = 0)
        {
            return _api_internal.triu(m: m, k: k);
        }

        public static _Symbol arange(int start, int? stop = null, int step = 1, DType dtype = null, Context ctx = null)
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

        public static SymbolList split(_Symbol ary, int[] indices_or_sections, int axis = 0)
        {
            return (SymbolList)_api_internal.split(ary: ary, indices_or_sections: indices_or_sections, axis: axis, multi: true);
        }

        public static SymbolList array_split(_Symbol ary, int[] indices_or_sections, int axis = 0)
        {
            return (SymbolList)_api_internal.array_split(ary: ary, indices_or_sections: indices_or_sections, axis: axis, multi: true);
        }

        public static SymbolList vsplit(_Symbol ary, int[] indices_or_sections)
        {
            return (SymbolList)_api_internal.vsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static SymbolList hsplit(_Symbol ary, int[] indices_or_sections)
        {
            return (SymbolList)_api_internal.hsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static SymbolList dsplit(_Symbol ary, int[] indices_or_sections)
        {
            return (SymbolList)_api_internal.dsplit(ary: ary, indices_or_sections: indices_or_sections, multi: true);
        }

        public static _Symbol concatenate(SymbolList seq, int axis = 0, _Symbol @out = null)
        {
            @out = _api_internal.concatenate(axis: axis, seq: seq);
            return @out;
        }

        public static _Symbol append(_Symbol arr, _Symbol values, int? axis = null)
        {
            return _api_internal.concatenate(arr: arr, values: values, axis: axis);
        }

        public static _Symbol stack(SymbolList arrays, int axis = 0, _Symbol @out = null)
        {
            @out = _api_internal.stack(axis: axis, arrays: arrays);
            return @out;
        }

        public static _Symbol vstack(SymbolList arrays, _Symbol @out = null)
        {
            @out = _api_internal.vstack(arrays: arrays);
            return @out;
        }

        public static _Symbol row_stack(SymbolList arrays)
        {
            var @out = _api_internal.vstack(arrays: arrays);
            return @out;
        }

        public static _Symbol column_stack(SymbolList tup)
        {
            return _api_internal.column_stack(tup: tup);
        }

        public static _Symbol hstack(SymbolList arrays)
        {
            return _api_internal.hstack(arrays: arrays);
        }

        public static _Symbol dstack(SymbolList arrays)
        {
            return _api_internal.dstack(arrays: arrays);
        }

        public static _Symbol maximum(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.maximum(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol maximum(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.maximum(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol maximum(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.maximum(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol fmax(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.fmax(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol fmax(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.fmax(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol fmax(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.fmax(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol minimum(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.minimum(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol minimum(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.minimum(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol minimum(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.minimum(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol fmin(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.fmin(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol fmin(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.fmin(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol fmin(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.fmin(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol max(_Symbol a, int? axis = null, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.max(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol min(_Symbol a, int? axis = null, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.min(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol swapaxes(_Symbol a, int axis1, int axis2)
        {
            return _api_internal.swapaxes(a: a, dim1: axis1, dim2: axis2);
        }

        public static _Symbol clip(_Symbol a, float? a_min, float? a_max, _Symbol @out = null)
        {
            if (a_min == null && a_max == null)
            {
                throw new Exception("array_clip: must set either max or min");
            }

            @out = _api_internal.clip(a: a, a_min: a_min, a_max: a_max);
            return @out;
        }

        public static _Symbol argmax(_Symbol a, int? axis = null, _Symbol @out = null)
        {
            @out = _api_internal.argmax(a: a, axis: axis);
            return @out;
        }

        public static _Symbol argmin(_Symbol a, int? axis = null, _Symbol @out = null)
        {
            @out = _api_internal.argmin(a: a, axis: axis);
            return @out;
        }

        public static _Symbol amax(_Symbol a, int? axis = null, bool keepdims = false, _Symbol @out = null)
        {
            @out = _api_internal.amax(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol amin(_Symbol a, int? axis = null, bool keepdims = false, _Symbol @out = null)
        {
            @out = _api_internal.amin(a: a, axis: axis, keepdims: keepdims);
            return @out;
        }

        public static _Symbol average(_Symbol a, int? axis = null, _Symbol weights = null, bool returned = false, _Symbol @out = null)
        {
            @out = _api_internal.average(a: a, weights: weights, axis: axis, returned: returned);
            return @out;
        }

        public static _Symbol mean(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.mean(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static _Symbol std(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.std(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static _Symbol delete(_Symbol arr, int obj, int? axis = null)
        {
            return _api_internal.delete(arr: arr, obj: obj, axis: axis);
        }

        public static _Symbol delete(_Symbol arr, Slice obj, int? axis = null)
        {
            return _api_internal.delete(arr, start: obj.Begin, stop: obj.End, step: obj.Step, axis: axis);
        }

        public static _Symbol delete(_Symbol arr, _Symbol obj, int? axis = null)
        {
            return _api_internal.delete(arr: arr, obj: obj, axis: axis);
        }

        public static _Symbol var(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null, bool keepdims = false)
        {
            @out = _api_internal.var(a: a, axis: axis, dtype: dtype, keepdims: keepdims);
            return @out;
        }

        public static _Symbol indices(Shape dimensions, DType dtype = null, Context ctx = null)
        {
            return _api_internal.indices(dimensions: dimensions, dtype: dtype, ctx: ctx);
        }

        public static _Symbol copysign(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.copysign(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol copysign(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.copysign(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol copysign(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.copysign(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol ravel(_Symbol x, string order = "C")
        {
            if (order == "F")
            {
                throw new NotSupportedException($"order {order} is not supported");
            }

            return reshape(x, -1);
        }

        public static SymbolList unravel_index(_Symbol indices, Shape shape, string order = "x")
        {
            if (order == "F")
            {
                throw new NotSupportedException($"order {order} is not supported");
            }

            return _api_internal.unravel_index_fallback(indices: indices, shape: shape, multi: true);
        }

        public static SymbolList flatnonzero(_Symbol a)
        {
            return nonzero(ravel(a));
        }

        public static SymbolList diag_indices_from(_Symbol arr)
        {
            return _api_internal.diag_indices_from(arr: arr, multi: true);
        }

        public static _Symbol hanning(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.hanning(M: M, dtype: dtype, ctx: ctx);
        }

        public static _Symbol hamming(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.hamming(M: M, dtype: dtype, ctx: ctx);
        }

        public static _Symbol blackman(int M, DType dtype = null, Context ctx = null)
        {
            return _api_internal.blackman(M: M, dtype: dtype, ctx: ctx);
        }

        public static _Symbol flip(_Symbol m, int? axis = null, _Symbol @out = null)
        {
            @out = _api_internal.flip(m: m, axis: axis);
            return @out;
        }

        public static _Symbol flipud(_Symbol m)
        {
            return flip(m, 0);
        }

        public static _Symbol fliplr(_Symbol m)
        {
            return flip(m, 1);
        }

        public static _Symbol around(_Symbol x, int decimals = 0, _Symbol @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static _Symbol round(_Symbol x, int decimals = 0, _Symbol @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static _Symbol round_(_Symbol x, int decimals = 0, _Symbol @out = null)
        {
            return _api_internal.around(x: x, decimals: decimals);
        }

        public static _Symbol arctan2(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: x2);
        }

        public static _Symbol arctan2(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol arctan2(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.arctan2(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol hypot(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.hypot(x1: x1, x2: x2);
        }

        public static _Symbol hypot(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.arctan2(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol hypot(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.hypot(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol bitwise_and(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_and(x1: x1, x2: x2);
        }

        public static _Symbol bitwise_and(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_and(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol bitwise_and(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_and(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol bitwise_xor(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_xor(x1: x1, x2: x2);
        }

        public static _Symbol bitwise_xor(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_xor(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol bitwise_xor(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_xor(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol bitwise_or(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_or(x1: x1, x2: x2);
        }

        public static _Symbol bitwise_or(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_or(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol bitwise_or(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.bitwise_or(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol ldexp(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.ldexp(x1: x1, x2: x2);
        }

        public static _Symbol ldexp(_Symbol x1, float x2, _Symbol @out = null)
        {
            return _api_internal.ldexp(x1: x1, x2: full_like(x1, x2));
        }

        public static _Symbol ldexp(float x1, _Symbol x2, _Symbol @out = null)
        {
            return _api_internal.ldexp(x1: full_like(x2, x1), x2: x2);
        }

        public static _Symbol vdot(_Symbol a, _Symbol b)
        {
            return tensordot(ravel(a), ravel(b), 1);
        }

        public static _Symbol inner(_Symbol a, _Symbol b)
        {
            return tensordot(a, b, new Shape(-1, -1));
        }

        public static _Symbol outer(_Symbol a, _Symbol b)
        {
            return tensordot(reshape(a, new Shape(-1)), reshape(b, new Shape(-1)), new Shape(0));
        }

        public static _Symbol cross(_Symbol a, _Symbol b, int axisa = -1, int axisb = -1, int axisc = -1, int? axis = null)
        {
            if (axis != null)
            {
                axisa = axis.Value;
                axisb = axis.Value;
                axisc = axis.Value;
            }

            return _api_internal.cross(a: a, b: b, axisa: axisa, axisb: axisb, axisc: axisc);
        }

        public static _Symbol kron(_Symbol a, _Symbol b)
        {
            return _api_internal.kron(a: a, b: b);
        }

        public static _Symbol equal(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.equal(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol equal(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol equal(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol not_equal(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.not_equal(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol not_equal(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.not_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol not_equal(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.not_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol greater(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.greater(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol greater(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.greater(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol greater(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.greater(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol less(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.less(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol less(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.less(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol less(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.less(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol logical_and(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_and(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol logical_and(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_and(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol logical_and(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_and(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol logical_or(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_or(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol logical_or(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_or(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol logical_or(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_or(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol logical_xor(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_xor(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol logical_xor(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_xor(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol logical_xor(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.logical_xor(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol greater_equal(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.greater_equal(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol greater_equal(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.greater_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol greater_equal(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.greater_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol less_equal(_Symbol x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.less_equal(x1: x1, x2: x2);
            return @out;
        }

        public static _Symbol less_equal(_Symbol x1, float x2, _Symbol @out = null)
        {
            @out = _api_internal.less_equal(x1: x1, x2: full_like(x1, x2));
            return @out;
        }

        public static _Symbol less_equal(float x1, _Symbol x2, _Symbol @out = null)
        {
            @out = _api_internal.less_equal(x1: full_like(x2, x1), x2: x2);
            return @out;
        }

        public static _Symbol roll(_Symbol a, int shift, int? axis = null)
        {
            return _api_internal.roll(a: a, shift: shift, axis: axis);
        }

        public static _Symbol roll(_Symbol a, int[] shift, int? axis = null)
        {
            return _api_internal.roll(a: a, shift: new Shape(shift), axis: axis);
        }

        public static _Symbol rot90(_Symbol m, int k = 1, params int[] axes)
        {
            return _api_internal.rot90(m: m, k: k, axes: new Shape(axes));
        }

        public static _Symbol einsum(string subscripts, SymbolList operands, _Symbol @out = null, bool optimize = false)
        {
            return _api_internal.einsum(subscripts: subscripts, optimize_arg: Convert.ToInt32(optimize), operands: operands);
        }

        public static _Symbol insert(_Symbol arr, int obj, _Symbol values, int? axis = null)
        {
            return _api_internal.insert_scalar(arr: arr, values: values, obj: obj, axis: axis);
        }

        public static _Symbol insert(_Symbol arr, Slice obj, _Symbol values, int? axis = null)
        {
            return _api_internal.insert_slice(arr: arr, values: values, start: obj.Begin, stop: obj.End, step: obj.Step, axis: axis);
        }

        public static _Symbol insert(_Symbol arr, _Symbol obj, _Symbol values, int? axis = null)
        {
            return _api_internal.insert_tensor(arr: arr, obj: obj, values: values, axis: axis);
        }

        public static SymbolList nonzero(_Symbol a)
        {
            SymbolList @out = _api_internal.nonzero(a: a, multi: true);
            @out = @out.Select(x => transpose(x)).ToList();
            return @out;
        }

        public static _Symbol percentile(_Symbol a, _Symbol q, int? axis = null, _Symbol @out = null, bool? overwrite_input = null, string interpolation = "linear", bool keepdims = false)
        {
            if (overwrite_input != null)
            {
                throw new NotSupportedException("overwrite_input is not supported yet");
            }

            @out = _api_internal.percentile(a: a, q: q, axis: axis, interpolation: interpolation, keepdims: keepdims);
            return @out;
        }

        public static _Symbol quantile(_Symbol a, _Symbol q, int? axis = null, _Symbol @out = null, bool? overwrite_input = null, string interpolation = "linear", bool keepdims = false)
        {
            if (overwrite_input != null)
            {
                throw new NotSupportedException("overwrite_input is not supported yet");
            }

            @out = _api_internal.percentile(a: a, q: q * 100, axis: axis, interpolation: interpolation, keepdims: keepdims);
            return @out;
        }

        public static _Symbol diff(_Symbol a, int n = 1, int axis = -1, _Symbol prepend = null, _Symbol append = null)
        {
            if (prepend != null || append != null)
            {
                throw new NotSupportedException("prepend and append options are not supported yet");
            }

            return _api_internal.diff(a: a, n: n, axis: axis);
        }

        public static _Symbol ediff1d(_Symbol ary, _Symbol to_end = null, _Symbol to_begin = null)
        {
            return _api_internal.ediff1d(ary: ary, to_end: to_end, to_begin: to_begin);
        }

        public static _Symbol resize(_Symbol a, Shape new_shape)
        {
            return _api_internal.resize_fallback(a: a, new_shape: new_shape);
        }

        public static _Symbol interp(_Symbol x, float[] xp, float[] fp, float? left = null, float? right = null, float? period = null)
        {
            return _api_internal.interp(xp: xp, fp: fp, x: x, left: left, right: right, period: period);
        }

        public static _Symbol full_like(_Symbol a, double fill_value, DType dtype = null, string order = "C", Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.full_like(a: a, fill_value: fill_value, dtype: dtype, ctx: ctx);
            return @out;
        }

        public static _Symbol zeros_like(_Symbol a, DType dtype = null, string order = "C", Context ctx = null, _Symbol @out = null)
        {
            return full_like(a, 0, dtype, order, ctx, @out);
        }

        public static _Symbol ones_like(_Symbol a, DType dtype = null, string order = "C", Context ctx = null, _Symbol @out = null)
        {
            return full_like(a, 1, dtype, order, ctx, @out);
        }

        public static _Symbol fill_diagonal(_Symbol a, float[] val, bool wrap = false)
        {
            return _api_internal.fill_diagonal(a, val, wrap, a);
        }

        public static _Symbol fill_diagonal(_Symbol a, float val, bool wrap = false)
        {
            return fill_diagonal(a, new float[] { val }, wrap);
        }

        public static _Symbol squeeze(_Symbol a, int? axis = null)
        {
            return _api_internal.squeeze(x: a, axis: axis);
        }

        public static _Symbol isnan(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.isnan(x: x);
            return @out;
        }

        public static _Symbol isinf(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.isinf(x: x);
            return @out;
        }

        public static _Symbol isposinf(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.isposinf(x: x);
            return @out;
        }

        public static _Symbol isneginf(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.isneginf(x: x);
            return @out;
        }

        public static _Symbol isfinite(_Symbol x, _Symbol @out = null)
        {
            @out = _api_internal.isfinite(x: x);
            return @out;
        }

        public static _Symbol where(_Symbol condition, _Symbol x = null, _Symbol y = null)
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

        public static _Symbol polyval(_Symbol p, _Symbol x)
        {
            return _api_internal.polyval(p: p, x: x);
        }

        public static _Symbol bincount(_Symbol x, _Symbol weights = null, int minlength = 0)
        {
            if (minlength < 0)
            {
                throw new Exception("Minlength value should greater than 0");
            }

            return _api_internal.bincount(x: x, weights: weights, minlength: minlength);
        }

        public static _Symbol atleast_1d(SymbolList arys)
        {
            return _api_internal.atleast_1d(arys: arys, multi: true);
        }

        public static _Symbol atleast_2d(SymbolList arys)
        {
            return _api_internal.atleast_2d(arys: arys, multi: true);
        }

        public static _Symbol atleast_3d(SymbolList arys)
        {
            return _api_internal.atleast_3d(arys: arys, multi: true);
        }

        public static _Symbol pad(_Symbol x, int[] pad_width = null, string mode = "constant", float constant_values = 0, string reflect_type = "even")
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

        public static _Symbol prod(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null, bool keepdims = false, float? initial = null)
        {
            @out = _api_internal.prod(a: a, axis: axis, dtype: dtype, keepdims: keepdims, initial: initial);
            return @out;
        }

        public static _Symbol dot(_Symbol a, _Symbol b, _Symbol @out = null)
        {
            @out = _api_internal.dot(a: a, b: b);
            return @out;
        }

        public static _Symbol cumsum(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null)
        {
            return _api_internal.cumsum(a: a, axis: axis, dtype: dtype, @out: @out);
        }

        public static _Symbol reshape(_Symbol a, Shape newshape, bool reverse = false, string order = "C")
        {
            return _api_internal.reshape(a: a, newshape: newshape, reverse: reverse, order: order);
        }

        public static _Symbol moveaxis(_Symbol a, int source, int destination)
        {
            return _api_internal.moveaxis(a: a, source: source, destination: destination);
        }

        public static _Symbol moveaxis(_Symbol a, int[] source, int[] destination)
        {
            return _api_internal.moveaxis(a: a, source: source, destination: destination);
        }

        public static _Symbol copy(_Symbol a)
        {
            return _api_internal.copy(a: a);
        }

        public static _Symbol rollaxis(_Symbol a, int axis, int start = 0)
        {
            return _api_internal.rollaxis(a: a, axis: axis, start: start);
        }

        public static _Symbol diag(_Symbol v, int k = 0)
        {
            return _api_internal.diag(v: v, k: k);
        }

        public static _Symbol diagflat(_Symbol v, int k = 0)
        {
            return _api_internal.diagflat(v: v, k: k);
        }

        public static _Symbol diagonal(_Symbol a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            return _api_internal.diagonal(a: a, offset: offset, axis1: axis1, axis2: axis2);
        }

        public static _Symbol sum(_Symbol a, int? axis = null, DType dtype = null, _Symbol @out = null, bool keepdims = false, float? initial = null)
        {
            @out = _api_internal.sum(a: a, axis: axis, dtype: dtype, keepdims: keepdims, initial: initial);
            return @out;
        }

        public static _Symbol meshgrid(_Symbol[] xi, string indexing = "xy", bool sparse = false, bool copy = true)
        {
            throw new NotImplementedException();
        }
    }
}
