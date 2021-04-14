/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using System.Collections.Generic;
using System.Diagnostics;

namespace MxNet
{
    public partial class NDArray
    {
        private static readonly List<string> PickModeConvert = new List<string> {"clip", "wrap"};

        private static readonly List<string> NormOutDtypeConvert = new List<string>
            {"float16", "float32", "float64", "int32", "int64", "int8"};

        private static readonly List<string>
            CastStorageStypeConvert = new List<string> {"default", "row_sparse", "csr"};

        private static readonly List<string> TopkRetTypConvert = new List<string> {"both", "indices", "mask", "value"};

        /// <summary>
        ///     <para>Returns indices of the maximum values along an axis.</para>
        ///     <para> </para>
        ///     <para>In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrence</para>
        ///     <para>are returned.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.,  1.,  2.],</para>
        ///     <para>       [ 3.,  4.,  5.]]</para>
        ///     <para> </para>
        ///     <para>  // argmax along axis 0</para>
        ///     <para>  argmax(x, axis=0) = [ 1.,  1.,  1.]</para>
        ///     <para> </para>
        ///     <para>  // argmax along axis 1</para>
        ///     <para>  argmax(x, axis=1) = [ 2.,  2.]</para>
        ///     <para> </para>
        ///     <para>  // argmax along axis 1 keeping same dims as an input array</para>
        ///     <para>  argmax(x, axis=1, keepdims=True) = [[ 2.],</para>
        ///     <para>                                      [ 2.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L52</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis along which to perform the reduction. Negative values means indexing from right to left.
        ///     ``Requires axis to be set as int, because global reduction is not supported yet.``
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Argmax(int? axis = null, bool keepdims = false)
        {
            return new Operator("argmax")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns indices of the minimum values along an axis.</para>
        ///     <para> </para>
        ///     <para>In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrence</para>
        ///     <para>are returned.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.,  1.,  2.],</para>
        ///     <para>       [ 3.,  4.,  5.]]</para>
        ///     <para> </para>
        ///     <para>  // argmin along axis 0</para>
        ///     <para>  argmin(x, axis=0) = [ 0.,  0.,  0.]</para>
        ///     <para> </para>
        ///     <para>  // argmin along axis 1</para>
        ///     <para>  argmin(x, axis=1) = [ 0.,  0.]</para>
        ///     <para> </para>
        ///     <para>  // argmin along axis 1 keeping same dims as an input array</para>
        ///     <para>  argmin(x, axis=1, keepdims=True) = [[ 0.],</para>
        ///     <para>                                      [ 0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L77</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis along which to perform the reduction. Negative values means indexing from right to left.
        ///     ``Requires axis to be set as int, because global reduction is not supported yet.``
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Argmin(int? axis = null, bool keepdims = false)
        {
            return new Operator("argmin")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns argmax indices of each channel from the input array.</para>
        ///     <para> </para>
        ///     <para>The result will be an NDArray of shape (num_channel,).</para>
        ///     <para> </para>
        ///     <para>In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence</para>
        ///     <para>are returned.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.,  1.,  2.],</para>
        ///     <para>       [ 3.,  4.,  5.]]</para>
        ///     <para> </para>
        ///     <para>  argmax_channel(x) = [ 2.,  2.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L97</para>
        /// </summary>
        /// <param name="data">The input array</param>
        /// <returns>returns new NDArray</returns>
        public NDArray ArgmaxChannel()
        {
            return new Operator("argmax_channel")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Picks elements from an input array according to the input indices along the given axis.</para>
        ///     <para> </para>
        ///     <para>Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be</para>
        ///     <para>an output array of shape ``(i0,)`` with::</para>
        ///     <para> </para>
        ///     <para>  output[i] = input[i, indices[i]]</para>
        ///     <para> </para>
        ///     <para>By default, if any index mentioned is too large, it is replaced by the index that addresses</para>
        ///     <para>the last element along an axis (the `clip` mode).</para>
        ///     <para> </para>
        ///     <para>This function supports n-dimensional input and (n-1)-dimensional indices arrays.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.,  2.],</para>
        ///     <para>       [ 3.,  4.],</para>
        ///     <para>       [ 5.,  6.]]</para>
        ///     <para> </para>
        ///     <para>  // picks elements with specified indices along axis 0</para>
        ///     <para>  pick(x, y=[0,1], 0) = [ 1.,  4.]</para>
        ///     <para> </para>
        ///     <para>  // picks elements with specified indices along axis 1</para>
        ///     <para>  pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]</para>
        ///     <para> </para>
        ///     <para>  y = [[ 1.],</para>
        ///     <para>       [ 0.],</para>
        ///     <para>       [ 2.]]</para>
        ///     <para> </para>
        ///     <para>  // picks elements with specified indices along axis 1 using 'wrap' mode</para>
        ///     <para>  // to place indicies that would normally be out of bounds</para>
        ///     <para>  pick(x, y=[2,-1,-2], 1, mode='wrap') = [ 1.,  4.,  5.]</para>
        ///     <para> </para>
        ///     <para>  y = [[ 1.],</para>
        ///     <para>       [ 0.],</para>
        ///     <para>       [ 2.]]</para>
        ///     <para> </para>
        ///     <para>  // picks elements with specified indices along axis 1 and dims are maintained</para>
        ///     <para>  pick(x,y, 1, keepdims=True) = [[ 2.],</para>
        ///     <para>                                 [ 3.],</para>
        ///     <para>                                 [ 6.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_index.cc:L154</para>
        /// </summary>
        /// <param name="data">The input array</param>
        /// <param name="index">The index array</param>
        /// <param name="axis">
        ///     int or None. The axis to picking the elements. Negative values means indexing from right to left. If
        ///     is `None`, the elements in the index w.r.t the flattened input will be picked.
        /// </param>
        /// <param name="keepdims">If true, the axis where we pick the elements is left in the result as dimension with size one.</param>
        /// <param name="mode">
        ///     Specify how out-of-bound indices behave. Default is "clip". "clip" means clip to the range. So, if
        ///     all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.
        ///     "wrap" means to wrap around.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray Pick(NDArray index, int? axis = -1, bool keepdims = false, PickMode mode = PickMode.Clip)
        {
            return new Operator("pick")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("mode", MxUtil.EnumToString<PickMode>(mode, PickModeConvert))
                .SetInput("data", this)
                .SetInput("index", index)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the sum of array elements over given axes.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para> </para>
        ///     <para>  `sum` and `sum_axis` are equivalent.</para>
        ///     <para>  For ndarray of csr storage type summation along axis 0 and axis 1 is supported.</para>
        ///     <para>  Setting keepdims or exclude to True will cause a fallback to dense operator.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  data = [[[1, 2], [2, 3], [1, 3]],</para>
        ///     <para>          [[1, 4], [4, 3], [5, 2]],</para>
        ///     <para>          [[7, 1], [7, 2], [7, 3]]]</para>
        ///     <para> </para>
        ///     <para>  sum(data, axis=1)</para>
        ///     <para>  [[  4.   8.]</para>
        ///     <para>   [ 10.   9.]</para>
        ///     <para>   [ 21.   6.]]</para>
        ///     <para> </para>
        ///     <para>  sum(data, axis=[1,2])</para>
        ///     <para>  [ 12.  19.  27.]</para>
        ///     <para> </para>
        ///     <para>  data = [[1, 2, 0],</para>
        ///     <para>          [3, 0, 1],</para>
        ///     <para>          [4, 1, 0]]</para>
        ///     <para> </para>
        ///     <para>  csr = cast_storage(data, 'csr')</para>
        ///     <para> </para>
        ///     <para>  sum(csr, axis=0)</para>
        ///     <para>  [ 8.  3.  1.]</para>
        ///     <para> </para>
        ///     <para>  sum(csr, axis=1)</para>
        ///     <para>  [ 3.  4.  5.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L116</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sum(Shape axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Sum(int axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public float Sum()
        {
            var x = Sum(null);
            return x.AsScalar<float>();
        }

        /// <summary>
        ///     <para>Computes the mean of array elements over given axes.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L132</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Mean(Shape axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("mean")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Mean(int axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("mean")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public float Mean()
        {
            var x = Mean(null);
            return x.AsScalar<float>();
        }

        /// <summary>
        ///     <para>Computes the product of array elements over given axes.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L147</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Prod(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("prod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Prod(int axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("prod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L162</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Nansum(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("nansum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L177</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Nanprod(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("nanprod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the max of array elements over given axes.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L191</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Max(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("max")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Max(int axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("max")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the min of array elements over given axes.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L205</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Min(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("min")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Min(int axis, bool keepdims = false, bool exclude = false)
        {
            return new Operator("min")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Broadcasts the input array over particular axes.</para>
        ///     <para> </para>
        ///     <para>Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to</para>
        ///     <para>`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   // given x of shape (1,2,1)</para>
        ///     <para>   x = [[[ 1.],</para>
        ///     <para>         [ 2.]]]</para>
        ///     <para> </para>
        ///     <para>   // broadcast x on on axis 2</para>
        ///     <para>   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],</para>
        ///     <para>                                         [ 2.,  2.,  2.]]]</para>
        ///     <para>   // broadcast x on on axes 0 and 2</para>
        ///     <para>   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],</para>
        ///     <para>                                                 [ 2.,  2.,  2.]],</para>
        ///     <para>                                                [[ 1.,  1.,  1.],</para>
        ///     <para>                                                 [ 2.,  2.,  2.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L238</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">The axes to perform the broadcasting.</param>
        /// <param name="size">Target sizes of the broadcasting axes.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray BroadcastAxis(Shape axis = null, Shape size = null)
        {
            if (axis == null) axis = new Shape();
            if (size == null) size = new Shape();

            return new Operator("broadcast_axis")
                .SetParam("axis", axis)
                .SetParam("size", size)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Broadcasts the input array to a new shape.</para>
        ///     <para> </para>
        ///     <para>Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations</para>
        ///     <para>with arrays of different shapes efficiently without creating multiple copies of arrays.</para>
        ///     <para>
        ///         Also see, `Broadcasting
        ///         <https:// docs.scipy.org/ doc/ numpy/ user/ basics.broadcasting.html>`_ for more explanation.
        ///     </para>
        ///     <para> </para>
        ///     <para>Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to</para>
        ///     <para>`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.</para>
        ///     <para> </para>
        ///     <para>For example::</para>
        ///     <para> </para>
        ///     <para>   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],</para>
        ///     <para>                                           [ 1.,  2.,  3.]])</para>
        ///     <para> </para>
        ///     <para>The dimension which you do not want to change can also be kept as `0` which means copy the original value.</para>
        ///     <para>So with `shape=(2,0)`, we will obtain the same result as in the above example.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L262</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="shape">
        ///     The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A =
        ///     broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray BroadcastTo(Shape shape = null)
        {
            if (shape == null) shape = new Shape();

            return new Operator("broadcast_to")
                .SetParam("shape", shape)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the norm on an NDArray.</para>
        ///     <para> </para>
        ///     <para>This operator computes the norm on an NDArray with the specified axis, depending</para>
        ///     <para>on the value of the ord parameter. By default, it computes the L2 norm on the entire</para>
        ///     <para>array. Currently only ord=2 supports sparse ndarrays.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[[1, 2],</para>
        ///     <para>        [3, 4]],</para>
        ///     <para>       [[2, 2],</para>
        ///     <para>        [5, 6]]]</para>
        ///     <para> </para>
        ///     <para>  norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]</para>
        ///     <para>                            [5.3851647 6.3245554]]</para>
        ///     <para> </para>
        ///     <para>  norm(x, ord=1, axis=1) = [[4., 6.],</para>
        ///     <para>                            [7., 8.]]</para>
        ///     <para> </para>
        ///     <para>  rsp = x.cast_storage('row_sparse')</para>
        ///     <para> </para>
        ///     <para>  norm(rsp) = [5.47722578]</para>
        ///     <para> </para>
        ///     <para>  csr = x.cast_storage('csr')</para>
        ///     <para> </para>
        ///     <para>  norm(csr) = [5.47722578]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L350</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="ord">Order of the norm. Currently ord=1 and ord=2 is supported.</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,      and the matrix
        ///     norms of these matrices are computed.
        /// </param>
        /// <param name="out_dtype">The data type of the output.</param>
        /// <param name="keepdims">If this is set to `True`, the reduced axis is left in the result as dimension with size one.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Norm(int ord = 2, Shape axis = null, NormOutDtype? out_dtype = null, bool keepdims = false)
        {
            return new Operator("norm")
                .SetParam("ord", ord)
                .SetParam("axis", axis)
                .SetParam("out_dtype", MxUtil.EnumToString(out_dtype, NormOutDtypeConvert))
                .SetParam("keepdims", keepdims)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Casts tensor storage type to the new type.</para>
        ///     <para> </para>
        ///     <para>When an NDArray with default storage type is cast to csr or row_sparse storage,</para>
        ///     <para>the result is compact, which means:</para>
        ///     <para> </para>
        ///     <para>- for csr, zero values will not be retained</para>
        ///     <para>- for row_sparse, row slices of all zeros will not be retained</para>
        ///     <para> </para>
        ///     <para>The storage type of ``cast_storage`` output depends on stype parameter:</para>
        ///     <para> </para>
        ///     <para>- cast_storage(csr, 'default') = default</para>
        ///     <para>- cast_storage(row_sparse, 'default') = default</para>
        ///     <para>- cast_storage(default, 'csr') = csr</para>
        ///     <para>- cast_storage(default, 'row_sparse') = row_sparse</para>
        ///     <para>- cast_storage(csr, 'csr') = csr</para>
        ///     <para>- cast_storage(row_sparse, 'row_sparse') = row_sparse</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>    dense = [[ 0.,  1.,  0.],</para>
        ///     <para>             [ 2.,  0.,  3.],</para>
        ///     <para>             [ 0.,  0.,  0.],</para>
        ///     <para>             [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para>    # cast to row_sparse storage type</para>
        ///     <para>    rsp = cast_storage(dense, 'row_sparse')</para>
        ///     <para>    rsp.indices = [0, 1]</para>
        ///     <para>    rsp.values = [[ 0.,  1.,  0.],</para>
        ///     <para>                  [ 2.,  0.,  3.]]</para>
        ///     <para> </para>
        ///     <para>    # cast to csr storage type</para>
        ///     <para>    csr = cast_storage(dense, 'csr')</para>
        ///     <para>    csr.indices = [1, 0, 2]</para>
        ///     <para>    csr.values = [ 1.,  2.,  3.]</para>
        ///     <para>    csr.indptr = [0, 1, 3, 3, 3]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\cast_storage.cc:L71</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="stype">Output storage type.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray ToSType(StorageStype stype)
        {
            if (stype ==  StorageStype.Csr && this.Shape.Dimension != 2)
            {
                throw new System.Exception("To convert to a CSR, the NDArray should be 2 Dimensional. Current shape is " + this.Shape);
            }

            if (this.SType == stype)
                return this;

            return new Operator("cast_storage")
                .SetParam("stype", MxUtil.EnumToString<StorageStype>(stype, CastStorageStypeConvert))
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Extracts a diagonal or constructs a diagonal array.</para>
        ///     <para> </para>
        ///     <para>``diag``'s behavior depends on the input array dimensions:</para>
        ///     <para> </para>
        ///     <para>- 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.</para>
        ///     <para>- N-D arrays: extracts the diagonals of the sub-arrays with axes specified by ``axis1`` and ``axis2``.</para>
        ///     <para>  The output shape would be decided by removing the axes numbered ``axis1`` and ``axis2`` from the</para>
        ///     <para>  input shape and appending to the result a new axis with the size of the diagonals in question.</para>
        ///     <para> </para>
        ///     <para>  For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2`` are 0 and 2</para>
        ///     <para>  respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[1, 2, 3],</para>
        ///     <para>       [4, 5, 6]]</para>
        ///     <para> </para>
        ///     <para>  diag(x) = [1, 5]</para>
        ///     <para> </para>
        ///     <para>  diag(x, k=1) = [2, 6]</para>
        ///     <para> </para>
        ///     <para>  diag(x, k=-1) = [4]</para>
        ///     <para> </para>
        ///     <para>  x = [1, 2, 3]</para>
        ///     <para> </para>
        ///     <para>  diag(x) = [[1, 0, 0],</para>
        ///     <para>             [0, 2, 0],</para>
        ///     <para>             [0, 0, 3]]</para>
        ///     <para> </para>
        ///     <para>  diag(x, k=1) = [[0, 1, 0],</para>
        ///     <para>                  [0, 0, 2],</para>
        ///     <para>                  [0, 0, 0]]</para>
        ///     <para> </para>
        ///     <para>  diag(x, k=-1) = [[0, 0, 0],</para>
        ///     <para>                   [1, 0, 0],</para>
        ///     <para>                   [0, 2, 0]]</para>
        ///     <para> </para>
        ///     <para>  x = [[[1, 2],</para>
        ///     <para>        [3, 4]],</para>
        ///     <para> </para>
        ///     <para>       [[5, 6],</para>
        ///     <para>        [7, 8]]]</para>
        ///     <para> </para>
        ///     <para>  diag(x) = [[1, 7],</para>
        ///     <para>             [2, 8]]</para>
        ///     <para> </para>
        ///     <para>  diag(x, k=1) = [[3],</para>
        ///     <para>                  [4]]</para>
        ///     <para> </para>
        ///     <para>  diag(x, axis1=-2, axis2=-1) = [[1, 4],</para>
        ///     <para>                                 [5, 8]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\diag_op.cc:L87</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="k">
        ///     Diagonal in question. The default is 0. Use k>0 for diagonals above the main diagonal, and k
        ///     <0 for diagonals below the main diagonal. If input has shape ( S0 S1) k must be between - S0 and S1
        /// </param>
        /// <param name="axis1">The first axis of the sub-arrays of interest. Ignored when the input is a 1-D array.</param>
        /// <param name="axis2">The second axis of the sub-arrays of interest. Ignored when the input is a 1-D array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Diag(int k = 0, int axis1 = 0, int axis2 = 1)
        {
            return new Operator("diag")
                .SetParam("k", k)
                .SetParam("axis1", axis1)
                .SetParam("axis2", axis2)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Reshape some or all dimensions of `lhs` to have the same shape as some or all dimensions of `rhs`.</para>
        ///     <para> </para>
        ///     <para>Returns a **view** of the `lhs` array with a new shape without altering any data.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [1, 2, 3, 4, 5, 6]</para>
        ///     <para>  y = [[0, -4], [3, 2], [2, 2]]</para>
        ///     <para>  reshape_like(x, y) = [[1, 2], [3, 4], [5, 6]]</para>
        ///     <para> </para>
        ///     <para>More precise control over how dimensions are inherited is achieved by specifying \</para>
        ///     <para>slices over the `lhs` and `rhs` array dimensions. Only the sliced `lhs` dimensions \</para>
        ///     <para>are reshaped to the `rhs` sliced dimensions, with the non-sliced `lhs` dimensions staying the same.</para>
        ///     <para> </para>
        ///     <para>  Examples::</para>
        ///     <para> </para>
        ///     <para>
        ///         - lhs shape = (30,7), rhs shape = (15,2,4), lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2, output shape =
        ///         (15,2,7)
        ///     </para>
        ///     <para>
        ///         - lhs shape = (3, 5), rhs shape = (1,15,4), lhs_begin=0, lhs_end=2, rhs_begin=1, rhs_end=2, output shape =
        ///         (15)
        ///     </para>
        ///     <para> </para>
        ///     <para>
        ///         Negative indices are supported, and `None` can be used for either `lhs_end` or `rhs_end` to indicate the end
        ///         of the range.
        ///     </para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>
        ///         - lhs shape = (30, 12), rhs shape = (4, 2, 2, 3), lhs_begin=-1, lhs_end=None, rhs_begin=1, rhs_end=None,
        ///         output shape = (30, 2, 2, 3)
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L469</para>
        /// </summary>
        /// <param name="lhs">First input.</param>
        /// <param name="rhs">Second input.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray ReshapeLike(NDArray lhs, NDArray rhs)
        {
            return new Operator("reshape_like")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns a 1D int64 array containing the shape of data.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  shape_array([[1,2,3,4], [5,6,7,8]]) = [2,4]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L529</para>
        /// </summary>
        /// <param name="data">Input Array.</param>
        /// <param name="lhs_begin">
        ///     Defaults to 0. The beginning index along which the lhs dimensions are to be reshaped. Supports
        ///     negative indices.
        /// </param>
        /// <param name="lhs_end">
        ///     Defaults to None. The ending index along which the lhs dimensions are to be used for reshaping.
        ///     Supports negative indices.
        /// </param>
        /// <param name="rhs_begin">
        ///     Defaults to 0. The beginning index along which the rhs dimensions are to be used for reshaping.
        ///     Supports negative indices.
        /// </param>
        /// <param name="rhs_end">
        ///     Defaults to None. The ending index along which the rhs dimensions are to be used for reshaping.
        ///     Supports negative indices.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray ShapeArray(int? lhs_begin = null, int? lhs_end = null, int? rhs_begin = null,
            int? rhs_end = null)
        {
            return new Operator("shape_array")
                .SetParam("lhs_begin", lhs_begin)
                .SetParam("lhs_end", lhs_end)
                .SetParam("rhs_begin", rhs_begin)
                .SetParam("rhs_end", rhs_end)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns a 1D int64 array containing the size of data.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  size_array([[1,2,3,4], [5,6,7,8]]) = [8]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L581</para>
        /// </summary>
        /// <param name="data">Input Array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray SizeArray()
        {
            return new Operator("size_array")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Casts all elements of the input to a new type.</para>
        ///     <para> </para>
        ///     <para>.. note:: ``Cast`` is deprecated. Use ``cast`` instead.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   cast([0.9, 1.3], dtype='int32') = [0, 1]</para>
        ///     <para>   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]</para>
        ///     <para>   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L619</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="dtype">Output data type.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Cast(DType dtype)
        {
            return new Operator("cast")
                .SetParam("dtype", dtype)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Numerical negative of the argument, element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``negative`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - negative(default) = default</para>
        ///     <para>   - negative(row_sparse) = row_sparse</para>
        ///     <para>   - negative(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Negative()
        {
            return new Operator("negative")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the reciprocal of the argument, element-wise.</para>
        ///     <para> </para>
        ///     <para>Calculates 1/x.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L663</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Reciprocal()
        {
            return new Operator("reciprocal")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise absolute value of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   abs([-2, 0, 3]) = [2, 0, 3]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``abs`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - abs(default) = default</para>
        ///     <para>   - abs(row_sparse) = row_sparse</para>
        ///     <para>   - abs(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L685</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Abs()
        {
            return new Operator("abs")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise sign of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   sign([-2, 0, 3]) = [-1, 0, 1]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``sign`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - sign(default) = default</para>
        ///     <para>   - sign(row_sparse) = row_sparse</para>
        ///     <para>   - sign(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L704</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sign()
        {
            return new Operator("sign")
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Sigmoid()
        {
            return new Operator("sigmoid")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise rounded value to the nearest integer of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``round`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>  - round(default) = default</para>
        ///     <para>  - round(row_sparse) = row_sparse</para>
        ///     <para>  - round(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L723</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Round()
        {
            return new Operator("round")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise rounded value to the nearest integer of the input.</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.</para>
        ///     <para>   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``rint`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - rint(default) = default</para>
        ///     <para>   - rint(row_sparse) = row_sparse</para>
        ///     <para>   - rint(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L744</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Rint()
        {
            return new Operator("rint")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise ceiling of the input.</para>
        ///     <para> </para>
        ///     <para>The ceil of the scalar x is the smallest integer i, such that i >= x.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``ceil`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - ceil(default) = default</para>
        ///     <para>   - ceil(row_sparse) = row_sparse</para>
        ///     <para>   - ceil(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L763</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Ceil()
        {
            return new Operator("ceil")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise floor of the input.</para>
        ///     <para> </para>
        ///     <para>
        ///         The floor of the scalar x is the largest integer i, such that i <= x.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``floor`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - floor(default) = default</para>
        ///     <para>   - floor(row_sparse) = row_sparse</para>
        ///     <para>   - floor(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L782</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Floor()
        {
            return new Operator("floor")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Return the element-wise truncated value of the input.</para>
        ///     <para> </para>
        ///     <para>The truncated value of the scalar x is the nearest integer i which is closer to</para>
        ///     <para>zero than x is. In short, the fractional part of the signed number x is discarded.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``trunc`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - trunc(default) = default</para>
        ///     <para>   - trunc(row_sparse) = row_sparse</para>
        ///     <para>   - trunc(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L802</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Trunc()
        {
            return new Operator("trunc")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise rounded value to the nearest \</para>
        ///     <para>integer towards zero of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``fix`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - fix(default) = default</para>
        ///     <para>   - fix(row_sparse) = row_sparse</para>
        ///     <para>   - fix(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L820</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
       
        public NDArray Fix()
        {
            return new Operator("fix")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise squared value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   square(x) = x^2</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   square([2, 3, 4]) = [4, 9, 16]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``square`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - square(default) = default</para>
        ///     <para>   - square(row_sparse) = row_sparse</para>
        ///     <para>   - square(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L840</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Square()
        {
            return new Operator("square")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise square-root value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   \textrm{sqrt}(x) = \sqrt{x}</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   sqrt([4, 9, 16]) = [2, 3, 4]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``sqrt`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - sqrt(default) = default</para>
        ///     <para>   - sqrt(row_sparse) = row_sparse</para>
        ///     <para>   - sqrt(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L863</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sqrt()
        {
            return new Operator("sqrt")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse square-root value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   rsqrt(x) = 1/\sqrt{x}</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``rsqrt`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L883</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Rsqrt()
        {
            return new Operator("rsqrt")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise cube-root value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   cbrt(x) = \sqrt[3]{x}</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   cbrt([1, 8, -125]) = [1, 2, -5]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``cbrt`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - cbrt(default) = default</para>
        ///     <para>   - cbrt(row_sparse) = row_sparse</para>
        ///     <para>   - cbrt(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L906</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Cbrt()
        {
            return new Operator("cbrt")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise gauss error function of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   erf([0, -1., 10.]) = [0., -0.8427, 1.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L920</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Erf()
        {
            return new Operator("erf")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse gauss error function of the input.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   erfinv([0, 0.5., -1.]) = [0., 0.4769, -inf]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L936</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Erfinv()
        {
            return new Operator("erfinv")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse cube-root value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   rcbrt(x) = 1/\sqrt[3]{x}</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L955</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Rcbrt()
        {
            return new Operator("rcbrt")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise exponential value of the input.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   exp(x) = e^x \approx 2.718^x</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``exp`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L978</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Exp()
        {
            return new Operator("exp")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise Natural logarithmic value of the input.</para>
        ///     <para> </para>
        ///     <para>The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``</para>
        ///     <para> </para>
        ///     <para>The storage type of ``log`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L990</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Log()
        {
            return new Operator("log")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise Base-10 logarithmic value of the input.</para>
        ///     <para> </para>
        ///     <para>``10**log10(x) = x``</para>
        ///     <para> </para>
        ///     <para>The storage type of ``log10`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1003</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Log10()
        {
            return new Operator("log10")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise Base-2 logarithmic value of the input.</para>
        ///     <para> </para>
        ///     <para>``2**log2(x) = x``</para>
        ///     <para> </para>
        ///     <para>The storage type of ``log2`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1015</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Log2()
        {
            return new Operator("log2")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise ``log(1 + x)`` value of the input.</para>
        ///     <para> </para>
        ///     <para>This function is more accurate than ``log(1 + x)``  for small ``x`` so that</para>
        ///     <para>:math:`1+x\approx 1`</para>
        ///     <para> </para>
        ///     <para>The storage type of ``log1p`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - log1p(default) = default</para>
        ///     <para>   - log1p(row_sparse) = row_sparse</para>
        ///     <para>   - log1p(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1040</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Log1P()
        {
            return new Operator("log1p")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns ``exp(x) - 1`` computed element-wise on the input.</para>
        ///     <para> </para>
        ///     <para>This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``expm1`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - expm1(default) = default</para>
        ///     <para>   - expm1(row_sparse) = row_sparse</para>
        ///     <para>   - expm1(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L1058</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Expm1()
        {
            return new Operator("expm1")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the gamma function (extension of the factorial function \</para>
        ///     <para>to the reals), computed element-wise on the input array.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``gamma`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Gamma()
        {
            return new Operator("gamma")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise log of the absolute value of the gamma function \</para>
        ///     <para>of the input.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``gammaln`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Gammaln()
        {
            return new Operator("gammaln")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the result of logical NOT (!) function</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para>  logical_not([-2., 0., 1.]) = [0., 1., 0.]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray LogicalNot()
        {
            return new Operator("logical_not")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the element-wise sine of the input array.</para>
        ///     <para> </para>
        ///     <para>The input should be in radians (:math:`2\pi` rad equals 360 degrees).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``sin`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - sin(default) = default</para>
        ///     <para>   - sin(row_sparse) = row_sparse</para>
        ///     <para>   - sin(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L46</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sin()
        {
            return new Operator("sin")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the element-wise cosine of the input array.</para>
        ///     <para> </para>
        ///     <para>The input should be in radians (:math:`2\pi` rad equals 360 degrees).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``cos`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L63</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Cos()
        {
            return new Operator("cos")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the element-wise tangent of the input array.</para>
        ///     <para> </para>
        ///     <para>The input should be in radians (:math:`2\pi` rad equals 360 degrees).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``tan`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - tan(default) = default</para>
        ///     <para>   - tan(row_sparse) = row_sparse</para>
        ///     <para>   - tan(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L83</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Tan()
        {
            return new Operator("tan")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse sine of the input array.</para>
        ///     <para> </para>
        ///     <para>The input should be in the range `[-1, 1]`.</para>
        ///     <para>The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arcsin`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - arcsin(default) = default</para>
        ///     <para>   - arcsin(row_sparse) = row_sparse</para>
        ///     <para>   - arcsin(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L104</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arcsin()
        {
            return new Operator("arcsin")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse cosine of the input array.</para>
        ///     <para> </para>
        ///     <para>The input should be in range `[-1, 1]`.</para>
        ///     <para>The output is in the closed interval :math:`[0, \pi]`</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arccos`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L123</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arccos()
        {
            return new Operator("arccos")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns element-wise inverse tangent of the input array.</para>
        ///     <para> </para>
        ///     <para>The output is in the closed interval :math:`[-\pi/2, \pi/2]`</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arctan`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - arctan(default) = default</para>
        ///     <para>   - arctan(row_sparse) = row_sparse</para>
        ///     <para>   - arctan(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L144</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arctan()
        {
            return new Operator("arctan")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Converts each element of the input array from radians to degrees.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``degrees`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - degrees(default) = default</para>
        ///     <para>   - degrees(row_sparse) = row_sparse</para>
        ///     <para>   - degrees(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L163</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Degrees()
        {
            return new Operator("degrees")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Converts each element of the input array from degrees to radians.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``radians`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - radians(default) = default</para>
        ///     <para>   - radians(row_sparse) = row_sparse</para>
        ///     <para>   - radians(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L182</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Radians()
        {
            return new Operator("radians")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the hyperbolic sine of the input array, computed element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   sinh(x) = 0.5\times(exp(x) - exp(-x))</para>
        ///     <para> </para>
        ///     <para>The storage type of ``sinh`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - sinh(default) = default</para>
        ///     <para>   - sinh(row_sparse) = row_sparse</para>
        ///     <para>   - sinh(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L201</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sinh()
        {
            return new Operator("sinh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the hyperbolic cosine  of the input array, computed element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   cosh(x) = 0.5\times(exp(x) + exp(-x))</para>
        ///     <para> </para>
        ///     <para>The storage type of ``cosh`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L216</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Cosh()
        {
            return new Operator("cosh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the hyperbolic tangent of the input array, computed element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   tanh(x) = sinh(x) / cosh(x)</para>
        ///     <para> </para>
        ///     <para>The storage type of ``tanh`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - tanh(default) = default</para>
        ///     <para>   - tanh(row_sparse) = row_sparse</para>
        ///     <para>   - tanh(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L234</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Tanh()
        {
            return new Operator("tanh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the element-wise inverse hyperbolic sine of the input array, \</para>
        ///     <para>computed element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arcsinh`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - arcsinh(default) = default</para>
        ///     <para>   - arcsinh(row_sparse) = row_sparse</para>
        ///     <para>   - arcsinh(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L250</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arcsinh()
        {
            return new Operator("arcsinh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the element-wise inverse hyperbolic cosine of the input array, \</para>
        ///     <para>computed element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arccosh`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L264</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arccosh()
        {
            return new Operator("arccosh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the element-wise inverse hyperbolic tangent of the input array, \</para>
        ///     <para>computed element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``arctanh`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - arctanh(default) = default</para>
        ///     <para>   - arctanh(row_sparse) = row_sparse</para>
        ///     <para>   - arctanh(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_trig.cc:L281</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Arctanh()
        {
            return new Operator("arctanh")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Return an array of zeros with the same shape, type and storage type</para>
        ///     <para>as the input array.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``zeros_like`` output depends on the storage type of the input</para>
        ///     <para> </para>
        ///     <para>- zeros_like(row_sparse) = row_sparse</para>
        ///     <para>- zeros_like(csr) = csr</para>
        ///     <para>- zeros_like(default) = default</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.,  1.,  1.],</para>
        ///     <para>       [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>  zeros_like(x) = [[ 0.,  0.,  0.],</para>
        ///     <para>                   [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <returns>returns new NDArray</returns>
        public NDArray ZerosLike()
        {
            return new Operator("zeros_like")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Return an array of ones with the same shape and type</para>
        ///     <para>as the input array.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.,  0.,  0.],</para>
        ///     <para>       [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para>  ones_like(x) = [[ 1.,  1.,  1.],</para>
        ///     <para>                  [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <returns>returns new NDArray</returns>
        public NDArray OnesLike()
        {
            return new Operator("ones_like")
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Permutes the dimensions of an array.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1, 2],</para>
        ///     <para>       [ 3, 4]]</para>
        ///     <para> </para>
        ///     <para>  transpose(x) = [[ 1.,  3.],</para>
        ///     <para>                  [ 2.,  4.]]</para>
        ///     <para> </para>
        ///     <para>  x = [[[ 1.,  2.],</para>
        ///     <para>        [ 3.,  4.]],</para>
        ///     <para> </para>
        ///     <para>       [[ 5.,  6.],</para>
        ///     <para>        [ 7.,  8.]]]</para>
        ///     <para> </para>
        ///     <para>  transpose(x) = [[[ 1.,  5.],</para>
        ///     <para>                   [ 3.,  7.]],</para>
        ///     <para> </para>
        ///     <para>                  [[ 2.,  6.],</para>
        ///     <para>                   [ 4.,  8.]]]</para>
        ///     <para> </para>
        ///     <para>  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],</para>
        ///     <para>                                 [ 5.,  6.]],</para>
        ///     <para> </para>
        ///     <para>                                [[ 3.,  4.],</para>
        ///     <para>                                 [ 7.,  8.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L399</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="axes">Target axis order. By default the axes will be inverted.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Transpose(Shape axes = null)
        {
            if (axes == null) axes = new Shape();

            return new Operator("transpose")
                .SetParam("axes", axes)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Inserts a new axis of size 1 into the array shape</para>
        ///     <para> </para>
        ///     <para>For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``</para>
        ///     <para>will return a new array with shape ``(2,1,3,4)``.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L440</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="axis">
        ///     Position where new axis is to be inserted. Suppose that the input `NDArray`'s dimension is `ndim`,
        ///     the range of the inserted axis is `[-ndim, ndim]`
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray ExpandDims(int axis, bool inPlace = false)
        {
            if (!inPlace)
            {
                return new Operator("expand_dims")
                    .SetParam("axis", axis)
                    .SetInput("data", this)
                    .Invoke();
            }
            else
            {
                var new_shape = this.Shape.Data;
                Debug.Assert((-new_shape.Count - 1 <= axis) && (axis <= new_shape.Count), $"axis {axis} is out of range for {new_shape.Count}d array");
                if (axis < 0)
                {
                    axis += new_shape.Count + 1;
                }

                new_shape.Insert(axis, 1);
                return this.Reshape(new Shape(new_shape));
            }
        }

        /// <summary>
        ///     <para>Slices a region of the array.</para>
        ///     <para> </para>
        ///     <para>.. note:: ``crop`` is deprecated. Use ``slice`` instead.</para>
        ///     <para> </para>
        ///     <para>This function returns a sliced array between the indices given</para>
        ///     <para>by `begin` and `end` with the corresponding `step`.</para>
        ///     <para> </para>
        ///     <para>For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,</para>
        ///     <para>slice operation with ``begin=(b_0, b_1...b_m-1)``,</para>
        ///     <para>``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,</para>
        ///     <para>
        ///         where m <= n, results in an array with the shape</para>
        ///     <para>``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.</para>
        ///     <para> </para>
        ///     <para>The resulting array's *k*-th dimension contains elements</para>
        ///     <para>from the *k*-th dimension of the input array starting</para>
        ///     <para>from index ``b_k`` (inclusive) with step ``s_k``</para>
        ///     <para>until reaching ``e_k`` (exclusive).</para>
        ///     <para> </para>
        ///     <para>If the *k*-th elements are `None` in the sequence of `begin`, `end`,</para>
        ///     <para>and `step`, the following rule will be used to set default values.</para>
        ///     <para>If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;</para>
        ///     <para>else, set `b_k=d_k-1`, `e_k=-1`.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``slice`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- slice(csr) = csr</para>
        ///     <para>- otherwise, ``slice`` generates output with default storage</para>
        ///     <para> </para>
        ///     <para>.. note:: When input data storage type is csr, it only supports</para>
        ///     <para>   step=(), or step=(None,), or step=(1,) to generate a csr output.</para>
        ///     <para>   For other step parameter values, it falls back to slicing</para>
        ///     <para>   a dense tensor.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[  1.,   2.,   3.,   4.],</para>
        ///     <para>       [  5.,   6.,   7.,   8.],</para>
        ///     <para>       [  9.,  10.,  11.,  12.]]</para>
        ///     <para> </para>
        ///     <para>  slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],</para>
        ///     <para>                                     [ 6.,  7.,  8.]]</para>
        ///     <para>  slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],</para>
        ///     <para>                                                            [5.,  7.],</para>
        ///     <para>                                                            [1.,  3.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L530</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="begin">starting indices for the slice operation, supports negative indices.</param>
        /// <param name="end">ending indices for the slice operation, supports negative indices.</param>
        /// <param name="step">step for the slice operation, supports negative values.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Slice(Shape begin, Shape end, Shape step = null)
        {
            if (step == null) step = new Shape();

            return new Operator("slice")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Clips (limits) the values in an array.</para>
        ///     <para> </para>
        ///     <para>Given an interval, values outside the interval are clipped to the interval edges.</para>
        ///     <para>Clipping ``x`` between `a_min` and `a_x` would be::</para>
        ///     <para> </para>
        ///     <para>   clip(x, a_min, a_max) = max(min(x, a_max), a_min))</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]</para>
        ///     <para> </para>
        ///     <para>    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \</para>
        ///     <para>parameter values:</para>
        ///     <para> </para>
        ///     <para>   - clip(default) = default</para>
        ///     <para>
        ///         - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
        ///     </para>
        ///     <para>
        ///         - clip(csr, a_min <= 0, a_max >= 0) = csr
        ///     </para>
        ///     <para>
        ///         - clip(row_sparse, a_min < 0, a_max < 0) = default</para>
        ///     <para>   - clip(row_sparse, a_min > 0, a_max > 0) = default</para>
        ///     <para>
        ///         - clip(csr, a_min < 0, a_max < 0) = csr</para>
        ///     <para>   - clip(csr, a_min > 0, a_max > 0) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L747</para>
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="a_min">Minimum value</param>
        /// <param name="a_max">Maximum value</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Clip(float a_min, float a_max)
        {
            return new Operator("clip")
                .SetParam("a_min", a_min)
                .SetParam("a_max", a_max)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Repeats elements of an array.</para>
        ///     <para> </para>
        ///     <para>By default, ``repeat`` flattens the input array into 1-D and then repeats the</para>
        ///     <para>elements::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1, 2],</para>
        ///     <para>       [ 3, 4]]</para>
        ///     <para> </para>
        ///     <para>  repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]</para>
        ///     <para> </para>
        ///     <para>The parameter ``axis`` specifies the axis along which to perform repeat::</para>
        ///     <para> </para>
        ///     <para>  repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],</para>
        ///     <para>                                  [ 3.,  3.,  4.,  4.]]</para>
        ///     <para> </para>
        ///     <para>  repeat(x, repeats=2, axis=0) = [[ 1.,  2.],</para>
        ///     <para>                                  [ 1.,  2.],</para>
        ///     <para>                                  [ 3.,  4.],</para>
        ///     <para>                                  [ 3.,  4.]]</para>
        ///     <para> </para>
        ///     <para>  repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],</para>
        ///     <para>                                   [ 3.,  3.,  4.,  4.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L820</para>
        /// </summary>
        /// <param name="data">Input data array</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">
        ///     The axis along which to repeat values. The negative numbers are interpreted counting from the
        ///     backward. By default, use the flattened input array, and return a flat output array.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray Repeat(int repeats, int? axis = null)
        {
            return new Operator("repeat")
                .SetParam("repeats", repeats)
                .SetParam("axis", axis)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Repeats the whole array multiple times.</para>
        ///     <para> </para>
        ///     <para>If ``reps`` has length *d*, and input array has dimension of *n*. There are</para>
        ///     <para>three cases:</para>
        ///     <para> </para>
        ///     <para>- **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::</para>
        ///     <para> </para>
        ///     <para>    x = [[1, 2],</para>
        ///     <para>         [3, 4]]</para>
        ///     <para> </para>
        ///     <para>    tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                           [ 3.,  4.,  3.,  4.,  3.,  4.],</para>
        ///     <para>                           [ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                           [ 3.,  4.,  3.,  4.,  3.,  4.]]</para>
        ///     <para> </para>
        ///     <para>- **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for</para>
        ///     <para>  an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>    tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],</para>
        ///     <para>                          [ 3.,  4.,  3.,  4.]]</para>
        ///     <para> </para>
        ///     <para>- **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a</para>
        ///     <para>  shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::</para>
        ///     <para> </para>
        ///     <para>    tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                              [ 3.,  4.,  3.,  4.,  3.,  4.],</para>
        ///     <para>                              [ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                              [ 3.,  4.,  3.,  4.,  3.,  4.]],</para>
        ///     <para> </para>
        ///     <para>                             [[ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                              [ 3.,  4.,  3.,  4.,  3.,  4.],</para>
        ///     <para>                              [ 1.,  2.,  1.,  2.,  1.,  2.],</para>
        ///     <para>                              [ 3.,  4.,  3.,  4.,  3.,  4.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L881</para>
        /// </summary>
        /// <param name="data">Input data array</param>
        /// <param name="reps">
        ///     The number of times for repeating the tensor a. Each dim size of reps must be a positive integer. If reps has
        ///     length d, the result will have dimension of max(d, a.ndim); If a.ndim
        ///     < d, a is promoted to be d-dimensional by prepending new axes. If a.ndim>
        ///         d, reps is promoted to a.ndim by
        ///         pre-pending 1's to it.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray Tile(Shape reps)
        {
            return new Operator("tile")
                .SetParam("reps", reps)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Reverses the order of elements along given axis while preserving array shape.</para>
        ///     <para> </para>
        ///     <para>Note: reverse and flip are equivalent. We use reverse in the following examples.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.,  1.,  2.,  3.,  4.],</para>
        ///     <para>       [ 5.,  6.,  7.,  8.,  9.]]</para>
        ///     <para> </para>
        ///     <para>  reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],</para>
        ///     <para>                        [ 0.,  1.,  2.,  3.,  4.]]</para>
        ///     <para> </para>
        ///     <para>  reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],</para>
        ///     <para>                        [ 9.,  8.,  7.,  6.,  5.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L922</para>
        /// </summary>
        /// <param name="data">Input data array</param>
        /// <param name="axis">The axis which to reverse elements.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Reverse(Shape axis)
        {
            return new Operator("reverse")
                .SetParam("axis", axis)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Splits an array along a particular axis into multiple sub-arrays.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x  = [[[ 1.]</para>
        ///     <para>          [ 2.]]</para>
        ///     <para>         [[ 3.]</para>
        ///     <para>          [ 4.]]</para>
        ///     <para>         [[ 5.]</para>
        ///     <para>          [ 6.]]]</para>
        ///     <para>   x.shape = (3, 2, 1)</para>
        ///     <para> </para>
        ///     <para>   y = split_v2(x, axis=1, indices_or_sections=2) // a list of 2 arrays with shape (3, 1, 1)</para>
        ///     <para>   y = [[[ 1.]]</para>
        ///     <para>        [[ 3.]]</para>
        ///     <para>        [[ 5.]]]</para>
        ///     <para> </para>
        ///     <para>       [[[ 2.]]</para>
        ///     <para>        [[ 4.]]</para>
        ///     <para>        [[ 6.]]]</para>
        ///     <para> </para>
        ///     <para>   y[0].shape = (3, 1, 1)</para>
        ///     <para> </para>
        ///     <para>   z = split_v2(x, axis=0, indices_or_sections=3) // a list of 3 arrays with shape (1, 2, 1)</para>
        ///     <para>   z = [[[ 1.]</para>
        ///     <para>         [ 2.]]]</para>
        ///     <para> </para>
        ///     <para>       [[[ 3.]</para>
        ///     <para>         [ 4.]]]</para>
        ///     <para> </para>
        ///     <para>       [[[ 5.]</para>
        ///     <para>         [ 6.]]]</para>
        ///     <para> </para>
        ///     <para>   z[0].shape = (1, 2, 1)</para>
        ///     <para> </para>
        ///     <para>   w = split_v2(x, axis=0, indices_or_sections=(1,)) // a list of 2 arrays with shape [(1, 2, 1), (2, 2, 1)]</para>
        ///     <para>   w = [[[ 1.]</para>
        ///     <para>         [ 2.]]]</para>
        ///     <para> </para>
        ///     <para>       [[[3.]</para>
        ///     <para>         [4.]]</para>
        ///     <para> </para>
        ///     <para>        [[5.]</para>
        ///     <para>         [6.]]]</para>
        ///     <para> </para>
        ///     <para>  w[0].shape = (1, 2, 1)</para>
        ///     <para>  w[1].shape = (2, 2, 1)</para>
        ///     <para> </para>
        ///     <para>`squeeze_axis=True` removes the axis with length 1 from the shapes of the output arrays.</para>
        ///     <para>**Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only</para>
        ///     <para>along the `axis` which it is split.</para>
        ///     <para>Also `squeeze_axis` can be set to true only if ``input.shape[axis] == indices_or_sections``.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   z = split_v2(x, axis=0, indices_or_sections=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)</para>
        ///     <para>   z = [[ 1.]</para>
        ///     <para>        [ 2.]]</para>
        ///     <para> </para>
        ///     <para>       [[ 3.]</para>
        ///     <para>        [ 4.]]</para>
        ///     <para> </para>
        ///     <para>       [[ 5.]</para>
        ///     <para>        [ 6.]]</para>
        ///     <para>   z[0].shape = (2, 1)</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L1214</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="indices">
        ///     Indices of splits. The elements should denote the boundaries of at which split is performed along
        ///     the `axis`.
        /// </param>
        /// <param name="axis">Axis along which to split.</param>
        /// <param name="squeeze_axis">
        ///     If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that
        ///     setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also
        ///     `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.
        /// </param>
        /// <param name="sections">Number of sections if equally splitted. Default to 0 which means split by indices.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray SplitV2(Shape indices, int axis = 1, bool squeeze_axis = false, int sections = 0)
        {
            return new Operator("_split_v2")
                .SetParam("indices", indices)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetParam("sections", sections)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArrayList Split(int num_outputs, int axis = 1, bool squeeze_axis = false)
        {
            return nd.Split(this, num_outputs, axis, squeeze_axis);
        }

        /// <summary>
        ///     <para>Returns the top *k* elements in an input array along the given axis.</para>
        ///     <para> The returned elements will be sorted.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.3,  0.2,  0.4],</para>
        ///     <para>       [ 0.1,  0.3,  0.2]]</para>
        ///     <para> </para>
        ///     <para>  // returns an index of the largest element on last axis</para>
        ///     <para>  topk(x) = [[ 2.],</para>
        ///     <para>             [ 1.]]</para>
        ///     <para> </para>
        ///     <para>  // returns the value of top-2 largest elements on last axis</para>
        ///     <para>  topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],</para>
        ///     <para>                                   [ 0.3,  0.2]]</para>
        ///     <para> </para>
        ///     <para>  // returns the value of top-2 smallest elements on last axis</para>
        ///     <para>  topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],</para>
        ///     <para>                                               [ 0.1 ,  0.2]]</para>
        ///     <para> </para>
        ///     <para>  // returns the value of top-2 largest elements on axis 0</para>
        ///     <para>  topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],</para>
        ///     <para>                                           [ 0.1,  0.2,  0.2]]</para>
        ///     <para> </para>
        ///     <para>  // flattens and then returns list of both values and indices</para>
        ///     <para>  topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\ordering_op.cc:L64</para>
        /// </summary>
        /// <param name="data">The input array</param>
        /// <param name="axis">
        ///     Axis along which to choose the top k indices. If not given, the flattened array is used. Default is
        ///     -1.
        /// </param>
        /// <param name="k">
        ///     Number of top elements to select, should be always smaller than or equal to the element number in the given axis. A
        ///     global sort is performed if set k < 1.</param>
        /// <param name="ret_typ">
        ///     The return type. "value" means to return the top k values, "indices" means to return the indices
        ///     of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means
        ///     to return a list of both values and indices of top k elements.
        /// </param>
        /// <param name="is_ascend">
        ///     Whether to choose k largest or k smallest elements. Top K largest elements will be chosen if
        ///     set to false.
        /// </param>
        /// <param name="dtype">
        ///     DType of the output indices when ret_typ is "indices" or "both". An error will be raised if the
        ///     selected data type cannot precisely represent the indices.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArrayList Topk(int? axis = -1, int k = 1, TopkRetTyp ret_typ = TopkRetTyp.Indices, bool is_ascend = false,
            DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;
            NDArrayList outputs = new NDArrayList();
            new Operator("topk")
                .SetParam("axis", axis)
                .SetParam("k", k)
                .SetParam("ret_typ", MxUtil.EnumToString<TopkRetTyp>(ret_typ, TopkRetTypConvert))
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", this)
                .Invoke(outputs);

            return outputs;
        }

        /// <summary>
        ///     <para>Returns a sorted copy of an input array along the given axis.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1, 4],</para>
        ///     <para>       [ 3, 1]]</para>
        ///     <para> </para>
        ///     <para>  // sorts along the last axis</para>
        ///     <para>  sort(x) = [[ 1.,  4.],</para>
        ///     <para>             [ 1.,  3.]]</para>
        ///     <para> </para>
        ///     <para>  // flattens and then sorts</para>
        ///     <para>  sort(x) = [ 1.,  1.,  3.,  4.]</para>
        ///     <para> </para>
        ///     <para>  // sorts along the first axis</para>
        ///     <para>  sort(x, axis=0) = [[ 1.,  1.],</para>
        ///     <para>                     [ 3.,  4.]]</para>
        ///     <para> </para>
        ///     <para>  // in a descend order</para>
        ///     <para>  sort(x, is_ascend=0) = [[ 4.,  1.],</para>
        ///     <para>                          [ 3.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\ordering_op.cc:L127</para>
        /// </summary>
        /// <param name="data">The input array</param>
        /// <param name="axis">
        ///     Axis along which to choose sort the input tensor. If not given, the flattened array is used. Default
        ///     is -1.
        /// </param>
        /// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray Sort(int? axis = -1, bool is_ascend = true)
        {
            return new Operator("sort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>Returns the indices that would sort an input array along the given axis.</para>
        ///     <para> </para>
        ///     <para>This function performs sorting along the given axis and returns an array of indices having same shape</para>
        ///     <para>as an input array that index data in sorted order.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 0.3,  0.2,  0.4],</para>
        ///     <para>       [ 0.1,  0.3,  0.2]]</para>
        ///     <para> </para>
        ///     <para>  // sort along axis -1</para>
        ///     <para>  argsort(x) = [[ 1.,  0.,  2.],</para>
        ///     <para>                [ 0.,  2.,  1.]]</para>
        ///     <para> </para>
        ///     <para>  // sort along axis 0</para>
        ///     <para>  argsort(x, axis=0) = [[ 1.,  0.,  1.]</para>
        ///     <para>                        [ 0.,  1.,  0.]]</para>
        ///     <para> </para>
        ///     <para>  // flatten and then sort</para>
        ///     <para>  argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\ordering_op.cc:L177</para>
        /// </summary>
        /// <param name="data">The input array</param>
        /// <param name="axis">Axis along which to sort the input tensor. If not given, the flattened array is used. Default is -1.</param>
        /// <param name="is_ascend">Whether to sort in ascending or descending order.</param>
        /// <param name="dtype">
        ///     DType of the output indices. It is only valid when ret_typ is "indices" or "both". An error will be
        ///     raised if the selected data type cannot precisely represent the indices.
        /// </param>
        /// <returns>returns new NDArray</returns>
        public NDArray Argsort(int? axis = -1, bool is_ascend = true, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("argsort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>
        ///         Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a
        ///         single multi index is given by a column of the input matrix. The leading dimension may be left unspecified by
        ///         using -1 as placeholder.
        ///     </para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   A = [[3,6,6],[4,5,1]]</para>
        ///     <para>   ravel(A, shape=(7,6)) = [22,41,37]</para>
        ///     <para>   ravel(A, shape=(-1,6)) = [22,41,37]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\ravel.cc:L42</para>
        /// </summary>
        /// <param name="data">Batch of multi-indices</param>
        /// <param name="shape">Shape of the array into which the multi-indices apply.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray RavelMultiIndex(Shape shape = null)
        {
            return new Operator("_ravel_multi_index")
                .SetParam("shape", shape)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>
        ///         Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a
        ///         single multi index is given by a column of the output matrix. The leading dimension may be left unspecified by
        ///         using -1 as placeholder.
        ///     </para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   A = [22,41,37]</para>
        ///     <para>   unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]</para>
        ///     <para>   unravel(A, shape=(-1,6)) = [[3,6,6],[4,5,1]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\ravel.cc:L67</para>
        /// </summary>
        /// <param name="data">Array of flat indices</param>
        /// <param name="shape">Shape of the array into which the multi-indices apply.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray UnravelIndex(Shape shape = null)
        {
            return new Operator("_unravel_index")
                .SetParam("shape", shape)
                .SetInput("data", this)
                .Invoke();
        }

        /// <summary>
        ///     <para>pick rows specified by user input index array from a row sparse matrix</para>
        ///     <para>and save them in the output sparse matrix.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  data = [[1, 2], [3, 4], [5, 6]]</para>
        ///     <para>  indices = [0, 1, 3]</para>
        ///     <para>  shape = (4, 2)</para>
        ///     <para>  rsp_in = row_sparse(data, indices)</para>
        ///     <para>  to_retain = [0, 3]</para>
        ///     <para>  rsp_out = retain(rsp_in, to_retain)</para>
        ///     <para>  rsp_out.values = [[1, 2], [5, 6]]</para>
        ///     <para>  rsp_out.indices = [0, 3]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``retain`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- retain(row_sparse, default) = row_sparse</para>
        ///     <para>- otherwise, ``retain`` is not supported</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\sparse_retain.cc:L53</para>
        /// </summary>
        /// <param name="data">The input array for sparse_retain operator.</param>
        /// <param name="indices">The index array of rows ids that will be retained.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray SparseRetain(NDArray indices)
        {
            return new Operator("_sparse_retain")
                .SetInput("data", this)
                .SetInput("indices", indices)
                .Invoke();
        }

        /// <summary>
        ///     <para>Computes the square sum of array elements over a given axis</para>
        ///     <para>for row-sparse matrix. This is a temporary solution for fusing ops square and</para>
        ///     <para>sum together for row-sparse matrix to save memory for storing gradients.</para>
        ///     <para>It will become deprecated once the functionality of fusing operators is finished</para>
        ///     <para>in the future.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])</para>
        ///     <para>  rsp = dns.tostype('row_sparse')</para>
        ///     <para>  sum = mx.nd._internal._square_sum(rsp, axis=1)</para>
        ///     <para>  sum = [0, 5, 0, 25, 0]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\square_sum.cc:L63</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="axis">
        ///     The axis or axes along which to perform the reduction.      The default, `axis=()`, will compute
        ///     over all elements into a      scalar array with shape `(1,)`.      If `axis` is int, a reduction is performed on a
        ///     particular axis.      If `axis` is a tuple of ints, a reduction is performed on all the axes      specified in the
        ///     tuple.      If `exclude` is true, reduction will be performed on the axes that are      NOT in axis instead.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="keepdims">If this is set to `True`, the reduced axes are left in the result as dimension with size one.</param>
        /// <param name="exclude">Whether to perform reduction on axis that are NOT in axis instead.</param>
        /// <returns>returns new NDArray</returns>
        public NDArray SquareSum(Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("_square_sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", this)
                .Invoke();
        }

        public NDArray Fix(int value)
        {
            return new Operator("_full")
                .SetParam("shape", this.Shape)
                .SetParam("value", value)
                .SetParam("ctx", this.Context)
                .SetParam("dtype", this.DataType.Name)
                .Invoke();
        }

        public NDArray ScatterSetNd(NDArray value_nd, NDArray indices)
        {
            return new Operator("_full")
                .SetInput("lhs", this)
                .SetInput("rhs", value_nd)
                .SetInput("indices", indices)
                .SetParam("shape", this.Shape)
                .Invoke();
        }
    }
}