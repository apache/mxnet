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
using System;
using System.Collections.Generic;

namespace MxNet
{
    [Obsolete("Legacy API after MxNet v2, will be deprecated in v3", false)]
    public class NDContribApi
    {
        private static readonly List<string> ContribBoxNmsInFormatConvert = new List<string> {"center", "corner"};
        private static readonly List<string> ContribBoxNmsOutFormatConvert = new List<string> {"center", "corner"};

        private static readonly List<string> ContribBoxIouFormatConvert = new List<string> {"center", "corner"};

        private static readonly List<string> ContribDequantizeOutTypeConvert = new List<string> {"float32"};

        private static readonly List<string> ContribQuantizeOutTypeConvert = new List<string> {"int8", "uint8"};

        private static readonly List<string> ContribQuantizeV2OutTypeConvert =
            new List<string> {"auto", "int8", "uint8"};

        private static readonly List<string> ContribQuantizedActActTypeConvert =
            new List<string> {"relu", "sigmoid", "softrelu", "softsign", "tanh"};

        private static readonly List<string> ContribQuantizedConvCudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        private static readonly List<string> ContribQuantizedConvLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"};

        private static readonly List<string> ContribQuantizedPoolingPoolTypeConvert =
            new List<string> {"avg", "lp", "max", "sum"};

        private static readonly List<string> ContribQuantizedPoolingPoolingConventionConvert =
            new List<string> {"full", "same", "valid"};

        private static readonly List<string> ContribQuantizedPoolingLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"};

        private static readonly List<string> ContribRequantizeOutTypeConvert =
            new List<string> {"auto", "int8", "uint8"};

        private static readonly List<string> ContribDeformableconvolutionLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW"};

        /// <summary>
        ///     <para> </para>
        ///     <para>Applies a 2D adaptive average pooling over a 4D input with the shape of (NCHW).</para>
        ///     <para>The pooling kernel and stride sizes are automatically chosen for desired output sizes.</para>
        ///     <para> </para>
        ///     <para>- If a single integer is provided for output_size, the output size is \</para>
        ///     <para>  (N x C x output_size x output_size) for any input (NCHW).</para>
        ///     <para> </para>
        ///     <para>- If a tuple of integers (height, width) are provided for output_size, the output size is \</para>
        ///     <para>  (N x C x height x width) for any input (NCHW).</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\adaptive_avg_pooling.cc:L214</para>
        /// </summary>
        /// <param name="data">Input data</param>
        /// <param name="output_size">int (output size) or a tuple of int for output (height, width).</param>
        /// <returns>returns new symbol</returns>
        public NDArray AdaptiveAvgPooling2D(NDArray data, Shape output_size = null)
        {
            if (output_size == null) output_size = new Shape();

            return new Operator("_contrib_AdaptiveAvgPooling2D")
                .SetParam("output_size", output_size)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para>Perform 2D resizing (upsampling or downsampling) for 4D input using bilinear interpolation.</para>
        ///     <para> </para>
        ///     <para>Expected input is a 4 dimensional NDArray (NCHW) and the output</para>
        ///     <para>with the shape of (N x C x height x width). </para>
        ///     <para>The key idea of bilinear interpolation is to perform linear interpolation</para>
        ///     <para>first in one direction, and then again in the other direction. See the wikipedia of</para>
        ///     <para>`Bilinear interpolation  <https:// en.wikipedia.org/ wiki/ Bilinear_interpolation>`_</para>
        ///     <para>for more details.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\bilinear_resize.cc:L175</para>
        /// </summary>
        /// <param name="data">Input data</param>
        /// <param name="height">output height (required, but ignored if scale_height is defined)</param>
        /// <param name="width">output width (required, but ignored if scale_width is defined)</param>
        /// <param name="scale_height">sampling scale of the height (optional, ignores height if defined)</param>
        /// <param name="scale_width">sampling scale of the scale_width (optional, ignores width if defined)</param>
        /// <returns>returns new symbol</returns>
        public NDArray BilinearResize2D(NDArray data, int height = 1, int width = 1, float? scale_height = null,
            float? scale_width = null)
        {
            return new Operator("_contrib_BilinearResize2D")
                .SetParam("height", height)
                .SetParam("width", width)
                .SetParam("scale_height", scale_height)
                .SetParam("scale_width", scale_width)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para>Given an n-d NDArray data, and a 1-d NDArray index,</para>
        ///     <para>the operator produces an un-predeterminable shaped n-d NDArray out,</para>
        ///     <para>which stands for the rows in x where the corresonding element in index is non-zero.</para>
        ///     <para> </para>
        ///     <para>>>> data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])</para>
        ///     <para>>>> index = mx.nd.array([0, 1, 0])</para>
        ///     <para>>>> out = mx.nd.contrib.boolean_mask(data, index)</para>
        ///     <para>>>> out</para>
        ///     <para> </para>
        ///     <para>[[4. 5. 6.]]</para>
        ///     <para>
        ///         <NDArray 1 x3 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\boolean_mask.cc:L199</para>
        /// </summary>
        /// <param name="data">Data</param>
        /// <param name="index">Mask</param>
        /// <param name="axis">An integer that represents the axis in NDArray to mask from.</param>
        /// <returns>returns new symbol</returns>
        public NDArray BooleanMask(NDArray data, NDArray index, int axis = 0)
        {
            return new Operator("_contrib_boolean_mask")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("index", index)
                .Invoke();
        }

        /// <summary>
        ///     <para>Apply non-maximum suppression to input.</para>
        ///     <para> </para>
        ///     <para>The output will be sorted in descending order according to `score`. Boxes with</para>
        ///     <para>overlaps larger than `overlap_thresh`, smaller scores and background boxes</para>
        ///     <para>will be removed and filled with -1, the corresponding position will be recorded</para>
        ///     <para>for backward propogation.</para>
        ///     <para> </para>
        ///     <para>During back-propagation, the gradient will be copied to the original</para>
        ///     <para>position according to the input index. For positions that have been suppressed,</para>
        ///     <para>the in_grad will be assigned 0.</para>
        ///     <para>In summary, gradients are sticked to its boxes, will either be moved or discarded</para>
        ///     <para>according to its original index in input.</para>
        ///     <para> </para>
        ///     <para>Input requirements::</para>
        ///     <para> </para>
        ///     <para>  1. Input tensor have at least 2 dimensions, (n, k), any higher dims will be regarded</para>
        ///     <para>  as batch, e.g. (a, b, c, d, n, k) == (a*b*c*d, n, k)</para>
        ///     <para>  2. n is the number of boxes in each batch</para>
        ///     <para>  3. k is the width of each box item.</para>
        ///     <para> </para>
        ///     <para>By default, a box is [id, score, xmin, ymin, xmax, ymax, ...],</para>
        ///     <para>additional elements are allowed.</para>
        ///     <para> </para>
        ///     <para>- `id_index`: optional, use -1 to ignore, useful if `force_suppress=False`, which means</para>
        ///     <para>  we will skip highly overlapped boxes if one is `apple` while the other is `car`.</para>
        ///     <para> </para>
        ///     <para>- `background_id`: optional, default=-1, class id for background boxes, useful</para>
        ///     <para>  when `id_index >= 0` which means boxes with background id will be filtered before nms.</para>
        ///     <para> </para>
        ///     <para>- `coord_start`: required, default=2, the starting index of the 4 coordinates.</para>
        ///     <para>  Two formats are supported:</para>
        ///     <para> </para>
        ///     <para>    - `corner`: [xmin, ymin, xmax, ymax]</para>
        ///     <para> </para>
        ///     <para>    - `center`: [x, y, width, height]</para>
        ///     <para> </para>
        ///     <para>- `score_index`: required, default=1, box score/confidence.</para>
        ///     <para>  When two boxes overlap IOU > `overlap_thresh`, the one with smaller score will be suppressed.</para>
        ///     <para> </para>
        ///     <para>- `in_format` and `out_format`: default='corner', specify in/out box formats.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],</para>
        ///     <para>       [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]</para>
        ///     <para>  box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0,</para>
        ///     <para>      force_suppress=True, in_format='corner', out_typ='corner') =</para>
        ///     <para>      [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],</para>
        ///     <para>       [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]</para>
        ///     <para>  out_grad = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],</para>
        ///     <para>              [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]</para>
        ///     <para>  # exe.backward</para>
        ///     <para>  in_grad = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0, 0],</para>
        ///     <para>             [0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\bounding_box.cc:L93</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="overlap_thresh">Overlapping(IoU) threshold to suppress object with smaller score.</param>
        /// <param name="valid_thresh">Filter input boxes to those whose scores greater than valid_thresh.</param>
        /// <param name="topk">Apply nms to topk boxes with descending scores, -1 to no restriction.</param>
        /// <param name="coord_start">Start index of the consecutive 4 coordinates.</param>
        /// <param name="score_index">Index of the scores/confidence of boxes.</param>
        /// <param name="id_index">Optional, index of the class categories, -1 to disable.</param>
        /// <param name="background_id">Optional, id of the background class which will be ignored in nms.</param>
        /// <param name="force_suppress">
        ///     Optional, if set false and id_index is provided, nms will only apply to boxes belongs to
        ///     the same category
        /// </param>
        /// <param name="in_format">
        ///     The input box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax],
        ///     "center" means boxes are encodes as [x, y, width, height].
        /// </param>
        /// <param name="out_format">
        ///     The output box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax],
        ///     "center" means boxes are encodes as [x, y, width, height].
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray BoxNms(NDArray data, float overlap_thresh = 0.5f, float valid_thresh = 0f, int topk = -1,
            int coord_start = 2, int score_index = 1, int id_index = -1, int background_id = -1,
            bool force_suppress = false, ContribBoxNmsInFormat in_format = ContribBoxNmsInFormat.Corner,
            ContribBoxNmsOutFormat out_format = ContribBoxNmsOutFormat.Corner)
        {
            return new Operator("_contrib_box_nms")
                .SetParam("overlap_thresh", overlap_thresh)
                .SetParam("valid_thresh", valid_thresh)
                .SetParam("topk", topk)
                .SetParam("coord_start", coord_start)
                .SetParam("score_index", score_index)
                .SetParam("id_index", id_index)
                .SetParam("background_id", background_id)
                .SetParam("force_suppress", force_suppress)
                .SetParam("in_format",
                    MxUtil.EnumToString<ContribBoxNmsInFormat>(in_format, ContribBoxNmsInFormatConvert))
                .SetParam("out_format",
                    MxUtil.EnumToString<ContribBoxNmsOutFormat>(out_format, ContribBoxNmsOutFormatConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public NDArray BoxNonMaximumSupression(NDArray data, float overlap_thresh = 0.5f, float valid_thresh = 0f, int topk = -1,
           int coord_start = 2, int score_index = 1, int id_index = -1, int background_id = -1,
           bool force_suppress = false, ContribBoxNmsInFormat in_format = ContribBoxNmsInFormat.Corner,
           ContribBoxNmsOutFormat out_format = ContribBoxNmsOutFormat.Corner)
        {
            return new Operator("_contrib_box_non_maximum_suppression")
                .SetParam("overlap_thresh", overlap_thresh)
                .SetParam("valid_thresh", valid_thresh)
                .SetParam("topk", topk)
                .SetParam("coord_start", coord_start)
                .SetParam("score_index", score_index)
                .SetParam("id_index", id_index)
                .SetParam("background_id", background_id)
                .SetParam("force_suppress", force_suppress)
                .SetParam("in_format",
                    MxUtil.EnumToString<ContribBoxNmsInFormat>(in_format, ContribBoxNmsInFormatConvert))
                .SetParam("out_format",
                    MxUtil.EnumToString<ContribBoxNmsOutFormat>(out_format, ContribBoxNmsOutFormatConvert))
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Bounding box overlap of two arrays.</para>
        ///     <para>  The overlap is defined as Intersection-over-Union, aka, IOU.</para>
        ///     <para>  - lhs: (a_1, a_2, ..., a_n, 4) array</para>
        ///     <para>  - rhs: (b_1, b_2, ..., b_n, 4) array</para>
        ///     <para>  - output: (a_1, a_2, ..., a_n, b_1, b_2, ..., b_n) array</para>
        ///     <para> </para>
        ///     <para>  Note::</para>
        ///     <para> </para>
        ///     <para>    Zero gradients are back-propagated in this op for now.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>    x = [[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]]</para>
        ///     <para>    y = [[0.25, 0.25, 0.75, 0.75]]</para>
        ///     <para>    box_iou(x, y, format='corner') = [[0.1428], [0.1428]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\bounding_box.cc:L134</para>
        /// </summary>
        /// <param name="lhs">The first input</param>
        /// <param name="rhs">The second input</param>
        /// <param name="format">
        ///     The box encoding type.  "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center"
        ///     means boxes are encodes as [x, y, width, height].
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray BoxIou(NDArray lhs, NDArray rhs, ContribBoxIouFormat format = ContribBoxIouFormat.Corner)
        {
            return new Operator("_contrib_box_iou")
                .SetParam("format", MxUtil.EnumToString<ContribBoxIouFormat>(format, ContribBoxIouFormatConvert))
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        /// <summary>
        ///     <para>Compute bipartite matching.</para>
        ///     <para>  The matching is performed on score matrix with shape [B, N, M]</para>
        ///     <para>  - B: batch_size</para>
        ///     <para>  - N: number of rows to match</para>
        ///     <para>  - M: number of columns as reference to be matched against.</para>
        ///     <para> </para>
        ///     <para>  Returns:</para>
        ///     <para>  x : matched column indices. -1 indicating non-matched elements in rows.</para>
        ///     <para>  y : matched row indices.</para>
        ///     <para> </para>
        ///     <para>  Note::</para>
        ///     <para> </para>
        ///     <para>    Zero gradients are back-propagated in this op for now.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>    s = [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]</para>
        ///     <para>    x, y = bipartite_matching(x, threshold=1e-12, is_ascend=False)</para>
        ///     <para>    x = [1, -1, 0]</para>
        ///     <para>    y = [2, 0]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\bounding_box.cc:L180</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="is_ascend">Use ascend order for scores instead of descending. Please set threshold accordingly.</param>
        /// <param name="threshold">
        ///     Ignore matching when score
        ///     < thresh, if is_ascend= false, or ignore score> thresh, if is_ascend=true.
        /// </param>
        /// <param name="topk">Limit the number of matches to topk, set -1 for no limit</param>
        /// <returns>returns new symbol</returns>
        public NDArray BipartiteMatching(NDArray data, float threshold, bool is_ascend = false, int topk = -1)
        {
            return new Operator("_contrib_bipartite_matching")
                .SetParam("is_ascend", is_ascend)
                .SetParam("threshold", threshold)
                .SetParam("topk", topk)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator samples sub-graphs from a csr graph via an</para>
        ///     <para>uniform probability. The operator is designed for DGL.</para>
        ///     <para> </para>
        ///     <para>The operator outputs three sets of NDArrays to represent the sampled results</para>
        ///     <para>(the number of NDArrays in each set is the same as the number of seed NDArrays):</para>
        ///     <para>1) a set of 1D NDArrays containing the sampled vertices, 2) a set of CSRNDArrays representing</para>
        ///     <para>the sampled edges, 3) a set of 1D NDArrays indicating the layer where a vertex is sampled.</para>
        ///     <para>The first set of 1D NDArrays have a length of max_num_vertices+1. The last element in an NDArray</para>
        ///     <para>indicate the acutal number of vertices in a subgraph. The third set of NDArrays have a length</para>
        ///     <para>of max_num_vertices, and the valid number of vertices is the same as the ones in the first set.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>  shape = (5, 5)</para>
        ///     <para>  data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)</para>
        ///     <para>  indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)</para>
        ///     <para>  indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)</para>
        ///     <para>  a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)</para>
        ///     <para>  a.asnumpy()</para>
        ///     <para>  seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)</para>
        ///     <para>
        ///         out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1, num_neighbor=2,
        ///         max_num_vertices=5)
        ///     </para>
        ///     <para> </para>
        ///     <para>  out[0]</para>
        ///     <para>  [0 1 2 3 4 5]</para>
        ///     <para>
        ///         <NDArray 6 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para>  out[1].asnumpy()</para>
        ///     <para>  array([[ 0,  1,  0,  3,  0],</para>
        ///     <para>         [ 5,  0,  0,  7,  0],</para>
        ///     <para>         [ 9,  0,  0, 11,  0],</para>
        ///     <para>         [13,  0, 15,  0,  0],</para>
        ///     <para>         [17,  0, 19,  0,  0]])</para>
        ///     <para> </para>
        ///     <para>  out[2]</para>
        ///     <para>  [0 0 0 0 0]</para>
        ///     <para>
        ///         <NDArray 5 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L784</para>
        /// </summary>
        /// <param name="csr_matrix">csr matrix</param>
        /// <param name="seed_arrays">seed vertices</param>
        /// <param name="num_args">Number of input NDArray.</param>
        /// <param name="num_hops">Number of hops.</param>
        /// <param name="num_neighbor">Number of neighbor.</param>
        /// <param name="max_num_vertices">Max number of vertices.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DglCsrNeighborUniformSample(NDArray csr_matrix, NDArrayList seed_arrays, int num_args,
            int num_hops = 1, int num_neighbor = 2, int max_num_vertices = 100)
        {
            return new Operator("_contrib_dgl_csr_neighbor_uniform_sample")
                .SetParam("num_args", num_args)
                .SetParam("num_hops", num_hops)
                .SetParam("num_neighbor", num_neighbor)
                .SetParam("max_num_vertices", max_num_vertices)
                .SetInput("csr_matrix", csr_matrix)
                .SetInput(seed_arrays)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator samples sub-graph from a csr graph via an</para>
        ///     <para>non-uniform probability. The operator is designed for DGL.</para>
        ///     <para> </para>
        ///     <para>The operator outputs four sets of NDArrays to represent the sampled results</para>
        ///     <para>(the number of NDArrays in each set is the same as the number of seed NDArrays):</para>
        ///     <para>1) a set of 1D NDArrays containing the sampled vertices, 2) a set of CSRNDArrays representing</para>
        ///     <para>the sampled edges, 3) a set of 1D NDArrays with the probability that vertices are sampled,</para>
        ///     <para>4) a set of 1D NDArrays indicating the layer where a vertex is sampled.</para>
        ///     <para>The first set of 1D NDArrays have a length of max_num_vertices+1. The last element in an NDArray</para>
        ///     <para>indicate the acutal number of vertices in a subgraph. The third and fourth set of NDArrays have a length</para>
        ///     <para>of max_num_vertices, and the valid number of vertices is the same as the ones in the first set.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>  shape = (5, 5)</para>
        ///     <para>  prob = mx.nd.array([0.9, 0.8, 0.2, 0.4, 0.1], dtype=np.float32)</para>
        ///     <para>  data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)</para>
        ///     <para>  indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)</para>
        ///     <para>  indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)</para>
        ///     <para>  a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)</para>
        ///     <para>  seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)</para>
        ///     <para>
        ///         out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1,
        ///         num_neighbor=2, max_num_vertices=5)
        ///     </para>
        ///     <para> </para>
        ///     <para>  out[0]</para>
        ///     <para>  [0 1 2 3 4 5]</para>
        ///     <para>
        ///         <NDArray 6 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para>  out[1].asnumpy()</para>
        ///     <para>  array([[ 0,  1,  2,  0,  0],</para>
        ///     <para>         [ 5,  0,  6,  0,  0],</para>
        ///     <para>         [ 9, 10,  0,  0,  0],</para>
        ///     <para>         [13, 14,  0,  0,  0],</para>
        ///     <para>         [ 0, 18, 19,  0,  0]])</para>
        ///     <para> </para>
        ///     <para>  out[2]</para>
        ///     <para>  [0.9 0.8 0.2 0.4 0.1]</para>
        ///     <para>
        ///         <NDArray 5 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para>  out[3]</para>
        ///     <para>  [0 0 0 0 0]</para>
        ///     <para>
        ///         <NDArray 5 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L883</para>
        /// </summary>
        /// <param name="csr_matrix">csr matrix</param>
        /// <param name="probability">probability vector</param>
        /// <param name="seed_arrays">seed vertices</param>
        /// <param name="num_args">Number of input NDArray.</param>
        /// <param name="num_hops">Number of hops.</param>
        /// <param name="num_neighbor">Number of neighbor.</param>
        /// <param name="max_num_vertices">Max number of vertices.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DglCsrNeighborNonUniformSample(NDArray csr_matrix, NDArray probability, NDArrayList seed_arrays,
            int num_args, int num_hops = 1, int num_neighbor = 2, int max_num_vertices = 100)
        {
            return new Operator("_contrib_dgl_csr_neighbor_non_uniform_sample")
                .SetParam("num_args", num_args)
                .SetParam("num_hops", num_hops)
                .SetParam("num_neighbor", num_neighbor)
                .SetParam("max_num_vertices", max_num_vertices)
                .SetInput("csr_matrix", csr_matrix)
                .SetInput("probability", probability)
                .SetInput(seed_arrays)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator constructs an induced subgraph for</para>
        ///     <para>a given set of vertices from a graph. The operator accepts multiple</para>
        ///     <para>sets of vertices as input. For each set of vertices, it returns a pair</para>
        ///     <para>of CSR matrices if return_mapping is True: the first matrix contains edges</para>
        ///     <para>with new edge Ids, the second matrix contains edges with the original</para>
        ///     <para>edge Ids.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>     x=[[1, 0, 0, 2],</para>
        ///     <para>       [3, 0, 4, 0],</para>
        ///     <para>       [0, 5, 0, 0],</para>
        ///     <para>       [0, 6, 7, 0]]</para>
        ///     <para>     v = [0, 1, 2]</para>
        ///     <para>     dgl_subgraph(x, v, return_mapping=True) =</para>
        ///     <para>       [[1, 0, 0],</para>
        ///     <para>        [2, 0, 3],</para>
        ///     <para>        [0, 4, 0]],</para>
        ///     <para>       [[1, 0, 0],</para>
        ///     <para>        [3, 0, 4],</para>
        ///     <para>        [0, 5, 0]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L1140</para>
        /// </summary>
        /// <param name="graph">Input graph where we sample vertices.</param>
        /// <param name="data">The input arrays that include data arrays and states.</param>
        /// <param name="num_args">Number of input arguments, including all symbol inputs.</param>
        /// <param name="return_mapping">Return mapping of vid and eid between the subgraph and the parent graph.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DglSubgraph(NDArray graph, NDArrayList data, int num_args, bool return_mapping)
        {
            return new Operator("_contrib_dgl_subgraph")
                .SetParam("num_args", num_args)
                .SetParam("return_mapping", return_mapping)
                .SetInput("graph", graph)
                .SetInput(data)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator implements the edge_id function for a graph</para>
        ///     <para>stored in a CSR matrix (the value of the CSR stores the edge Id of the graph).</para>
        ///     <para>output[i] = input[u[i], v[i]] if there is an edge between u[i] and v[i]],</para>
        ///     <para>otherwise output[i] will be -1. Both u and v should be 1D vectors.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>      x = [[ 1, 0, 0 ],</para>
        ///     <para>           [ 0, 2, 0 ],</para>
        ///     <para>           [ 0, 0, 3 ]]</para>
        ///     <para>      u = [ 0, 0, 1, 1, 2, 2 ]</para>
        ///     <para>      v = [ 0, 1, 1, 2, 0, 2 ]</para>
        ///     <para>      edge_id(x, u, v) = [ 1, -1, 2, -1, -1, 3 ]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``edge_id`` output depends on storage types of inputs</para>
        ///     <para>  - edge_id(csr, default, default) = default</para>
        ///     <para>  - default and rsp inputs are not supported</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L1321</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="u">u ndarray</param>
        /// <param name="v">v ndarray</param>
        /// <returns>returns new symbol</returns>
        public NDArray EdgeId(NDArray data, NDArray u, NDArray v)
        {
            return new Operator("_contrib_edge_id")
                .SetInput("data", data)
                .SetInput("u", u)
                .SetInput("v", v)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator converts a CSR matrix whose values are edge Ids</para>
        ///     <para>to an adjacency matrix whose values are ones. The output CSR matrix always has</para>
        ///     <para>the data value of float32.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1, 0, 0 ],</para>
        ///     <para>       [ 0, 2, 0 ],</para>
        ///     <para>       [ 0, 0, 3 ]]</para>
        ///     <para>  dgl_adjacency(x) =</para>
        ///     <para>      [[ 1, 0, 0 ],</para>
        ///     <para>       [ 0, 1, 0 ],</para>
        ///     <para>       [ 0, 0, 1 ]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L1393</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <returns>returns new symbol</returns>
        public NDArray DglAdjacency(NDArray data)
        {
            return new Operator("_contrib_dgl_adjacency")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator compacts a CSR matrix generated by</para>
        ///     <para>dgl_csr_neighbor_uniform_sample and dgl_csr_neighbor_non_uniform_sample.</para>
        ///     <para>The CSR matrices generated by these two operators may have many empty</para>
        ///     <para>rows at the end and many empty columns. This operator removes these</para>
        ///     <para>empty rows and empty columns.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>   .. code:: python</para>
        ///     <para> </para>
        ///     <para>  shape = (5, 5)</para>
        ///     <para>  data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)</para>
        ///     <para>  indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)</para>
        ///     <para>  indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)</para>
        ///     <para>  a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)</para>
        ///     <para>  seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)</para>
        ///     <para>  out = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(a, seed, num_args=2, num_hops=1,</para>
        ///     <para>          num_neighbor=2, max_num_vertices=6)</para>
        ///     <para>  subg_v = out[0]</para>
        ///     <para>  subg = out[1]</para>
        ///     <para>  compact = mx.nd.contrib.dgl_graph_compact(subg, subg_v,</para>
        ///     <para>          graph_sizes=(subg_v[-1].asnumpy()[0]), return_mapping=False)</para>
        ///     <para> </para>
        ///     <para>  compact.asnumpy()</para>
        ///     <para>  array([[0, 0, 0, 1, 0],</para>
        ///     <para>         [2, 0, 3, 0, 0],</para>
        ///     <para>         [0, 4, 0, 0, 5],</para>
        ///     <para>         [0, 6, 0, 0, 7],</para>
        ///     <para>         [8, 9, 0, 0, 0]])</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\dgl_graph.cc:L1582</para>
        /// </summary>
        /// <param name="graph_data">Input graphs and input vertex Ids.</param>
        /// <param name="num_args">Number of input arguments.</param>
        /// <param name="return_mapping">Return mapping of vid and eid between the subgraph and the parent graph.</param>
        /// <param name="graph_sizes">the number of vertices in each graph.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DglGraphCompact(NDArrayList graph_data, int num_args, bool return_mapping,
            Tuple<double> graph_sizes)
        {
            return new Operator("_contrib_dgl_graph_compact")
                .SetParam("num_args", num_args)
                .SetParam("return_mapping", return_mapping)
                .SetParam("graph_sizes", graph_sizes)
                .SetInput(graph_data)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operator implements the gradient multiplier function.</para>
        ///     <para>In forward pass it acts as an identity transform. During backpropagation it</para>
        ///     <para>multiplies the gradient from the subsequent level by a scalar factor lambda and passes it to</para>
        ///     <para>the preceding layer.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\gradient_multiplier_op.cc:L78</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="scalar">lambda multiplier</param>
        /// <returns>returns new symbol</returns>
        public NDArray Gradientmultiplier(NDArray data, float scalar)
        {
            return new Operator("_contrib_gradientmultiplier")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Copies the elements of a `new_tensor` into the `old_tensor`.</para>
        ///     <para> </para>
        ///     <para>This operator copies the elements by selecting the indices in the order given in `index`.</para>
        ///     <para>The output will be a new tensor containing the rest elements of old tensor and</para>
        ///     <para>the copied elements of new tensor.</para>
        ///     <para>For example, if `index[i] == j`, then the `i` th row of `new_tensor` is copied to the</para>
        ///     <para>`j` th row of output.</para>
        ///     <para> </para>
        ///     <para>The `index` must be a vector and it must have the same size with the `0` th dimension of</para>
        ///     <para>`new_tensor`. Also, the `0` th dimension of old_tensor must `>=` the `0` th dimension of</para>
        ///     <para>`new_tensor`, or an error will be raised.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>    x = mx.nd.zeros((5,3))</para>
        ///     <para>    t = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])</para>
        ///     <para>    index = mx.nd.array([0,4,2])</para>
        ///     <para> </para>
        ///     <para>    mx.nd.contrib.index_copy(x, index, t)</para>
        ///     <para> </para>
        ///     <para>    [[1. 2. 3.]</para>
        ///     <para>     [0. 0. 0.]</para>
        ///     <para>     [7. 8. 9.]</para>
        ///     <para>     [0. 0. 0.]</para>
        ///     <para>     [4. 5. 6.]]</para>
        ///     <para>
        ///         <NDArray 5 x3 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\index_copy.cc:L183</para>
        /// </summary>
        /// <param name="old_tensor">Old tensor</param>
        /// <param name="index_vector">Index vector</param>
        /// <param name="new_tensor">New tensor to be copied</param>
        /// <returns>returns new symbol</returns>
        public NDArray IndexCopy(NDArray old_tensor, NDArray index_vector, NDArray new_tensor)
        {
            return new Operator("_contrib_index_copy")
                .SetInput("old_tensor", old_tensor)
                .SetInput("index_vector", index_vector)
                .SetInput("new_tensor", new_tensor)
                .Invoke();
        }

        /// <summary>
        ///     <para>Number of stored values for a sparse tensor, including explicit zeros.</para>
        ///     <para> </para>
        ///     <para>This operator only supports CSR matrix on CPU.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\nnz.cc:L177</para>
        /// </summary>
        /// <param name="data">Input</param>
        /// <param name="axis">Select between the number of values across the whole matrix, in each column, or in each row.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Getnnz(NDArray data, int? axis = null)
        {
            return new Operator("_contrib_getnnz")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Update function for Group AdaGrad optimizer.</para>
        ///     <para> </para>
        ///     <para>Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,</para>
        ///     <para>and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but</para>
        ///     <para>uses only a single learning rate for every row of the parameter array.</para>
        ///     <para> </para>
        ///     <para>Updates are applied by::</para>
        ///     <para> </para>
        ///     <para>    grad = clip(grad * rescale_grad, clip_gradient)</para>
        ///     <para>    history += mean(square(grad), axis=1, keepdims=True)</para>
        ///     <para>    div = grad / sqrt(history + float_stable_eps)</para>
        ///     <para>    weight -= div * lr</para>
        ///     <para> </para>
        ///     <para>Weights are updated lazily if the gradient is sparse.</para>
        ///     <para> </para>
        ///     <para>Note that non-zero values for the weight decay option are not supported.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\optimizer_op.cc:L71</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="history">History</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="epsilon">Epsilon for numerical stability</param>
        /// <returns>returns new symbol</returns>
        public NDArray GroupAdagradUpdate(NDArray weight, NDArray grad, NDArray history, float lr,
            float rescale_grad = 1f, float clip_gradient = -1f, float epsilon = 1e-05f)
        {
            return new Operator("_contrib_group_adagrad_update")
                .SetParam("lr", lr)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("epsilon", epsilon)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("history", history)
                .Invoke();
        }

        /// <summary>
        ///     <para>This operators implements the quadratic function.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>    f(x) = ax^2+bx+c</para>
        ///     <para> </para>
        ///     <para>where :math:`x` is an input tensor and all operations</para>
        ///     <para>in the function are element-wise.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[1, 2], [3, 4]]</para>
        ///     <para>  y = quadratic(data=x, a=1, b=2, c=3)</para>
        ///     <para>  y = [[6, 11], [18, 27]]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``quadratic`` output depends on storage types of inputs</para>
        ///     <para>  - quadratic(csr, a, b, 0) = csr</para>
        ///     <para>  - quadratic(default, a, b, c) = default</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\quadratic_op.cc:L50</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="a">Coefficient of the quadratic term in the quadratic function.</param>
        /// <param name="b">Coefficient of the linear term in the quadratic function.</param>
        /// <param name="c">Constant term in the quadratic function.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Quadratic(NDArray data, float a = 0f, float b = 0f, float c = 0f)
        {
            return new Operator("_contrib_quadratic")
                .SetParam("a", a)
                .SetParam("b", b)
                .SetParam("c", c)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para>This operator takes a 4D feature map as an input array and region proposals as `rois`,</para>
        ///     <para>then align the feature map over sub-regions of input and produces a fixed-sized output array.</para>
        ///     <para>This operator is typically used in Faster R-CNN & Mask R-CNN networks.</para>
        ///     <para> </para>
        ///     <para>Different from ROI pooling, ROI Align removes the harsh quantization, properly aligning</para>
        ///     <para>the extracted features with the input. RoIAlign computes the value of each sampling point</para>
        ///     <para>by bilinear interpolation from the nearby grid points on the feature map. No quantization is</para>
        ///     <para>performed on any coordinates involved in the RoI, its bins, or the sampling points.</para>
        ///     <para>Bilinear interpolation is used to compute the exact values of the</para>
        ///     <para>input features at four regularly sampled locations in each RoI bin.</para>
        ///     <para>Then the feature map can be aggregated by avgpooling.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>References</para>
        ///     <para>----------</para>
        ///     <para> </para>
        ///     <para>He, Kaiming, et al. "Mask R-CNN." ICCV, 2017</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\roi_align.cc:L538</para>
        /// </summary>
        /// <param name="data">Input data to the pooling operator, a 4D Feature maps</param>
        /// <param name="rois">Bounding box coordinates, a 2D array</param>
        /// <param name="pooled_size">ROI Align output roi feature map height and width: (h, w)</param>
        /// <param name="spatial_scale">
        ///     Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        ///     of total stride in convolutional layers
        /// </param>
        /// <param name="sample_ratio">Optional sampling ratio of ROI align, using adaptive size by default.</param>
        /// <param name="position_sensitive">
        ///     Whether to perform position-sensitive RoI pooling. PSRoIPooling is first proposaled by
        ///     R-FCN and it can reduce the input channels by ph*pw times, where (ph, pw) is the pooled_size
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray ROIAlign(NDArray data, NDArray rois, Shape pooled_size, float spatial_scale,
            int sample_ratio = -1, bool position_sensitive = false)
        {
            return new Operator("_contrib_ROIAlign")
                .SetParam("pooled_size", pooled_size)
                .SetParam("spatial_scale", spatial_scale)
                .SetParam("sample_ratio", sample_ratio)
                .SetParam("position_sensitive", position_sensitive)
                .SetInput("data", data)
                .SetInput("rois", rois)
                .Invoke();
        }

        public NDArray SyncBatchNorm(NDArray data, NDArray gamma, NDArray beta, NDArray moving_mean, NDArray moving_var,
            string key, float eps = 0.001f, float momentum = 0.9f, bool fix_gamma = true, bool use_global_stats = false,
            bool output_mean_var = false, int ndev = 1)
        {
            return new Operator("_contrib_SyncBatchNorm")
                .SetParam("eps", eps)
                .SetParam("momentum", momentum)
                .SetParam("fix_gamma", fix_gamma)
                .SetParam("use_global_stats", use_global_stats)
                .SetParam("output_mean_var", output_mean_var)
                .SetParam("ndev", ndev)
                .SetParam("key", key)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .SetInput("moving_mean", moving_mean)
                .SetInput("moving_var", moving_var)
                .Invoke();
        }

        /// <summary>
        ///     <para>Rescale the input by the square root of the channel dimension.</para>
        ///     <para> </para>
        ///     <para>   out = data / sqrt(data.shape[-1])</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\transformer.cc:L38</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DivSqrtDim(NDArray data)
        {
            return new Operator("_contrib_div_sqrt_dim")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Dequantize the input tensor into a float tensor.</para>
        ///     <para>min_range and max_range are scalar floats that specify the range for</para>
        ///     <para>the output data.</para>
        ///     <para> </para>
        ///     <para>When input data type is `uint8`, the output is calculated using the following equation:</para>
        ///     <para> </para>
        ///     <para>`out[i] = in[i] * (max_range - min_range) / 255.0`,</para>
        ///     <para> </para>
        ///     <para>When input data type is `int8`, the output is calculate using the following equation</para>
        ///     <para>by keep zero centered for the quantized value:</para>
        ///     <para> </para>
        ///     <para>`out[i] = in[i] * MaxAbs(min_range, max_range) / 127.0`,</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\dequantize.cc:L83</para>
        /// </summary>
        /// <param name="data">A ndarray/symbol of type `uint8`</param>
        /// <param name="min_range">The minimum scalar value possibly produced for the input in float32</param>
        /// <param name="max_range">The maximum scalar value possibly produced for the input in float32</param>
        /// <param name="out_type">Output data type.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Dequantize(NDArray data, NDArray min_range, NDArray max_range,
            ContribDequantizeOutType out_type = ContribDequantizeOutType.Float32)
        {
            return new Operator("_contrib_dequantize")
                .SetParam("out_type",
                    MxUtil.EnumToString<ContribDequantizeOutType>(out_type, ContribDequantizeOutTypeConvert))
                .SetInput("data", data)
                .SetInput("min_range", min_range)
                .SetInput("max_range", max_range)
                .Invoke();
        }

        /// <summary>
        ///     <para>Quantize a input tensor from float to `out_type`,</para>
        ///     <para>with user-specified `min_range` and `max_range`.</para>
        ///     <para> </para>
        ///     <para>min_range and max_range are scalar floats that specify the range for</para>
        ///     <para>the input data.</para>
        ///     <para> </para>
        ///     <para>When out_type is `uint8`, the output is calculated using the following equation:</para>
        ///     <para> </para>
        ///     <para>`out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range) + 0.5`,</para>
        ///     <para> </para>
        ///     <para>where `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`.</para>
        ///     <para> </para>
        ///     <para>When out_type is `int8`, the output is calculate using the following equation</para>
        ///     <para>by keep zero centered for the quantized value:</para>
        ///     <para> </para>
        ///     <para>`out[i] = sign(in[i]) * min(abs(in[i] * scale + 0.5f, quantized_range)`,</para>
        ///     <para> </para>
        ///     <para>where</para>
        ///     <para>`quantized_range = MinAbs(max(int8), min(int8))` and</para>
        ///     <para>`scale = quantized_range / MaxAbs(min_range, max_range).`</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propagation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantize.cc:L74</para>
        /// </summary>
        /// <param name="data">A ndarray/symbol of type `float32`</param>
        /// <param name="min_range">The minimum scalar value possibly produced for the input</param>
        /// <param name="max_range">The maximum scalar value possibly produced for the input</param>
        /// <param name="out_type">Output data type.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Quantize(NDArray data, NDArray min_range, NDArray max_range,
            ContribQuantizeOutType out_type = ContribQuantizeOutType.Uint8)
        {
            return new Operator("_contrib_quantize")
                .SetParam("out_type",
                    MxUtil.EnumToString<ContribQuantizeOutType>(out_type, ContribQuantizeOutTypeConvert))
                .SetInput("data", data)
                .SetInput("min_range", min_range)
                .SetInput("max_range", max_range)
                .Invoke();
        }

        /// <summary>
        ///     <para>Quantize a input tensor from float to `out_type`,</para>
        ///     <para>with user-specified `min_calib_range` and `max_calib_range` or the input range collected at runtime.</para>
        ///     <para> </para>
        ///     <para>Output `min_range` and `max_range` are scalar floats that specify the range for the input data.</para>
        ///     <para> </para>
        ///     <para>When out_type is `uint8`, the output is calculated using the following equation:</para>
        ///     <para> </para>
        ///     <para>`out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range) + 0.5`,</para>
        ///     <para> </para>
        ///     <para>where `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`.</para>
        ///     <para> </para>
        ///     <para>When out_type is `int8`, the output is calculate using the following equation</para>
        ///     <para>by keep zero centered for the quantized value:</para>
        ///     <para> </para>
        ///     <para>`out[i] = sign(in[i]) * min(abs(in[i] * scale + 0.5f, quantized_range)`,</para>
        ///     <para> </para>
        ///     <para>where</para>
        ///     <para>`quantized_range = MinAbs(max(int8), min(int8))` and</para>
        ///     <para>`scale = quantized_range / MaxAbs(min_range, max_range).`</para>
        ///     <para> </para>
        ///     <para>When out_type is `auto`, the output type is automatically determined by min_calib_range if presented.</para>
        ///     <para>If min_calib_range < 0.0f, the output type will be int8, otherwise will be uint8.</para>
        ///     <para>If min_calib_range isn't presented, the output type will be int8.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propagation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantize_v2.cc:L92</para>
        /// </summary>
        /// <param name="data">A ndarray/symbol of type `float32`</param>
        /// <param name="out_type">
        ///     Output data type. `auto` can be specified to automatically determine output type according to
        ///     min_calib_range.
        /// </param>
        /// <param name="min_calib_range">
        ///     The minimum scalar value in the form of float32. If present, it will be used to quantize
        ///     the fp32 data into int8 or uint8.
        /// </param>
        /// <param name="max_calib_range">
        ///     The maximum scalar value in the form of float32. If present, it will be used to quantize
        ///     the fp32 data into int8 or uint8.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizeV2(NDArray data, ContribQuantizeV2OutType out_type = ContribQuantizeV2OutType.Int8,
            float? min_calib_range = null, float? max_calib_range = null)
        {
            return new Operator("_contrib_quantize_v2")
                .SetParam("out_type",
                    MxUtil.EnumToString<ContribQuantizeV2OutType>(out_type, ContribQuantizeV2OutTypeConvert))
                .SetParam("min_calib_range", min_calib_range)
                .SetParam("max_calib_range", max_calib_range)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Activation operator for input and output data type of int8.</para>
        ///     <para>The input and output data comes with min and max thresholds for quantizing</para>
        ///     <para>the float32 data into int8.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>     This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para>     This operator only supports `relu`</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantized_activation.cc:L91</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="min_data">Minimum value of data.</param>
        /// <param name="max_data">Maximum value of data.</param>
        /// <param name="act_type">Activation function to be applied.</param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedAct(NDArray data, NDArray min_data, NDArray max_data,
            ContribQuantizedActActType act_type)
        {
            return new Operator("_contrib_quantized_act")
                .SetParam("act_type",
                    MxUtil.EnumToString<ContribQuantizedActActType>(act_type, ContribQuantizedActActTypeConvert))
                .SetInput("data", data)
                .SetInput("min_data", min_data)
                .SetInput("max_data", max_data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Joins input arrays along a given axis.</para>
        ///     <para> </para>
        ///     <para>The dimensions of the input arrays should be the same except the axis along</para>
        ///     <para>which they will be concatenated.</para>
        ///     <para>The dimension of the output array along the concatenated axis will be equal</para>
        ///     <para>to the sum of the corresponding dimensions of the input arrays.</para>
        ///     <para>All inputs with different min/max will be rescaled by using largest [min, max] pairs.</para>
        ///     <para>If any input holds int8, then the output will be int8. Otherwise output will be uint8.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantized_concat.cc:L108</para>
        /// </summary>
        /// <param name="data">List of arrays to concatenate</param>
        /// <param name="num_args">Number of inputs to be concated.</param>
        /// <param name="dim">the dimension to be concated.</param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedConcat(NDArrayList data, int num_args, int dim = 1)
        {
            return new Operator("_contrib_quantized_concat")
                .SetParam("num_args", num_args)
                .SetParam("dim", dim)
                .SetInput(data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Convolution operator for input, weight and bias data type of int8,</para>
        ///     <para>and accumulates in type int32 for the output. For each argument, two more arguments of type</para>
        ///     <para>float32 must be provided representing the thresholds of quantizing argument from data</para>
        ///     <para>type float32 to int8. The final outputs contain the convolution result in int32, and min</para>
        ///     <para>and max thresholds representing the threholds for quantizing the float32 output into int32.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantized_conv.cc:L137</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="weight">weight.</param>
        /// <param name="bias">bias.</param>
        /// <param name="min_data">Minimum value of data.</param>
        /// <param name="max_data">Maximum value of data.</param>
        /// <param name="min_weight">Minimum value of weight.</param>
        /// <param name="max_weight">Maximum value of weight.</param>
        /// <param name="min_bias">Minimum value of bias.</param>
        /// <param name="max_bias">Maximum value of bias.</param>
        /// <param name="kernel">Convolution kernel size: (w,), (h, w) or (d, h, w)</param>
        /// <param name="stride">Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.</param>
        /// <param name="dilate">Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.</param>
        /// <param name="pad">Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.</param>
        /// <param name="num_filter">Convolution filter(channel) number</param>
        /// <param name="num_group">Number of group partitions.</param>
        /// <param name="workspace">
        ///     Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages. When
        ///     CUDNN is not used, it determines the effective batch size of the convolution kernel. When CUDNN is used, it
        ///     controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is
        ///     used.
        /// </param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="cudnn_tune">Whether to pick convolution algo by running performance test.</param>
        /// <param name="cudnn_off">Turn off cudnn for this layer.</param>
        /// <param name="layout">
        ///     Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and
        ///     NCDHW for 3d.NHWC and NDHWC are only supported on GPU.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedConv(NDArray data, NDArray weight, NDArray bias, NDArray min_data, NDArray max_data,
            NDArray min_weight, NDArray max_weight, NDArray min_bias, NDArray max_bias, Shape kernel, uint num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, uint num_group = 1, ulong workspace = 1024,
            bool no_bias = false, ContribQuantizedConvCudnnTune? cudnn_tune = null, bool cudnn_off = false,
            ContribQuantizedConvLayout? layout = null)
        {
            if (stride == null) stride = new Shape();
            if (dilate == null) dilate = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("_contrib_quantized_conv")
                .SetParam("kernel", kernel)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("pad", pad)
                .SetParam("num_filter", num_filter)
                .SetParam("num_group", num_group)
                .SetParam("workspace", workspace)
                .SetParam("no_bias", no_bias)
                .SetParam("cudnn_tune", MxUtil.EnumToString(cudnn_tune, ContribQuantizedConvCudnnTuneConvert))
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("layout", MxUtil.EnumToString(layout, ContribQuantizedConvLayoutConvert))
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .SetInput("min_data", min_data)
                .SetInput("max_data", max_data)
                .SetInput("min_weight", min_weight)
                .SetInput("max_weight", max_weight)
                .SetInput("min_bias", min_bias)
                .SetInput("max_bias", max_bias)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">A ndarray/symbol of type `float32`</param>
        /// <param name="min_data">The minimum scalar value possibly produced for the data</param>
        /// <param name="max_data">The maximum scalar value possibly produced for the data</param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedFlatten(NDArray data, NDArray min_data, NDArray max_data)
        {
            return new Operator("_contrib_quantized_flatten")
                .SetInput("data", data)
                .SetInput("min_data", min_data)
                .SetInput("max_data", max_data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Fully Connected operator for input, weight and bias data type of int8,</para>
        ///     <para>and accumulates in type int32 for the output. For each argument, two more arguments of type</para>
        ///     <para>float32 must be provided representing the thresholds of quantizing argument from data</para>
        ///     <para>type float32 to int8. The final outputs contain the convolution result in int32, and min</para>
        ///     <para>and max thresholds representing the threholds for quantizing the float32 output into int32.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantized_fully_connected.cc:L307</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="weight">weight.</param>
        /// <param name="bias">bias.</param>
        /// <param name="min_data">Minimum value of data.</param>
        /// <param name="max_data">Maximum value of data.</param>
        /// <param name="min_weight">Minimum value of weight.</param>
        /// <param name="max_weight">Maximum value of weight.</param>
        /// <param name="min_bias">Minimum value of bias.</param>
        /// <param name="max_bias">Maximum value of bias.</param>
        /// <param name="num_hidden">Number of hidden nodes of the output.</param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="flatten">Whether to collapse all but the first axis of the input data tensor.</param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedFullyConnected(NDArray data, NDArray weight, NDArray bias, NDArray min_data,
            NDArray max_data, NDArray min_weight, NDArray max_weight, NDArray min_bias, NDArray max_bias,
            int num_hidden, bool no_bias = false, bool flatten = true)
        {
            return new Operator("_contrib_quantized_fully_connected")
                .SetParam("num_hidden", num_hidden)
                .SetParam("no_bias", no_bias)
                .SetParam("flatten", flatten)
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .SetInput("min_data", min_data)
                .SetInput("max_data", max_data)
                .SetInput("min_weight", min_weight)
                .SetInput("max_weight", max_weight)
                .SetInput("min_bias", min_bias)
                .SetInput("max_bias", max_bias)
                .Invoke();
        }

        /// <summary>
        ///     <para>Pooling operator for input and output data type of int8.</para>
        ///     <para>The input and output data comes with min and max thresholds for quantizing</para>
        ///     <para>the float32 data into int8.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para>    This operator only supports `pool_type` of `avg` or `max`.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\quantized_pooling.cc:L145</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="min_data">Minimum value of data.</param>
        /// <param name="max_data">Maximum value of data.</param>
        /// <param name="kernel">Pooling kernel size: (y, x) or (d, y, x)</param>
        /// <param name="pool_type">Pooling type to be applied.</param>
        /// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
        /// <param name="cudnn_off">Turn off cudnn pooling and use MXNet pooling operator. </param>
        /// <param name="pooling_convention">Pooling convention to be applied.</param>
        /// <param name="stride">Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.</param>
        /// <param name="pad">Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.</param>
        /// <param name="p_value">Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.</param>
        /// <param name="count_include_pad">
        ///     Only used for AvgPool, specify whether to count padding elements for
        ///     averagecalculation. For example, with a 5*5 kernel on a 3*3 corner of a image,the sum of the 9 valid elements will
        ///     be divided by 25 if this is set to true,or it will be divided by 9 if this is set to false. Defaults to true.
        /// </param>
        /// <param name="layout">
        ///     Set layout for input and output. Empty for    default layout: NCW for 1d, NCHW for 2d and NCDHW
        ///     for 3d.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray QuantizedPooling(NDArray data, NDArray min_data, NDArray max_data, Shape kernel = null,
            ContribQuantizedPoolingPoolType pool_type = ContribQuantizedPoolingPoolType.Max, bool global_pool = false,
            bool cudnn_off = false,
            ContribQuantizedPoolingPoolingConvention pooling_convention =
                ContribQuantizedPoolingPoolingConvention.Valid, Shape stride = null, Shape pad = null,
            int? p_value = null, bool? count_include_pad = null, ContribQuantizedPoolingLayout? layout = null)
        {
            if (kernel == null) kernel = new Shape();
            if (stride == null) stride = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("_contrib_quantized_pooling")
                .SetParam("kernel", kernel)
                .SetParam("pool_type",
                    MxUtil.EnumToString<ContribQuantizedPoolingPoolType>(pool_type,
                        ContribQuantizedPoolingPoolTypeConvert))
                .SetParam("global_pool", global_pool)
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("pooling_convention",
                    MxUtil.EnumToString<ContribQuantizedPoolingPoolingConvention>(pooling_convention,
                        ContribQuantizedPoolingPoolingConventionConvert))
                .SetParam("stride", stride)
                .SetParam("pad", pad)
                .SetParam("p_value", p_value)
                .SetParam("count_include_pad", count_include_pad)
                .SetParam("layout", MxUtil.EnumToString(layout, ContribQuantizedPoolingLayoutConvert))
                .SetInput("data", data)
                .SetInput("min_data", min_data)
                .SetInput("max_data", max_data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Given data that is quantized in int32 and the corresponding thresholds,</para>
        ///     <para>requantize the data into int8 using min and max thresholds either calculated at runtime</para>
        ///     <para>or from calibration. It's highly recommended to pre-calucate the min and max thresholds</para>
        ///     <para>through calibration since it is able to save the runtime of the operator and improve the</para>
        ///     <para>inference accuracy.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>    This operator only supports forward propogation. DO NOT use it in training.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\quantization\requantize.cc:L60</para>
        /// </summary>
        /// <param name="data">A ndarray/symbol of type `int32`</param>
        /// <param name="min_range">The original minimum scalar value in the form of float32 used for quantizing data into int32.</param>
        /// <param name="max_range">The original maximum scalar value in the form of float32 used for quantizing data into int32.</param>
        /// <param name="out_type">
        ///     Output data type. `auto` can be specified to automatically determine output type according to
        ///     min_calib_range.
        /// </param>
        /// <param name="min_calib_range">
        ///     The minimum scalar value in the form of float32 obtained through calibration. If present,
        ///     it will be used to requantize the int32 data into int8.
        /// </param>
        /// <param name="max_calib_range">
        ///     The maximum scalar value in the form of float32 obtained through calibration. If present,
        ///     it will be used to requantize the int32 data into int8.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray Requantize(NDArray data, NDArray min_range, NDArray max_range,
            ContribRequantizeOutType out_type = ContribRequantizeOutType.Int8, float? min_calib_range = null,
            float? max_calib_range = null)
        {
            return new Operator("_contrib_requantize")
                .SetParam("out_type",
                    MxUtil.EnumToString<ContribRequantizeOutType>(out_type, ContribRequantizeOutTypeConvert))
                .SetParam("min_calib_range", min_calib_range)
                .SetParam("max_calib_range", max_calib_range)
                .SetInput("data", data)
                .SetInput("min_range", min_range)
                .SetInput("max_range", max_range)
                .Invoke();
        }

        /// <summary>
        ///     <para>Maps integer indices to vector representations (embeddings).</para>
        ///     <para> </para>
        ///     <para>note:: ``contrib.SparseEmbedding`` is deprecated, use ``Embedding`` instead.</para>
        ///     <para> </para>
        ///     <para>This operator maps words to real-valued vectors in a high-dimensional space,</para>
        ///     <para>called word embeddings. These embeddings can capture semantic and syntactic properties of the words.</para>
        ///     <para>For example, it has been noted that in the learned embedding spaces, similar words tend</para>
        ///     <para>to be close to each other and dissimilar words far apart.</para>
        ///     <para> </para>
        ///     <para>For an input array of shape (d1, ..., dK),</para>
        ///     <para>the shape of an output array is (d1, ..., dK, output_dim).</para>
        ///     <para>All the input values should be integers in the range [0, input_dim).</para>
        ///     <para> </para>
        ///     <para>If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be</para>
        ///     <para>(ip0, op0).</para>
        ///     <para> </para>
        ///     <para>The storage type of the gradient will be `row_sparse`.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para> </para>
        ///     <para>    `SparseEmbedding` is designed for the use case where `input_dim` is very large (e.g. 100k).</para>
        ///     <para>    The operator is available on both CPU and GPU.</para>
        ///     <para>    When `deterministic` is set to `True`, the accumulation of gradients follows a</para>
        ///     <para>    deterministic order if a feature appears multiple times in the input. However, the</para>
        ///     <para>    accumulation is usually slower when the order is enforced on GPU.</para>
        ///     <para>    When the operator is used on the GPU, the recommended value for `deterministic` is `True`.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  input_dim = 4</para>
        ///     <para>  output_dim = 5</para>
        ///     <para> </para>
        ///     <para>  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)</para>
        ///     <para>  y = [[  0.,   1.,   2.,   3.,   4.],</para>
        ///     <para>       [  5.,   6.,   7.,   8.,   9.],</para>
        ///     <para>       [ 10.,  11.,  12.,  13.,  14.],</para>
        ///     <para>       [ 15.,  16.,  17.,  18.,  19.]]</para>
        ///     <para> </para>
        ///     <para>  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]</para>
        ///     <para>  x = [[ 1.,  3.],</para>
        ///     <para>       [ 0.,  2.]]</para>
        ///     <para> </para>
        ///     <para>  // Mapped input x to its vector representation y.</para>
        ///     <para>  SparseEmbedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],</para>
        ///     <para>                                 [ 15.,  16.,  17.,  18.,  19.]],</para>
        ///     <para> </para>
        ///     <para>                                [[  0.,   1.,   2.,   3.,   4.],</para>
        ///     <para>                                 [ 10.,  11.,  12.,  13.,  14.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\indexing_op.cc:L595</para>
        /// </summary>
        /// <param name="data">The input array to the embedding operator.</param>
        /// <param name="weight">The embedding weight matrix.</param>
        /// <param name="input_dim">Vocabulary size of the input indices.</param>
        /// <param name="output_dim">Dimension of the embedding vectors.</param>
        /// <param name="dtype">Data type of weight.</param>
        /// <param name="sparse_grad">
        ///     Compute row sparse gradient in the backward calculation. If set to True, the grad's storage
        ///     type is row_sparse.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray SparseEmbedding(NDArray data, NDArray weight, int input_dim, int output_dim, DType dtype = null,
            bool sparse_grad = false)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_contrib_SparseEmbedding")
                .SetParam("input_dim", input_dim)
                .SetParam("output_dim", output_dim)
                .SetParam("dtype", dtype)
                .SetParam("sparse_grad", sparse_grad)
                .SetInput("data", data)
                .SetInput("weight", weight)
                .Invoke();
        }

        /// <summary>
        ///     <para>Apply CountSketch to input: map a d-dimension data to k-dimension data"</para>
        ///     <para> </para>
        ///     <para>.. note:: `count_sketch` is only available on GPU.</para>
        ///     <para> </para>
        ///     <para>Assume input data has shape (N, d), sign hash table s has shape (N, d),</para>
        ///     <para>index hash table h has shape (N, d) and mapping dimension out_dim = k,</para>
        ///     <para>each element in s is either +1 or -1, each element in h is random integer from 0 to k-1.</para>
        ///     <para>Then the operator computs:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   out[h[i]] += data[i] * s[i]</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   out_dim = 5</para>
        ///     <para>   x = [[1.2, 2.5, 3.4],[3.2, 5.7, 6.6]]</para>
        ///     <para>   h = [[0, 3, 4]]</para>
        ///     <para>   s = [[1, -1, 1]]</para>
        ///     <para>   mx.contrib.ndarray.count_sketch(data=x, h=h, s=s, out_dim = 5) = [[1.2, 0, 0, -2.5, 3.4],</para>
        ///     <para>                                                                     [3.2, 0, 0, -5.7, 6.6]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\count_sketch.cc:L67</para>
        /// </summary>
        /// <param name="data">Input data to the CountSketchOp.</param>
        /// <param name="h">The index vector</param>
        /// <param name="s">The sign vector</param>
        /// <param name="out_dim">The output dimension.</param>
        /// <param name="processing_batch_size">How many sketch vectors to process at one time.</param>
        /// <returns>returns new symbol</returns>
        public NDArray CountSketch(NDArray data, NDArray h, NDArray s, int out_dim, int processing_batch_size = 32)
        {
            return new Operator("_contrib_count_sketch")
                .SetParam("out_dim", out_dim)
                .SetParam("processing_batch_size", processing_batch_size)
                .SetInput("data", data)
                .SetInput("h", h)
                .SetInput("s", s)
                .Invoke();
        }

        /// <summary>
        ///     <para>Compute 2-D deformable convolution on 4-D input.</para>
        ///     <para> </para>
        ///     <para>The deformable convolution operation is described in https://arxiv.org/abs/1703.06211</para>
        ///     <para> </para>
        ///     <para>For 2-D deformable convolution, the shapes are</para>
        ///     <para> </para>
        ///     <para>- **data**: *(batch_size, channel, height, width)*</para>
        ///     <para>- **offset**: *(batch_size, num_deformable_group * kernel[0] * kernel[1], height, width)*</para>
        ///     <para>- **weight**: *(num_filter, channel, kernel[0], kernel[1])*</para>
        ///     <para>- **bias**: *(num_filter,)*</para>
        ///     <para>- **out**: *(batch_size, num_filter, out_height, out_width)*.</para>
        ///     <para> </para>
        ///     <para>Define::</para>
        ///     <para> </para>
        ///     <para>  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1</para>
        ///     <para> </para>
        ///     <para>then we have::</para>
        ///     <para> </para>
        ///     <para>  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])</para>
        ///     <para>  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])</para>
        ///     <para> </para>
        ///     <para>If ``no_bias`` is set to be true, then the ``bias`` term is ignored.</para>
        ///     <para> </para>
        ///     <para>The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,</para>
        ///     <para>width)*.</para>
        ///     <para> </para>
        ///     <para>If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``</para>
        ///     <para>evenly into *g* parts along the channel axis, and also evenly split ``weight``</para>
        ///     <para>along the first dimension. Next compute the convolution on the *i*-th part of</para>
        ///     <para>the data with the *i*-th weight part. The output is obtained by concating all</para>
        ///     <para>the *g* results.</para>
        ///     <para> </para>
        ///     <para>If ``num_deformable_group`` is larger than 1, denoted by *dg*, then split the</para>
        ///     <para>input ``offset`` evenly into *dg* parts along the channel axis, and also evenly</para>
        ///     <para>split ``out`` evenly into *dg* parts along the channel axis. Next compute the</para>
        ///     <para>deformable convolution, apply the *i*-th part of the offset part on the *i*-th</para>
        ///     <para>out.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Both ``weight`` and ``bias`` are learnable parameters.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\deformable_convolution.cc:L100</para>
        /// </summary>
        /// <param name="data">Input data to the DeformableConvolutionOp.</param>
        /// <param name="offset">Input offset to the DeformableConvolutionOp.</param>
        /// <param name="weight">Weight matrix.</param>
        /// <param name="bias">Bias parameter.</param>
        /// <param name="kernel">Convolution kernel size: (h, w) or (d, h, w)</param>
        /// <param name="stride">Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.</param>
        /// <param name="dilate">Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.</param>
        /// <param name="pad">Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.</param>
        /// <param name="num_filter">Convolution filter(channel) number</param>
        /// <param name="num_group">Number of group partitions.</param>
        /// <param name="num_deformable_group">Number of deformable group partitions.</param>
        /// <param name="workspace">Maximum temperal workspace allowed for convolution (MB).</param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="layout">
        ///     Set layout for input, output and weight. Empty for    default layout: NCW for 1d, NCHW for 2d and
        ///     NCDHW for 3d.
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray DeformableConvolution(NDArray data, NDArray offset, NDArray weight, NDArray bias, Shape kernel,
            uint num_filter, Shape stride = null, Shape dilate = null, Shape pad = null, uint num_group = 1,
            uint num_deformable_group = 1, ulong workspace = 1024, bool no_bias = false,
            ContribDeformableconvolutionLayout? layout = null)
        {
            if (stride == null) stride = new Shape();
            if (dilate == null) dilate = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("_contrib_DeformableConvolution")
                .SetParam("kernel", kernel)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("pad", pad)
                .SetParam("num_filter", num_filter)
                .SetParam("num_group", num_group)
                .SetParam("num_deformable_group", num_deformable_group)
                .SetParam("workspace", workspace)
                .SetParam("no_bias", no_bias)
                .SetParam("layout", MxUtil.EnumToString(layout, ContribDeformableconvolutionLayoutConvert))
                .SetInput("data", data)
                .SetInput("offset", offset)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .Invoke();
        }

        /// <summary>
        ///     <para>Performs deformable position-sensitive region-of-interest pooling on inputs.</para>
        ///     <para>
        ///         The DeformablePSROIPooling operation is described in https://arxiv.org/abs/1703.06211 .batch_size will change
        ///         to the number of region bounding boxes after DeformablePSROIPooling
        ///     </para>
        /// </summary>
        /// <param name="data">Input data to the pooling operator, a 4D Feature maps</param>
        /// <param name="rois">
        ///     Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are
        ///     top left and down right corners of designated region of interest. batch_index indicates the index of corresponding
        ///     image in the input data
        /// </param>
        /// <param name="trans">transition parameter</param>
        /// <param name="spatial_scale">
        ///     Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        ///     of total stride in convolutional layers
        /// </param>
        /// <param name="output_dim">fix output dim</param>
        /// <param name="group_size">fix group size</param>
        /// <param name="pooled_size">fix pooled size</param>
        /// <param name="part_size">fix part size</param>
        /// <param name="sample_per_part">fix samples per part</param>
        /// <param name="trans_std">fix transition std</param>
        /// <param name="no_trans">Whether to disable trans parameter.</param>
        /// <returns>returns new symbol</returns>
        public NDArray DeformablePSROIPooling(Symbol data, Symbol rois, Symbol trans, float spatial_scale,
            int output_dim, int group_size, int pooled_size, int part_size = 0, int sample_per_part = 1,
            float trans_std = 0f, bool no_trans = false)
        {
            return new Operator("_contrib_DeformablePSROIPooling")
                .SetParam("data", data)
                .SetParam("rois", rois)
                .SetParam("trans", trans)
                .SetParam("spatial_scale", spatial_scale)
                .SetParam("output_dim", output_dim)
                .SetParam("group_size", group_size)
                .SetParam("pooled_size", pooled_size)
                .SetParam("part_size", part_size)
                .SetParam("sample_per_part", sample_per_part)
                .SetParam("trans_std", trans_std)
                .SetParam("no_trans", no_trans)
                .Invoke();
        }

        /// <summary>
        ///     <para>Apply 1D FFT to input"</para>
        ///     <para> </para>
        ///     <para>.. note:: `fft` is only available on GPU.</para>
        ///     <para> </para>
        ///     <para>Currently accept 2 input data shapes: (N, d) or (N1, N2, N3, d), data can only be real numbers.</para>
        ///     <para>The output data has shape: (N, 2*d) or (N1, N2, N3, 2*d). The format is: [real0, imag0, real1, imag1, ...].</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   data = np.random.normal(0,1,(3,4))</para>
        ///     <para>   out = mx.contrib.ndarray.fft(data = mx.nd.array(data,ctx = mx.gpu(0)))</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\fft.cc:L56</para>
        /// </summary>
        /// <param name="data">Input data to the FFTOp.</param>
        /// <param name="compute_size">Maximum size of sub-batch to be forwarded at one time</param>
        /// <returns>returns new symbol</returns>
        public NDArray Fft(NDArray data, int compute_size = 128)
        {
            return new Operator("_contrib_fft")
                .SetParam("compute_size", compute_size)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Apply 1D ifft to input"</para>
        ///     <para> </para>
        ///     <para>.. note:: `ifft` is only available on GPU.</para>
        ///     <para> </para>
        ///     <para>
        ///         Currently accept 2 input data shapes: (N, d) or (N1, N2, N3, d). Data is in format: [real0, imag0, real1,
        ///         imag1, ...].
        ///     </para>
        ///     <para>Last dimension must be an even number.</para>
        ///     <para>The output data has shape: (N, d/2) or (N1, N2, N3, d/2). It is only the real part of the result.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   data = np.random.normal(0,1,(3,4))</para>
        ///     <para>   out = mx.contrib.ndarray.ifft(data = mx.nd.array(data,ctx = mx.gpu(0)))</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\ifft.cc:L58</para>
        /// </summary>
        /// <param name="data">Input data to the IFFTOp.</param>
        /// <param name="compute_size">Maximum size of sub-batch to be forwarded at one time</param>
        /// <returns>returns new symbol</returns>
        public NDArray Ifft(NDArray data, int compute_size = 128)
        {
            return new Operator("_contrib_ifft")
                .SetParam("compute_size", compute_size)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Generate region proposals via RPN</para>
        /// </summary>
        /// <param name="cls_prob">Score of how likely proposal is object.</param>
        /// <param name="bbox_pred">BBox Predicted deltas from anchors for proposals</param>
        /// <param name="im_info">Image size and scale.</param>
        /// <param name="rpn_pre_nms_top_n">Number of top scoring boxes to keep after applying NMS to RPN proposals</param>
        /// <param name="rpn_post_nms_top_n">
        ///     Overlap threshold used for non-maximumsuppresion(suppress boxes with IoU >= this
        ///     threshold
        /// </param>
        /// <param name="threshold">NMS value, below which to suppress.</param>
        /// <param name="rpn_min_size">Minimum height or width in proposal</param>
        /// <param name="scales">Used to generate anchor windows by enumerating scales</param>
        /// <param name="ratios">Used to generate anchor windows by enumerating ratios</param>
        /// <param name="feature_stride">
        ///     The size of the receptive field each unit in the convolution layer of the rpn,for example
        ///     the product of all stride's prior to this layer.
        /// </param>
        /// <param name="output_score">Add score to outputs</param>
        /// <param name="iou_loss">Usage of IoU Loss</param>
        /// <returns>returns new symbol</returns>
        public NDArray MultiProposal(NDArray cls_prob, NDArray bbox_pred, NDArray im_info, int rpn_pre_nms_top_n = 6000,
            int rpn_post_nms_top_n = 300, float threshold = 0.7f, int rpn_min_size = 16, Tuple<double> scales = null,
            Tuple<double> ratios = null, int feature_stride = 16, bool output_score = false, bool iou_loss = false)
        {
            if (scales == null) scales = new Tuple<double>(4, 8, 16, 32);
            if (ratios == null) ratios = new Tuple<double>(0.5, 1, 2);

            return new Operator("_contrib_MultiProposal")
                .SetParam("rpn_pre_nms_top_n", rpn_pre_nms_top_n)
                .SetParam("rpn_post_nms_top_n", rpn_post_nms_top_n)
                .SetParam("threshold", threshold)
                .SetParam("rpn_min_size", rpn_min_size)
                .SetParam("scales", scales)
                .SetParam("ratios", ratios)
                .SetParam("feature_stride", feature_stride)
                .SetParam("output_score", output_score)
                .SetParam("iou_loss", iou_loss)
                .SetInput("cls_prob", cls_prob)
                .SetInput("bbox_pred", bbox_pred)
                .SetInput("im_info", im_info)
                .Invoke();
        }

        /// <summary>
        ///     <para>Convert multibox detection predictions.</para>
        /// </summary>
        /// <param name="cls_prob">Class probabilities.</param>
        /// <param name="loc_pred">Location regression predictions.</param>
        /// <param name="anchor">Multibox prior anchor boxes</param>
        /// <param name="clip">Clip out-of-boundary boxes.</param>
        /// <param name="threshold">Threshold to be a positive prediction.</param>
        /// <param name="background_id">Background id.</param>
        /// <param name="nms_threshold">Non-maximum suppression threshold.</param>
        /// <param name="force_suppress">Suppress all detections regardless of class_id.</param>
        /// <param name="variances">Variances to be decoded from box regression output.</param>
        /// <param name="nms_topk">Keep maximum top k detections before nms, -1 for no limit.</param>
        /// <returns>returns new symbol</returns>
        public NDArray MultiBoxDetection(NDArray cls_prob, NDArray loc_pred, NDArray anchor, bool clip = true,
            float threshold = 0.01f, int background_id = 0, float nms_threshold = 0.5f, bool force_suppress = false,
            Tuple<double> variances = null, int nms_topk = -1)
        {
            if (variances == null) variances = new Tuple<double>(0.1, 0.1, 0.2, 0.2);

            return new Operator("_contrib_MultiBoxDetection")
                .SetParam("clip", clip)
                .SetParam("threshold", threshold)
                .SetParam("background_id", background_id)
                .SetParam("nms_threshold", nms_threshold)
                .SetParam("force_suppress", force_suppress)
                .SetParam("variances", variances)
                .SetParam("nms_topk", nms_topk)
                .SetInput("cls_prob", cls_prob)
                .SetInput("loc_pred", loc_pred)
                .SetInput("anchor", anchor)
                .Invoke();
        }

        /// <summary>
        ///     <para>Generate prior(anchor) boxes from data, sizes and ratios.</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="sizes">List of sizes of generated MultiBoxPriores.</param>
        /// <param name="ratios">List of aspect ratios of generated MultiBoxPriores.</param>
        /// <param name="clip">Whether to clip out-of-boundary boxes.</param>
        /// <param name="steps">Priorbox step across y and x, -1 for auto calculation.</param>
        /// <param name="offsets">Priorbox center offsets, y and x respectively</param>
        /// <returns>returns new symbol</returns>
        public NDArray MultiBoxPrior(NDArray data, Tuple<double> sizes = null, Tuple<double> ratios = null,
            bool clip = false, Tuple<double> steps = null, Tuple<double> offsets = null)
        {
            if (sizes == null) sizes = new Tuple<double>(1);
            if (ratios == null) ratios = new Tuple<double>(1);
            if (steps == null) steps = new Tuple<double>(-1, -1);
            if (offsets == null) offsets = new Tuple<double>(0.5, 0.5);

            return new Operator("_contrib_MultiBoxPrior")
                .SetParam("sizes", sizes)
                .SetParam("ratios", ratios)
                .SetParam("clip", clip)
                .SetParam("steps", steps)
                .SetParam("offsets", offsets)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Compute Multibox training targets</para>
        /// </summary>
        /// <param name="anchor">Generated anchor boxes.</param>
        /// <param name="label">Object detection labels.</param>
        /// <param name="cls_pred">Class predictions.</param>
        /// <param name="overlap_threshold">Anchor-GT overlap threshold to be regarded as a positive match.</param>
        /// <param name="ignore_label">Label for ignored anchors.</param>
        /// <param name="negative_mining_ratio">Max negative to positive samples ratio, use -1 to disable mining</param>
        /// <param name="negative_mining_thresh">Threshold used for negative mining.</param>
        /// <param name="minimum_negative_samples">Minimum number of negative samples.</param>
        /// <param name="variances">Variances to be encoded in box regression target.</param>
        /// <returns>returns new symbol</returns>
        public NDArray MultiBoxTarget(NDArray anchor, NDArray label, NDArray cls_pred, float overlap_threshold = 0.5f,
            float ignore_label = -1f, float negative_mining_ratio = -1f, float negative_mining_thresh = 0.5f,
            int minimum_negative_samples = 0, Tuple<double> variances = null)
        {
            if (variances == null) variances = new Tuple<double>(0.1, 0.1, 0.2, 0.2);

            return new Operator("_contrib_MultiBoxTarget")
                .SetParam("overlap_threshold", overlap_threshold)
                .SetParam("ignore_label", ignore_label)
                .SetParam("negative_mining_ratio", negative_mining_ratio)
                .SetParam("negative_mining_thresh", negative_mining_thresh)
                .SetParam("minimum_negative_samples", minimum_negative_samples)
                .SetParam("variances", variances)
                .SetInput("anchor", anchor)
                .SetInput("label", label)
                .SetInput("cls_pred", cls_pred)
                .Invoke();
        }

        /// <summary>
        ///     <para>Generate region proposals via RPN</para>
        /// </summary>
        /// <param name="cls_prob">Score of how likely proposal is object.</param>
        /// <param name="bbox_pred">BBox Predicted deltas from anchors for proposals</param>
        /// <param name="im_info">Image size and scale.</param>
        /// <param name="rpn_pre_nms_top_n">Number of top scoring boxes to keep after applying NMS to RPN proposals</param>
        /// <param name="rpn_post_nms_top_n">
        ///     Overlap threshold used for non-maximumsuppresion(suppress boxes with IoU >= this
        ///     threshold
        /// </param>
        /// <param name="threshold">NMS value, below which to suppress.</param>
        /// <param name="rpn_min_size">Minimum height or width in proposal</param>
        /// <param name="scales">Used to generate anchor windows by enumerating scales</param>
        /// <param name="ratios">Used to generate anchor windows by enumerating ratios</param>
        /// <param name="feature_stride">
        ///     The size of the receptive field each unit in the convolution layer of the rpn,for example
        ///     the product of all stride's prior to this layer.
        /// </param>
        /// <param name="output_score">Add score to outputs</param>
        /// <param name="iou_loss">Usage of IoU Loss</param>
        /// <returns>returns new symbol</returns>
        public NDArray Proposal(NDArray cls_prob, NDArray bbox_pred, NDArray im_info, int rpn_pre_nms_top_n = 6000,
            int rpn_post_nms_top_n = 300, float threshold = 0.7f, int rpn_min_size = 16, Tuple<double> scales = null,
            Tuple<double> ratios = null, int feature_stride = 16, bool output_score = false, bool iou_loss = false)
        {
            if (scales == null) scales = new Tuple<double>(4, 8, 16, 32);
            if (ratios == null) ratios = new Tuple<double>(0.5, 1, 2);

            return new Operator("_contrib_Proposal")
                .SetParam("rpn_pre_nms_top_n", rpn_pre_nms_top_n)
                .SetParam("rpn_post_nms_top_n", rpn_post_nms_top_n)
                .SetParam("threshold", threshold)
                .SetParam("rpn_min_size", rpn_min_size)
                .SetParam("scales", scales)
                .SetParam("ratios", ratios)
                .SetParam("feature_stride", feature_stride)
                .SetParam("output_score", output_score)
                .SetParam("iou_loss", iou_loss)
                .SetInput("cls_prob", cls_prob)
                .SetInput("bbox_pred", bbox_pred)
                .SetInput("im_info", im_info)
                .Invoke();
        }

        /// <summary>
        ///     <para>
        ///         Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop
        ///         input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output
        ///         indicated by pooled_size. batch_size will change to the number of region bounding boxes after PSROIPooling
        ///     </para>
        /// </summary>
        /// <param name="data">Input data to the pooling operator, a 4D Feature maps</param>
        /// <param name="rois">
        ///     Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are
        ///     top left and down right corners of designated region of interest. batch_index indicates the index of corresponding
        ///     image in the input data
        /// </param>
        /// <param name="spatial_scale">
        ///     Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        ///     of total stride in convolutional layers
        /// </param>
        /// <param name="output_dim">fix output dim</param>
        /// <param name="pooled_size">fix pooled size</param>
        /// <param name="group_size">fix group size</param>
        /// <returns>returns new symbol</returns>
        public NDArray PSROIPooling(Symbol data, Symbol rois, float spatial_scale, int output_dim, int pooled_size,
            int group_size = 0)
        {
            return new Operator("_contrib_PSROIPooling")
                .SetParam("data", data)
                .SetParam("rois", rois)
                .SetParam("spatial_scale", spatial_scale)
                .SetParam("output_dim", output_dim)
                .SetParam("pooled_size", pooled_size)
                .SetParam("group_size", group_size)
                .Invoke();
        }

        public static NDArray MultiAdamWUpdate(NDArrayList weight, NDArrayList grad, NDArrayList mean, NDArrayList var,
          NDArrayList rescale_grad, float lr, float eta, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f,
          float wd = 0f, float clip_gradient = -1f)
        {
            throw new NotImplementedException();

        }

        public NDArray BatchNormWithReLU(NDArray data, NDArray gamma, NDArray beta, NDArray moving_mean,
            NDArray moving_var, double eps = 0.001, float momentum = 0.9f, bool fix_gamma = true,
            bool use_global_stats = false, bool output_mean_var = false, int axis = 1, bool cudnn_off = false)
        {
            return new Operator("BatchNormWithReLU")
                .SetParam("eps", eps)
                .SetParam("momentum", momentum)
                .SetParam("fix_gamma", fix_gamma)
                .SetParam("use_global_stats", use_global_stats)
                .SetParam("output_mean_var", output_mean_var)
                .SetParam("axis", axis)
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .SetInput("moving_mean", moving_mean)
                .SetInput("moving_var", moving_var)
                .Invoke();
        }
    }
}