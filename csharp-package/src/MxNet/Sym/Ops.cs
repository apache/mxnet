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
    public partial class sym
    {
        public static SymImgApi Image = new SymImgApi();
        public static SymContribApi Contrib = new SymContribApi();

        internal static readonly List<string> LeakyreluActTypeConvert =
            new List<string> {"elu", "gelu", "leaky", "prelu", "rrelu", "selu"};

        internal static readonly List<string> ActivationActTypeConvert =
            new List<string> {"relu", "sigmoid", "softrelu", "softsign", "tanh"};

        internal static readonly List<string> ConvolutionCudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        internal static readonly List<string> ConvolutionLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"};

        internal static readonly List<string> CtclossBlankLabelConvert = new List<string> {"first", "last"};

        internal static readonly List<string> DeconvolutionCudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        internal static readonly List<string> DeconvolutionLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"};

        internal static readonly List<string> DropoutModeConvert = new List<string> {"always", "training"};

        internal static readonly List<string> PoolingPoolTypeConvert = new List<string> {"avg", "lp", "max", "sum"};

        internal static readonly List<string> PoolingPoolingConventionConvert =
            new List<string> {"full", "same", "valid"};

        internal static readonly List<string> PoolingLayoutConvert = new List<string>
            {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"};

        internal static readonly List<string> SoftmaxactivationModeConvert = new List<string> {"channel", "instance"};

        internal static readonly List<string> UpsamplingSampleTypeConvert = new List<string> {"bilinear", "nearest"};
        internal static readonly List<string> UpsamplingMultiInputModeConvert = new List<string> {"concat", "sum"};

        internal static readonly List<string> PadModeConvert = new List<string> {"constant", "edge", "reflect"};

        internal static readonly List<string> RNNModeConvert = new List<string> {"gru", "lstm", "rnn_relu", "rnn_tanh"};

        internal static readonly List<string> SoftmaxoutputNormalizationConvert =
            new List<string> {"batch", "null", "valid"};

        internal static readonly List<string> PickModeConvert = new List<string> {"clip", "wrap"};

        internal static readonly List<string> NormOutDtypeConvert = new List<string>
            {"float16", "float32", "float64", "int32", "int64", "int8"};

        internal static readonly List<string>
            CastStorageStypeConvert = new List<string> {"csr", "default", "row_sparse"};

        internal static readonly List<string> DotForwardStypeConvert = new List<string> {"csr", "default", "row_sparse"};

        internal static readonly List<string> BatchDotForwardStypeConvert =
            new List<string> {"csr", "default", "row_sparse"};

        internal static readonly List<string> TakeModeConvert = new List<string> {"clip", "raise", "wrap"};

        internal static readonly List<string> TopkRetTypConvert = new List<string> {"both", "indices", "mask", "value"};

        internal static readonly List<string> ConvolutionV1CudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        internal static readonly List<string> ConvolutionV1LayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NDHWC", "NHWC"};

        internal static readonly List<string> GridgeneratorTransformTypeConvert = new List<string> {"affine", "warp"};

        internal static readonly List<string> L2normalizationModeConvert =
            new List<string> {"channel", "instance", "spatial"};

        internal static readonly List<string> MakelossNormalizationConvert = new List<string> {"batch", "null", "valid"};

        internal static readonly List<string> PoolingV1PoolTypeConvert = new List<string> {"avg", "max", "sum"};
        internal static readonly List<string> PoolingV1PoolingConventionConvert = new List<string> {"full", "valid"};

        internal static readonly List<string> SpatialtransformerTransformTypeConvert = new List<string> {"affine"};
        internal static readonly List<string> SpatialtransformerSamplerTypeConvert = new List<string> {"bilinear"};

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <returns>returns new symbol</returns>
        public static Symbol CustomFunction(string symbol_name = "")
        {
            return new Operator("_CustomFunction")
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">input data list</param>
        /// <returns>returns new symbol</returns>
        public static Symbol CachedOp(SymbolList data, string symbol_name = "")
        {
            return new Operator("_CachedOp")
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Decode image with OpenCV. </para>
        ///     <para>Note: return image in RGB by default, instead of OpenCV's default BGR.</para>
        /// </summary>
        /// <param name="buf">Buffer containing binary encoded image</param>
        /// <param name="flag">Convert decoded image to grayscale (0) or color (1).</param>
        /// <param name="to_rgb">Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Cvimdecode(Symbol buf, int flag = 1, bool to_rgb = true, string symbol_name = "")
        {
            return new Operator("_cvimdecode")
                .SetParam("buf", buf)
                .SetParam("flag", flag)
                .SetParam("to_rgb", to_rgb)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Read and decode image with OpenCV. </para>
        ///     <para>Note: return image in RGB by default, instead of OpenCV's default BGR.</para>
        /// </summary>
        /// <param name="filename">Name of the image file to be loaded.</param>
        /// <param name="flag">Convert decoded image to grayscale (0) or color (1).</param>
        /// <param name="to_rgb">Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Cvimread(string filename, int flag = 1, bool to_rgb = true, string symbol_name = "")
        {
            return new Operator("_cvimread")
                .SetParam("filename", filename)
                .SetParam("flag", flag)
                .SetParam("to_rgb", to_rgb)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Resize image with OpenCV. </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source image</param>
        /// <param name="w">Width of resized image.</param>
        /// <param name="h">Height of resized image.</param>
        /// <param name="interp">Interpolation method (default=cv2.INTER_LINEAR).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Cvimresize(Symbol data, int w, int h, int interp = 1, string symbol_name = "")
        {
            return new Operator("_cvimresize")
                .SetParam("data", data)
                .SetParam("w", w)
                .SetParam("h", h)
                .SetParam("interp", interp)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Pad image border with OpenCV. </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source image</param>
        /// <param name="top">Top margin.</param>
        /// <param name="bot">Bottom margin.</param>
        /// <param name="left">Left margin.</param>
        /// <param name="right">Right margin.</param>
        /// <param name="type">Filling type (default=cv2.BORDER_CONSTANT).</param>
        /// <param name="values">Fill with value(RGB[A] or gray), up to 4 channels.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol CvcopyMakeBorder(Symbol data, int top, int bot, int left, int right, int type = 0,
            Tuple<double> values = null, string symbol_name = "")
        {
            if (values == null) values = new Tuple<double>();

            return new Operator("_cvcopyMakeBorder")
                .SetParam("data", data)
                .SetParam("top", top)
                .SetParam("bot", bot)
                .SetParam("left", left)
                .SetParam("right", right)
                .SetParam("type", type)
                .SetParam("values", values)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">input data</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Copyto(Symbol data, string symbol_name = "")
        {
            return new Operator("_copyto")
                .SetParam("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Place holder for variable who cannot perform gradient</para>
        /// </summary>
        /// <returns>returns new symbol</returns>
        public static Symbol NoGradient(string symbol_name = "")
        {
            return new Operator("_NoGradient")
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Batch normalization.</para>
        ///     <para> </para>
        ///     <para>This operator is DEPRECATED. Perform BatchNorm on the input.</para>
        ///     <para> </para>
        ///     <para>Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as</para>
        ///     <para>well as offset ``beta``.</para>
        ///     <para> </para>
        ///     <para>Assume the input has more than one dimension and we normalize along axis 1.</para>
        ///     <para>We first compute the mean and variance along this axis:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  data\_mean[i] = mean(data[:,i,:,...]) \\</para>
        ///     <para>  data\_var[i] = var(data[:,i,:,...])</para>
        ///     <para> </para>
        ///     <para>Then compute the normalized output, which has the same shape as input, as following:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]</para>
        ///     <para> </para>
        ///     <para>Both *mean* and *var* returns a scalar by treating the input as a vector.</para>
        ///     <para> </para>
        ///     <para>Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``</para>
        ///     <para>have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and</para>
        ///     <para>``data_var`` as well, which are needed for the backward pass.</para>
        ///     <para> </para>
        ///     <para>Besides the inputs and the outputs, this operator accepts two auxiliary</para>
        ///     <para>states, ``moving_mean`` and ``moving_var``, which are *k*-length</para>
        ///     <para>vectors. They are global statistics for the whole dataset, which are updated</para>
        ///     <para>by::</para>
        ///     <para> </para>
        ///     <para>  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)</para>
        ///     <para>  moving_var = moving_var * momentum + data_var * (1 - momentum)</para>
        ///     <para> </para>
        ///     <para>If ``use_global_stats`` is set to be true, then ``moving_mean`` and</para>
        ///     <para>``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute</para>
        ///     <para>the output. It is often used during inference.</para>
        ///     <para> </para>
        ///     <para>Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,</para>
        ///     <para>then set ``gamma`` to 1 and its gradient to 0.</para>
        ///     <para> </para>
        ///     <para>There's no sparse support for this operator, and it will exhibit problematic behavior if used with</para>
        ///     <para>sparse tensors.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\batch_norm_v1.cc:L95</para>
        /// </summary>
        /// <param name="data">Input data to batch normalization</param>
        /// <param name="gamma">gamma array</param>
        /// <param name="beta">beta array</param>
        /// <param name="eps">Epsilon to prevent div 0</param>
        /// <param name="momentum">Momentum for moving average</param>
        /// <param name="fix_gamma">Fix gamma while training</param>
        /// <param name="use_global_stats">
        ///     Whether use global moving statistics instead of local batch-norm. This will force change
        ///     batch-norm into a scale shift operator.
        /// </param>
        /// <param name="output_mean_var">Output All,normal mean and var</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BatchNormV1(Symbol data, Symbol gamma, Symbol beta, float eps = 0.001f,
            float momentum = 0.9f, bool fix_gamma = true, bool use_global_stats = false, bool output_mean_var = false,
            string symbol_name = "")
        {
            return new Operator("BatchNorm_v1")
                .SetParam("eps", eps)
                .SetParam("momentum", momentum)
                .SetParam("fix_gamma", fix_gamma)
                .SetParam("use_global_stats", use_global_stats)
                .SetParam("output_mean_var", output_mean_var)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for multi-precision AdamW optimizer.</para>
        ///     <para> </para>
        ///     <para>AdamW is seen as a modification of Adam by decoupling the weight decay from the</para>
        ///     <para>optimization steps taken w.r.t. the loss function.</para>
        ///     <para> </para>
        ///     <para>Adam update consists of the following steps, where g represents gradient and m, v</para>
        ///     <para>are 1st and 2nd order moment estimates (mean and variance).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\</para>
        ///     <para> v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\</para>
        ///     <para> W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon } + wd W_{t-1})</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> m = beta1*m + (1-beta1)*grad</para>
        ///     <para> v = beta2*v + (1-beta2)*(grad**2)</para>
        ///     <para> w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)</para>
        ///     <para> </para>
        ///     <para>Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,</para>
        ///     <para>the update is skipped.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\adamw.cc:L77</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mean">Moving mean</param>
        /// <param name="var">Moving variance</param>
        /// <param name="weight32">Weight32</param>
        /// <param name="rescale_grad">Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, the update is skipped.</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="beta1">The decay rate for the 1st moment estimates.</param>
        /// <param name="beta2">The decay rate for the 2nd moment estimates.</param>
        /// <param name="epsilon">A small constant for numerical stability.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="eta">Learning rate schedule multiplier</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MpAdamWUpdate(Symbol weight, Symbol grad, Symbol mean, Symbol var, Symbol weight32,
            Symbol rescale_grad, float lr, float eta, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f,
            float wd = 0f, float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("_mp_adamw_update")
                .SetParam("lr", lr)
                .SetParam("beta1", beta1)
                .SetParam("beta2", beta2)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("eta", eta)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mean", mean)
                .SetInput("var", var)
                .SetInput("weight32", weight32)
                .SetInput("rescale_grad", rescale_grad)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for AdamW optimizer. AdamW is seen as a modification of</para>
        ///     <para>Adam by decoupling the weight decay from the optimization steps taken w.r.t. the loss function.</para>
        ///     <para> </para>
        ///     <para>Adam update consists of the following steps, where g represents gradient and m, v</para>
        ///     <para>are 1st and 2nd order moment estimates (mean and variance).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\</para>
        ///     <para> v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\</para>
        ///     <para> W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon } + wd W_{t-1})</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> m = beta1*m + (1-beta1)*grad</para>
        ///     <para> v = beta2*v + (1-beta2)*(grad**2)</para>
        ///     <para> w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)</para>
        ///     <para> </para>
        ///     <para>Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,</para>
        ///     <para>the update is skipped.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\adamw.cc:L120</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mean">Moving mean</param>
        /// <param name="var">Moving variance</param>
        /// <param name="rescale_grad">Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, the update is skipped.</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="beta1">The decay rate for the 1st moment estimates.</param>
        /// <param name="beta2">The decay rate for the 2nd moment estimates.</param>
        /// <param name="epsilon">A small constant for numerical stability.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="eta">Learning rate schedule multiplier</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol AdamWUpdate(Symbol weight, Symbol grad, Symbol mean, Symbol var, Symbol rescale_grad,
            float lr, float eta, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f, float wd = 0f,
            float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("_adamw_update")
                .SetParam("lr", lr)
                .SetParam("beta1", beta1)
                .SetParam("beta2", beta2)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("eta", eta)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mean", mean)
                .SetInput("var", var)
                .SetInput("rescale_grad", rescale_grad)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes the Khatri-Rao product of the input matrices.</para>
        ///     <para> </para>
        ///     <para>Given a collection of :math:`n` input matrices,</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},</para>
        ///     <para> </para>
        ///     <para>the (column-wise) Khatri-Rao product is defined as the matrix,</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},</para>
        ///     <para> </para>
        ///     <para>where the :math:`k` th column is equal to the column-wise outer product</para>
        ///     <para>:math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth</para>
        ///     <para>column of the ith matrix.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  >>> A = mx.nd.array([[1, -1],</para>
        ///     <para>  >>>                  [2, -3]])</para>
        ///     <para>  >>> B = mx.nd.array([[1, 4],</para>
        ///     <para>  >>>                  [2, 5],</para>
        ///     <para>  >>>                  [3, 6]])</para>
        ///     <para>  >>> C = mx.nd.khatri_rao(A, B)</para>
        ///     <para>  >>> print(C.asnumpy())</para>
        ///     <para>  [[  1.  -4.]</para>
        ///     <para>   [  2.  -5.]</para>
        ///     <para>   [  3.  -6.]</para>
        ///     <para>   [  2. -12.]</para>
        ///     <para>   [  4. -15.]</para>
        ///     <para>   [  6. -18.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\contrib\krprod.cc:L108</para>
        /// </summary>
        /// <param name="args">Positional input matrices</param>
        /// <returns>returns new symbol</returns>
        public static Symbol KhatriRao(SymbolList args, string symbol_name = "")
        {
            return new Operator("khatri_rao")
                .SetInput(args)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Run a for loop over an NDArray with user-defined computation</para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\control_flow.cc:1256</para>
        /// </summary>
        /// <param name="fn">Input graph.</param>
        /// <param name="data">The input arrays that include data arrays and states.</param>
        /// <param name="num_args">Number of inputs.</param>
        /// <param name="num_outputs">The number of outputs of the subgraph.</param>
        /// <param name="num_out_data">The number of output data of the subgraph.</param>
        /// <param name="in_state_locs">The locations of loop states among the inputs.</param>
        /// <param name="in_data_locs">The locations of input data among the inputs.</param>
        /// <param name="remain_locs">The locations of remaining data among the inputs.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Foreach(Symbol fn, SymbolList data, int num_args, int num_outputs, int num_out_data,
            Tuple<double> in_state_locs, Tuple<double> in_data_locs, Tuple<double> remain_locs, string symbol_name = "")
        {
            return new Operator("_foreach")
                .SetParam("fn", fn)
                .SetParam("num_args", num_args)
                .SetParam("num_outputs", num_outputs)
                .SetParam("num_out_data", num_out_data)
                .SetParam("in_state_locs", in_state_locs)
                .SetParam("in_data_locs", in_data_locs)
                .SetParam("remain_locs", remain_locs)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Run a while loop over with user-defined condition and computation</para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\control_flow.cc:1317</para>
        /// </summary>
        /// <param name="cond">Input graph for the loop condition.</param>
        /// <param name="func">Input graph for the loop body.</param>
        /// <param name="data">The input arrays that include data arrays and states.</param>
        /// <param name="num_args">Number of input arguments, including cond and func as two symbol inputs.</param>
        /// <param name="num_outputs">The number of outputs of the subgraph.</param>
        /// <param name="num_out_data">The number of outputs from the function body.</param>
        /// <param name="max_iterations">Maximum number of iterations.</param>
        /// <param name="cond_input_locs">The locations of cond's inputs in the given inputs.</param>
        /// <param name="func_input_locs">The locations of func's inputs in the given inputs.</param>
        /// <param name="func_var_locs">The locations of loop_vars among func's inputs.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol WhileLoop(Symbol cond, Symbol func, SymbolList data, int num_args, int num_outputs,
            int num_out_data, int max_iterations, Tuple<double> cond_input_locs, Tuple<double> func_input_locs,
            Tuple<double> func_var_locs, string symbol_name = "")
        {
            return new Operator("_while_loop")
                .SetParam("cond", cond)
                .SetParam("func", func)
                .SetParam("num_args", num_args)
                .SetParam("num_outputs", num_outputs)
                .SetParam("num_out_data", num_out_data)
                .SetParam("max_iterations", max_iterations)
                .SetParam("cond_input_locs", cond_input_locs)
                .SetParam("func_input_locs", func_input_locs)
                .SetParam("func_var_locs", func_var_locs)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Run a if-then-else using user-defined condition and computation</para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\control_flow.cc:1379</para>
        /// </summary>
        /// <param name="cond">Input graph for the condition.</param>
        /// <param name="then_branch">Input graph for the then branch.</param>
        /// <param name="else_branch">Input graph for the else branch.</param>
        /// <param name="data">The input arrays that include data arrays and states.</param>
        /// <param name="num_args">Number of input arguments, including cond, then and else as three symbol inputs.</param>
        /// <param name="num_outputs">The number of outputs of the subgraph.</param>
        /// <param name="cond_input_locs">The locations of cond's inputs in the given inputs.</param>
        /// <param name="then_input_locs">The locations of then's inputs in the given inputs.</param>
        /// <param name="else_input_locs">The locations of else's inputs in the given inputs.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Cond(Symbol cond, Symbol then_branch, Symbol else_branch, SymbolList data, int num_args,
            int num_outputs, Tuple<double> cond_input_locs, Tuple<double> then_input_locs,
            Tuple<double> else_input_locs, string symbol_name = "")
        {
            return new Operator("_cond")
                .SetParam("cond", cond)
                .SetParam("then_branch", then_branch)
                .SetParam("else_branch", else_branch)
                .SetParam("num_args", num_args)
                .SetParam("num_outputs", num_outputs)
                .SetParam("cond_input_locs", cond_input_locs)
                .SetParam("then_input_locs", then_input_locs)
                .SetParam("else_input_locs", else_input_locs)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Apply a custom operator implemented in a frontend language (like Python).</para>
        ///     <para> </para>
        ///     <para>Custom operators should override required methods like `forward` and `backward`.</para>
        ///     <para>The custom operator must be registered before it can be used.</para>
        ///     <para>Please check the tutorial here: http://mxnet.io/faq/new_op.html.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\custom\custom.cc:L546</para>
        /// </summary>
        /// <param name="data">Input data for the custom operator.</param>
        /// <param name="op_type">
        ///     Name of the custom operator. This is the name that is passed to `mx.operator.register` to
        ///     register the operator.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Custom(SymbolList data, string op_type, string symbol_name = "")
        {
            return new Operator("Custom")
                .SetParam("op_type", op_type)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Apply a sparse regularization to the output a sigmoid activation function.</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="sparseness_target">The sparseness target</param>
        /// <param name="penalty">The tradeoff parameter for the sparseness penalty</param>
        /// <param name="momentum">The momentum for running average</param>
        /// <returns>returns new symbol</returns>
        public static Symbol IdentityAttachKLSparseReg(Symbol data, float sparseness_target = 0.1f,
            float penalty = 0.001f, float momentum = 0.9f, string symbol_name = "")
        {
            return new Operator("IdentityAttachKLSparseReg")
                .SetParam("sparseness_target", sparseness_target)
                .SetParam("penalty", penalty)
                .SetParam("momentum", momentum)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies Leaky rectified linear unit activation element-wise to the input.</para>
        ///     <para> </para>
        ///     <para>Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`</para>
        ///     <para>when the input is negative and has a slope of one when input is positive.</para>
        ///     <para> </para>
        ///     <para>The following modified ReLU Activation functions are supported:</para>
        ///     <para> </para>
        ///     <para>- *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`</para>
        ///     <para>- *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha * (exp(x) - 1))` where</para>
        ///     <para>  *lambda = 1.0507009873554804934193349852946* and *alpha = 1.6732632423543772848170429916717*.</para>
        ///     <para>- *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`</para>
        ///     <para>- *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.</para>
        ///     <para>- *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from</para>
        ///     <para>  *[lower_bound, upper_bound)* for training, while fixed to be</para>
        ///     <para>  *(lower_bound+upper_bound)/2* for inference.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\leaky_relu.cc:L65</para>
        /// </summary>
        /// <param name="data">Input data to activation function.</param>
        /// <param name="gamma">
        ///     Slope parameter for PReLU. Only required when act_type is 'prelu'. It should be either a vector of
        ///     size 1, or the same size as the second dimension of data.
        /// </param>
        /// <param name="act_type">Activation function to be applied.</param>
        /// <param name="slope">Init slope for the activation. (For leaky and elu only)</param>
        /// <param name="lower_bound">Lower bound of random slope. (For rrelu only)</param>
        /// <param name="upper_bound">Upper bound of random slope. (For rrelu only)</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LeakyReLU(Symbol data, NDArray gamma = null,
            ReluActType act_type = ReluActType.Leaky, float slope = 0.25f, float lower_bound = 0.125f,
            float upper_bound = 0.334f, string symbol_name = "")
        {
            return new Operator("LeakyReLU")
                .SetParam("act_type", MxUtil.EnumToString<ReluActType>(act_type, LeakyreluActTypeConvert))
                .SetParam("slope", slope)
                .SetParam("lower_bound", lower_bound)
                .SetParam("upper_bound", upper_bound)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Calculate cross entropy of softmax output and one-hot label.</para>
        ///     <para> </para>
        ///     <para>- This operator computes the cross entropy in two steps:</para>
        ///     <para>  - Applies softmax function on the input array.</para>
        ///     <para>  - Computes and returns the cross entropy loss between the softmax output and the labels.</para>
        ///     <para> </para>
        ///     <para>- The softmax function and cross entropy loss is given by:</para>
        ///     <para> </para>
        ///     <para>  - Softmax Function:</para>
        ///     <para> </para>
        ///     <para>  .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}</para>
        ///     <para> </para>
        ///     <para>  - Cross Entropy Function:</para>
        ///     <para> </para>
        ///     <para>  .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[1, 2, 3],</para>
        ///     <para>       [11, 7, 5]]</para>
        ///     <para> </para>
        ///     <para>  label = [2, 0]</para>
        ///     <para> </para>
        ///     <para>  softmax(x) = [[0.09003057, 0.24472848, 0.66524094],</para>
        ///     <para>                [0.97962922, 0.01794253, 0.00242826]]</para>
        ///     <para> </para>
        ///     <para>  softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\loss_binary_op.cc:L59</para>
        /// </summary>
        /// <param name="data">Input data</param>
        /// <param name="label">Input label</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SoftmaxCrossEntropy(Symbol data, Symbol label, string symbol_name = "")
        {
            return new Operator("softmax_cross_entropy")
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies an activation function element-wise to the input.</para>
        ///     <para> </para>
        ///     <para>The following activation functions are supported:</para>
        ///     <para> </para>
        ///     <para>- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`</para>
        ///     <para>- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`</para>
        ///     <para>- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`</para>
        ///     <para>- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`</para>
        ///     <para>- `softsign`: :math:`y = \frac{x}{1 + abs(x)}`</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\activation.cc:L167</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="act_type">Activation function to be applied.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Activation(Symbol data, ActivationType act_type, string symbol_name = "")
        {
            return new Operator("Activation")
                .SetParam("act_type", MxUtil.EnumToString<ActivationType>(act_type, ActivationActTypeConvert))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Batch normalization.</para>
        
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\batch_norm.cc:L574</para>
        /// </summary>
        /// <param name="data">Input data to batch normalization</param>
        /// <param name="gamma">gamma array</param>
        /// <param name="beta">beta array</param>
        /// <param name="moving_mean">running mean of input</param>
        /// <param name="moving_var">running variance of input</param>
        /// <param name="eps">
        ///     Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using
        ///     cudnn (usually 1e-5)
        /// </param>
        /// <param name="momentum">Momentum for moving average</param>
        /// <param name="fix_gamma">Fix gamma while training</param>
        /// <param name="use_global_stats">
        ///     Whether use global moving statistics instead of local batch-norm. This will force change
        ///     batch-norm into a scale shift operator.
        /// </param>
        /// <param name="output_mean_var">Output the mean and inverse std </param>
        /// <param name="axis">Specify which shape axis the channel is specified</param>
        /// <param name="cudnn_off">Do not select CUDNN operator, if available</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BatchNorm(Symbol data, Symbol gamma, Symbol beta, Symbol moving_mean, Symbol moving_var,
            double eps = 0.001, float momentum = 0.9f, bool fix_gamma = true, bool use_global_stats = false,
            bool output_mean_var = false, int axis = 1, bool cudnn_off = false, string symbol_name = "")
        {
            return new Operator("BatchNorm")
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
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Joins input arrays along a given axis.</para>
        ///     <para> </para>
        ///     <para>.. note:: `Concat` is deprecated. Use `concat` instead.</para>
        ///     <para> </para>
        ///     <para>The dimensions of the input arrays should be the same except the axis along</para>
        ///     <para>which they will be concatenated.</para>
        ///     <para>The dimension of the output array along the concatenated axis will be equal</para>
        ///     <para>to the sum of the corresponding dimensions of the input arrays.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``concat`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- concat(csr, csr, ..., csr, dim=0) = csr</para>
        ///     <para>- otherwise, ``concat`` generates output with default storage</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[1,1],[2,2]]</para>
        ///     <para>   y = [[3,3],[4,4],[5,5]]</para>
        ///     <para>   z = [[6,6], [7,7],[8,8]]</para>
        ///     <para> </para>
        ///     <para>   concat(x,y,z,dim=0) = [[ 1.,  1.],</para>
        ///     <para>                          [ 2.,  2.],</para>
        ///     <para>                          [ 3.,  3.],</para>
        ///     <para>                          [ 4.,  4.],</para>
        ///     <para>                          [ 5.,  5.],</para>
        ///     <para>                          [ 6.,  6.],</para>
        ///     <para>                          [ 7.,  7.],</para>
        ///     <para>                          [ 8.,  8.]]</para>
        ///     <para> </para>
        ///     <para>   Note that you cannot concat x,y,z along dimension 1 since dimension</para>
        ///     <para>   0 is not the same for all the input arrays.</para>
        ///     <para> </para>
        ///     <para>   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],</para>
        ///     <para>                         [ 4.,  4.,  7.,  7.],</para>
        ///     <para>                         [ 5.,  5.,  8.,  8.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\concat.cc:L371</para>
        /// </summary>
        /// <param name="data">List of arrays to concatenate</param>
        /// <param name="num_args">Number of inputs to be concated.</param>
        /// <param name="dim">the dimension to be concated.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Concat(SymbolList data, int dim = 1, string symbol_name = "")
        {
            return new Operator("Concat")
                .SetParam("num_args", data.Length)
                .SetParam("dim", dim)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">List of arrays to concatenate</param>
        /// <param name="num_args">Number of inputs to be concated.</param>
        /// <param name="dim">the dimension to be concated.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RnnParamConcat(SymbolList data, int num_args, int dim = 1, string symbol_name = "")
        {
            return new Operator("_rnn_param_concat")
                .SetParam("num_args", num_args)
                .SetParam("dim", dim)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Compute *N*-D convolution on *(N+2)*-D input.</para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\convolution.cc:L472</para>
        /// </summary>
        /// <param name="data">Input data to the ConvolutionOp.</param>
        /// <param name="weight">Weight matrix.</param>
        /// <param name="bias">Bias parameter.</param>
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
        public static Symbol Convolution(Symbol data, Symbol weight, Symbol bias, Shape kernel, int num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, int num_group = 1, ulong workspace = 1024,
            bool no_bias = true, ConvolutionCudnnTune? cudnn_tune = null, bool cudnn_off = false,
            ConvolutionLayout? layout = null, string symbol_name = "")
        {
            if (stride == null) stride = new Shape();
            if (dilate == null) dilate = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("Convolution")
                .SetParam("kernel", kernel)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("pad", pad)
                .SetParam("num_filter", num_filter)
                .SetParam("num_group", num_group)
                .SetParam("workspace", workspace)
                .SetParam("no_bias", no_bias)
                .SetParam("cudnn_tune", MxUtil.EnumToString(cudnn_tune, ConvolutionCudnnTuneConvert))
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("layout", MxUtil.EnumToString(layout, ConvolutionLayoutConvert))
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Connectionist Temporal Classification Loss.</para>
        ///     <para> </para>
        ///     <para>.. note:: The existing alias ``contrib_CTCLoss`` is deprecated.</para>
        ///     <para> </para>
        ///     <para>The shapes of the inputs and outputs:</para>
        ///     <para> </para>
        ///     <para>- **data**: `(sequence_length, batch_size, alphabet_size)`</para>
        ///     <para>- **label**: `(batch_size, label_sequence_length)`</para>
        ///     <para>- **out**: `(batch_size)`</para>
        ///     <para> </para>
        ///     <para>The `data` tensor consists of sequences of activation vectors (without applying softmax),</para>
        ///     <para>with i-th channel in the last dimension corresponding to i-th label</para>
        ///     <para>for i between 0 and alphabet_size-1 (i.e always 0-indexed).</para>
        ///     <para>Alphabet size should include one additional value reserved for blank label.</para>
        ///     <para>When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for</para>
        ///     <para>activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be</para>
        ///     <para>reserved for blank label.</para>
        ///     <para> </para>
        ///     <para>``label`` is an index matrix of integers. When `blank_label` is ``"first"``,</para>
        ///     <para>the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,</para>
        ///     <para>when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.</para>
        ///     <para> </para>
        ///     <para>If a sequence of labels is shorter than *label_sequence_length*, use the special</para>
        ///     <para>padding value at the end of the sequence to conform it to the correct</para>
        ///     <para>length. The padding value is `0` when `blank_label` is ``"first"``, and `-1` otherwise.</para>
        ///     <para> </para>
        ///     <para>For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three sequences</para>
        ///     <para>'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the labels as</para>
        ///     <para>`{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in data tensor.</para>
        ///     <para>The resulting `label` tensor should be padded to be::</para>
        ///     <para> </para>
        ///     <para>  [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]</para>
        ///     <para> </para>
        ///     <para>When `blank_label` is ``"last"``, we can index the labels as</para>
        ///     <para>`{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label in data tensor.</para>
        ///     <para>The resulting `label` tensor should be padded to be::</para>
        ///     <para> </para>
        ///     <para>  [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]</para>
        ///     <para> </para>
        ///     <para>``out`` is a list of CTC loss values, one per example in the batch.</para>
        ///     <para> </para>
        ///     <para>See *Connectionist Temporal Classification: Labelling Unsegmented</para>
        ///     <para>Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more</para>
        ///     <para>information on the definition and the algorithm.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\ctc_loss.cc:L100</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="label">Ground-truth labels for the loss.</param>
        /// <param name="data_lengths">Lengths of data for each of the samples. Only required when use_data_lengths is true.</param>
        /// <param name="label_lengths">Lengths of labels for each of the samples. Only required when use_label_lengths is true.</param>
        /// <param name="use_data_lengths">
        ///     Whether the data lenghts are decided by `data_lengths`. If false, the lengths are equal
        ///     to the max sequence length.
        /// </param>
        /// <param name="use_label_lengths">
        ///     Whether the label lenghts are decided by `label_lengths`, or derived from
        ///     `padding_mask`. If false, the lengths are derived from the first occurrence of the value of `padding_mask`. The
        ///     value of `padding_mask` is ``0`` when first CTC label is reserved for blank, and ``-1`` when last label is reserved
        ///     for blank. See `blank_label`.
        /// </param>
        /// <param name="blank_label">
        ///     Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label
        ///     values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If
        ///     "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in
        ///     the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol CTCLoss(Symbol data, Symbol label, Symbol data_lengths, Symbol label_lengths,
            bool use_data_lengths = false, bool use_label_lengths = false,
            CtclossBlankLabel blank_label = CtclossBlankLabel.First, string symbol_name = "")
        {
            return new Operator("CTCLoss")
                .SetParam("use_data_lengths", use_data_lengths)
                .SetParam("use_label_lengths", use_label_lengths)
                .SetParam("blank_label", MxUtil.EnumToString<CtclossBlankLabel>(blank_label, CtclossBlankLabelConvert))
                .SetInput("data", data)
                .SetInput("label", label)
                .SetInput("data_lengths", data_lengths)
                .SetInput("label_lengths", label_lengths)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>
        ///         Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This
        ///         operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually
        ///         reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a
        ///         larger output while preserving the connectivity pattern.
        ///     </para>
        /// </summary>
        /// <param name="data">Input tensor to the deconvolution operation.</param>
        /// <param name="weight">Weights representing the kernel.</param>
        /// <param name="bias">Bias added to the result after the deconvolution operation.</param>
        /// <param name="kernel">
        ///     Deconvolution kernel size: (w,), (h, w) or (d, h, w). This is same as the kernel size used for the
        ///     corresponding convolution
        /// </param>
        /// <param name="stride">
        ///     The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). Defaults to 1 for
        ///     each dimension.
        /// </param>
        /// <param name="dilate">
        ///     Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). Defaults to 1 for each
        ///     dimension.
        /// </param>
        /// <param name="pad">
        ///     The amount of implicit zero padding added during convolution for each dimension of the input: (w,),
        ///     (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice. If `target_shape` is set, `pad` will be ignored and
        ///     a padding that will generate the target shape will be used. Defaults to no padding.
        /// </param>
        /// <param name="adj">
        ///     Adjustment for output shape: (w,), (h, w) or (d, h, w). If `target_shape` is set, `adj` will be
        ///     ignored and computed accordingly.
        /// </param>
        /// <param name="target_shape">Shape of the output tensor: (w,), (h, w) or (d, h, w).</param>
        /// <param name="num_filter">Number of output filters.</param>
        /// <param name="num_group">Number of groups partition.</param>
        /// <param name="workspace">
        ///     Maximum temporary workspace allowed (MB) in deconvolution.This parameter has two usages. When
        ///     CUDNN is not used, it determines the effective batch size of the deconvolution kernel. When CUDNN is used, it
        ///     controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is
        ///     used.
        /// </param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="cudnn_tune">Whether to pick convolution algorithm by running performance test.</param>
        /// <param name="cudnn_off">Turn off cudnn for this layer.</param>
        /// <param name="layout">
        ///     Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and
        ///     NCDHW for 3d.NHWC and NDHWC are only supported on GPU.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Deconvolution(NDArray data, NDArray weight, NDArray bias, Shape kernel, uint num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, Shape adj = null, Shape target_shape = null,
            uint num_group = 1, ulong workspace = 512, bool no_bias = true, DeconvolutionCudnnTune? cudnn_tune = null,
            bool cudnn_off = false, DeconvolutionLayout? layout = null, string symbol_name = "")
        {
            if (stride == null) stride = new Shape();
            if (dilate == null) dilate = new Shape();
            if (pad == null) pad = new Shape();
            if (adj == null) adj = new Shape();
            if (target_shape == null) target_shape = new Shape();

            return new Operator("Deconvolution")
                .SetParam("kernel", kernel)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("pad", pad)
                .SetParam("adj", adj)
                .SetParam("target_shape", target_shape)
                .SetParam("num_filter", num_filter)
                .SetParam("num_group", num_group)
                .SetParam("workspace", workspace)
                .SetParam("no_bias", no_bias)
                .SetParam("cudnn_tune", MxUtil.EnumToString(cudnn_tune, DeconvolutionCudnnTuneConvert))
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("layout", MxUtil.EnumToString(layout, DeconvolutionLayoutConvert))
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies dropout operation to input array.</para>
        ///     <para> </para>
        ///     <para>- During training, each element of the input is set to zero with probability p.</para>
        ///     <para>  The whole array is rescaled by :math:`1/(1-p)` to keep the expected</para>
        ///     <para>  sum of the input unchanged.</para>
        ///     <para> </para>
        ///     <para>- During testing, this operator does not change the input if mode is 'training'.</para>
        ///     <para>  If mode is 'always', the same computaion as during training will be applied.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  random.seed(998)</para>
        ///     <para>  input_array = array([[3., 0.5,  -0.5,  2., 7.],</para>
        ///     <para>                      [2., -0.4,   7.,  3., 0.2]])</para>
        ///     <para>  a = symbol.Variable('a')</para>
        ///     <para>  dropout = symbol.Dropout(a, p = 0.2)</para>
        ///     <para>  executor = dropout.simple_bind(a = input_array.shape)</para>
        ///     <para> </para>
        ///     <para>  ## If training</para>
        ///     <para>  executor.forward(is_train = True, a = input_array)</para>
        ///     <para>  executor.outputs</para>
        ///     <para>  [[ 3.75   0.625 -0.     2.5    8.75 ]</para>
        ///     <para>   [ 2.5   -0.5    8.75   3.75   0.   ]]</para>
        ///     <para> </para>
        ///     <para>  ## If testing</para>
        ///     <para>  executor.forward(is_train = False, a = input_array)</para>
        ///     <para>  executor.outputs</para>
        ///     <para>  [[ 3.     0.5   -0.5    2.     7.   ]</para>
        ///     <para>   [ 2.    -0.4    7.     3.     0.2  ]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\dropout.cc:L76</para>
        /// </summary>
        /// <param name="data">Input array to which dropout will be applied.</param>
        /// <param name="p">Fraction of the input that gets dropped out during training time.</param>
        /// <param name="mode">Whether to only turn on dropout during training or to also turn on for inference.</param>
        /// <param name="axes">Axes for variational dropout kernel.</param>
        /// <param name="cudnn_off">Whether to turn off cudnn in dropout operator. This option is ignored if axes is specified.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Dropout(Symbol data, float p = 0.5f, DropoutMode mode = DropoutMode.Training,
            Shape axes = null, bool? cudnn_off = false, string symbol_name = "")
        {
            if (axes == null) axes = new Shape();

            return new Operator("Dropout")
                .SetParam("p", p)
                .SetParam("mode", MxUtil.EnumToString<DropoutMode>(mode, DropoutModeConvert))
                .SetParam("axes", axes)
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies a linear transformation: :math:`Y = XW^T + b`.</para>
        ///     <para> </para>
        ///     <para>If ``flatten`` is set to be true, then the shapes are:</para>
        ///     <para> </para>
        ///     <para>- **data**: `(batch_size, x1, x2, ..., xn)`</para>
        ///     <para>- **weight**: `(num_hidden, x1 * x2 * ... * xn)`</para>
        ///     <para>- **bias**: `(num_hidden,)`</para>
        ///     <para>- **out**: `(batch_size, num_hidden)`</para>
        ///     <para> </para>
        ///     <para>If ``flatten`` is set to be false, then the shapes are:</para>
        ///     <para> </para>
        ///     <para>- **data**: `(x1, x2, ..., xn, input_dim)`</para>
        ///     <para>- **weight**: `(num_hidden, input_dim)`</para>
        ///     <para>- **bias**: `(num_hidden,)`</para>
        ///     <para>- **out**: `(x1, x2, ..., xn, num_hidden)`</para>
        ///     <para> </para>
        ///     <para>The learnable parameters include both ``weight`` and ``bias``.</para>
        ///     <para> </para>
        ///     <para>If ``no_bias`` is set to be true, then the ``bias`` term is ignored.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para> </para>
        ///     <para>    The sparse support for FullyConnected is limited to forward evaluation with `row_sparse`</para>
        ///     <para>    weight and bias, where the length of `weight.indices` and `bias.indices` must be equal</para>
        ///     <para>    to `num_hidden`. This could be useful for model inference with `row_sparse` weights</para>
        ///     <para>    trained with importance sampling or noise contrastive estimation.</para>
        ///     <para> </para>
        ///     <para>    To compute linear transformation with 'csr' sparse data, sparse.dot is recommended instead</para>
        ///     <para>    of sparse.FullyConnected.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\fully_connected.cc:L277</para>
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <param name="weight">Weight matrix.</param>
        /// <param name="bias">Bias parameter.</param>
        /// <param name="num_hidden">Number of hidden nodes of the output.</param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="flatten">Whether to collapse all but the first axis of the input data tensor.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol FullyConnected(Symbol data, Symbol weight, Symbol bias = null, int num_hidden = 0,
            bool no_bias = true, bool flatten = true, string symbol_name = "")
        {
            return new Operator("FullyConnected")
                .SetParam("num_hidden", num_hidden)
                .SetParam("no_bias", no_bias)
                .SetParam("flatten", flatten)
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Layer normalization.</para>
        ///     <para> </para>
        ///     <para>Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as</para>
        ///     <para>well as offset ``beta``.</para>
        ///     <para> </para>
        ///     <para>Assume the input has more than one dimension and we normalize along axis 1.</para>
        ///     <para>We first compute the mean and variance along this axis and then </para>
        ///     <para>compute the normalized output, which has the same shape as input, as following:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta</para>
        ///     <para> </para>
        ///     <para>Both ``gamma`` and ``beta`` are learnable parameters.</para>
        ///     <para> </para>
        ///     <para>Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.</para>
        ///     <para> </para>
        ///     <para>Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``</para>
        ///     <para>have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and</para>
        ///     <para>``data_std``. Note that no gradient will be passed through these two outputs.</para>
        ///     <para> </para>
        ///     <para>The parameter ``axis`` specifies which axis of the input shape denotes</para>
        ///     <para>the 'channel' (separately normalized groups).  The default is -1, which sets the channel</para>
        ///     <para>axis to be the last item in the input shape.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\layer_norm.cc:L94</para>
        /// </summary>
        /// <param name="data">Input data to layer normalization</param>
        /// <param name="gamma">gamma array</param>
        /// <param name="beta">beta array</param>
        /// <param name="axis">
        ///     The axis to perform layer normalization. Usually, this should be be axis of the channel dimension.
        ///     Negative values means indexing from right to left.
        /// </param>
        /// <param name="eps">An `epsilon` parameter to prevent division by 0.</param>
        /// <param name="output_mean_var">Output the mean and std calculated along the given axis.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LayerNorm(Symbol data, Symbol gamma, Symbol beta, int axis = -1, float eps = 1e-05f,
            bool output_mean_var = false, string symbol_name = "")
        {
            return new Operator("LayerNorm")
                .SetParam("axis", axis)
                .SetParam("eps", eps)
                .SetParam("output_mean_var", output_mean_var)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies local response normalization to the input.</para>
        ///     <para> </para>
        ///     <para>The local response normalization layer performs "lateral inhibition" by normalizing</para>
        ///     <para>over local input regions.</para>
        ///     <para> </para>
        ///     <para>If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position</para>
        ///     <para>:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized</para>
        ///     <para>activity :math:`b_{x,y}^{i}` is given by the expression:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>
        ///         b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \frac{\alpha}{n} \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1,
        ///         i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}
        ///     </para>
        ///     <para> </para>
        ///     <para>
        ///         where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the
        ///         total
        ///     </para>
        ///     <para>number of kernels in the layer.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\lrn.cc:L164</para>
        /// </summary>
        /// <param name="data">Input data to LRN</param>
        /// <param name="alpha">The variance scaling parameter :math:`lpha` in the LRN expression.</param>
        /// <param name="beta">The power parameter :math:`eta` in the LRN expression.</param>
        /// <param name="knorm">The parameter :math:`k` in the LRN expression.</param>
        /// <param name="nsize">normalization window width in elements.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LRN(Symbol data, uint nsize, float alpha = 0.0001f, float beta = 0.75f, float knorm = 2f,
            string symbol_name = "")
        {
            return new Operator("LRN")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetParam("knorm", knorm)
                .SetParam("nsize", nsize)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs pooling on the input.</para>
        ///     <para> </para>
        ///     <para>The shapes for 1-D pooling are</para>
        ///     <para> </para>
        ///     <para>- **data** and **out**: *(batch_size, channel, width)* (NCW layout) or</para>
        ///     <para>  *(batch_size, width, channel)* (NWC layout),</para>
        ///     <para> </para>
        ///     <para>The shapes for 2-D pooling are</para>
        ///     <para> </para>
        ///     <para>- **data** and **out**: *(batch_size, channel, height, width)* (NCHW layout) or</para>
        ///     <para>  *(batch_size, height, width, channel)* (NHWC layout),</para>
        ///     <para> </para>
        ///     <para>    out_height = f(height, kernel[0], pad[0], stride[0])</para>
        ///     <para>    out_width = f(width, kernel[1], pad[1], stride[1])</para>
        ///     <para> </para>
        ///     <para>The definition of *f* depends on ``pooling_convention``, which has two options:</para>
        ///     <para> </para>
        ///     <para>- **valid** (default)::</para>
        ///     <para> </para>
        ///     <para>    f(x, k, p, s) = floor((x+2*p-k)/s)+1</para>
        ///     <para> </para>
        ///     <para>- **full**, which is compatible with Caffe::</para>
        ///     <para> </para>
        ///     <para>    f(x, k, p, s) = ceil((x+2*p-k)/s)+1</para>
        ///     <para> </para>
        ///     <para>But ``global_pool`` is set to be true, then do a global pooling, namely reset</para>
        ///     <para>``kernel=(height, width)``.</para>
        ///     <para> </para>
        ///     <para>Three pooling options are supported by ``pool_type``:</para>
        ///     <para> </para>
        ///     <para>- **avg**: average pooling</para>
        ///     <para>- **max**: max pooling</para>
        ///     <para>- **sum**: sum pooling</para>
        ///     <para>- **lp**: Lp pooling</para>
        ///     <para> </para>
        ///     <para>For 3-D pooling, an additional *depth* dimension is added before</para>
        ///     <para>*height*. Namely the input data and output will have shape *(batch_size, channel, depth,</para>
        ///     <para>height, width)* (NCDHW layout) or *(batch_size, depth, height, width, channel)* (NDHWC layout).</para>
        ///     <para> </para>
        ///     <para>Notes on Lp pooling:</para>
        ///     <para> </para>
        ///     <para>Lp pooling was first introduced by this paper: https://arxiv.org/pdf/1204.3968.pdf.</para>
        ///     <para>L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.</para>
        ///     <para>We can see that Lp pooling stands between those two, in practice the most common value for p is 2.</para>
        ///     <para> </para>
        ///     <para>For each window ``X``, the mathematical expression for Lp pooling is:</para>
        ///     <para> </para>
        ///     <para>:math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\pooling.cc:L416</para>
        /// </summary>
        /// <param name="data">Input data to the pooling operator.</param>
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
        public static Symbol Pooling(Symbol data, Shape kernel = null, PoolingType pool_type = PoolingType.Max,
            bool global_pool = false, bool cudnn_off = false,
            PoolingConvention pooling_convention = PoolingConvention.Valid, Shape stride = null,
            Shape pad = null, int? p_value = null, bool? count_include_pad = null, string layout = null,
            string symbol_name = "")
        {
            if (kernel == null) kernel = new Shape();
            if (stride == null) stride = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("Pooling")
                .SetParam("kernel", kernel)
                .SetParam("pool_type", MxUtil.EnumToString<PoolingType>(pool_type, PoolingPoolTypeConvert))
                .SetParam("global_pool", global_pool)
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("pooling_convention",
                    MxUtil.EnumToString<PoolingConvention>(pooling_convention, PoolingPoolingConventionConvert))
                .SetParam("stride", stride)
                .SetParam("pad", pad)
                .SetParam("p_value", p_value)
                .SetParam("count_include_pad", count_include_pad)
                .SetParam("layout", layout)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies the softmax function.</para>
        ///     <para> </para>
        ///     <para>The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}</para>
        ///     <para> </para>
        ///     <para>for :math:`j = 1, ..., K`</para>
        ///     <para> </para>
        ///     <para>t is the temperature parameter in softmax function. By default, t equals 1.0</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.  1.  1.]</para>
        ///     <para>       [ 1.  1.  1.]]</para>
        ///     <para> </para>
        ///     <para>  softmax(x,axis=0) = [[ 0.5  0.5  0.5]</para>
        ///     <para>                       [ 0.5  0.5  0.5]]</para>
        ///     <para> </para>
        ///     <para>  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],</para>
        ///     <para>                       [ 0.33333334,  0.33333334,  0.33333334]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\softmax.cc:L93</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="axis">The axis along which to compute softmax.</param>
        /// <param name="temperature">Temperature parameter in softmax</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to the same as input's dtype if not
        ///     defined (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Softmax(Symbol data, int axis = -1, double? temperature = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("softmax")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies the softmin function.</para>
        ///     <para> </para>
        ///     <para>The resulting array contains elements in the range (0,1) and the elements along the given axis sum</para>
        ///     <para>up to 1.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   softmin(\mathbf{z/t})_j = \frac{e^{-z_j/t}}{\sum_{k=1}^K e^{-z_k/t}}</para>
        ///     <para> </para>
        ///     <para>for :math:`j = 1, ..., K`</para>
        ///     <para> </para>
        ///     <para>t is the temperature parameter in softmax function. By default, t equals 1.0</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.  2.  3.]</para>
        ///     <para>       [ 3.  2.  1.]]</para>
        ///     <para> </para>
        ///     <para>  softmin(x,axis=0) = [[ 0.88079703,  0.5,  0.11920292],</para>
        ///     <para>                       [ 0.11920292,  0.5,  0.88079703]]</para>
        ///     <para> </para>
        ///     <para>  softmin(x,axis=1) = [[ 0.66524094,  0.24472848,  0.09003057],</para>
        ///     <para>                       [ 0.09003057,  0.24472848,  0.66524094]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\softmax.cc:L153</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="axis">The axis along which to compute softmax.</param>
        /// <param name="temperature">Temperature parameter in softmax</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to the same as input's dtype if not
        ///     defined (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Softmin(Symbol data, int axis = -1, double? temperature = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("softmin")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes the log softmax of the input.</para>
        ///     <para>This is equivalent to computing softmax followed by log.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  >>> x = mx.nd.array([1, 2, .1])</para>
        ///     <para>  >>> mx.nd.log_softmax(x).asnumpy()</para>
        ///     <para>  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)</para>
        ///     <para> </para>
        ///     <para>  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )</para>
        ///     <para>  >>> mx.nd.log_softmax(x, axis=0).asnumpy()</para>
        ///     <para>  array([[-0.34115392, -0.69314718, -1.24115396],</para>
        ///     <para>         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="axis">The axis along which to compute softmax.</param>
        /// <param name="temperature">Temperature parameter in softmax</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to the same as input's dtype if not
        ///     defined (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogSoftmax(Symbol data, int axis = -1, double? temperature = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("log_softmax")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies softmax activation to input. This is intended for internal layers.</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para> </para>
        ///     <para>  This operator has been deprecated, please use `softmax`.</para>
        ///     <para> </para>
        ///     <para>If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.</para>
        ///     <para>This is the default mode.</para>
        ///     <para> </para>
        ///     <para>If `mode` = ``channel``, this operator will compute a k-class softmax at each position</para>
        ///     <para>of each instance, where `k` = ``num_channel``. This mode can only be used when the input array</para>
        ///     <para>has at least 3 dimensions.</para>
        ///     <para>This can be used for `fully convolutional network`, `image segmentation`, etc.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],</para>
        ///     <para>  >>>                            [2., -.4, 7.,   3., 0.2]])</para>
        ///     <para>  >>> softmax_act = mx.nd.SoftmaxActivation(input_array)</para>
        ///     <para>  >>> print softmax_act.asnumpy()</para>
        ///     <para>  [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]</para>
        ///     <para>   [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\nn\softmax_activation.cc:L59</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="mode">
        ///     Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance.
        ///     If set to ``channel``, It computes cross channel softmax for each position of each instance.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SoftmaxActivation(Symbol data, SoftmaxMode mode = SoftmaxMode.Instance,
            string symbol_name = "")
        {
            return new Operator("SoftmaxActivation")
                .SetParam("mode", MxUtil.EnumToString<SoftmaxMode>(mode, SoftmaxactivationModeConvert))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>
        ///         Performs nearest neighbor/bilinear up sampling to inputs. Bilinear upsampling makes use of deconvolution.
        ///         Therefore, provide 2 inputs - data and weight.
        ///     </para>
        /// </summary>
        /// <param name="data">
        ///     Array of tensors to upsample. For bilinear upsampling, there should be 2 inputs - 1 data and 1
        ///     weight.
        /// </param>
        /// <param name="scale">Up sampling scale</param>
        /// <param name="num_filter">
        ///     Input filter. Only used by bilinear sample_type.Since bilinear upsampling uses deconvolution,
        ///     num_filters is set to the number of channels.
        /// </param>
        /// <param name="sample_type">upsampling method</param>
        /// <param name="multi_input_mode">
        ///     How to handle multiple input. concat means concatenate upsampled images along the
        ///     channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
        /// </param>
        /// <param name="num_args">
        ///     Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of
        ///     output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling
        ///     this must be 2; 1 input and 1 weight.
        /// </param>
        /// <param name="workspace">Tmp workspace for deconvolution (MB)</param>
        /// <returns>returns new symbol</returns>
        public static Symbol UpSampling(SymbolList data, int scale, UpsamplingSampleType sample_type, int num_args,
            int num_filter = 0, UpsamplingMultiInputMode multi_input_mode = UpsamplingMultiInputMode.Concat,
            ulong workspace = 512, string symbol_name = "")
        {
            return new Operator("UpSampling")
                .SetParam("scale", scale)
                .SetParam("num_filter", num_filter)
                .SetParam("sample_type",
                    MxUtil.EnumToString<UpsamplingSampleType>(sample_type, UpsamplingSampleTypeConvert))
                .SetParam("multi_input_mode",
                    MxUtil.EnumToString<UpsamplingMultiInputMode>(multi_input_mode, UpsamplingMultiInputModeConvert))
                .SetParam("num_args", num_args)
                .SetParam("workspace", workspace)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for SignSGD optimizer.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> W_t = W_{t-1} - \eta_t \text{sign}(g_t)</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> weight = weight - learning_rate * sign(gradient)</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   - sparse ndarray not supported for this optimizer yet.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L59</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SignsgdUpdate(Symbol weight, Symbol grad, float lr, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("signsgd_update")
                .SetParam("lr", lr)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>SIGN momentUM (Signum) optimizer.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> m_t = \beta m_{t-1} + (1 - \beta) g_t\\</para>
        ///     <para> W_t = W_{t-1} - \eta_t \text{sign}(m_t)</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> state = momentum * state + (1-momentum) * gradient</para>
        ///     <para> weight = weight - learning_rate * sign(state)</para>
        ///     <para> </para>
        ///     <para>Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   - sparse ndarray not supported for this optimizer yet.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L88</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mom">Momentum</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="wd_lh">
        ///     The amount of weight decay that does not go into gradient/momentum calculationsotherwise do weight
        ///     decay algorithmically only.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SignumUpdate(Symbol weight, Symbol grad, Symbol mom, float lr, float momentum = 0f,
            float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, float wd_lh = 0f,
            string symbol_name = "")
        {
            return new Operator("signum_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("wd_lh", wd_lh)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for Stochastic Gradient Descent (SDG) optimizer.</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> weight = weight - learning_rate * (gradient + wd * weight)</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L325</para>
        /// </summary>
        /// <param name="data">Weights</param>
        /// <param name="lrs">Learning rates.</param>
        /// <param name="wds">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="num_weights">Number of updated weights.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MultiSgdUpdate(SymbolList data, float[] lrs, float[] wds,
            float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1, string symbol_name = "")
        {
            return new Operator("multi_sgd_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Momentum update function for Stochastic Gradient Descent (SGD) optimizer.</para>
        ///     <para> </para>
        ///     <para>Momentum update has better convergence rates on neural networks. Mathematically it looks</para>
        ///     <para>like below:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  v_1 = \alpha * \nabla J(W_0)\\</para>
        ///     <para>  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\</para>
        ///     <para>  W_t = W_{t-1} + v_t</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para>  v = momentum * v - learning_rate * gradient</para>
        ///     <para>  weight += v</para>
        ///     <para> </para>
        ///     <para>Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L370</para>
        /// </summary>
        /// <param name="data">Weights, gradients and momentum</param>
        /// <param name="lrs">Learning rates.</param>
        /// <param name="wds">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="num_weights">Number of updated weights.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MultiSgdMomUpdate(SymbolList data, float[] lrs, float[] wds, float momentum = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1, string symbol_name = "")
        {
            return new Operator("multi_sgd_mom_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("momentum", momentum)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> weight = weight - learning_rate * (gradient + wd * weight)</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L413</para>
        /// </summary>
        /// <param name="data">Weights</param>
        /// <param name="lrs">Learning rates.</param>
        /// <param name="wds">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="num_weights">Number of updated weights.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MultiMpSgdUpdate(SymbolList data, float[] lrs, float[] wds,
            float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1, string symbol_name = "")
        {
            return new Operator("multi_mp_sgd_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Momentum update function for multi-precision Stochastic Gradient Descent (SGD) optimizer.</para>
        ///     <para> </para>
        ///     <para>Momentum update has better convergence rates on neural networks. Mathematically it looks</para>
        ///     <para>like below:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  v_1 = \alpha * \nabla J(W_0)\\</para>
        ///     <para>  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\</para>
        ///     <para>  W_t = W_{t-1} + v_t</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para>  v = momentum * v - learning_rate * gradient</para>
        ///     <para>  weight += v</para>
        ///     <para> </para>
        ///     <para>Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L468</para>
        /// </summary>
        /// <param name="data">Weights</param>
        /// <param name="lrs">Learning rates.</param>
        /// <param name="wds">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="num_weights">Number of updated weights.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MultiMpSgdMomUpdate(SymbolList data, float[] lrs, float[] wds,
            float momentum = 0f, float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1,
            string symbol_name = "")
        {
            return new Operator("multi_mp_sgd_mom_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("momentum", momentum)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for Stochastic Gradient Descent (SGD) optimizer.</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> weight = weight - learning_rate * (gradient + wd * weight)</para>
        ///     <para> </para>
        ///     <para>However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,</para>
        ///     <para>only the row slices whose indices appear in grad.indices are updated::</para>
        ///     <para> </para>
        ///     <para> for row in gradient.indices:</para>
        ///     <para>     weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L520</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="lazy_update">If true, lazy updates are applied if gradient's stype is row_sparse.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SgdUpdate(Symbol weight, Symbol grad, float lr, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, bool lazy_update = true, string symbol_name = "")
        {
            return new Operator("sgd_update")
                .SetParam("lr", lr)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Momentum update function for Stochastic Gradient Descent (SGD) optimizer.</para>
        ///     <para> </para>
        ///     <para>Momentum update has better convergence rates on neural networks. Mathematically it looks</para>
        ///     <para>like below:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  v_1 = \alpha * \nabla J(W_0)\\</para>
        ///     <para>  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\</para>
        ///     <para>  W_t = W_{t-1} + v_t</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para>  v = momentum * v - learning_rate * gradient</para>
        ///     <para>  weight += v</para>
        ///     <para> </para>
        ///     <para>Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.</para>
        ///     <para> </para>
        ///     <para>However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage</para>
        ///     <para>type is the same as momentum's storage type,</para>
        ///     <para>only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::</para>
        ///     <para> </para>
        ///     <para>  for row in gradient.indices:</para>
        ///     <para>      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]</para>
        ///     <para>      weight[row] += v[row]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L561</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mom">Momentum</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="lazy_update">
        ///     If true, lazy updates are applied if gradient's stype is row_sparse and both weight and
        ///     momentum have the same stype
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SgdMomUpdate(Symbol weight, Symbol grad, Symbol mom, float lr, float momentum = 0f,
            float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, bool lazy_update = true,
            string symbol_name = "")
        {
            return new Operator("sgd_mom_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Updater function for multi-precision sgd optimizer</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">gradient</param>
        /// <param name="weight32">Weight32</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="lazy_update">If true, lazy updates are applied if gradient's stype is row_sparse.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MpSgdUpdate(Symbol weight, Symbol grad, Symbol weight32, float lr, float wd = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, bool lazy_update = true, string symbol_name = "")
        {
            return new Operator("mp_sgd_update")
                .SetParam("lr", lr)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("weight32", weight32)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Updater function for multi-precision sgd optimizer</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mom">Momentum</param>
        /// <param name="weight32">Weight32</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="momentum">The decay rate of momentum estimates at each epoch.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="lazy_update">
        ///     If true, lazy updates are applied if gradient's stype is row_sparse and both weight and
        ///     momentum have the same stype
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol MpSgdMomUpdate(Symbol weight, Symbol grad, Symbol mom, Symbol weight32, float lr,
            float momentum = 0f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f,
            bool lazy_update = true, string symbol_name = "")
        {
            return new Operator("mp_sgd_mom_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .SetInput("weight32", weight32)
                .CreateSymbol(symbol_name);
        }

        public static Symbol NAGMomUpdate(NDArray weight, NDArray grad, NDArray mom, float lr, float momentum = 0,
           float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("nag_mom_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .CreateSymbol(symbol_name);
        }

        public static Symbol MPNAGMomUpdate(NDArray weight, NDArray grad, NDArray mom, NDArray weight32, float lr,
            float momentum = 0, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("mp_nag_mom_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .SetInput("weight32", weight32)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>The FTML optimizer described in</para>
        ///     <para>*FTML - Follow the Moving Leader in Deep Learning*,</para>
        ///     <para>available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\</para>
        ///     <para> d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)</para>
        ///     <para> \sigma_t = d_t - \beta_1 d_{t-1}</para>
        ///     <para> z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}</para>
        ///     <para> W_t = - \frac{ z_t }{ d_t }</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L636</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="d">Internal state ``d_t``</param>
        /// <param name="v">Internal state ``v_t``</param>
        /// <param name="z">Internal state ``z_t``</param>
        /// <param name="lr">Learning rate.</param>
        /// <param name="beta1">Generally close to 0.5.</param>
        /// <param name="beta2">Generally close to 1.</param>
        /// <param name="epsilon">Epsilon to prevent div 0.</param>
        /// <param name="t">Number of update.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_grad">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol FtmlUpdate(Symbol weight, Symbol grad, Symbol d, Symbol v, Symbol z, float lr, int t,
            float beta1 = 0.6f, float beta2 = 0.999f, double epsilon = 1e-08, float wd = 0f, float rescale_grad = 1f,
            float clip_grad = -1f, string symbol_name = "")
        {
            return new Operator("ftml_update")
                .SetParam("lr", lr)
                .SetParam("beta1", beta1)
                .SetParam("beta2", beta2)
                .SetParam("epsilon", epsilon)
                .SetParam("t", t)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_grad", clip_grad)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("d", d)
                .SetInput("v", v)
                .SetInput("z", z)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for Adam optimizer. Adam is seen as a generalization</para>
        ///     <para>of AdaGrad.</para>
        ///     <para> </para>
        ///     <para>Adam update consists of the following steps, where g represents gradient and m, v</para>
        ///     <para>are 1st and 2nd order moment estimates (mean and variance).</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para> g_t = \nabla J(W_{t-1})\\</para>
        ///     <para> m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\</para>
        ///     <para> v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\</para>
        ///     <para> W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> m = beta1*m + (1-beta1)*grad</para>
        ///     <para> v = beta2*v + (1-beta2)*(grad**2)</para>
        ///     <para> w += - learning_rate * m / (sqrt(v) + epsilon)</para>
        ///     <para> </para>
        ///     <para>However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage</para>
        ///     <para>type of weight is the same as those of m and v,</para>
        ///     <para>only the row slices whose indices appear in grad.indices are updated (for w, m and v)::</para>
        ///     <para> </para>
        ///     <para> for row in grad.indices:</para>
        ///     <para>     m[row] = beta1*m[row] + (1-beta1)*grad[row]</para>
        ///     <para>     v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)</para>
        ///     <para>     w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L684</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="mean">Moving mean</param>
        /// <param name="var">Moving variance</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="beta1">The decay rate for the 1st moment estimates.</param>
        /// <param name="beta2">The decay rate for the 2nd moment estimates.</param>
        /// <param name="epsilon">A small constant for numerical stability.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="lazy_update">
        ///     If true, lazy updates are applied if gradient's stype is row_sparse and all of w, m and v
        ///     have the same stype
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol AdamUpdate(Symbol weight, Symbol grad, Symbol mean, Symbol var, float lr,
            float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, bool lazy_update = true, string symbol_name = "")
        {
            return new Operator("adam_update")
                .SetParam("lr", lr)
                .SetParam("beta1", beta1)
                .SetParam("beta2", beta2)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mean", mean)
                .SetInput("var", var)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for `RMSProp` optimizer.</para>
        ///     <para> </para>
        ///     <para>`RMSprop` is a variant of stochastic gradient descent where the gradients are</para>
        ///     <para>divided by a cache which grows with the sum of squares of recent gradients?</para>
        ///     <para> </para>
        ///     <para>`RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively</para>
        ///     <para>tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for</para>
        ///     <para>each parameter monotonically over the course of training.</para>
        ///     <para>While this is analytically motivated for convex optimizations, it may not be ideal</para>
        ///     <para>for non-convex problems. `RMSProp` deals with this heuristically by allowing the</para>
        ///     <para>learning rates to rebound as the denominator decays over time.</para>
        ///     <para> </para>
        ///     <para>Define the Root Mean Square (RMS) error criterion of the gradient as</para>
        ///     <para>:math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents</para>
        ///     <para>gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.</para>
        ///     <para> </para>
        ///     <para>The :math:`E[g^2]_t` is given by:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2</para>
        ///     <para> </para>
        ///     <para>The update step is</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t</para>
        ///     <para> </para>
        ///     <para>The RMSProp code follows the version in</para>
        ///     <para>http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf</para>
        ///     <para>Tieleman & Hinton, 2012.</para>
        ///     <para> </para>
        ///     <para>Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate</para>
        ///     <para>:math:`\eta` to be 0.001.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L742</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="n">n</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="gamma1">The decay rate of momentum estimates.</param>
        /// <param name="epsilon">A small constant for numerical stability.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="clip_weights">
        ///     Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RmspropUpdate(Symbol weight, Symbol grad, Symbol n, float lr, float gamma1 = 0.95f,
            float epsilon = 1e-08f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f,
            float clip_weights = -1f, string symbol_name = "")
        {
            return new Operator("rmsprop_update")
                .SetParam("lr", lr)
                .SetParam("gamma1", gamma1)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("clip_weights", clip_weights)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("n", n)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for RMSPropAlex optimizer.</para>
        ///     <para> </para>
        ///     <para>`RMSPropAlex` is non-centered version of `RMSProp`.</para>
        ///     <para> </para>
        ///     <para>Define :math:`E[g^2]_t` is the decaying average over past squared gradient and</para>
        ///     <para>:math:`E[g]_t` is the decaying average over past gradient.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\</para>
        ///     <para>  E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\</para>
        ///     <para>  \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\</para>
        ///     <para> </para>
        ///     <para>The update step is</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  \theta_{t+1} = \theta_t + \Delta_t</para>
        ///     <para> </para>
        ///     <para>The RMSPropAlex code follows the version in</para>
        ///     <para>http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.</para>
        ///     <para> </para>
        ///     <para>Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`</para>
        ///     <para>to be 0.9 and the learning rate :math:`\eta` to be 0.0001.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L781</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="n">n</param>
        /// <param name="g">g</param>
        /// <param name="delta">delta</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="gamma1">Decay rate.</param>
        /// <param name="gamma2">Decay rate.</param>
        /// <param name="epsilon">A small constant for numerical stability.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <param name="clip_weights">
        ///     Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RmspropalexUpdate(Symbol weight, Symbol grad, Symbol n, Symbol g, Symbol delta, float lr,
            float gamma1 = 0.95f, float gamma2 = 0.9f, float epsilon = 1e-08f, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, float clip_weights = -1f, string symbol_name = "")
        {
            return new Operator("rmspropalex_update")
                .SetParam("lr", lr)
                .SetParam("gamma1", gamma1)
                .SetParam("gamma2", gamma2)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("clip_weights", clip_weights)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("n", n)
                .SetInput("g", g)
                .SetInput("delta", delta)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for Ftrl optimizer.</para>
        ///     <para>Referenced from *Ad Click Prediction: a View from the Trenches*, available at</para>
        ///     <para>http://dl.acm.org/citation.cfm?id=2488200.</para>
        ///     <para> </para>
        ///     <para>It updates the weights using::</para>
        ///     <para> </para>
        ///     <para> rescaled_grad = clip(grad * rescale_grad, clip_gradient)</para>
        ///     <para> z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate</para>
        ///     <para> n += rescaled_grad**2</para>
        ///     <para> w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)</para>
        ///     <para> </para>
        ///     <para>If w, z and n are all of ``row_sparse`` storage type,</para>
        ///     <para>only the row slices whose indices appear in grad.indices are updated (for w, z and n)::</para>
        ///     <para> </para>
        ///     <para> for row in grad.indices:</para>
        ///     <para>     rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)</para>
        ///     <para>
        ///         z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] /
        ///         learning_rate
        ///     </para>
        ///     <para>     n[row] += rescaled_grad[row]**2</para>
        ///     <para>
        ///         w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row])
        ///         > lamda1)
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L821</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="z">z</param>
        /// <param name="n">Square of grad</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="lamda1">The L1 regularization coefficient.</param>
        /// <param name="beta">Per-Coordinate Learning Rate beta.</param>
        /// <param name="wd">
        ///     Weight decay augments the objective function with a regularization term that penalizes large weights.
        ///     The penalty scales with the square of the magnitude of each weight.
        /// </param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol FtrlUpdate(Symbol weight, Symbol grad, Symbol z, Symbol n, float lr, float lamda1 = 0.01f,
            float beta = 1f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, string symbol_name = "")
        {
            return new Operator("ftrl_update")
                .SetParam("lr", lr)
                .SetParam("lamda1", lamda1)
                .SetParam("beta", beta)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("z", z)
                .SetInput("n", n)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Update function for AdaGrad optimizer.</para>
        ///     <para> </para>
        ///     <para>Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,</para>
        ///     <para>and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.</para>
        ///     <para> </para>
        ///     <para>Updates are applied by::</para>
        ///     <para> </para>
        ///     <para>    rescaled_grad = clip(grad * rescale_grad, clip_gradient)</para>
        ///     <para>    history = history + square(rescaled_grad)</para>
        ///     <para>    w = w - learning_rate * rescaled_grad / sqrt(history + epsilon)</para>
        ///     <para> </para>
        ///     <para>Note that non-zero values for the weight decay option are not supported.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\optimizer_op.cc:L854</para>
        /// </summary>
        /// <param name="weight">Weight</param>
        /// <param name="grad">Gradient</param>
        /// <param name="history">History</param>
        /// <param name="lr">Learning rate</param>
        /// <param name="epsilon">epsilon</param>
        /// <param name="wd">weight decay</param>
        /// <param name="rescale_grad">Rescale gradient to grad = rescale_grad*grad.</param>
        /// <param name="clip_gradient">
        ///     Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SparseAdagradUpdate(Symbol weight, Symbol grad, Symbol history, float lr,
            float epsilon = 1e-07f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f,
            string symbol_name = "")
        {
            return new Operator("_sparse_adagrad_update")
                .SetParam("lr", lr)
                .SetParam("epsilon", epsilon)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("history", history)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Pads an input array with a constant or edge values of the array.</para>
        ///     <para> </para>
        ///     <para>.. note:: `Pad` is deprecated. Use `pad` instead.</para>
        ///     <para> </para>
        ///     <para>.. note:: Current implementation only supports 4D and 5D input arrays with padding applied</para>
        ///     <para>   only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.</para>
        ///     <para> </para>
        ///     <para>This operation pads an input array with either a `constant_value` or edge values</para>
        ///     <para>along each axis of the input array. The amount of padding is specified by `pad_width`.</para>
        ///     <para> </para>
        ///     <para>`pad_width` is a tuple of integer padding widths for each axis of the format</para>
        ///     <para>``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``</para>
        ///     <para>where ``N`` is the number of dimensions of the array.</para>
        ///     <para> </para>
        ///     <para>For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values</para>
        ///     <para>to add before and after the elements of the array along dimension ``N``.</para>
        ///     <para>The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,</para>
        ///     <para>``after_2`` must be 0.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[[[  1.   2.   3.]</para>
        ///     <para>          [  4.   5.   6.]]</para>
        ///     <para> </para>
        ///     <para>         [[  7.   8.   9.]</para>
        ///     <para>          [ 10.  11.  12.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>        [[[ 11.  12.  13.]</para>
        ///     <para>          [ 14.  15.  16.]]</para>
        ///     <para> </para>
        ///     <para>         [[ 17.  18.  19.]</para>
        ///     <para>          [ 20.  21.  22.]]]]</para>
        ///     <para> </para>
        ///     <para>   pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =</para>
        ///     <para> </para>
        ///     <para>         [[[[  1.   1.   2.   3.   3.]</para>
        ///     <para>            [  1.   1.   2.   3.   3.]</para>
        ///     <para>            [  4.   4.   5.   6.   6.]</para>
        ///     <para>            [  4.   4.   5.   6.   6.]]</para>
        ///     <para> </para>
        ///     <para>           [[  7.   7.   8.   9.   9.]</para>
        ///     <para>            [  7.   7.   8.   9.   9.]</para>
        ///     <para>            [ 10.  10.  11.  12.  12.]</para>
        ///     <para>            [ 10.  10.  11.  12.  12.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>          [[[ 11.  11.  12.  13.  13.]</para>
        ///     <para>            [ 11.  11.  12.  13.  13.]</para>
        ///     <para>            [ 14.  14.  15.  16.  16.]</para>
        ///     <para>            [ 14.  14.  15.  16.  16.]]</para>
        ///     <para> </para>
        ///     <para>           [[ 17.  17.  18.  19.  19.]</para>
        ///     <para>            [ 17.  17.  18.  19.  19.]</para>
        ///     <para>            [ 20.  20.  21.  22.  22.]</para>
        ///     <para>            [ 20.  20.  21.  22.  22.]]]]</para>
        ///     <para> </para>
        ///     <para>   pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =</para>
        ///     <para> </para>
        ///     <para>         [[[[  0.   0.   0.   0.   0.]</para>
        ///     <para>            [  0.   1.   2.   3.   0.]</para>
        ///     <para>            [  0.   4.   5.   6.   0.]</para>
        ///     <para>            [  0.   0.   0.   0.   0.]]</para>
        ///     <para> </para>
        ///     <para>           [[  0.   0.   0.   0.   0.]</para>
        ///     <para>            [  0.   7.   8.   9.   0.]</para>
        ///     <para>            [  0.  10.  11.  12.   0.]</para>
        ///     <para>            [  0.   0.   0.   0.   0.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>          [[[  0.   0.   0.   0.   0.]</para>
        ///     <para>            [  0.  11.  12.  13.   0.]</para>
        ///     <para>            [  0.  14.  15.  16.   0.]</para>
        ///     <para>            [  0.   0.   0.   0.   0.]]</para>
        ///     <para> </para>
        ///     <para>           [[  0.   0.   0.   0.   0.]</para>
        ///     <para>            [  0.  17.  18.  19.   0.]</para>
        ///     <para>            [  0.  20.  21.  22.   0.]</para>
        ///     <para>            [  0.   0.   0.   0.   0.]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\pad.cc:L766</para>
        /// </summary>
        /// <param name="data">An n-dimensional input array.</param>
        /// <param name="mode">
        ///     Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the
        ///     input array "reflect" pads by reflecting values with respect to the edges.
        /// </param>
        /// <param name="pad_width">
        ///     Widths of the padding regions applied to the edges of each axis. It is a tuple of integer
        ///     padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length
        ///     ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but
        ///     flattened.
        /// </param>
        /// <param name="constant_value">The value used for padding when `mode` is "constant".</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Pad(Symbol data, PadMode mode, Shape pad_width, double constant_value = 0,
            string symbol_name = "")
        {
            return new Operator("Pad")
                .SetParam("mode", MxUtil.EnumToString<PadMode>(mode, PadModeConvert))
                .SetParam("pad_width", pad_width)
                .SetParam("constant_value", constant_value)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Flattens the input array into a 2-D array by collapsing the higher dimensions.</para>
        ///     <para> </para>
        ///     <para>.. note:: `Flatten` is deprecated. Use `flatten` instead.</para>
        ///     <para> </para>
        ///     <para>For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes</para>
        ///     <para>the input array into an output array of shape ``(d1, d2*...*dk)``.</para>
        ///     <para> </para>
        ///     <para>Note that the bahavior of this function is different from numpy.ndarray.flatten,</para>
        ///     <para>which behaves similar to mxnet.ndarray.reshape((-1,)).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>    x = [[</para>
        ///     <para>        [1,2,3],</para>
        ///     <para>        [4,5,6],</para>
        ///     <para>        [7,8,9]</para>
        ///     <para>    ],</para>
        ///     <para>    [    [1,2,3],</para>
        ///     <para>        [4,5,6],</para>
        ///     <para>        [7,8,9]</para>
        ///     <para>    ]],</para>
        ///     <para> </para>
        ///     <para>    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],</para>
        ///     <para>       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L315</para>
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Flatten(Symbol data, string symbol_name = "")
        {
            return new Operator("Flatten")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>uniform distributions on the intervals given by *[low,high)*.</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as input arrays.</para>
        ///     <para>Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input values at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input arrays.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   low = [ 0.0, 2.5 ]</para>
        ///     <para>   high = [ 1.0, 3.7 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_uniform(low, high) = [ 0.40451524,  3.18687344]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],</para>
        ///     <para>                                           [ 3.18687344,  3.68352246]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L277</para>
        /// </summary>
        /// <param name="low">Lower bounds of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <param name="high">Upper bounds of the distributions.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleUniform(Symbol low, Symbol high, Shape shape = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_sample_uniform")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("low", low)
                .SetInput("high", high)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as input arrays.</para>
        ///     <para>Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input values at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input arrays.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   mu = [ 0.0, 2.5 ]</para>
        ///     <para>   sigma = [ 1.0, 3.7 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_normal(mu, sigma) = [-0.56410581,  0.95934606]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],</para>
        ///     <para>                                          [ 0.95934606,  4.48287058]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L279</para>
        /// </summary>
        /// <param name="mu">Means of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <param name="sigma">Standard deviations of the distributions.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleNormal(Symbol mu, Symbol sigma, Shape shape = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_sample_normal")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("mu", mu)
                .SetInput("sigma", sigma)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>gamma distributions with parameters *alpha* (shape) and *beta* (scale).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as input arrays.</para>
        ///     <para>Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input values at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input arrays.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   alpha = [ 0.0, 2.5 ]</para>
        ///     <para>   beta = [ 1.0, 0.7 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],</para>
        ///     <para>                                           [ 2.25797319,  1.70734084]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L282</para>
        /// </summary>
        /// <param name="alpha">Alpha (shape) parameters of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <param name="beta">Beta (scale) parameters of the distributions.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleGamma(Symbol alpha, Symbol beta, Shape shape = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_sample_gamma")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("alpha", alpha)
                .SetInput("beta", beta)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>exponential distributions with parameters lambda (rate).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as an input array.</para>
        ///     <para>Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input value at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input array.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   lam = [ 1.0, 8.5 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_exponential(lam) = [ 0.51837951,  0.09994757]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],</para>
        ///     <para>                                         [ 0.09994757,  0.50447971]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L284</para>
        /// </summary>
        /// <param name="lam">Lambda (rate) parameters of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleExponential(Symbol lam, Shape shape = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_sample_exponential")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("lam", lam)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>Poisson distributions with parameters lambda (rate).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as an input array.</para>
        ///     <para>Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input value at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input array.</para>
        ///     <para> </para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   lam = [ 1.0, 8.5 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_poisson(lam) = [  0.,  13.]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_poisson(lam, shape=(2)) = [[  0.,   4.],</para>
        ///     <para>                                     [ 13.,   8.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L286</para>
        /// </summary>
        /// <param name="lam">Lambda (rate) parameters of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SamplePoisson(Symbol lam, Shape shape = null, DType dtype = null, string symbol_name = "")
        {
            return new Operator("_sample_poisson")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("lam", lam)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as input arrays.</para>
        ///     <para>Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input values at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input arrays.</para>
        ///     <para> </para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   k = [ 20, 49 ]</para>
        ///     <para>   p = [ 0.4 , 0.77 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_negative_binomial(k, p) = [ 15.,  16.]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],</para>
        ///     <para>                                                [ 16.,  12.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L289</para>
        /// </summary>
        /// <param name="k">Limits of unsuccessful experiments.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <param name="p">Failure probabilities in each experiment.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleNegativeBinomial(Symbol k, Symbol p, Shape shape = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_sample_negative_binomial")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("k", k)
                .SetInput("p", p)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple</para>
        ///     <para>generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).</para>
        ///     <para> </para>
        ///     <para>The parameters of the distributions are provided as input arrays.</para>
        ///     <para>Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*</para>
        ///     <para>be the shape specified as the parameter of the operator, and *m* be the dimension</para>
        ///     <para>of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.</para>
        ///     <para> </para>
        ///     <para>For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*</para>
        ///     <para>will be an *m*-dimensional array that holds randomly drawn samples from the distribution</para>
        ///     <para>which is parameterized by the input values at index *i*. If the shape parameter of the</para>
        ///     <para>operator is not set, then one sample will be drawn per distribution and the output array</para>
        ///     <para>has the same shape as the input arrays.</para>
        ///     <para> </para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   mu = [ 2.0, 2.5 ]</para>
        ///     <para>   alpha = [ 1.0, 0.1 ]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],</para>
        ///     <para>                                                                 [ 3.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\multisample_op.cc:L293</para>
        /// </summary>
        /// <param name="mu">Means of the distributions.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <param name="alpha">Alpha (dispersion) parameters of the distributions.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleGeneralizedNegativeBinomial(Symbol mu, Symbol alpha, Shape shape = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_sample_generalized_negative_binomial")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("mu", mu)
                .SetInput("alpha", alpha)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Concurrent sampling from multiple multinomial distributions.</para>
        ///     <para> </para>
        ///     <para>*data* is an *n* dimensional array whose last dimension has length *k*, where</para>
        ///     <para>*k* is the number of possible outcomes of each multinomial distribution. This</para>
        ///     <para>operator will draw *shape* samples from each distribution. If shape is empty</para>
        ///     <para>one sample will be drawn from each distribution.</para>
        ///     <para> </para>
        ///     <para>If *get_prob* is true, a second array containing log likelihood of the drawn</para>
        ///     <para>samples will also be returned. This is usually used for reinforcement learning</para>
        ///     <para>where you can provide reward as head gradient for this array to estimate</para>
        ///     <para>gradient.</para>
        ///     <para> </para>
        ///     <para>Note that the input distribution must be normalized, i.e. *data* must sum to</para>
        ///     <para>1 along its last axis.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]</para>
        ///     <para> </para>
        ///     <para>   // Draw a single sample for each distribution</para>
        ///     <para>   sample_multinomial(probs) = [3, 0]</para>
        ///     <para> </para>
        ///     <para>   // Draw a vector containing two samples for each distribution</para>
        ///     <para>   sample_multinomial(probs, shape=(2)) = [[4, 2],</para>
        ///     <para>                                           [0, 0]]</para>
        ///     <para> </para>
        ///     <para>   // requests log likelihood</para>
        ///     <para>   sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]</para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">Distribution probabilities. Must sum to one on the last axis.</param>
        /// <param name="shape">Shape to be sampled from each random distribution.</param>
        /// <param name="get_prob">
        ///     Whether to also return the log probability of sampled result. This is usually used for
        ///     differentiating through stochastic variables, e.g. in reinforcement learning.
        /// </param>
        /// <param name="dtype">DType of the output in case this can't be inferred.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleMultinomial(Symbol data, Shape shape = null, bool get_prob = false,
            DType dtype = null, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Int32;

            return new Operator("_sample_multinomial")
                .SetParam("shape", shape)
                .SetParam("get_prob", get_prob)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a uniform distribution.</para>
        ///     <para> </para>
        ///     <para>.. note:: The existing alias ``uniform`` is deprecated.</para>
        ///     <para> </para>
        ///     <para>Samples are uniformly distributed over the half-open interval *[low, high)*</para>
        ///     <para>(includes *low*, but excludes *high*).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],</para>
        ///     <para>                                          [ 0.54488319,  0.84725171]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L96</para>
        /// </summary>
        /// <param name="low">Lower bound of the distribution.</param>
        /// <param name="high">Upper bound of the distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomUniform(float low = 0f, float high = 1f, Shape shape = null, Context ctx = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_uniform")
                .SetParam("low", low)
                .SetParam("high", high)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a normal (Gaussian) distribution.</para>
        ///     <para> </para>
        ///     <para>.. note:: The existing alias ``normal`` is deprecated.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*</para>
        ///     <para>(standard deviation).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],</para>
        ///     <para>                                          [-1.23474145,  1.55807114]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L113</para>
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Standard deviation of the distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomNormal(float loc = 0f, float scale = 1f, Shape shape = null, Context ctx = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_normal")
                .SetParam("loc", loc)
                .SetParam("scale", scale)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a gamma distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],</para>
        ///     <para>                                            [ 3.91697288,  3.65933681]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L125</para>
        /// </summary>
        /// <param name="alpha">Alpha parameter (shape) of the gamma distribution.</param>
        /// <param name="beta">Beta parameter (scale) of the gamma distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomGamma(float alpha = 1f, float beta = 1f, Shape shape = null, Context ctx = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_gamma")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from an exponential distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],</para>
        ///     <para>                                      [ 0.04146638,  0.31715935]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L137</para>
        /// </summary>
        /// <param name="lam">Lambda parameter (rate) of the exponential distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomExponential(float lam = 1f, Shape shape = null, Context ctx = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_exponential")
                .SetParam("lam", lam)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a Poisson distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],</para>
        ///     <para>                                  [ 4.,  6.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L150</para>
        /// </summary>
        /// <param name="lam">Lambda parameter (rate) of the Poisson distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomPoisson(float lam = 1f, Shape shape = null, Context ctx = null, DType dtype = null,
            string symbol_name = "")
        {
            return new Operator("_random_poisson")
                .SetParam("lam", lam)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a negative binomial distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a negative binomial distribution parametrized by</para>
        ///     <para>*k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],</para>
        ///     <para>                                                 [ 2.,  5.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L164</para>
        /// </summary>
        /// <param name="k">Limit of unsuccessful experiments.</param>
        /// <param name="p">Failure probability in each experiment.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomNegativeBinomial(int k = 1, float p = 1f, Shape shape = null, Context ctx = null,
            DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_negative_binomial")
                .SetParam("k", k)
                .SetParam("p", p)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a generalized negative binomial distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a generalized negative binomial distribution parametrized by</para>
        ///     <para>*mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the</para>
        ///     <para>number of unsuccessful experiments (generalized to real numbers).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],</para>
        ///     <para>                                                                    [ 6.,  4.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L179</para>
        /// </summary>
        /// <param name="mu">Mean of the negative binomial distribution.</param>
        /// <param name="alpha">Alpha (dispersion) parameter of the negative binomial distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">
        ///     DType of the output in case this can't be inferred. Defaults to float32 if not defined
        ///     (dtype=None).
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomGeneralizedNegativeBinomial(float mu = 1f, float alpha = 1f, Shape shape = null,
            Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_generalized_negative_binomial")
                .SetParam("mu", mu)
                .SetParam("alpha", alpha)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a discrete uniform distribution.</para>
        ///     <para> </para>
        ///     <para>Samples are uniformly distributed over the half-open interval *[low, high)*</para>
        ///     <para>(includes *low*, but excludes *high*).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   randint(low=0, high=5, shape=(2,2)) = [[ 0,  2],</para>
        ///     <para>                                          [ 3,  1]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L193</para>
        /// </summary>
        /// <param name="low">Lower bound of the distribution.</param>
        /// <param name="high">Upper bound of the distribution.</param>
        /// <param name="shape">Shape of the output.</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.</param>
        /// <param name="dtype">DType of the output in case this can't be inferred. Defaults to int32 if not defined (dtype=None).</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomRandint(float[] low, float[] high, Shape shape = null,
            Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            return new Operator("_random_randint")
                .SetParam("low", low)
                .SetParam("high", high)
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a uniform distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are uniformly distributed over the half-open interval *[low, high)*</para>
        ///     <para>(includes *low*, but excludes *high*).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   uniform(low=0, high=1, data=ones(2,2)) = [[ 0.60276335,  0.85794562],</para>
        ///     <para>                                             [ 0.54488319,  0.84725171]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L208</para>
        /// </summary>
        /// <param name="low">Lower bound of the distribution.</param>
        /// <param name="high">Upper bound of the distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomUniformLike(Symbol data, float low = 0f, float high = 1f, string symbol_name = "")
        {
            return new Operator("_random_uniform_like")
                .SetParam("low", low)
                .SetParam("high", high)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a normal (Gaussian) distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*</para>
        ///     <para>(standard deviation).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   normal(loc=0, scale=1, data=ones(2,2)) = [[ 1.89171135, -1.16881478],</para>
        ///     <para>                                             [-1.23474145,  1.55807114]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L220</para>
        /// </summary>
        /// <param name="loc">Mean of the distribution.</param>
        /// <param name="scale">Standard deviation of the distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomNormalLike(Symbol data, float loc = 0f, float scale = 1f, string symbol_name = "")
        {
            return new Operator("_random_normal_like")
                .SetParam("loc", loc)
                .SetParam("scale", scale)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a gamma distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   gamma(alpha=9, beta=0.5, data=ones(2,2)) = [[ 7.10486984,  3.37695289],</para>
        ///     <para>                                               [ 3.91697288,  3.65933681]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L231</para>
        /// </summary>
        /// <param name="alpha">Alpha parameter (shape) of the gamma distribution.</param>
        /// <param name="beta">Beta parameter (scale) of the gamma distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomGammaLike(Symbol data, float alpha = 1f, float beta = 1f, string symbol_name = "")
        {
            return new Operator("_random_gamma_like")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from an exponential distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   exponential(lam=4, data=ones(2,2)) = [[ 0.0097189 ,  0.08999364],</para>
        ///     <para>                                         [ 0.04146638,  0.31715935]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L242</para>
        /// </summary>
        /// <param name="lam">Lambda parameter (rate) of the exponential distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomExponentialLike(Symbol data, float lam = 1f, string symbol_name = "")
        {
            return new Operator("_random_exponential_like")
                .SetParam("lam", lam)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a Poisson distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   poisson(lam=4, data=ones(2,2)) = [[ 5.,  2.],</para>
        ///     <para>                                     [ 4.,  6.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L254</para>
        /// </summary>
        /// <param name="lam">Lambda parameter (rate) of the Poisson distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomPoissonLike(Symbol data, float lam = 1f, string symbol_name = "")
        {
            return new Operator("_random_poisson_like")
                .SetParam("lam", lam)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a negative binomial distribution according to the input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a negative binomial distribution parametrized by</para>
        ///     <para>*k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   negative_binomial(k=3, p=0.4, data=ones(2,2)) = [[ 4.,  7.],</para>
        ///     <para>                                                    [ 2.,  5.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L267</para>
        /// </summary>
        /// <param name="k">Limit of unsuccessful experiments.</param>
        /// <param name="p">Failure probability in each experiment.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomNegativeBinomialLike(Symbol data, int k = 1, float p = 1f, string symbol_name = "")
        {
            return new Operator("_random_negative_binomial_like")
                .SetParam("k", k)
                .SetParam("p", p)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from a generalized negative binomial distribution according to the</para>
        ///     <para>input array shape.</para>
        ///     <para> </para>
        ///     <para>Samples are distributed according to a generalized negative binomial distribution parametrized by</para>
        ///     <para>*mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the</para>
        ///     <para>number of unsuccessful experiments (generalized to real numbers).</para>
        ///     <para>Samples will always be returned as a floating point data type.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   generalized_negative_binomial(mu=2.0, alpha=0.3, data=ones(2,2)) = [[ 2.,  1.],</para>
        ///     <para>                                                                       [ 6.,  4.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\sample_op.cc:L283</para>
        /// </summary>
        /// <param name="mu">Mean of the negative binomial distribution.</param>
        /// <param name="alpha">Alpha (dispersion) parameter of the negative binomial distribution.</param>
        /// <param name="data">The input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RandomGeneralizedNegativeBinomialLike(Symbol data, float mu = 1f, float alpha = 1f,
            string symbol_name = "")
        {
            return new Operator("_random_generalized_negative_binomial_like")
                .SetParam("mu", mu)
                .SetParam("alpha", alpha)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Randomly shuffle the elements.</para>
        ///     <para> </para>
        ///     <para>This shuffles the array along the first axis.</para>
        ///     <para>The order of the elements in each subarray does not change.</para>
        ///     <para>For example, if a 2D array is given, the order of the rows randomly changes,</para>
        ///     <para>but the order of the elements in each row does not change.</para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">Data to be shuffled.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Shuffle(Symbol data, string symbol_name = "")
        {
            return new Operator("_shuffle")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Draw random samples from an an approximately log-uniform</para>
        ///     <para>or Zipfian distribution without replacement.</para>
        ///     <para> </para>
        ///     <para>This operation takes a 2-D shape `(batch_size, num_sampled)`,</para>
        ///     <para>and randomly generates *num_sampled* samples from the range of integers [0, range_max)</para>
        ///     <para>for each instance in the batch.</para>
        ///     <para> </para>
        ///     <para>The elements in each instance are drawn without replacement from the base distribution.</para>
        ///     <para>The base distribution for this operator is an approximately log-uniform or Zipfian distribution:</para>
        ///     <para> </para>
        ///     <para>  P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)</para>
        ///     <para> </para>
        ///     <para>Additionaly, it also returns the number of trials used to obtain `num_sampled` samples for</para>
        ///     <para>each instance in the batch.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   samples, trials = _sample_unique_zipfian(750000, shape=(4, 8192))</para>
        ///     <para>   unique(samples[0]) = 8192</para>
        ///     <para>   unique(samples[3]) = 8192</para>
        ///     <para>   trials[0] = 16435</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\random\unique_sample_op.cc:L66</para>
        /// </summary>
        /// <param name="range_max">The number of possible classes.</param>
        /// <param name="shape">
        ///     2-D shape of the output, where shape[0] is the batch size, and shape[1] is the number of candidates
        ///     to sample for each batch.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SampleUniqueZipfian(int range_max, Shape shape = null, string symbol_name = "")
        {
            return new Operator("_sample_unique_zipfian")
                .SetParam("range_max", range_max)
                .SetParam("shape", shape)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes and optimizes for squared loss during backward propagation.</para>
        ///     <para>Just outputs ``data`` during forward propagation.</para>
        ///     <para> </para>
        ///     <para>
        ///         If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target
        ///         value,
        ///     </para>
        ///     <para>then the squared loss estimated over :math:`n` samples is defined as</para>
        ///     <para> </para>
        ///     <para>
        ///         :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i
        ///         - \hat{\textbf{y}}_i  \rVert_2`
        ///     </para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   Use the LinearRegressionOutput as the final output layer of a net.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``label`` can be ``default`` or ``csr``</para>
        ///     <para> </para>
        ///     <para>- LinearRegressionOutput(default, default) = default</para>
        ///     <para>- LinearRegressionOutput(default, csr) = default</para>
        ///     <para> </para>
        ///     <para>
        ///         By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression
        ///         outputs of a training example.
        ///     </para>
        ///     <para>The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\regression_output.cc:L92</para>
        /// </summary>
        /// <param name="data">Input data to the function.</param>
        /// <param name="label">Input label to the function.</param>
        /// <param name="grad_scale">Scale the gradient by a float factor</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinearRegressionOutput(Symbol data, Symbol label, float grad_scale = 1f,
            string symbol_name = "")
        {
            return new Operator("LinearRegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes mean absolute error of the input.</para>
        ///     <para> </para>
        ///     <para>MAE is a risk metric corresponding to the expected value of the absolute error.</para>
        ///     <para> </para>
        ///     <para>
        ///         If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target
        ///         value,
        ///     </para>
        ///     <para>then the mean absolute error (MAE) estimated over :math:`n` samples is defined as</para>
        ///     <para> </para>
        ///     <para>
        ///         :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i -
        ///         \hat{\textbf{y}}_i \rVert_1`
        ///     </para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   Use the MAERegressionOutput as the final output layer of a net.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``label`` can be ``default`` or ``csr``</para>
        ///     <para> </para>
        ///     <para>- MAERegressionOutput(default, default) = default</para>
        ///     <para>- MAERegressionOutput(default, csr) = default</para>
        ///     <para> </para>
        ///     <para>
        ///         By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression
        ///         outputs of a training example.
        ///     </para>
        ///     <para>The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\regression_output.cc:L120</para>
        /// </summary>
        /// <param name="data">Input data to the function.</param>
        /// <param name="label">Input label to the function.</param>
        /// <param name="grad_scale">Scale the gradient by a float factor</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MAERegressionOutput(Symbol data, Symbol label, float grad_scale = 1f,
            string symbol_name = "")
        {
            return new Operator("MAERegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies a logistic function to the input.</para>
        ///     <para> </para>
        ///     <para>The logistic function, also known as the sigmoid function, is computed as</para>
        ///     <para>:math:`\frac{1}{1+exp(-\textbf{x})}`.</para>
        ///     <para> </para>
        ///     <para>Commonly, the sigmoid is used to squash the real-valued output of a linear model</para>
        ///     <para>:math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.</para>
        ///     <para>It is suitable for binary classification or probability prediction tasks.</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>   Use the LogisticRegressionOutput as the final output layer of a net.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``label`` can be ``default`` or ``csr``</para>
        ///     <para> </para>
        ///     <para>- LogisticRegressionOutput(default, default) = default</para>
        ///     <para>- LogisticRegressionOutput(default, csr) = default</para>
        ///     <para> </para>
        ///     <para>The loss function used is the Binary Cross Entropy Loss:</para>
        ///     <para> </para>
        ///     <para>:math:`-{(y\log(p) + (1 - y)\log(1 - p))}`</para>
        ///     <para> </para>
        ///     <para>
        ///         Where `y` is the ground truth probability of positive outcome for a given example, and `p` the probability
        ///         predicted by the model. By default, gradients of this loss function are scaled by factor `1/m`, where m is the
        ///         number of regression outputs of a training example.
        ///     </para>
        ///     <para>The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\regression_output.cc:L152</para>
        /// </summary>
        /// <param name="data">Input data to the function.</param>
        /// <param name="label">Input label to the function.</param>
        /// <param name="grad_scale">Scale the gradient by a float factor</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogisticRegressionOutput(Symbol data, Symbol label, float grad_scale = 1f,
            string symbol_name = "")
        {
            return new Operator("LogisticRegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are</para>
        ///     <para>implemented, with both multi-layer and bidirectional support.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\rnn.cc:L219</para>
        /// </summary>
        /// <param name="data">Input data to RNN</param>
        /// <param name="parameters">Vector of all RNN trainable parameters concatenated</param>
        /// <param name="state">initial hidden state of the RNN</param>
        /// <param name="state_cell">initial cell state for LSTM networks (only for LSTM)</param>
        /// <param name="state_size">size of the state for each layer</param>
        /// <param name="num_layers">number of stacked layers</param>
        /// <param name="bidirectional">whether to use bidirectional recurrent layers</param>
        /// <param name="mode">the type of RNN to compute</param>
        /// <param name="p">drop rate of the dropout on the outputs of each RNN layer, except the last layer.</param>
        /// <param name="state_outputs">Whether to have the states as symbol outputs.</param>
        /// <param name="projection_size">size of project size</param>
        /// <param name="lstm_state_clip_min">
        ///     Minimum clip value of LSTM states. This option must be used together with
        ///     lstm_state_clip_max.
        /// </param>
        /// <param name="lstm_state_clip_max">
        ///     Maximum clip value of LSTM states. This option must be used together with
        ///     lstm_state_clip_min.
        /// </param>
        /// <param name="lstm_state_clip_nan">
        ///     Whether to stop NaN from propagating in state by clipping it to min/max. If clipping
        ///     range is not specified, this option is ignored.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol RNN(Symbol data, Symbol parameters, Symbol state, Symbol state_cell, uint state_size,
            uint num_layers, RNNMode mode, bool bidirectional = false, float p = 0f, bool state_outputs = false,
            int? projection_size = null, double? lstm_state_clip_min = null, double? lstm_state_clip_max = null,
            bool lstm_state_clip_nan = false, string symbol_name = "")
        {
            return new Operator("RNN")
                .SetParam("state_size", state_size)
                .SetParam("num_layers", num_layers)
                .SetParam("bidirectional", bidirectional)
                .SetParam("mode", MxUtil.EnumToString<RNNMode>(mode, RNNModeConvert))
                .SetParam("p", p)
                .SetParam("state_outputs", state_outputs)
                .SetParam("projection_size", projection_size)
                .SetParam("lstm_state_clip_min", lstm_state_clip_min)
                .SetParam("lstm_state_clip_max", lstm_state_clip_max)
                .SetParam("lstm_state_clip_nan", lstm_state_clip_nan)
                .SetInput("data", data)
                .SetInput("parameters", parameters)
                .SetInput("state", state)
                .SetInput("state_cell", state_cell)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Splits an array along a particular axis into multiple sub-arrays.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\slice_channel.cc:L107</para>
        /// </summary>
        /// <param name="data">The input</param>
        /// <param name="num_outputs">Number of splits. Note that this should evenly divide the length of the `axis`.</param>
        /// <param name="axis">Axis along which to split.</param>
        /// <param name="squeeze_axis">
        ///     If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that
        ///     setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also
        ///     `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Split(Symbol data, int num_outputs, int axis = 1, bool squeeze_axis = false,
            string symbol_name = "")
        {
            return new Operator("split")
                .SetParam("num_outputs", num_outputs)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes the gradient of cross entropy loss with respect to softmax output.</para>
        ///     <para> </para>
        ///     <para>- This operator computes the gradient in two steps.</para>
        ///     <para>  The cross entropy loss does not actually need to be computed.</para>
        ///     <para> </para>
        ///     <para>  - Applies softmax function on the input array.</para>
        ///     <para>  - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.</para>
        ///     <para> </para>
        ///     <para>- The softmax function, cross entropy loss and gradient is given by:</para>
        ///     <para> </para>
        ///     <para>  - Softmax Function:</para>
        ///     <para> </para>
        ///     <para>    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}</para>
        ///     <para> </para>
        ///     <para>  - Cross Entropy Function:</para>
        ///     <para> </para>
        ///     <para>    .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)</para>
        ///     <para> </para>
        ///     <para>  - The gradient of cross entropy loss w.r.t softmax output:</para>
        ///     <para> </para>
        ///     <para>    .. math:: \text{gradient} = \text{output} - \text{label}</para>
        ///     <para> </para>
        ///     <para>- During forward propagation, the softmax function is computed for each instance in the input array.</para>
        ///     <para> </para>
        ///     <para>  For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is</para>
        ///     <para>  :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`</para>
        ///     <para>  and `multi_output` to specify the way to compute softmax:</para>
        ///     <para> </para>
        ///     <para>  - By default, `preserve_shape` is ``false``. This operator will reshape the input array</para>
        ///     <para>    into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for</para>
        ///     <para>    each row in the reshaped array, and afterwards reshape it back to the original shape</para>
        ///     <para>    :math:`(d_1, d_2, ..., d_n)`.</para>
        ///     <para>  - If `preserve_shape` is ``true``, the softmax function will be computed along</para>
        ///     <para>    the last axis (`axis` = ``-1``).</para>
        ///     <para>  - If `multi_output` is ``true``, the softmax function will be computed along</para>
        ///     <para>    the second axis (`axis` = ``1``).</para>
        ///     <para> </para>
        ///     <para>- During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.</para>
        ///     <para>  The provided label can be a one-hot label array or a probability label array.</para>
        ///     <para> </para>
        ///     <para>  - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances</para>
        ///     <para>    with a particular label to be ignored during backward propagation. **This has no effect when</para>
        ///     <para>    softmax `output` has same shape as `label`**.</para>
        ///     <para> </para>
        ///     <para>    Example::</para>
        ///     <para> </para>
        ///     <para>      data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]</para>
        ///     <para>      label = [1,0,2,3]</para>
        ///     <para>      ignore_label = 1</para>
        ///     <para>      SoftmaxOutput(data=data, label = label,\</para>
        ///     <para>                    multi_output=true, use_ignore=true,\</para>
        ///     <para>                    ignore_label=ignore_label)</para>
        ///     <para>      ## forward softmax output</para>
        ///     <para>      [[ 0.0320586   0.08714432  0.23688284  0.64391428]</para>
        ///     <para>       [ 0.25        0.25        0.25        0.25      ]</para>
        ///     <para>       [ 0.25        0.25        0.25        0.25      ]</para>
        ///     <para>       [ 0.25        0.25        0.25        0.25      ]]</para>
        ///     <para>      ## backward gradient output</para>
        ///     <para>      [[ 0.    0.    0.    0.  ]</para>
        ///     <para>       [-0.75  0.25  0.25  0.25]</para>
        ///     <para>       [ 0.25  0.25 -0.75  0.25]</para>
        ///     <para>       [ 0.25  0.25  0.25 -0.75]]</para>
        ///     <para>      ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.</para>
        ///     <para> </para>
        ///     <para>  - The parameter `grad_scale` can be used to rescale the gradient, which is often used to</para>
        ///     <para>    give each loss function different weights.</para>
        ///     <para> </para>
        ///     <para>  - This operator also supports various ways to normalize the gradient by `normalization`,</para>
        ///     <para>    The `normalization` is applied if softmax output has different shape than the labels.</para>
        ///     <para>    The `normalization` mode can be set to the followings:</para>
        ///     <para> </para>
        ///     <para>    - ``'null'``: do nothing.</para>
        ///     <para>    - ``'batch'``: divide the gradient by the batch size.</para>
        ///     <para>    - ``'valid'``: divide the gradient by the number of instances which are not ignored.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\softmax_output.cc:L230</para>
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="label">Ground truth label.</param>
        /// <param name="grad_scale">Scales the gradient by a float factor.</param>
        /// <param name="ignore_label">
        ///     The instances whose `labels` == `ignore_label` will be ignored during backward, if
        ///     `use_ignore` is set to ``true``).
        /// </param>
        /// <param name="multi_output">
        ///     If set to ``true``, the softmax function will be computed along axis ``1``. This is applied
        ///     when the shape of input array differs from the shape of label array.
        /// </param>
        /// <param name="use_ignore">If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.</param>
        /// <param name="preserve_shape">If set to ``true``, the softmax function will be computed along the last axis (``-1``).</param>
        /// <param name="normalization">Normalizes the gradient.</param>
        /// <param name="out_grad">Multiplies gradient with output gradient element-wise.</param>
        /// <param name="smooth_alpha">
        ///     Constant for computing a label smoothed version of cross-entropyfor the backwards pass.
        ///     This constant gets subtracted from theone-hot encoding of the gold label and distributed uniformly toall other
        ///     labels.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SoftmaxOutput(Symbol data, Symbol label, float grad_scale = 1f, float ignore_label = -1f,
            bool multi_output = false, bool use_ignore = false, bool preserve_shape = false,
            SoftmaxoutputNormalization normalization = SoftmaxoutputNormalization.Valid, bool out_grad = false,
            float smooth_alpha = 0f, string symbol_name = "")
        {
            return new Operator("SoftmaxOutput")
                .SetParam("grad_scale", grad_scale)
                .SetParam("ignore_label", ignore_label)
                .SetParam("multi_output", multi_output)
                .SetParam("use_ignore", use_ignore)
                .SetParam("preserve_shape", preserve_shape)
                .SetParam("normalization",
                    MxUtil.EnumToString<SoftmaxoutputNormalization>(normalization, SoftmaxoutputNormalizationConvert))
                .SetParam("out_grad", out_grad)
                .SetParam("smooth_alpha", smooth_alpha)
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Interchanges two axes of an array.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[1, 2, 3]])</para>
        ///     <para>  swapaxes(x, 0, 1) = [[ 1],</para>
        ///     <para>                       [ 2],</para>
        ///     <para>                       [ 3]]</para>
        ///     <para> </para>
        ///     <para>  x = [[[ 0, 1],</para>
        ///     <para>        [ 2, 3]],</para>
        ///     <para>       [[ 4, 5],</para>
        ///     <para>        [ 6, 7]]]  // (2,2,2) array</para>
        ///     <para> </para>
        ///     <para> swapaxes(x, 0, 2) = [[[ 0, 4],</para>
        ///     <para>                       [ 2, 6]],</para>
        ///     <para>                      [[ 1, 5],</para>
        ///     <para>                       [ 3, 7]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\swapaxis.cc:L70</para>
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="dim1">the first axis to be swapped.</param>
        /// <param name="dim2">the second axis to be swapped.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SwapAxis(Symbol data, uint dim1 = 0, uint dim2 = 0, string symbol_name = "")
        {
            return new Operator("SwapAxis")
                .SetParam("dim1", dim1)
                .SetParam("dim2", dim2)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

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
        /// <returns>returns new symbol</returns>
        public static Symbol Argmax(Symbol data, int? axis = null, bool keepdims = false, string symbol_name = "")
        {
            return new Operator("argmax")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Argmin(Symbol data, int? axis = null, bool keepdims = false, string symbol_name = "")
        {
            return new Operator("argmin")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol ArgmaxChannel(Symbol data, string symbol_name = "")
        {
            return new Operator("argmax_channel")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Pick(Symbol data, Symbol index, int? axis = -1, bool keepdims = false,
            PickMode mode = PickMode.Clip, string symbol_name = "")
        {
            return new Operator("pick")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("mode", MxUtil.EnumToString<PickMode>(mode, PickModeConvert))
                .SetInput("data", data)
                .SetInput("index", index)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sum(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        public static Symbol Sum(Symbol data, int axis, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return Sum(data, new Shape(axis), keepdims, exclude, symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Mean(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("mean")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        public static Symbol Mean(Symbol data, int axis, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return Mean(data, new Shape(axis), keepdims, exclude, symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Prod(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("prod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Nansum(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("nansum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Nanprod(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("nanprod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Max(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("max")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Min(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("min")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastAxis(Symbol data, Shape axis = null, Shape size = null, string symbol_name = "")
        {
            if (axis == null) axis = new Shape();
            if (size == null) size = new Shape();

            return new Operator("broadcast_axis")
                .SetParam("axis", axis)
                .SetParam("size", size)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastTo(Symbol data, Shape shape = null, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();

            return new Operator("broadcast_to")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Broadcasts lhs to have the same shape as rhs.</para>
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
        ///     <para>   broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],</para>
        ///     <para>                                                   [ 1.,  2.,  3.]])</para>
        ///     <para> </para>
        ///     <para>   broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\broadcast_reduce_op_value.cc:L315</para>
        /// </summary>
        /// <param name="lhs">First input.</param>
        /// <param name="rhs">Second input.</param>
        /// <param name="lhs_axes">Axes to perform broadcast on in the first input array</param>
        /// <param name="rhs_axes">Axes to copy from the second input array</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLike(Symbol lhs, Symbol rhs, Shape lhs_axes = null, Shape rhs_axes = null,
            string symbol_name = "")
        {
            return new Operator("broadcast_like")
                .SetParam("lhs_axes", lhs_axes)
                .SetParam("rhs_axes", rhs_axes)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Norm(Symbol data, int ord = 2, Shape axis = null, NormOutDtype? out_dtype = null,
            bool keepdims = false, string symbol_name = "")
        {
            return new Operator("norm")
                .SetParam("ord", ord)
                .SetParam("axis", axis)
                .SetParam("out_dtype", MxUtil.EnumToString(out_dtype, NormOutDtypeConvert))
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol CastStorage(Symbol data, StorageStype stype, string symbol_name = "")
        {
            return new Operator("cast_storage")
                .SetParam("stype", MxUtil.EnumToString<StorageStype>(stype, CastStorageStypeConvert))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Return the elements, either from x or y, depending on the condition.</para>
        ///     <para> </para>
        ///     <para>Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,</para>
        ///     <para>depending on the elements from condition are true or false. x and y must have the same shape.</para>
        ///     <para>If condition has the same shape as x, each element in the output array is from x if the</para>
        ///     <para>corresponding element in the condition is true, and from y if false.</para>
        ///     <para> </para>
        ///     <para>If condition does not have the same shape as x, it must be a 1D array whose size is</para>
        ///     <para>the same as x's first dimension size. Each row of the output array is from x's row</para>
        ///     <para>if the corresponding element from condition is true, and from y's row if false.</para>
        ///     <para> </para>
        ///     <para>Note that all non-zero values are interpreted as ``True`` in condition.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[1, 2], [3, 4]]</para>
        ///     <para>  y = [[5, 6], [7, 8]]</para>
        ///     <para>  cond = [[0, 1], [-1, 0]]</para>
        ///     <para> </para>
        ///     <para>  where(cond, x, y) = [[5, 2], [3, 8]]</para>
        ///     <para> </para>
        ///     <para>  csr_cond = cast_storage(cond, 'csr')</para>
        ///     <para> </para>
        ///     <para>  where(csr_cond, x, y) = [[5, 2], [3, 8]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\control_flow_op.cc:L57</para>
        /// </summary>
        /// <param name="condition">condition array</param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>returns new symbol</returns>
        public static Symbol Where(Symbol condition, Symbol x, Symbol y, string symbol_name = "")
        {
            return new Operator("where")
                .SetInput("condition", condition)
                .SetInput("x", x)
                .SetInput("y", y)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Diag(Symbol data, int k = 0, int axis1 = 0, int axis2 = 1, string symbol_name = "")
        {
            return new Operator("diag")
                .SetParam("k", k)
                .SetParam("axis1", axis1)
                .SetParam("axis2", axis2)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Dot product of two arrays.</para>
        ///     <para> </para>
        ///     <para>``dot``'s behavior depends on the input array dimensions:</para>
        ///     <para> </para>
        ///     <para>- 1-D arrays: inner product of vectors</para>
        ///     <para>- 2-D arrays: matrix multiplication</para>
        ///     <para>- N-D arrays: a sum product over the last axis of the first input and the first</para>
        ///     <para>  axis of the second input</para>
        ///     <para> </para>
        ///     <para>  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the</para>
        ///     <para>  result array will have shape `(n,m,r,s)`. It is computed by::</para>
        ///     <para> </para>
        ///     <para>    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))</para>
        ///     <para>    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))</para>
        ///     <para>    dot(x,y)[0,0,1,1] = 0</para>
        ///     <para>    sum(x[0,0,:]*y[:,1,1]) = 0</para>
        ///     <para> </para>
        ///     <para>The storage type of ``dot`` output depends on storage types of inputs, transpose option and</para>
        ///     <para>forward_stype option for output storage type. Implemented sparse operations include:</para>
        ///     <para> </para>
        ///     <para>- dot(default, default, transpose_a=True/False, transpose_b=True/False) = default</para>
        ///     <para>- dot(csr, default, transpose_a=True) = default</para>
        ///     <para>- dot(csr, default, transpose_a=True) = row_sparse</para>
        ///     <para>- dot(csr, default) = default</para>
        ///     <para>- dot(csr, row_sparse) = default</para>
        ///     <para>- dot(default, csr) = csr (CPU only)</para>
        ///     <para>- dot(default, csr, forward_stype='default') = default</para>
        ///     <para>- dot(default, csr, transpose_b=True, forward_stype='default') = default</para>
        ///     <para> </para>
        ///     <para>If the combination of input storage types and forward_stype does not match any of the</para>
        ///     <para>above patterns, ``dot`` will fallback and generate output with default storage.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para> </para>
        ///     <para>    If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs will be</para>
        ///     <para>    "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad</para>
        ///     <para>    and Adam. Note that by default lazy updates is turned on, which may perform differently</para>
        ///     <para>    from standard updates. For more details, please check the Optimization API at:</para>
        ///     <para>    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\dot.cc:L77</para>
        /// </summary>
        /// <param name="lhs">The first input</param>
        /// <param name="rhs">The second input</param>
        /// <param name="transpose_a">If true then transpose the first input before dot.</param>
        /// <param name="transpose_b">If true then transpose the second input before dot.</param>
        /// <param name="forward_stype">
        ///     The desired storage type of the forward output given by user, if thecombination of input
        ///     storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand
        ///     still produce an output of the desired storage type.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Dot(Symbol lhs, Symbol rhs, bool transpose_a = false, bool transpose_b = false,
            DotForwardStype? forward_stype = null, string symbol_name = "")
        {
            return new Operator("dot")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("forward_stype", MxUtil.EnumToString(forward_stype, DotForwardStypeConvert))
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Batchwise dot product.</para>
        ///     <para> </para>
        ///     <para>``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and</para>
        ///     <para>``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.</para>
        ///     <para> </para>
        ///     <para>For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape</para>
        ///     <para>`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,</para>
        ///     <para>which is computed by::</para>
        ///     <para> </para>
        ///     <para>   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\dot.cc:L125</para>
        /// </summary>
        /// <param name="lhs">The first input</param>
        /// <param name="rhs">The second input</param>
        /// <param name="transpose_a">If true then transpose the first input before dot.</param>
        /// <param name="transpose_b">If true then transpose the second input before dot.</param>
        /// <param name="forward_stype">
        ///     The desired storage type of the forward output given by user, if thecombination of input
        ///     storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand
        ///     still produce an output of the desired storage type.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol BatchDot(Symbol lhs, Symbol rhs, bool transpose_a = false, bool transpose_b = false,
            BatchDotForwardStype? forward_stype = null, string symbol_name = "")
        {
            return new Operator("batch_dot")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("forward_stype", MxUtil.EnumToString(forward_stype, BatchDotForwardStypeConvert))
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise sum of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>`broadcast_plus` is an alias to the function `broadcast_add`.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_add(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                          [ 2.,  2.,  2.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_plus(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                           [ 2.,  2.,  2.]]</para>
        ///     <para> </para>
        ///     <para>Supported sparse operations:</para>
        ///     <para> </para>
        ///     <para>   broadcast_add(csr, dense(1D)) = dense</para>
        ///     <para>   broadcast_add(dense(1D), csr) = dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L58</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastAdd(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise difference of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>`broadcast_minus` is an alias to the function `broadcast_sub`.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_sub(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                          [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_minus(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                            [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para>Supported sparse operations:</para>
        ///     <para> </para>
        ///     <para>   broadcast_sub/minus(csr, dense(1D)) = dense</para>
        ///     <para>   broadcast_sub/minus(dense(1D), csr) = dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L106</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastSub(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_sub")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise product of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_mul(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                          [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>Supported sparse operations:</para>
        ///     <para> </para>
        ///     <para>   broadcast_mul(csr, dense(1D)) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L146</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastMul(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_mul")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise division of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 6.,  6.,  6.],</para>
        ///     <para>        [ 6.,  6.,  6.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 2.],</para>
        ///     <para>        [ 3.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_div(x, y) = [[ 3.,  3.,  3.],</para>
        ///     <para>                          [ 2.,  2.,  2.]]</para>
        ///     <para> </para>
        ///     <para>Supported sparse operations:</para>
        ///     <para> </para>
        ///     <para>   broadcast_div(csr, dense(1D)) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L187</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastDiv(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise modulo of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 8.,  8.,  8.],</para>
        ///     <para>        [ 8.,  8.,  8.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 2.],</para>
        ///     <para>        [ 3.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_mod(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                          [ 2.,  2.,  2.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_basic.cc:L222</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastMod(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_mod")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns result of first array elements raised to powers from second array, element-wise with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_power(x, y) = [[ 2.,  2.,  2.],</para>
        ///     <para>                            [ 4.,  4.,  4.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L45</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastPower(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_power")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise maximum of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>This function compares two input arrays and returns a new array having the element-wise maxima.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_maximum(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                              [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L80</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastMaximum(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_maximum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns element-wise minimum of the input arrays with broadcasting.</para>
        ///     <para> </para>
        ///     <para>This function compares two input arrays and returns a new array having the element-wise minima.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_maximum(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                              [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L115</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastMinimum(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_minimum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> Returns the hypotenuse of a right angled triangle, given its "legs"</para>
        ///     <para>with broadcasting.</para>
        ///     <para> </para>
        ///     <para>It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 3.,  3.,  3.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 4.],</para>
        ///     <para>        [ 4.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_hypot(x, y) = [[ 5.,  5.,  5.],</para>
        ///     <para>                            [ 5.,  5.,  5.]]</para>
        ///     <para> </para>
        ///     <para>   z = [[ 0.],</para>
        ///     <para>        [ 4.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_hypot(x, z) = [[ 3.,  3.,  3.],</para>
        ///     <para>                            [ 5.,  5.,  5.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_extended.cc:L156</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastHypot(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_hypot")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_equal(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                            [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L46</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                                [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L64</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastNotEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_not_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_greater(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                              [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L82</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastGreater(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_greater")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                                    [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L100</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastGreaterEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_greater_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_lesser(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                             [ 0.,  0.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L118</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLesser(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_lesser")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>
        ///         Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                                   [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L136</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLesserEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_lesser_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **logical and** with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  1.],</para>
        ///     <para>        [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 0.],</para>
        ///     <para>        [ 1.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_logical_and(x, y) = [[ 0.,  0.,  0.],</para>
        ///     <para>                                  [ 1.,  1.,  1.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L154</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLogicalAnd(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_logical_and")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **logical or** with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  0.],</para>
        ///     <para>        [ 1.,  1.,  0.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 1.],</para>
        ///     <para>        [ 0.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],</para>
        ///     <para>                                 [ 1.,  1.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L172</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLogicalOr(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_logical_or")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns the result of element-wise **logical xor** with broadcasting.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[ 1.,  1.,  0.],</para>
        ///     <para>        [ 1.,  1.,  0.]]</para>
        ///     <para> </para>
        ///     <para>   y = [[ 1.],</para>
        ///     <para>        [ 0.]]</para>
        ///     <para> </para>
        ///     <para>   broadcast_logical_xor(x, y) = [[ 0.,  0.,  1.],</para>
        ///     <para>                                  [ 1.,  1.,  0.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_broadcast_op_logic.cc:L190</para>
        /// </summary>
        /// <param name="lhs">First input to the function</param>
        /// <param name="rhs">Second input to the function</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BroadcastLogicalXor(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("broadcast_logical_xor")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Adds arguments element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``elemwise_add`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>   - elemwise_add(row_sparse, row_sparse) = row_sparse</para>
        ///     <para>   - elemwise_add(csr, csr) = csr</para>
        ///     <para>   - elemwise_add(default, csr) = default</para>
        ///     <para>   - elemwise_add(csr, default) = default</para>
        ///     <para>   - elemwise_add(default, rsp) = default</para>
        ///     <para>   - elemwise_add(rsp, default) = default</para>
        ///     <para>   - otherwise, ``elemwise_add`` generates output with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ElemwiseAdd(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("elemwise_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol GradAdd(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_grad_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Subtracts arguments element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``elemwise_sub`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>   - elemwise_sub(row_sparse, row_sparse) = row_sparse</para>
        ///     <para>   - elemwise_sub(csr, csr) = csr</para>
        ///     <para>   - elemwise_sub(default, csr) = default</para>
        ///     <para>   - elemwise_sub(csr, default) = default</para>
        ///     <para>   - elemwise_sub(default, rsp) = default</para>
        ///     <para>   - elemwise_sub(rsp, default) = default</para>
        ///     <para>   - otherwise, ``elemwise_sub`` generates output with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ElemwiseSub(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("elemwise_sub")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Multiplies arguments element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``elemwise_mul`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>   - elemwise_mul(default, default) = default</para>
        ///     <para>   - elemwise_mul(row_sparse, row_sparse) = row_sparse</para>
        ///     <para>   - elemwise_mul(default, row_sparse) = row_sparse</para>
        ///     <para>   - elemwise_mul(row_sparse, default) = row_sparse</para>
        ///     <para>   - elemwise_mul(csr, csr) = csr</para>
        ///     <para>   - otherwise, ``elemwise_mul`` generates output with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ElemwiseMul(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("elemwise_mul")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Divides arguments element-wise.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``elemwise_div`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ElemwiseDiv(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("elemwise_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Mod(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_mod")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Power(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_power")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Maximum(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_maximum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Minimum(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_minimum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Given the "legs" of a right triangle, return its hypotenuse.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_op_extended.cc:L79</para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Hypot(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_hypot")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Equal(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol NotEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_not_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Greater(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_greater")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol GreaterEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_greater_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Lesser(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_lesser")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LesserEqual(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_lesser_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalAnd(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_logical_and")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalOr(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_logical_or")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalXor(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_logical_xor")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol PlusScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_plus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MinusScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_minus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RminusScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_rminus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Multiply an array with a scalar.</para>
        ///     <para> </para>
        ///     <para>``_mul_scalar`` only operates on data array of input if input is sparse.</para>
        ///     <para> </para>
        ///     <para>For example, if input of shape (100, 100) has only 2 non zero elements,</para>
        ///     <para>i.e. input.data = [5, 6], scalar = nan,</para>
        ///     <para>it will result output.data = [nan, nan] instead of 10000 nans.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L149</para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MulScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_mul_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Divide an array with a scalar.</para>
        ///     <para> </para>
        ///     <para>``_div_scalar`` only operates on data array of input if input is sparse.</para>
        ///     <para> </para>
        ///     <para>For example, if input of shape (100, 100) has only 2 non zero elements,</para>
        ///     <para>i.e. input.data = [5, 6], scalar = nan,</para>
        ///     <para>it will result output.data = [nan, nan] instead of 10000 nans.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_scalar_op_basic.cc:L171</para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol DivScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_div_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RdivScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_rdiv_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ModScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_mod_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RmodScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_rmod_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MaximumScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_maximum_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MinimumScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_minimum_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol PowerScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_power_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol RpowerScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_rpower_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol HypotScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_hypot_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Calculate Smooth L1 Loss(lhs, scalar) by summing</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>    f(x) =</para>
        ///     <para>    \begin{cases}</para>
        ///     <para>
        ///         (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\</para>
        ///     <para>    |x|-0.5/\sigma^2,& \text{otherwise}</para>
        ///     <para>    \end{cases}</para>
        ///     <para> </para>
        ///     <para>where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]</para>
        ///     <para>  smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_binary_scalar_op_extended.cc:L104</para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SmoothL1(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("smooth_l1")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol EqualScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol NotEqualScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_not_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol GreaterScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_greater_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol GreaterEqualScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_greater_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LesserScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_lesser_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LesserEqualScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_lesser_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalAndScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_logical_and_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalOrScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_logical_or_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalXorScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_logical_xor_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Divides arguments element-wise.  If the left-hand-side input is 'row_sparse', then</para>
        ///     <para>only the values which exist in the left-hand sparse array are computed.  The 'missing' values</para>
        ///     <para>are ignored.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``_scatter_elemwise_div`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- _scatter_elemwise_div(row_sparse, row_sparse) = row_sparse</para>
        ///     <para>- _scatter_elemwise_div(row_sparse, dense) = row_sparse</para>
        ///     <para>- _scatter_elemwise_div(row_sparse, csr) = row_sparse</para>
        ///     <para>- otherwise, ``_scatter_elemwise_div`` behaves exactly like elemwise_div and generates output</para>
        ///     <para>with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">first input</param>
        /// <param name="rhs">second input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ScatterElemwiseDiv(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_scatter_elemwise_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Adds a scalar to a tensor element-wise.  If the left-hand-side input is</para>
        ///     <para>'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.</para>
        ///     <para>The 'missing' values are ignored.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``_scatter_plus_scalar`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- _scatter_plus_scalar(row_sparse, scalar) = row_sparse</para>
        ///     <para>- _scatter_plus_scalar(csr, scalar) = csr</para>
        ///     <para>- otherwise, ``_scatter_plus_scalar`` behaves exactly like _plus_scalar and generates output</para>
        ///     <para>with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ScatterPlusScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_scatter_plus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Subtracts a scalar to a tensor element-wise.  If the left-hand-side input is</para>
        ///     <para>'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.</para>
        ///     <para>The 'missing' values are ignored.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``_scatter_minus_scalar`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- _scatter_minus_scalar(row_sparse, scalar) = row_sparse</para>
        ///     <para>- _scatter_minus_scalar(csr, scalar) = csr</para>
        ///     <para>- otherwise, ``_scatter_minus_scalar`` behaves exactly like _minus_scalar and generates output</para>
        ///     <para>with default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">source input</param>
        /// <param name="scalar">scalar input</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ScatterMinusScalar(Symbol data, float scalar, string symbol_name = "")
        {
            return new Operator("_scatter_minus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Adds all input arguments element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n</para>
        ///     <para> </para>
        ///     <para>``add_n`` is potentially more efficient than calling ``add`` by `n` times.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``add_n`` output depends on storage types of inputs</para>
        ///     <para> </para>
        ///     <para>- add_n(row_sparse, row_sparse, ..) = row_sparse</para>
        ///     <para>- add_n(default, csr, default) = default</para>
        ///     <para>- add_n(any input combinations longer than 4 (>4) with at least one default type) = default</para>
        ///     <para>- otherwise, ``add_n`` falls all inputs back to default storage and generates default storage</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_sum.cc:L156</para>
        /// </summary>
        /// <param name="args">Positional input arguments</param>
        /// <returns>returns new symbol</returns>
        public static Symbol AddN(SymbolList args, string symbol_name = "")
        {
            return new Operator("add_n")
                .SetInput(args)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes rectified linear activation.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   max(features, 0)</para>
        ///     <para> </para>
        ///     <para>The storage type of ``relu`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - relu(default) = default</para>
        ///     <para>   - relu(row_sparse) = row_sparse</para>
        ///     <para>   - relu(csr) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L85</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Relu(Symbol data, string symbol_name = "")
        {
            return new Operator("relu")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes sigmoid of x element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   y = 1 / (1 + exp(-x))</para>
        ///     <para> </para>
        ///     <para>The storage type of ``sigmoid`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L101</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Sigmoid(Symbol data, string symbol_name = "")
        {
            return new Operator("sigmoid")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes hard sigmoid of x element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   y = max(0, min(1, alpha * x + beta))</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L115</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <param name="alpha">Slope of hard sigmoid</param>
        /// <param name="beta">Bias of hard sigmoid.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol HardSigmoid(Symbol data, float alpha = 0.2f, float beta = 0.5f, string symbol_name = "")
        {
            return new Operator("hard_sigmoid")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes softsign of x element-wise.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>   y = x / (1 + abs(x))</para>
        ///     <para> </para>
        ///     <para>The storage type of ``softsign`` output is always dense</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L145</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Softsign(Symbol data, string symbol_name = "")
        {
            return new Operator("softsign")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns a copy of the input.</para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:200</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Copy(Symbol data, string symbol_name = "")
        {
            return new Operator("_copy")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Stops gradient computation.</para>
        ///     <para> </para>
        ///     <para>Stops the accumulated gradient of the inputs from flowing through this operator</para>
        ///     <para>in the backward direction. In other words, this operator prevents the contribution</para>
        ///     <para>of its inputs to be taken into account for computing gradients.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  v1 = [1, 2]</para>
        ///     <para>  v2 = [0, 1]</para>
        ///     <para>  a = Variable('a')</para>
        ///     <para>  b = Variable('b')</para>
        ///     <para>  b_stop_grad = stop_gradient(3 * b)</para>
        ///     <para>  loss = MakeLoss(b_stop_grad + a)</para>
        ///     <para> </para>
        ///     <para>  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))</para>
        ///     <para>  executor.forward(is_train=True, a=v1, b=v2)</para>
        ///     <para>  executor.outputs</para>
        ///     <para>  [ 1.  5.]</para>
        ///     <para> </para>
        ///     <para>  executor.backward()</para>
        ///     <para>  executor.grad_arrays</para>
        ///     <para>  [ 0.  0.]</para>
        ///     <para>  [ 1.  1.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L281</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BlockGrad(Symbol data, string symbol_name = "")
        {
            return new Operator("BlockGrad")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Make your own loss function in network construction.</para>
        ///     <para> </para>
        ///     <para>This operator accepts a customized loss function symbol as a terminal loss and</para>
        ///     <para>the symbol should be an operator with no backward dependency.</para>
        ///     <para>The output of this function is the gradient of loss with respect to the input data.</para>
        ///     <para> </para>
        ///     <para>For example, if you are a making a cross entropy loss function. Assume ``out`` is the</para>
        ///     <para>predicted output and ``label`` is the true label, then the cross entropy can be defined as::</para>
        ///     <para> </para>
        ///     <para>  cross_entropy = label * log(out) + (1 - label) * log(1 - out)</para>
        ///     <para>  loss = make_loss(cross_entropy)</para>
        ///     <para> </para>
        ///     <para>We will need to use ``make_loss`` when we are creating our own loss function or we want to</para>
        ///     <para>combine multiple loss functions. Also we may want to stop some variables' gradients</para>
        ///     <para>from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.</para>
        ///     <para> </para>
        ///     <para>The storage type of ``make_loss`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - make_loss(default) = default</para>
        ///     <para>   - make_loss(row_sparse) = row_sparse</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\elemwise_unary_op_basic.cc:L314</para>
        /// </summary>
        /// <param name="data">The input array.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol MakeLoss(Symbol data, string symbol_name = "")
        {
            return new Operator("make_loss")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">First input.</param>
        /// <param name="rhs">Second input.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol IdentityWithAttrLikeRhs(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_identity_with_attr_like_rhs")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol ReshapeLike(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("reshape_like")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol ShapeArray(Symbol data, int? lhs_begin = null, int? lhs_end = null, int? rhs_begin = null,
            int? rhs_end = null, string symbol_name = "")
        {
            return new Operator("shape_array")
                .SetParam("lhs_begin", lhs_begin)
                .SetParam("lhs_end", lhs_end)
                .SetParam("rhs_begin", rhs_begin)
                .SetParam("rhs_end", rhs_end)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol SizeArray(Symbol data, string symbol_name = "")
        {
            return new Operator("size_array")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Cast(Symbol data, DType dtype, string symbol_name = "")
        {
            return new Operator("Cast")
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Negative(Symbol data, string symbol_name = "")
        {
            return new Operator("negative")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Reciprocal(Symbol data, string symbol_name = "")
        {
            return new Operator("reciprocal")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Abs(Symbol data, string symbol_name = "")
        {
            return new Operator("abs")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sign(Symbol data, string symbol_name = "")
        {
            return new Operator("sign")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Round(Symbol data, string symbol_name = "")
        {
            return new Operator("round")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Rint(Symbol data, string symbol_name = "")
        {
            return new Operator("rint")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Ceil(Symbol data, string symbol_name = "")
        {
            return new Operator("ceil")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Floor(Symbol data, string symbol_name = "")
        {
            return new Operator("floor")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Trunc(Symbol data, string symbol_name = "")
        {
            return new Operator("trunc")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Fix(Symbol data, string symbol_name = "")
        {
            return new Operator("fix")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Square(Symbol data, string symbol_name = "")
        {
            return new Operator("square")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sqrt(Symbol data, string symbol_name = "")
        {
            return new Operator("sqrt")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Rsqrt(Symbol data, string symbol_name = "")
        {
            return new Operator("rsqrt")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Cbrt(Symbol data, string symbol_name = "")
        {
            return new Operator("cbrt")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Erf(Symbol data, string symbol_name = "")
        {
            return new Operator("erf")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Erfinv(Symbol data, string symbol_name = "")
        {
            return new Operator("erfinv")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Rcbrt(Symbol data, string symbol_name = "")
        {
            return new Operator("rcbrt")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Exp(Symbol data, string symbol_name = "")
        {
            return new Operator("exp")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Log(Symbol data, string symbol_name = "")
        {
            return new Operator("log")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Log10(Symbol data, string symbol_name = "")
        {
            return new Operator("log10")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Log2(Symbol data, string symbol_name = "")
        {
            return new Operator("log2")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Log1P(Symbol data, string symbol_name = "")
        {
            return new Operator("log1p")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Expm1(Symbol data, string symbol_name = "")
        {
            return new Operator("expm1")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Gamma(Symbol data, string symbol_name = "")
        {
            return new Operator("gamma")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Gammaln(Symbol data, string symbol_name = "")
        {
            return new Operator("gammaln")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol LogicalNot(Symbol data, string symbol_name = "")
        {
            return new Operator("logical_not")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sin(Symbol data, string symbol_name = "")
        {
            return new Operator("sin")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Cos(Symbol data, string symbol_name = "")
        {
            return new Operator("cos")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Tan(Symbol data, string symbol_name = "")
        {
            return new Operator("tan")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arcsin(Symbol data, string symbol_name = "")
        {
            return new Operator("arcsin")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arccos(Symbol data, string symbol_name = "")
        {
            return new Operator("arccos")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arctan(Symbol data, string symbol_name = "")
        {
            return new Operator("arctan")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Degrees(Symbol data, string symbol_name = "")
        {
            return new Operator("degrees")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Radians(Symbol data, string symbol_name = "")
        {
            return new Operator("radians")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sinh(Symbol data, string symbol_name = "")
        {
            return new Operator("sinh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Cosh(Symbol data, string symbol_name = "")
        {
            return new Operator("cosh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Tanh(Symbol data, string symbol_name = "")
        {
            return new Operator("tanh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arcsinh(Symbol data, string symbol_name = "")
        {
            return new Operator("arcsinh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arccosh(Symbol data, string symbol_name = "")
        {
            return new Operator("arccosh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Arctanh(Symbol data, string symbol_name = "")
        {
            return new Operator("arctanh")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>This operators implements the histogram function.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para>  x = [[0, 1], [2, 2], [3, 4]]</para>
        ///     <para>  histo, bin_edges = histogram(data=x, bin_bounds=[], bin_cnt=5, range=(0,5))</para>
        ///     <para>  histo = [1, 1, 2, 1, 1]</para>
        ///     <para>  bin_edges = [0., 1., 2., 3., 4.]</para>
        ///     <para>  histo, bin_edges = histogram(data=x, bin_bounds=[0., 2.1, 3.])</para>
        ///     <para>  histo = [4, 1]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\histogram.cc:L136</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="bins">Input ndarray</param>
        /// <param name="bin_cnt">Number of bins for uniform case</param>
        /// <param name="range">
        ///     The lower and upper range of the bins. if not provided, range is simply (a.min(), a.max()). values
        ///     outside the range are ignored. the first element of the range must be less than or equal to the second. range
        ///     affects the automatic bin computation as well. while bin width is computed to be optimal based on the actual data
        ///     within range, the bin count will fill the entire range including portions containing no data.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Histogram(Symbol data, Symbol bins, int? bin_cnt = null, Tuple<double> range = null,
            string symbol_name = "")
        {
            return new Operator("_histogram")
                .SetParam("bin_cnt", bin_cnt)
                .SetParam("range", range)
                .SetInput("data", data)
                .SetInput("bins", bins)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Maps integer indices to vector representations (embeddings).</para>
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
        ///     <para>By default, if any index mentioned is too large, it is replaced by the index that addresses</para>
        ///     <para>the last vector in an embedding matrix.</para>
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
        ///     <para>  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],</para>
        ///     <para>                            [ 15.,  16.,  17.,  18.,  19.]],</para>
        ///     <para> </para>
        ///     <para>                           [[  0.,   1.,   2.,   3.,   4.],</para>
        ///     <para>                            [ 10.,  11.,  12.,  13.,  14.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>The storage type of weight can be either row_sparse or default.</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para> </para>
        ///     <para>    If "sparse_grad" is set to True, the storage type of gradient w.r.t weights will be</para>
        ///     <para>    "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad</para>
        ///     <para>    and Adam. Note that by default lazy updates is turned on, which may perform differently</para>
        ///     <para>    from standard updates. For more details, please check the Optimization API at:</para>
        ///     <para>    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\indexing_op.cc:L519</para>
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
        public static Symbol Embedding(Symbol data, Symbol weight, int input_dim, int output_dim, DType dtype = null,
            bool sparse_grad = false, string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("Embedding")
                .SetParam("input_dim", input_dim)
                .SetParam("output_dim", output_dim)
                .SetParam("dtype", dtype)
                .SetParam("sparse_grad", sparse_grad)
                .SetInput("data", data)
                .SetInput("weight", weight)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Takes elements from an input array along the given axis.</para>
        ///     <para> </para>
        ///     <para>This function slices the input array along a particular axis with the provided indices.</para>
        ///     <para> </para>
        ///     <para>Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis</para>
        ///     <para>dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them</para>
        ///     <para>in an output tensor of rank q + (r - 1).</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [4.  5.  6.]</para>
        ///     <para> </para>
        ///     <para>  // Trivial case, take the second element along the first axis.</para>
        ///     <para> </para>
        ///     <para>  take(x, [1]) = [ 5. ]</para>
        ///     <para> </para>
        ///     <para>  // The other trivial case, axis=-1, take the third element along the first axis</para>
        ///     <para> </para>
        ///     <para>  take(x, [3], axis=-1, mode='clip') = [ 6. ]</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.,  2.],</para>
        ///     <para>       [ 3.,  4.],</para>
        ///     <para>       [ 5.,  6.]]</para>
        ///     <para> </para>
        ///     <para>  // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0</para>
        ///     <para> </para>
        ///     <para>  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],</para>
        ///     <para>                             [ 3.,  4.]],</para>
        ///     <para> </para>
        ///     <para>                            [[ 3.,  4.],</para>
        ///     <para>                             [ 5.,  6.]]]</para>
        ///     <para> </para>
        ///     <para>  // In this case we will get rows 0 and 1, then 1 and 2 (calculated by wrapping around).</para>
        ///     <para>  // Along axis 1</para>
        ///     <para> </para>
        ///     <para>  take(x, [[0, 3], [-1, -2]], axis=1, mode='wrap') = [[[ 1.  2.]</para>
        ///     <para>                                                       [ 2.  1.]]</para>
        ///     <para> </para>
        ///     <para>                                                      [[ 3.  4.]</para>
        ///     <para>                                                       [ 4.  3.]]</para>
        ///     <para> </para>
        ///     <para>                                                      [[ 5.  6.]</para>
        ///     <para>                                                       [ 6.  5.]]]</para>
        ///     <para> </para>
        ///     <para>The storage type of ``take`` output depends upon the input storage type:</para>
        ///     <para> </para>
        ///     <para>   - take(default, default) = default</para>
        ///     <para>   - take(csr, default, axis=0) = csr</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\indexing_op.cc:L695</para>
        /// </summary>
        /// <param name="a">The input array.</param>
        /// <param name="indices">The indices of the values to be extracted.</param>
        /// <param name="axis">
        ///     The axis of input array to be taken.For input tensor of rank r, it could be in the range of [-r,
        ///     r-1]
        /// </param>
        /// <param name="mode">
        ///     Specify how out-of-bound indices bahave. Default is "clip". "clip" means clip to the range. So, if
        ///     all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.
        ///     "wrap" means to wrap around.  "raise" means to raise an error, not supported yet.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Take(Symbol a, Symbol indices, int axis = 0, TakeMode mode = TakeMode.Clip,
            string symbol_name = "")
        {
            return new Operator("take")
                .SetParam("axis", axis)
                .SetParam("mode", MxUtil.EnumToString<TakeMode>(mode, TakeModeConvert))
                .SetInput("a", a)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Takes elements from a data batch.</para>
        ///     <para> </para>
        ///     <para>.. note::</para>
        ///     <para>  `batch_take` is deprecated. Use `pick` instead.</para>
        ///     <para> </para>
        ///     <para>Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be</para>
        ///     <para>an output array of shape ``(i0,)`` with::</para>
        ///     <para> </para>
        ///     <para>  output[i] = input[i, indices[i]]</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[ 1.,  2.],</para>
        ///     <para>       [ 3.,  4.],</para>
        ///     <para>       [ 5.,  6.]]</para>
        ///     <para> </para>
        ///     <para>  // takes elements with specified indices</para>
        ///     <para>  batch_take(x, [0,1,0]) = [ 1.  4.  5.]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\indexing_op.cc:L753</para>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <param name="indices">The index array</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BatchTake(Symbol a, Symbol indices, string symbol_name = "")
        {
            return new Operator("batch_take")
                .SetInput("a", a)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Returns a one-hot array.</para>
        ///     <para> </para>
        ///     <para>The locations represented by `indices` take value `on_value`, while all</para>
        ///     <para>other locations take value `off_value`.</para>
        ///     <para> </para>
        ///     <para>`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result</para>
        ///     <para>in an output array of shape ``(i0, i1, d)`` with::</para>
        ///     <para> </para>
        ///     <para>  output[i,j,:] = off_value</para>
        ///     <para>  output[i,j,indices[i,j]] = on_value</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]</para>
        ///     <para>                           [ 1.  0.  0.]</para>
        ///     <para>                           [ 0.  0.  1.]</para>
        ///     <para>                           [ 1.  0.  0.]]</para>
        ///     <para> </para>
        ///     <para>  one_hot([1,0,2,0], 3, on_value=8, off_value=1,</para>
        ///     <para>          dtype='int32') = [[1 8 1]</para>
        ///     <para>                            [8 1 1]</para>
        ///     <para>                            [1 1 8]</para>
        ///     <para>                            [8 1 1]]</para>
        ///     <para> </para>
        ///     <para>  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]</para>
        ///     <para>                                      [ 1.  0.  0.]]</para>
        ///     <para> </para>
        ///     <para>                                     [[ 0.  1.  0.]</para>
        ///     <para>                                      [ 1.  0.  0.]]</para>
        ///     <para> </para>
        ///     <para>                                     [[ 0.  0.  1.]</para>
        ///     <para>                                      [ 1.  0.  0.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\indexing_op.cc:L799</para>
        /// </summary>
        /// <param name="indices">array of locations where to set on_value</param>
        /// <param name="depth">Depth of the one hot dimension.</param>
        /// <param name="on_value">The value assigned to the locations represented by indices.</param>
        /// <param name="off_value">The value assigned to the locations not represented by indices.</param>
        /// <param name="dtype">DType of the output</param>
        /// <returns>returns new symbol</returns>
        public static Symbol OneHot(Symbol indices, int depth, double on_value = 1, double off_value = 0,
            DType dtype = null, string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("one_hot")
                .SetParam("depth", depth)
                .SetParam("on_value", on_value)
                .SetParam("off_value", off_value)
                .SetParam("dtype", dtype)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Gather elements or slices from `data` and store to a tensor whose</para>
        ///     <para>shape is defined by `indices`.</para>
        ///     <para> </para>
        ///     <para>Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape</para>
        ///     <para>`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,</para>
        ///     <para>
        ///         where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.</para>
        ///     <para> </para>
        ///     <para>The elements in output is defined as follows::</para>
        ///     <para> </para>
        ///     <para>  output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],</para>
        ///     <para>                                                      ...,</para>
        ///     <para>                                                      indices[M-1, y_0, ..., y_{K-1}],</para>
        ///     <para>                                                      x_M, ..., x_{N-1}]</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  data = [[0, 1], [2, 3]]</para>
        ///     <para>  indices = [[1, 1, 0], [0, 1, 0]]</para>
        ///     <para>  gather_nd(data, indices) = [2, 3, 0]</para>
        ///     <para> </para>
        ///     <para>  data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]</para>
        ///     <para>  indices = [[0, 1], [1, 0]]</para>
        ///     <para>  gather_nd(data, indices) = [[3, 4], [5, 6]]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">data</param>
        /// <param name="indices">indices</param>
        /// <returns>returns new symbol</returns>
        public static Symbol GatherNd(Symbol data, Symbol indices, string symbol_name = "")
        {
            return new Operator("gather_nd")
                .SetInput("data", data)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Scatters data into a new tensor according to indices.</para>
        ///     <para> </para>
        ///     <para>Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape</para>
        ///     <para>`(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,</para>
        ///     <para>
        ///         where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.</para>
        ///     <para> </para>
        ///     <para>The elements in output is defined as follows::</para>
        ///     <para> </para>
        ///     <para>  output[indices[0, y_0, ..., y_{K-1}],</para>
        ///     <para>         ...,</para>
        ///     <para>         indices[M-1, y_0, ..., y_{K-1}],</para>
        ///     <para>         x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]</para>
        ///     <para> </para>
        ///     <para>all other entries in output are 0.</para>
        ///     <para> </para>
        ///     <para>.. warning::</para>
        ///     <para> </para>
        ///     <para>    If the indices have duplicates, the result will be non-deterministic and</para>
        ///     <para>    the gradient of `scatter_nd` will not be correct!!</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  data = [2, 3, 0]</para>
        ///     <para>  indices = [[1, 1, 0], [0, 1, 0]]</para>
        ///     <para>  shape = (2, 2)</para>
        ///     <para>  scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]</para>
        ///     <para> </para>
        ///     <para>  data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]</para>
        ///     <para>  indices = [[0, 1], [1, 1]]</para>
        ///     <para>  shape = (2, 2, 2, 2)</para>
        ///     <para>  scatter_nd(data, indices, shape) = [[[[0, 0],</para>
        ///     <para>                                        [0, 0]],</para>
        ///     <para> </para>
        ///     <para>                                       [[1, 2],</para>
        ///     <para>                                        [3, 4]]],</para>
        ///     <para> </para>
        ///     <para>                                      [[[0, 0],</para>
        ///     <para>                                        [0, 0]],</para>
        ///     <para> </para>
        ///     <para>                                       [[5, 6],</para>
        ///     <para>                                        [7, 8]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">data</param>
        /// <param name="indices">indices</param>
        /// <param name="shape">Shape of output.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ScatterNd(Symbol data, Symbol indices, Shape shape, string symbol_name = "")
        {
            return new Operator("scatter_nd")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>This operator has the same functionality as scatter_nd</para>
        ///     <para>except that it does not reset the elements not indexed by the input</para>
        ///     <para>index `NDArray` in the input data `NDArray`. output should be explicitly</para>
        ///     <para>given and be the same as lhs.</para>
        ///     <para> </para>
        ///     <para>.. note:: This operator is for internal use only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  data = [2, 3, 0]</para>
        ///     <para>  indices = [[1, 1, 0], [0, 1, 0]]</para>
        ///     <para>  out = [[1, 1], [1, 1]]</para>
        ///     <para>  _scatter_set_nd(lhs=out, rhs=data, indices=indices, out=out)</para>
        ///     <para>  out = [[0, 1], [2, 3]]</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">source input</param>
        /// <param name="rhs">value to assign</param>
        /// <param name="indices">indices</param>
        /// <param name="shape">Shape of output.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ScatterSetNd(Symbol lhs, Symbol rhs, Symbol indices, Shape shape, string symbol_name = "")
        {
            return new Operator("_scatter_set_nd")
                .SetParam("shape", shape)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>fill target with zeros without default dtype</para>
        /// </summary>
        /// <param name="shape">The shape of the output</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol ZerosWithoutDtype(Shape shape = null, Context ctx = null, DType dtype = null,
            string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_zeros_without_dtype")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>fill target with zeros</para>
        /// </summary>
        /// <param name="shape">The shape of the output</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Zeros(Shape shape = null, Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_zeros")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Return a 2-D array with ones on the diagonal and zeros elsewhere.</para>
        /// </summary>
        /// <param name="N">Number of rows in the output.</param>
        /// <param name="M">Number of columns in the output. If 0, defaults to N</param>
        /// <param name="k">
        ///     Index of the diagonal. 0 (the default) refers to the main diagonal.A positive value refers to an upper
        ///     diagonal.A negative value to a lower diagonal.
        /// </param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Eye(int N, int M = 0, int k = 0, Context ctx = null, DType dtype = null,
            string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_eye")
                .SetParam("N", N)
                .SetParam("M", M)
                .SetParam("k", k)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>fill target with ones</para>
        /// </summary>
        /// <param name="shape">The shape of the output</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Ones(Shape shape = null, Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_ones")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        public static Symbol Empty(Shape shape = null, Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_empty")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>fill target with a scalar value</para>
        /// </summary>
        /// <param name="shape">The shape of the output</param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <param name="value">Value with which to fill newly created tensor</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Full(double value, Shape shape = null, Context ctx = null, DType dtype = null,
            string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_full")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .SetParam("value", value)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Return evenly spaced values within a given interval. Similar to Numpy</para>
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">
        ///     End of interval. The interval does not include this value, except in some cases where step is not an
        ///     integer and floating point round-off affects the length of out.
        /// </param>
        /// <param name="step">Spacing between values.</param>
        /// <param name="repeat">
        ///     The repeating time of all elements. E.g repeat=3, the element a will be repeated three times -->
        ///     a, a, a.
        /// </param>
        /// <param name="infer_range">
        ///     When set to True, infer the stop position from the start, step, repeat, and output tensor
        ///     size.
        /// </param>
        /// <param name="ctx">Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.</param>
        /// <param name="dtype">Target data type.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Arange(double start, double? stop = null, double step = 1, int repeat = 1,
            bool infer_range = false, Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_arange")
                .SetParam("start", start)
                .SetParam("stop", stop)
                .SetParam("step", step)
                .SetParam("repeat", repeat)
                .SetParam("infer_range", infer_range)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol ZerosLike(Symbol data, string symbol_name = "")
        {
            return new Operator("zeros_like")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol OnesLike(Symbol data, string symbol_name = "")
        {
            return new Operator("ones_like")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs general matrix multiplication and accumulation.</para>
        ///     <para>Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape</para>
        ///     <para>on the leading *n-2* dimensions.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, the BLAS3 function *gemm* is performed:</para>
        ///     <para> </para>
        ///     <para>   *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*</para>
        ///     <para> </para>
        ///     <para>Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or</para>
        ///     <para>matrix transposition (depending on *transpose_a*, *transpose_b*).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices</para>
        ///     <para>are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis*</para>
        ///     <para>parameter. By default, the trailing two dimensions will be used for matrix encoding.</para>
        ///     <para> </para>
        ///     <para>For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes</para>
        ///     <para>calls. For example let *A*, *B*, *C* be 5 dimensional tensors. Then gemm(*A*, *B*, *C*, axis=1) is equivalent</para>
        ///     <para>to the following without the overhead of the additional swapaxis operations::</para>
        ///     <para> </para>
        ///     <para>    A1 = swapaxes(A, dim1=1, dim2=3)</para>
        ///     <para>    B1 = swapaxes(B, dim1=1, dim2=3)</para>
        ///     <para>    C = swapaxes(C, dim1=1, dim2=3)</para>
        ///     <para>    C = gemm(A1, B1, C)</para>
        ///     <para>    C = swapaxis(C, dim1=1, dim2=3)</para>
        ///     <para> </para>
        ///     <para>When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE</para>
        ///     <para>and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use</para>
        ///     <para>pseudo-float16 precision (float32 math with float16 I/O) precision in order to use</para>
        ///     <para>Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix multiply-add</para>
        ///     <para>   A = [[1.0, 1.0], [1.0, 1.0]]</para>
        ///     <para>   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]</para>
        ///     <para>   C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]</para>
        ///     <para>   gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)</para>
        ///     <para>           = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix multiply-add</para>
        ///     <para>   A = [[[1.0, 1.0]], [[0.1, 0.1]]]</para>
        ///     <para>   B = [[[1.0, 1.0]], [[0.1, 0.1]]]</para>
        ///     <para>   C = [[[10.0]], [[0.01]]]</para>
        ///     <para>   gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)</para>
        ///     <para>           = [[[104.0]], [[0.14]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L87</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices</param>
        /// <param name="B">Tensor of input matrices</param>
        /// <param name="C">Tensor of input matrices</param>
        /// <param name="transpose_a">Multiply with transposed of first input (A).</param>
        /// <param name="transpose_b">Multiply with transposed of second input (B).</param>
        /// <param name="alpha">Scalar factor multiplied with A*B.</param>
        /// <param name="beta">Scalar factor multiplied with C.</param>
        /// <param name="axis">Axis corresponding to the matrix rows.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgGemm(Symbol A, Symbol B, Symbol C, bool transpose_a = false,
            bool transpose_b = false, double alpha = 1, double beta = 1, int axis = -2, string symbol_name = "")
        {
            return new Operator("_linalg_gemm")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetParam("axis", axis)
                .SetInput("A", A)
                .SetInput("B", B)
                .SetInput("C", C)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs general matrix multiplication.</para>
        ///     <para>Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape</para>
        ///     <para>on the leading *n-2* dimensions.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, the BLAS3 function *gemm* is performed:</para>
        ///     <para> </para>
        ///     <para>   *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)</para>
        ///     <para> </para>
        ///     <para>Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix</para>
        ///     <para>transposition (depending on *transpose_a*, *transpose_b*).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices</para>
        ///     <para>are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis*</para>
        ///     <para>parameter. By default, the trailing two dimensions will be used for matrix encoding.</para>
        ///     <para> </para>
        ///     <para>For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes</para>
        ///     <para>calls. For example let *A*, *B* be 5 dimensional tensors. Then gemm(*A*, *B*, axis=1) is equivalent to</para>
        ///     <para>the following without the overhead of the additional swapaxis operations::</para>
        ///     <para> </para>
        ///     <para>    A1 = swapaxes(A, dim1=1, dim2=3)</para>
        ///     <para>    B1 = swapaxes(B, dim1=1, dim2=3)</para>
        ///     <para>    C = gemm2(A1, B1)</para>
        ///     <para>    C = swapaxis(C, dim1=1, dim2=3)</para>
        ///     <para> </para>
        ///     <para>When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE</para>
        ///     <para>and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use</para>
        ///     <para>pseudo-float16 precision (float32 math with float16 I/O) precision in order to use</para>
        ///     <para>Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix multiply</para>
        ///     <para>   A = [[1.0, 1.0], [1.0, 1.0]]</para>
        ///     <para>   B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]</para>
        ///     <para>   gemm2(A, B, transpose_b=True, alpha=2.0)</para>
        ///     <para>            = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix multiply</para>
        ///     <para>   A = [[[1.0, 1.0]], [[0.1, 0.1]]]</para>
        ///     <para>   B = [[[1.0, 1.0]], [[0.1, 0.1]]]</para>
        ///     <para>   gemm2(A, B, transpose_b=True, alpha=2.0)</para>
        ///     <para>           = [[[4.0]], [[0.04 ]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L161</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices</param>
        /// <param name="B">Tensor of input matrices</param>
        /// <param name="transpose_a">Multiply with transposed of first input (A).</param>
        /// <param name="transpose_b">Multiply with transposed of second input (B).</param>
        /// <param name="alpha">Scalar factor multiplied with A*B.</param>
        /// <param name="axis">Axis corresponding to the matrix row indices.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgGemm2(Symbol A, Symbol B, bool transpose_a = false, bool transpose_b = false,
            double alpha = 1, int axis = -2, string symbol_name = "")
        {
            return new Operator("_linalg_gemm2")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("alpha", alpha)
                .SetParam("axis", axis)
                .SetInput("A", A)
                .SetInput("B", B)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs Cholesky factorization of a symmetric positive-definite matrix.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, the Cholesky factor *B* of the symmetric, positive definite matrix *A* is</para>
        ///     <para>computed. *B* is triangular (entries of upper or lower triangle are all zero), has</para>
        ///     <para>positive diagonal entries, and:</para>
        ///     <para> </para>
        ///     <para>  *A* = *B* \* *B*\ :sup:`T`  if *lower* = *true*</para>
        ///     <para>  *A* = *B*\ :sup:`T` \* *B*  if *lower* = *false*</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs</para>
        ///     <para>(batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix factorization</para>
        ///     <para>   A = [[4.0, 1.0], [1.0, 4.25]]</para>
        ///     <para>   potrf(A) = [[2.0, 0], [0.5, 2.0]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix factorization</para>
        ///     <para>   A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]</para>
        ///     <para>   potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L212</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices to be decomposed</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgPotrf(Symbol A, string symbol_name = "")
        {
            return new Operator("_linalg_potrf")
                .SetInput("A", A)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs matrix inversion from a Cholesky factorization.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, *A* is a triangular matrix (entries of upper or lower triangle are all zero)</para>
        ///     <para>with positive diagonal. We compute:</para>
        ///     <para> </para>
        ///     <para>  *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1` if *lower* = *true*</para>
        ///     <para>  *out* = *A*\ :sup:`-1` \* *A*\ :sup:`-T` if *lower* = *false*</para>
        ///     <para> </para>
        ///     <para>In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix</para>
        ///     <para>*B* (obtained by *potrf*), then</para>
        ///     <para> </para>
        ///     <para>  *out* = *B*\ :sup:`-1`</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs</para>
        ///     <para>(batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>.. note:: Use this operator only if you are certain you need the inverse of *B*, and</para>
        ///     <para>          cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution</para>
        ///     <para>          (*trsm*). The latter is numerically much safer, and also cheaper.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix inverse</para>
        ///     <para>   A = [[2.0, 0], [0.5, 2.0]]</para>
        ///     <para>   potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix inverse</para>
        ///     <para>   A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]</para>
        ///     <para>   potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],</para>
        ///     <para>               [[0.06641, -0.01562], [-0.01562, 0,0625]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L273</para>
        /// </summary>
        /// <param name="A">Tensor of lower triangular matrices</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgPotri(Symbol A, string symbol_name = "")
        {
            return new Operator("_linalg_potri")
                .SetInput("A", A)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs multiplication with a lower triangular matrix.</para>
        ///     <para>Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape</para>
        ///     <para>on the leading *n-2* dimensions.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, *A* must be triangular. The operator performs the BLAS3 function</para>
        ///     <para>*trmm*:</para>
        ///     <para> </para>
        ///     <para>   *out* = *alpha* \* *op*\ (*A*) \* *B*</para>
        ///     <para> </para>
        ///     <para>if *rightside=False*, or</para>
        ///     <para> </para>
        ///     <para>   *out* = *alpha* \* *B* \* *op*\ (*A*)</para>
        ///     <para> </para>
        ///     <para>if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the</para>
        ///     <para>identity or the matrix transposition (depending on *transpose*).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs</para>
        ///     <para>(batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single triangular matrix multiply</para>
        ///     <para>   A = [[1.0, 0], [1.0, 1.0]]</para>
        ///     <para>   B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]</para>
        ///     <para>   trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]</para>
        ///     <para> </para>
        ///     <para>   // Batch triangular matrix multiply</para>
        ///     <para>   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]</para>
        ///     <para>   B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]</para>
        ///     <para>   trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],</para>
        ///     <para>                            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L331</para>
        /// </summary>
        /// <param name="A">Tensor of lower triangular matrices</param>
        /// <param name="B">Tensor of matrices</param>
        /// <param name="transpose">Use transposed of the triangular matrix</param>
        /// <param name="rightside">Multiply triangular matrix from the right to non-triangular one.</param>
        /// <param name="lower">True if the triangular matrix is lower triangular, false if it is upper triangular.</param>
        /// <param name="alpha">Scalar factor to be applied to the result.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgTrmm(Symbol A, Symbol B, bool transpose = false, bool rightside = false,
            bool lower = true, double alpha = 1, string symbol_name = "")
        {
            return new Operator("_linalg_trmm")
                .SetParam("transpose", transpose)
                .SetParam("rightside", rightside)
                .SetParam("lower", lower)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .SetInput("B", B)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Solves matrix equation involving a lower triangular matrix.</para>
        ///     <para>Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape</para>
        ///     <para>on the leading *n-2* dimensions.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, *A* must be triangular. The operator performs the BLAS3 function</para>
        ///     <para>*trsm*, solving for *out* in:</para>
        ///     <para> </para>
        ///     <para>   *op*\ (*A*) \* *out* = *alpha* \* *B*</para>
        ///     <para> </para>
        ///     <para>if *rightside=False*, or</para>
        ///     <para> </para>
        ///     <para>   *out* \* *op*\ (*A*) = *alpha* \* *B*</para>
        ///     <para> </para>
        ///     <para>if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the</para>
        ///     <para>identity or the matrix transposition (depending on *transpose*).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs</para>
        ///     <para>(batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix solve</para>
        ///     <para>   A = [[1.0, 0], [1.0, 1.0]]</para>
        ///     <para>   B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]</para>
        ///     <para>   trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix solve</para>
        ///     <para>   A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]</para>
        ///     <para>   B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],</para>
        ///     <para>        [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]</para>
        ///     <para>   trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],</para>
        ///     <para>                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L394</para>
        /// </summary>
        /// <param name="A">Tensor of lower triangular matrices</param>
        /// <param name="B">Tensor of matrices</param>
        /// <param name="transpose">Use transposed of the triangular matrix</param>
        /// <param name="rightside">Multiply triangular matrix from the right to non-triangular one.</param>
        /// <param name="lower">True if the triangular matrix is lower triangular, false if it is upper triangular.</param>
        /// <param name="alpha">Scalar factor to be applied to the result.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgTrsm(Symbol A, Symbol B, bool transpose = false, bool rightside = false,
            bool lower = true, double alpha = 1, string symbol_name = "")
        {
            return new Operator("_linalg_trsm")
                .SetParam("transpose", transpose)
                .SetParam("rightside", rightside)
                .SetParam("lower", lower)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .SetInput("B", B)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes the sum of the logarithms of the diagonal elements of a square matrix.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, *A* must be square with positive diagonal entries. We sum the natural</para>
        ///     <para>logarithms of the diagonal elements, the result has shape (1,).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all</para>
        ///     <para>inputs (batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix reduction</para>
        ///     <para>   A = [[1.0, 1.0], [1.0, 7.0]]</para>
        ///     <para>   sumlogdiag(A) = [1.9459]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix reduction</para>
        ///     <para>   A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]</para>
        ///     <para>   sumlogdiag(A) = [1.9459, 3.9318]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L443</para>
        /// </summary>
        /// <param name="A">Tensor of square matrices</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgSumlogdiag(Symbol A, string symbol_name = "")
        {
            return new Operator("_linalg_sumlogdiag")
                .SetInput("A", A)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Multiplication of matrix with its transpose.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, the operator performs the BLAS3 function *syrk*:</para>
        ///     <para> </para>
        ///     <para>  *out* = *alpha* \* *A* \* *A*\ :sup:`T`</para>
        ///     <para> </para>
        ///     <para>if *transpose=False*, or</para>
        ///     <para> </para>
        ///     <para>  *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*</para>
        ///     <para> </para>
        ///     <para>if *transpose=True*.</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *syrk* is performed separately on the trailing two dimensions for all</para>
        ///     <para>inputs (batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single matrix multiply</para>
        ///     <para>   A = [[1., 2., 3.], [4., 5., 6.]]</para>
        ///     <para>   syrk(A, alpha=1., transpose=False)</para>
        ///     <para>            = [[14., 32.],</para>
        ///     <para>               [32., 77.]]</para>
        ///     <para>   syrk(A, alpha=1., transpose=True)</para>
        ///     <para>            = [[17., 22., 27.],</para>
        ///     <para>               [22., 29., 36.],</para>
        ///     <para>               [27., 36., 45.]]</para>
        ///     <para> </para>
        ///     <para>   // Batch matrix multiply</para>
        ///     <para>   A = [[[1., 1.]], [[0.1, 0.1]]]</para>
        ///     <para>   syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L499</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices</param>
        /// <param name="transpose">Use transpose of input matrix.</param>
        /// <param name="alpha">Scalar factor to be applied to the result.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgSyrk(Symbol A, bool transpose = false, double alpha = 1, string symbol_name = "")
        {
            return new Operator("_linalg_syrk")
                .SetParam("transpose", transpose)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>LQ factorization for general matrix.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*</para>
        ///     <para>
        ///         must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ</para>
        ///     <para>factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so</para>
        ///     <para>that:</para>
        ///     <para> </para>
        ///     <para>   *A* = *L* \* *Q*</para>
        ///     <para> </para>
        ///     <para>Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,</para>
        ///     <para>and *Q* is row-orthonormal, meaning that</para>
        ///     <para> </para>
        ///     <para>   *Q* \* *Q*\ :sup:`T`</para>
        ///     <para> </para>
        ///     <para>is equal to the identity matrix of shape *(x, x)*.</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all</para>
        ///     <para>inputs (batch mode).</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single LQ factorization</para>
        ///     <para>   A = [[1., 2., 3.], [4., 5., 6.]]</para>
        ///     <para>   Q, L = gelqf(A)</para>
        ///     <para>   Q = [[-0.26726124, -0.53452248, -0.80178373],</para>
        ///     <para>        [0.87287156, 0.21821789, -0.43643578]]</para>
        ///     <para>   L = [[-3.74165739, 0.],</para>
        ///     <para>        [-8.55235974, 1.96396101]]</para>
        ///     <para> </para>
        ///     <para>   // Batch LQ factorization</para>
        ///     <para>   A = [[[1., 2., 3.], [4., 5., 6.]],</para>
        ///     <para>        [[7., 8., 9.], [10., 11., 12.]]]</para>
        ///     <para>   Q, L = gelqf(A)</para>
        ///     <para>   Q = [[[-0.26726124, -0.53452248, -0.80178373],</para>
        ///     <para>         [0.87287156, 0.21821789, -0.43643578]],</para>
        ///     <para>        [[-0.50257071, -0.57436653, -0.64616234],</para>
        ///     <para>         [0.7620735, 0.05862104, -0.64483142]]]</para>
        ///     <para>   L = [[[-3.74165739, 0.],</para>
        ///     <para>         [-8.55235974, 1.96396101]],</para>
        ///     <para>        [[-13.92838828, 0.],</para>
        ///     <para>         [-19.09768702, 0.52758934]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L567</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices to be factorized</param>
        /// <returns>returns new symbol</returns>
        public static Symbol LinalgGelqf(Symbol A, string symbol_name = "")
        {
            return new Operator("_linalg_gelqf")
                .SetInput("A", A)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Eigendecomposition for symmetric matrix.</para>
        ///     <para>Input is a tensor *A* of dimension *n >= 2*.</para>
        ///     <para> </para>
        ///     <para>If *n=2*, *A* must be symmetric, of shape *(x, x)*. We compute the eigendecomposition,</para>
        ///     <para>resulting in the orthonormal matrix *U* of eigenvectors, shape *(x, x)*, and the</para>
        ///     <para>vector *L* of eigenvalues, shape *(x,)*, so that:</para>
        ///     <para> </para>
        ///     <para>   *U* \* *A* = *diag(L)* \* *U*</para>
        ///     <para> </para>
        ///     <para>Here:</para>
        ///     <para> </para>
        ///     <para>   *U* \* *U*\ :sup:`T` = *U*\ :sup:`T` \* *U* = *I*</para>
        ///     <para> </para>
        ///     <para>
        ///         where *I* is the identity matrix. Also, *L(0) <= L(1) <= L(2) <= ...* (ascending order).</para>
        ///     <para> </para>
        ///     <para>If *n>2*, *syevd* is performed separately on the trailing two dimensions of *A* (batch</para>
        ///     <para>mode). In this case, *U* has *n* dimensions like *A*, and *L* has *n-1* dimensions.</para>
        ///     <para> </para>
        ///     <para>.. note:: The operator supports float32 and float64 data types only.</para>
        ///     <para> </para>
        ///     <para>.. note:: Derivatives for this operator are defined only if *A* is such that all its</para>
        ///     <para>          eigenvalues are distinct, and the eigengaps are not too small. If you need</para>
        ///     <para>          gradients, do not apply this operator to matrices with multiple eigenvalues.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>   // Single symmetric eigendecomposition</para>
        ///     <para>   A = [[1., 2.], [2., 4.]]</para>
        ///     <para>   U, L = syevd(A)</para>
        ///     <para>   U = [[0.89442719, -0.4472136],</para>
        ///     <para>        [0.4472136, 0.89442719]]</para>
        ///     <para>   L = [0., 5.]</para>
        ///     <para> </para>
        ///     <para>   // Batch symmetric eigendecomposition</para>
        ///     <para>   A = [[[1., 2.], [2., 4.]],</para>
        ///     <para>        [[1., 2.], [2., 5.]]]</para>
        ///     <para>   U, L = syevd(A)</para>
        ///     <para>   U = [[[0.89442719, -0.4472136],</para>
        ///     <para>         [0.4472136, 0.89442719]],</para>
        ///     <para>        [[0.92387953, -0.38268343],</para>
        ///     <para>         [0.38268343, 0.92387953]]]</para>
        ///     <para>   L = [[0., 5.],</para>
        ///     <para>        [0.17157288, 5.82842712]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\la_op.cc:L636</para>
        /// </summary>
        /// <param name="A">Tensor of input matrices to be factorized</param>
        /// <returns>returns new symbol</returns>
        public static (Symbol, Symbol) LinalgSyevd(Symbol A, string symbol_name = "")
        {
            var result = new Operator("_linalg_syevd")
                .SetInput("A", A)
                .CreateSymbol(symbol_name);

            if (result.ListOutputs().Count > 1)
                return (result[0], result[1]);

            return (result, null);
        }

        /// <summary>
        ///     <para>Reshapes the input array.</para>
        ///     <para> </para>
        ///     <para>.. note:: ``Reshape`` is deprecated, use ``reshape``</para>
        ///     <para> </para>
        ///     <para>Given an array and a shape, this function returns a copy of the array in the new shape.</para>
        ///     <para>
        ///         The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the
        ///         input array.
        ///     </para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]</para>
        ///     <para> </para>
        ///     <para>
        ///         Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of
        ///         each is explained below:
        ///     </para>
        ///     <para> </para>
        ///     <para>- ``0``  copy this dimension from the input to the output shape.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)</para>
        ///     <para>  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)</para>
        ///     <para> </para>
        ///     <para>- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions</para>
        ///     <para>  keeping the size of the new array same as that of the input array.</para>
        ///     <para>  At most one dimension of shape can be -1.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)</para>
        ///     <para>  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)</para>
        ///     <para>  - input shape = (2,3,4), shape=(-1,), output shape = (24,)</para>
        ///     <para> </para>
        ///     <para>- ``-2`` copy all/remainder of the input dimensions to the output shape.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)</para>
        ///     <para>  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)</para>
        ///     <para>  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)</para>
        ///     <para> </para>
        ///     <para>- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)</para>
        ///     <para>  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)</para>
        ///     <para>  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)</para>
        ///     <para>  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)</para>
        ///     <para> </para>
        ///     <para>
        ///         - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain
        ///         -1).
        ///     </para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)</para>
        ///     <para>  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)</para>
        ///     <para> </para>
        ///     <para>If the argument `reverse` is set to 1, then the special values are inferred from right to left.</para>
        ///     <para> </para>
        ///     <para>  Example::</para>
        ///     <para> </para>
        ///     <para>  - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)</para>
        ///     <para>  - with reverse=1, output shape will be (50,4).</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L222</para>
        /// </summary>
        /// <param name="data">Input data to reshape.</param>
        /// <param name="shape">The target shape</param>
        /// <param name="reverse">If true then the special values are inferred from right to left</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Reshape(Symbol data, Shape shape = null, bool reverse = false, string symbol_name = "")
        {
            if (shape == null) shape = new Shape();

            return new Operator("Reshape")
                .SetParam("shape", shape)
                .SetParam("reverse", reverse)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Transpose(Symbol data, Shape axes = null, string symbol_name = "")
        {
            if (axes == null) axes = new Shape();

            return new Operator("transpose")
                .SetParam("axes", axes)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol ExpandDims(Symbol data, int axis, string symbol_name = "")
        {
            return new Operator("expand_dims")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Slice(Symbol data, Shape begin, Shape end, Shape step = null, string symbol_name = "")
        {
            if (step == null) step = new Shape();

            return new Operator("slice")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        public static SymbolList SliceChannel(Symbol data, int num_outputs, int axis = 1, bool squeeze_axis = false, string symbol_name = "")
        {
            return new Operator("SliceChannel")
                .SetParam("num_outputs", num_outputs)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetInput("data", data)
                .CreateSymbol(symbol_name).ToList();
        }

        /// <summary>
        ///     <para>Assign the rhs to a cropped subset of lhs.</para>
        ///     <para> </para>
        ///     <para>Requirements</para>
        ///     <para>------------</para>
        ///     <para>- output should be explicitly given and be the same as lhs.</para>
        ///     <para>- lhs and rhs are of the same data type, and on the same device.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:559</para>
        /// </summary>
        /// <param name="lhs">Source input</param>
        /// <param name="rhs">value to assign</param>
        /// <param name="begin">starting indices for the slice operation, supports negative indices.</param>
        /// <param name="end">ending indices for the slice operation, supports negative indices.</param>
        /// <param name="step">step for the slice operation, supports negative values.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceAssign(Symbol lhs, Symbol rhs, Shape begin, Shape end, Shape step = null,
            string symbol_name = "")
        {
            if (step == null) step = new Shape();

            return new Operator("_slice_assign")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>(Assign the scalar to a cropped subset of the input.</para>
        ///     <para> </para>
        ///     <para>Requirements</para>
        ///     <para>------------</para>
        ///     <para>- output should be explicitly given and be the same as input</para>
        ///     <para>)</para>
        ///     <para> </para>
        ///     <para>From:C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:584</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="scalar">The scalar value for assignment.</param>
        /// <param name="begin">starting indices for the slice operation, supports negative indices.</param>
        /// <param name="end">ending indices for the slice operation, supports negative indices.</param>
        /// <param name="step">step for the slice operation, supports negative values.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceAssignScalar(Symbol data, Shape begin, Shape end, double scalar = 0,
            Shape step = null, string symbol_name = "")
        {
            if (step == null) step = new Shape();

            return new Operator("_slice_assign_scalar")
                .SetParam("scalar", scalar)
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Slices along a given axis.</para>
        ///     <para> </para>
        ///     <para>Returns an array slice along a given `axis` starting from the `begin` index</para>
        ///     <para>to the `end` index.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [[  1.,   2.,   3.,   4.],</para>
        ///     <para>       [  5.,   6.,   7.,   8.],</para>
        ///     <para>       [  9.,  10.,  11.,  12.]]</para>
        ///     <para> </para>
        ///     <para>  slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],</para>
        ///     <para>                                           [  9.,  10.,  11.,  12.]]</para>
        ///     <para> </para>
        ///     <para>  slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],</para>
        ///     <para>                                           [  5.,   6.],</para>
        ///     <para>                                           [  9.,  10.]]</para>
        ///     <para> </para>
        ///     <para>  slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],</para>
        ///     <para>                                             [  6.,   7.],</para>
        ///     <para>                                             [ 10.,  11.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L620</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="axis">Axis along which to be sliced, supports negative indexes.</param>
        /// <param name="begin">The beginning index along the axis to be sliced,  supports negative indexes.</param>
        /// <param name="end">The ending index along the axis to be sliced,  supports negative indexes.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceAxis(Symbol data, int axis, int begin, int? end, string symbol_name = "")
        {
            return new Operator("slice_axis")
                .SetParam("axis", axis)
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Slices a region of the array like the shape of another array.</para>
        ///     <para> </para>
        ///     <para>This function is similar to ``slice``, however, the `begin` are always `0`s</para>
        ///     <para>and `end` of specific axes are inferred from the second input `shape_like`.</para>
        ///     <para> </para>
        ///     <para>Given the second `shape_like` input of ``shape=(d_0, d_1, ..., d_n-1)``,</para>
        ///     <para>a ``slice_like`` operator with default empty `axes`, it performs the</para>
        ///     <para>following operation:</para>
        ///     <para> </para>
        ///     <para>`` out = slice(input, begin=(0, 0, ..., 0), end=(d_0, d_1, ..., d_n-1))``.</para>
        ///     <para> </para>
        ///     <para>When `axes` is not empty, it is used to speficy which axes are being sliced.</para>
        ///     <para> </para>
        ///     <para>Given a 4-d input data, ``slice_like`` operator with ``axes=(0, 2, -1)``</para>
        ///     <para>will perform the following operation:</para>
        ///     <para> </para>
        ///     <para>`` out = slice(input, begin=(0, 0, 0, 0), end=(d_0, None, d_2, d_3))``.</para>
        ///     <para> </para>
        ///     <para>Note that it is allowed to have first and second input with different dimensions,</para>
        ///     <para>however, you have to make sure the `axes` are specified and not exceeding the</para>
        ///     <para>dimension limits.</para>
        ///     <para> </para>
        ///     <para>For example, given `input_1` with ``shape=(2,3,4,5)`` and `input_2` with</para>
        ///     <para>``shape=(1,2,3)``, it is not allowed to use:</para>
        ///     <para> </para>
        ///     <para>`` out = slice_like(a, b)`` because ndim of `input_1` is 4, and ndim of `input_2`</para>
        ///     <para>is 3.</para>
        ///     <para> </para>
        ///     <para>The following is allowed in this situation:</para>
        ///     <para> </para>
        ///     <para>`` out = slice_like(a, b, axes=(0, 2))``</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[  1.,   2.,   3.,   4.],</para>
        ///     <para>       [  5.,   6.,   7.,   8.],</para>
        ///     <para>       [  9.,  10.,  11.,  12.]]</para>
        ///     <para> </para>
        ///     <para>  y = [[  0.,   0.,   0.],</para>
        ///     <para>       [  0.,   0.,   0.]]</para>
        ///     <para> </para>
        ///     <para>  slice_like(x, y) = [[ 1.,  2.,  3.]</para>
        ///     <para>                      [ 5.,  6.,  7.]]</para>
        ///     <para>  slice_like(x, y, axes=(0, 1)) = [[ 1.,  2.,  3.]</para>
        ///     <para>                                   [ 5.,  6.,  7.]]</para>
        ///     <para>  slice_like(x, y, axes=(0)) = [[ 1.,  2.,  3.,  4.]</para>
        ///     <para>                                [ 5.,  6.,  7.,  8.]]</para>
        ///     <para>  slice_like(x, y, axes=(-1)) = [[  1.,   2.,   3.]</para>
        ///     <para>                                 [  5.,   6.,   7.]</para>
        ///     <para>                                 [  9.,  10.,  11.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L689</para>
        /// </summary>
        /// <param name="data">Source input</param>
        /// <param name="shape_like">Shape like input</param>
        /// <param name="axes">
        ///     List of axes on which input data will be sliced according to the corresponding size of the second
        ///     input. By default will slice on all axes. Negative axes are supported.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol SliceLike(Symbol data, Symbol shape_like, Shape axes = null, string symbol_name = "")
        {
            if (axes == null) axes = new Shape();

            return new Operator("slice_like")
                .SetParam("axes", axes)
                .SetInput("data", data)
                .SetInput("shape_like", shape_like)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Clip(Symbol data, float a_min, float a_max, string symbol_name = "")
        {
            return new Operator("clip")
                .SetParam("a_min", a_min)
                .SetParam("a_max", a_max)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Repeat(Symbol data, int repeats, int? axis = null, string symbol_name = "")
        {
            return new Operator("repeat")
                .SetParam("repeats", repeats)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Tile(Symbol data, Shape reps, string symbol_name = "")
        {
            return new Operator("tile")
                .SetParam("reps", reps)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        public static Symbol Flip(Symbol data, int axis, string symbol_name = "")
        {
            return new Operator("reverse")
                .SetParam("axis", new Shape(axis))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Reverse(Symbol data, Shape axis, string symbol_name = "")
        {
            return new Operator("reverse")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Join a sequence of arrays along a new axis.</para>
        ///     <para> </para>
        ///     <para>The axis parameter specifies the index of the new axis in the dimensions of the</para>
        ///     <para>result. For example, if axis=0 it will be the first dimension and if axis=-1 it</para>
        ///     <para>will be the last dimension.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  x = [1, 2]</para>
        ///     <para>  y = [3, 4]</para>
        ///     <para> </para>
        ///     <para>  stack(x, y) = [[1, 2],</para>
        ///     <para>                 [3, 4]]</para>
        ///     <para>  stack(x, y, axis=1) = [[1, 3],</para>
        ///     <para>                         [2, 4]]</para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">List of arrays to stack</param>
        /// <param name="axis">The axis in the result array along which the input arrays are stacked.</param>
        /// <param name="num_args">Number of inputs to be stacked.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Stack(SymbolList data, int num_args, int axis = 0, string symbol_name = "")
        {
            return new Operator("stack")
                .SetParam("axis", axis)
                .SetParam("num_args", num_args)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Remove single-dimensional entries from the shape of an array.</para>
        ///     <para>Same behavior of defining the output tensor shape as numpy.squeeze for the most of cases.</para>
        ///     <para>See the following note for exception.</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  data = [[[0], [1], [2]]]</para>
        ///     <para>  squeeze(data) = [0, 1, 2]</para>
        ///     <para>  squeeze(data, axis=0) = [[0], [1], [2]]</para>
        ///     <para>  squeeze(data, axis=2) = [[0, 1, 2]]</para>
        ///     <para>  squeeze(data, axis=(0, 2)) = [0, 1, 2]</para>
        ///     <para> </para>
        ///     <para>.. Note::</para>
        ///     <para>  The output of this operator will keep at least one dimension not removed. For example,</para>
        ///     <para>  squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.</para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">data to squeeze</param>
        /// <param name="axis">
        ///     Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape
        ///     entry greater than one, an error is raised.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Squeeze(SymbolList data, Shape axis = null, string symbol_name = "")
        {
            return new Operator("squeeze")
                .SetParam("axis", axis)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Rearranges(permutes) data from depth into blocks of spatial data.</para>
        ///     <para>Similar to ONNX DepthToSpace operator:</para>
        ///     <para>https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace.</para>
        ///     <para>The output is a new tensor where the values from depth dimension are moved in spatial blocks </para>
        ///     <para>to height and width dimension. The reverse of this operation is ``space_to_depth``.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>    \begin{gather*}</para>
        ///     <para>
        ///         x \prime = reshape(x, [N, block\_size, block\_size, C / (block\_size ^ 2), H * block\_size, W *
        ///         block\_size]) \\
        ///     </para>
        ///     <para>    x \prime \prime = transpose(x \prime, [0, 3, 4, 1, 5, 2]) \\</para>
        ///     <para>    y = reshape(x \prime \prime, [N, C / (block\_size ^ 2), H * block\_size, W * block\_size])</para>
        ///     <para>    \end{gather*}</para>
        ///     <para> </para>
        ///     <para>
        ///         where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`: [batch, channels, height,
        ///         width]
        ///     </para>
        ///     <para>
        ///         and :math:`y` is the output tensor of layout :math:`[N, C / (block\_size ^ 2), H * block\_size, W *
        ///         block\_size]`
        ///     </para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[[[0, 1, 2],</para>
        ///     <para>         [3, 4, 5]],</para>
        ///     <para>        [[6, 7, 8],</para>
        ///     <para>         [9, 10, 11]],</para>
        ///     <para>        [[12, 13, 14],</para>
        ///     <para>         [15, 16, 17]],</para>
        ///     <para>        [[18, 19, 20],</para>
        ///     <para>         [21, 22, 23]]]]</para>
        ///     <para> </para>
        ///     <para>  depth_to_space(x, 2) = [[[[0, 6, 1, 7, 2, 8],</para>
        ///     <para>                            [12, 18, 13, 19, 14, 20],</para>
        ///     <para>                            [3, 9, 4, 10, 5, 11],</para>
        ///     <para>                            [15, 21, 16, 22, 17, 23]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L1074</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="block_size">Blocks of [block_size. block_size] are moved</param>
        /// <returns>returns new symbol</returns>
        public static Symbol DepthToSpace(Symbol data, int block_size, string symbol_name = "")
        {
            return new Operator("depth_to_space")
                .SetParam("block_size", block_size)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Rearranges(permutes) blocks of spatial data into depth.</para>
        ///     <para>Similar to ONNX SpaceToDepth operator:</para>
        ///     <para>https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth </para>
        ///     <para> </para>
        ///     <para>The output is a new tensor where the values from height and width dimension are </para>
        ///     <para>moved to the depth dimension. The reverse of this operation is ``depth_to_space``.</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>    \begin{gather*}</para>
        ///     <para>    x \prime = reshape(x, [N, C, H / block\_size, block\_size, W / block\_size, block\_size]) \\</para>
        ///     <para>    x \prime \prime = transpose(x \prime, [0, 3, 5, 1, 2, 4]) \\</para>
        ///     <para>    y = reshape(x \prime \prime, [N, C * (block\_size ^ 2), H / block\_size, W / block\_size])</para>
        ///     <para>    \end{gather*}</para>
        ///     <para> </para>
        ///     <para>
        ///         where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`: [batch, channels, height,
        ///         width]
        ///     </para>
        ///     <para>
        ///         and :math:`y` is the output tensor of layout :math:`[N, C * (block\_size ^ 2), H / block\_size, W /
        ///         block\_size]`
        ///     </para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[[[0, 6, 1, 7, 2, 8],</para>
        ///     <para>         [12, 18, 13, 19, 14, 20],</para>
        ///     <para>         [3, 9, 4, 10, 5, 11],</para>
        ///     <para>         [15, 21, 16, 22, 17, 23]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>  space_to_depth(x, 2) = [[[[0, 1, 2],</para>
        ///     <para>                            [3, 4, 5]],</para>
        ///     <para>                           [[6, 7, 8],</para>
        ///     <para>                            [9, 10, 11]],</para>
        ///     <para>                           [[12, 13, 14],</para>
        ///     <para>                            [15, 16, 17]],</para>
        ///     <para>                           [[18, 19, 20],</para>
        ///     <para>                            [21, 22, 23]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\tensor\matrix_op.cc:L1128</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="block_size">Blocks of [block_size. block_size] are moved</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SpaceToDepth(Symbol data, int block_size, string symbol_name = "")
        {
            return new Operator("space_to_depth")
                .SetParam("block_size", block_size)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Splits an array along a particular axis into multiple sub-arrays.</para>
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
        /// <returns>returns new symbol</returns>
        public static Symbol SplitV2(Symbol data, Shape indices, int axis = 1, bool squeeze_axis = false,
            int sections = 0, string symbol_name = "")
        {
            return new Operator("_split_v2")
                .SetParam("indices", indices)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetParam("sections", sections)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Topk(Symbol data, int? axis = -1, int k = 1, TopkRetTyp ret_typ = TopkRetTyp.Indices,
            bool is_ascend = false, DType dtype = null, string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("topk")
                .SetParam("axis", axis)
                .SetParam("k", k)
                .SetParam("ret_typ", MxUtil.EnumToString<TopkRetTyp>(ret_typ, TopkRetTypConvert))
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Sort(Symbol data, int? axis = -1, bool is_ascend = true, string symbol_name = "")
        {
            return new Operator("sort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol Argsort(Symbol data, int? axis = -1, bool is_ascend = true, DType dtype = null,
            string symbol_name = "")
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("argsort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol RavelMultiIndex(Symbol data, Shape shape = null, string symbol_name = "")
        {
            return new Operator("_ravel_multi_index")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol UnravelIndex(Symbol data, Shape shape = null, string symbol_name = "")
        {
            return new Operator("_unravel_index")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol SparseRetain(Symbol data, Symbol indices, string symbol_name = "")
        {
            return new Operator("_sparse_retain")
                .SetInput("data", data)
                .SetInput("indices", indices)
                .CreateSymbol(symbol_name);
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
        /// <returns>returns new symbol</returns>
        public static Symbol SquareSum(Symbol data, Shape axis = null, bool keepdims = false, bool exclude = false,
            string symbol_name = "")
        {
            return new Operator("_square_sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies bilinear sampling to input feature map.</para>
        ///     <para> </para>
        ///     <para>
        ///         Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is
        ///         very similar to remap function in OpenCV,
        ///     </para>
        ///     <para>except that the operator has the backward pass.</para>
        ///     <para> </para>
        ///     <para>Given :math:`data` and :math:`grid`, then the output is computed by</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\</para>
        ///     <para>  y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\</para>
        ///     <para>  output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})</para>
        ///     <para> </para>
        ///     <para>
        ///         :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes
        ///         the bilinear interpolation kernel.
        ///     </para>
        ///     <para>
        ///         The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0],
        ///         data.shape[1], grid.shape[2], grid.shape[3]).
        ///     </para>
        ///     <para> </para>
        ///     <para>The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].</para>
        ///     <para> </para>
        ///     <para>BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.</para>
        ///     <para>GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.</para>
        ///     <para>
        ///         If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of
        ///         GridGenerator.
        ///     </para>
        ///     <para> </para>
        ///     <para>Example 1::</para>
        ///     <para> </para>
        ///     <para>  ## Zoom out data two times</para>
        ///     <para>  data = array([[[[1, 4, 3, 6],</para>
        ///     <para>                  [1, 8, 8, 9],</para>
        ///     <para>                  [0, 4, 1, 5],</para>
        ///     <para>                  [1, 0, 1, 3]]]])</para>
        ///     <para> </para>
        ///     <para>  affine_matrix = array([[2, 0, 0],</para>
        ///     <para>                         [0, 2, 0]])</para>
        ///     <para> </para>
        ///     <para>  affine_matrix = reshape(affine_matrix, shape=(1, 6))</para>
        ///     <para> </para>
        ///     <para>  grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))</para>
        ///     <para> </para>
        ///     <para>  out = BilinearSampler(data, grid)</para>
        ///     <para> </para>
        ///     <para>  out</para>
        ///     <para>  [[[[ 0,   0,     0,   0],</para>
        ///     <para>     [ 0,   3.5,   6.5, 0],</para>
        ///     <para>     [ 0,   1.25,  2.5, 0],</para>
        ///     <para>     [ 0,   0,     0,   0]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Example 2::</para>
        ///     <para> </para>
        ///     <para>  ## shift data horizontally by -1 pixel</para>
        ///     <para> </para>
        ///     <para>  data = array([[[[1, 4, 3, 6],</para>
        ///     <para>                  [1, 8, 8, 9],</para>
        ///     <para>                  [0, 4, 1, 5],</para>
        ///     <para>                  [1, 0, 1, 3]]]])</para>
        ///     <para> </para>
        ///     <para>  warp_maxtrix = array([[[[1, 1, 1, 1],</para>
        ///     <para>                          [1, 1, 1, 1],</para>
        ///     <para>                          [1, 1, 1, 1],</para>
        ///     <para>                          [1, 1, 1, 1]],</para>
        ///     <para>                         [[0, 0, 0, 0],</para>
        ///     <para>                          [0, 0, 0, 0],</para>
        ///     <para>                          [0, 0, 0, 0],</para>
        ///     <para>                          [0, 0, 0, 0]]]])</para>
        ///     <para> </para>
        ///     <para>  grid = GridGenerator(data=warp_matrix, transform_type='warp')</para>
        ///     <para>  out = BilinearSampler(data, grid)</para>
        ///     <para> </para>
        ///     <para>  out</para>
        ///     <para>  [[[[ 4,  3,  6,  0],</para>
        ///     <para>     [ 8,  8,  9,  0],</para>
        ///     <para>     [ 4,  1,  5,  0],</para>
        ///     <para>     [ 0,  1,  3,  0]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\bilinear_sampler.cc:L256</para>
        /// </summary>
        /// <param name="data">Input data to the BilinearsamplerOp.</param>
        /// <param name="grid">Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src</param>
        /// <param name="cudnn_off">whether to turn cudnn off</param>
        /// <returns>returns new symbol</returns>
        public static Symbol BilinearSampler(Symbol data, Symbol grid, bool? cudnn_off = null, string symbol_name = "")
        {
            return new Operator("BilinearSampler")
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .SetInput("grid", grid)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>This operator is DEPRECATED. Apply convolution to input then add a bias.</para>
        /// </summary>
        /// <param name="data">Input data to the ConvolutionV1Op.</param>
        /// <param name="weight">Weight matrix.</param>
        /// <param name="bias">Bias parameter.</param>
        /// <param name="kernel">convolution kernel size: (h, w) or (d, h, w)</param>
        /// <param name="stride">convolution stride: (h, w) or (d, h, w)</param>
        /// <param name="dilate">convolution dilate: (h, w) or (d, h, w)</param>
        /// <param name="pad">pad for convolution: (h, w) or (d, h, w)</param>
        /// <param name="num_filter">convolution filter(channel) number</param>
        /// <param name="num_group">
        ///     Number of group partitions. Equivalent to slicing input into num_group    partitions, apply
        ///     convolution on each, then concatenate the results
        /// </param>
        /// <param name="workspace">
        ///     Maximum temporary workspace allowed for convolution (MB).This parameter determines the
        ///     effective batch size of the convolution kernel, which may be smaller than the given batch size. Also, the workspace
        ///     will be automatically enlarged to make sure that we can run the kernel with batch_size=1
        /// </param>
        /// <param name="no_bias">Whether to disable bias parameter.</param>
        /// <param name="cudnn_tune">
        ///     Whether to pick convolution algo by running performance test.    Leads to higher startup time
        ///     but may give faster speed. Options are:    'off': no tuning    'limited_workspace': run test and pick the fastest
        ///     algorithm that doesn't exceed workspace limit.    'fastest': pick the fastest algorithm and ignore workspace limit.
        ///     If set to None (default), behavior is determined by environment    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for
        ///     off,    1 for limited workspace (default), 2 for fastest.
        /// </param>
        /// <param name="cudnn_off">Turn off cudnn for this layer.</param>
        /// <param name="layout">
        ///     Set layout for input, output and weight. Empty for    default layout: NCHW for 2d and NCDHW for
        ///     3d.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol ConvolutionV1(Symbol data, Symbol weight, Symbol bias, Shape kernel, uint num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, uint num_group = 1, ulong workspace = 1024,
            bool no_bias = false, ConvolutionV1CudnnTune? cudnn_tune = null, bool cudnn_off = false,
            ConvolutionV1Layout? layout = null, string symbol_name = "")
        {
            if (stride == null) stride = new Shape();
            if (dilate == null) dilate = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("Convolution_v1")
                .SetParam("kernel", kernel)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("pad", pad)
                .SetParam("num_filter", num_filter)
                .SetParam("num_group", num_group)
                .SetParam("workspace", workspace)
                .SetParam("no_bias", no_bias)
                .SetParam("cudnn_tune", MxUtil.EnumToString(cudnn_tune, ConvolutionV1CudnnTuneConvert))
                .SetParam("cudnn_off", cudnn_off)
                .SetParam("layout", MxUtil.EnumToString(layout, ConvolutionV1LayoutConvert))
                .SetInput("data", data)
                .SetInput("weight", weight)
                .SetInput("bias", bias)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies correlation to inputs.</para>
        ///     <para> </para>
        ///     <para>The correlation layer performs multiplicative patch comparisons between two feature maps.</para>
        ///     <para> </para>
        ///     <para>
        ///         Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being
        ///         their width, height, and number of channels,
        ///     </para>
        ///     <para>
        ///         the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from
        ///         :math:`f_{2}`.
        ///     </para>
        ///     <para> </para>
        ///     <para>
        ///         For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at
        ///         :math:`x_{1}` in the first map and
        ///     </para>
        ///     <para>:math:`x_{2}` in the second map is then defined as:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>   c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}( x_{1} + o), f_{2}( x_{2} + o)></para>
        ///     <para> </para>
        ///     <para>for a square patch of size :math:`K:=2k+1`.</para>
        ///     <para> </para>
        ///     <para>
        ///         Note that the equation above is identical to one step of a convolution in neural networks, but instead of
        ///         convolving data with a filter, it convolves data with other
        ///     </para>
        ///     <para>data. For this reason, it has no training weights.</para>
        ///     <para> </para>
        ///     <para>
        ///         Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch
        ///         combinations involves :math:`w^{2}*h^{2}` such computations.
        ///     </para>
        ///     <para> </para>
        ///     <para>
        ///         Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations
        ///         :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
        ///     </para>
        ///     <para>
        ///         by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}`
        ///         globally and to quantize :math:`x_{2}` within the neighborhood
        ///     </para>
        ///     <para>centered around :math:`x_{1}`.</para>
        ///     <para> </para>
        ///     <para>The final output is defined by the following expression:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para>  out[n, q, i, j] = c(x_{i, j}, x_{q})</para>
        ///     <para> </para>
        ///     <para>
        ///         where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the
        ///         :math:`q^{th}` neighborhood of :math:`x_{i,j}`.
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\correlation.cc:L198</para>
        /// </summary>
        /// <param name="data1">Input data1 to the correlation.</param>
        /// <param name="data2">Input data2 to the correlation.</param>
        /// <param name="kernel_size">kernel size for Correlation must be an odd number</param>
        /// <param name="max_displacement">Max displacement of Correlation </param>
        /// <param name="stride1">stride1 quantize data1 globally</param>
        /// <param name="stride2">stride2 quantize data2 within the neighborhood centered around data1</param>
        /// <param name="pad_size">pad for Correlation</param>
        /// <param name="is_multiply">operation type is either multiplication or subduction</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Correlation(Symbol data1, Symbol data2, uint kernel_size = 1, uint max_displacement = 1,
            uint stride1 = 1, uint stride2 = 1, uint pad_size = 0, bool is_multiply = true, string symbol_name = "")
        {
            return new Operator("Correlation")
                .SetParam("kernel_size", kernel_size)
                .SetParam("max_displacement", max_displacement)
                .SetParam("stride1", stride1)
                .SetParam("stride2", stride2)
                .SetParam("pad_size", pad_size)
                .SetParam("is_multiply", is_multiply)
                .SetInput("data1", data1)
                .SetInput("data2", data2)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>.. note:: `Crop` is deprecated. Use `slice` instead.</para>
        ///     <para> </para>
        ///     <para>Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or</para>
        ///     <para>with width and height of the second input symbol, i.e., with one input, we need h_w to</para>
        ///     <para>specify the crop height and width, otherwise the second input symbol's size will be used</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\crop.cc:L50</para>
        /// </summary>
        /// <param name="data">Tensor or List of Tensors, the second input will be used as crop_like shape reference</param>
        /// <param name="num_args">
        ///     Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width,
        ///     else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here
        /// </param>
        /// <param name="offset">crop offset coordinate: (y, x)</param>
        /// <param name="h_w">crop height and width: (h, w)</param>
        /// <param name="center_crop">
        ///     If set to true, then it will use be the center_crop,or it will crop using the shape of
        ///     crop_like
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol Crop(SymbolList data, int num_args, Shape offset = null, Shape h_w = null,
            bool center_crop = false, string symbol_name = "")
        {
            if (offset == null) offset = new Shape();
            if (h_w == null) h_w = new Shape();

            return new Operator("Crop")
                .SetParam("data", data)
                .SetParam("num_args", num_args)
                .SetParam("offset", offset)
                .SetParam("h_w", h_w)
                .SetParam("center_crop", center_crop)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Special op to copy data cross device</para>
        /// </summary>
        /// <returns>returns new symbol</returns>
        public static Symbol CrossDeviceCopy(string symbol_name = "")
        {
            return new Operator("_CrossDeviceCopy")
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Stub for implementing an operator implemented in native frontend language.</para>
        /// </summary>
        /// <param name="data">Input data for the custom operator.</param>
        /// <param name="info"></param>
        /// <param name="need_top_grad">Whether this layer needs out grad for backward. Should be false for loss layers.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Native(SymbolList data, IntPtr info, bool need_top_grad = true, string symbol_name = "")
        {
            return new Operator("_Native")
                .SetParam("info", info)
                .SetParam("need_top_grad", need_top_grad)
                .SetInput(data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Generates 2D sampling grid for bilinear sampling.</para>
        /// </summary>
        /// <param name="data">Input data to the function.</param>
        /// <param name="transform_type">
        ///     The type of transformation. For `affine`, input data should be an affine matrix of size
        ///     (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
        /// </param>
        /// <param name="target_shape">
        ///     Specifies the output shape (H, W). This is required if transformation type is `affine`. If
        ///     transformation type is `warp`, this parameter is ignored.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol GridGenerator(Symbol data, GridgeneratorTransformType transform_type,
            Shape target_shape = null, string symbol_name = "")
        {
            if (target_shape == null) target_shape = new Shape();

            return new Operator("GridGenerator")
                .SetParam("transform_type",
                    MxUtil.EnumToString<GridgeneratorTransformType>(transform_type, GridgeneratorTransformTypeConvert))
                .SetParam("target_shape", target_shape)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies instance normalization to the n-dimensional input array.</para>
        ///     <para> </para>
        ///     <para>This operator takes an n-dimensional input array where (n>2) and normalizes</para>
        ///     <para>the input using the following formula:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>  out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta</para>
        ///     <para> </para>
        ///     <para>This layer is similar to batch normalization layer (`BatchNorm`)</para>
        ///     <para>with two differences: first, the normalization is</para>
        ///     <para>carried out per example (instance), not over a batch. Second, the</para>
        ///     <para>same normalization is applied both at test and train time. This</para>
        ///     <para>operation is also known as `contrast normalization`.</para>
        ///     <para> </para>
        ///     <para>If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],</para>
        ///     <para>`gamma` and `beta` parameters must be vectors of shape [channel].</para>
        ///     <para> </para>
        ///     <para>This implementation is based on paper:</para>
        ///     <para> </para>
        ///     <para>.. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,</para>
        ///     <para>   D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).</para>
        ///     <para> </para>
        ///     <para>Examples::</para>
        ///     <para> </para>
        ///     <para>  // Input of shape (2,1,2)</para>
        ///     <para>  x = [[[ 1.1,  2.2]],</para>
        ///     <para>       [[ 3.3,  4.4]]]</para>
        ///     <para> </para>
        ///     <para>  // gamma parameter of length 1</para>
        ///     <para>  gamma = [1.5]</para>
        ///     <para> </para>
        ///     <para>  // beta parameter of length 1</para>
        ///     <para>  beta = [0.5]</para>
        ///     <para> </para>
        ///     <para>  // Instance normalization is calculated with the above formula</para>
        ///     <para>  InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],</para>
        ///     <para>                                [[-0.99752653,  1.99752724]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\instance_norm.cc:L95</para>
        /// </summary>
        /// <param name="data">An n-dimensional input array (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].</param>
        /// <param name="gamma">A vector of length 'channel', which multiplies the normalized input.</param>
        /// <param name="beta">A vector of length 'channel', which is added to the product of the normalized input and the weight.</param>
        /// <param name="eps">An `epsilon` parameter to prevent division by 0.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol InstanceNorm(Symbol data, Symbol gamma, Symbol beta, float eps = 0.001f,
            string symbol_name = "")
        {
            return new Operator("InstanceNorm")
                .SetParam("eps", eps)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Normalize the input array using the L2 norm.</para>
        ///     <para> </para>
        ///     <para>For 1-D NDArray, it computes::</para>
        ///     <para> </para>
        ///     <para>  out = data / sqrt(sum(data ** 2) + eps)</para>
        ///     <para> </para>
        ///     <para>For N-D NDArray, if the input array has shape (N, N, ..., N),</para>
        ///     <para> </para>
        ///     <para>with ``mode`` = ``instance``, it normalizes each instance in the multidimensional</para>
        ///     <para>array by its L2 norm.::</para>
        ///     <para> </para>
        ///     <para>  for i in 0...N</para>
        ///     <para>    out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)</para>
        ///     <para> </para>
        ///     <para>with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::</para>
        ///     <para> </para>
        ///     <para>  for i in 0...N</para>
        ///     <para>    out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)</para>
        ///     <para> </para>
        ///     <para>with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position</para>
        ///     <para>in the array by its L2 norm.::</para>
        ///     <para> </para>
        ///     <para>  for dim in 2...N</para>
        ///     <para>    for i in 0...N</para>
        ///     <para>
        ///         out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) +
        ///         eps)
        ///     </para>
        ///     <para>          -dim-</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[[1,2],</para>
        ///     <para>        [3,4]],</para>
        ///     <para>       [[2,2],</para>
        ///     <para>        [5,6]]]</para>
        ///     <para> </para>
        ///     <para>  L2Normalization(x, mode='instance')</para>
        ///     <para>  =[[[ 0.18257418  0.36514837]</para>
        ///     <para>     [ 0.54772252  0.73029673]]</para>
        ///     <para>    [[ 0.24077171  0.24077171]</para>
        ///     <para>     [ 0.60192931  0.72231513]]]</para>
        ///     <para> </para>
        ///     <para>  L2Normalization(x, mode='channel')</para>
        ///     <para>  =[[[ 0.31622776  0.44721359]</para>
        ///     <para>     [ 0.94868326  0.89442718]]</para>
        ///     <para>    [[ 0.37139067  0.31622776]</para>
        ///     <para>     [ 0.92847669  0.94868326]]]</para>
        ///     <para> </para>
        ///     <para>  L2Normalization(x, mode='spatial')</para>
        ///     <para>  =[[[ 0.44721359  0.89442718]</para>
        ///     <para>     [ 0.60000002  0.80000001]]</para>
        ///     <para>    [[ 0.70710677  0.70710677]</para>
        ///     <para>     [ 0.6401844   0.76822126]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\l2_normalization.cc:L196</para>
        /// </summary>
        /// <param name="data">Input array to normalize.</param>
        /// <param name="eps">A small constant for numerical stability.</param>
        /// <param name="mode">Specify the dimension along which to compute L2 norm.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol L2Normalization(Symbol data, float eps = 1e-10f,
            L2normalizationMode mode = L2normalizationMode.Instance, string symbol_name = "")
        {
            return new Operator("L2Normalization")
                .SetParam("eps", eps)
                .SetParam("mode", MxUtil.EnumToString<L2normalizationMode>(mode, L2normalizationModeConvert))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Make your own loss function in network construction.</para>
        ///     <para> </para>
        ///     <para>This operator accepts a customized loss function symbol as a terminal loss and</para>
        ///     <para>the symbol should be an operator with no backward dependency.</para>
        ///     <para>The output of this function is the gradient of loss with respect to the input data.</para>
        ///     <para> </para>
        ///     <para>For example, if you are a making a cross entropy loss function. Assume ``out`` is the</para>
        ///     <para>predicted output and ``label`` is the true label, then the cross entropy can be defined as::</para>
        ///     <para> </para>
        ///     <para>  cross_entropy = label * log(out) + (1 - label) * log(1 - out)</para>
        ///     <para>  loss = MakeLoss(cross_entropy)</para>
        ///     <para> </para>
        ///     <para>We will need to use ``MakeLoss`` when we are creating our own loss function or we want to</para>
        ///     <para>combine multiple loss functions. Also we may want to stop some variables' gradients</para>
        ///     <para>from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.</para>
        ///     <para> </para>
        ///     <para>In addition, we can give a scale to the loss by setting ``grad_scale``,</para>
        ///     <para>so that the gradient of the loss will be rescaled in the backpropagation.</para>
        ///     <para> </para>
        ///     <para>.. note:: This operator should be used as a Symbol instead of NDArray.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\make_loss.cc:L71</para>
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="grad_scale">Gradient scale as a supplement to unary and binary operators</param>
        /// <param name="valid_thresh">
        ///     clip each element in the array to 0 when it is less than ``valid_thresh``. This is used when
        ///     ``normalization`` is set to ``'valid'``.
        /// </param>
        /// <param name="normalization">
        ///     If this is set to null, the output gradient will not be normalized. If this is set to
        ///     batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be
        ///     divided by the number of valid input elements.
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol MakeLoss(Symbol data, float grad_scale = 1f, float valid_thresh = 0f,
            MakelossNormalization normalization = MakelossNormalization.Null, string symbol_name = "")
        {
            return new Operator("MakeLoss")
                .SetParam("grad_scale", grad_scale)
                .SetParam("valid_thresh", valid_thresh)
                .SetParam("normalization",
                    MxUtil.EnumToString<MakelossNormalization>(normalization, MakelossNormalizationConvert))
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>This operator is DEPRECATED.</para>
        ///     <para>Perform pooling on the input.</para>
        ///     <para> </para>
        ///     <para>The shapes for 2-D pooling is</para>
        ///     <para> </para>
        ///     <para>- **data**: *(batch_size, channel, height, width)*</para>
        ///     <para>- **out**: *(batch_size, num_filter, out_height, out_width)*, with::</para>
        ///     <para> </para>
        ///     <para>    out_height = f(height, kernel[0], pad[0], stride[0])</para>
        ///     <para>    out_width = f(width, kernel[1], pad[1], stride[1])</para>
        ///     <para> </para>
        ///     <para>The definition of *f* depends on ``pooling_convention``, which has two options:</para>
        ///     <para> </para>
        ///     <para>- **valid** (default)::</para>
        ///     <para> </para>
        ///     <para>    f(x, k, p, s) = floor((x+2*p-k)/s)+1</para>
        ///     <para> </para>
        ///     <para>- **full**, which is compatible with Caffe::</para>
        ///     <para> </para>
        ///     <para>    f(x, k, p, s) = ceil((x+2*p-k)/s)+1</para>
        ///     <para> </para>
        ///     <para>But ``global_pool`` is set to be true, then do a global pooling, namely reset</para>
        ///     <para>``kernel=(height, width)``.</para>
        ///     <para> </para>
        ///     <para>Three pooling options are supported by ``pool_type``:</para>
        ///     <para> </para>
        ///     <para>- **avg**: average pooling</para>
        ///     <para>- **max**: max pooling</para>
        ///     <para>- **sum**: sum pooling</para>
        ///     <para> </para>
        ///     <para>1-D pooling is special case of 2-D pooling with *weight=1* and</para>
        ///     <para>*kernel[1]=1*.</para>
        ///     <para> </para>
        ///     <para>For 3-D pooling, an additional *depth* dimension is added before</para>
        ///     <para>*height*. Namely the input data will have shape *(batch_size, channel, depth,</para>
        ///     <para>height, width)*.</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\pooling_v1.cc:L104</para>
        /// </summary>
        /// <param name="data">Input data to the pooling operator.</param>
        /// <param name="kernel">pooling kernel size: (y, x) or (d, y, x)</param>
        /// <param name="pool_type">Pooling type to be applied.</param>
        /// <param name="global_pool">Ignore kernel size, do global pooling based on current input feature map. </param>
        /// <param name="pooling_convention">Pooling convention to be applied.</param>
        /// <param name="stride">stride: for pooling (y, x) or (d, y, x)</param>
        /// <param name="pad">pad for pooling: (y, x) or (d, y, x)</param>
        /// <returns>returns new symbol</returns>
        public static Symbol PoolingV1(Symbol data, Shape kernel = null,
            PoolingV1PoolType pool_type = PoolingV1PoolType.Max, bool global_pool = false,
            PoolingV1PoolingConvention pooling_convention = PoolingV1PoolingConvention.Valid, Shape stride = null,
            Shape pad = null, string symbol_name = "")
        {
            if (kernel == null) kernel = new Shape();
            if (stride == null) stride = new Shape();
            if (pad == null) pad = new Shape();

            return new Operator("Pooling_v1")
                .SetParam("kernel", kernel)
                .SetParam("pool_type", MxUtil.EnumToString<PoolingV1PoolType>(pool_type, PoolingV1PoolTypeConvert))
                .SetParam("global_pool", global_pool)
                .SetParam("pooling_convention",
                    MxUtil.EnumToString<PoolingV1PoolingConvention>(pooling_convention,
                        PoolingV1PoolingConventionConvert))
                .SetParam("stride", stride)
                .SetParam("pad", pad)
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Performs region of interest(ROI) pooling on the input array.</para>
        ///     <para> </para>
        ///     <para>ROI pooling is a variant of a max pooling layer, in which the output size is fixed and</para>
        ///     <para>region of interest is a parameter. Its purpose is to perform max pooling on the inputs</para>
        ///     <para>of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net</para>
        ///     <para>layer mostly used in training a `Fast R-CNN` network for object detection.</para>
        ///     <para> </para>
        ///     <para>This operator takes a 4D feature map as an input array and region proposals as `rois`,</para>
        ///     <para>then it pools over sub-regions of input and produces a fixed-sized output array</para>
        ///     <para>regardless of the ROI size.</para>
        ///     <para> </para>
        ///     <para>To crop the feature map accordingly, you can resize the bounding box coordinates</para>
        ///     <para>by changing the parameters `rois` and `spatial_scale`.</para>
        ///     <para> </para>
        ///     <para>The cropped feature maps are pooled by standard max pooling operation to a fixed size output</para>
        ///     <para>indicated by a `pooled_size` parameter. batch_size will change to the number of region</para>
        ///     <para>bounding boxes after `ROIPooling`.</para>
        ///     <para> </para>
        ///     <para>The size of each region of interest doesn't have to be perfectly divisible by</para>
        ///     <para>the number of pooling sections(`pooled_size`).</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>  x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],</para>
        ///     <para>         [  6.,   7.,   8.,   9.,  10.,  11.],</para>
        ///     <para>         [ 12.,  13.,  14.,  15.,  16.,  17.],</para>
        ///     <para>         [ 18.,  19.,  20.,  21.,  22.,  23.],</para>
        ///     <para>         [ 24.,  25.,  26.,  27.,  28.,  29.],</para>
        ///     <para>         [ 30.,  31.,  32.,  33.,  34.,  35.],</para>
        ///     <para>         [ 36.,  37.,  38.,  39.,  40.,  41.],</para>
        ///     <para>         [ 42.,  43.,  44.,  45.,  46.,  47.]]]]</para>
        ///     <para> </para>
        ///     <para>  // region of interest i.e. bounding box coordinates.</para>
        ///     <para>  y = [[0,0,0,4,4]]</para>
        ///     <para> </para>
        ///     <para>  // returns array of shape (2,2) according to the given roi with max pooling.</para>
        ///     <para>  ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],</para>
        ///     <para>                                    [ 26.,  28.]]]]</para>
        ///     <para> </para>
        ///     <para>  // region of interest is changed due to the change in `spacial_scale` parameter.</para>
        ///     <para>  ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],</para>
        ///     <para>                                    [ 19.,  21.]]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\roi_pooling.cc:L295</para>
        /// </summary>
        /// <param name="data">The input array to the pooling operator,  a 4D Feature maps </param>
        /// <param name="rois">
        ///     Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2)
        ///     are top left and bottom right corners of designated region of interest. `batch_index` indicates the index of
        ///     corresponding image in the input array
        /// </param>
        /// <param name="pooled_size">ROI pooling output shape (h,w) </param>
        /// <param name="spatial_scale">
        ///     Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        ///     of total stride in convolutional layers
        /// </param>
        /// <returns>returns new symbol</returns>
        public static Symbol ROIPooling(Symbol data, Symbol rois, Shape pooled_size, float spatial_scale,
            string symbol_name = "")
        {
            return new Operator("ROIPooling")
                .SetParam("pooled_size", pooled_size)
                .SetParam("spatial_scale", spatial_scale)
                .SetInput("data", data)
                .SetInput("rois", rois)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Takes the last element of a sequence.</para>
        ///     <para> </para>
        ///     <para>This function takes an n-dimensional input array of the form</para>
        ///     <para>[max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array</para>
        ///     <para>of the form [batch_size, other_feature_dims].</para>
        ///     <para> </para>
        ///     <para>Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be</para>
        ///     <para>an input array of positive ints of dimension [batch_size]. To use this parameter,</para>
        ///     <para>set `use_sequence_length` to `True`, otherwise each example in the batch is assumed</para>
        ///     <para>to have the max sequence length.</para>
        ///     <para> </para>
        ///     <para>.. note:: Alternatively, you can also use `take` operator.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[[  1.,   2.,   3.],</para>
        ///     <para>         [  4.,   5.,   6.],</para>
        ///     <para>         [  7.,   8.,   9.]],</para>
        ///     <para> </para>
        ///     <para>        [[ 10.,   11.,   12.],</para>
        ///     <para>         [ 13.,   14.,   15.],</para>
        ///     <para>         [ 16.,   17.,   18.]],</para>
        ///     <para> </para>
        ///     <para>        [[  19.,   20.,   21.],</para>
        ///     <para>         [  22.,   23.,   24.],</para>
        ///     <para>         [  25.,   26.,   27.]]]</para>
        ///     <para> </para>
        ///     <para>   // returns last sequence when sequence_length parameter is not used</para>
        ///     <para>   SequenceLast(x) = [[  19.,   20.,   21.],</para>
        ///     <para>                      [  22.,   23.,   24.],</para>
        ///     <para>                      [  25.,   26.,   27.]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length is used</para>
        ///     <para>   SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =</para>
        ///     <para>            [[  1.,   2.,   3.],</para>
        ///     <para>             [  4.,   5.,   6.],</para>
        ///     <para>             [  7.,   8.,   9.]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length is used</para>
        ///     <para>   SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =</para>
        ///     <para>            [[  1.,    2.,   3.],</para>
        ///     <para>             [  13.,  14.,  15.],</para>
        ///     <para>             [  25.,  26.,  27.]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\sequence_last.cc:L100</para>
        /// </summary>
        /// <param name="data">
        ///     n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where
        ///     n>2
        /// </param>
        /// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
        /// <param name="use_sequence_length">
        ///     If set to true, this layer takes in an extra input parameter `sequence_length` to
        ///     specify variable length sequence
        /// </param>
        /// <param name="axis">The sequence axis. Only values of 0 and 1 are currently supported.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SequenceLast(Symbol data, Symbol sequence_length, bool use_sequence_length = false,
            int axis = 0, string symbol_name = "")
        {
            return new Operator("SequenceLast")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Sets all elements outside the sequence to a constant value.</para>
        ///     <para> </para>
        ///     <para>This function takes an n-dimensional input array of the form</para>
        ///     <para>[max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.</para>
        ///     <para> </para>
        ///     <para>Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`</para>
        ///     <para>should be an input array of positive ints of dimension [batch_size].</para>
        ///     <para>To use this parameter, set `use_sequence_length` to `True`,</para>
        ///     <para>otherwise each example in the batch is assumed to have the max sequence length and</para>
        ///     <para>this operator works as the `identity` operator.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[[  1.,   2.,   3.],</para>
        ///     <para>         [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>        [[  7.,   8.,   9.],</para>
        ///     <para>         [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>        [[ 13.,  14.,   15.],</para>
        ///     <para>         [ 16.,  17.,   18.]]]</para>
        ///     <para> </para>
        ///     <para>   // Batch 1</para>
        ///     <para>   B1 = [[  1.,   2.,   3.],</para>
        ///     <para>         [  7.,   8.,   9.],</para>
        ///     <para>         [ 13.,  14.,  15.]]</para>
        ///     <para> </para>
        ///     <para>   // Batch 2</para>
        ///     <para>   B2 = [[  4.,   5.,   6.],</para>
        ///     <para>         [ 10.,  11.,  12.],</para>
        ///     <para>         [ 16.,  17.,  18.]]</para>
        ///     <para> </para>
        ///     <para>   // works as identity operator when sequence_length parameter is not used</para>
        ///     <para>   SequenceMask(x) = [[[  1.,   2.,   3.],</para>
        ///     <para>                       [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>                      [[  7.,   8.,   9.],</para>
        ///     <para>                       [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>                      [[ 13.,  14.,   15.],</para>
        ///     <para>                       [ 16.,  17.,   18.]]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length [1,1] means 1 of each batch will be kept</para>
        ///     <para>   // and other rows are masked with default mask value = 0</para>
        ///     <para>   SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =</para>
        ///     <para>                [[[  1.,   2.,   3.],</para>
        ///     <para>                  [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>                 [[  0.,   0.,   0.],</para>
        ///     <para>                  [  0.,   0.,   0.]],</para>
        ///     <para> </para>
        ///     <para>                 [[  0.,   0.,   0.],</para>
        ///     <para>                  [  0.,   0.,   0.]]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept</para>
        ///     <para>   // and other rows are masked with value = 1</para>
        ///     <para>   SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =</para>
        ///     <para>                [[[  1.,   2.,   3.],</para>
        ///     <para>                  [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>                 [[  7.,   8.,   9.],</para>
        ///     <para>                  [  10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>                 [[   1.,   1.,   1.],</para>
        ///     <para>                  [  16.,  17.,  18.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\sequence_mask.cc:L186</para>
        /// </summary>
        /// <param name="data">
        ///     n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where
        ///     n>2
        /// </param>
        /// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
        /// <param name="use_sequence_length">
        ///     If set to true, this layer takes in an extra input parameter `sequence_length` to
        ///     specify variable length sequence
        /// </param>
        /// <param name="value">The value to be used as a mask.</param>
        /// <param name="axis">The sequence axis. Only values of 0 and 1 are currently supported.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SequenceMask(Symbol data, Symbol sequence_length, bool use_sequence_length = false,
            float value = 0f, int axis = 0, string symbol_name = "")
        {
            return new Operator("SequenceMask")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("value", value)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Reverses the elements of each sequence.</para>
        ///     <para> </para>
        ///     <para>
        ///         This function takes an n-dimensional input array of the form [max_sequence_length, batch_size,
        ///         other_feature_dims]
        ///     </para>
        ///     <para>and returns an array of the same shape.</para>
        ///     <para> </para>
        ///     <para>Parameter `sequence_length` is used to handle variable-length sequences.</para>
        ///     <para>`sequence_length` should be an input array of positive ints of dimension [batch_size].</para>
        ///     <para>To use this parameter, set `use_sequence_length` to `True`,</para>
        ///     <para>otherwise each example in the batch is assumed to have the max sequence length.</para>
        ///     <para> </para>
        ///     <para>Example::</para>
        ///     <para> </para>
        ///     <para>   x = [[[  1.,   2.,   3.],</para>
        ///     <para>         [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>        [[  7.,   8.,   9.],</para>
        ///     <para>         [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>        [[ 13.,  14.,   15.],</para>
        ///     <para>         [ 16.,  17.,   18.]]]</para>
        ///     <para> </para>
        ///     <para>   // Batch 1</para>
        ///     <para>   B1 = [[  1.,   2.,   3.],</para>
        ///     <para>         [  7.,   8.,   9.],</para>
        ///     <para>         [ 13.,  14.,  15.]]</para>
        ///     <para> </para>
        ///     <para>   // Batch 2</para>
        ///     <para>   B2 = [[  4.,   5.,   6.],</para>
        ///     <para>         [ 10.,  11.,  12.],</para>
        ///     <para>         [ 16.,  17.,  18.]]</para>
        ///     <para> </para>
        ///     <para>   // returns reverse sequence when sequence_length parameter is not used</para>
        ///     <para>   SequenceReverse(x) = [[[ 13.,  14.,   15.],</para>
        ///     <para>                          [ 16.,  17.,   18.]],</para>
        ///     <para> </para>
        ///     <para>                         [[  7.,   8.,   9.],</para>
        ///     <para>                          [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>                         [[  1.,   2.,   3.],</para>
        ///     <para>                          [  4.,   5.,   6.]]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length [2,2] means 2 rows of</para>
        ///     <para>   // both batch B1 and B2 will be reversed.</para>
        ///     <para>   SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =</para>
        ///     <para>                     [[[  7.,   8.,   9.],</para>
        ///     <para>                       [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>                      [[  1.,   2.,   3.],</para>
        ///     <para>                       [  4.,   5.,   6.]],</para>
        ///     <para> </para>
        ///     <para>                      [[ 13.,  14.,   15.],</para>
        ///     <para>                       [ 16.,  17.,   18.]]]</para>
        ///     <para> </para>
        ///     <para>   // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3</para>
        ///     <para>   // will be reversed.</para>
        ///     <para>   SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =</para>
        ///     <para>                    [[[  7.,   8.,   9.],</para>
        ///     <para>                      [ 16.,  17.,  18.]],</para>
        ///     <para> </para>
        ///     <para>                     [[  1.,   2.,   3.],</para>
        ///     <para>                      [ 10.,  11.,  12.]],</para>
        ///     <para> </para>
        ///     <para>                     [[ 13.,  14,   15.],</para>
        ///     <para>                      [  4.,   5.,   6.]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\sequence_reverse.cc:L122</para>
        /// </summary>
        /// <param name="data">n-dimensional input array of the form [max_sequence_length, batch_size, other dims] where n>2 </param>
        /// <param name="sequence_length">vector of sequence lengths of the form [batch_size]</param>
        /// <param name="use_sequence_length">
        ///     If set to true, this layer takes in an extra input parameter `sequence_length` to
        ///     specify variable length sequence
        /// </param>
        /// <param name="axis">The sequence axis. Only 0 is currently supported.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SequenceReverse(Symbol data, Symbol sequence_length, bool use_sequence_length = false,
            int axis = 0, string symbol_name = "")
        {
            return new Operator("SequenceReverse")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Applies a spatial transformer to input feature map.</para>
        /// </summary>
        /// <param name="data">Input data to the SpatialTransformerOp.</param>
        /// <param name="loc">
        ///     localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the
        ///     weight and bias with identity tranform.
        /// </param>
        /// <param name="target_shape">output shape(h, w) of spatial transformer: (y, x)</param>
        /// <param name="transform_type">transformation type</param>
        /// <param name="sampler_type">sampling type</param>
        /// <param name="cudnn_off">whether to turn cudnn off</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SpatialTransformer(Symbol data, Symbol loc, SpatialtransformerTransformType transform_type,
            SpatialtransformerSamplerType sampler_type, Shape target_shape = null, bool? cudnn_off = null,
            string symbol_name = "")
        {
            if (target_shape == null) target_shape = new Shape();

            return new Operator("SpatialTransformer")
                .SetParam("target_shape", target_shape)
                .SetParam("transform_type",
                    MxUtil.EnumToString<SpatialtransformerTransformType>(transform_type,
                        SpatialtransformerTransformTypeConvert))
                .SetParam("sampler_type",
                    MxUtil.EnumToString<SpatialtransformerSamplerType>(sampler_type,
                        SpatialtransformerSamplerTypeConvert))
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .SetInput("loc", loc)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Computes support vector machine based transformation of the input.</para>
        ///     <para> </para>
        ///     <para>This tutorial demonstrates using SVM as output layer for classification instead of softmax:</para>
        ///     <para>https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.</para>
        ///     <para> </para>
        ///     <para> </para>
        /// </summary>
        /// <param name="data">Input data for SVM transformation.</param>
        /// <param name="label">Class label for the input data.</param>
        /// <param name="margin">The loss function penalizes outputs that lie outside this margin. Default margin is 1.</param>
        /// <param name="regularization_coefficient">
        ///     Regularization parameter for the SVM. This balances the tradeoff between
        ///     coefficient size and error.
        /// </param>
        /// <param name="use_linear">Whether to use L1-SVM objective. L2-SVM objective is used by default.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol SVMOutput(Symbol data, Symbol label, float margin = 1f,
            float regularization_coefficient = 1f, bool use_linear = false, string symbol_name = "")
        {
            return new Operator("SVMOutput")
                .SetParam("margin", margin)
                .SetParam("regularization_coefficient", regularization_coefficient)
                .SetParam("use_linear", use_linear)
                .SetInput("data", data)
                .SetInput("label", label)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para> </para>
        /// </summary>
        /// <param name="lhs">Left operand to the function.</param>
        /// <param name="rhs">Right operand to the function.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol OnehotEncode(Symbol lhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("_onehot_encode")
                .SetParam("lhs", lhs)
                .SetParam("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>
        ///         Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs
        ///         and values indicated by mhs. This function assume rhs uses 0-based index.
        ///     </para>
        /// </summary>
        /// <param name="lhs">Left operand to the function.</param>
        /// <param name="mhs">Middle operand to the function.</param>
        /// <param name="rhs">Right operand to the function.</param>
        /// <returns>returns new symbol</returns>
        public static Symbol FillElement0Index(Symbol lhs, Symbol mhs, Symbol rhs, string symbol_name = "")
        {
            return new Operator("fill_element_0index")
                .SetParam("lhs", lhs)
                .SetParam("mhs", mhs)
                .SetParam("rhs", rhs)
                .CreateSymbol(symbol_name);
        }

        /// <summary>
        ///     <para>Decode an image, clip to (x0, y0, x1, y1), subtract mean, and write to buffer</para>
        /// </summary>
        /// <param name="mean">image mean</param>
        /// <param name="index">buffer position for output</param>
        /// <param name="x0">x0</param>
        /// <param name="y0">y0</param>
        /// <param name="x1">x1</param>
        /// <param name="y1">y1</param>
        /// <param name="c">channel</param>
        /// <param name="size">length of str_img</param>
        /// <returns>returns new symbol</returns>
        public static Symbol Imdecode(Symbol mean, int index, int x0, int y0, int x1, int y1, int c, int size,
            string symbol_name = "")
        {
            return new Operator("_imdecode")
                .SetParam("index", index)
                .SetParam("x0", x0)
                .SetParam("y0", y0)
                .SetParam("x1", x1)
                .SetParam("y1", y1)
                .SetParam("c", c)
                .SetParam("size", size)
                .SetInput("mean", mean)
                .CreateSymbol(symbol_name);
        }

        public static Symbol Linspace(float start, float stop, int num, bool endpoint = true, Context ctx = null, DType dtype = null, string symbol_name = "")
        {
            if (ctx == null)
                ctx = Context.CurrentContext;

            if (dtype == null)
                dtype = DType.Float32;

            return new Operator("linspace")
                .SetParam("start", start)
                .SetParam("stop", stop)
                .SetParam("num", num)
                .SetParam("endpoint", endpoint)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .CreateSymbol(symbol_name);
        }

        public static Symbol StopGradient(Symbol data, string symbol_name = "")
        {
            return new Operator("stop_gradient")
                .SetInput("data", data)
                .CreateSymbol(symbol_name);
        }

        public static Symbol GroupNorm(Symbol data, Symbol gamma, Symbol beta, float eps = 0.001f,
            string symbol_name = "")
        {
            return new Operator("GroupNorm")
                .SetParam("eps", eps)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .CreateSymbol(symbol_name);
        }

        public static SymbolList MultiSumSq(SymbolList arrays, int num_arrays)
        {
            return new Operator("multi_sum_sq")
                .SetInput(arrays).SetParam("num_arrays", num_arrays)
                .CreateSymbol().ToList();
        }
    }
}