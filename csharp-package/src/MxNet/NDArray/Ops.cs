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
using System.Linq;
using System.Runtime.InteropServices;
using MxNet.Interop;
using MxNet.Numpy;

namespace MxNet
{
    [Obsolete("Legacy API after MxNet v2, will be deprecated in v3", false)]
    public partial class nd
    {
        public static NDImgApi Image = new NDImgApi();
        public static NDContribApi Contrib = new NDContribApi();

        private static readonly List<string> LeakyreluActTypeConvert =
            new List<string> {"elu", "gelu", "leaky", "prelu", "rrelu", "selu"};

        private static readonly List<string> ActivationActTypeConvert =
            new List<string> {"relu", "sigmoid", "softrelu", "softsign", "tanh"};

        private static readonly List<string> ConvolutionCudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        private static readonly List<string> ConvolutionLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"};

        private static readonly List<string> CtclossBlankLabelConvert = new List<string> {"first", "last"};

        private static readonly List<string> DeconvolutionCudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        private static readonly List<string> DeconvolutionLayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"};

        private static readonly List<string> DropoutModeConvert = new List<string> {"always", "training"};

        private static readonly List<string> PoolingPoolTypeConvert = new List<string> {"avg", "lp", "max", "sum"};

        private static readonly List<string> PoolingPoolingConventionConvert =
            new List<string> {"full", "same", "valid"};

        private static readonly List<string> PoolingLayoutConvert = new List<string>
            {"NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"};

        private static readonly List<string> SoftmaxactivationModeConvert = new List<string> {"channel", "instance"};

        private static readonly List<string> UpsamplingSampleTypeConvert = new List<string> {"bilinear", "nearest"};
        private static readonly List<string> UpsamplingMultiInputModeConvert = new List<string> {"concat", "sum"};

        private static readonly List<string> PadModeConvert = new List<string> {"constant", "edge", "reflect"};

        private static readonly List<string> RNNModeConvert = new List<string> {"gru", "lstm", "rnn_relu", "rnn_tanh"};

        private static readonly List<string> SoftmaxoutputNormalizationConvert =
            new List<string> {"batch", "null", "valid"};

        private static readonly List<string> PickModeConvert = new List<string> {"clip", "wrap"};

        private static readonly List<string> NormOutDtypeConvert = new List<string>
            {"float16", "float32", "float64", "int32", "int64", "int8"};

        private static readonly List<string>
            CastStorageStypeConvert = new List<string> {"csr", "default", "row_sparse"};

        private static readonly List<string> DotForwardStypeConvert = new List<string> {"csr", "default", "row_sparse"};

        private static readonly List<string> BatchDotForwardStypeConvert =
            new List<string> {"csr", "default", "row_sparse"};

        private static readonly List<string> TakeModeConvert = new List<string> {"clip", "raise", "wrap"};

        private static readonly List<string> TopkRetTypConvert = new List<string> {"both", "indices", "mask", "value"};

        private static readonly List<string> ConvolutionV1CudnnTuneConvert =
            new List<string> {"fastest", "limited_workspace", "off"};

        private static readonly List<string> ConvolutionV1LayoutConvert =
            new List<string> {"NCDHW", "NCHW", "NDHWC", "NHWC"};

        private static readonly List<string> GridgeneratorTransformTypeConvert = new List<string> {"affine", "warp"};

        private static readonly List<string> L2normalizationModeConvert =
            new List<string> {"channel", "instance", "spatial"};

        private static readonly List<string> MakelossNormalizationConvert = new List<string> {"batch", "null", "valid"};

        private static readonly List<string> PoolingV1PoolTypeConvert = new List<string> {"avg", "max", "sum"};
        private static readonly List<string> PoolingV1PoolingConventionConvert = new List<string> {"full", "valid"};

        private static readonly List<string> SpatialtransformerTransformTypeConvert = new List<string> {"affine"};
        private static readonly List<string> SpatialtransformerSamplerTypeConvert = new List<string> {"bilinear"};

        public static NDArray CustomFunction()
        {
            return new Operator("_CustomFunction")
                .Invoke();
        }

        public static NDArray CachedOp(NDArrayList data)
        {
            return new Operator("_CachedOp")
                .SetInput(data)
                .Invoke();
        }

        public static NDArray Cvimdecode(byte[] buf, int flag = 1, bool to_rgb = true)
        {
            return new Operator("_cvimdecode")
                .SetInput("buf", new NDArray(buf))
                .SetParam("flag", flag)
                .SetParam("to_rgb", to_rgb)
                .Invoke();
        }

        public static NDArray Cvimread(string filename, int flag = 1, bool to_rgb = true)
        {
            return new Operator("_cvimread")
                .SetParam("filename", filename)
                .SetParam("flag", flag)
                .SetParam("to_rgb", to_rgb)
                .Invoke();
        }

        public static NDArray Cvimresize(NDArray data, int w, int h, int interp = 1)
        {
            return new Operator("_cvimresize")
                .SetInput("data", data)
                .SetParam("w", w)
                .SetParam("h", h)
                .SetParam("interp", interp)
                .Invoke();
        }

        public static NDArray CvcopyMakeBorder(NDArray data, int top, int bot, int left, int right, int type = 0,
            Tuple<double> values = null)
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
                .Invoke();
        }

        public static NDArray CopyTo(NDArray data)
        {
            return new Operator("_copyto")
                .SetParam("data", data)
                .Invoke();
        }

        public static NDArray Array(Array data, Context ctx = null)
        {
            DType dtype = DType.InferDtype(data);

            return new NDArray(data, ctx, dtype);
        }
        
        public static NDArray NoGradient()
        {
            return new Operator("_NoGradient")
                .Invoke();
        }

        public static NDArray BatchNormV1(NDArray data, NDArray gamma, NDArray beta, float eps = 0.001f,
            float momentum = 0.9f, bool fix_gamma = true, bool use_global_stats = false, bool output_mean_var = false)
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
                .Invoke();
        }

        public static NDArray MpAdamWUpdate(NDArray weight, NDArray grad, NDArray mean, NDArray var, NDArray weight32,
            NDArray rescale_grad, float lr, float eta, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f,
            float wd = 0f, float clip_gradient = -1f)
        {
            rescale_grad = GetRescaleGrad(rescale_grad, ctx: weight.Context);

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
                .Invoke();
        }

        public static NDArray AdamWUpdate(NDArray weight, NDArray grad, NDArray mean, NDArray var, NDArray rescale_grad,
            float lr, float eta, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f, float wd = 0f,
            float clip_gradient = -1f)
        {
            rescale_grad = GetRescaleGrad(rescale_grad, ctx: weight.Context);
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
                .Invoke();
        }

        public static NDArray KhatriRao(NDArrayList args)
        {
            return new Operator("khatri_rao")
                .SetInput(args)
                .Invoke();
        }

        public static NDArray Foreach(NDArray fn, NDArrayList data, int num_args, int num_outputs, int num_out_data,
            Tuple<double> in_state_locs, Tuple<double> in_data_locs, Tuple<double> remain_locs)
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
                .Invoke();
        }

        public static NDArray WhileLoop(NDArray cond, NDArray func, NDArrayList data, int num_args, int num_outputs,
            int num_out_data, int max_iterations, Tuple<double> cond_input_locs, Tuple<double> func_input_locs,
            Tuple<double> func_var_locs)
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
                .Invoke();
        }

        public static NDArray Cond(NDArray cond, NDArray then_branch, NDArray else_branch, NDArrayList data,
            int num_args, int num_outputs, Tuple<double> cond_input_locs, Tuple<double> then_input_locs,
            Tuple<double> else_input_locs)
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
                .Invoke();
        }

        public static NDArray Custom(NDArrayList data, string op_type)
        {
            return new Operator("Custom")
                .SetParam("op_type", op_type)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray IdentityAttachKLSparseReg(NDArray data, float sparseness_target = 0.1f,
            float penalty = 0.001f, float momentum = 0.9f)
        {
            return new Operator("IdentityAttachKLSparseReg")
                .SetParam("sparseness_target", sparseness_target)
                .SetParam("penalty", penalty)
                .SetParam("momentum", momentum)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LeakyReLU(NDArray data, NDArray gamma = null,
            ReluActType act_type = ReluActType.Leaky, float slope = 0.25f, float lower_bound = 0.125f,
            float upper_bound = 0.334f)
        {
            return new Operator("LeakyReLU")
                .SetParam("act_type", MxUtil.EnumToString<ReluActType>(act_type, LeakyreluActTypeConvert))
                .SetParam("slope", slope)
                .SetParam("lower_bound", lower_bound)
                .SetParam("upper_bound", upper_bound)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .Invoke();
        }

        public static NDArray SoftmaxCrossEntropy(NDArray data, NDArray label)
        {
            return new Operator("softmax_cross_entropy")
                .SetInput("data", data)
                .SetInput("label", label)
                .Invoke();
        }

        public static NDArray Activation(NDArray data, ActivationType act_type)
        {
            return new Operator("Activation")
                .SetParam("act_type", MxUtil.EnumToString<ActivationType>(act_type, ActivationActTypeConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BatchNorm(NDArray data, NDArray gamma, NDArray beta, NDArray moving_mean,
            NDArray moving_var, double eps = 0.001, float momentum = 0.9f, bool fix_gamma = true,
            bool use_global_stats = false, bool output_mean_var = false, int axis = 1, bool cudnn_off = false)
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
                .Invoke();
        }

        public static NDArray Concat(NDArrayList data, int dim = 1)
        {
            return new Operator("concat")
                .SetParam("dim", dim)
                .SetParam("num_args", data.Length)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray RnnParamConcat(NDArrayList data, int num_args, int dim = 1)
        {
            return new Operator("_rnn_param_concat")
                .SetParam("num_args", num_args)
                .SetParam("dim", dim)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray Convolution(NDArray data, NDArray weight, NDArray bias, Shape kernel, int num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, int num_group = 1, ulong workspace = 1024,
            bool no_bias = false, ConvolutionCudnnTune? cudnn_tune = null, bool cudnn_off = false,
            ConvolutionLayout? layout = null)
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
                .Invoke();
        }

        public static NDArray CTCLoss(NDArray data, NDArray label, NDArray data_lengths, NDArray label_lengths,
            bool use_data_lengths = false, bool use_label_lengths = false,
            CtclossBlankLabel blank_label = CtclossBlankLabel.First)
        {
            return new Operator("CTCLoss")
                .SetParam("use_data_lengths", use_data_lengths)
                .SetParam("use_label_lengths", use_label_lengths)
                .SetParam("blank_label", MxUtil.EnumToString<CtclossBlankLabel>(blank_label, CtclossBlankLabelConvert))
                .SetInput("data", data)
                .SetInput("label", label)
                .SetInput("data_lengths", data_lengths)
                .SetInput("label_lengths", label_lengths)
                .Invoke();
        }

        public static NDArray Deconvolution(NDArray data, NDArray weight, NDArray bias, Shape kernel, uint num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, Shape adj = null, Shape target_shape = null,
            uint num_group = 1, ulong workspace = 512, bool no_bias = true, DeconvolutionCudnnTune? cudnn_tune = null,
            bool cudnn_off = false, DeconvolutionLayout? layout = null)
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
                .Invoke();
        }

        public static NDArray Dropout(NDArray data, float p = 0.5f, DropoutMode mode = DropoutMode.Training,
            Shape axes = null, bool? cudnn_off = false)
        {
            if (axes == null) axes = new Shape();

            return new Operator("Dropout")
                .SetParam("p", p)
                .SetParam("mode", MxUtil.EnumToString<DropoutMode>(mode, DropoutModeConvert))
                .SetParam("axes", axes)
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray FullyConnected(NDArray data, NDArray weight, NDArray bias, int num_hidden,
            bool no_bias = false, bool flatten = true)
        {
            //return new Operator("FullyConnected")
            //.SetParam("num_hidden", num_hidden)
            //.SetParam("no_bias", no_bias)
            //.SetParam("flatten", flatten)
            //.SetInput("data", data)
            //.SetInput("weight", weight)
            //.SetInput("bias", bias)
            //.Invoke();

            return new Operator("FullyConnected").Set(data, weight, bias, num_hidden, no_bias, flatten).Invoke();
        }

        public static NDArray LayerNorm(NDArray data, NDArray gamma, NDArray beta, int axis = -1, float eps = 1e-05f,
            bool output_mean_var = false)
        {
            return new Operator("LayerNorm")
                .SetParam("axis", axis)
                .SetParam("eps", eps)
                .SetParam("output_mean_var", output_mean_var)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .Invoke();
        }

        public static NDArray LRN(NDArray data, uint nsize, float alpha = 0.0001f, float beta = 0.75f, float knorm = 2f)
        {
            return new Operator("LRN")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetParam("knorm", knorm)
                .SetParam("nsize", nsize)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Pooling(NDArray data, Shape kernel = null,
            PoolingType pool_type = PoolingType.Max, bool global_pool = false, bool cudnn_off = false,
            PoolingConvention pooling_convention = PoolingConvention.Valid, Shape stride = null,
            Shape pad = null, int? p_value = null, bool? count_include_pad = null, string layout = null)
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
                .Invoke();
        }

        public static NDArray Softmax(NDArray data, int axis = -1, double? temperature = null, DType dtype = null)
        {
            return new Operator("softmax")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Softmin(NDArray data, int axis = -1, double? temperature = null, DType dtype = null)
        {
            return new Operator("softmin")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LogSoftmax(NDArray data, int axis = -1, double? temperature = null, DType dtype = null)
        {
            return new Operator("log_softmax")
                .SetParam("axis", axis)
                .SetParam("temperature", temperature)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SoftmaxActivation(NDArray data,
            SoftmaxMode mode = SoftmaxMode.Instance)
        {
            return new Operator("SoftmaxActivation")
                .SetParam("mode", MxUtil.EnumToString<SoftmaxMode>(mode, SoftmaxactivationModeConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray UpSampling(NDArrayList data, int scale, UpsamplingSampleType sample_type, int num_args,
            int num_filter = 0, UpsamplingMultiInputMode multi_input_mode = UpsamplingMultiInputMode.Concat,
            ulong workspace = 512)
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
                .Invoke();
        }

        public static NDArray SignsgdUpdate(NDArray weight, NDArray grad, float lr, float wd = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f)
        {
            return new Operator("signsgd_update")
                .SetParam("lr", lr)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .Invoke();
        }

        public static NDArray SignumUpdate(NDArray weight, NDArray grad, NDArray mom, float lr, float momentum = 0f,
            float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, float wd_lh = 0f)
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
                .Invoke();
        }

        public static NDArrayList MultiSgdUpdate(NDArrayList data, float[] lrs, float[] wds, float rescale_grad = 1f,
            float clip_gradient = -1f, int num_weights = 1, NDArrayList outputs = null)
        {
            new Operator("multi_sgd_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .Invoke(outputs);
            return outputs.ToArray();
        }

        public static NDArrayList MultiSgdMomUpdate(NDArrayList data, float[] lrs, float[] wds, float momentum = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1, NDArrayList outputs = null)
        {
            new Operator("multi_sgd_mom_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("momentum", momentum)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .Invoke(outputs);

            return outputs.ToArray();
        }

        public static NDArrayList MultiMpSgdUpdate(NDArrayList data, float[] lrs, float[] wds, float rescale_grad = 1f,
            float clip_gradient = -1f, int num_weights = 1, NDArrayList outputs = null)
        {
            new Operator("multi_mp_sgd_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .Invoke(outputs);

            return outputs.ToArray();
        }

        public static NDArrayList MultiMpSgdMomUpdate(NDArrayList data, float[] lrs, float[] wds, float momentum = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, int num_weights = 1, NDArrayList outputs = null)
        {
            new Operator("multi_mp_sgd_mom_update")
                .SetParam("lrs", lrs)
                .SetParam("wds", wds)
                .SetParam("momentum", momentum)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("num_weights", num_weights)
                .SetInput(data)
                .Invoke(outputs);

            return outputs.ToArray();
        }

        public static NDArray SgdUpdate(NDArray weight, NDArray grad, float lr, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, bool lazy_update = true)
        {
            var output = new NDArray();
            new Operator("sgd_update")
                .SetParam("lr", lr)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetParam("weight", weight)
                .Set("grad", grad)
                .Invoke(weight);

            return weight;
        }

        public static NDArray SgdMomUpdate(NDArray weight, NDArray grad, NDArray mom, float lr, float momentum = 0f,
            float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f, bool lazy_update = true)
        {
            new Operator("sgd_mom_update")
                .SetParam("lr", lr)
                .SetParam("momentum", momentum)
                .SetParam("wd", wd)
                .SetParam("rescale_grad", rescale_grad)
                .SetParam("clip_gradient", clip_gradient)
                .SetParam("lazy_update", lazy_update)
                .SetInput("weight", weight)
                .SetInput("grad", grad)
                .SetInput("mom", mom)
                .Invoke(weight);

            return weight;
        }

        public static NDArray MpSgdUpdate(NDArray weight, NDArray grad, NDArray weight32, float lr, float wd = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, bool lazy_update = true)
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
                .Invoke();
        }

        public static NDArray MpSgdMomUpdate(NDArray weight, NDArray grad, NDArray mom, NDArray weight32, float lr,
            float momentum = 0f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f,
            bool lazy_update = true)
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
                .Invoke();
        }

        public static NDArray FtmlUpdate(NDArray weight, NDArray grad, NDArray d, NDArray v, NDArray z, float lr, int t,
            float beta1 = 0.6f, float beta2 = 0.999f, double epsilon = 1e-08, float wd = 0f, float rescale_grad = 1f,
            float clip_grad = -1f)
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
                .Invoke();
        }

        public static NDArray AdamUpdate(NDArray weight, NDArray grad, NDArray mean, NDArray var, float lr,
            float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-08f, float wd = 0f, float rescale_grad = 1f,
            float clip_gradient = -1f, bool lazy_update = true)
        {
            new Operator("adam_update")
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
                .Invoke(weight);

            return weight;
        }

        public static NDArray RmspropUpdate(NDArray weight, NDArray grad, NDArray n, float lr, float gamma1 = 0.95f,
            float epsilon = 1e-08f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f,
            float clip_weights = -1f)
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
                .Invoke();
        }

        public static NDArray RmspropalexUpdate(NDArray weight, NDArray grad, NDArray n, NDArray g, NDArray delta,
            float lr, float gamma1 = 0.95f, float gamma2 = 0.9f, float epsilon = 1e-08f, float wd = 0f,
            float rescale_grad = 1f, float clip_gradient = -1f, float clip_weights = -1f)
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
                .Invoke();
        }

        public static NDArray FtrlUpdate(NDArray weight, NDArray grad, NDArray z, NDArray n, float lr,
            float lamda1 = 0.01f, float beta = 1f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f)
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
                .Invoke();
        }

        public static NDArray NAGMomUpdate(NDArray weight, NDArray grad, NDArray mom, float lr, float momentum = 0,
            float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f)
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
                .Invoke();
        }

        public static NDArray MPNAGMomUpdate(NDArray weight, NDArray grad, NDArray mom, NDArray weight32, float lr,
            float momentum = 0, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f)
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
                .Invoke();
        }

        public static NDArray SparseAdagradUpdate(NDArray weight, NDArray grad, NDArray history, float lr,
            float epsilon = 1e-07f, float wd = 0f, float rescale_grad = 1f, float clip_gradient = -1f)
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
                .Invoke();
        }

        public static NDArray Pad(NDArray data, PadMode mode, Shape pad_width, double constant_value = 0)
        {
            return new Operator("pad")
                .SetParam("mode", MxUtil.EnumToString<PadMode>(mode, PadModeConvert))
                .SetParam("pad_width", pad_width)
                .SetParam("constant_value", constant_value)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Flatten(NDArray data)
        {
            return new Operator("Flatten")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SampleUniform(NDArray low, NDArray high, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_uniform")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("low", low)
                .SetInput("high", high)
                .Invoke();
        }

        public static NDArray SampleNormal(NDArray mu, NDArray sigma, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_normal")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("mu", mu)
                .SetInput("sigma", sigma)
                .Invoke();
        }

        public static NDArray SampleGamma(NDArray alpha, NDArray beta, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_gamma")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("alpha", alpha)
                .SetInput("beta", beta)
                .Invoke();
        }

        public static NDArray SampleExponential(NDArray lam, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_exponential")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("lam", lam)
                .Invoke();
        }

        public static NDArray SamplePoisson(NDArray lam, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_poisson")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("lam", lam)
                .Invoke();
        }

        public static NDArray SampleNegativeBinomial(NDArray k, NDArray p, Shape shape = null, DType dtype = null)
        {
            return new Operator("_sample_negative_binomial")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("k", k)
                .SetInput("p", p)
                .Invoke();
        }

        public static NDArray SampleGeneralizedNegativeBinomial(NDArray mu, NDArray alpha, Shape shape = null,
            DType dtype = null)
        {
            return new Operator("_sample_generalized_negative_binomial")
                .SetParam("shape", shape)
                .SetParam("dtype", dtype)
                .SetInput("mu", mu)
                .SetInput("alpha", alpha)
                .Invoke();
        }

        public static NDArray SampleMultinomial(NDArray data, Shape shape = null, bool get_prob = false,
            DType dtype = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Int32;

            return new Operator("_sample_multinomial")
                .SetParam("shape", shape)
                .SetParam("get_prob", get_prob)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Shuffle(NDArray data)
        {
            return new Operator("_shuffle")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SampleUniqueZipfian(int range_max, Shape shape = null)
        {
            return new Operator("_sample_unique_zipfian")
                .SetParam("range_max", range_max)
                .SetParam("shape", shape)
                .Invoke();
        }

        public static NDArray LinearRegressionOutput(NDArray data, NDArray label, float grad_scale = 1f)
        {
            return new Operator("LinearRegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .Invoke();
        }

        public static NDArray MAERegressionOutput(NDArray data, NDArray label, float grad_scale = 1f)
        {
            return new Operator("MAERegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .Invoke();
        }

        public static NDArray LogisticRegressionOutput(NDArray data, NDArray label, float grad_scale = 1f)
        {
            return new Operator("LogisticRegressionOutput")
                .SetParam("grad_scale", grad_scale)
                .SetInput("data", data)
                .SetInput("label", label)
                .Invoke();
        }

        public static NDArray RNN(NDArray data, NDArray parameters, NDArray state, NDArray state_cell, uint state_size,
            uint num_layers, RNNMode mode, bool bidirectional = false, float p = 0f, bool state_outputs = false,
            int? projection_size = null, double? lstm_state_clip_min = null, double? lstm_state_clip_max = null,
            bool lstm_state_clip_nan = false)
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
                .Invoke();
        }

        public static NDArrayList SliceChannel(NDArray data, int num_outputs, int axis = 1, bool squeeze_axis = false)
        {
            NDArrayList ret = new NDArrayList();
            new Operator("SliceChannel")
                .SetParam("num_outputs", num_outputs)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetInput("data", data)
                .Invoke(ret);

            return ret;
        }

        public static NDArray SoftmaxOutput(NDArray data, NDArray label, float grad_scale = 1f,
            float ignore_label = -1f, bool multi_output = false, bool use_ignore = false, bool preserve_shape = false,
            SoftmaxoutputNormalization normalization = SoftmaxoutputNormalization.Null, bool out_grad = false,
            float smooth_alpha = 0f)
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
                .Invoke();
        }

        public static NDArray SwapAxis(NDArray data, uint dim1 = 0, uint dim2 = 0)
        {
            return new Operator("SwapAxis")
                .SetParam("dim1", dim1)
                .SetParam("dim2", dim2)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Argmax(NDArray data, int? axis = null, bool keepdims = false)
        {
            return new Operator("argmax")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Argmin(NDArray data, int? axis = null, bool keepdims = false)
        {
            return new Operator("argmin")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray ArgmaxChannel(NDArray data)
        {
            return new Operator("argmax_channel")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Pick(NDArray data, NDArray index, int? axis = -1, bool keepdims = false,
            PickMode mode = PickMode.Clip)
        {
            return new Operator("pick")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("mode", MxUtil.EnumToString<PickMode>(mode, PickModeConvert))
                .SetInput("data", data)
                .SetInput("index", index)
                .Invoke();
        }

        public static NDArray Sum(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sum(NDArray data, int axis, bool keepdims = false, bool exclude = false)
        {
            return Sum(data, new Shape(axis), keepdims, exclude);
        }

        public static NDArray Mean(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("mean")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Mean(NDArray data, int axis, bool keepdims = false, bool exclude = false)
        {
            return Mean(data, new Shape(axis), keepdims, exclude);
        }

        public static NDArray Prod(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("prod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Nansum(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("nansum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Nanprod(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("nanprod")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Max(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("max")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Min(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("min")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BroadcastAxis(NDArray data, Shape axis = null, Shape size = null)
        {
            if (axis == null) axis = new Shape();
            if (size == null) size = new Shape();

            return new Operator("broadcast_axis")
                .SetParam("axis", axis)
                .SetParam("size", size)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BroadcastTo(NDArray data, Shape shape = null)
        {
            if (shape == null) shape = new Shape();

            return new Operator("broadcast_to")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BroadcastLike(NDArray lhs, NDArray rhs, Shape lhs_axes = null, Shape rhs_axes = null)
        {
            return new Operator("broadcast_like")
                .SetParam("lhs_axes", lhs_axes)
                .SetParam("rhs_axes", rhs_axes)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Norm(NDArray data, int ord = 2, Shape axis = null, NormOutDtype? out_dtype = null,
            bool keepdims = false)
        {
            return new Operator("norm")
                .SetParam("ord", ord)
                .SetParam("axis", axis)
                .SetParam("out_dtype", MxUtil.EnumToString(out_dtype, NormOutDtypeConvert))
                .SetParam("keepdims", keepdims)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray CastStorage(NDArray data, StorageStype stype)
        {
            return new Operator("cast_storage")
                .SetParam("stype", MxUtil.EnumToString<StorageStype>(stype, CastStorageStypeConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Where(NDArray condition, NDArray x = null, NDArray y = null)
        {
            return new Operator("where")
                .SetInput("condition", condition)
                .SetInput("x", x)
                .SetInput("y", y)
                .Invoke();
        }

        public static NDArray Diag(NDArray data, int k = 0, int axis1 = 0, int axis2 = 1)
        {
            return new Operator("diag")
                .SetParam("k", k)
                .SetParam("axis1", axis1)
                .SetParam("axis2", axis2)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Dot(NDArray lhs, NDArray rhs, bool transpose_a = false, bool transpose_b = false,
            DotForwardStype? forward_stype = null)
        {
            return new Operator("dot")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("forward_stype", MxUtil.EnumToString(forward_stype, DotForwardStypeConvert))
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BatchDot(NDArray lhs, NDArray rhs, bool transpose_a = false, bool transpose_b = false,
            BatchDotForwardStype? forward_stype = null)
        {
            return new Operator("batch_dot")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("forward_stype", MxUtil.EnumToString(forward_stype, BatchDotForwardStypeConvert))
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastAdd(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastSub(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_sub")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastMul(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_mul")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastDiv(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastMod(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_mod")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastPower(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_power")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastMaximum(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_maximum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastMinimum(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_minimum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastHypot(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_hypot")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastNotEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_not_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastGreater(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_greater")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastGreaterEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_greater_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastLesser(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_lesser")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastLesserEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_lesser_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastLogicalAnd(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_logical_and")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastLogicalOr(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_logical_or")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray BroadcastLogicalXor(NDArray lhs, NDArray rhs)
        {
            return new Operator("broadcast_logical_xor")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ElemwiseAdd(NDArray lhs, NDArray rhs)
        {
            return new Operator("elemwise_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray GradAdd(NDArray lhs, NDArray rhs)
        {
            return new Operator("_grad_add")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ElemwiseSub(NDArray lhs, NDArray rhs)
        {
            return new Operator("elemwise_sub")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ElemwiseMul(NDArray lhs, NDArray rhs)
        {
            return new Operator("elemwise_mul")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ElemwiseDiv(NDArray lhs, NDArray rhs)
        {
            return new Operator("elemwise_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Mod(NDArray lhs, NDArray rhs)
        {
            return new Operator("_mod")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Power(NDArray lhs, NDArray rhs)
        {
            return new Operator("_power")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Maximum(NDArray lhs, NDArray rhs)
        {
            return new Operator("_maximum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Minimum(NDArray lhs, NDArray rhs)
        {
            return new Operator("_minimum")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Hypot(NDArray lhs, NDArray rhs)
        {
            return new Operator("_hypot")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Equal(NDArray lhs, NDArray rhs)
        {
            return new Operator("_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray NotEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("_not_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Greater(NDArray lhs, NDArray rhs)
        {
            return new Operator("_greater")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray GreaterEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("_greater_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray Lesser(NDArray lhs, NDArray rhs)
        {
            return new Operator("_lesser")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray LesserEqual(NDArray lhs, NDArray rhs)
        {
            return new Operator("_lesser_equal")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray LogicalAnd(NDArray lhs, NDArray rhs)
        {
            return new Operator("_logical_and")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray LogicalOr(NDArray lhs, NDArray rhs)
        {
            return new Operator("_logical_or")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray LogicalXor(NDArray lhs, NDArray rhs)
        {
            return new Operator("_logical_xor")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray PlusScalar(NDArray data, float scalar)
        {
            return new Operator("_plus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MinusScalar(NDArray data, float scalar)
        {
            return new Operator("_minus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray RminusScalar(NDArray data, float scalar)
        {
            return new Operator("_rminus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MulScalar(NDArray data, float scalar)
        {
            return new Operator("_mul_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray DivScalar(NDArray data, float scalar)
        {
            return new Operator("_div_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray RdivScalar(NDArray data, float scalar)
        {
            return new Operator("_rdiv_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray ModScalar(NDArray data, float scalar)
        {
            return new Operator("_mod_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray RmodScalar(NDArray data, float scalar)
        {
            return new Operator("_rmod_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MaximumScalar(NDArray data, float scalar)
        {
            return new Operator("_maximum_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MinimumScalar(NDArray data, float scalar)
        {
            return new Operator("_minimum_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray PowerScalar(NDArray data, float scalar)
        {
            return new Operator("_power_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray RpowerScalar(NDArray data, float scalar)
        {
            return new Operator("_rpower_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray HypotScalar(NDArray data, float scalar)
        {
            return new Operator("_hypot_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SmoothL1(NDArray data, float scalar)
        {
            return new Operator("smooth_l1")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray EqualScalar(NDArray data, float scalar)
        {
            return new Operator("_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray NotEqualScalar(NDArray data, float scalar)
        {
            return new Operator("_not_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray GreaterScalar(NDArray data, float scalar)
        {
            return new Operator("_greater_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray GreaterEqualScalar(NDArray data, float scalar)
        {
            return new Operator("_greater_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LesserScalar(NDArray data, float scalar)
        {
            return new Operator("_lesser_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LesserEqualScalar(NDArray data, float scalar)
        {
            return new Operator("_lesser_equal_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LogicalAndScalar(NDArray data, float scalar)
        {
            return new Operator("_logical_and_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LogicalOrScalar(NDArray data, float scalar)
        {
            return new Operator("_logical_or_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LogicalXorScalar(NDArray data, float scalar)
        {
            return new Operator("_logical_xor_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray ScatterElemwiseDiv(NDArray lhs, NDArray rhs)
        {
            return new Operator("_scatter_elemwise_div")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ScatterPlusScalar(NDArray data, float scalar)
        {
            return new Operator("_scatter_plus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray ScatterMinusScalar(NDArray data, float scalar)
        {
            return new Operator("_scatter_minus_scalar")
                .SetParam("scalar", scalar)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray AddN(NDArrayList args)
        {
            return new Operator("add_n")
                .SetInput(args)
                .Invoke();
        }

        public static NDArray Relu(NDArray data)
        {
            return new Operator("relu")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sigmoid(NDArray data)
        {
            return new Operator("sigmoid")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray HardSigmoid(NDArray data, float alpha = 0.2f, float beta = 0.5f)
        {
            return new Operator("hard_sigmoid")
                .SetParam("alpha", alpha)
                .SetParam("beta", beta)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Softsign(NDArray data)
        {
            return new Operator("softsign")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Copy(NDArray data)
        {
            return new Operator("_copy")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BlockGrad(NDArray data)
        {
            return new Operator("BlockGrad")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MakeLoss(NDArray data)
        {
            return new Operator("make_loss")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray IdentityWithAttrLikeRhs(NDArray lhs, NDArray rhs)
        {
            return new Operator("_identity_with_attr_like_rhs")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ReshapeLike(NDArray lhs, NDArray rhs)
        {
            return new Operator("reshape_like")
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray ShapeArray(NDArray data, int? lhs_begin = null, int? lhs_end = null,
            int? rhs_begin = null, int? rhs_end = null)
        {
            return new Operator("shape_array")
                .SetParam("lhs_begin", lhs_begin)
                .SetParam("lhs_end", lhs_end)
                .SetParam("rhs_begin", rhs_begin)
                .SetParam("rhs_end", rhs_end)
                .SetInput("data", data)
                .Invoke();
        }
        public static NDArray SizeArray(NDArray data)
        {
            return new Operator("size_array")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Cast(NDArray data, DType dtype)
        {
            return new Operator("cast")
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Negative(NDArray data)
        {
            return new Operator("negative")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Reciprocal(NDArray data)
        {
            return new Operator("reciprocal")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Abs(NDArray data)
        {
            return new Operator("abs")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sign(NDArray data)
        {
            return new Operator("sign")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Round(NDArray data)
        {
            return new Operator("round")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Rint(NDArray data)
        {
            return new Operator("rint")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Ceil(NDArray data)
        {
            return new Operator("ceil")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Floor(NDArray data)
        {
            return new Operator("floor")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Trunc(NDArray data)
        {
            return new Operator("trunc")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Fix(NDArray data)
        {
            return new Operator("fix")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Square(NDArray data)
        {
            return new Operator("square")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sqrt(NDArray data)
        {
            return new Operator("sqrt")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Rsqrt(NDArray data)
        {
            return new Operator("rsqrt")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Cbrt(NDArray data)
        {
            return new Operator("cbrt")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Erf(NDArray data)
        {
            return new Operator("erf")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Erfinv(NDArray data)
        {
            return new Operator("erfinv")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Rcbrt(NDArray data)
        {
            return new Operator("rcbrt")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Exp(NDArray data)
        {
            return new Operator("exp")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Log(NDArray data)
        {
            return new Operator("log")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Log10(NDArray data)
        {
            return new Operator("log10")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Log2(NDArray data)
        {
            return new Operator("log2")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Log1P(NDArray data)
        {
            return new Operator("log1p")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Expm1(NDArray data)
        {
            return new Operator("expm1")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Gamma(NDArray data)
        {
            return new Operator("gamma")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Gammaln(NDArray data)
        {
            return new Operator("gammaln")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LogicalNot(NDArray data)
        {
            return new Operator("logical_not")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sin(NDArray data)
        {
            return new Operator("sin")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Cos(NDArray data)
        {
            return new Operator("cos")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Tan(NDArray data)
        {
            return new Operator("tan")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arcsin(NDArray data)
        {
            return new Operator("arcsin")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arccos(NDArray data)
        {
            return new Operator("arccos")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arctan(NDArray data)
        {
            return new Operator("arctan")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Degrees(NDArray data)
        {
            return new Operator("degrees")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Radians(NDArray data)
        {
            return new Operator("radians")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sinh(NDArray data)
        {
            return new Operator("sinh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Cosh(NDArray data)
        {
            return new Operator("cosh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Tanh(NDArray data)
        {
            return new Operator("tanh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arcsinh(NDArray data)
        {
            return new Operator("arcsinh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arccosh(NDArray data)
        {
            return new Operator("arccosh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Arctanh(NDArray data)
        {
            return new Operator("arctanh")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Histogram(NDArray data, NDArray bins, int? bin_cnt = null, Tuple<double> range = null)
        {
            return new Operator("_histogram")
                .SetParam("bin_cnt", bin_cnt)
                .SetParam("range", range)
                .SetInput("data", data)
                .SetInput("bins", bins)
                .Invoke();
        }

        public static NDArray Embedding(NDArray data, NDArray weight, int input_dim, int output_dim, DType dtype = null,
            bool sparse_grad = false)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("Embedding")
                .SetParam("input_dim", input_dim)
                .SetParam("output_dim", output_dim)
                .SetParam("dtype", dtype)
                .SetParam("sparse_grad", sparse_grad)
                .SetInput("data", data)
                .SetInput("weight", weight)
                .Invoke();
        }

        public static NDArray Take(NDArray a, NDArray indices, int axis = 0, TakeMode mode = TakeMode.Clip)
        {
            return new Operator("take")
                .SetParam("axis", axis)
                .SetParam("mode", MxUtil.EnumToString<TakeMode>(mode, TakeModeConvert))
                .SetInput("a", a)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray BatchTake(NDArray a, NDArray indices)
        {
            return new Operator("batch_take")
                .SetInput("a", a)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray OneHot(NDArray indices, int depth, double on_value = 1, double off_value = 0,
            DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("one_hot")
                .SetParam("depth", depth)
                .SetParam("on_value", on_value)
                .SetParam("off_value", off_value)
                .SetParam("dtype", dtype)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray GatherNd(NDArray data, NDArray indices)
        {
            return new Operator("gather_nd")
                .SetInput("data", data)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray ScatterNd(NDArray data, NDArray indices, Shape shape)
        {
            return new Operator("scatter_nd")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray ScatterSetNd(NDArray lhs, NDArray rhs, NDArray indices, Shape shape)
        {
            return new Operator("_scatter_set_nd")
                .SetParam("shape", shape)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray ZerosWithoutDtype(Shape shape = null, Context ctx = null, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_zeros_without_dtype")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .Invoke();
        }

        public static NDArray Zeros(Shape shape = null, Context ctx = null, DType dtype = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_zeros")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx.ToString())
                .SetParam("dtype", dtype)
                .Invoke();
        }

        public static NDArray Eye(Tuple<double> N, int M = 0, int k = 0, Context ctx = null, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_eye")
                .SetParam("N", N)
                .SetParam("M", M)
                .SetParam("k", k)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .Invoke();
        }

        public static NDArray Ones(Shape shape = null, Context ctx = null, DType dtype = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_ones")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .Invoke();
        }

        public static NDArray Empty(Shape shape = null, Context ctx = null, DType dtype = null)
        {
            if (shape == null) shape = new Shape();
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_empty")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .Invoke();
        }

        public static NDArray Full(double value, Shape shape = null, Context ctx = null, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("_full")
                .SetParam("shape", shape)
                .SetParam("ctx", ctx)
                .SetParam("dtype", dtype)
                .SetParam("value", value)
                .Invoke();
        }

        public static NDArray Arange(int start, int? stop = null, int step = 1, int repeat = 1,
            bool infer_range = false, Context ctx = null, DType dtype = null)
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
                .Invoke();
        }

        public static NDArray ZerosLike(NDArray data)
        {
            return new Operator("zeros_like")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray OnesLike(NDArray data)
        {
            return new Operator("ones_like")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray LinalgGemm(NDArray A, NDArray B, NDArray C, bool transpose_a = false,
            bool transpose_b = false, double alpha = 1, double beta = 1, int axis = -2)
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
                .Invoke();
        }
        
        public static NDArray LinalgGemm2(NDArray A, NDArray B, bool transpose_a = false, bool transpose_b = false,
            double alpha = 1, int axis = -2)
        {
            return new Operator("_linalg_gemm2")
                .SetParam("transpose_a", transpose_a)
                .SetParam("transpose_b", transpose_b)
                .SetParam("alpha", alpha)
                .SetParam("axis", axis)
                .SetInput("A", A)
                .SetInput("B", B)
                .Invoke();
        }

        public static NDArray LinalgPotrf(NDArray A)
        {
            return new Operator("_linalg_potrf")
                .SetInput("A", A)
                .Invoke();
        }

        public static NDArray LinalgPotri(NDArray A)
        {
            return new Operator("_linalg_potri")
                .SetInput("A", A)
                .Invoke();
        }

        public static NDArray LinalgTrmm(NDArray A, NDArray B, bool transpose = false, bool rightside = false,
            bool lower = true, double alpha = 1)
        {
            return new Operator("_linalg_trmm")
                .SetParam("transpose", transpose)
                .SetParam("rightside", rightside)
                .SetParam("lower", lower)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .SetInput("B", B)
                .Invoke();
        }

        public static NDArray LinalgTrsm(NDArray A, NDArray B, bool transpose = false, bool rightside = false,
            bool lower = true, double alpha = 1)
        {
            return new Operator("_linalg_trsm")
                .SetParam("transpose", transpose)
                .SetParam("rightside", rightside)
                .SetParam("lower", lower)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .SetInput("B", B)
                .Invoke();
        }

        public static NDArray LinalgSumlogdiag(NDArray A)
        {
            return new Operator("_linalg_sumlogdiag")
                .SetInput("A", A)
                .Invoke();
        }

        public static NDArray LinalgSyrk(NDArray A, bool transpose = false, double alpha = 1)
        {
            return new Operator("_linalg_syrk")
                .SetParam("transpose", transpose)
                .SetParam("alpha", alpha)
                .SetInput("A", A)
                .Invoke();
        }

        public static NDArray LinalgGelqf(NDArray A)
        {
            return new Operator("_linalg_gelqf")
                .SetInput("A", A)
                .Invoke();
        }

        public static (NDArray, NDArray) LinalgSyevd(NDArray A)
        {
            var outputs = new NDArrayList();
            new Operator("_linalg_syevd")
                .SetInput("A", A)
                .Invoke(outputs);
            if (outputs.Length == 1)
                return (outputs[0], null);
            return (outputs[0], outputs[1]);
        }

        public static NDArray Reshape(NDArray data, Shape shape = null, bool reverse = false)
        {
            if (shape == null) shape = new Shape();
            new Operator("Reshape")
                .SetParam("shape", shape)
                .SetParam("reverse", reverse)
                .SetInput("data", data)
                .Invoke();

            return data;
        }

        public static NDArray Transpose(NDArray data, Shape axes = null)
        {
            if (axes == null) axes = new Shape();

            return new Operator("transpose")
                .SetParam("axes", axes)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray ExpandDims(NDArray data, int axis)
        {
            return new Operator("expand_dims")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Slice(NDArray data, Shape begin, Shape end, Shape step = null)
        {
            if (step == null) step = new Shape();

            return new Operator("slice")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SliceAssign(NDArray lhs, NDArray rhs, Shape begin, Shape end, Shape step = null)
        {
            if (step == null) step = new Shape();

            return new Operator("_slice_assign")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("lhs", lhs)
                .SetInput("rhs", rhs)
                .Invoke();
        }

        public static NDArray SliceAssignScalar(NDArray data, Shape begin, Shape end, double scalar = 0,
            Shape step = null)
        {
            if (step == null) step = new Shape();

            NDArray output = new NDArray(data.Shape, dtype: data.DataType);

            new Operator("_slice_assign_scalar")
                .SetParam("scalar", scalar)
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", data)
                .Invoke(output);

            return output;
        }

        public static NDArray SliceAxis(NDArray data, int axis, int begin, int? end)
        {
            return new Operator("slice_axis")
                .SetParam("axis", axis)
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SliceLike(NDArray data, NDArray shape_like, Shape axes = null)
        {
            if (axes == null) axes = new Shape();

            return new Operator("slice_like")
                .SetParam("axes", axes)
                .SetInput("data", data)
                .SetInput("shape_like", shape_like)
                .Invoke();
        }

        public static NDArray Clip(NDArray data, float a_min, float a_max)
        {
            return new Operator("clip")
                .SetParam("a_min", a_min)
                .SetParam("a_max", a_max)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Repeat(NDArray data, int repeats, int? axis = null)
        {
            return new Operator("repeat")
                .SetParam("repeats", repeats)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Tile(NDArray data, Shape reps)
        {
            return new Operator("tile")
                .SetParam("reps", reps)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Reverse(NDArray data, Shape axis)
        {
            return new Operator("reverse")
                .SetParam("axis", axis)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Flip(NDArray data, int axis)
        {
            return new Operator("reverse")
                .SetParam("axis", new Shape(axis))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Stack(NDArrayList data, int num_args, int axis = 0)
        {
            return new Operator("stack")
                .SetParam("axis", axis)
                .SetParam("num_args", num_args)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray Squeeze(NDArrayList data, Shape axis = null)
        {
            return new Operator("squeeze")
                .SetParam("axis", axis)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray DepthToSpace(NDArray data, int block_size)
        {
            return new Operator("depth_to_space")
                .SetParam("block_size", block_size)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SpaceToDepth(NDArray data, int block_size)
        {
            return new Operator("space_to_depth")
                .SetParam("block_size", block_size)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArrayList SplitV2(NDArray data, Shape indices, int axis = 1, bool squeeze_axis = false,
            int sections = 0)
        {
            var outputs = new NDArrayList();
            new Operator("_split_v2")
                .SetParam("indices", indices)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetParam("sections", sections)
                .SetInput("data", data)
                .Invoke(outputs);

            return outputs.ToArray();
        }

        public static NDArrayList Split(NDArray data, int num_outputs, int axis = 1, bool squeeze_axis = false)
        {
            var outputs = new NDArrayList();
            new Operator("split")
                .SetParam("num_outputs", num_outputs)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .SetInput("data", data)
                .Invoke(outputs);
            return outputs.ToArray();
        }

        public static NDArray Topk(NDArray data, int? axis = -1, int k = 1, TopkRetTyp ret_typ = TopkRetTyp.Indices,
            bool is_ascend = false, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("topk")
                .SetParam("axis", axis)
                .SetParam("k", k)
                .SetParam("ret_typ", MxUtil.EnumToString<TopkRetTyp>(ret_typ, TopkRetTypConvert))
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Sort(NDArray data, int? axis = -1, bool is_ascend = true)
        {
            return new Operator("sort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray Argsort(NDArray data, int? axis = -1, bool is_ascend = true, DType dtype = null)
        {
            if (dtype == null) dtype = DType.Float32;

            return new Operator("argsort")
                .SetParam("axis", axis)
                .SetParam("is_ascend", is_ascend)
                .SetParam("dtype", dtype)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray RavelMultiIndex(NDArray data, Shape shape = null)
        {
            return new Operator("_ravel_multi_index")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray UnravelIndex(NDArray data, Shape shape = null)
        {
            return new Operator("_unravel_index")
                .SetParam("shape", shape)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray SparseRetain(NDArray data, NDArray indices)
        {
            return new Operator("_sparse_retain")
                .SetInput("data", data)
                .SetInput("indices", indices)
                .Invoke();
        }

        public static NDArray SquareSum(NDArray data, Shape axis = null, bool keepdims = false, bool exclude = false)
        {
            return new Operator("_square_sum")
                .SetParam("axis", axis)
                .SetParam("keepdims", keepdims)
                .SetParam("exclude", exclude)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray BilinearSampler(NDArray data, NDArray grid, bool? cudnn_off = null)
        {
            return new Operator("BilinearSampler")
                .SetParam("cudnn_off", cudnn_off)
                .SetInput("data", data)
                .SetInput("grid", grid)
                .Invoke();
        }

        public static NDArray ConvolutionV1(NDArray data, NDArray weight, NDArray bias, Shape kernel, uint num_filter,
            Shape stride = null, Shape dilate = null, Shape pad = null, uint num_group = 1, ulong workspace = 1024,
            bool no_bias = false, ConvolutionV1CudnnTune? cudnn_tune = null, bool cudnn_off = false,
            ConvolutionV1Layout? layout = null)
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
                .Invoke();
        }

        public static NDArray Correlation(NDArray data1, NDArray data2, uint kernel_size = 1, uint max_displacement = 1,
            uint stride1 = 1, uint stride2 = 1, uint pad_size = 0, bool is_multiply = true)
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
                .Invoke();
        }

        public static NDArray Crop(NDArrayList data, int num_args, Shape offset = null, Shape h_w = null,
            bool center_crop = false)
        {
            if (offset == null) offset = new Shape();
            if (h_w == null) h_w = new Shape();

            return new Operator("Crop")
                .SetParam("data", data)
                .SetParam("num_args", num_args)
                .SetParam("offset", offset)
                .SetParam("h_w", h_w)
                .SetParam("center_crop", center_crop)
                .Invoke();
        }

        public static NDArray CrossDeviceCopy()
        {
            return new Operator("_CrossDeviceCopy")
                .Invoke();
        }

        public static NDArray Native(NDArrayList data, IntPtr info, bool need_top_grad = true)
        {
            return new Operator("_Native")
                .SetParam("info", info)
                .SetParam("need_top_grad", need_top_grad)
                .SetInput(data)
                .Invoke();
        }

        public static NDArray GridGenerator(NDArray data, GridgeneratorTransformType transform_type,
            Shape target_shape = null)
        {
            if (target_shape == null) target_shape = new Shape();

            return new Operator("GridGenerator")
                .SetParam("transform_type",
                    MxUtil.EnumToString<GridgeneratorTransformType>(transform_type, GridgeneratorTransformTypeConvert))
                .SetParam("target_shape", target_shape)
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray InstanceNorm(NDArray data, NDArray gamma, NDArray beta, float eps = 0.001f)
        {
            return new Operator("InstanceNorm")
                .SetParam("eps", eps)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .Invoke();
        }

        public static NDArray L2Normalization(NDArray data, float eps = 1e-10f,
            L2normalizationMode mode = L2normalizationMode.Instance)
        {
            return new Operator("L2Normalization")
                .SetParam("eps", eps)
                .SetParam("mode", MxUtil.EnumToString<L2normalizationMode>(mode, L2normalizationModeConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray MakeLoss(NDArray data, float grad_scale = 1f, float valid_thresh = 0f,
            MakelossNormalization normalization = MakelossNormalization.Null)
        {
            return new Operator("MakeLoss")
                .SetParam("grad_scale", grad_scale)
                .SetParam("valid_thresh", valid_thresh)
                .SetParam("normalization",
                    MxUtil.EnumToString<MakelossNormalization>(normalization, MakelossNormalizationConvert))
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray PoolingV1(NDArray data, Shape kernel = null,
            PoolingV1PoolType pool_type = PoolingV1PoolType.Max, bool global_pool = false,
            PoolingV1PoolingConvention pooling_convention = PoolingV1PoolingConvention.Valid, Shape stride = null,
            Shape pad = null)
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
                .Invoke();
        }

        public static NDArray ROIPooling(NDArray data, NDArray rois, Shape pooled_size, float spatial_scale)
        {
            return new Operator("ROIPooling")
                .SetParam("pooled_size", pooled_size)
                .SetParam("spatial_scale", spatial_scale)
                .SetInput("data", data)
                .SetInput("rois", rois)
                .Invoke();
        }

        public static NDArray SequenceLast(NDArray data, NDArray sequence_length, bool use_sequence_length = false,
            int axis = 0)
        {
            return new Operator("SequenceLast")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .Invoke();
        }

        public static NDArray SequenceMask(NDArray data, NDArray sequence_length, bool use_sequence_length = false,
            float value = 0f, int axis = 0)
        {
            return new Operator("SequenceMask")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("value", value)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .Invoke();
        }

        public static NDArray SequenceReverse(NDArray data, NDArray sequence_length, bool use_sequence_length = false,
            int axis = 0)
        {
            return new Operator("SequenceReverse")
                .SetParam("use_sequence_length", use_sequence_length)
                .SetParam("axis", axis)
                .SetInput("data", data)
                .SetInput("sequence_length", sequence_length)
                .Invoke();
        }

        public static NDArray SpatialTransformer(NDArray data, NDArray loc,
            SpatialtransformerTransformType transform_type, SpatialtransformerSamplerType sampler_type,
            Shape target_shape = null, bool? cudnn_off = null)
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
                .Invoke();
        }

        public static NDArray SVMOutput(NDArray data, NDArray label, float margin = 1f,
            float regularization_coefficient = 1f, bool use_linear = false)
        {
            return new Operator("SVMOutput")
                .SetParam("margin", margin)
                .SetParam("regularization_coefficient", regularization_coefficient)
                .SetParam("use_linear", use_linear)
                .SetInput("data", data)
                .SetInput("label", label)
                .Invoke();
        }

        public static NDArray OnehotEncode(NDArray lhs, NDArray rhs)
        {
            return new Operator("_onehot_encode")
                .SetParam("lhs", lhs)
                .SetParam("rhs", rhs)
                .Invoke();
        }

        public static NDArray FillElement0Index(NDArray lhs, NDArray mhs, NDArray rhs)
        {
            return new Operator("fill_element_0index")
                .SetParam("lhs", lhs)
                .SetParam("mhs", mhs)
                .SetParam("rhs", rhs)
                .Invoke();
        }

        public static NDArray Imdecode(NDArray mean, int index, int x0, int y0, int x1, int y1, int c, int size)
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
                .Invoke();
        }

        public static NDArray Linspace(float start, float stop, int num, bool endpoint = true, Context ctx = null, DType dtype = null)
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
                .Invoke();
        }

        public static NDArray StopGradient(NDArray data)
        {
            return new Operator("stop_gradient")
                .SetInput("data", data)
                .Invoke();
        }

        public static NDArray GroupNorm(NDArray data, NDArray gamma, NDArray beta, float eps = 0.001f)
        {
            return new Operator("GroupNorm")
                .SetParam("eps", eps)
                .SetInput("data", data)
                .SetInput("gamma", gamma)
                .SetInput("beta", beta)
                .Invoke();
        }

        public static NDArrayList FlattenList(List<NDArrayList> nested_list)
        {
            NDArrayList result = new NDArrayList();
            foreach (var item in nested_list)
            {
                result.Add(item);
            }

            return result;
        }

        public static NDArray GetRescaleGrad(double rescale_grad, Context ctx = null)
        {
            return nd.Full(shape: new Shape(1), value: rescale_grad, ctx: ctx);
        }

        public static NDArray GetRescaleGrad(NDArray rescale_grad, Context ctx = null)
        {
            return rescale_grad.AsInContext(ctx);
        }

        public static NDArrayList MultiSumSq(NDArrayList arrays, int num_arrays)
        {
            NDArrayList ret = new NDArrayList();
            new Operator("multi_sum_sq")
                .SetInput(arrays).SetParam("num_arrays", num_arrays)
                .Invoke(ret);

            return ret;
        }

        public static NDArrayList ResetArrays(NDArrayList arrays)
        {
            NDArrayList ret = new NDArrayList();
            new Operator("reset_arrays")
                .SetInput(arrays).SetParam("num_arrays", arrays.Length)
                .Invoke(ret);

            return ret;
        }
    }
}