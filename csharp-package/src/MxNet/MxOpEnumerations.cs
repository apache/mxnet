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
namespace MxNet
{
    /// <summary>
    ///     <para>The input box encoding type. </para>
    ///     <para>
    ///         "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y,
    ///         width, height].
    ///     </para>
    /// </summary>
    public enum ContribBoxNmsInFormat
    {
        Center,
        Corner
    }

    /// <summary>
    ///     <para>The output box encoding type. </para>
    ///     <para>
    ///         "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y,
    ///         width, height].
    ///     </para>
    /// </summary>
    public enum ContribBoxNmsOutFormat
    {
        Center,
        Corner
    }

    /// <summary>
    ///     <para>The box encoding type. </para>
    ///     <para>
    ///         "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y,
    ///         width, height].
    ///     </para>
    /// </summary>
    public enum ContribBoxIouFormat
    {
        Center,
        Corner
    }

    /// <summary>
    ///     <para>Activation function to be applied.</para>
    /// </summary>
    public enum ReluActType
    {
        Elu,
        Gelu,
        Leaky,
        Prelu,
        Rrelu,
        Selu
    }

    /// <summary>
    ///     <para>Activation function to be applied.</para>
    /// </summary>
    public enum ActivationType
    {
        Relu,
        Sigmoid,
        Softrelu,
        Softsign,
        Tanh
    }

    /// <summary>
    ///     <para>Whether to pick convolution algo by running performance test.</para>
    /// </summary>
    public enum ConvolutionCudnnTune
    {
        Fastest,
        LimitedWorkspace,
        Off
    }

    /// <summary>
    ///     <para>Set layout for input, output and weight. Empty for</para>
    ///     <para>    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are only supported on GPU.</para>
    /// </summary>
    public enum ConvolutionLayout
    {
        NCDHW,
        NCHW,
        NCW,
        NDHWC,
        NHWC
    }

    /// <summary>
    ///     <para>
    ///         Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for
    ///         tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last",
    ///         last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the
    ///         vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.
    ///     </para>
    /// </summary>
    public enum CtclossBlankLabel
    {
        First,
        Last
    }

    /// <summary>
    ///     <para>Whether to pick convolution algorithm by running performance test.</para>
    /// </summary>
    public enum DeconvolutionCudnnTune
    {
        Fastest,
        LimitedWorkspace,
        Off
    }

    /// <summary>
    ///     <para>
    ///         Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for
    ///         3d.NHWC and NDHWC are only supported on GPU.
    ///     </para>
    /// </summary>
    public enum DeconvolutionLayout
    {
        NCDHW,
        NCHW,
        NCW,
        NDHWC,
        NHWC
    }

    /// <summary>
    ///     <para>Whether to only turn on dropout during training or to also turn on for inference.</para>
    /// </summary>
    public enum DropoutMode
    {
        Always,
        Training
    }

    /// <summary>
    ///     <para>Pooling type to be applied.</para>
    /// </summary>
    public enum PoolingType
    {
        Avg,
        Lp,
        Max,
        Sum
    }

    /// <summary>
    ///     <para>Pooling convention to be applied.</para>
    /// </summary>
    public enum PoolingConvention
    {
        Full,
        Same,
        Valid
    }

    /// <summary>
    ///     <para>Set layout for input and output. Empty for</para>
    ///     <para>    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.</para>
    /// </summary>
    public enum PoolingLayout
    {
        NCDHW,
        NCHW,
        NCW,
        NDHWC,
        NHWC,
        NWC
    }

    /// <summary>
    ///     <para>
    ///         Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set
    ///         to ``channel``, It computes cross channel softmax for each position of each instance.
    ///     </para>
    /// </summary>
    public enum SoftmaxMode
    {
        Channel,
        Instance
    }

    /// <summary>
    ///     <para>upsampling method</para>
    /// </summary>
    public enum UpsamplingSampleType
    {
        Bilinear,
        Nearest
    }

    /// <summary>
    ///     <para>
    ///         How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum
    ///         means add all images together, only available for nearest neighbor upsampling.
    ///     </para>
    /// </summary>
    public enum UpsamplingMultiInputMode
    {
        Concat,
        Sum
    }

    /// <summary>
    ///     <para>
    ///         Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input
    ///         array "reflect" pads by reflecting values with respect to the edges.
    ///     </para>
    /// </summary>
    public enum PadMode
    {
        Constant,
        Edge,
        Reflect
    }

    /// <summary>
    ///     <para>Output data type.</para>
    /// </summary>
    public enum ContribDequantizeOutType
    {
        Float32
    }

    /// <summary>
    ///     <para>Output data type.</para>
    /// </summary>
    public enum ContribQuantizeOutType
    {
        Int8,
        Uint8
    }

    /// <summary>
    ///     <para>
    ///         Output data type. `auto` can be specified to automatically determine output type according to
    ///         min_calib_range.
    ///     </para>
    /// </summary>
    public enum ContribQuantizeV2OutType
    {
        Auto,
        Int8,
        Uint8
    }

    /// <summary>
    ///     <para>Activation function to be applied.</para>
    /// </summary>
    public enum ContribQuantizedActActType
    {
        Relu,
        Sigmoid,
        Softrelu,
        Softsign,
        Tanh
    }

    /// <summary>
    ///     <para>Whether to pick convolution algo by running performance test.</para>
    /// </summary>
    public enum ContribQuantizedConvCudnnTune
    {
        Fastest,
        LimitedWorkspace,
        Off
    }

    /// <summary>
    ///     <para>Set layout for input, output and weight. Empty for</para>
    ///     <para>    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are only supported on GPU.</para>
    /// </summary>
    public enum ContribQuantizedConvLayout
    {
        NCDHW,
        NCHW,
        NCW,
        NDHWC,
        NHWC
    }

    /// <summary>
    ///     <para>Pooling type to be applied.</para>
    /// </summary>
    public enum ContribQuantizedPoolingPoolType
    {
        Avg,
        Lp,
        Max,
        Sum
    }

    /// <summary>
    ///     <para>Pooling convention to be applied.</para>
    /// </summary>
    public enum ContribQuantizedPoolingPoolingConvention
    {
        Full,
        Same,
        Valid
    }

    /// <summary>
    ///     <para>Set layout for input and output. Empty for</para>
    ///     <para>    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.</para>
    /// </summary>
    public enum ContribQuantizedPoolingLayout
    {
        NCDHW,
        NCHW,
        NCW,
        NDHWC,
        NHWC,
        NWC
    }

    /// <summary>
    ///     <para>
    ///         Output data type. `auto` can be specified to automatically determine output type according to
    ///         min_calib_range.
    ///     </para>
    /// </summary>
    public enum ContribRequantizeOutType
    {
        Auto,
        Int8,
        Uint8
    }

    /// <summary>
    ///     <para>the type of RNN to compute</para>
    /// </summary>
    public enum RNNMode
    {
        Gru,
        Lstm,
        RnnRelu,
        RnnTanh
    }

    /// <summary>
    ///     <para>Normalizes the gradient.</para>
    /// </summary>
    public enum SoftmaxoutputNormalization
    {
        Batch,
        Null,
        Valid
    }

    /// <summary>
    ///     <para>
    ///         Specify how out-of-bound indices behave. Default is "clip". "clip" means clip to the range. So, if all
    ///         indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.
    ///         "wrap" means to wrap around.
    ///     </para>
    /// </summary>
    public enum PickMode
    {
        Clip,
        Wrap
    }

    /// <summary>
    ///     <para>The data type of the output.</para>
    /// </summary>
    public enum NormOutDtype
    {
        Float16,
        Float32,
        Float64,
        Int32,
        Int64,
        Int8
    }

    /// <summary>
    ///     <para>Output storage type.</para>
    /// </summary>
    public enum StorageStype
    {
        Default = 0,
        RowSparse,
        Csr,
        Undefined
    }

    /// <summary>
    ///     <para>
    ///         The desired storage type of the forward output given by user, if thecombination of input storage types and
    ///         this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce
    ///         an output of the desired storage type.
    ///     </para>
    /// </summary>
    public enum DotForwardStype
    {
        Default = 0,
        RowSparse,
        Csr
    }

    /// <summary>
    ///     <para>
    ///         The desired storage type of the forward output given by user, if thecombination of input storage types and
    ///         this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce
    ///         an output of the desired storage type.
    ///     </para>
    /// </summary>
    public enum BatchDotForwardStype
    {
        Default = 0,
        RowSparse,
        Csr
    }

    /// <summary>
    ///     <para>
    ///         Specify how out-of-bound indices bahave. Default is "clip". "clip" means clip to the range. So, if all
    ///         indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.
    ///         "wrap" means to wrap around.  "raise" means to raise an error, not supported yet.
    ///     </para>
    /// </summary>
    public enum TakeMode
    {
        Clip,
        Raise,
        Wrap
    }

    /// <summary>
    ///     <para>The return type.</para>
    ///     <para>
    ///         "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask"
    ///         means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of
    ///         both values and indices of top k elements.
    ///     </para>
    /// </summary>
    public enum TopkRetTyp
    {
        Both,
        Indices,
        Mask,
        Value
    }

    /// <summary>
    ///     <para>Set layout for input, output and weight. Empty for</para>
    ///     <para>    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.</para>
    /// </summary>
    public enum ContribDeformableconvolutionLayout
    {
        NCDHW,
        NCHW,
        NCW
    }

    /// <summary>
    ///     <para>Whether to pick convolution algo by running performance test.</para>
    ///     <para>    Leads to higher startup time but may give faster speed. Options are:</para>
    ///     <para>    'off': no tuning</para>
    ///     <para>    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.</para>
    ///     <para>    'fastest': pick the fastest algorithm and ignore workspace limit.</para>
    ///     <para>    If set to None (default), behavior is determined by environment</para>
    ///     <para>    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,</para>
    ///     <para>    1 for limited workspace (default), 2 for fastest.</para>
    /// </summary>
    public enum ConvolutionV1CudnnTune
    {
        Fastest,
        LimitedWorkspace,
        Off
    }

    /// <summary>
    ///     <para>Set layout for input, output and weight. Empty for</para>
    ///     <para>    default layout: NCHW for 2d and NCDHW for 3d.</para>
    /// </summary>
    public enum ConvolutionV1Layout
    {
        NCDHW,
        NCHW,
        NDHWC,
        NHWC
    }

    /// <summary>
    ///     <para>
    ///         The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For
    ///         `warp`, input data should be an optical flow of size (batch, 2, h, w).
    ///     </para>
    /// </summary>
    public enum GridgeneratorTransformType
    {
        Affine,
        Warp
    }

    /// <summary>
    ///     <para>Specify the dimension along which to compute L2 norm.</para>
    /// </summary>
    public enum L2normalizationMode
    {
        Channel,
        Instance,
        Spatial
    }

    /// <summary>
    ///     <para>
    ///         If this is set to null, the output gradient will not be normalized. If this is set to batch, the output
    ///         gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the
    ///         number of valid input elements.
    ///     </para>
    /// </summary>
    public enum MakelossNormalization
    {
        Batch,
        Null,
        Valid
    }

    /// <summary>
    ///     <para>Pooling type to be applied.</para>
    /// </summary>
    public enum PoolingV1PoolType
    {
        Avg,
        Max,
        Sum
    }

    /// <summary>
    ///     <para>Pooling convention to be applied.</para>
    /// </summary>
    public enum PoolingV1PoolingConvention
    {
        Full,
        Valid
    }

    /// <summary>
    ///     <para>transformation type</para>
    /// </summary>
    public enum SpatialtransformerTransformType
    {
        Affine
    }

    /// <summary>
    ///     <para>sampling type</para>
    /// </summary>
    public enum SpatialtransformerSamplerType
    {
        Bilinear
    }
}