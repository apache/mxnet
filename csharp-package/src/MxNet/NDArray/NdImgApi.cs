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

namespace MxNet
{
    [Obsolete("Legacy API after MxNet v2, will be deprecated in v3", false)]
    public class NDImgApi
    {
        /// <summary>
        ///     <para>Crop an image NDArray of shape (H x W x C) or (N x H x W x C) </para>
        ///     <para>to the given size.</para>
        ///     <para>Example:</para>
        ///     <para>    .. code-block:: python</para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        mx.nd.image.crop(image, 1, 1, 2, 2)</para>
        ///     <para>            [[[144  34   4]</para>
        ///     <para>              [ 82 157  38]]</para>
        ///     <para> </para>
        ///     <para>             [[156 111 230]</para>
        ///     <para>              [177  25  15]]]</para>
        ///     <para>
        ///         <NDArray 2 x2x3 @ cpu(0)>
        ///     </para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        mx.nd.image.crop(image, 1, 1, 2, 2)            </para>
        ///     <para>            [[[[ 35 198  50]</para>
        ///     <para>               [242  94 168]]</para>
        ///     <para> </para>
        ///     <para>              [[223 119 129]</para>
        ///     <para>               [249  14 154]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>              [[[137 215 106]</para>
        ///     <para>                [ 79 174 133]]</para>
        ///     <para> </para>
        ///     <para>               [[116 142 109]</para>
        ///     <para>                [ 35 239  50]]]]</para>
        ///     <para>
        ///         <NDArray 2 x2x2x3 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\crop.cc:L65</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="x">Left boundary of the cropping area.</param>
        /// <param name="y">Top boundary of the cropping area.</param>
        /// <param name="width">Width of the cropping area.</param>
        /// <param name="height">Height of the cropping area.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Crop(NDArray data, int x, int y, int width, int height)
        {
            return new Operator("_image_crop")
                .SetParam("x", x)
                .SetParam("y", y)
                .SetParam("width", width)
                .SetParam("height", height)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Converts an image NDArray of shape (H x W x C) or (N x H x W x C) </para>
        ///     <para>with values in the range [0, 255] to a tensor NDArray of shape (C x H x W) or (N x C x H x W)</para>
        ///     <para>with values in the range [0, 1)</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para>    .. code-block:: python</para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        to_tensor(image)</para>
        ///     <para>            [[[ 0.85490197  0.72156864]</para>
        ///     <para>              [ 0.09019608  0.74117649]</para>
        ///     <para>              [ 0.61960787  0.92941177]</para>
        ///     <para>              [ 0.96470588  0.1882353 ]]</para>
        ///     <para>             [[ 0.6156863   0.73725492]</para>
        ///     <para>              [ 0.46666667  0.98039216]</para>
        ///     <para>              [ 0.44705883  0.45490196]</para>
        ///     <para>              [ 0.01960784  0.8509804 ]]</para>
        ///     <para>             [[ 0.39607844  0.03137255]</para>
        ///     <para>              [ 0.72156864  0.52941179]</para>
        ///     <para>              [ 0.16470589  0.7647059 ]</para>
        ///     <para>              [ 0.05490196  0.70588237]]]</para>
        ///     <para>
        ///         <NDArray 3 x4x2 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        to_tensor(image)</para>
        ///     <para>            [[[[0.11764706 0.5803922 ]</para>
        ///     <para>               [0.9411765  0.10588235]</para>
        ///     <para>               [0.2627451  0.73333335]</para>
        ///     <para>               [0.5647059  0.32156864]]</para>
        ///     <para>              [[0.7176471  0.14117648]</para>
        ///     <para>               [0.75686276 0.4117647 ]</para>
        ///     <para>               [0.18431373 0.45490196]</para>
        ///     <para>               [0.13333334 0.6156863 ]]</para>
        ///     <para>              [[0.6392157  0.5372549 ]</para>
        ///     <para>               [0.52156866 0.47058824]</para>
        ///     <para>               [0.77254903 0.21568628]</para>
        ///     <para>               [0.01568628 0.14901961]]]</para>
        ///     <para>             [[[0.6117647  0.38431373]</para>
        ///     <para>               [0.6784314  0.6117647 ]</para>
        ///     <para>               [0.69411767 0.96862745]</para>
        ///     <para>               [0.67058825 0.35686275]]</para>
        ///     <para>              [[0.21960784 0.9411765 ]</para>
        ///     <para>               [0.44705883 0.43529412]</para>
        ///     <para>               [0.09803922 0.6666667 ]</para>
        ///     <para>               [0.16862746 0.1254902 ]]</para>
        ///     <para>              [[0.6156863  0.9019608 ]</para>
        ///     <para>               [0.35686275 0.9019608 ]</para>
        ///     <para>               [0.05882353 0.6509804 ]</para>
        ///     <para>               [0.20784314 0.7490196 ]]]]</para>
        ///     <para>
        ///         <NDArray 2 x3x4x2 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L91</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <returns>returns new symbol</returns>
        public NDArray ToTensor(NDArray data)
        {
            return new Operator("_image_to_tensor")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and</para>
        ///     <para>    standard deviation.</para>
        ///     <para> </para>
        ///     <para>    Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,</para>
        ///     <para>    this transform normalizes each channel of the input tensor with:</para>
        ///     <para> </para>
        ///     <para>.. math::</para>
        ///     <para> </para>
        ///     <para>        output[i] = (input[i] - m\ :sub:`i`\ ) / s\ :sub:`i`</para>
        ///     <para> </para>
        ///     <para>    If mean or std is scalar, the same value will be applied to all channels.</para>
        ///     <para> </para>
        ///     <para>    Default value for mean is 0.0 and stand deviation is 1.0.</para>
        ///     <para> </para>
        ///     <para>Example:</para>
        ///     <para> </para>
        ///     <para>    .. code-block:: python</para>
        ///     <para>        image = mx.nd.random.uniform(0, 1, (3, 4, 2))</para>
        ///     <para>        normalize(image, mean=(0, 1, 2), std=(3, 2, 1))</para>
        ///     <para>            [[[ 0.18293785  0.19761486]</para>
        ///     <para>              [ 0.23839645  0.28142193]</para>
        ///     <para>              [ 0.20092112  0.28598186]</para>
        ///     <para>              [ 0.18162774  0.28241724]]</para>
        ///     <para>             [[-0.2881726  -0.18821815]</para>
        ///     <para>              [-0.17705294 -0.30780914]</para>
        ///     <para>              [-0.2812064  -0.3512327 ]</para>
        ///     <para>              [-0.05411351 -0.4716435 ]]</para>
        ///     <para>             [[-1.0363373  -1.7273437 ]</para>
        ///     <para>              [-1.6165586  -1.5223348 ]</para>
        ///     <para>              [-1.208275   -1.1878313 ]</para>
        ///     <para>              [-1.4711051  -1.5200229 ]]]</para>
        ///     <para>
        ///         <NDArray 3 x4x2 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para>        image = mx.nd.random.uniform(0, 1, (2, 3, 4, 2))</para>
        ///     <para>        normalize(image, mean=(0, 1, 2), std=(3, 2, 1))</para>
        ///     <para>            [[[[ 0.18934818  0.13092826]</para>
        ///     <para>               [ 0.3085322   0.27869293]</para>
        ///     <para>               [ 0.02367868  0.11246539]</para>
        ///     <para>               [ 0.0290431   0.2160573 ]]</para>
        ///     <para>              [[-0.4898908  -0.31587923]</para>
        ///     <para>               [-0.08369008 -0.02142242]</para>
        ///     <para>               [-0.11092162 -0.42982462]</para>
        ///     <para>               [-0.06499392 -0.06495637]]</para>
        ///     <para>              [[-1.0213816  -1.526392  ]</para>
        ///     <para>               [-1.2008414  -1.1990893 ]</para>
        ///     <para>               [-1.5385206  -1.4795225 ]</para>
        ///     <para>               [-1.2194707  -1.3211205 ]]]</para>
        ///     <para>             [[[ 0.03942481  0.24021089]</para>
        ///     <para>               [ 0.21330701  0.1940066 ]</para>
        ///     <para>               [ 0.04778443  0.17912441]</para>
        ///     <para>               [ 0.31488964  0.25287187]]</para>
        ///     <para>              [[-0.23907584 -0.4470462 ]</para>
        ///     <para>               [-0.29266903 -0.2631998 ]</para>
        ///     <para>               [-0.3677222  -0.40683383]</para>
        ///     <para>               [-0.11288315 -0.13154092]]</para>
        ///     <para>              [[-1.5438497  -1.7834496 ]</para>
        ///     <para>               [-1.431566   -1.8647819 ]</para>
        ///     <para>               [-1.9812102  -1.675859  ]</para>
        ///     <para>               [-1.3823645  -1.8503251 ]]]]</para>
        ///     <para>
        ///         <NDArray 2 x3x4x2 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L165</para>
        /// </summary>
        /// <param name="data">Input ndarray</param>
        /// <param name="mean">Sequence of means for each channel. Default value is 0.</param>
        /// <param name="std">Sequence of standard deviations for each channel. Default value is 1.</param>
        /// <returns>returns new symbol</returns>
        public NDArray Normalize(NDArray data, Tuple<float> mean = null, Tuple<float> std = null)
        {
            if (mean == null) mean = new Tuple<float>(0, 0, 0, 0);
            if (std == null) std = new Tuple<float>(1, 1, 1, 1);
            
            return new Operator("_image_normalize")
                .SetParam("mean", mean)
                .SetParam("std", std)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L192</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <returns>returns new symbol</returns>
        public NDArray FlipLeftRight(NDArray data)
        {
            return new Operator("_image_flip_left_right")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L196</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomFlipLeftRight(NDArray data)
        {
            return new Operator("_image_random_flip_left_right")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L200</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <returns>returns new symbol</returns>
        public NDArray FlipTopBottom(NDArray data)
        {
            return new Operator("_image_flip_top_bottom")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L204</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomFlipTopBottom(NDArray data)
        {
            return new Operator("_image_random_flip_top_bottom")
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L208</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="min_factor">Minimum factor.</param>
        /// <param name="max_factor">Maximum factor.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomBrightness(NDArray data, float min_factor, float max_factor)
        {
            return new Operator("_image_random_brightness")
                .SetParam("min_factor", min_factor)
                .SetParam("max_factor", max_factor)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L214</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="min_factor">Minimum factor.</param>
        /// <param name="max_factor">Maximum factor.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomContrast(NDArray data, float min_factor, float max_factor)
        {
            return new Operator("_image_random_contrast")
                .SetParam("min_factor", min_factor)
                .SetParam("max_factor", max_factor)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L221</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="min_factor">Minimum factor.</param>
        /// <param name="max_factor">Maximum factor.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomSaturation(NDArray data, float min_factor, float max_factor)
        {
            return new Operator("_image_random_saturation")
                .SetParam("min_factor", min_factor)
                .SetParam("max_factor", max_factor)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L228</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="min_factor">Minimum factor.</param>
        /// <param name="max_factor">Maximum factor.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomHue(NDArray data, float min_factor, float max_factor)
        {
            return new Operator("_image_random_hue")
                .SetParam("min_factor", min_factor)
                .SetParam("max_factor", max_factor)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L235</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="brightness">How much to jitter brightness.</param>
        /// <param name="contrast">How much to jitter contrast.</param>
        /// <param name="saturation">How much to jitter saturation.</param>
        /// <param name="hue">How much to jitter hue.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomColorJitter(NDArray data, float brightness, float contrast, float saturation, float hue)
        {
            return new Operator("_image_random_color_jitter")
                .SetParam("brightness", brightness)
                .SetParam("contrast", contrast)
                .SetParam("saturation", saturation)
                .SetParam("hue", hue)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Adjust the lighting level of the input. Follow the AlexNet style.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L242</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="alpha">The lighting alphas for the R, G, B channels.</param>
        /// <returns>returns new symbol</returns>
        public NDArray AdjustLighting(NDArray data, Tuple<double> alpha)
        {
            return new Operator("_image_adjust_lighting")
                .SetParam("alpha", alpha)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Randomly add PCA noise. Follow the AlexNet style.</para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\image_random.cc:L249</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="alpha_std">Level of the lighting noise.</param>
        /// <returns>returns new symbol</returns>
        public NDArray RandomLighting(NDArray data, float alpha_std = 0.05f)
        {
            return new Operator("_image_random_lighting")
                .SetParam("alpha_std", alpha_std)
                .SetInput("data", data)
                .Invoke();
        }

        /// <summary>
        ///     <para>Resize an image NDArray of shape (H x W x C) or (N x H x W x C) </para>
        ///     <para>to the given size</para>
        ///     <para>Example:</para>
        ///     <para>    .. code-block:: python</para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        mx.nd.image.resize(image, (3, 3))</para>
        ///     <para>            [[[124 111 197]</para>
        ///     <para>              [158  80 155]</para>
        ///     <para>              [193  50 112]]</para>
        ///     <para> </para>
        ///     <para>             [[110 100 113]</para>
        ///     <para>              [134 165 148]</para>
        ///     <para>              [157 231 182]]</para>
        ///     <para> </para>
        ///     <para>             [[202 176 134]</para>
        ///     <para>              [174 191 149]</para>
        ///     <para>              [147 207 164]]]</para>
        ///     <para>
        ///         <NDArray 3 x3x3 @ cpu(0)>
        ///     </para>
        ///     <para>        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)</para>
        ///     <para>        mx.nd.image.resize(image, (2, 2))            </para>
        ///     <para>            [[[[ 59 133  80]</para>
        ///     <para>               [187 114 153]]</para>
        ///     <para> </para>
        ///     <para>              [[ 38 142  39]</para>
        ///     <para>               [207 131 124]]]</para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>              [[[117 125 136]</para>
        ///     <para>               [191 166 150]]</para>
        ///     <para> </para>
        ///     <para>              [[129  63 113]</para>
        ///     <para>               [182 109  48]]]]</para>
        ///     <para>
        ///         <NDArray 2 x2x2x3 @ cpu(0)>
        ///     </para>
        ///     <para> </para>
        ///     <para> </para>
        ///     <para>Defined in C:\Jenkins\workspace\mxnet\mxnet\src\operator\image\resize.cc:L70</para>
        /// </summary>
        /// <param name="data">The input.</param>
        /// <param name="size">Size of new image. Could be (width, height) or (size)</param>
        /// <param name="keep_ratio">Whether to resize the short edge or both edges to `size`, if size is give as an integer.</param>
        /// <param name="interp">
        ///     Interpolation method for resizing. By default uses bilinear interpolationOptions are INTER_NEAREST
        ///     - a nearest-neighbor interpolationINTER_LINEAR - a bilinear interpolationINTER_AREA - resampling using pixel area
        ///     relationINTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhoodINTER_LANCZOS4 - a Lanczos interpolation
        ///     over 8x8 pixel neighborhoodNote that the GPU version only support bilinear interpolation(1) and the result on cpu
        ///     would be slightly different from gpu.It uses opencv resize function which tend to align center on cpuwhile using
        ///     contrib.bilinearResize2D which aligns corner on gpu
        /// </param>
        /// <returns>returns new symbol</returns>
        public NDArray Resize(NDArray data, Shape size = null, bool keep_ratio = false, int interp = 1)
        {
            if (size == null) size = new Shape();

            return new Operator("_image_resize")
                .SetParam("size", size)
                .SetParam("keep_ratio", keep_ratio)
                .SetParam("interp", interp)
                .SetInput("data", data)
                .Invoke();
        }
    }
}