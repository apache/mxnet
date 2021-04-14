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
using System.Diagnostics;
using MxNet.Numpy;
using OpenCvSharp;
using Random = System.Random;

namespace MxNet.Image
{
    public enum ImgInterp
    {
        Nearest_Neighbors = 0,
        Bilinear = 1,
        Area_Based = 2,
        Bicubic = 3,
        Lanczos = 4,
        Cubic_Area_Bilinear = 9,
        Random_Select = 10
    }

    public class Img
    {
        public static ndarray ImRead(string filename, int flag = 1, bool to_rgb = false)
        {
            Mat mat = Cv2.ImRead(filename, (ImreadModes)flag);
            if (to_rgb)
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);
            return mat;
            //return nd.Cvimread(filename, flag, to_rgb);
        }

        public static ndarray ImResize(ndarray src, int w, int h, InterpolationFlags interp = InterpolationFlags.Linear)
        {
            //Mat mat = new Mat();
            //Mat input = src;
            //Cv2.Resize(input, input, new Size(w, h), interpolation: interp);
            //return input;
            return nd.Cvimresize(src, w, h, (int) interp);
        }

        public static void ImShow(ndarray x, string winname = "", bool wait = true)
        {
            if (winname == "")
                winname = "test";

            bool transpose = true;

            if (x.shape.Dimension == 4)
                x = x.reshape(x.shape[1], x.shape[2], x.shape[3]).AsType(DType.UInt8);
            else
                x = x.AsType(DType.UInt8);

            if (x.shape[0] > 3)
                transpose = false;

            if (transpose)
                x = x.transpose(new Shape(1, 2, 0));
            ndarray.WaitAll();
            Mat mat = x;
            
            Cv2.ImShow(winname, mat);
            ndarray.WaitAll();
            if (wait)
                Cv2.WaitKey();
        }

        public static ndarray ImDecode(byte[] buf, int flag = 1, bool to_rgb = true)
        {
            return nd.Cvimdecode(buf, flag, to_rgb);
        }

        public static (int, int) ScaleDown((int, int) src_size, (int, int) size)
        {
            var (w, h) = size;
            var (sw, sh) = src_size;
            if (sh < h)
                (w, h) = (w * sh / h, sh);
            if (sw < w)
                (w, h) = (sw, h * sw / w);

            return (w, h);
        }

        public static ndarray CopyMakeBorder(ndarray src, int top, int bot, int left, int right,
            BorderTypes type = BorderTypes.Constant)
        {
            return nd.CvcopyMakeBorder(src, top, bot, left, right, (int) type);
        }

        public static InterpolationFlags GetInterp(InterpolationFlags interp, (int, int, int, int)? sizes = null)
        {
            if (interp == InterpolationFlags.Cubic)
            {
                if (sizes.HasValue)
                {
                    var (oh, ow, nh, nw) = sizes.Value;

                    if (nh > oh && nw > ow)
                        return InterpolationFlags.Area;
                    if (nh < oh && nw < ow)
                        return InterpolationFlags.Cubic;
                    return InterpolationFlags.Linear;
                }

                return InterpolationFlags.Area;
            }

            return interp;
        }

        public static ndarray ResizeShort(ndarray src, int size, InterpolationFlags interp = InterpolationFlags.Linear)
        {
            var (h, w, _) = (src.shape[0], src.shape[1], src.shape[2]);
            int new_h;
            int new_w;
            if (h > w)
                (new_h, new_w) = (size * (int)Math.Round((double)h / w), size);
            else
                (new_h, new_w) = (size, size * (int)Math.Round((double)w / h));

            return ImResize(src, new_w, new_h, GetInterp(interp, (h, w, new_h, new_w)));
        }

        public static ndarray FixedCrop(ndarray src, int x0, int y0, int w, int h,
            (int, int)? size = null, InterpolationFlags interp = InterpolationFlags.Area)
        {
            var output = nd.Slice(src, new Shape(y0, x0, 0), new Shape(y0 + h, x0 + w, src.shape[2]));
            if (size.HasValue && size.Value.Item1 != w && size.Value.Item2 != h)
            {
                var sizes = (h, w, size.Value.Item2, size.Value.Item1);
                output = ImResize(output, size.Value.Item1, size.Value.Item2, GetInterp(interp, sizes));
            }

            return output;
        }

        public static (ndarray, (int, int, int, int)) RandomCrop(ndarray src, (int, int) size,
            InterpolationFlags interp = InterpolationFlags.Area)
        {
            var (h, w, _) = (src.shape[0], src.shape[1], src.shape[2]);
            var (new_w, new_h) = ScaleDown((w, h), size);
            var x0 = new Random().Next(0, w - new_w);
            var y0 = new Random().Next(0, h - new_h);
            var output = FixedCrop(src, x0, y0, new_w, new_h, size, interp);
            return (output, (x0, y0, new_w, new_h));
        }

        public static (ndarray, (int, int, int, int)) CenterCrop(ndarray src, (int, int) size,
            InterpolationFlags interp = InterpolationFlags.Area)
        {
            var (h, w, _) = (src.shape[0], src.shape[1], src.shape[2]);
            var (new_w, new_h) = ScaleDown((w, h), size);
            var x0 = (w - new_w) / 2;
            var y0 = (h - new_h) / 2;
            var output = FixedCrop(src, x0, y0, new_w, new_h, size, interp);
            return (output, (x0, y0, new_w, new_h));
        }

        public static ndarray ColorNormalize(ndarray src, ndarray mean, ndarray std = null)
        {
            if (mean != null)
                src = src - mean;

            if (std != null)
                src = src / std;

            return src;
        }

        public static (ndarray, (int, int, int, int)) RandomSizeCrop(ndarray src, (int, int) size, (float, float) area,
            (float, float) ratio, InterpolationFlags interp = InterpolationFlags.Area)
        {
            var (h, w, _) = (src.shape[0], src.shape[1], src.shape[2]);
            var src_area = h * w;
            for (var i = 0; i < 10; i++)
            {
                float target_area = FloatRnd.Uniform(area.Item1, area.Item2) * src_area;
                var log_ratio = ((float) Math.Log(ratio.Item1), (float) Math.Log(ratio.Item2));
                var new_ratio = Math.Exp(FloatRnd.Uniform(log_ratio.Item1, log_ratio.Item2));
                var new_w = (int) Math.Round(Math.Pow(target_area * new_ratio, 2));
                var new_h = (int) Math.Round(Math.Pow(target_area / new_ratio, 2));
                if (new_w <= w && new_h <= h)
                {
                    var x0 = new Random().Next(0, w - new_w);
                    var y0 = new Random().Next(0, h - new_h);

                    var @out = FixedCrop(src, x0, y0, new_w, new_h, size, interp);
                    return (@out, (x0, y0, new_w, new_h));
                }
            }

            return CenterCrop(src, size, interp);
        }

        public static ndarray ImRotate(ndarray src, float rotation_degrees, bool zoom_in, bool zoom_out)
        {
            ndarray globalscale;
            ndarray scale_y;
            ndarray scale_x;
            if (zoom_in && zoom_out) {
                throw new Exception("`zoom_in` and `zoom_out` cannot be both True");
            }
            if (src.dtype.Name != np.Float32.Name) {
                throw new Exception("Only `float32` images are supported by this function");
            }
            // handles the case in which a single image is passed to this function
            var expanded = false;
            if (src.ndim == 3) {
                expanded = true;
                src = src.expand_dims(axis: 0);
            } else if (src.ndim != 4) {
                throw new Exception("Only 3D and 4D are supported by this function");
            }
            
            // when a scalar is passed we wrap it into an array
            var rotation_degrees_array = np.full(new Shape(src.shape[0]), rotation_degrees);
            
            rotation_degrees_array = rotation_degrees_array.AsInContext(src.ctx);
            var rotation_rad = np.pi * rotation_degrees_array / 180;
            // reshape the rotations angle in order to be broadcasted
            // over the `src` tensor
            rotation_rad = rotation_rad.expand_dims(axis: 1).expand_dims(axis: 2);
            var h = src.shape[2];
            var w = src.shape[3];
            // Generate a grid centered at the center of the image
            float hscale = (h - 1) / 2;
            float wscale = (w - 1) / 2;
            var h_matrix = (np.repeat(np.arange(h, ctx: src.ctx).Cast(np.Float32).reshape(h, 1), w, axis: 1) - hscale).expand_dims(axis: 0);
            var w_matrix = (np.repeat(np.arange(w, ctx: src.ctx).Cast(np.Float32).reshape(1, w), h, axis: 0) - wscale).expand_dims(axis: 0);
            // perform rotation on the grid
            var c_alpha = np.cos(rotation_rad);
            var s_alpha = np.sin(rotation_rad);
            var w_matrix_rot = w_matrix * c_alpha - h_matrix * s_alpha;
            var h_matrix_rot = w_matrix * s_alpha + h_matrix * c_alpha;
            // NOTE: grid normalization must be performed after the rotation
            //       to keep the aspec ratio
            w_matrix_rot = w_matrix_rot / wscale;
            h_matrix_rot = h_matrix_rot / hscale;
            
            // compute the scale factor in case `zoom_in` or `zoom_out` are True
            if (zoom_in || zoom_out) {
                var rho_corner = Math.Sqrt(h * h + w * w);
                var ang_corner = np.arctan(h / w);
                var corner1_x_pos = np.abs(rho_corner * np.cos(ang_corner + np.abs(rotation_rad)));
                var corner1_y_pos = np.abs(rho_corner * np.sin(ang_corner + np.abs(rotation_rad)));
                var corner2_x_pos = np.abs(rho_corner * np.cos(ang_corner - np.abs(rotation_rad)));
                var corner2_y_pos = np.abs(rho_corner * np.sin(ang_corner - np.abs(rotation_rad)));
                var max_x = np.maximum(corner1_x_pos, corner2_x_pos);
                var max_y = np.maximum(corner1_y_pos, corner2_y_pos);
                if (zoom_out) {
                    scale_x = max_x / w;
                    scale_y = max_y / h;
                    globalscale = np.maximum(scale_x, scale_y);
                } else {
                    scale_x = w / max_x;
                    scale_y = h / max_y;
                    globalscale = np.minimum(scale_x, scale_y);
                }
                globalscale = globalscale.expand_dims(axis: 3);
            } else {
                globalscale = 1;
            }
            var grid = np.concatenate(new NDArrayList(w_matrix_rot.expand_dims(axis: 1), h_matrix_rot.expand_dims(axis: 1)), axis: 1);
            grid = grid * globalscale;
            
            ndarray rot_img = nd.BilinearSampler(src, grid);
          
            return rot_img;
        }
        
        public static ndarray ImRotate(ndarray src, ndarray rotation_degrees_array, bool zoom_in, bool zoom_out)
        {
            ndarray globalscale;
            ndarray scale_y;
            ndarray scale_x;
            if (zoom_in && zoom_out) {
                throw new Exception("`zoom_in` and `zoom_out` cannot be both True");
            }
            if (src.dtype.Name != np.Float32.Name) {
                throw new Exception("Only `float32` images are supported by this function");
            }
            // handles the case in which a single image is passed to this function
            var expanded = false;
            if (src.ndim == 3) {
                expanded = true;
                src = src.expand_dims(axis: 0);
            } else if (src.ndim != 4) {
                throw new Exception("Only 3D and 4D are supported by this function");
            }

            rotation_degrees_array = rotation_degrees_array.AsInContext(src.ctx);
            var rotation_rad = np.pi * rotation_degrees_array / 180;
            // reshape the rotations angle in order to be broadcasted
            // over the `src` tensor
            rotation_rad = rotation_rad.expand_dims(axis: 1).expand_dims(axis: 2);
            var h = src.shape[2];
            var w = src.shape[3];
            // Generate a grid centered at the center of the image
            float hscale = (h - 1) / 2;
            float wscale = (w - 1) / 2;
            var h_matrix = (np.repeat(np.arange(h, ctx: src.ctx).Cast(np.Float32).reshape(h, 1), w, axis: 1) - hscale).expand_dims(axis: 0);
            var w_matrix = (np.repeat(np.arange(w, ctx: src.ctx).Cast(np.Float32).reshape(1, w), h, axis: 0) - wscale).expand_dims(axis: 0);
            // perform rotation on the grid
            var c_alpha = np.cos(rotation_rad);
            var s_alpha = np.sin(rotation_rad);
            var w_matrix_rot = w_matrix * c_alpha - h_matrix * s_alpha;
            var h_matrix_rot = w_matrix * s_alpha + h_matrix * c_alpha;
            // NOTE: grid normalization must be performed after the rotation
            //       to keep the aspec ratio
            w_matrix_rot = w_matrix_rot / wscale;
            h_matrix_rot = h_matrix_rot / hscale;
            
            // compute the scale factor in case `zoom_in` or `zoom_out` are True
            if (zoom_in || zoom_out) {
                var rho_corner = Math.Sqrt(h * h + w * w);
                var ang_corner = np.arctan(h / w);
                var corner1_x_pos = np.abs(rho_corner * np.cos(ang_corner + np.abs(rotation_rad)));
                var corner1_y_pos = np.abs(rho_corner * np.sin(ang_corner + np.abs(rotation_rad)));
                var corner2_x_pos = np.abs(rho_corner * np.cos(ang_corner - np.abs(rotation_rad)));
                var corner2_y_pos = np.abs(rho_corner * np.sin(ang_corner - np.abs(rotation_rad)));
                var max_x = np.maximum(corner1_x_pos, corner2_x_pos);
                var max_y = np.maximum(corner1_y_pos, corner2_y_pos);
                if (zoom_out) {
                    scale_x = max_x / w;
                    scale_y = max_y / h;
                    globalscale = np.maximum(scale_x, scale_y);
                } else {
                    scale_x = w / max_x;
                    scale_y = h / max_y;
                    globalscale = np.minimum(scale_x, scale_y);
                }
                globalscale = globalscale.expand_dims(axis: 3);
            } else {
                globalscale = 1;
            }
            var grid = np.concatenate(new NDArrayList(w_matrix_rot.expand_dims(axis: 1), h_matrix_rot.expand_dims(axis: 1)), axis: 1);
            grid = grid * globalscale;
            
            ndarray rot_img = nd.BilinearSampler(src, grid);
          
            return rot_img;
        }

        public static ndarray RandomRotate(ndarray src, (float, float) angle_limits, bool zoom_in, bool zoom_out)
        {
            ndarray rotation_degrees = null;
            if (src.ndim == 3) {
                rotation_degrees = np.random.uniform(angle_limits.Item1, angle_limits.Item2);
            } else {
                var n = src.shape[0];
                rotation_degrees = np.random.uniform(angle_limits.Item1, angle_limits.Item2, size: new Shape(n));
            }
            return ImRotate(src, rotation_degrees, zoom_in: zoom_in, zoom_out: zoom_out);
        }

        public static Augmenter[] CreateAugmenter(Shape data_shape, int resize = 0, bool rand_crop = false,
            bool rand_resize = false, bool rand_mirror = false, ndarray mean = null, ndarray std = null,
            float brightness = 0, float contrast = 0, float saturation = 0, float hue = 0, float pca_noise = 0,
            float rand_gray = 0, InterpolationFlags inter_method = InterpolationFlags.Area)
        {
            var auglist = new List<Augmenter>();
            if (resize > 0) {
                auglist.Add(new ResizeAug(resize, inter_method));
            }
            var crop_size = (data_shape[2], data_shape[1]);
            if (rand_resize) {
                Debug.Assert(rand_crop);
                auglist.Add(new RandomSizedCropAug(crop_size, (0.08f, 0.08f), (3.0f / 4.0f, 4.0f / 3.0f), inter_method));
            } else if (rand_crop) {
                auglist.Add(new RandomCropAug(crop_size, inter_method));
            } else {
                auglist.Add(new CenterCropAug(crop_size, inter_method));
            }
            if (rand_mirror) {
                auglist.Add(new HorizontalFlipAug(0.5f));
            }

            auglist.Add(new CastAug());
            if (brightness > 0 || contrast > 0 || saturation > 0) {
                auglist.Add(new ColorJitterAug(brightness, contrast, saturation));
            }

            if (hue > 0) {
                auglist.Add(new HueJitterAug(hue));
            }

            if (pca_noise > 0) {
                var eigval = np.array(new float[] {
                    55.46f,
                    4.794f,
                    1.148f
                });

                var eigvec = np.array(new float[,] {
                    {
                        -0.5675f,
                        0.7192f,
                        0.4009f
                    },
                    {
                        -0.5808f,
                        -0.0045f,
                        -0.814f
                    },
                    {
                        -0.5836f,
                        -0.6948f,
                        0.4203f
                    }
                });
                auglist.Add(new LightingAug(pca_noise, eigval, eigvec));
            }
            if (rand_gray > 0) {
                auglist.Add(new RandomGrayAug(rand_gray));
            }
            if (mean != null) {
                Debug.Assert(new List<int> {
                    1,
                    3
                }.Contains(mean.shape[0]));
            }

            if (std != null)
            {
                Debug.Assert(new List<int> {
                    1,
                    3
                }.Contains(std.shape[0]));
            }

            if (mean != null || std != null) {
                auglist.Add(new ColorNormalizeAug(mean, std));
            }

            return auglist.ToArray();
        }
    }
}