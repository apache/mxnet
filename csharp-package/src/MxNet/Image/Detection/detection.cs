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
using OpenCvSharp;

namespace MxNet.Image
{
    public class Detection
    {
        public static DetRandomSelectAug CreateMultiRandCropAugmenter(float min_object_covered = 0.1f,
            (float, float)? aspect_ratio_range = null,
            (float, float)? area_range = null, float min_eject_coverage = 0.3f,
            int max_attempts = 50, float skip_prob = 0)
        {
            throw new NotImplementedException();
        }

        public static DetAugmenter CreateDetAugmenter(Shape data_shape, int resize = 0, float rand_crop = 0,
            float rand_pad = 0, float rand_gray = 0,
            bool rand_mirror = false, NDArray mean = null, NDArray std = null, float brightness = 0,
            float contrast = 0, float saturation = 0, float pca_noise = 0, float hue = 0,
            InterpolationFlags inter_method = InterpolationFlags.Cubic,
            float min_object_covered = 0.1f, (float, float)? aspect_ratio_range = null,
            (float, float)? area_range = null, float min_eject_coverage = 0.3f,
            int max_attempts = 50, float pad_val = 127)
        {
            throw new NotImplementedException();
        }
    }
}