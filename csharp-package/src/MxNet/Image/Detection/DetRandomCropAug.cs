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
using NumpyDotNet;

namespace MxNet.Image
{
    public class DetRandomCropAug : DetAugmenter
    {
        private readonly bool enabled;

        public DetRandomCropAug(float min_object_covered = 0.1f, float min_eject_coverage = 0.3f,
            (float, float)? aspect_ratio_range = null,
            (float, float)? area_range = null, int max_attempts = 50)
        {
            MinObjectCovered = min_object_covered;
            MinEjectCovered = min_eject_coverage;
            AspectRatioRange = aspect_ratio_range.HasValue ? aspect_ratio_range.Value : (0.75f, 1.33f);
            AreaRange = area_range.HasValue ? area_range.Value : (0.05f, 1);
            MinEjectCoverage = min_eject_coverage;
            MaxAttempts = max_attempts;
            enabled = false;
            if (AreaRange.Item2 <= 0 || AreaRange.Item1 > AreaRange.Item2)
                Logger.Warning("Skip DetRandomCropAug due to invalid area_range: " + AreaRange.ToValueString());
            else if (AspectRatioRange.Item1 > AspectRatioRange.Item2 || AspectRatioRange.Item1 <= 0)
                Logger.Warning("Skip DetRandomCropAug due to invalid aspect_ratio_range: " +
                               aspect_ratio_range.ToValueString());
            else
                enabled = true;
        }

        public float MinObjectCovered { get; set; }

        public float MinEjectCovered { get; set; }

        public (float, float) AspectRatioRange { get; set; }

        public (float, float) AreaRange { get; set; }

        public float MinEjectCoverage { get; set; }

        public int MaxAttempts { get; set; }

        public override (NDArray, NDArray) Call(NDArray src, NDArray label)
        {
            var (x, y, w, h, crop) = RandomCropProposal(label, src.Shape[0], src.Shape[1]);
            label = crop != null ? crop : label;
            src = Img.FixedCrop(src, x, y, w, h);
            return (src, crop);
        }

        private NDArray CalculateAreas(NDArray label)
        {
            var heights = nd.MaximumScalar(label[":,3"] - label[":,1"], 0);
            var widths = nd.MaximumScalar(label[":,2"] - label[":,0"], 0);

            return heights * widths;
        }

        private NDArray Intersect(NDArray label, float xmin, float ymin, float xmax, float ymax)
        {
            var left = nd.MaximumScalar(label[":,0"], xmin);
            var right = nd.MinimumScalar(label[":,2"], xmax);
            var top = nd.MaximumScalar(label[":,1"], ymin);
            var bot = nd.MinimumScalar(label[":,3"], ymax);

            var invalid = nd.Where(nd.LogicalOr(nd.GreaterEqual(left, right), nd.GreaterEqual(top, bot)));
            var @out = label.Copy();
            @out[":,0"] = left;
            @out[":,1"] = top;
            @out[":,2"] = right;
            @out[":,3"] = bot;
            @out[invalid + ",:"] = nd.ZerosLike(@out[invalid + ",:"]);
            return @out;
        }

        private bool CheckSatisfyConstraints(NDArray label, int xmin, int ymin, int xmax, int ymax, int width,
            int height)
        {
            if ((xmax - xmin) * (ymax - ymin) < 2)
                return false;
            var x1 = (float) xmin / width;
            var y1 = (float) ymin / height;
            var x2 = (float) xmax / width;
            var y2 = (float) ymax / height;
            var object_areas = CalculateAreas(label[":,1:"]);
            var valid_objects = nd.Where(nd.GreaterScalar(object_areas * width * height, 2));
            if (valid_objects.Size < 1)
                return false;
            var intersects = Intersect(label[valid_objects + ",1:"], x1, y1, x2, y2);
            var coverages = CalculateAreas(intersects) / object_areas[valid_objects];
            coverages = coverages[nd.Where(nd.GreaterScalar(coverages, 0))];
            return coverages.Size > 0 && nd.Min(coverages).AsScalar<float>() > MinObjectCovered;
        }

        private NDArray UpdateLabels(NDArray label, int[] crop_box, int height, int width)
        {
            var xmin = (float) crop_box[0] / width;
            var ymin = (float) crop_box[1] / height;
            var w = (float) crop_box[2] / width;
            var h = (float) crop_box[3] / height;
            var @out = label.Copy();
            @out[":,(1, 3)"] -= xmin; //ToDo: Support x[:,(1,3)]
            @out[":,(2, 4)"] -= ymin;
            @out[":,(1, 3)"] /= w;
            @out[":,(2, 4)"] /= h;
            @out[":,1:5"] = nd.MaximumScalar(@out[":,1:5"], 0);
            @out[":,1:5"] = nd.MinusScalar(@out[":,1:5"], 1);
            var coverage = CalculateAreas(@out[":,1:"]) * w * h / CalculateAreas(label[":,1:"]);
            var valid = nd.LogicalAnd(@out[":,3"] > @out[":,1"], @out[":,4"] > @out[":,2"]);
            valid = nd.LogicalAnd(valid, coverage > MinEjectCoverage);
            valid = nd.Where(valid);
            if (valid.Size < 1)
                return null;
            @out = @out[valid.Shape + ",:"];
            return @out;
        }

        private (int, int, int, int, NDArray) RandomCropProposal(NDArray label, int height, int width)
        {
            if (!enabled || height <= 0 || width <= 0)
                return (0, 0, 0, 0, null);

            var min_area = AreaRange.Item1 * height * width;
            var max_area = AreaRange.Item2 * height * width;
            for (var i = 0; i < MaxAttempts; i++)
            {
                float ratio = FloatRnd.Uniform(AspectRatioRange.Item1, AspectRatioRange.Item2);
                if (ratio <= 0)
                    continue;
                var h = (int) Math.Round(Math.Pow(min_area / ratio, 2));
                var max_h = (int) Math.Round(Math.Pow(max_area / ratio, 2));
                if (Math.Round(max_h * ratio) > width)
                    max_h = (int) ((width + 0.4999999) /
                                   ratio); //find smallest max_h satifying round(max_h * ratio) <= width


                if (max_h > height)
                    max_h = height;
                if (h > max_h)
                    h = max_h;
                if (h < max_h)
                    h = IntRnd.Uniform(h, max_h); // generate random h in range [h, max_h]

                var w = (int) Math.Round(h * ratio);
                if (w <= width)
                    throw new Exception("Error: w <= width");

                var area = w * h;
                if (area < min_area)
                {
                    h += 1;
                    w = (int) Math.Round(h * ratio);
                    area = w * h;
                }

                if (area > max_area)
                {
                    h -= 1;
                    w = (int) Math.Round(h * ratio);
                    area = w * h;
                }

                if (!(min_area <= area) && area <= max_area && 0 <= w && w <= width && 0 <= h && h <= height)
                    continue;

                NDArray new_label = null;

                var y = IntRnd.Uniform(0, Math.Max(0, height - h));
                var x = IntRnd.Uniform(0, Math.Max(0, width - w));
                if (CheckSatisfyConstraints(label, x, y, x + w, y + h, width, height))
                    new_label = UpdateLabels(label, new int[] {x, y, w, h}, height, width);
                if (new_label != null)
                    return (x, y, w, h, new_label);
            }

            return (0, 0, 0, 0, null);
        }
    }
}