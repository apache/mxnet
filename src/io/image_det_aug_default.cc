/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_det_aug_default.cc
 * \brief Default augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include "./image_augmenter.h"
#include "../common/utils.h"

namespace mxnet {
namespace io {

using nnvm::Tuple;

namespace image_det_aug_default_enum {
enum ImageDetAugDefaultCropEmitMode {kCenter, kOverlap};
enum ImageDetAugDefaultResizeMode {kForce, kShrink, kFit};
}

/*! \brief image detection augmentation parameters*/
struct DefaultImageDetAugmentParam : public dmlc::Parameter<DefaultImageDetAugmentParam> {
  /*! \brief resize shorter edge to size before applying other augmentations */
  int resize;
  /*! \brief probability we do random cropping, use prob <= 0 to disable */
  float rand_crop_prob;
  /*! \brief min crop scales */
  Tuple<float> min_crop_scales;
  /*! \brief max crop scales */
  Tuple<float> max_crop_scales;
  /*! \brief min crop aspect ratios */
  Tuple<float> min_crop_aspect_ratios;
  /*! \brief max crop aspect ratios */
  Tuple<float> max_crop_aspect_ratios;
  /*! \brief min IOUs between ground-truths and crop boxes */
  Tuple<float> min_crop_overlaps;
  /*! \brief max IOUs between ground-truths and crop boxes */
  Tuple<float> max_crop_overlaps;
  /*! \brief min itersection/gt_area between ground-truths and crop boxes */
  Tuple<float> min_crop_sample_coverages;
  /*! \brief max itersection/gt_area between ground-truths and crop boxes */
  Tuple<float> max_crop_sample_coverages;
  /*! \brief min itersection/crop_area between ground-truths and crop boxes */
  Tuple<float> min_crop_object_coverages;
  /*! \brief max itersection/crop_area between ground-truths and crop boxes */
  Tuple<float> max_crop_object_coverages;
  /*! \brief number of crop samplers, skip random crop if <= 0 */
  int num_crop_sampler;
  /*! \beief 0-emit ground-truth if center out of crop area
   * 1-emit if overlap < emit_overlap_thresh
   */
  int crop_emit_mode;
  /*! \brief ground-truth emition threshold specific for crop_emit_mode == 1 */
  float emit_overlap_thresh;
  /*! \brief maximum trials for cropping, skip cropping if fails exceed this number */
  Tuple<int> max_crop_trials;
  /*! \brief random padding prob */
  float rand_pad_prob;
  /*!< \brief maximum padding scale */
  float max_pad_scale;
  /*! \brief max random in H channel */
  int max_random_hue;
  /*! \brief random H prob */
  float random_hue_prob;
  /*! \brief max random in S channel */
  int max_random_saturation;
  /*! \brief random saturation prob */
  float random_saturation_prob;
  /*! \brief max random in L channel */
  int max_random_illumination;
  /*! \brief random illumination change prob */
  float random_illumination_prob;
  /*! \brief max random contrast */
  float max_random_contrast;
  /*! \brief random contrast prob */
  float random_contrast_prob;
  /*! \brief random mirror prob */
  float rand_mirror_prob;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief shape of the image data */
  TShape data_shape;
  /*! \brief resize mode, 0-force
   * 1-Shrink to data_shape, preserve ratio,
   * 2-fit to data_shape, preserve ratio
   */
  int resize_mode;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DefaultImageDetAugmentParam) {
    DMLC_DECLARE_FIELD(resize).set_default(-1)
        .describe("Augmentation Param: scale shorter edge to size "
                  "before applying other augmentations, -1 to disable.");
    DMLC_DECLARE_FIELD(rand_crop_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability of random cropping, <= 0 to disable");
    DMLC_DECLARE_FIELD(min_crop_scales).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Min crop scales.");
    DMLC_DECLARE_FIELD(max_crop_scales).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Max crop scales.");
    DMLC_DECLARE_FIELD(min_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Min crop aspect ratios.");
    DMLC_DECLARE_FIELD(max_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Max crop aspect ratios.");
    DMLC_DECLARE_FIELD(min_crop_overlaps).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum crop IOU between crop_box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_overlaps).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum crop IOU between crop_box and ground-truth.");
    DMLC_DECLARE_FIELD(min_crop_sample_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum ratio of intersect/crop_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_sample_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum ratio of intersect/crop_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(min_crop_object_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum ratio of intersect/gt_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(max_crop_object_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum ratio of intersect/gt_area "
                  "between crop box and ground-truths.");
    DMLC_DECLARE_FIELD(num_crop_sampler).set_default(1)
        .describe("Augmentation Param: Number of crop samplers.");
    DMLC_DECLARE_FIELD(crop_emit_mode)
        .add_enum("center", image_det_aug_default_enum::kCenter)
        .add_enum("overlap", image_det_aug_default_enum::kOverlap)
        .set_default(image_det_aug_default_enum::kCenter)
        .describe("Augmentation Param: Emition mode for invalid ground-truths after crop. "
                  "center: emit if centroid of object is out of crop region; "
                  "overlap: emit if overlap is less than emit_overlap_thresh. ");
    DMLC_DECLARE_FIELD(emit_overlap_thresh).set_default(0.3f)
        .describe("Augmentation Param: Emit overlap thresh for emit mode overlap only.");
    DMLC_DECLARE_FIELD(max_crop_trials).set_default(Tuple<int>({25}))
        .describe("Augmentation Param: Skip cropping if fail crop trail count "
                  "exceeds this number.");
    DMLC_DECLARE_FIELD(rand_pad_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability for random padding.");
    DMLC_DECLARE_FIELD(max_pad_scale).set_default(1.0f)
        .describe("Augmentation Param: Maximum padding scale.");
    DMLC_DECLARE_FIELD(max_random_hue).set_default(0)
        .describe("Augmentation Param: Maximum random value of H channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_hue_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random hue.");
    DMLC_DECLARE_FIELD(max_random_saturation).set_default(0)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_saturation_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random saturation.");
    DMLC_DECLARE_FIELD(max_random_illumination).set_default(0)
        .describe("Augmentation Param: Maximum random value of L channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_illumination_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random illumination.");
    DMLC_DECLARE_FIELD(max_random_contrast).set_default(0)
        .describe("Augmentation Param: Maximum random value of delta contrast.");
    DMLC_DECLARE_FIELD(random_contrast_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply random contrast.");
    DMLC_DECLARE_FIELD(rand_mirror_prob).set_default(0.0f)
        .describe("Augmentation Param: Probability to apply horizontal flip aka. mirror.");
    DMLC_DECLARE_FIELD(fill_value).set_default(127)
        .describe("Augmentation Param: Filled color value while padding.");
    DMLC_DECLARE_FIELD(inter_method).set_default(1)
        .describe("Augmentation Param: 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Dataset Param: Shape of each instance generated by the DataIter.");
    DMLC_DECLARE_FIELD(resize_mode)
      .add_enum("force", image_det_aug_default_enum::kForce)
      .add_enum("shrink", image_det_aug_default_enum::kShrink)
      .add_enum("fit", image_det_aug_default_enum::kFit)
      .set_default(image_det_aug_default_enum::kForce)
      .describe("Augmentation Param: How image data fit in data_shape. "
                "force: force reshape to data_shape regardless of aspect ratio; "
                "shrink: ensure each side fit in data_shape, preserve aspect ratio; "
                "fit: fit image to data_shape, preserve ratio, will upscale if applicable.");
  }
};

DMLC_REGISTER_PARAMETER(DefaultImageDetAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListDefaultDetAugParams() {
  return DefaultImageDetAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV
using Rect = cv::Rect_<float>;

#ifdef _MSC_VER
#define M_PI CV_PI
#endif

/*! \brief helper class for better detection label handling */
class ImageDetLabel {
 public:
  /*! \brief Helper struct to store the coordinates and id for each object */
  struct ImageDetObject {
    float id;
    float left;
    float top;
    float right;
    float bottom;
    std::vector<float> extra;  // store extra info other than id and coordinates

    /*! \brief Return converted Rect object */
    Rect ToRect() const {
      return Rect(left, top, right - left, bottom - top);
    }

     /*! \brief Return projected coordinates according to new region */
     ImageDetObject Project(Rect box) const {
       ImageDetObject ret = *this;
       ret.left = std::max(0.f, (ret.left - box.x) / box.width);
       ret.top = std::max(0.f, (ret.top - box.y) / box.height);
       ret.right = std::min(1.f, (ret.right - box.x) / box.width);
       ret.bottom = std::min(1.f, (ret.bottom - box.y) / box.height);
       return ret;
     }

     /*! \brief Return Horizontally fliped coordinates */
     ImageDetObject HorizontalFlip() const {
       ImageDetObject ret = *this;
       ret.left = 1.f - this->right;
       ret.right = 1.f - this->left;
       return ret;
     }
  };  // struct ImageDetObject

  /*! \brief constructor from raw array of detection labels */
  explicit ImageDetLabel(const std::vector<float> &raw_label) {
    FromArray(raw_label);
  }

  /*! \brief construct from raw array with following format
   * header_width, object_width, (extra_headers...),
   * [id, xmin, ymin, xmax, ymax, (extra_object_info)] x N
   */
  void FromArray(const std::vector<float> &raw_label) {
    int label_width = static_cast<int>(raw_label.size());
    CHECK_GE(label_width, 7);  // at least 2(header) + 5(1 object)
    int header_width = static_cast<int>(raw_label[0]);
    CHECK_GE(header_width, 2);
    object_width_ = static_cast<int>(raw_label[1]);
    CHECK_GE(object_width_, 5);  // id, x1, y1, x2, y2...
    header_.assign(raw_label.begin(), raw_label.begin() + header_width);
    int num = (label_width - header_width) / object_width_;
    CHECK_EQ((label_width - header_width) % object_width_, 0);
    objects_.reserve(num);
    for (int i = header_width; i < label_width; i += object_width_) {
      ImageDetObject obj;
      auto it = raw_label.cbegin() + i;
      obj.id = *(it++);
      obj.left = *(it++);
      obj.top = *(it++);
      obj.right = *(it++);
      obj.bottom = *(it++);
      obj.extra.assign(it, it - 5 + object_width_);
      if (obj.right > obj.left && obj.bottom > obj.top) {
        objects_.push_back(obj);
      }
    }
  }

  /*! \brief Convert back to raw array */
  std::vector<float> ToArray() const {
    std::vector<float> out(header_);
    out.reserve(out.size() + objects_.size() * object_width_);
    for (auto& obj : objects_) {
      out.push_back(obj.id);
      out.push_back(obj.left);
      out.push_back(obj.top);
      out.push_back(obj.right);
      out.push_back(obj.bottom);
      out.insert(out.end(), obj.extra.begin(), obj.extra.end());
    }
    return out;
  }

  /*! \brief Intersection over Union between two rects */
  static float RectIOU(Rect a, Rect b) {
    float intersect = (a & b).area();
    if (intersect <= 0.f) return 0.f;
    return intersect / (a.area() + b.area() - intersect);
  }

  /*! \brief try crop image with given crop_box
   * return false if fail to meet any of the constraints
   * convert all objects if success
   */
  bool TryCrop(const Rect crop_box,
    const float min_crop_overlap, const float max_crop_overlap,
    const float min_crop_sample_coverage, const float max_crop_sample_coverage,
    const float min_crop_object_coverage, const float max_crop_object_coverage,
    const int crop_emit_mode, const float emit_overlap_thresh) {
    if (objects_.size() < 1) {
      return true;  // no object, raise error or just skip?
    }
    // check if crop_box valid
    bool valid = false;
    if (min_crop_overlap > 0.f || max_crop_overlap < 1.f ||
        min_crop_sample_coverage > 0.f || max_crop_sample_coverage < 1.f ||
        min_crop_object_coverage > 0.f || max_crop_object_coverage < 1.f) {
      for (auto& obj : objects_) {
        Rect gt_box = obj.ToRect();
        if (min_crop_overlap > 0.f || max_crop_overlap < 1.f) {
          float ovp = RectIOU(crop_box, gt_box);
          if (ovp < min_crop_overlap || ovp > max_crop_overlap) {
            continue;
          }
        }
        if (min_crop_sample_coverage > 0.f || max_crop_sample_coverage < 1.f) {
          float c = (crop_box & gt_box).area() / crop_box.area();
          if (c < min_crop_sample_coverage || c > max_crop_sample_coverage) {
            continue;
          }
        }
        if (min_crop_object_coverage > 0.f || max_crop_object_coverage < 1.f) {
          float c = (crop_box & gt_box).area() / gt_box.area();
          if (c < min_crop_object_coverage || c > max_crop_object_coverage) {
            continue;
          }
        }
        valid = true;
        break;
      }
    } else {
      valid = true;
    }

    if (!valid) return false;
    // transform ground-truth labels
    std::vector<ImageDetObject> new_objects;
    for (auto iter = objects_.begin(); iter != objects_.end(); ++iter) {
      if (image_det_aug_default_enum::kCenter == crop_emit_mode) {
        float center_x = (iter->left + iter->right) * 0.5f;
        float center_y = (iter->top + iter->bottom) * 0.5f;
        if (!crop_box.contains(cv::Point2f(center_x, center_y))) {
          continue;
        }
        new_objects.push_back(iter->Project(crop_box));
      } else if (image_det_aug_default_enum::kOverlap == crop_emit_mode) {
        Rect gt_box = iter->ToRect();
        float overlap = (crop_box & gt_box).area() / gt_box.area();
        if (overlap > emit_overlap_thresh) {
          new_objects.push_back(iter->Project(crop_box));
        }
      }
    }
    if (new_objects.size() < 1) return false;
    objects_ = new_objects;  // replace the old objects
    return true;
  }

  /*! \brief try pad image with given pad_box
   * convert all objects afterwards
   */
  bool TryPad(const Rect pad_box) {
    // update all objects inplace
    for (auto it = objects_.begin(); it != objects_.end(); ++it) {
      *it = it->Project(pad_box);
    }
    return true;
  }

  /*! \brief flip image and object coordinates horizontally */
  bool TryMirror() {
    // flip all objects horizontally
    for (auto it = objects_.begin(); it != objects_.end(); ++it) {
      *it = it->HorizontalFlip();
    }
    return true;
  }

 private:
  /*! \brief width for each object information, 5 at least */
  int object_width_;
  /*! \brief vector to store original header info */
  std::vector<float> header_;
  /*! \brief storing objects in more convenient formats */
  std::vector<ImageDetObject> objects_;
};  // class ImageDetLabel

/*! \brief helper class to do image augmentation */
class DefaultImageDetAugmenter : public ImageAugmenter {
 public:
  // contructor
  DefaultImageDetAugmenter() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);

    CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
     (param_.inter_method >= 9 && param_.inter_method <= 10))
      << "invalid inter_method: valid value 0,1,2,3,9,10";

    // validate crop parameters
    ValidateCropParameters(&param_.min_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.min_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(&param_.max_crop_trials, param_.num_crop_sampler);
    for (int i = 0; i < param_.num_crop_sampler; ++i) {
      CHECK_GE(param_.min_crop_scales[i], 0.0f);
      CHECK_LE(param_.max_crop_scales[i], 1.0f);
      CHECK_GT(param_.max_crop_scales[i], param_.min_crop_scales[i]);
      CHECK_GE(param_.min_crop_aspect_ratios[i], 0.0f);
      CHECK_GE(param_.max_crop_aspect_ratios[i], param_.min_crop_aspect_ratios[i]);
      CHECK_GE(param_.max_crop_overlaps[i], param_.min_crop_overlaps[i]);
      CHECK_GE(param_.max_crop_sample_coverages[i], param_.min_crop_sample_coverages[i]);
      CHECK_GE(param_.max_crop_object_coverages[i], param_.min_crop_object_coverages[i]);
    }
    CHECK_GE(param_.emit_overlap_thresh, 0.0f);
  }
  /*!
   * \brief get interpolation method with given inter_method, 0-CV_INTER_NN 1-CV_INTER_LINEAR 2-CV_INTER_CUBIC
   * \ 3-CV_INTER_AREA 4-CV_INTER_LANCZOS4 9-AUTO(cubic for enlarge, area for shrink, bilinear for others) 10-RAND
   */
  int GetInterMethod(int inter_method, int old_width, int old_height, int new_width,
    int new_height, common::RANDOM_ENGINE *prnd) {
    if (inter_method == 9) {
      if (new_width > old_width && new_height > old_height) {
        return 2;  // CV_INTER_CUBIC for enlarge
      } else if (new_width <old_width && new_height < old_height) {
        return 3;  // CV_INTER_AREA for shrink
      } else {
        return 1;  // CV_INTER_LINEAR for others
      }
      } else if (inter_method == 10) {
      std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
      return rand_uniform_int(*prnd);
    } else {
      return inter_method;
    }
  }

  /*! \brief Check number of crop samplers and given parameters */
  template<typename DType>
  void ValidateCropParameters(nnvm::Tuple<DType> *param, const int num_sampler) {
    if (num_sampler == 1) {
      CHECK_EQ(param->ndim(), 1);
    } else if (num_sampler > 1) {
      if (param->ndim() == 1) {
        std::vector<DType> vec(num_sampler, (*param)[0]);
        param->assign(vec.begin(), vec.end());
      } else {
        CHECK_EQ(param->ndim(), num_sampler) << "# of parameters/crop_samplers mismatch ";
      }
    }
  }

  /*! \brief Generate crop box region given cropping parameters */
  Rect GenerateCropBox(const float min_crop_scale,
    const float max_crop_scale, const float min_crop_aspect_ratio,
    const float max_crop_aspect_ratio, common::RANDOM_ENGINE *prnd,
    const float img_aspect_ratio) {
    float new_scale = std::uniform_real_distribution<float>(
        min_crop_scale, max_crop_scale)(*prnd) + 1e-12f;
    float min_ratio = std::max<float>(min_crop_aspect_ratio / img_aspect_ratio,
        new_scale * new_scale);
    float max_ratio = std::min<float>(max_crop_aspect_ratio / img_aspect_ratio,
        1. / (new_scale * new_scale));
    float new_ratio = std::sqrt(std::uniform_real_distribution<float>(
        min_ratio, max_ratio)(*prnd));
    float new_width = std::min(1.f, new_scale * new_ratio);
    float new_height = std::min(1.f, new_scale / new_ratio);
    float x0 = std::uniform_real_distribution<float>(0.f, 1 - new_width)(*prnd);
    float y0 = std::uniform_real_distribution<float>(0.f, 1 - new_height)(*prnd);
    return Rect(x0, y0, new_width, new_height);
  }

  /*! \brief Generate padding box region given padding parameters */
  Rect GeneratePadBox(const float max_pad_scale,
    common::RANDOM_ENGINE *prnd, const float threshold = 1.05f) {
      float new_scale = std::uniform_real_distribution<float>(
        1.f, max_pad_scale)(*prnd);
      if (new_scale < threshold) return Rect(0, 0, 0, 0);
      auto rand_uniform = std::uniform_real_distribution<float>(0.f, new_scale - 1);
      float x0 = rand_uniform(*prnd);
      float y0 = rand_uniform(*prnd);
      return Rect(-x0, -y0, new_scale, new_scale);
    }

  cv::Mat Process(const cv::Mat &src, std::vector<float> *label,
                  common::RANDOM_ENGINE *prnd) override {
    using mshadow::index_t;
    cv::Mat res;
    if (param_.resize != -1) {
      int new_height, new_width;
      if (src.rows > src.cols) {
        new_height = param_.resize*src.rows/src.cols;
        new_width = param_.resize;
      } else {
        new_height = param_.resize;
        new_width = param_.resize*src.cols/src.rows;
      }
      int interpolation_method = GetInterMethod(param_.inter_method,
                   src.cols, src.rows, new_width, new_height, prnd);
      cv::resize(src, res, cv::Size(new_width, new_height),
                   0, 0, interpolation_method);
    } else {
      res = src;
    }

    // build a helper class for processing labels
    ImageDetLabel det_label(*label);
    // random engine
    std::uniform_real_distribution<float> rand_uniform(0, 1);

    // color space augmentation
    if (param_.random_hue_prob > 0.f || param_.random_saturation_prob > 0.f ||
        param_.random_illumination_prob > 0.f || param_.random_contrast_prob > 0.f) {
      std::uniform_real_distribution<float> uniform_range(-1.f, 1.f);
      int h = uniform_range(*prnd) * param_.max_random_hue;
      int s = uniform_range(*prnd) * param_.max_random_saturation;
      int l = uniform_range(*prnd) * param_.max_random_illumination;
      float c = uniform_range(*prnd) * param_.max_random_contrast;
      h = rand_uniform(*prnd) < param_.random_hue_prob ? h : 0;
      s = rand_uniform(*prnd) < param_.random_saturation_prob ? s : 0;
      l = rand_uniform(*prnd) < param_.random_illumination_prob ? l : 0;
      c = rand_uniform(*prnd) < param_.random_contrast_prob ? c : 0;
      if (h != 0 || s != 0 || l != 0) {
        int temp[3] = {h, l, s};
        int limit[3] = {180, 255, 255};
        cv::cvtColor(res, res, CV_BGR2HLS);
        for (int i = 0; i < res.rows; ++i) {
          for (int j = 0; j < res.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
              int v = res.at<cv::Vec3b>(i, j)[k];
              v += temp[k];
              v = std::max(0, std::min(limit[k], v));
              res.at<cv::Vec3b>(i, j)[k] = v;
            }
          }
        }
        cv::cvtColor(res, res, CV_HLS2BGR);
      }
      if (fabs(c) > 1e-3) {
        cv::Mat tmp = res;
        tmp.convertTo(res, -1, c + 1.f, 0);
      }
    }

    // random mirror logic
    if (param_.rand_mirror_prob > 0 && rand_uniform(*prnd) < param_.rand_mirror_prob) {
      if (det_label.TryMirror()) {
        // flip image
        cv::flip(res, temp_, 1);
        res = temp_;
      }
    }

    // random padding logic
    if (param_.rand_pad_prob > 0 && param_.max_pad_scale > 1.f) {
      if (rand_uniform(*prnd) < param_.rand_pad_prob) {
        Rect pad_box = GeneratePadBox(param_.max_pad_scale, prnd);
        if (pad_box.area() > 0) {
          if (det_label.TryPad(pad_box)) {
            // pad image
            temp_ = res;
            int left = static_cast<int>(-pad_box.x * res.cols);
            int top = static_cast<int>(-pad_box.y * res.rows);
            int right = static_cast<int>((pad_box.width + pad_box.x - 1) * res.cols);
            int bot = static_cast<int>((pad_box.height + pad_box.y - 1) * res.rows);
            cv::copyMakeBorder(temp_, res, top, bot, left, right, cv::BORDER_ISOLATED,
              cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
          }
        }
      }
    }

    // random crop logic
    if (param_.rand_crop_prob > 0 && param_.num_crop_sampler > 0) {
      if (rand_uniform(*prnd) < param_.rand_crop_prob) {
        // random crop sampling logic: randomly pick a sampler, return if success
        // continue to next sampler if failed(exceed max_trial)
        // return original sample if every sampler has failed
        std::vector<int> indices(param_.num_crop_sampler);
        for (int i = 0; i < param_.num_crop_sampler; ++i) {
          indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), *prnd);
        int num_processed = 0;
        for (auto idx : indices) {
          if (num_processed > 0) break;
          for (int t = 0; t < param_.max_crop_trials[idx]; ++t) {
            Rect crop_box = GenerateCropBox(param_.min_crop_scales[idx],
              param_.max_crop_scales[idx], param_.min_crop_aspect_ratios[idx],
              param_.max_crop_aspect_ratios[idx], prnd,
              static_cast<float>(res.cols) / res.rows);
            if (det_label.TryCrop(crop_box, param_.min_crop_overlaps[idx],
                param_.max_crop_overlaps[idx], param_.min_crop_sample_coverages[idx],
                param_.max_crop_sample_coverages[idx], param_.min_crop_object_coverages[idx],
                param_.max_crop_object_coverages[idx], param_.crop_emit_mode,
                param_.emit_overlap_thresh)) {
              ++num_processed;
              // crop image
              int left = static_cast<int>(crop_box.x * res.cols);
              int top = static_cast<int>(crop_box.y * res.rows);
              int width = static_cast<int>(crop_box.width * res.cols);
              int height = static_cast<int>(crop_box.height * res.rows);
              res = res(cv::Rect(left, top, width, height));
              break;
            }
          }
        }
      }
    }

    if (image_det_aug_default_enum::kForce == param_.resize_mode) {
      // force resize to specified data_shape, regardless of aspect ratio
      int new_height = param_.data_shape[1];
      int new_width = param_.data_shape[2];
      int interpolation_method = GetInterMethod(param_.inter_method,
                   res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, res, cv::Size(new_width, new_height),
                   0, 0, interpolation_method);
    } else if (image_det_aug_default_enum::kShrink == param_.resize_mode) {
      // try to keep original size, shrink if too large
      float h = param_.data_shape[1];
      float w = param_.data_shape[2];
      if (res.rows > h || res.cols > w) {
        float ratio = std::min(h / res.rows, w / res.cols);
        int new_height = ratio * res.rows;
        int new_width = ratio * res.cols;
        int interpolation_method = GetInterMethod(param_.inter_method,
                     res.cols, res.rows, new_width, new_height, prnd);
        cv::resize(res, res, cv::Size(new_width, new_height),
                    0, 0, interpolation_method);
      }
    } else if (image_det_aug_default_enum::kFit == param_.resize_mode) {
      float h = param_.data_shape[1];
      float w = param_.data_shape[2];
      float ratio = std::min(h / res.rows, w / res.cols);
      int new_height = ratio * res.rows;
      int new_width = ratio * res.cols;
      int interpolation_method = GetInterMethod(param_.inter_method,
                   res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, res, cv::Size(new_width, new_height),
                  0, 0, interpolation_method);
    }

    *label = det_label.ToArray();  // put back processed labels
    return res;
  }

 private:
  // temporal space
  cv::Mat temp_;
  // parameters
  DefaultImageDetAugmentParam param_;
};

MXNET_REGISTER_IMAGE_AUGMENTER(det_aug_default)
.describe("default detection augmenter")
.set_body([]() {
    return new DefaultImageDetAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
