/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_detection_aug_default.cc
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

#if MXNET_USE_OPENCV
// Registers
// namespace dmlc {
// DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
// }  // namespace dmlc
#endif

namespace mxnet {
namespace io {

namespace image_detection_aug_default_enum {
enum ImageDetectionAugDefaultCropEmitMode {kCenter, kOverlap};
}

using nnvm::Tuple;
using Rect = cv::Rect_<float>;
/*! \brief image augmentation parameters*/
struct DefaultImageDetectionAugmentParam : public dmlc::Parameter<DefaultImageDetectionAugmentParam> {
  /*! \brief resize shorter edge to size before applying other augmentations */
  int resize;
  /*! \brief whether we do random cropping */
  float rand_crop_prob;
  /*! \brief where to nonrandom crop on y */
  Tuple<float> min_crop_scales;
  /*! \brief where to nonrandom crop on x */
  Tuple<float> max_crop_scales;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  Tuple<float> min_crop_aspect_ratios;
  /*! \brief max aspect ratio */
  Tuple<float> max_crop_aspect_ratios;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  Tuple<float> min_crop_overlaps;
  /*! \brief max crop size */
  Tuple<float> max_crop_overlaps;
  /*! \brief min crop size */
  Tuple<float> min_crop_sample_coverages;
  /*! \brief max scale ratio */
  Tuple<float> max_crop_sample_coverages;
  /*! \brief min scale_ratio */
  Tuple<float> min_crop_object_coverages;
  /*! \brief min image size */
  Tuple<float> max_crop_object_coverages;
  int num_crop_sampler;
  int crop_emit_mode;
  float emit_overlap_thresh;
  /*! \brief */
  Tuple<int> max_crop_trials;
  /*! \brief random padding prob */
  float rand_pad_prob;
  /*!< \brief maximum padding scale */
  float max_pad_scale;
  // int max_img_size;
  // int min_img_size;
  /*! \brief max image size */

  /*! \brief max random in H channel */
  int max_random_hue;
  float random_hue_prob;
  /*! \brief max random in S channel */
  int max_random_saturation;
  float random_saturation_prob;
  /*! \brief max random in L channel */
  int max_random_illumination;
  float random_illumination_prob;
  float max_random_contrast;
  float random_contrast_prob;
  /*! \brief rotate angle */
  float rand_mirror_prob;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief shape of the image data*/
  TShape data_shape;

  /*! \brief mean value for r channel */
  float mean_r;
  /*! \brief mean value for g channel */
  float mean_g;
  /*! \brief mean value for b channel */
  float mean_b;
  /*! \brief mean value for alpha channel */
  float mean_a;
  /*! \brief scale on color space */
  float std_r;
  float std_g;
  float std_b;
  float std_a;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DefaultImageDetectionAugmentParam) {
    DMLC_DECLARE_FIELD(resize).set_default(-1)
        .describe("Augmentation Param: scale shorter edge to size "
                  "before applying other augmentations, -1 to disable.");
    DMLC_DECLARE_FIELD(rand_crop_prob).set_default(0.0f)
        .describe("Augmentation Param: Whether to random crop on the image");
    DMLC_DECLARE_FIELD(min_crop_scales).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Where to nonrandom crop on y.");
    DMLC_DECLARE_FIELD(max_crop_scales).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Where to nonrandom crop on x.");
    DMLC_DECLARE_FIELD(min_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: rotated randomly in [-max_rotate_angle, max_rotate_angle].");
    DMLC_DECLARE_FIELD(max_crop_aspect_ratios).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: denotes the max ratio of random aspect ratio augmentation.");
    DMLC_DECLARE_FIELD(min_crop_overlaps).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: denotes the max random shearing ratio.");
    DMLC_DECLARE_FIELD(max_crop_overlaps).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum crop size.");
    DMLC_DECLARE_FIELD(min_crop_sample_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum crop size.");
    DMLC_DECLARE_FIELD(max_crop_sample_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Maximum scale ratio.");
    DMLC_DECLARE_FIELD(min_crop_object_coverages).set_default(Tuple<float>({0.0f}))
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(max_crop_object_coverages).set_default(Tuple<float>({1.0f}))
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(num_crop_sampler).set_default(1)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(crop_emit_mode)
        .add_enum("center", image_detection_aug_default_enum::kCenter)
        .add_enum("overlap", image_detection_aug_default_enum::kOverlap)
        .set_default(image_detection_aug_default_enum::kCenter)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(emit_overlap_thresh).set_default(0.5f)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(max_crop_trials).set_default(Tuple<int>({25}))
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(rand_pad_prob).set_default(0.0f)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(max_pad_scale).set_default(1.0f)
        .describe("Augmentation Param: Minimum scale ratio.");
    // DMLC_DECLARE_FIELD(max_img_size).set_default(1e10)
    //     .describe("Augmentation Param: Maximum image size after resizing.");
    // DMLC_DECLARE_FIELD(min_img_size).set_default(0.0)
    //     .describe("Augmentation Param: Minimum image size after resizing.");
    DMLC_DECLARE_FIELD(max_random_hue).set_default(0)
        .describe("Augmentation Param: Maximum random value of H channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_hue_prob).set_default(0.0f)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(max_random_saturation).set_default(0)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_saturation_prob).set_default(0.0f)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(max_random_illumination).set_default(0)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_illumination_prob).set_default(0.0f)
        .describe("Augmentation Param: Maximum random value of L channel in HSL color space.");
    DMLC_DECLARE_FIELD(max_random_contrast).set_default(0)
        .describe("Augmentation Param: Maximum random value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_contrast_prob).set_default(0.0f)
        .describe("Augmentation Param: Maximum random value of L channel in HSL color space.");
    DMLC_DECLARE_FIELD(rand_mirror_prob).set_default(0.0f)
        .describe("Augmentation Param: Rotate angle.");
    DMLC_DECLARE_FIELD(fill_value).set_default(127)
        .describe("Augmentation Param: Filled color value while padding.");
    DMLC_DECLARE_FIELD(inter_method).set_default(1)
        .describe("Augmentation Param: 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3)
        .describe("Dataset Param: Shape of each instance generated by the DataIter.");
    DMLC_DECLARE_FIELD(mean_r).set_default(0.0f)
        .describe("Augmentation Param: Mean value on R channel.");
    DMLC_DECLARE_FIELD(mean_g).set_default(0.0f)
        .describe("Augmentation Param: Mean value on G channel.");
    DMLC_DECLARE_FIELD(mean_b).set_default(0.0f)
        .describe("Augmentation Param: Mean value on B channel.");
    DMLC_DECLARE_FIELD(mean_a).set_default(0.0f)
        .describe("Augmentation Param: Mean value on Alpha channel.");
    DMLC_DECLARE_FIELD(std_r).set_default(0.0f)
        .describe("Augmentation Param: Mean value on R channel.");
    DMLC_DECLARE_FIELD(std_g).set_default(0.0f)
        .describe("Augmentation Param: Mean value on G channel.");
    DMLC_DECLARE_FIELD(std_b).set_default(0.0f)
        .describe("Augmentation Param: Mean value on B channel.");
    DMLC_DECLARE_FIELD(std_a).set_default(0.0f)
        .describe("Augmentation Param: Mean value on Alpha channel.");
  }
};

DMLC_REGISTER_PARAMETER(DefaultImageDetectionAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListDefaultDetectionAugParams() {
  return DefaultImageDetectionAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV

#ifdef _MSC_VER
#define M_PI CV_PI
#endif

// struct Rect {
//   float left;
//   float top;
//   float right;
//   float bottom;
//   Rect(float l, float t, float r, float b):
//     left(l), top(t), right(r), bottom(b) {
//     CHECK_GE(r, l);
//     CHECK_GE(b, t);
//   }
// };
//
// class BBox : public cv::Rect_<float> {
// public:
//   float overlap(const BBox &r) const {
//     float inter = (*this & r).area();
//     if (inter <= 0.f) return 0.f;
//     return inter / (area() + r.area() - inter);
//   }
// };

/*! \brief helper class for better detection label handling */
class ImageDetectionLabel {
 public:
   struct ImageDetectionObject {
     float id;
     float left;
     float top;
     float right;
     float bottom;
     std::vector<float> extra;
     Rect ToRect() const {
       return Rect(left, top, right - left, bottom - top);
     }

     ImageDetectionObject Project(Rect box) const {
       ImageDetectionObject ret = *this;
       ret.left = (ret.left - box.x) / box.width;
       ret.top = (ret.top - box.y) / box.height;
       ret.right = (ret.right - box.x) / box.width;
       ret.bottom = (ret.bottom - box.y) / box.height;
       return ret;
     }
   };
   explicit ImageDetectionLabel(std::vector<float> &raw_label) {
     FromArray(raw_label);
   }

   void FromArray(std::vector<float> &raw_label) {
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
       ImageDetectionObject obj;
       obj.id = raw_label[i++];
       obj.left = raw_label[i++];
       obj.top = raw_label[i++];
       obj.right = raw_label[i++];
       obj.bottom = raw_label[i++];
       obj.extra.assign(raw_label.begin() + i, raw_label.begin() + i + object_width_ - 5);
       objects_.push_back(obj);
       CHECK_GT(obj.right, obj.left);
       CHECK_GT(obj.bottom, obj.top);
     }
   }

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

  //  static float RectIntersect(Rect a, Rect b) {
  //    float w = std::min(a.right, b.right) - std::max(a.left, b.left);
  //    if (w < 0) return 0.f;
  //    float h = std::min(a.bottom, b.bottom) - std::max(a.top, b.top);
  //    if (h < 0) return 0.f;
  //    return w * h;
  //  }
   //
  //  static float RectSize(Rect a) {
  //    return std::max(0.f, a.right - a.left) * std::max(0.f, a.bottom - a.top);
  //  }
   //
  //  static float RectUnion(Rect a, Rect b) {
  //    return RectSize(a) + RectSize(b) - RectIntersect(a, b);
  //  }
   //
   static float RectOverlap(Rect a, Rect b) {
     float intersect = (a & b).area();
     if (intersect <= 0.f) return 0.f;
     return intersect / (a.area() + b.area() - intersect);
   }

   bool TryCrop(Rect crop_box,
     const float min_crop_overlap, const float max_crop_overlap,
     const float min_crop_sample_coverage, const float max_crop_sample_coverage,
     const float min_crop_object_coverage, const float max_crop_object_coverage,
     const int crop_emit_mode, const float emit_overlap_thresh) {
    if (objects_.size() < 1) {
      return true;  // no object, raise error or just skip?
    }
    // check if crop_box valid
    bool valid = false;
    if (min_crop_overlap > 0.f && max_crop_overlap < 1.f &&
        min_crop_sample_coverage > 0.f && max_crop_sample_coverage < 1.f &&
        min_crop_object_coverage > 0.f && max_crop_object_coverage < 1.f) {
      for (auto& obj : objects_) {
        Rect gt_box = obj.ToRect();
        if (min_crop_overlap > 0.f || max_crop_overlap < 1.f) {
          float ovp = RectOverlap(crop_box, gt_box);
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
    std::vector<ImageDetectionObject> new_objects;
    for (auto iter = objects_.begin(); iter != objects_.end(); ++iter) {
      if (image_detection_aug_default_enum::kCenter == crop_emit_mode) {
        float center_x = (iter->left + iter->right) * 0.5f;
        float center_y = (iter->top + iter->bottom) * 0.5f;
        if (!crop_box.contains(cv::Point2f(center_x, center_y))) {
          continue;
        }
        new_objects.push_back(iter->Project(crop_box));
      } else if (image_detection_aug_default_enum::kOverlap == crop_emit_mode) {
        float overlap = RectOverlap(crop_box, iter->ToRect());
        if (overlap > emit_overlap_thresh) {
          new_objects.push_back(iter->Project(crop_box));
        }
      }
    }
    if (new_objects.size() < 1) return false;
    objects_ = new_objects;  // replace the old objects
    return true;
   }

 private:
   /*! \brief  */
   int object_width_;
   std::vector<float> header_;
   std::vector<ImageDetectionObject> objects_;
};  // class ImageDetectionLabel

/*! \brief helper class to do image augmentation */
class DefaultImageDetectionAugmenter : public ImageAugmenter {
 public:
  // contructor
  DefaultImageDetectionAugmenter() {
    rotateM_ = cv::Mat(2, 3, CV_32F);
  }
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);
    for (size_t i = 0; i < kwargs_left.size(); i++) {
        if (!strcmp(kwargs_left[i].first.c_str(), "rotate_list")) {
          const char* val = kwargs_left[i].second.c_str();
          const char *end = val + strlen(val);
          char buf[128];
          while (val < end) {
            sscanf(val, "%[^,]", buf);
            val += strlen(buf) + 1;
            rotate_list_.push_back(atoi(buf));
          }
        }
    }
    // validate crop parameters
    ValidateCropParameters(param_.min_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_scales, param_.num_crop_sampler);
    ValidateCropParameters(param_.min_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_aspect_ratios, param_.num_crop_sampler);
    ValidateCropParameters(param_.min_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_overlaps, param_.num_crop_sampler);
    ValidateCropParameters(param_.min_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_sample_coverages, param_.num_crop_sampler);
    ValidateCropParameters(param_.min_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_object_coverages, param_.num_crop_sampler);
    ValidateCropParameters(param_.max_crop_trials, param_.num_crop_sampler);
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

  template<typename DType>
  void ValidateCropParameters(nnvm::Tuple<DType> &param, const int num_sampler) {
    if (num_sampler == 1) {
      CHECK_EQ(param.ndim(), 1);
    } else if (num_sampler > 1) {
      if (param.ndim() == 1) {
        std::vector<DType> vec(num_sampler, param[0]);
        param.assign(vec.begin(), vec.end());
      } else {
        CHECK_EQ(param.ndim(), num_sampler) << "# of parameters/crop_samplers mismatch ";
      }
    }
  }

  Rect GenerateCropBox(const float min_crop_scale,
    const float max_crop_scale, const float min_crop_aspect_ratio,
    const float max_crop_aspect_ratio, common::RANDOM_ENGINE *prnd) {
    float new_scale = std::uniform_real_distribution<float>(
      min_crop_scale, max_crop_scale)(*prnd) + 1e-12f;
    float min_ratio = std::max<float>(min_crop_aspect_ratio, new_scale * new_scale);
    float max_ratio = std::min<float>(max_crop_aspect_ratio, new_scale * new_scale);
    float new_ratio = std::sqrt(std::uniform_real_distribution<float>(
      min_ratio, max_ratio)(*prnd));
    float new_width = new_scale * new_ratio;
    float new_height = new_scale / new_ratio;
    float x0 = std::uniform_real_distribution<float>(0.f, 1 - new_width)(*prnd);
    float y0 = std::uniform_real_distribution<float>(0.f, 1 - new_height)(*prnd);
    return Rect(x0, y0, new_width, new_height);
  }

  cv::Mat Process(const cv::Mat &src, std::vector<float> &label,
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
      CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
       (param_.inter_method >= 9 && param_.inter_method <= 10))
        << "invalid inter_method: valid value 0,1,2,3,9,10";
      int interpolation_method = GetInterMethod(param_.inter_method,
                   src.cols, src.rows, new_width, new_height, prnd);
      cv::resize(src, res, cv::Size(new_width, new_height),
                   0, 0, interpolation_method);
    } else {
      res = src;
    }

    // normal augmentation by affine transformation.
    // if (param_.max_rotate_angle > 0 || param_.max_shear_ratio > 0.0f
    //     || param_.rotate > 0 || rotate_list_.size() > 0 || param_.max_random_scale != 1.0
    //     || param_.min_random_scale != 1.0 || param_.max_aspect_ratio != 0.0f
    //     || param_.max_img_size != 1e10f || param_.min_img_size != 0.0f) {
    //   std::uniform_real_distribution<float> rand_uniform(0, 1);
    //   // shear
    //   float s = rand_uniform(*prnd) * param_.max_shear_ratio * 2 - param_.max_shear_ratio;
    //   // rotate
    //   int angle = std::uniform_int_distribution<int>(
    //       -param_.max_rotate_angle, param_.max_rotate_angle)(*prnd);
    //   if (param_.rotate > 0) angle = param_.rotate;
    //   if (rotate_list_.size() > 0) {
    //     angle = rotate_list_[std::uniform_int_distribution<int>(0, rotate_list_.size() - 1)(*prnd)];
    //   }
    //   float a = cos(angle / 180.0 * M_PI);
    //   float b = sin(angle / 180.0 * M_PI);
    //   // scale
    //   float scale = rand_uniform(*prnd) *
    //       (param_.max_random_scale - param_.min_random_scale) + param_.min_random_scale;
    //   // aspect ratio
    //   float ratio = rand_uniform(*prnd) *
    //       param_.max_aspect_ratio * 2 - param_.max_aspect_ratio + 1;
    //   float hs = 2 * scale / (1 + ratio);
    //   float ws = ratio * hs;
    //   // new width and height
    //   float new_width = std::max(param_.min_img_size,
    //                              std::min(param_.max_img_size, scale * res.cols));
    //   float new_height = std::max(param_.min_img_size,
    //                               std::min(param_.max_img_size, scale * res.rows));
    //   cv::Mat M(2, 3, CV_32F);
    //   M.at<float>(0, 0) = hs * a - s * b * ws;
    //   M.at<float>(1, 0) = -b * ws;
    //   M.at<float>(0, 1) = hs * b + s * a * ws;
    //   M.at<float>(1, 1) = a * ws;
    //   float ori_center_width = M.at<float>(0, 0) * res.cols + M.at<float>(0, 1) * res.rows;
    //   float ori_center_height = M.at<float>(1, 0) * res.cols + M.at<float>(1, 1) * res.rows;
    //   M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
    //   M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
    //   CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
    //     (param_.inter_method >= 9 && param_.inter_method <= 10))
    //      << "invalid inter_method: valid value 0,1,2,3,9,10";
    //   int interpolation_method = GetInterMethod(param_.inter_method,
    //                 res.cols, res.rows, new_width, new_height, prnd);
    //   cv::warpAffine(res, temp_, M, cv::Size(new_width, new_height),
    //                  interpolation_method,
    //                  cv::BORDER_CONSTANT,
    //                  cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
    //   res = temp_;
    // }

    // pad logic
    // if (param_.pad > 0) {
    //   cv::copyMakeBorder(res, res, param_.pad, param_.pad, param_.pad, param_.pad,
    //                      cv::BORDER_CONSTANT,
    //                      cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
    // }

    // crop logic
    // if (param_.max_crop_size != -1 || param_.min_crop_size != -1) {
    //   CHECK(res.cols >= param_.max_crop_size && res.rows >= \
    //           param_.max_crop_size && param_.max_crop_size >= param_.min_crop_size)
    //       << "input image size smaller than max_crop_size";
    //   index_t rand_crop_size =
    //       std::uniform_int_distribution<index_t>(param_.min_crop_size, param_.max_crop_size)(*prnd);
    //   index_t y = res.rows - rand_crop_size;
    //   index_t x = res.cols - rand_crop_size;
    //   if (param_.rand_crop != 0) {
    //     y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
    //     x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
    //   } else {
    //     y /= 2; x /= 2;
    //   }
    //   cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
    //   int interpolation_method = GetInterMethod(param_.inter_method, rand_crop_size, rand_crop_size,
    //                                             param_.data_shape[2], param_.data_shape[1], prnd);
    //   cv::resize(res(roi), res, cv::Size(param_.data_shape[2], param_.data_shape[1])
    //             , 0, 0, interpolation_method);
    // } else {
    //   CHECK(static_cast<index_t>(res.rows) >= param_.data_shape[1]
    //         && static_cast<index_t>(res.cols) >= param_.data_shape[2])
    //       << "input image size smaller than input shape";
    //   index_t y = res.rows - param_.data_shape[1];
    //   index_t x = res.cols - param_.data_shape[2];
    //   if (param_.rand_crop != 0) {
    //     y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
    //     x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
    //   } else {
    //     y /= 2; x /= 2;
    //   }
    //   cv::Rect roi(x, y, param_.data_shape[2], param_.data_shape[1]);
    //   res = res(roi);
    // }

    // build a helper class for processing labels
    ImageDetectionLabel det_label(label);

    // random crop logic
    if (param_.rand_crop_prob > 0 && param_.num_crop_sampler > 0) {
      std::uniform_real_distribution<float> rand_uniform(0, 1);
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
              param_.max_crop_aspect_ratios[idx], prnd);
            if (det_label.TryCrop(crop_box, param_.min_crop_overlaps[idx],
                param_.max_crop_overlaps[idx], param_.min_crop_sample_coverages[idx],
                param_.max_crop_sample_coverages[idx], param_.min_crop_object_coverages[idx],
                param_.max_crop_object_coverages[idx], param_.crop_emit_mode,
                param_.emit_overlap_thresh)) {
              ++num_processed;
              // crop image
              int left = static_cast<int>(crop_box.x * src.cols);
              int top = static_cast<int>(crop_box.y * src.rows);
              int width = static_cast<int>(crop_box.width * src.cols);
              int height = static_cast<int>(crop_box.height * src.rows);
              res = res(cv::Rect(left, top, width, height));
              break;
            }
          }
        }
      }
    }

    // random padding logic
    if (param_.rand_pad_prob > 0 && param_.max_pad_scale > 1.f) {

    }

    // color space augmentation
    // if (param_.random_h != 0 || param_.random_s != 0 || param_.random_l != 0) {
    //   std::uniform_real_distribution<float> rand_uniform(0, 1);
    //   cvtColor(res, res, CV_BGR2HLS);
    //   int h = rand_uniform(*prnd) * param_.random_h * 2 - param_.random_h;
    //   int s = rand_uniform(*prnd) * param_.random_s * 2 - param_.random_s;
    //   int l = rand_uniform(*prnd) * param_.random_l * 2 - param_.random_l;
    //   int temp[3] = {h, l, s};
    //   int limit[3] = {180, 255, 255};
    //   for (int i = 0; i < res.rows; ++i) {
    //     for (int j = 0; j < res.cols; ++j) {
    //       for (int k = 0; k < 3; ++k) {
    //         int v = res.at<cv::Vec3b>(i, j)[k];
    //         v += temp[k];
    //         v = std::max(0, std::min(limit[k], v));
    //         res.at<cv::Vec3b>(i, j)[k] = v;
    //       }
    //     }
    //   }
    //   cvtColor(res, res, CV_HLS2BGR);
    // }
    label = det_label.ToArray();  // put back processed labels
    return res;
  }

 private:
  // temporal space
  cv::Mat temp_;
  // rotation param
  cv::Mat rotateM_;
  // parameters
  DefaultImageDetectionAugmentParam param_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
};

// ImageAugmenter* ImageAugmenter::Create(const std::string& name) {
//   return dmlc::Registry<ImageAugmenterReg>::Find(name)->body();
// }

MXNET_REGISTER_IMAGE_AUGMENTER(detection_aug_default)
.describe("default detection augmenter")
.set_body([]() {
    return new DefaultImageDetectionAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
