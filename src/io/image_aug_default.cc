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
 * \file image_aug_default.cc
 * \brief Default augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include "./image_augmenter.h"
#include "../common/utils.h"

#if MXNET_USE_OPENCV
// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
}  // namespace dmlc
#endif

namespace mxnet {
namespace io {

/*! \brief image augmentation parameters*/
struct DefaultImageAugmentParam : public dmlc::Parameter<DefaultImageAugmentParam> {
  /*! \brief resize shorter edge to size before applying other augmentations */
  int resize;
  /*! \brief whether we do random cropping */
  bool rand_crop;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle;
  /*! \brief max aspect ratio */
  float max_aspect_ratio;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio;
  /*! \brief max crop size */
  int max_crop_size;
  /*! \brief min crop size */
  int min_crop_size;
  /*! \brief max scale ratio */
  float max_random_scale;
  /*! \brief min scale_ratio */
  float min_random_scale;
  /*! \brief min image size */
  float min_img_size;
  /*! \brief max image size */
  float max_img_size;
  /*! \brief max random in H channel */
  int random_h;
  /*! \brief max random in S channel */
  int random_s;
  /*! \brief max random in L channel */
  int random_l;
  /*! \brief rotate angle */
  int rotate;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief padding size */
  int pad;
  /*! \brief shape of the image data*/
  TShape data_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DefaultImageAugmentParam) {
    DMLC_DECLARE_FIELD(resize).set_default(-1)
        .describe("Down scale the shorter edge to a new size  "
                  "before applying other augmentations.");
    DMLC_DECLARE_FIELD(rand_crop).set_default(false)
        .describe("If or not randomly crop the image");
    DMLC_DECLARE_FIELD(max_rotate_angle).set_default(0.0f)
        .describe("Rotate by a random degree in ``[-v, v]``");
    DMLC_DECLARE_FIELD(max_aspect_ratio).set_default(0.0f)
        .describe("Change the aspect (namely width/height) to a random value "
                  "in ``[1 - max_aspect_ratio, 1 + max_aspect_ratio]``");
    DMLC_DECLARE_FIELD(max_shear_ratio).set_default(0.0f)
        .describe("Apply a shear transformation (namely ``(x,y)->(x+my,y)``) "
                  "with ``m`` randomly chose from "
                  "``[-max_shear_ratio, max_shear_ratio]``");
    DMLC_DECLARE_FIELD(max_crop_size).set_default(-1)
        .describe("Crop both width and height into a random size in "
                  "``[min_crop_size, max_crop_size]``");
    DMLC_DECLARE_FIELD(min_crop_size).set_default(-1)
        .describe("Crop both width and height into a random size in "
                  "``[min_crop_size, max_crop_size]``");
    DMLC_DECLARE_FIELD(max_random_scale).set_default(1.0f)
        .describe("Resize into ``[width*s, height*s]`` with ``s`` randomly"
                  " chosen from ``[min_random_scale, max_random_scale]``");
    DMLC_DECLARE_FIELD(min_random_scale).set_default(1.0f)
        .describe("Resize into ``[width*s, height*s]`` with ``s`` randomly"
                  " chosen from ``[min_random_scale, max_random_scale]``");
    DMLC_DECLARE_FIELD(max_img_size).set_default(1e10f)
        .describe("Set the maximal width and height after all resize and"
                  " rotate argumentation  are applied");
    DMLC_DECLARE_FIELD(min_img_size).set_default(0.0f)
        .describe("Set the minimal width and height after all resize and"
                  " rotate argumentation  are applied");
    DMLC_DECLARE_FIELD(random_h).set_default(0)
        .describe("Add a random value in ``[-random_h, random_h]`` to "
                  "the H channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_s).set_default(0)
        .describe("Add a random value in ``[-random_s, random_s]`` to "
                  "the S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_l).set_default(0)
        .describe("Add a random value in ``[-random_l, random_l]`` to "
                  "the L channel in HSL color space.");
    DMLC_DECLARE_FIELD(rotate).set_default(-1.0f)
        .describe("Rotate by an angle. If set, it overwrites the ``max_rotate_angle`` option.");
    DMLC_DECLARE_FIELD(fill_value).set_default(255)
        .describe("Set the padding pixes value into ``fill_value``.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("The shape of a output image.");
    DMLC_DECLARE_FIELD(inter_method).set_default(1)
        .describe("The interpolation method: 0-NN 1-bilinear 2-cubic 3-area "
                  "4-lanczos4 9-auto 10-rand.");
    DMLC_DECLARE_FIELD(pad).set_default(0)
        .describe("Change size from ``[width, height]`` into "
                  "``[pad + width + pad, pad + height + pad]`` by padding pixes");
  }
};

DMLC_REGISTER_PARAMETER(DefaultImageAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListDefaultAugParams() {
  return DefaultImageAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV

#ifdef _MSC_VER
#define M_PI CV_PI
#endif
/*! \brief helper class to do image augmentation */
class DefaultImageAugmenter : public ImageAugmenter {
 public:
  // contructor
  DefaultImageAugmenter() {
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
    if (param_.max_rotate_angle > 0 || param_.max_shear_ratio > 0.0f
        || param_.rotate > 0 || rotate_list_.size() > 0 || param_.max_random_scale != 1.0
        || param_.min_random_scale != 1.0 || param_.max_aspect_ratio != 0.0f
        || param_.max_img_size != 1e10f || param_.min_img_size != 0.0f) {
      std::uniform_real_distribution<float> rand_uniform(0, 1);
      // shear
      float s = rand_uniform(*prnd) * param_.max_shear_ratio * 2 - param_.max_shear_ratio;
      // rotate
      int angle = std::uniform_int_distribution<int>(
          -param_.max_rotate_angle, param_.max_rotate_angle)(*prnd);
      if (param_.rotate > 0) angle = param_.rotate;
      if (rotate_list_.size() > 0) {
        angle = rotate_list_[std::uniform_int_distribution<int>(0, rotate_list_.size() - 1)(*prnd)];
      }
      float a = cos(angle / 180.0 * M_PI);
      float b = sin(angle / 180.0 * M_PI);
      // scale
      float scale = rand_uniform(*prnd) *
          (param_.max_random_scale - param_.min_random_scale) + param_.min_random_scale;
      // aspect ratio
      float ratio = rand_uniform(*prnd) *
          param_.max_aspect_ratio * 2 - param_.max_aspect_ratio + 1;
      float hs = 2 * scale / (1 + ratio);
      float ws = ratio * hs;
      // new width and height
      float new_width = std::max(param_.min_img_size,
                                 std::min(param_.max_img_size, scale * res.cols));
      float new_height = std::max(param_.min_img_size,
                                  std::min(param_.max_img_size, scale * res.rows));
      cv::Mat M(2, 3, CV_32F);
      M.at<float>(0, 0) = hs * a - s * b * ws;
      M.at<float>(1, 0) = -b * ws;
      M.at<float>(0, 1) = hs * b + s * a * ws;
      M.at<float>(1, 1) = a * ws;
      float ori_center_width = M.at<float>(0, 0) * res.cols + M.at<float>(0, 1) * res.rows;
      float ori_center_height = M.at<float>(1, 0) * res.cols + M.at<float>(1, 1) * res.rows;
      M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
      M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
      CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
        (param_.inter_method >= 9 && param_.inter_method <= 10))
         << "invalid inter_method: valid value 0,1,2,3,9,10";
      int interpolation_method = GetInterMethod(param_.inter_method,
                    res.cols, res.rows, new_width, new_height, prnd);
      cv::warpAffine(res, temp_, M, cv::Size(new_width, new_height),
                     interpolation_method,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
      res = temp_;
    }

    // pad logic
    if (param_.pad > 0) {
      cv::copyMakeBorder(res, res, param_.pad, param_.pad, param_.pad, param_.pad,
                         cv::BORDER_CONSTANT,
                         cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
    }

    // crop logic
    if (param_.max_crop_size != -1 || param_.min_crop_size != -1) {
      CHECK(res.cols >= param_.max_crop_size && res.rows >= \
              param_.max_crop_size && param_.max_crop_size >= param_.min_crop_size)
          << "input image size smaller than max_crop_size";
      index_t rand_crop_size =
          std::uniform_int_distribution<index_t>(param_.min_crop_size, param_.max_crop_size)(*prnd);
      index_t y = res.rows - rand_crop_size;
      index_t x = res.cols - rand_crop_size;
      if (param_.rand_crop != 0) {
        y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
        x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
      } else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
      int interpolation_method = GetInterMethod(param_.inter_method, rand_crop_size, rand_crop_size,
                                                param_.data_shape[2], param_.data_shape[1], prnd);
      cv::resize(res(roi), res, cv::Size(param_.data_shape[2], param_.data_shape[1])
                , 0, 0, interpolation_method);
    } else {
      CHECK(static_cast<index_t>(res.rows) >= param_.data_shape[1]
            && static_cast<index_t>(res.cols) >= param_.data_shape[2])
          << "input image size smaller than input shape";
      index_t y = res.rows - param_.data_shape[1];
      index_t x = res.cols - param_.data_shape[2];
      if (param_.rand_crop != 0) {
        y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
        x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
      } else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, param_.data_shape[2], param_.data_shape[1]);
      res = res(roi);
    }

    // color space augmentation
    if (param_.random_h != 0 || param_.random_s != 0 || param_.random_l != 0) {
      std::uniform_real_distribution<float> rand_uniform(0, 1);
      cvtColor(res, res, CV_BGR2HLS);
      int h = rand_uniform(*prnd) * param_.random_h * 2 - param_.random_h;
      int s = rand_uniform(*prnd) * param_.random_s * 2 - param_.random_s;
      int l = rand_uniform(*prnd) * param_.random_l * 2 - param_.random_l;
      int temp[3] = {h, l, s};
      int limit[3] = {180, 255, 255};
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
      cvtColor(res, res, CV_HLS2BGR);
    }
    return res;
  }

 private:
  // temporal space
  cv::Mat temp_;
  // rotation param
  cv::Mat rotateM_;
  // parameters
  DefaultImageAugmentParam param_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
};

ImageAugmenter* ImageAugmenter::Create(const std::string& name) {
  return dmlc::Registry<ImageAugmenterReg>::Find(name)->body();
}

MXNET_REGISTER_IMAGE_AUGMENTER(aug_default)
.describe("default augmenter")
.set_body([]() {
    return new DefaultImageAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
