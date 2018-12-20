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
 *  Copyright (c) 2016 by Contributors
 * \file crop-inl.h
 * \brief the image crop operator implementation
 */

#ifndef MXNET_OPERATOR_IMAGE_CROP_INL_H_
#define MXNET_OPERATOR_IMAGE_CROP_INL_H_


#include <algorithm>
#include <vector>

#include "mxnet/base.h"
#include "dmlc/optional.h"
#include "image_utils.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace op {
namespace image {

struct CropParam : public dmlc::Parameter<CropParam> {
  int x;
  int y;
  int width;
  int height;
  dmlc::optional<nnvm::Tuple<int>> size;
  dmlc::optional<int> interp;
  DMLC_DECLARE_PARAMETER(CropParam) {
    DMLC_DECLARE_FIELD(x)
    .describe("Left boundary of the cropping area.");
    DMLC_DECLARE_FIELD(y)
    .describe("Top boundary of the cropping area.");
    DMLC_DECLARE_FIELD(width)
    .describe("Width of the cropping area.");
    DMLC_DECLARE_FIELD(height)
    .describe("Top boundary of the cropping area");
    DMLC_DECLARE_FIELD(size)
    .describe("Size of image for resizing after crop. Could be (width, height) or size");
    DMLC_DECLARE_FIELD(interp)
    .describe("Interpolation method for resizing. By default uses bilinear interpolation"
        "Options are INTER_NEAREST - a nearest-neighbor interpolation"
        "INTER_LINEAR - a bilinear interpolation"
        "INTER_AREA - resampling using pixel area relation"
        "INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood"
        "INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood");
  }
};

inline bool CropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();

  const auto& ishape = (*in_attrs)[0];
  const CropParam& param = nnvm::get<CropParam>(attrs.parsed);
  const bool has_size = param.size.has_value();
  const bool has_interp = param.interp.has_value();

  CHECK(has_size == has_interp)
      << "Missing input. Either size or interp is not assigned the value.";
  CHECK((param.height > 0) && (param.width > 0))
      << "Input height and width must be greater than 0";
  if (has_size) {
    int height;
    int width;
    CHECK(param.size.value().ndim() == 1 || param.size.value().ndim() == 2)
        << "Input size must be int size or (width, height)";
    if (param.size.value().ndim() == 1) {
      CHECK_GE(param.size.value()[0], 0) << "Input size must be greater than 0";
      height = param.size.value()[0];
      width = param.size.value()[0];
    } else {
      CHECK((param.size.value()[0] >0) && (param.size.value()[1] > 0))
          << "Input width and height must be greater than 0";
      height = param.size.value()[1];
      width = param.size.value()[0];
    }
    if (ishape.ndim() == 3) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({height, width, ishape[C]}));
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], height, width, ishape[kC]}));
    }
  } else {
    if (ishape.ndim() == 3) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({param.height, param.width, ishape[C]}));
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], param.height, param.width, ishape[kC]}));
    }
  }
  return true;
}

// overloading version of CropImpl to do resize after crop
inline void CropImpl(int x,
                      int y,
                      int height,
                      int width,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      int input_index = 0,
                      int output_index = 0) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "opencv doesn't support fp16";
  // mapping to opencv matrix element type according to channel
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  if (inputs[0].ndim() == 3) {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[2]);
    cv::Mat buf(inputs[0].shape_[H], inputs[0].shape_[W], cv_type, inputs[0].dptr_);
    cv::Mat dst(outputs[0].shape_[H], outputs[0].shape_[W], cv_type, outputs[0].dptr_);
    cv::Rect roi(x, y, width, height);
    buf(roi).copyTo(dst);
    CHECK(!dst.empty());
    CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
  } else {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[3]);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      cv::Mat buf(inputs[0].shape_[kH], inputs[0].shape_[kW], cv_type,
        inputs[0].dptr<DType>() + input_index);
      cv::Mat dst(outputs[0].shape_[kH], outputs[0].shape_[kW], cv_type,
        outputs[0].dptr<DType>() + output_index);
      cv::Rect roi(x, y, width, height);
      buf(roi).copyTo(dst);
      CHECK(!dst.empty());
      CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr<DType>() + output_index);
    });
  }
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image crop operator.";
#endif  // MXNET_USE_OPENCV
}

inline void CropImpl(int x,
                      int y,
                      int height,
                      int width,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const SizeParam &size,
                      int interp,
                      int input_index = 0,
                      int output_index = 0) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "opencv doesn't support fp16";
  // mapping to opencv matrix element type according to channel
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  if (inputs[0].ndim() == 3) {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[2]);
    cv::Mat buf(inputs[0].shape_[H], inputs[0].shape_[W], cv_type, inputs[0].dptr_);
    cv::Mat dst(outputs[0].shape_[H], outputs[0].shape_[W], cv_type, outputs[0].dptr_);
    cv::Rect roi(x, y, width, height);
    cv::resize(buf(roi), dst, cv::Size(size.width, size.height), 0, 0, interp);
    CHECK(!dst.empty());
    CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
  } else {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[3]);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      cv::Mat buf(inputs[0].shape_[kH], inputs[0].shape_[kW], cv_type,
        inputs[0].dptr<DType>() + input_index);
      cv::Mat dst(outputs[0].shape_[kH], outputs[0].shape_[kW], cv_type,
        outputs[0].dptr<DType>() + output_index);
      cv::Rect roi(x, y, width, height);
      cv::resize(buf(roi), dst, cv::Size(size.width, size.height), 0, 0, interp);
      CHECK(!dst.empty());
      CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr<DType>() + output_index);
    });
  }
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image crop operator.";
#endif  // MXNET_USE_OPENCV
}

inline void Crop(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const CropParam& param = nnvm::get<CropParam>(attrs.parsed);
  const bool has_size = param.size.has_value();
  auto need_resize = false;
  SizeParam size;
  CHECK((param.height > 0) && (param.width > 0))
      << "Input height and width must be greater than 0";
  if (has_size) {
    CHECK(param.size.value().ndim() == 1 || param.size.value().ndim() == 2)
        << "Input size must be int size or (width, height)";
    if (param.size.value().ndim() == 1) {
      CHECK_GT(param.size.value()[0], 0)
        << "Input size should greater than 0, but got "
        << param.size.value()[0];
      size = SizeParam(param.size.value()[0], param.size.value()[0]);
    } else {
      CHECK_GT(param.size.value()[0], 0)
        << "Input width in size should greater than 0, but got "
        << param.size.value()[0];
      CHECK_GT(param.size.value()[1], 0)
        << "Input height in size should greater than 0, but got "
        << param.size.value()[1];
      size = SizeParam(param.size.value()[1], param.size.value()[0]);
    }
    // if size given is not the same as input height, width, resize it.
    if ((param.height != size.height) || (param.width != size.width)) {
      need_resize = true;
    }
  }

  if (inputs[0].ndim() == 3) {
    if (need_resize) {
      CropImpl(param.x, param.y, param.height, param.width,
        inputs, outputs, size, param.interp.value());
    } else {
      CropImpl(param.x, param.y, param.height, param.width, inputs, outputs);
    }
  } else {
    const auto batch_size = inputs[0].shape_[0];
    const auto input_offset = inputs[0].shape_[1] * inputs[0].shape_[2] * inputs[0].shape_[3];
    int output_offset;
    if (need_resize) {
      output_offset = size.height * size.width * outputs[0].shape_[3];
    } else {
      output_offset = param.width * param.height * outputs[0].shape_[3];
    }
    #pragma omp parallel for
    for (auto i = 0; i < batch_size; ++i) {
      if (need_resize) {
        CropImpl(param.x, param.y, param.height, param.width,
          inputs, outputs, size, param.interp.value(), input_offset * i, output_offset * i);
      } else {
        CropImpl(param.x, param.y, param.height, param.width,
          inputs, outputs, input_offset * i, output_offset * i);
      }
    }
  }
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CROP_INL_H_
