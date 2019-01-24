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
 *  Copyright (c) 2018 by Contributors
 * \file image_utils.h
 * \brief the image operator utility function implementation
 */

#ifndef MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_

#include <vector>
#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace op {
namespace image {

enum ImageLayout {H, W, C};
enum BatchImageLayout {N, kH, kW, kC};

struct SizeParam {
  int height;
  int width;
  SizeParam() {
    height = 0;
    width = 0;
  }
  SizeParam(int height_, int width_) {
    height = height_;
    width = width_;
  }
};  // struct SizeParam

template<typename T>
inline SizeParam GetHeightAndWidthFromSize(const T& param) {
  int h, w;
  if (param.size.ndim() == 1) {
    h = param.size[0];
    w = param.size[0];
  } else {
    // size should be (w, h) instead of (h, w)
    h = param.size[1];
    w = param.size[0];
  }
  return SizeParam(h, w);
}

// Scales down crop size if it's larger than image size.
inline SizeParam ScaleDown(const SizeParam& src,
                            const SizeParam& size) {
  const auto src_h = src.height;
  const auto src_w = src.width;
  auto dst_h = size.height;
  auto dst_w = size.width;

  if (src_h < dst_h) {
    dst_w = static_cast<int>((dst_w * src_h) / dst_h);
    dst_h = src_h;
  }
  if (src_w < dst_w) {
    dst_h = static_cast<int>((dst_h * src_w) / dst_w);
    dst_w = src_w;
  }
  return SizeParam(dst_h, dst_w);
}

inline void CropImpl(const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      int x,
                      int y,
                      int height,
                      int width,
                      const SizeParam &size,
                      int interp,
                      int input_index = 0,
                      int output_index = 0) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "opencv doesn't support fp16";
  // mapping to opencv matrix element type according to channel
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_],
    inputs[0].shape_[(inputs[0].ndim() == 3) ? C : kC]);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    cv::Mat buf(inputs[0].shape_[(inputs[0].ndim() == 3) ? H : kH],
      inputs[0].shape_[(inputs[0].ndim() == 3) ? W : kW], cv_type,
      inputs[0].dptr<DType>() + input_index);
    cv::Mat dst(outputs[0].shape_[(inputs[0].ndim() == 3) ? H : kH],
      outputs[0].shape_[(inputs[0].ndim() == 3) ? W : kW], cv_type,
      outputs[0].dptr<DType>() + output_index);
    cv::Rect roi(x, y, width, height);
    if ((size.height != 0 && size.width != 0) &&
      ((height != size.height) || (width != size.width))) {
      cv::resize(buf(roi), dst, cv::Size(size.width, size.height), 0, 0, interp);
    } else {
      buf(roi).copyTo(dst);
    }
    CHECK(!dst.empty());
    CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr<DType>() + output_index);
  });
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image crop operator.";
#endif  // MXNET_USE_OPENCV
}

inline void CenterCropImpl(const std::vector<TBlob> &inputs,
                            const std::vector<TBlob> &outputs,
                            const SizeParam &size,
                            int interp) {
  int h, w;
  if (inputs[0].ndim() == 3) {
    h = inputs[0].shape_[0];
    w = inputs[0].shape_[1];
  } else {
    h = inputs[0].shape_[1];
    w = inputs[0].shape_[2];
  }
  const auto new_size = ScaleDown(SizeParam(h, w), size);
  const auto x0 = static_cast<int>((w - new_size.width) / 2);
  const auto y0 = static_cast<int>((h - new_size.height) / 2);
  if (inputs[0].ndim() == 3) {
    CropImpl(inputs, outputs, x0, y0, new_size.height, new_size.width, size, interp);
  } else {
    const auto batch_size = inputs[0].shape_[N];
    const auto input_step = inputs[0].shape_[kH] * inputs[0].shape_[kW] * inputs[0].shape_[kC];
    int output_step;
    if ((new_size.height != size.height) || (new_size.width != size.width)) {
      output_step = size.height * size.width * outputs[0].shape_[kC];
    } else {
      output_step = new_size.height * new_size.width * outputs[0].shape_[kC];
    }
    #pragma omp parallel for
    for (auto i = 0; i < batch_size; ++i) {
      CropImpl(inputs, outputs, x0, y0, new_size.height, new_size.width,
        size, interp, input_step * i, output_step * i);
    }
  }
}

}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_UTILS_H_
