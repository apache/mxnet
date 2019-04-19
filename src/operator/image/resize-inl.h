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
* \file resize-inl.h
* \brief image resize operator using opencv and only support bilinear resize
* \author Jake Lee
*/
#ifndef MXNET_OPERATOR_IMAGE_RESIZE_INL_H_
#define MXNET_OPERATOR_IMAGE_RESIZE_INL_H_

#include <mxnet/base.h>
#include <vector>

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "image_utils.h"

#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
namespace op {
namespace image {

using namespace mshadow;

#if MXNET_USE_CUDA
template<typename DType, typename T, typename Acctype>
void ResizeImplCUDA(Stream<gpu> *s,
                      const T input,
                      const T output);
#endif  // MXNET_USE_CUDA

struct ResizeParam : public dmlc::Parameter<ResizeParam> {
  mxnet::Tuple<int> size;
  bool keep_ratio;
  int interp;
  DMLC_DECLARE_PARAMETER(ResizeParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(mxnet::Tuple<int>())
    .describe("Size of new image. Could be (width, height) or (size)");
    DMLC_DECLARE_FIELD(keep_ratio)
    .describe("Whether to resize the short edge or both edges to `size`, "
      "if size is give as an integer.")
    .set_default(false);
    DMLC_DECLARE_FIELD(interp)
    .set_default(1)
    .describe("Interpolation method for resizing. By default uses bilinear interpolation"
        "Options are INTER_NEAREST - a nearest-neighbor interpolation"
        "INTER_LINEAR - a bilinear interpolation"
        "INTER_AREA - resampling using pixel area relation"
        "INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood"
        "INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood"
        "Note that the GPU version only support bilinear interpolation(1)"
        " and the result on cpu would be slightly different from gpu."
        "It uses opencv resize function which tend to align center on cpu"
        "while using contrib.bilinearResize2D which aligns corner on gpu");
  }
};
// handle the keep ratio param
inline SizeParam GetHeightAndWidth(int data_h,
                                    int data_w,
                                    const ResizeParam& param) {
  CHECK((param.size.ndim() == 1) || (param.size.ndim() == 2))
      << "Input size dimension must be 1 or 2, but got "
      << param.size.ndim();
  int resized_h;
  int resized_w;
  if (param.size.ndim() == 1) {
    CHECK_GT(param.size[0], 0)
      << "Input size should be greater than 0, but got "
      << param.size[0];
    if (!param.keep_ratio) {
      resized_h = param.size[0];
      resized_w = param.size[0];
    } else {
      if (data_h > data_w) {
        resized_w = param.size[0];
        resized_h = static_cast<int>(data_h * resized_w / data_w);
      } else {
        resized_h = param.size[0];
        resized_w = static_cast<int>(data_w * resized_h / data_h);
      }
    }
  } else {
    CHECK_GT(param.size[0], 0)
        << "Input width should be greater than 0, but got "
        << param.size[0];
    CHECK_GT(param.size[1], 0)
        << "Input height should be greater than 0, but got "
        << param.size[1];
    resized_h = param.size[1];
    resized_w = param.size[0];
  }
  return SizeParam(resized_h, resized_w);
}

inline bool ResizeShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  SizeParam size;
  if (ishape.ndim() == 3) {
    size = GetHeightAndWidth(ishape[H], ishape[W], param);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({size.height, size.width, ishape[C]}));
  } else {
    size = GetHeightAndWidth(ishape[kH], ishape[kW], param);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0,
      mxnet::TShape({ishape[N], size.height, size.width, ishape[kC]}));
  }
  return true;
}

inline void ResizeImpl(const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const int height,
                      const int width,
                      const int interp,
                      const int input_index = 0,
                      const int output_index = 0) {
#if MXNET_USE_OPENCV
  CHECK_NE(inputs[0].type_flag_, mshadow::kFloat16) << "opencv image mat doesn't support fp16";
  CHECK((inputs[0].type_flag_ != mshadow::kInt32) || (inputs[0].type_flag_ != mshadow::kInt64))
      << "opencv resize doesn't support int32, int64";
  // mapping to opencv matrix element type according to channel
  const int DTYPE[] = {CV_32F, CV_64F, -1, CV_8U, CV_32S};
  if (inputs[0].ndim() == 3) {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[C]);
    cv::Mat buf(inputs[0].shape_[H], inputs[0].shape_[W], cv_type, inputs[0].dptr_);
    cv::Mat dst(outputs[0].shape_[H], outputs[0].shape_[W], cv_type, outputs[0].dptr_);
    cv::resize(buf, dst, cv::Size(width, height), 0, 0, interp);
    CHECK(!dst.empty());
    CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr_);
  } else {
    const int cv_type = CV_MAKETYPE(DTYPE[inputs[0].type_flag_], inputs[0].shape_[kC]);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      cv::Mat buf(inputs[0].shape_[kH], inputs[0].shape_[kW], cv_type,
        inputs[0].dptr<DType>() + input_index);
      cv::Mat dst(outputs[0].shape_[kH], outputs[0].shape_[kW], cv_type,
        outputs[0].dptr<DType>() + output_index);
      cv::resize(buf, dst, cv::Size(width, height), 0, 0, interp);
      CHECK(!dst.empty());
      CHECK_EQ(static_cast<void*>(dst.ptr()), outputs[0].dptr<DType>() + output_index);
    });
  }
#else
  LOG(FATAL) << "Build with USE_OPENCV=1 for image resize operator.";
#endif  // MXNET_USE_OPENCV
}

template <typename xpu>
inline void Resize(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  SizeParam size;
  if (std::is_same<xpu, gpu>::value) {
#if MXNET_USE_CUDA
    CHECK(param.interp == 1) << "interp should be 1 for using Resize on GPU.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      if (inputs[0].ndim() == 3) {
        Tensor<gpu, 3, DType> input = inputs[0].get<gpu, 3, DType>(s);
        Tensor<gpu, 3, DType> output = outputs[0].get<gpu, 3, DType>(s);
        ResizeImplCUDA<DType, Tensor<gpu, 3, DType>, float>(s, input, output);
      } else {
        Tensor<gpu, 4, DType> input = inputs[0].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> output = outputs[0].get<gpu, 4, DType>(s);
        ResizeImplCUDA<DType, Tensor<gpu, 4, DType>, float>(s, input, output);
      }
    });
#endif  // MXNET_USE_CUDA
  } else if (inputs[0].ndim() == 3) {
    size = GetHeightAndWidth(inputs[0].shape_[H], inputs[0].shape_[W], param);
    ResizeImpl(inputs, outputs, size.height, size.width, param.interp);
  } else {
    size = GetHeightAndWidth(inputs[0].shape_[kH], inputs[0].shape_[kW], param);
    const auto batch_size = inputs[0].shape_[N];
    const auto input_step = inputs[0].shape_[kH] * inputs[0].shape_[kW] * inputs[0].shape_[kC];
    const auto output_step = size.height * size.width * inputs[0].shape_[kC];
    #pragma omp parallel for
    for (auto i = 0; i < batch_size; ++i) {
      ResizeImpl(inputs, outputs, size.height, size.width,
        param.interp, i * input_step, i * output_step);
    }
  }
}

}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_RESIZE_INL_H_
