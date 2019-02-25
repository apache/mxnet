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
#include "../../common/static_array.h"
#include "../tensor/matrix_op-inl.h"
#include "resize-inl.h"

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

inline void CropImpl(int x,
                      int y,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const OpContext &ctx,
                      const std::vector<OpReqType> &req) {
  using namespace mshadow;
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    Stream<cpu>* s = ctx.get_stream<cpu>();
    common::StaticArray<index_t, ndim> begin = {0}, step = {1};
    if (ndim == 3) {
      begin[0] = y;
      begin[1] = x;
    } else {
      begin[1] = y;
      begin[2] = x;
    }
    Tensor<cpu, ndim, float> input_tensor = data.get<cpu, ndim, float>(s);
    Tensor<cpu, ndim, float> output_tensor = out.get<cpu, ndim, float>(s);
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      size_t num_threads = out.shape_.FlatTo2D()[0];
      mxnet_op::Kernel<slice_forward<ndim, Req, cpu>, cpu>::Launch(s, num_threads,
          output_tensor.dptr_, input_tensor.dptr_, 
          input_tensor.shape_, output_tensor.shape_, begin, step);
    })
  })
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
    CropImpl(param.x, param.y, inputs, outputs, ctx, req);
    if (need_resize) {
      ResizeImpl(inputs, outputs, size.height, size.width, param.interp.value());
    }
  } else {
    CropImpl(param.x, param.y, inputs, outputs, ctx, req);
    if (need_resize) {
      const auto batch_size = inputs[0].shape_[0];
      const auto input_step = inputs[0].shape_[1] * inputs[0].shape_[2] * inputs[0].shape_[3];
      int output_step = size.height * size.width * outputs[0].shape_[3];
      #pragma omp parallel for
      for (auto i = 0; i < batch_size; ++i) {
        ResizeImpl(inputs, outputs, size.height, size.width, param.interp.value(),
        i * input_step, i * output_step);
      }
    }    
  }
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CROP_INL_H_
