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

#ifndef MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_
#define MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_


#include <algorithm>
#include <vector>

#include "mxnet/base.h"
#include "dmlc/optional.h"

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "crop-inl.h"
#include "image_utils.h"

namespace mxnet {
namespace op {
namespace image {

struct CenterCropParam : public dmlc::Parameter<CenterCropParam> {
  nnvm::Tuple<int> size;
  int interp;
  DMLC_DECLARE_PARAMETER(CenterCropParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(nnvm::Tuple<int>())
    .describe("Size of output image. Could be (width, height) or (size)");
    DMLC_DECLARE_FIELD(interp)
    .describe("Interpolation method for resizing. By default uses bilinear"
        "interpolation. See OpenCV's resize function for available choices.");
  }
};

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

bool CenterCropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const CenterCropParam& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const auto size = GetHeightAndWidthFromSize(param);

  CHECK((size.height > 0) && (size.width > 0))
      << "Input height and width must be greater than 0";
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({size.height, size.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], size.height, size.width, ishape[kC]}));
  }

  return true;
}

void CenterCrop(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  CHECK((inputs[0].ndim() == 3) || (inputs[0].ndim() == 4))
      << "Input data must be (h, w, c) or (n, h, w, c)";
  const CenterCropParam& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const auto size = GetHeightAndWidthFromSize(param);
  auto need_resize = false;
  int h, w;
  if (inputs[0].ndim() == 3) {
    h = inputs[0].shape_[0];
    w = inputs[0].shape_[1];
  } else {
    h = inputs[0].shape_[1];
    w = inputs[0].shape_[2];
  }
  const auto new_size = ScaleDown(SizeParam(h, w), size);
  if ((new_size.height != size.height) || (new_size.width != size.width)) {
    need_resize = true;
  }
  const auto x0 = static_cast<int>((w - new_size.width) / 2);
  const auto y0 = static_cast<int>((h - new_size.height) / 2);
  if (inputs[0].ndim() == 3) {
    if (need_resize) {
      CropImpl(x0, y0, new_size.height, new_size.width, inputs, outputs, size, param.interp);
    } else {
      CropImpl(x0, y0, new_size.height, new_size.width, inputs, outputs);
    }
  } else {
    const auto batch_size = inputs[0].shape_[0];
    const auto input_offset = inputs[0].shape_[kH] * inputs[0].shape_[kW] * inputs[0].shape_[kC];
    int output_offset;
    if (need_resize) {
      output_offset = size.height * size.width * outputs[0].shape_[kC];
    } else {
      output_offset = new_size.height * new_size.width * outputs[0].shape_[kC];
    }
    #pragma omp parallel for
    for (auto i = 0; i < batch_size; ++i) {
      if (need_resize) {
        CropImpl(x0, y0, new_size.height, new_size.width, inputs, outputs,
          size, param.interp, input_offset * i, output_offset * i);
      } else {
        CropImpl(x0, y0, new_size.height, new_size.width, inputs, outputs,
          input_offset * i, output_offset * i);
      }
    }
  }
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_
