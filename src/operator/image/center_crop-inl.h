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
 * \file centor_crop-inl.h
 * \brief the image centor crop operator implementation
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
  mxnet::Tuple<int> size;
  int interp;
  DMLC_DECLARE_PARAMETER(CenterCropParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(mxnet::Tuple<int>())
    .describe("Size of output image. Could be (width, height) or (size)");
  }
};

// Scales down crop size if it's larger than image size.
inline ImageSize ScaleDown(const ImageSize& src,
                           const ImageSize& size) {
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
  return ImageSize(dst_h, dst_w);
}

inline bool CenterCropShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const CenterCropParam& param = nnvm::get<CenterCropParam>(attrs.parsed);

  ImageSize param_size(param.size);
  ImageSize input_size(ishape[ishape.ndim() - 3], ishape[ishape.ndim() - 2]);
  ImageSize new_size = ScaleDown(input_size, param_size);

  CHECK((param_size.height > 0) && (param_size.width > 0))
    << "Input height and width must be greater than 0";

  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0,
      TShape({new_size.height, new_size.width, ishape[ishape.ndim() - 1]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0,
      TShape({ishape[0], new_size.height, new_size.width, ishape[ishape.ndim() - 1]}));
  }
  return true;
}

template<typename xpu>
inline void CenterCropOpForward(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<TBlob> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  CHECK((inputs[0].ndim() == 3) || (inputs[0].ndim() == 4))
    << "Input data must be (h, w, c) or (n, h, w, c)";
  const TBlob& data = inputs[0];
  const auto& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const ImageSize input_size(data.size((data.ndim() == 3) ? 0 : 1),
    data.size((data.ndim() == 3) ? 1 : 2));
  const ImageSize size(param.size);
  const ImageSize new_size = ScaleDown(input_size, size);

  const int x = static_cast<int>((input_size.width - new_size.width) / 2);
  const int y = static_cast<int>((input_size.height - new_size.height) / 2);
  CropImpl<xpu>(x, y, new_size.width, new_size.height, inputs, outputs, ctx, req);
}

template<typename xpu>
inline void CenterCropOpBackward(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<TBlob> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const auto& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const TBlob& input_grad = outputs[0];
  const ImageSize input_size(input_grad.size((input_grad.ndim() == 3) ? 0 : 1),
    input_grad.size((input_grad.ndim() == 3) ? 1 : 2));
  const ImageSize size(param.size);
  const ImageSize new_size = ScaleDown(input_size, size);

  const int x = static_cast<int>((input_size.width - new_size.width) / 2);
  const int y = static_cast<int>((input_size.height - new_size.height) / 2);
  CropBackwardImpl<xpu>(x, y, new_size.width, new_size.height, inputs, outputs, ctx, req);
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_
