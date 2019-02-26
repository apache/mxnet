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
 *  Copyright (c) 2019 by Contributors
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

namespace mxnet {
namespace op {
namespace image {

struct CropParam : public dmlc::Parameter<CropParam> {
  int x;
  int y;
  int width;
  int height;
  DMLC_DECLARE_PARAMETER(CropParam) {
    DMLC_DECLARE_FIELD(x)
    .describe("Left boundary of the cropping area.");
    DMLC_DECLARE_FIELD(y)
    .describe("Top boundary of the cropping area.");
    DMLC_DECLARE_FIELD(width)
    .describe("Width of the cropping area.");
    DMLC_DECLARE_FIELD(height)
    .describe("Top boundary of the cropping area");
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

  CHECK((param.height > 0) && (param.width > 0))
      << "Input height and width must be greater than 0";
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({param.height, param.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], param.height, param.width, ishape[kC]}));
  }
  return true;
}

inline void CropImpl(int x,
                      int y,
                      int width,
                      int height,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const OpContext &ctx,
                      const std::vector<OpReqType> &req) {
  using namespace mshadow;
  // invalid param
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    CHECK(x + width <= data.shape_[ndim - 2])
        << " x + width should not be greater than input width";
    CHECK(y + height <= data.shape_[ndim - 3])
        << " y + height should not be greater than input height";
    Stream<cpu>* s = ctx.get_stream<cpu>();
    common::StaticArray<index_t, ndim> begin = {0}, step = {1};
    if (ndim == 3) {
      begin[0] = y;
      begin[1] = x;
    } else {
      begin[1] = y;
      begin[2] = x;
    }
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      Tensor<cpu, ndim, DType> input_tensor = data.get<cpu, ndim, DType>(s);
      Tensor<cpu, ndim, DType> output_tensor = out.get<cpu, ndim, DType>(s);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = out.shape_.FlatTo2D()[0];
        mxnet_op::Kernel<slice_forward<ndim, Req, cpu>, cpu>::Launch(s, num_threads,
          output_tensor.dptr_, input_tensor.dptr_,
          input_tensor.shape_, output_tensor.shape_, begin, step);
      })
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
  CHECK((param.height > 0) && (param.width > 0))
      << "Input height and width must be greater than 0";

  CropImpl(param.x, param.y, param.width, param.height, inputs, outputs, ctx, req);
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CROP_INL_H_
