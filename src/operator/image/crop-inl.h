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
    .describe("Height of the cropping area.");
  }
};

inline bool CropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  if (in_attrs->at(0).ndim() == 3U) {
    CHECK((in_attrs->at(0)[2] == 1) || (in_attrs->at(0)[2] == 3))
      << "Expect channel of the input image is 1 or 3, but got"
      << in_attrs->at(0)[2];
  } else if (in_attrs->at(0).ndim() == 4U) {
    CHECK((in_attrs->at(0)[3] == 1) || (in_attrs->at(0)[3] == 3))
      << "Expect channel of the input image is 1 or 3, but got"
      << in_attrs->at(0)[3];
  } else {
    LOG(FATAL) << "Image Crop expects inputs of 3D (h, w, c) or 4D (n, h, w, c). But got "
      << in_attrs->at(0).ndim();
  }

  const auto& ishape = (*in_attrs)[0];
  const CropParam& param = nnvm::get<CropParam>(attrs.parsed);

  CHECK((param.height > 0) && (param.width > 0))
    << "Input height and width must be greater than 0";
  CHECK(param.x + param.width <= ishape[ishape.ndim() - 2])
    << " x + width should not be greater than input width";
  CHECK(param.y + param.height <= ishape[ishape.ndim() - 3])
    << " y + height should not be greater than input height";
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({param.height, param.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], param.height, param.width, ishape[kC]}));
  }
  return true;
}

template<typename xpu>
inline void CropImpl(int x,
                      int y,
                      int width,
                      int height,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const OpContext &ctx,
                      const std::vector<OpReqType> &req) {
  using namespace mshadow;
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    Stream<xpu>* s = ctx.get_stream<xpu>();
    common::StaticArray<index_t, ndim> begin = {0}, step = {1};
    if (ndim == 3) {
      begin[0] = y;
      begin[1] = x;
    } else {
      begin[1] = y;
      begin[2] = x;
    }
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = out.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= out.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_forward<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
          out.dptr<DType>(), data.dptr<DType>(),
          data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
inline void CropBackwardImpl(int x,
                      int y,
                      int width,
                      int height,
                      const std::vector<TBlob> &inputs,
                      const std::vector<TBlob> &outputs,
                      const OpContext &ctx,
                      const std::vector<OpReqType> &req) {
  using namespace mshadow;
  if (req[0] == kNullOp) return;
  const TBlob& output_grad = inputs[0];
  const TBlob& input_grad = outputs[0];
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (req[0] == kWriteTo) {
    Fill(s, input_grad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_backward_image_crop does not support kWriteInplace";
  }
  MXNET_NDIM_SWITCH(output_grad.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin = {0}, step = {1};
    if (ndim == 3) {
      begin[0] = y;
      begin[1] = x;
    } else {
      begin[1] = y;
      begin[2] = x;
    }
    MSHADOW_TYPE_SWITCH(output_grad.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = output_grad.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= output_grad.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
          input_grad.dptr<DType>(), output_grad.dptr<DType>(),
          input_grad.shape_.get<ndim>(), output_grad.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
inline void CropOpForward(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const CropParam& param = nnvm::get<CropParam>(attrs.parsed);
  CropImpl<xpu>(param.x, param.y, param.width, param.height, inputs, outputs, ctx, req);
}

template<typename xpu>
inline void CropOpBackward(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  const CropParam& param = nnvm::get<CropParam>(attrs.parsed);
  CropBackwardImpl<xpu>(param.x, param.y, param.width, param.height, inputs, outputs, ctx, req);
}

struct RandomCropParam : public dmlc::Parameter<RandomCropParam> {
  Tuple<float> xrange;
  Tuple<float> yrange;
  int width;
  int height;
  int interp;
  DMLC_DECLARE_PARAMETER(RandomCropParam) {
    DMLC_DECLARE_FIELD(xrange).set_default(Tuple<float>({0.f, 1.f}))
    .describe("Left boundaries of the cropping area.");
    DMLC_DECLARE_FIELD(yrange).set_default(Tuple<float>({0.f, 1.f}))
    .describe("Top boundaries of the cropping area.");
    DMLC_DECLARE_FIELD(width)
    .describe("The target image width");
    DMLC_DECLARE_FIELD(height)
    .describe("The target image height.");
    DMLC_DECLARE_FIELD(interp)
    .set_default(1)
    .describe("Interpolation method for resizing. By default uses bilinear interpolation"
        "Options are INTER_NEAREST - a nearest-neighbor interpolation"
        "INTER_LINEAR - a bilinear interpolation"
        "INTER_AREA - resampling using pixel area relation"
        "INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood"
        "INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood"
        "Note that the GPU version only support bilinear interpolation(1)");
  }
};

inline Tuple<int> GetSourceSize(const TShape& in_shape) {
  Tuple<int> ret;
  if (in_shape.ndim() == 3) {
    ret = Tuple<int>({int(in_shape[W]), int(in_shape[H])});
  } else if (in_shape.ndim() == 4) {
    ret = Tuple<int>({int(in_shape[kW]), int(in_shape[kH])});
  } else {
    LOG(FATAL) << "Image RandomCrop expects inputs of 3D (h, w, c) or 4D (n, h, w, c). But got "
      << in_shape.ndim();
  }
  return ret;
}

inline bool RandomCropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  if (in_attrs->at(0).ndim() == 3U) {
    CHECK((in_attrs->at(0)[2] == 1) || (in_attrs->at(0)[2] == 3))
      << "Expect channel of the input image is 1 or 3, but got"
      << in_attrs->at(0)[2];
  } else if (in_attrs->at(0).ndim() == 4U) {
    CHECK((in_attrs->at(0)[3] == 1) || (in_attrs->at(0)[3] == 3))
      << "Expect channel of the input image is 1 or 3, but got"
      << in_attrs->at(0)[3];
  } else {
    LOG(FATAL) << "Image RandomCrop expects inputs of 3D (h, w, c) or 4D (n, h, w, c). But got "
      << in_attrs->at(0).ndim();
  }

  const auto& ishape = (*in_attrs)[0];
  const RandomCropParam& param = nnvm::get<RandomCropParam>(attrs.parsed);

  CHECK((param.height > 0) && (param.width > 0))
    << "Input height and width must be greater than 0";
  CHECK((param.xrange.ndim() == 2) && (param.yrange.ndim() == 2))
    << "Param xrange and yrange must have two values each";
  CHECK((param.xrange[0] <= param.xrange[1]) && (param.xrange[0] >= 0) and (param.xrange[1] <=1))
    << "Invalid xrange, range should be within 0 and 1.0. Given: " << param.xrange;
  CHECK((param.yrange[0] <= param.yrange[1]) && (param.yrange[0] >= 0) and (param.yrange[1] <=1))
    << "Invalid yrange, range should be within 0 and 1.0. Given: " << param.yrange;

  // real output
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({param.height, param.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], param.height, param.width, ishape[kC]}));
  }

  // temp output
  auto src_shape = GetSourceSize(ishape);
  int sw = src_shape[0];
  int sh = src_shape[1];
  // scale down crop size if it's larger than image size
  float w = param.width;
  float h = param.height;
  if (sh < h) {
    w = w * sh / h;
    h = sh;
  }
  if (sw < w) {
    w = sw;
    h = h * sw / w;
  }
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(
      {static_cast<int>(h), static_cast<int>(w), ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(
      {ishape[N], static_cast<int>(h), static_cast<int>(w), ishape[kC]}));
  }
  return true;
}

template<typename xpu>
inline void RandomCropOpForward(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 2U) << "out, temp";
  CHECK_EQ(inputs.size(), 1U);
  const RandomCropParam& param = nnvm::get<RandomCropParam>(attrs.parsed);
  
  const TShape& dshape = inputs[0].shape_;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Random<cpu> *prnd = ctx.requested[0].get_random<cpu, real_t>(s);
  auto src_size = GetSourceSize(dshape);
  auto resize_size = GetSourceSize(outputs[1].shape_);
  // random left/top position
  float x = std::uniform_real_distribution<float>(
    param.xrange[0], param.xrange[1])(prnd->GetRndEngine()) * (src_size[0] - resize_size[0]);
  float y = std::uniform_real_distribution<float>(
    param.yrange[0], param.yrange[1])(prnd->GetRndEngine()) * (src_size[1] - resize_size[1]);
  // write x, y, w, h to temp workspace
  Tensor<cpu, 1> workspace = ctx.requested[1].get_space<cpu>(
    mshadow::Shape1(4), s);
  workspace.dptr_[0] = x;
  workspace.dptr_[1] = y;
  workspace.dptr_[2] = resize_size[0];
  workspace.dptr_[3] = resize_size[1];
  if (resize_size[0] == src_size[0] && resize_size[1] == src_size[1]) {
    // no need to resize
    CropImpl<xpu>(x, y, resize_size[0], resize_size[1], inputs, outputs, ctx, req);
  } else {
    std::vector<TBlob> hidden_outputs = {outputs[1]};
    CropImpl<xpu>(x, y, resize_size[0], resize_size[1], inputs, hidden_outputs, ctx, req);
    ResizeParam rparam;
    rparam.interp = param.interp;
    rparam.keep_ratio = false;
    rparam.size = Tuple<int>({param.width, param.height});
    ResizeImplWrapper<xpu>(rparam, ctx, hidden_outputs, outputs);
  }
}

template<typename xpu>
inline void RandomCropOpBackward(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs.size(), 2U);
  Tensor<cpu, 1> workspace = ctx.requested[1].get_space<cpu>(
    mshadow::Shape1(4), ctx.get_stream<cpu>());
  auto ptr = workspace.dptr_;
  CropBackwardImpl<xpu>(ptr[0], ptr[1], ptr[2], ptr[3], inputs, outputs, ctx, req);
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CROP_INL_H_
