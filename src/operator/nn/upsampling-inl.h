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
 * Copyright (c) 2015 by Contributors
 * \file upsampling-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_UPSAMPLING_INL_H_
#define MXNET_OPERATOR_NN_UPSAMPLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./deconvolution-inl.h"

namespace mxnet {
namespace op {

namespace up_enum {
enum UpSamplingOpInputs {kData, kWeight};
enum UpSamplingOpOutputs {kOut};
enum UpSamplingType {kNearest, kBilinear};
enum UpSamplingMultiInputMode {kConcat, kSum};
}  // namespace up_enum

struct UpSamplingParam : public dmlc::Parameter<UpSamplingParam> {
  int scale;
  int num_filter;
  int sample_type;
  int num_args;
  int multi_input_mode;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(UpSamplingParam) {
    DMLC_DECLARE_FIELD(scale)
    .set_range(1, 1000)
    .describe("Up sampling scale");
    DMLC_DECLARE_FIELD(num_filter)
    .describe("Input filter. Only used by bilinear sample_type.")
    .set_default(0);
    DMLC_DECLARE_FIELD(sample_type)
    .add_enum("nearest", up_enum::kNearest)
    .add_enum("bilinear", up_enum::kBilinear)
    .describe("upsampling method");
    DMLC_DECLARE_FIELD(multi_input_mode)
    .add_enum("concat", up_enum::kConcat)
    .add_enum("sum", up_enum::kSum)
    .set_default(up_enum::kConcat)
    .describe("How to handle multiple input. concat means concatenate upsampled "
    "images along the channel dimension. sum means add all images together, "
    "only available for nearest neighbor upsampling.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be upsampled. For nearest neighbor "
    "upsampling, this can be 1-N; the size of output will be"
    "(scale*h_0,scale*w_0) and all other inputs will be upsampled to the"
    "same size. For bilinear upsampling this must be 2; 1 input and 1 weight.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
    .describe("Tmp workspace for deconvolution (MB)");
  }
};  // struct UpSamplingParam

template<typename xpu, typename DType>
void UpSamplingForward(const OpContext &ctx, const UpSamplingParam &param,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(in_data.size(), static_cast<size_t>(param.num_args));
  CHECK_EQ(out_data.size(), 1U);
  if (req[up_enum::kOut] == kNullOp) {
    return;
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4, DType> out = out_data[up_enum::kOut].get<xpu, 4, DType>(s);
  if (param.num_args > 1) {
    int begin = 0;
    for (int i = 0; i < param.num_args; ++i) {
      Tensor<xpu, 4, DType> data = in_data[i].get<xpu, 4, DType>(s);
      int end = begin + data.size(1);
      int scale = out_data[up_enum::kOut].size(2)/in_data[i].size(2);
      if (param.multi_input_mode == up_enum::kSum) {
        if (i == 0) {
          Assign(out, req[up_enum::kOut], upsampling_nearest(data, scale));
        } else {
          out += upsampling_nearest(data, scale);
        }
      } else {
        Assign(slice<1>(out, begin, end), req[up_enum::kOut], upsampling_nearest(data, scale));
      }
      begin = end;
    }
  } else {
    Tensor<xpu, 4, DType> data = in_data[up_enum::kData].get<xpu, 4, DType>(s);
    Assign(out, req[up_enum::kOut], upsampling_nearest(data, param.scale));
  }
}

template<typename xpu, typename DType>
void UpSamplingBackward(const OpContext &ctx, const UpSamplingParam &param,
                        const TBlob &out_grad, const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(in_grad.size(), static_cast<size_t>(param.num_args));
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4, DType> grad = out_grad.get<xpu, 4, DType>(s);
  if (param.num_args > 1) {
    int begin = 0;
    for (int i = 0; i < param.num_args; ++i) {
      Tensor<xpu, 4, DType> input_grad = in_grad[i].get<xpu, 4, DType>(s);
      mshadow::Shape<2> in_shape = Shape2(input_grad.shape_[2], input_grad.shape_[3]);
      int end = begin + input_grad.size(1);
      int scale = grad.size(2)/in_shape[0];
      if (param.multi_input_mode == up_enum::kSum) {
        Assign(input_grad, req[i],
               pool<mshadow::red::sum>(grad,
                                       in_shape,
                                       scale,
                                       scale,
                                       scale,
                                       scale));
      } else {
        Assign(input_grad, req[i],
               pool<mshadow::red::sum>(slice<1>(grad, begin, end),
                                       in_shape,
                                       scale,
                                       scale,
                                       scale,
                                       scale));
      }
      begin = end;
    }
  } else {
    Tensor<xpu, 4, DType> input_grad = in_grad[up_enum::kData].get<xpu, 4, DType>(s);
    mshadow::Shape<2> in_shape = Shape2(input_grad.shape_[2], input_grad.shape_[3]);
    Assign(input_grad, req[up_enum::kData],
           pool<mshadow::red::sum>(grad,
                                   in_shape,
                                   param.scale,
                                   param.scale,
                                   param.scale,
                                   param.scale));
  }
}

static inline DeconvolutionParam GetDeconvolutionParam(const UpSamplingParam& param) {
  DeconvolutionParam p = DeconvolutionParam();
  int kernel = 2 * param.scale - param.scale % 2;
  int stride = param.scale;
  int pad = static_cast<int>(ceil((param.scale - 1) / 2.));
  p.workspace = param.workspace;
  p.num_group = param.num_filter;
  p.num_filter = param.num_filter;
  p.no_bias =  true;
  int shape[] = {1, 1};
  p.dilate = TShape(shape, shape + 2);
  shape[0] = shape[1] = kernel;
  p.kernel = TShape(shape, shape + 2);
  shape[0] = shape[1] = stride;
  p.stride = TShape(shape, shape + 2);
  shape[0] = shape[1] = pad;
  p.pad = TShape(shape, shape + 2);
  return p;
}

template<typename xpu>
void UpSamplingCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx, const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  if (param.sample_type == up_enum::kNearest) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[deconv::kData].type_flag_, DType, {
      UpSamplingForward<xpu, DType>(ctx, param, inputs, req, outputs);
    });
  } else if (param.sample_type == up_enum::kBilinear) {
    DeconvolutionParam p = GetDeconvolutionParam(param);
    _DeconvolutionCompute<xpu>(p, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Unknown sample type";
  }
}

template<typename xpu>
void UpSamplingGradCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  if (param.sample_type == up_enum::kNearest) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[deconv::kData].type_flag_, DType, {
      CHECK_EQ(inputs.size(), 1U);
      UpSamplingBackward<xpu, DType>(ctx, param, inputs[0], req, outputs);
    });
  } else if (param.sample_type == up_enum::kBilinear) {
    DeconvolutionParam p = GetDeconvolutionParam(param);
    _DeconvolutionGradCompute<xpu>(p, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Unknown sample type";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_UPSAMPLING_INL_H_
