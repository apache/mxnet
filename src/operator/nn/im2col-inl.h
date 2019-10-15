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
 * \file im2col-inl.h
 * \brief
 * \author Jiajun Wang
*/

#ifndef MXNET_OPERATOR_NN_IM2COL_INL_H_
#define MXNET_OPERATOR_NN_IM2COL_INL_H_
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./im2col.h"

namespace mxnet {
namespace op {

struct Im2colParam : public dmlc::Parameter<Im2colParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  DMLC_DECLARE_PARAMETER(Im2colParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("Convolution padding: (w,), (h, w) or (d, h, w). Defaults to no padding.");
  }

  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }
};  // struct Im2colParam


template<typename xpu>
void Im2colCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape in_shape = inputs[0].shape_;
  const mxnet::TShape out_shape = outputs[0].shape_;
  const index_t batch_size = in_shape[0];
  const index_t input_dim = in_shape.ProdShape(1, in_shape.ndim());
  const index_t output_dim = out_shape.ProdShape(1, out_shape.ndim());

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = out_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = in_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    for (index_t n = 0; n < batch_size; ++n) {
      im2col(s, inputs[0].dptr<DType>() + n * input_dim, in_shape,
             col_buffer_shape, param.kernel, param.pad, param.stride, param.dilate,
             outputs[0].dptr<DType>() + n * output_dim);
    }
  });
}

template<typename xpu>
void Im2colGradCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &out_grad = inputs[0];
  const TBlob &in_grad = outputs[0];

  const mxnet::TShape in_shape = in_grad.shape_;
  const mxnet::TShape out_shape = out_grad.shape_;
  const index_t batch_size = in_shape[0];
  const index_t input_dim = in_shape.ProdShape(1, in_shape.ndim());
  const index_t output_dim = out_shape.ProdShape(1, out_shape.ndim());

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = out_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = in_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    for (index_t n = 0; n < batch_size; ++n) {
      col2im(s, out_grad.dptr<DType>() + n * output_dim, in_grad.shape_, col_buffer_shape,
             param.kernel, param.pad, param.stride, param.dilate,
             in_grad.dptr<DType>() + n * input_dim, req[0]);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_IM2COL_INL_H_
