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
#include <vector>
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
    DMLC_DECLARE_FIELD(kernel).describe("Sliding kernel size: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("The stride between adjacent sliding blocks in spatial dimension: "
              "(w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w). "
              "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("The zero-value padding size on both sides of spatial dimension: "
              "(w,), (h, w) or (d, h, w). Defaults to no padding.");
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
  const mxnet::TShape im_shape = inputs[0].shape_;
  const mxnet::TShape col_shape = outputs[0].shape_;
  const index_t num = im_shape[0];

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = col_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = im_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 4, DType> im = inputs[0].get_with_shape<xpu, 4, DType>(
        Shape4(im_shape[0], im_shape[1], im_shape[2], im_shape[3]), s);
    Tensor<xpu, 3, DType> col = outputs[0].get_with_shape<xpu, 3, DType>(
        Shape3(col_shape[0], col_shape[1], col_shape[2]), s);

    if (req[0] == kNullOp) return;
    if (req[0] != kAddTo) {
      for (index_t n = 0; n < num; ++n) {
        im2col(s, im[n].dptr_, im_shape, col_buffer_shape,
               param.kernel, param.pad, param.stride, param.dilate, col[n].dptr_);
      }
    } else {
      Tensor<xpu, 2, DType> tcol = ctx.requested[0]
        .get_space_typed<xpu, 2, DType>(Shape2(col_shape[1], col_shape[2]), s);
      for (index_t n = 0; n < num; ++n) {
        im2col(s, im[n].dptr_, im_shape, col_buffer_shape,
               param.kernel, param.pad, param.stride, param.dilate, tcol.dptr_);
        Tensor<xpu, 2, DType> ocol = col[n];
        ocol += tcol;
      }
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

  const mxnet::TShape im_shape = outputs[0].shape_;
  const mxnet::TShape col_shape = inputs[0].shape_;
  const index_t num = im_shape[0];

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = col_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = im_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 4, DType> im_grad = outputs[0].get_with_shape<xpu, 4, DType>(
        Shape4(im_shape[0], im_shape[1], im_shape[2], im_shape[3]), s);
    Tensor<xpu, 3, DType> col_grad = inputs[0].get_with_shape<xpu, 3, DType>(
        Shape3(col_shape[0], col_shape[1], col_shape[2]), s);

    for (index_t n = 0; n < num; ++n) {
      col2im(s, col_grad[n].dptr_, im_shape, col_buffer_shape,
             param.kernel, param.pad, param.stride, param.dilate,
             im_grad[n].dptr_, req[0]);
    }
  });
}

struct Col2imParam : public dmlc::Parameter<Col2imParam> {
  mxnet::TShape output_size;
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  DMLC_DECLARE_PARAMETER(Col2imParam) {
    DMLC_DECLARE_FIELD(output_size)
    .describe("The spatial dimension of image array: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(kernel).describe("Sliding kernel size: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("The stride between adjacent sliding blocks in spatial dimension: "
              "(w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w). "
              "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("The zero-value padding size on both sides of spatial dimension: "
              "(w,), (h, w) or (d, h, w). Defaults to no padding.");
  }

  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }
};  // struct Col2imParam

template<typename xpu>
void Col2imCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const Col2imParam& param = nnvm::get<Col2imParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TShape im_shape = outputs[0].shape_;
  const mxnet::TShape col_shape = inputs[0].shape_;
  const index_t num = im_shape[0];

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = col_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = im_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 4, DType> im = outputs[0].get_with_shape<xpu, 4, DType>(
        Shape4(im_shape[0], im_shape[1], im_shape[2], im_shape[3]), s);
    Tensor<xpu, 3, DType> col = inputs[0].get_with_shape<xpu, 3, DType>(
        Shape3(col_shape[0], col_shape[1], col_shape[2]), s);

    for (index_t n = 0; n < num; ++n) {
      col2im(s, col[n].dptr_, im_shape, col_buffer_shape,
             param.kernel, param.pad, param.stride, param.dilate,
             im[n].dptr_, req[0]);
    }
  });
}

template<typename xpu>
void Col2imGradCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const Col2imParam& param = nnvm::get<Col2imParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const mxnet::TShape im_shape = inputs[0].shape_;
  const mxnet::TShape col_shape = outputs[0].shape_;
  const index_t batch_size = im_shape[0];

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape col_buffer_shape(1 + spatial_size, 1);
  col_buffer_shape[0] = im_shape[1];
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = im_shape[i + 2] + 2 * param.pad[i];
    const index_t output_size = (pad_size - param.DilatedKernelSize(i)) / param.stride[i] + 1;
    col_buffer_shape[i + 1] = output_size;
  }

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 4, DType> im_grad = inputs[0].get_with_shape<xpu, 4, DType>(
        Shape4(im_shape[0], im_shape[1], im_shape[2], im_shape[3]), s);
    Tensor<xpu, 3, DType> col_grad = outputs[0].get_with_shape<xpu, 3, DType>(
        Shape3(col_shape[0], col_shape[1], col_shape[2]), s);

    if (req[0] == kNullOp) return;
    if (req[0] != kAddTo) {
      for (index_t n = 0; n < batch_size; ++n) {
        im2col(s, im_grad[n].dptr_, im_shape, col_buffer_shape,
               param.kernel, param.pad, param.stride, param.dilate, col_grad[n].dptr_);
      }
    } else {
      Tensor<xpu, 2, DType> tgrad = ctx.requested[0]
        .get_space_typed<xpu, 2, DType>(Shape2(col_shape[1], col_shape[2]), s);
      for (index_t n = 0; n < batch_size; ++n) {
        im2col(s, im_grad[n].dptr_, im_shape, col_buffer_shape,
               param.kernel, param.pad, param.stride, param.dilate, tgrad.dptr_);
        Tensor<xpu, 2, DType> cgrad = col_grad[n];
        cgrad += tgrad;
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_IM2COL_INL_H_
