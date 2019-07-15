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
 * \file np_init_op.h
 * \brief CPU Implementation of numpy init op
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_

#include <vector>
#include <string>
#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"


namespace mxnet {
namespace op {

struct NumpyEyeParam : public dmlc::Parameter<NumpyEyeParam> {
  nnvm::dim_t N;
  dmlc::optional<nnvm::dim_t> M;
  nnvm::dim_t k;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(NumpyEyeParam) {
    DMLC_DECLARE_FIELD(N)
    .describe("Number of rows in the output.");
    DMLC_DECLARE_FIELD(M)
    .set_default(dmlc::optional<nnvm::dim_t>())
    .describe("Number of columns in the output. If None, defaults to N.");
    DMLC_DECLARE_FIELD(k)
    .set_default(0)
    .describe("Index of the diagonal. 0 (the default) refers to the main diagonal,"
              "a positive value refers to an upper diagonal."
              "and a negative value to a lower diagonal.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    MXNET_ADD_ALL_TYPES
    .describe("Data-type of the returned array.");
  }
};

inline bool NumpyRangeShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_shapes,
                            mxnet::ShapeVector* out_shapes) {
  const RangeParam& param = nnvm::get<RangeParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK_NE(param.step, 0) << "_npi_arange does not support step=0";
  CHECK_EQ(param.repeat, 1) << "_npi_arange only supports repeat=1, received " << param.repeat;
  CHECK(param.stop.has_value()) << "_npi_arange requires stop to have a value";
  double out_size = std::ceil((param.stop.value() - param.start) / param.step);
  if (out_size < 0) {
    out_size = 0;
  }
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, mxnet::TShape({static_cast<nnvm::dim_t>(out_size)}));
  return true;
}

inline bool NumpyEyeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const NumpyEyeParam& param = nnvm::get<NumpyEyeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  nnvm::dim_t M = param.M.has_value() ? param.M.value() : param.N;
  CHECK(param.N >= 0) << "negative dimensions are not allowed. N is " << param.N;
  CHECK(M >= 0) << "negative dimensions are not allowed. M is " << M;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(param.N, M));

  return out_attrs->at(0).ndim() != 0U;
}

template<typename xpu>
void NumpyEyeFill(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0) return;  // zero-size tensor
  const NumpyEyeParam& param = nnvm::get<NumpyEyeParam>(attrs.parsed);
  const nnvm::dim_t num_cols = param.M.has_value() ? param.M.value() : param.N;
  EyeFillImpl<xpu>(outputs[0], ctx, req, num_cols, param.N, param.k);
}


struct IndicesOpParam : public dmlc::Parameter<IndicesOpParam> {
  mxnet::TShape dimensions;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(IndicesOpParam) {
    DMLC_DECLARE_FIELD(dimensions)
    .describe("The shape of the grid.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kInt32)
      MXNET_ADD_ALL_TYPES
      .describe("Target data type.");
  }
};

inline bool NumpyIndicesShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_shapes,
                              mxnet::ShapeVector* out_shapes) {
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK_GE(param.dimensions.ndim(), 0)
    << "_npi_indices dimensions the number of dim must not be less than  0";
  mxnet::TShape param_dim = param.dimensions;
  if (!shape_is_known(param_dim)) return false;
  for (int i = 0; i < param_dim.ndim(); ++i) {
    CHECK_GE(param_dim[i], 0) << "_npi_indices dimensions must be non-negative";
  }
  const int indim = param.dimensions.ndim();
  mxnet::TShape ret(indim + 1, -1);
  ret[0] = indim;
  for (int i = 1; i < indim + 1; ++i) {
    ret[i] = param.dimensions[i-1];
  }
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, ret);
  return shape_is_known(out_shapes->at(0));
}

template<int req>
struct indices_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  const nnvm::dim_t value,
                                  const nnvm::dim_t N,
                                  const nnvm::dim_t dim_i,
                                  const nnvm::dim_t j,
                                  const nnvm::dim_t k,
                                  const nnvm::dim_t t) {
    KERNEL_ASSIGN(out[dim_i*N+N/(t*value)*j+i+k*N/t], req, static_cast<DType>(j));
  }
};

template<typename xpu>
void IndicesCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  const TBlob& out_data = outputs[0];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  dim_t indim = param.dimensions.ndim();
  dim_t t = 1;
  dim_t N = out_data.Size()/indim;
  dim_t value = 0;
  if (out_data.Size() == 0) return;
  if (req[0] != kNullOp) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          for (int i = 0; i < indim; ++i) {
            value = param.dimensions[i];
            for (int k = 0; k < t; ++k) {
              for (int j = 0; j < param.dimensions[i]; ++j) {
                Kernel<indices_fwd<req_type>, xpu>::Launch(s, N/(param.dimensions[i] * t),
                    out_data.dptr<DType>(), value, N, i, j, k, t);
              }
            }
            t = t * param.dimensions[i];
          }
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
