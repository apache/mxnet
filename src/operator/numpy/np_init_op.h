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

#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"


namespace mxnet {
namespace op {

struct NumpyEyeParam : public dmlc::Parameter<NumpyEyeParam> {
  nnvm::dim_t N;
  dmlc::optional<nnvm::dim_t> M;
  nnvm::dim_t k;
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
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int8", mshadow::kInt8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int64", mshadow::kInt64)
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
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(param.N, M));
  return true;
}

template<typename xpu>
void NumpyEyeFill(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0) return; // zero-size tensor
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyEyeParam& param = nnvm::get<NumpyEyeParam>(attrs.parsed);
  const TBlob& out_data = outputs[0];
  const nnvm::dim_t num_cols = param.M.has_value() ? param.M.value() : param.N;
  const nnvm::dim_t cnnz = std::max(num_cols - std::abs(param.k), (nnvm::dim_t)0);
  const nnvm::dim_t rnnz = std::max(param.N - std::abs(param.k), (nnvm::dim_t)0);
  const nnvm::dim_t nnz = param.k > 0 ? std::min(cnnz, param.N) :
                                        std::min(rnnz, num_cols);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Fill(s, out_data, req[0], static_cast<DType>(0));
      if (nnz > 0) {
        Kernel<eye_dns_fill<req_type>, xpu>::Launch(s, nnz, out_data.dptr<DType>(),
          std::max(static_cast<nnvm::dim_t>(0), param.k), param.k, num_cols);
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
