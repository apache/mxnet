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
 * \file prepare_b_op.cc
 * \brief Converts B matrices to intgemm's representation.
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "../../../../3rdparty/intgemm/aligned.h"
#include "../../../../3rdparty/intgemm/intgemm.h"

namespace mxnet {
namespace op {

struct PrepareBParam : public dmlc::Parameter<PrepareBParam> {
  float multiplier;
  DMLC_DECLARE_PARAMETER(PrepareBParam) {
    DMLC_DECLARE_FIELD(multiplier)
      .describe("Multiply floats by this constant before casting to int8.  Typically you would set this to 127.0 / max absolute value in B.");
  }
};
DMLC_REGISTER_PARAMETER(PrepareBParam);

inline bool PrepareBOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  // One in, one out.
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  const mxnet::TShape &shape = in_attrs->front();
  if (mxnet::ndim_is_known(shape)) {
    CHECK_GE(shape.ndim(), 2) << "Matrices have at least two dimensions.";
  }
  return !mxnet::op::shape_is_none(out_attrs->at(0));
}

inline bool PrepareBOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // This routine converts from float to int8.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  return true;
}

inline bool PrepareBOpStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  // Dense storage only.
  return storage_type_assign(&out_attrs->front(), kDefaultStorage, dispatch_mode, DispatchMode::kFCompute) &&
    storage_type_assign(&in_attrs->front(), kDefaultStorage, dispatch_mode, DispatchMode::kFCompute);
}

namespace {
void PrepareBOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const TBlob& in,
                       const TBlob& out) {
  CHECK_EQ(in.type_flag_, mshadow::kFloat32);
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());
  size_t B_cols = in.shape_.ProdShape(0, in.shape_.ndim() - 1);
  size_t inner = in.shape_[in.shape_.ndim() - 1];
  CHECK_EQ(inner % ::intgemm::Int8::kBTileRow, 0) << "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::kBTileRow;
  CHECK_EQ(B_cols % ::intgemm::Int8::kBTileCol, 0) << "intgemm requires B have a multiple of " << ::intgemm::Int8::kBTileCol << " columns inthe equation C = AB.";
 
  const float *B = in.dptr<float>();
  int8_t *quantB = out.dptr<int8_t>();
  const PrepareBParam& param = nnvm::get<PrepareBParam>(attrs.parsed);
  // TODO: eliminate transpose here by making a PrepareBColumnMajor.
  intgemm::AlignedVector<float> B_transpose(inner * B_cols);
  for (size_t i = 0; i < inner; ++i) {
    for (size_t j = 0; j < B_cols; ++j) {
      B_transpose[i * B_cols + j] = B[i + inner * j];
    }
  }
  ::intgemm::Int8::PrepareB(B_transpose.begin(), quantB, param.multiplier, inner, B_cols);
}
} // namespace

void PrepareBOpForwardExCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";
  CHECK_EQ(inputs[0].storage_type(), kDefaultStorage);
  CHECK_EQ(outputs[0].storage_type(), kDefaultStorage);
  PrepareBOpForward(attrs, ctx, inputs[0].data(), outputs[0].data());
}

void PrepareBOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";
  PrepareBOpForward(attrs, ctx, inputs[0], outputs[0]);
}

NNVM_REGISTER_OP(_contrib_intgemm_prepareb)
.describe(R"code(This operator converts a float32 matrix to intgemm's internal representation of B in preparation for the operation C = AB.  B should be provided in column-major order i.e. the last dimension of shape is the number of rows of B. This operator is not meant to be fast; it is meant to be run offline to quantize a model.

The float32 values are multiplied by the provided multiplier before casting to int8.

The internal representation of B is CPU dependent: AVX512BW, AVX2, and SSSE3 have different formats.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<PrepareBParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"B"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", PrepareBOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareBOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareBOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareBOpForwardCPU)
.set_attr<FComputeEx>("FComputeEx<cpu>", PrepareBOpForwardExCPU)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("B", "NDArray-or-Symbol", "Parameter matrix to be prepared for multiplication.")
.add_arguments(PrepareBParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
