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
 * \file prepare_weight_op.cc
 * \brief Converts weight matrices to intgemm's representation.
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "intgemm/intgemm.h"

namespace mxnet {
namespace op {

struct PrepareWeightParam : public dmlc::Parameter<PrepareWeightParam> {
  bool already_quantized;
  DMLC_DECLARE_PARAMETER(PrepareWeightParam) {
    DMLC_DECLARE_FIELD(already_quantized).set_default(false)
    .describe("Is the weight matrix already quantized?");
  }
};
DMLC_REGISTER_PARAMETER(PrepareWeightParam);

bool PrepareWeightOpShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector* in_attrs,
                    mxnet::ShapeVector* out_attrs) {
  // Optimal maximum parameter.
  CHECK_GE(in_attrs->size(), 1U) << "Need at least weight to quantize.";
  CHECK_LE(in_attrs->size(), 2U) << "weight and maximum for scaling.";
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  if (in_attrs->size() == 2U) {
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape(1, 1));
  }
  return shape_is_known(out_attrs->at(0));
}

bool PrepareWeightOpType(const nnvm::NodeAttrs& attrs,
                   std::vector<int>* in_attrs,
                   std::vector<int>* out_attrs) {
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  CHECK_GE(in_attrs->size(), 1U) << "Need at least weight to quantize.";
  CHECK_LE(in_attrs->size(), 2U) << "weight and maximum for scaling.";
  if (in_attrs->size() == 1U) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kInt8);
  } else if (in_attrs->size() == 2U) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  }
  return true;
}

bool PrepareWeightOpStorageType(const nnvm::NodeAttrs& attrs,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_GE(in_attrs->size(), 1U) << "Need at least weight to quantize.";
  CHECK_LE(in_attrs->size(), 2U) << "weight and maximum for scaling.";
  CHECK_EQ(out_attrs->size(), 1U);
  STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, 0, kDefaultStorage);
  STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, 0, kDefaultStorage);
  if (in_attrs->size() == 2U) {
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, 1, kDefaultStorage);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

void PrepareWeightOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  const PrepareWeightParam& params = nnvm::get<PrepareWeightParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), params.already_quantized ? 1U : 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";

  const TBlob &in = inputs.front();
  const TBlob &out = outputs.front();
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());
  size_t B_cols = in.shape_.ProdShape(0, in.shape_.ndim() - 1);
  size_t inner = in.shape_[in.shape_.ndim() - 1];
  CHECK_EQ(inner % ::intgemm::Int8::tile_info.b_rows, 0) <<
    "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::tile_info.b_rows;
  CHECK_EQ(B_cols % ::intgemm::Int8::tile_info.b_cols, 0) <<
    "intgemm requires the output dimension (the product of all but the last dimension of the "
    "weight matrix) to be a multiple of " << ::intgemm::Int8::tile_info.b_cols << ".";

  int8_t *quantB = out.dptr<int8_t>();
  CHECK_EQ(reinterpret_cast<intptr_t>(quantB) % 64, 0) <<
    "Pointers should be aligned to a multiple of 64.";
  CHECK(in.type_flag_ == mshadow::kFloat32 || in.type_flag_ == mshadow::kInt8) <<
    "Expected either 32-bit values to be quantized or 8-bit values to rearrange.";
  if (in.type_flag_ == mshadow::kInt8) {
    const int8_t *B = in.dptr<int8_t>();
    CHECK_EQ(reinterpret_cast<intptr_t>(B) % 64, 0) <<
      "Pointers should be aligned to a multiple of 64.";
    ::intgemm::Int8::PrepareBQuantizedTransposed(B, quantB, inner, B_cols);
  } else if (in.type_flag_ == mshadow::kFloat32) {
    const float *B = in.dptr<float>();
    CHECK_EQ(reinterpret_cast<intptr_t>(B) % 64, 0) <<
      "Pointers should be aligned to a multiple of 64.";
    ::intgemm::Int8::PrepareBTransposed(
        B,
        quantB,
        127.0 / *inputs[1].dptr<float>(),
        inner,
        B_cols);
  }
}

NNVM_REGISTER_OP(_contrib_intgemm_prepare_weight)
.add_alias("_npx_intgemm_prepare_weight")
.describe(R"code(This operator converts a weight matrix in column-major format to intgemm's internal fast representation of weight matrices.  MXNet customarily stores weight matrices in column-major (transposed) format. This operator is not meant to be fast; it is meant to be run offline to quantize a model.

In other words, it prepares weight for the operation C = data * weight^T.

If the provided weight matrix is float32, it will be quantized first.  The quantization function is (int8_t)(127.0 / max * weight) where multiplier is provided as argument 1 (the weight matrix is argument 0).  Then the matrix will be rearranged into the CPU-dependent format.

If the provided weight matrix is already int8, the matrix will only be rearranged into the CPU-dependent format.  This way one can quantize with intgemm_prepare_data (which just quantizes), store to disk in a consistent format, then at load time convert to CPU-dependent format with intgemm_prepare_weight.

The internal representation depends on register length.  So AVX512, AVX2, and SSSE3 have different formats.  AVX512BW and AVX512VNNI have the same representation.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<PrepareWeightParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const PrepareWeightParam& params = nnvm::get<PrepareWeightParam>(attrs.parsed);
  return params.already_quantized ? 1 : 2;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  const PrepareWeightParam& params = nnvm::get<PrepareWeightParam>(attrs.parsed);
  return params.already_quantized ?
    std::vector<std::string>{"weight"} : std::vector<std::string>{"weight", "maxabs"};
})
.set_attr<mxnet::FInferShape>("FInferShape", PrepareWeightOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareWeightOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareWeightOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareWeightOpForwardCPU)
.add_argument("weight", "NDArray-or-Symbol", "Parameter matrix to be prepared for multiplication.")
.add_argument(
    "maxabs",
    "NDArray-or-Symbol",
    "Maximum absolute value for scaling. The weights will be multipled by 127.0 / maxabs.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(PrepareWeightParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
