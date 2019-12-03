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

#include "prepare_op-common.h"
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

void PrepareWeightOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";

  const TBlob &in = inputs.front();
  const TBlob &out = outputs.front();
  CHECK_EQ(in.type_flag_, mshadow::kFloat32);
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());
  size_t B_cols = in.shape_.ProdShape(0, in.shape_.ndim() - 1);
  size_t inner = in.shape_[in.shape_.ndim() - 1];
  CHECK_EQ(inner % ::intgemm::Int8::kBTileRow, 0) << "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::kBTileRow;
  CHECK_EQ(B_cols % ::intgemm::Int8::kBTileCol, 0) << "intgemm requires the output dimension (the product of all but the last dimension of the weight matrix) to be a multiple of " << ::intgemm::Int8::kBTileCol << ".";

  const float *B = in.dptr<float>();
  int8_t *quantB = out.dptr<int8_t>();
  // TODO: eliminate transpose here by making a PrepareWeightColumnMajor.
  intgemm::AlignedVector<float> B_transpose(inner * B_cols);
  for (size_t i = 0; i < inner; ++i) {
    for (size_t j = 0; j < B_cols; ++j) {
      B_transpose[i * B_cols + j] = B[i + inner * j];
    }
  }
  const PrepareParam& param = nnvm::get<PrepareParam>(attrs.parsed);
  ::intgemm::Int8::PrepareB(B_transpose.begin(), quantB, param.multiplier, inner, B_cols);
}

NNVM_REGISTER_OP(_contrib_intgemm_prepare_weight)
.describe(R"code(This operator converts a float32 matrix to intgemm's internal representation of weight matrices in preparation for the operation C = data * weight.  weight should be provided in column-major order i.e. the last dimension of shape is the number of rows of weight.  This is the usual representation mxnet uses for weights. This operator is not meant to be fast; it is meant to be run offline to quantize a model.

The float32 values are multiplied by the provided multiplier before casting to int8.

The internal representation is CPU dependent: AVX512BW, AVX2, and SSSE3 have different formats.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<PrepareParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"weight"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", PrepareOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareWeightOpForwardCPU)
.add_argument("weight", "NDArray-or-Symbol", "Parameter matrix to be prepared for multiplication.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(PrepareParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
