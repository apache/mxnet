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
 * \file prepare_data_op.cc
 * \brief Converts data aka A matrices (typically activations) to intgemm's
 * representation for A in C=AB.  This just quantizes to int8 and bans -128.
 * The only difference from Quantize/QuantizeV2 is that it bans -128.
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

bool PrepareDataOpShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector* in_attrs,
                    mxnet::ShapeVector* out_attrs) {
  // data and maximum
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape(1, 1));

  return shape_is_known(out_attrs->at(0));
}

bool PrepareDataOpType(const nnvm::NodeAttrs& attrs,
                   std::vector<int>* in_attrs,
                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  // This routine converts from float to int8 with a scaling factor
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  return true;
}

bool PrepareDataOpStorageType(const nnvm::NodeAttrs& attrs,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, 0, kDefaultStorage);
  STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, 0, kDefaultStorage);
  STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, 1, kDefaultStorage);
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  return true;
}

void PrepareDataOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";
  const TBlob &in = inputs[0], &out = outputs[0];

  CHECK_EQ(in.type_flag_, mshadow::kFloat32);
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());

  const float *A = in.dptr<float>();
  int8_t *quantA = out.dptr<int8_t>();
  CHECK_EQ(reinterpret_cast<intptr_t>(A) % 64, 0);
  CHECK_EQ(reinterpret_cast<intptr_t>(quantA) % 64, 0);
  const float multiplier = 127.0 / *inputs[1].dptr<float>();
  ::intgemm::Int8::Quantize(A, quantA, multiplier, in.shape_.Size());
}

NNVM_REGISTER_OP(_contrib_intgemm_prepare_data)
.add_alias("_npx_intgemm_prepare_data")
.describe(R"code(This operator converts quantizes float32 to int8 while also banning -128.

It it suitable for preparing an data matrix for use by intgemm's C=data * weights operation.

The float32 values are scaled such that maxabs maps to 127. Typically maxabs = maxabsolute(A).
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "maxabs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", PrepareDataOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareDataOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareDataOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareDataOpForwardCPU)
.add_argument("data", "NDArray-or-Symbol", "Activation matrix to be prepared for multiplication.")
.add_argument(
    "maxabs",
    "NDArray-or-Symbol",
    "Maximum absolute value to be used for scaling.  (The values will be multiplied by 127.0 / "
      "maxabs.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet
