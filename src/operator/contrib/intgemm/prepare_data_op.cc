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

#include "prepare_op-common.h"
#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "../../../../3rdparty/intgemm/intgemm.h"

namespace mxnet {
namespace op {

void PrepareDataOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "intgemm only overwrites";
  const TBlob &in = inputs[0], &out = outputs[0];

  CHECK_EQ(in.type_flag_, mshadow::kFloat32);
  CHECK_EQ(out.type_flag_, mshadow::kInt8);
  CHECK(in.CheckContiguous());
  CHECK(out.CheckContiguous());
  size_t size = in.shape_.Size();
  CHECK_EQ(size % 16, 0) << "intgemm PrepareData requires the size be a multiple of 16.";

  const float *A = in.dptr<float>();
  int8_t *quantA = out.dptr<int8_t>();
  const PrepareParam& param = nnvm::get<PrepareParam>(attrs.parsed);
  ::intgemm::Int8::Quantize(A, quantA, param.multiplier, size);
}

NNVM_REGISTER_OP(_contrib_intgemm_prepare_data)
.describe(R"code(This operator converts quantizes float32 to int8 while also banning -128.

It it suitable for preparing an data matrix for use by intgemm's C=data * weights operation.

The float32 values are multiplied by the provided multiplier before casting to int8.  Typically this is 127.0 / maxabsolute(A).
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<PrepareParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", PrepareOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareDataOpForwardCPU)
.add_argument("data", "NDArray-or-Symbol", "Activation matrix to be prepared for multiplication.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(PrepareParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
