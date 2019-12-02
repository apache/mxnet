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
 * \file prepare_a_op.cc
 * \brief Converts A matrices (typically activations) to intgemm's 
 * representation for A in C=AB.  This just quantizes to int8 and bans -128.  
 * The only difference from Quantize/QuantizeV2 is that it bans -128.
 */

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "../../../../3rdparty/intgemm/intgemm.h"

namespace mxnet {
namespace op {

struct PrepareAParam : public dmlc::Parameter<PrepareAParam> {
  float multiplier;
  DMLC_DECLARE_PARAMETER(PrepareAParam) {
    DMLC_DECLARE_FIELD(multiplier)
      .describe("Multiply floats by this constant before casting to int8.  Typically you would set this to 127.0 / max absolute value in A.");
  }
};
DMLC_REGISTER_PARAMETER(PrepareAParam);

inline bool PrepareAOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  // One in, one out.
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return shape_is_known(out_attrs->at(0));
}

inline bool PrepareAOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // This routine converts from float to int8.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  return true;
}

inline bool PrepareAOpStorageType(const nnvm::NodeAttrs& attrs,
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


void PrepareAOpForwardCPU(const nnvm::NodeAttrs& attrs,
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
  CHECK_EQ(size % 16, 0) << "intgemm PrepareA requires the size be a multiple of 16.";
 
  const float *A = in.dptr<float>();
  int8_t *quantA = out.dptr<int8_t>();
  const PrepareAParam& param = nnvm::get<PrepareAParam>(attrs.parsed);
  ::intgemm::Int8::Quantize(A, quantA, param.multiplier, size);
}

NNVM_REGISTER_OP(_contrib_intgemm_preparea)
.describe(R"code(This operator converts quantizes float32 to int8 while also banning -128.

It it suitable for preparing an A matrix for use by intgemm's C=AB operation.

The float32 values are multiplied by the provided multiplier before casting to int8.  Typically this is 127.0 / maxabsolute(A).
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<PrepareAParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"A"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", PrepareAOpShape)
.set_attr<nnvm::FInferType>("FInferType", PrepareAOpType)
.set_attr<FInferStorageType>("FInferStorageType", PrepareAOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", PrepareAOpForwardCPU)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("A", "NDArray-or-Symbol", "Activation matrix to be prepared for multiplication.")
.add_arguments(PrepareAParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
