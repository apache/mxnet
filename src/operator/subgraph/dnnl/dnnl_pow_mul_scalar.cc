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
 * \file dnnl_pow_mul_scalar.cc
 * \brief DNNL pow_mul_scalar operator based on subgraph
 */

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_pow_mul_scalar-inl.h"
#include "operator/subgraph/common.h"

namespace mxnet {
namespace op {
bool DNNLPowMulScalarType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const DNNLPowMulScalarParam& param = nnvm::get<DNNLPowMulScalarParam>(attrs.parsed);
  bool scalar_is_int                 = param.exp_is_int && param.mul_is_int;
  if (common::is_int(in_attrs->at(0)) && !scalar_is_int) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  } else if (in_attrs->at(0) == mshadow::kBool) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, scalar_is_int ? mshadow::kInt64 : mshadow::kFloat64);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  }
  return out_attrs->at(0) != -1;
}

inline static bool DNNLPowMulScalarStorageType(const nnvm::NodeAttrs& attrs,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int>* in_attrs,
                                               std::vector<int>* out_attrs) {
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

template <typename OP>
static void ComputeOP(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      mshadow::Stream<cpu>* s,
                      const TBlob& input,
                      const TBlob& output,
                      const double scalar) {
  using namespace mshadow;
  using namespace mshadow::expr;
  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    auto temp_req    = input.dptr_ == output.dptr_ ? kWriteInplace : kWriteTo;
    TBlob temp_tblob = input;
    if (input.type_flag_ != output.type_flag_) {
      temp_tblob = TBlob(ctx.requested[0].get_space_typed<cpu, 1, DType>(Shape1(output.Size()), s));
      CastCompute<cpu>(attrs, ctx, {input}, {kWriteTo}, {temp_tblob});
    }
    MXNET_ASSIGN_REQ_SWITCH(temp_req, Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
          s, input.Size(), output.dptr<DType>(), temp_tblob.dptr<DType>(), DType(scalar));
    });
  });
}

static void PowMulScalarCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  DCHECK_EQ(inputs.size(), 1);
  DCHECK_EQ(outputs.size(), 1);
  using namespace mshadow;
  using namespace mshadow::expr;
  const DNNLPowMulScalarParam& param = nnvm::get<DNNLPowMulScalarParam>(attrs.parsed);
  // temp_mid_tblob is output of power operation and input of multiplication.
  // Its dtype depends on input dtype and scalar type.
  TBlob temp_mid_tblob;
  if ((common::is_int(inputs[0].type_flag_) || inputs[0].type_flag_ == kBool) &&
      !param.exp_is_int) {
    // If exponent is not an integer and data is of int or bool dtype, output of both power and
    // multiplication operations is of same dtype as outputs[0], therefore we can assign it as
    // temp_mid_tblob.
    temp_mid_tblob = outputs[0];
  } else if (inputs[0].type_flag_ == kBool) {
    // If exponent is an integer and input data is of bool dtype, output of the power operation is
    // of int64 dtype.
    temp_mid_tblob =
        TBlob(ctx.requested[0].get_space_typed<cpu, 1, int64_t>(Shape1(inputs[0].Size()), s));
  } else {
    // If both exponent and input data is of integer dtype, output of the power operation is of the
    // same dtype as its input, therefore we can assign inputs[0] as temp_mid_tblob.
    temp_mid_tblob = inputs[0];
  }
  ComputeOP<mshadow_op::power>(attrs, ctx, s, inputs[0], temp_mid_tblob, param.exponent);
  ComputeOP<mshadow_op::mul>(attrs, ctx, s, temp_mid_tblob, outputs[0], param.multiplier);
}

static void PowMulScalarComputeEx(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<mxnet::NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<mxnet::NDArray>& outputs) {
  if (SupportDNNL<DNNLTypeMode::FloatTypes>(inputs[0])) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLPowMulScalarForward<true>, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(PowMulScalarCompute, attrs, ctx, inputs, req, outputs);
  } else {
    FallBackCompute(PowMulScalarCompute, attrs, ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(_sg_pow_mul_scalar)
    .describe(R"code(_sg_pow_mul_scalar)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) { return 1; })
    .set_num_outputs([](const NodeAttrs& attrs) { return 1; })
    .set_attr_parser(ParamParser<DNNLPowMulScalarParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"input"};
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", DNNLPowMulScalarType)
    .set_attr<FInferStorageType>("FInferStorageType", DNNLPowMulScalarStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>", PowMulScalarCompute)
    .set_attr<FComputeEx>("FComputeEx<cpu>", PowMulScalarComputeEx)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
