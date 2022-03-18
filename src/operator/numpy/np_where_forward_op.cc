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
 * \file np_where_forward_op.cc
 * \brief CPU Implementation of numpy operator where
 */

#include "np_where_op-inl.h"
#include "../nn/dnnl/dnnl_where-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyWhereOpShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& operand1 = (*in_attrs)[0];
  mxnet::TShape& operand2 = (*in_attrs)[1];
  mxnet::TShape& operand3 = (*in_attrs)[2];

  if (operand1 == operand2 && operand2 == operand3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, operand1);
    return shape_is_known(out_attrs->at(0));
  }
  mxnet::TShape out(std::max({operand1.ndim(), operand2.ndim(), operand3.ndim()}), -1);
  const int b1 = out.ndim() - operand1.ndim();
  const int b2 = out.ndim() - operand2.ndim();
  const int b3 = out.ndim() - operand3.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int s1 = 1, s2 = 1, s3 = 1;
    if (i >= b1)
      s1 = operand1[i - b1];
    if (i >= b2)
      s2 = operand2[i - b2];
    if (i >= b3)
      s3 = operand3[i - b3];
    if (!(s1 == s2 && s2 == s3)) {
      CHECK((s1 == 1 && s2 == 1) || (s1 == 1 && s3 == 1) || (s2 == 1 && s3 == 1) ||
            (s1 == 1 && s2 == s3) || (s2 == 1 && s1 == s3) || (s3 == 1 && s1 == s2))
          << "Operands could not be broadcast together.";
      out[i] = std::max({s1, s2, s3});
    } else {
      out[i] = s1;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return shape_is_known(out);
}

inline bool NumpyWhereOpType(const nnvm::NodeAttrs& attrs,
                             std::vector<int>* in_attrs,
                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U) << "where operator takes 3 arguments (" << in_attrs->size()
                                 << " given)";
  CHECK_EQ(out_attrs->size(), 1U);
  std::vector<int> sub_in_attrs(in_attrs->begin() + 1, in_attrs->end());
  bool flag = ElemwiseType<2, 1>(attrs, &sub_in_attrs, out_attrs);
  return flag && (in_attrs->at(0) != -1);
}

inline bool NumpyWhereScalarOpType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  std::vector<int> sub_in_attrs(in_attrs->begin() + 1, in_attrs->end());
  bool flag = ElemwiseType<1, 1>(attrs, &sub_in_attrs, out_attrs);
  return flag && (in_attrs->at(0) != -1);
}

DMLC_REGISTER_PARAMETER(NumpyWhereScalarParam);
DMLC_REGISTER_PARAMETER(NumpyWhereScalar2Param);

#if MXNET_USE_ONEDNN == 1
static void WhereForwardEx(const nnvm::NodeAttrs& attrs,
                           const OpContext& op_ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  if (req[0] == kNullOp) {
    return;
  }
  if (SupportDNNLWhere(inputs)) {
    DNNL_OPCHECK_INIT(/*is backward*/ false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLWhereForward, attrs, op_ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(NumpyWhereOpForward<cpu>, attrs, op_ctx, inputs, req, outputs);
  } else {
    FallBackCompute(NumpyWhereOpForward<cpu>, attrs, op_ctx, inputs, req, outputs);
  }
}

inline static bool WhereInferStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int>* in_attrs,
                                         std::vector<int>* out_attrs) {
  return DNNLStorageType(attrs,
                         dev_mask,
                         /*support onednn*/ true,
                         dispatch_mode,
                         in_attrs,
                         out_attrs);
}
#endif  // MXNET_USE_ONEDNN == 1

NNVM_REGISTER_OP(_npi_where)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"condition", "x", "y"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyWhereOpShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyWhereOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{1, 0}, {2, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", NumpyWhereOpForward<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FComputeEx>("FComputeEx<cpu>", WhereForwardEx)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FInferStorageType>("FInferStorageType", WhereInferStorageType)
#endif
    .set_attr<nnvm::FGradient>(
        "FGradient",
        // Use the following lambda function instead of ElemwiseGradUseIn for best efficiency.
        // grad[condition] = 0; to calculate grad[x] and grad[y] we need only condition from input.
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          std::vector<nnvm::NodeEntry> ret;
          // make zero grad node for grad[condition]
          auto p =
              MakeNode("zeros_like", n->attrs.name + "_cond_backward", {n->inputs[0]}, nullptr, &n);
          ret.emplace_back(p);

          // make grad nodes for grad[x] and grad[y]
          std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
          heads.push_back(n->inputs[0]);  // only need condition to calculate gradients
          p             = nnvm::Node::Create();
          p->attrs.op   = nnvm::Op::Get("_backward_np_where");
          p->attrs.name = n->attrs.name + "_backward";
          p->attrs.dict = n->attrs.dict;
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          p->control_deps.emplace_back(n);
          p->inputs = std::move(heads);
          ret.emplace_back(p, 0, 0);
          ret.emplace_back(p, 1, 0);
          return ret;
        })
    .add_argument("condition", "NDArray-or-Symbol", "condition array")
    .add_argument("x", "NDArray-or-Symbol", "input x")
    .add_argument("y", "NDArray-or-Symbol", "input y");

NNVM_REGISTER_OP(_npi_where_lscalar)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyWhereScalarParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"condition", "x"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyWhereScalarOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{1, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", NumpyWhereScalarOpForward<cpu, true>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        // Use the following lambda function instead of ElemwiseGradUseIn
        // for best efficiency. grad[condition] = 0; to calculate grad[x] or grad[y]
        // we need only condition from input.
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          std::vector<nnvm::NodeEntry> ret;
          // make zero grad node for grad[condition]
          auto p =
              MakeNode("zeros_like", n->attrs.name + "_cond_backward", {n->inputs[0]}, nullptr, &n);
          ret.emplace_back(p);

          // make grad nodes for grad[x] and grad[y]
          std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
          heads.push_back(n->inputs[0]);  // only need condition to calculate gradients
          p             = nnvm::Node::Create();
          p->attrs.op   = nnvm::Op::Get("_backward_np_where_lscalar");
          p->attrs.name = n->attrs.name + "_backward";
          p->attrs.dict = n->attrs.dict;
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          p->control_deps.emplace_back(n);
          p->inputs = std::move(heads);
          ret.emplace_back(p, 0, 0);
          return ret;
        })
    .add_argument("condition", "NDArray-or-Symbol", "condition array")
    .add_argument("x", "NDArray-or-Symbol", "input x")
    .add_arguments(NumpyWhereScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_where_rscalar)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyWhereScalarParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"condition", "y"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyWhereScalarOpType)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{1, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", NumpyWhereScalarOpForward<cpu, false>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        // Use the following lambda function instead of ElemwiseGradUseIn
        // for best efficiency. grad[condition] = 0; to calculate grad[x] or grad[y]
        // we need only condition from input.
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          std::vector<nnvm::NodeEntry> ret;
          // make zero grad node for grad[condition]
          auto p =
              MakeNode("zeros_like", n->attrs.name + "_cond_backward", {n->inputs[0]}, nullptr, &n);
          ret.emplace_back(p);

          // make grad nodes for grad[x] and grad[y]
          std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
          heads.push_back(n->inputs[0]);  // only need condition to calculate gradients
          p             = nnvm::Node::Create();
          p->attrs.op   = nnvm::Op::Get("_backward_np_where_rscalar");
          p->attrs.name = n->attrs.name + "_backward";
          p->attrs.dict = n->attrs.dict;
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          p->control_deps.emplace_back(n);
          p->inputs = std::move(heads);
          ret.emplace_back(p, 0, 0);
          return ret;
        })
    .add_argument("condition", "NDArray-or-Symbol", "condition array")
    .add_argument("y", "NDArray-or-Symbol", "input y")
    .add_arguments(NumpyWhereScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_where_scalar2)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyWhereScalar2Param>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"condition"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType",
                                [](const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
                                  CHECK_EQ(in_attrs->size(), 1U);
                                  CHECK_EQ(out_attrs->size(), 1U);
                                  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
                                  return in_attrs->at(0) != -1;
                                })
    .set_attr<FCompute>("FCompute<cpu>", NumpyWhereScalar2OpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .add_argument("condition", "NDArray-or-Symbol", "condition array")
    .add_arguments(NumpyWhereScalar2Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
