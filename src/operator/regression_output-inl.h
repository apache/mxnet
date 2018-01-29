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
 * \file regression_ouput-inl.h
 * \brief Regression output operator.
*/
#ifndef MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
#define MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

/*!
 * \brief regression namespace
 */
namespace reg_enum {
enum RegressionOutputOpInputs {kData, kLabel};
enum RegressionOutputOutputs {kOut};
}  // reg_enum

struct RegressionOutputParam : public dmlc::Parameter<RegressionOutputParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(RegressionOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
  };
};

inline bool RegressionOpShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  using namespace mshadow;
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[data, label]";
  const TShape &dshape = in_attrs->at(0);
  if (dshape.ndim() == 0) return false;
  auto &lshape = (*in_attrs)[1];
  if (lshape.ndim() == 0) {
    // special treatment for 1D output, to allow 1D label by default.
    // Think about change convention later
    if (dshape.ndim() == 2 && dshape[1] == 1) {
      lshape = Shape1(dshape[0]);
    } else {
      lshape = dshape;
    }
  } else if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
    std::ostringstream os;
    os << "Shape inconsistent, Provided=" << lshape << ','
       << " inferred shape=" << dshape;
    throw ::mxnet::op::InferShapeError(os.str(), 1);
  }
  out_attrs->clear();
  out_attrs->push_back(dshape);
  return true;
}

template<typename xpu, typename ForwardOp>
void RegressionForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[reg_enum::kData].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[reg_enum::kOut], Req, {
      const DType* in_data = inputs[reg_enum::kData].dptr<DType>();
      DType* out_data = outputs[reg_enum::kOut].dptr<DType>();
      using namespace mxnet_op;
      Kernel<op_with_req<ForwardOp, Req>, xpu>::Launch(
        s, outputs[reg_enum::kOut].Size(), out_data, in_data);
    });
  });
}

template<typename xpu, typename BackwardOp>
void RegressionBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const RegressionOutputParam& param = nnvm::get<RegressionOutputParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // inputs are in_label, out_data
  // outputs are data_grad, label_grad
  MSHADOW_REAL_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      const DType* in_label = inputs[0].dptr<DType>();
      const DType* out_data = inputs[1].dptr<DType>();
      DType* data_grad = outputs[0].dptr<DType>();
      const real_t num_output = inputs[0].Size()/inputs[0].shape_[0];
      using namespace mxnet_op;
      Kernel<op_with_req<BackwardOp, Req>, xpu>::Launch(
        s, outputs[0].Size(), data_grad, out_data, in_label);
      Kernel<op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
        s, outputs[0].Size(), data_grad, data_grad,
        static_cast<DType>(param.grad_scale/num_output));
    });
  });
}

struct RegressionOpGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(n->inputs[reg_enum::kLabel]);
    heads.emplace_back(nnvm::NodeEntry{n, reg_enum::kOut, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
