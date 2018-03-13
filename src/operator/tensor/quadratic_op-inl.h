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
 * \file quad_function-inl.h
 * \brief Operator implementing quadratic function.
 * For using as an exmaple in the tutorial of adding operators
 * in MXNet backend.
 */
#ifndef MXNET_OPERATOR_TENSOR_QUADRATIC_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_QUADRATIC_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct QuadraticParam : public dmlc::Parameter<QuadraticParam> {
  float a, b, c;
  DMLC_DECLARE_PARAMETER(QuadraticParam) {
    DMLC_DECLARE_FIELD(a)
      .set_default(0.0)
      .describe("Coefficient of the quadratic term in the quadratic function.");
    DMLC_DECLARE_FIELD(b)
      .set_default(0.0)
      .describe("Coefficient of the linear term in the quadratic function.");
    DMLC_DECLARE_FIELD(c)
      .set_default(0.0)
      .describe("Constant term in the quadratic function.");
  }
};

inline bool QuadraticOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool QuadraticOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

template<int req>
struct quadratic_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const float a, const float b, const float c) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i] * (a * in_data[i] + b) + c);
  }
};

template<int req>
struct quadratic_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data, const float a, const float b) {
    KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * (2 * a * in_data[i] + b));
  }
};

template<typename xpu>
void QuadraticOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<quadratic_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          param.a, param.b, param.c);
    });
  });
}

template<typename xpu>
void QuadraticOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<quadratic_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>(), param.a, param.b);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_QUADRATIC_OP_INL_H_
