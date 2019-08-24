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
 * Copyright (c) 2018 by Contributors
 * \file constant-inl.h
*/

#ifndef MXNET_OPERATOR_CONTRIB_CONSTANT_INL_H_
#define MXNET_OPERATOR_CONTRIB_CONSTANT_INL_H_

#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct ConstantParam : public dmlc::Parameter<ConstantParam> {
  mxnet::Tuple<float> value;
  int dtype;
  DMLC_DECLARE_PARAMETER(ConstantParam) {
    DMLC_DECLARE_FIELD(value)
    .set_default({1.0f, 1.0f})
    .describe("The target shape");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    MXNET_ADD_ALL_TYPES
    .describe("Target data type.");
  }
};

inline bool ConstantShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
    const ConstantParam& param_ = nnvm::get<ConstantParam>(attrs.parsed);
    CHECK_EQ(in_attrs->size(), 0U);
    CHECK_EQ(out_attrs->size(), 1U);
    const double out_size = param_.value.end() - param_.value.begin();
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape({static_cast<nnvm::dim_t>(out_size)}));
    return true;
}

struct constant {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const mxnet::Tuple<float>& value,
                                  int n, int req, DType* out) {
    for (ptrdiff_t j = 0; j < n; j++) {
      KERNEL_ASSIGN(out[j], req, value[j]);
    }
  }
};

template<typename xpu, typename ParamType>
void ConstantForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<constant, xpu>::Launch(s,
                                  outputs[0].Size(),
                                  param.value,
                                  static_cast<DType>(param.value.ndim()),
                                  req[0],
                                  outputs[0].dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CONSTANT_INL_H_