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
 *  Copyright (c) 2019 by Contributors
 * \file reset_arrays-inl.h
 * \brief setting all array element values to zeros
 * \author Moises Hernandez-Fernandez, Andrei Ivanov
 */

#ifndef MXNET_OPERATOR_CONTRIB_RESET_ARRAYS_INL_H_
#define MXNET_OPERATOR_CONTRIB_RESET_ARRAYS_INL_H_

#include <vector>
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct ResetArraysParam : public dmlc::Parameter<ResetArraysParam> {
  int num_arrays;
  DMLC_DECLARE_PARAMETER(ResetArraysParam) {
    DMLC_DECLARE_FIELD(num_arrays)
    .describe("number of input arrays.");
  }
};

inline bool ResetArraysShape(const NodeAttrs& attrs,
                            std::vector<mxnet::TShape>* in_shape,
                            std::vector<mxnet::TShape>* out_shape) {
  const auto& param = dmlc::get<ResetArraysParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), param.num_arrays);
  for (auto s : *in_shape) {
    if (s.ndim() == 0)
      return false;
  }

  return true;
}

inline bool ResetArraysType(const NodeAttrs& attrs,
                           std::vector<int>* in_type,
                           std::vector<int>* out_type) {
  const auto& param = dmlc::get<ResetArraysParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.num_arrays);
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1)
      return false;
  }

  return true;
}

template<typename xpu>
void ResetMemory(void *pntr, size_t len, mshadow::Stream<xpu> *s);

template<typename xpu>
void ResetArrays(const nnvm::NodeAttrs& attrs,
                 const OpContext &ctx,
                 const std::vector<TBlob> &inputs,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &outputs) {
  auto s = ctx.get_stream<xpu>();
  const auto& param = nnvm::get<ResetArraysParam>(attrs.parsed);
  for (int i = 0; i < param.num_arrays; i++) {  // array index in inputs
    const size_t size = inputs[i].shape_.Size();
    MSHADOW_REAL_TYPE_SWITCH(inputs[i].type_flag_, DType,
      ResetMemory(inputs[i].FlatTo2D<xpu, DType>(s).dptr_, size * sizeof(DType), s);
    )
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_RESET_ARRAYS_INL_H_
