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
 * \file multi_sum_sq.cc
 * \brief vectorized sum or squared over multiple arrays operators
 * \author Clement Fuji Tsang, Andrei Ivanov, Moises Hernandez
 */

#include "./multi_sum_sq-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MultiSumSqParam);

NNVM_REGISTER_OP(multi_sum_sq)
.describe(R"code(Compute the sums of squares of multiple arrays
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return static_cast<uint32_t>(dmlc::get<MultiSumSqParam>(attrs.parsed).num_arrays);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<MultiSumSqParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiSumSqShape)
.set_attr<nnvm::FInferType>("FInferType", MultiSumSqType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const auto& param = dmlc::get<MultiSumSqParam>(attrs.parsed);
    const uint32_t num_args = param.num_arrays;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("array_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiSumSq<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol[]", "Arrays")
.add_arguments(MultiSumSqParam::__FIELDS__());

template<>
size_t GetRequiredStorageMultiSumSq<cpu>(const std::vector<TBlob> &inputs,
                                         int* param_max_chunks_per_tensor) {
  return 0;
}

template<typename DType>
inline void CalcSumSq(const std::vector<TBlob> &inputs, int n_inputs,
                      float *out_ptr, mshadow::Stream<cpu> *s) {
  int i;
  size_t j;
#pragma omp parallel for private(i, j)
  for (i = 0; i < n_inputs; ++i) {  // array index in inputs
    float sum = 0;
    const auto address = inputs[i].FlatTo2D<cpu, DType>(s).dptr_;
    const auto j_max = inputs[i].shape_.Size();
    for (j = 0; j < j_max; ++j)
      sum += address[j] * address[j];

    out_ptr[i] = sum;
  }
}

template<>
void MultiSumSqRun<cpu>(const std::vector<TBlob> &inputs, int n_inputs,
                        float *out_ptr, const OpContext &ctx) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType,
    CalcSumSq<DType>(inputs, n_inputs, out_ptr, ctx.get_stream<cpu>());
  )
}

}  // namespace op
}  // namespace mxnet
