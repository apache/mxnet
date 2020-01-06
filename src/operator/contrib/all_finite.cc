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
 * \file all_finite.cc 
 * \brief operator for checking if a group of array is all finite
 * \author Clement Fuji Tsang
 */
#include "./all_finite-inl.h"
#include <cmath>

namespace mxnet {
namespace op {

template<typename DType>
struct AllFiniteCPUKernel {
  MSHADOW_XINLINE static void Map(int i, const DType* in, float* out) {
    bool is_finite = true;
    is_finite = std::isfinite(static_cast<float>(in[i]))  ? is_finite : false;
    if (!is_finite) {
      out[0] = 0.;
    }
  }
};

inline void AllFiniteCPU(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<cpu>* s = ctx.get_stream<cpu>();
  const AllFiniteParam& op_param = nnvm::get<AllFiniteParam>(attrs.parsed);
  Tensor<cpu, 2, float> out = outputs[0].FlatTo2D<cpu, float>(s);
  if (op_param.init_output) {
    out = 1.;
  }
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<cpu, 2, DType> in = inputs[0].FlatTo2D<cpu, DType>(s);
    const int n = in.shape_.Size();
    Kernel<AllFiniteCPUKernel<DType>, cpu>::Launch(s, n, in.dptr_, out.dptr_);
  });
}

template<typename DType>
struct MultiAllFiniteCPUKernel {
  MSHADOW_XINLINE static void Map(int i, const MultiAllFiniteKernelParam<DType> param,
                                  float* out) {
    bool is_finite = true;
    for (int index = 0; index < param.count; ++index) {
      if ((size_t)i < param.sizes[index]) {
        is_finite = std::isfinite(static_cast<float>(param.arrays[index][i])) ? is_finite : false;
      }
    }
    if (!is_finite) {
      out[0] = 0.;
    }
  }
};

inline void MultiAllFiniteCPU(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<cpu>* s = ctx.get_stream<cpu>();
  const MultiAllFiniteParam& op_param = nnvm::get<MultiAllFiniteParam>(attrs.parsed);
  Tensor<cpu, 2, float> out = outputs[0].FlatTo2D<cpu, float>(s);
  if (op_param.init_output)
    out = 1.;
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MultiAllFiniteKernelParam<DType> param =
      FillMultiAllFiniteParam<cpu, DType>(op_param, ctx, inputs);
    Kernel<MultiAllFiniteCPUKernel<DType>, cpu>::Launch(s, param.max_size,
                                                       param, out.dptr_);
  });
}

DMLC_REGISTER_PARAMETER(AllFiniteParam);

NNVM_REGISTER_OP(all_finite)
.describe(R"code(Check if all the float numbers in the array are finite (used for AMP)
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AllFiniteParam>)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<TShape> *in_attrs,
     std::vector<TShape> *out_attrs){
    (*out_attrs)[0] = TShape({1});
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_attrs,
     std::vector<int> *out_attrs){
    (*out_attrs)[0] = mshadow::kFloat32;
    return true;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    std::vector<std::string> ret;
    ret.emplace_back("data");
    return ret;
  })
.add_argument("data", "NDArray", "Array")
.add_arguments(AllFiniteParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", AllFiniteCPU);

DMLC_REGISTER_PARAMETER(MultiAllFiniteParam);

NNVM_REGISTER_OP(multi_all_finite)
.describe(R"code(Check if all the float numbers in all the arrays are finite (used for AMP)
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const MultiAllFiniteParam& param = dmlc::get<MultiAllFiniteParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_arrays);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<MultiAllFiniteParam>)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<TShape> *in_attrs,
     std::vector<TShape> *out_attrs) {
    (*out_attrs)[0] = TShape({1});
    return true;
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_attrs,
     std::vector<int> *out_attrs) {
    (*out_attrs)[0] = mshadow::kFloat32;
    return true;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiAllFiniteParam>(attrs.parsed).num_arrays;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("array_") + std::to_string(i));
    }
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol[]", "Arrays")
.add_arguments(MultiAllFiniteParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", MultiAllFiniteCPU);

}  // namespace op
}  // namespace mxnet
