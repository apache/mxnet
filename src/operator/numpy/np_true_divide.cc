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
 * \file np_true_divide.cc
 * \brief CPU Implementation of true_divide operator.
 */
#if MXNET_USE_TVM_OP
#include "np_true_divide.h"
#include <mxnet/base.h>
#include <string>
#include <utility>
#include <vector>
#include "../tvmop/op_module.h"

namespace mxnet {
namespace op {

template <int num_inputs>
bool TrueDivideType(const nnvm::NodeAttrs& attrs,
                    std::vector<int>* in_attrs,
                    std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(num_inputs));
  CHECK_EQ(out_attrs->size(), 1U);
  for (const int dtype : *in_attrs) {
    if (dtype == -1) return false;
  }
  if (num_inputs == 2) {
    const int lhs_dtype = in_attrs->at(0);
    const int rhs_dtype = in_attrs->at(1);
    CHECK_EQ(lhs_dtype, rhs_dtype)
      << "_true_divide currently only supports same dtype for dividend and divisor";
  }
  if (IsIntType(in_attrs->at(0))) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  }
  return true;
}

static constexpr char func_true_divide_cpu[] = "true_divide";
static constexpr char func_true_divide_gpu[] = "cuda_true_divide";
static constexpr char func_true_divide_scalar_cpu[] = "true_divide_scalar";
static constexpr char func_true_divide_scalar_gpu[] = "cuda_true_divide_scalar";
static constexpr char func_rtrue_divide_scalar_cpu[] = "rtrue_divide_scalar";
static constexpr char func_rtrue_divide_scalar_gpu[] = "cuda_rtrue_divide_scalar";

TBlob PrependAxes(const TBlob& src, const int dst_ndim) {
  CHECK_LE(src.shape_.ndim(), dst_ndim);
  const int src_ndim = src.shape_.ndim();
  if (src_ndim == dst_ndim) return src;
  mxnet::TShape dst_shape(dst_ndim, 1);
  for (int i = dst_ndim - src_ndim; i < dst_ndim; ++i) {
    dst_shape[i] = src.shape_[i - dst_ndim + src_ndim];
  }
  return src.reshape(dst_shape);
}

template<const char* func>
void TVMBinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                               const mxnet::OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  const int ondim = outputs[0].shape_.ndim();
  const size_t num_args = inputs.size() + outputs.size();
  type_codes.resize(num_args);
  values.resize(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    tblobs[i] = PrependAxes(tblobs[i], ondim);
    type_codes[i] = kArrayHandle;
    values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
  }
  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], tblobs.size());
  tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
}

template<const char* func>
void TVMBinaryBroadcastScalarCompute(const nnvm::NodeAttrs& attrs,
                                     const mxnet::OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

  // prepare tblobs and TVMArgs
  std::vector<TBlob> tblobs = {inputs[0], outputs[0]};
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  const size_t num_args = 3;  // one input tensor, one scalar param, and one output
  type_codes.resize(num_args);
  values.resize(num_args);

  // input tensor setup
  type_codes[0] = kArrayHandle;
  values[0].v_handle = const_cast<DLTensor*>(&(tblobs[0].dltensor()));

  // scalar param
  type_codes[1] = kDLFloat;
  values[1].v_float64 = nnvm::get<double>(attrs.parsed);

  // output tensor
  type_codes[2] = kArrayHandle;
  values[2].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

  tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], 3);
  tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
}

NNVM_REGISTER_OP(_npi_true_divide)
.describe(R"code(
Returns a true division of the inputs, element-wise.

It currently only supports dtype float16, float32, and float64.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   _true_divide(x, y) = [[ 3.,  3.,  3.],
                         [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<2>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
  })
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMBinaryBroadcastCompute<func_true_divide_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMBinaryBroadcastCompute<func_true_divide_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_true_divide"})
.add_argument("lhs", "NDArray-or-Symbol", "Dividend array")
.add_argument("rhs", "NDArray-or-Symbol", "Divisor array");

NNVM_REGISTER_OP(_backward_true_divide)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseInFloat<cpu, mshadow_op::div_grad,
                                                                       mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_npi_true_divide_scalar)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>",
                    mxnet::op::TVMBinaryBroadcastScalarCompute<func_true_divide_scalar_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>",
                    mxnet::op::TVMBinaryBroadcastScalarCompute<func_true_divide_scalar_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_true_divide_scalar"})
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "float", "scalar input");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_backward_true_divide_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOpComputeFloat<cpu, mshadow_op::div>);

NNVM_REGISTER_OP(_npi_rtrue_divide_scalar)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
  attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>",
                    mxnet::op::TVMBinaryBroadcastScalarCompute<func_rtrue_divide_scalar_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>",
                    mxnet::op::TVMBinaryBroadcastScalarCompute<func_rtrue_divide_scalar_cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rtrue_divide_scalar"})
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "float", "scalar input");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_backward_rtrue_divide_scalar)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOpBackwardFloat<cpu, mshadow_op::rdiv_grad>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
