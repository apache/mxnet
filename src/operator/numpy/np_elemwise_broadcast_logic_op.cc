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
 * \file np_elemwise_binary_logic_op.cc
 * \brief CPU Implementation of basic logic functions for elementwise numpy binary
 * broadcast operator.
 */

#if MXNET_USE_TVM_OP
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
static constexpr char func_equal_cpu[] = "equal_cpu";
static constexpr char func_equal_gpu[] = "equal_gpu";
static constexpr char func_not_equal_cpu[] = "not_equal_cpu";
static constexpr char func_not_equal_gpu[] = "not_equal_gpu";
static constexpr char func_greater_cpu[] = "greater_cpu";
static constexpr char func_greater_gpu[] = "greater_gpu";
static constexpr char func_less_cpu[] = "less_cpu";
static constexpr char func_less_gpu[] = "less_gpu";
static constexpr char func_greater_equal_cpu[] = "greater_equal_cpu";
static constexpr char func_greater_equal_gpu[] = "greater_equal_gpu";
static constexpr char func_less_equal_cpu[] = "less_equal_cpu";
static constexpr char func_less_equal_gpu[] = "less_equal_gpu";
static constexpr char func_logical_and_cpu[] = "logical_and_cpu";
static constexpr char func_logical_and_gpu[] = "logical_and_gpu";
static constexpr char func_logical_or_cpu[] = "logical_or_cpu";
static constexpr char func_logical_or_gpu[] = "logical_or_gpu";
static constexpr char func_logical_xor_cpu[] = "logical_xor_cpu";
static constexpr char func_logical_xor_gpu[] = "logical_xor_gpu";
#pragma clang diagnostic pop

bool NumpyBinaryLogicOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (in_attrs->at(0) == -1 && in_attrs->at(1) == -1) return false;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return true;
}

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

struct TVMBinaryBroadcastCompute {
  const char* func;
  std::string OP;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

    // prepare tblobs and TVMArgs
    std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0]};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;

    const TBlob& a = inputs[0];
    const TBlob& b = inputs[1];

#if MXNET_USE_CUDA

    if (a.type_flag_ != b.type_flag_) {
      BinaryBroadcastRTCCompute mixedTypeCompute{OP};
      mixedTypeCompute(attrs, ctx, inputs, req, outputs);
      return;
    }

#endif  // MXNET_USE_CUDA

    const int ondim = outputs[0].shape_.ndim();
    const size_t num_args = inputs.size() + outputs.size();
    type_codes.resize(num_args);
    values.resize(num_args);
    for (size_t i = 0; i < num_args; ++i) {
      tblobs[i] = PrependAxes(tblobs[i], ondim);
      type_codes[i] = kTVMDLTensorHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
    }
    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], tblobs.size());
    tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

template<typename xpu, typename OP>
struct GetBinaryBroadcastCompute {
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_TVM_OP
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    if (outputs[0].shape_.Size() == 0U) return;  // skip zero-size tensor

    // prepare tblobs and TVMArgs
    std::vector<TBlob> tblobs = {inputs[0], inputs[1], outputs[0]};
    std::vector<int> type_codes;
    std::vector<TVMValue> values;

    const TBlob& a = inputs[0];
    const TBlob& b = inputs[1];
    if (a.type_flag_ != b.type_flag_) {
      if (outputs[0].shape_.Size() == 0U) return;
      mxnet::TShape new_lshape, new_rshape, new_oshape;
      const TBlob& lhs = inputs[0];
      const TBlob& rhs = inputs[1];
      const TBlob& out = outputs[0];
      int ndim = BinaryBroadcastShapeCompact(lhs.shape_, rhs.shape_, out.shape_,
                                            &new_lshape, &new_rshape, &new_oshape);
      if (!ndim) {
        ElemwiseBinaryOp::ComputeLogic<xpu, OP>(attrs, ctx, inputs, req, outputs);
      } else {
        if (req[0] == kNullOp) return;
        mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
        MSHADOW_TYPE_SWITCH_WITH_BOOL(lhs.type_flag_, DType, {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(rhs.type_flag_, EType, {
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
                                lhs.dptr<DType>(), rhs.dptr<EType>(),
                                out.dptr<bool>());
            });
          });
        });
      }
      return;
    }

    const int ondim = outputs[0].shape_.ndim();
    const size_t num_args = inputs.size() + outputs.size();
    type_codes.resize(num_args);
    values.resize(num_args);
    for (size_t i = 0; i < num_args; ++i) {
      tblobs[i] = PrependAxes(tblobs[i], ondim);
      type_codes[i] = kTVMDLTensorHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(tblobs[i].dltensor()));
    }
    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], tblobs.size());
    tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(name)                                 \
  NNVM_REGISTER_OP(_npi_##name)                                                       \
  .set_num_inputs(2)                                                                  \
  .set_num_outputs(1)                                                                 \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                                 \
  [](const NodeAttrs& attrs) {                                                        \
    return std::vector<std::string>{"lhs", "rhs"};                                    \
  })                                                                                  \
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)                  \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryLogicOpType)                   \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
  [](const NodeAttrs& attrs) {                                                        \
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                         \
  })                                                                                  \
  .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                          \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")            \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC(logical_xor);

#if MXNET_USE_TVM_OP

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(name)                          \
  NNVM_REGISTER_OP(_npi_##name)                                                    \
  .set_attr<FCompute>("FCompute<cpu>", GetBinaryBroadcastCompute<cpu,              \
                      mshadow_op::np_##name>{func_##name##_cpu})

#if MXNET_USE_CUDA

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(name)                          \
  NNVM_REGISTER_OP(_npi_##name)                                                    \
  .set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastCompute{func_##name##_gpu, "np_" #name})\

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(logical_xor);

#endif  // MXNET_USE_CUDA

#else

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(name)                                     \
  NNVM_REGISTER_OP(_npi_##name)                                                               \
  .set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastComputeLogic<cpu, mshadow_op::np_##name>)

#endif  // MXNET_USE_TVM_OP

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_CPU(logical_xor);

bool NumpyBinaryScalarLogicOpType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (in_attrs->at(0) == -1) return false;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kBool);
  return true;
}

struct TVMBinaryBroadcastScalarCompute {
  const char* func;
  void operator()(const nnvm::NodeAttrs& attrs,
                  const mxnet::OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
#if MXNET_USE_TVM_OP
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
    type_codes[0] = kTVMDLTensorHandle;
    values[0].v_handle = const_cast<DLTensor*>(&(tblobs[0].dltensor()));

    // scalar param
    type_codes[1] = kDLFloat;
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    values[1].v_float64 = param.scalar;

    // output tensor
    type_codes[2] = kTVMDLTensorHandle;
    values[2].v_handle = const_cast<DLTensor*>(&(tblobs[1].dltensor()));

    tvm::runtime::TVMArgs tvm_args(&values[0], &type_codes[0], 3);
    tvm::runtime::TVMOpModule::Get()->CallEx(func, ctx, tblobs, tvm_args);
#else
    LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag for compiling MXNet source code "
                  "to enable TVM-generated kernels for operator " << func;
#endif  // MXNET_USE_TVM_OP
  }
};

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                    \
  .set_num_inputs(1)                                                                        \
  .set_num_outputs(1)                                                                       \
  .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)                                     \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                                       \
  [](const NodeAttrs& attrs) {                                                              \
    return std::vector<std::string>{"data"};                                                \
  })                                                                                        \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                         \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarLogicOpType)                   \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                         \
  [](const NodeAttrs& attrs) {                                                              \
    return std::vector<std::pair<int, int> >{{0, 0}};                                       \
  })                                                                                        \
  .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                                \
  .add_argument("data", "NDArray-or-Symbol", "First input to the function")                 \
  .add_arguments(NumpyBinaryScalarParam::__FIELDS__())

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC(logical_xor);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
static constexpr char func_equal_scalar_cpu[] = "equal_scalar_cpu";
static constexpr char func_equal_scalar_gpu[] = "equal_scalar_gpu";
static constexpr char func_not_equal_scalar_cpu[] = "not_equal_scalar_cpu";
static constexpr char func_not_equal_scalar_gpu[] = "not_equal_scalar_gpu";
static constexpr char func_greater_scalar_cpu[] = "greater_scalar_cpu";
static constexpr char func_greater_scalar_gpu[] = "greater_scalar_gpu";
static constexpr char func_less_scalar_cpu[] = "less_scalar_cpu";
static constexpr char func_less_scalar_gpu[] = "less_scalar_gpu";
static constexpr char func_greater_equal_scalar_cpu[] = "greater_equal_scalar_cpu";
static constexpr char func_greater_equal_scalar_gpu[] = "greater_equal_scalar_gpu";
static constexpr char func_less_equal_scalar_cpu[] = "less_equal_scalar_cpu";
static constexpr char func_less_equal_scalar_gpu[] = "less_equal_scalar_gpu";
static constexpr char func_logical_and_scalar_cpu[] = "logical_and_scalar_cpu";
static constexpr char func_logical_and_scalar_gpu[] = "logical_and_scalar_gpu";
static constexpr char func_logical_or_scalar_cpu[] = "logical_or_scalar_cpu";
static constexpr char func_logical_or_scalar_gpu[] = "logical_or_scalar_gpu";
static constexpr char func_logical_xor_scalar_cpu[] = "logical_xor_scalar_cpu";
static constexpr char func_logical_xor_scalar_gpu[] = "logical_xor_scalar_gpu";
#pragma clang diagnostic pop

#if MXNET_USE_TVM_OP

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                        \
  .set_attr<FCompute>("FCompute<cpu>", TVMBinaryBroadcastScalarCompute{func_##name##_scalar_cpu})

#if MXNET_USE_CUDA

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(name)                                \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                        \
  .set_attr<FCompute>("FCompute<gpu>", TVMBinaryBroadcastScalarCompute{func_##name##_scalar_gpu})

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(logical_xor);

#endif  // MXNET_USE_CUDA

#else

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(name)                                  \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                          \
  .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::ComputeLogic<cpu, mshadow_op::np_##name>)  \
  .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {                        \
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};                             \
  })

#endif  // MXNET_USE_TVM_OP

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(less_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(logical_and);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(logical_or);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_CPU(logical_xor);

}  // namespace op
}  // namespace mxnet
