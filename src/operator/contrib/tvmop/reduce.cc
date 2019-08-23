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
 * Copyright (c) 2019 by Contributors
 * \file ufunc.cc
 * \brief
 * \author Yizhi Liu
 */
#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include <string>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"
#include "../../numpy/np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

static constexpr char func_sum_cpu[] = "tvm_sum";
static constexpr char func_sum_gpu[] = "cuda_tvm_sum";
// static constexpr char func_bakcward_vadd_cpu[] = "backward_vadd";
// static constexpr char func_bakcward_vadd_gpu[] = "cuda_backward_vadd";
static constexpr int max_dim = 5;

TBlob padding_reduce(const TBlob& tblob, const int max_dim) {
  TShape tshape(max_dim, 1);
  int ndim = tblob.shape_.ndim();
  for (int i = max_dim - ndim; i < max_dim; ++i) {
    tshape[i] = tblob.size(i - max_dim + ndim);
  }
  return tblob.reshape(tshape);
}

template<const char* func>
void TVMReduceCompute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mxnet;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);

  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);

  const TBlob input = inputs[0].reshape(src_shape);
  const TBlob output = outputs[0].reshape(dst_shape);

  int ndim = input.shape_.ndim();
  std::vector<int> dv, ov;
  TBlob data = padding_reduce(input, ndim);
  TBlob out = padding_reduce(output, ndim);
  int flag;
  if (data.size(0) != out.size(0)) {
    flag = 1;
  } else {
    flag = 0;
  }
  for (int i = 0; i < ndim; ++i) {
    if (i == 0 || (data.size(i) != out.size(i)) != (data.size(i - 1) != out.size(i - 1))) {
      dv.push_back(data.size(i));
    } else {
      dv.back() *= data.size(i);
    }
  }
  for (int i = dv.size(); i < max_dim; ++i) {
    dv.push_back(1);
  }
  for (int i = flag; i < dv.size(); i += 2) {
    ov.push_back(dv[i]);
  }
  TShape dshape(dv.begin(), dv.end());
  TShape oshape(ov.begin(), ov.end());
  TBlob data_tvm(data.reshape(dshape));
  TBlob output_tvm(out.reshape(oshape));
  std::string funcname = std::string(func) + "reduce1st_" + std::to_string(flag);
  // dispatch by req
  funcname += "otype_";
  switch (output.type_flag_) {
    case mshadow::kFloat16:
      funcname += "float16";
      break;
    case mshadow::kFloat32:
      funcname += "float32";
      break;
    case mshadow::kFloat64:
      funcname += "float64";
      break;
    default:
      LOG(FATAL) << "type " << output.type_flag_ << "not supported in tvm_sum";
      break;
  }
  tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx, {data_tvm, output_tvm, output_tvm});
}

NNVM_REGISTER_OP(_contrib_tvm_sum)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"a"};
      })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpySumType)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMReduceCompute<func_sum_cpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMReduceCompute<func_sum_gpu>)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr_parser(ParamParser<NumpyReduceAxesParam>)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_arguments(NumpyReduceAxesParam::__FIELDS__());

// NNVM_REGISTER_OP(_backward_contrib_tvm_vadd)
//     .set_num_inputs(1)
//     .set_num_outputs(2)
//     .set_attr<nnvm::TIsBackward>("TIsBackward", true)
// #if MXNET_USE_CUDA
//     .set_attr<mxnet::FCompute>("FCompute<gpu>",
//                                mxnet::op::TVMBinaryBackwardComputeUseNone<func_bakcward_vadd_gpu>)
// #endif  // MXNET_USE_CUDA
//     .set_attr<mxnet::FCompute>("FCompute<cpu>",
//                                mxnet::op::TVMBinaryBackwardComputeUseNone<func_bakcward_vadd_cpu>);
//
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
