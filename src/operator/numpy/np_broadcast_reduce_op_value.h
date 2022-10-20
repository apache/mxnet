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
 * \file np_broadcast_reduce_op_value.h
 * \brief Definition of broadcast and reduce functions based on value.
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_VALUE_H_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_VALUE_H_

#include <string>
#include <vector>

#if MXNET_USE_TVM_OP
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "np_broadcast_reduce_op.h"

#if MXNET_USE_ONEDNN
#include "../nn/dnnl/dnnl_reduce-inl.h"
#endif  // MXNET_USE_ONEDNN

namespace mxnet {
namespace op {

inline bool NumpySumType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    if (in_attrs->at(0) == mshadow::kBool) {
      CHECK(param.dtype.value() == mshadow::kInt32 || param.dtype.value() == mshadow::kInt64 ||
            param.dtype.value() == mshadow::kFloat32 || param.dtype.value() == mshadow::kFloat64)
          << "Only support the following output dtypes when input dtype is bool: "
             "int32, int64, float32, float64.";
    }
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else if (in_attrs->at(0) == mshadow::kBool) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

#if MXNET_USE_TVM_OP
static constexpr int max_reduce_ndim = 5;
TBlob PrependAxes(const TBlob& src, const int dst_ndim);
#endif  // MXNET_USE_TVM_OP

inline void TVMOpReduce(const OpContext& ctx,
                        const TBlob& input,
                        const dmlc::optional<mxnet::Tuple<int>>& axis,
                        const TBlob& output,
                        const OpReqType req,
                        const std::string& reducer_name) {
#if MXNET_USE_TVM_OP
  CHECK_GE(input.ndim(), output.ndim());
  CHECK_LE(input.ndim(), max_reduce_ndim)
      << "TVMOpReduce only supports ndim <= " << max_reduce_ndim;

  const TBlob expanded_output =
      (input.ndim() == output.ndim() ?
           output :
           output.reshape(NumpyReduceAxesShapeImpl(input.shape_, axis, true)));
  CHECK_EQ(input.ndim(), expanded_output.ndim());
  int reduce1st_dim = 0;
  if (input.ndim() > 0 && input.size(0) != expanded_output.size(0)) {
    reduce1st_dim = 1;
  }
  // collapse consecutive dimensions where reduction are performed or not performed
  std::vector<index_t> ishape_vec;
  for (int i = 0; i < input.ndim(); ++i) {
    if (i == 0 || ((input.size(i) != expanded_output.size(i)) !=
                   (input.size(i - 1) != expanded_output.size(i - 1)))) {
      ishape_vec.push_back(input.size(i));
    } else {
      ishape_vec.back() *= input.size(i);
    }
  }
  // append axes after collapsed ishape to reach the max ndim allowed
  for (int i = ishape_vec.size(); i < max_reduce_ndim; ++i) {
    ishape_vec.push_back(1);
  }
  std::vector<index_t> oshape_vec;
  for (size_t i = reduce1st_dim; i < ishape_vec.size(); i += 2) {
    oshape_vec.push_back(ishape_vec[i]);
  }
  TShape ishape(ishape_vec.begin(), ishape_vec.end()), oshape(oshape_vec.begin(), oshape_vec.end());
  TBlob input_tvm  = input.reshape(ishape);
  TBlob output_tvm = output.reshape(oshape);
  const std::string ctx_name =
      (ctx.run_ctx.ctx.dev_type == mxnet::Context::DeviceType::kCPU) ? "cpu" : "gpu";
  std::ostringstream func_name;
  func_name << reducer_name << "_"
            << (ctx.run_ctx.ctx.dev_type == mxnet::Context::DeviceType::kCPU ? "cpu" : "gpu")
            << "reduce1st_dim_" << reduce1st_dim << "req_"
            << (req == kWriteTo ? "kWriteTo" : "kAddTo");
  tvm::runtime::TVMOpModule::Get()->Call(func_name.str(), ctx, {input_tvm, output_tvm, output_tvm});
#else
  LOG(FATAL) << "Please add USE_TVM_OP=1 as a compile flag to enable TVM-generated kernels.";
#endif  // MXNET_USE_TVM_OP
}

inline bool NumpyReduceAxesNoDTypeType(const nnvm::NodeAttrs& attrs,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

inline bool IsIntType(const int dtype) {
  return (dtype == mshadow::kUint8 || dtype == mshadow::kInt32 || dtype == mshadow::kInt8 ||
          dtype == mshadow::kInt64);
}

inline bool NumpyMeanType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else {
    if (common::is_float(in_attrs->at(0))) {
      TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
      TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
    } else {
      TYPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::common::GetDefaultDtype());
    }
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

inline bool NumpyBroadcastToShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector* in_attrs,
                                  mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::shape_is_known(ishape))
    return false;
  const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
  CHECK_LE(ishape.ndim(), param.shape.ndim())
      << "shape " << ishape << " is not broadcastable to " << param.shape;
  TShape pshape = param.shape;
  for (int i = param.shape.ndim() - 1; i >= 0; --i) {
    int j = i - param.shape.ndim() + ishape.ndim();
    if (j < 0)
      break;
    if (pshape[i] == -2) {
      pshape[i] = ishape[j];
    }
    CHECK(ishape[j] == pshape[i] || ishape[j] == 1)
        << "shape " << ishape << " is not broadcastable to " << pshape;
  }
  CHECK(mxnet::shape_is_known(pshape))
      << "the objective shape for broadcasting array must be known";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, pshape);
  return true;
}

#if MXNET_USE_ONEDNN == 1
template <dnnl::algorithm reduction_alg>
static void DNNLReduceEx(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (SupportDNNLReduce<NumpyReduceAxesParam>(attrs, inputs[0], outputs[0])) {
    DNNLRun(DNNLReduceForward<NumpyReduceAxesParam, reduction_alg>,
            attrs,
            ctx,
            inputs[0],
            req[0],
            outputs[0]);
    return;
  } else {
    constexpr bool normalize = reduction_alg == dnnl::algorithm::reduction_mean;
    FallBackCompute(NumpyReduceAxesCompute<cpu, mshadow_op::sum, true, normalize>,
                    attrs,
                    ctx,
                    inputs,
                    req,
                    outputs);
    return;
  }
}

inline static bool NumpyReduceAxesStorageType(const nnvm::NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs) {
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);

  bool onednn_disptach = true;
  if (param.dtype.has_value()) {
    onednn_disptach = param.dtype.value() == mshadow::kFloat32;
  }

  return DNNLStorageType(attrs, dev_mask, onednn_disptach, dispatch_mode, in_attrs, out_attrs);
}
#endif

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_VALUE_H_
