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
 * Copyright (c) 2020 by Contributors
 * \file batch_norm_relu.cc
 * \brief
 * \author Xinyu Chen
*/

#include "../nn/batch_norm-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_batch_norm-inl.h"
#endif

namespace mxnet {
namespace op {

namespace batchnormrelu {

enum BatchNormWithReLUOpInputs {kData, kGamma, kBeta, kInMovingMean,
  kInMovingVar};  // kGamma: weights, kBeta: biases
enum BatchNormWithReLUOpOutputs {kOut, kMean, kVar, kWorkspace};  // req, out_data
enum BatchNormWithReLUOpResource {kTempSpace};
enum BatchNormWithReLUOpAuxiliary {kMovingMean, kMovingVar};  // aux_states

/*! \brief Default channel axis if none specified in the params */
constexpr int DEFAULT_AXIS = 1;
}  // namespace batchnormrelu

static bool BatchNormWithReLUShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector *in_shape,
                                   mxnet::ShapeVector *out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, MovingMean, MovingVar]";
  CHECK_EQ(out_shape->size(), 4U);
  const mxnet::TShape &dshape = in_shape->at(batchnormrelu::kData);

  const size_t channelAxis = static_cast<size_t>(param.axis < 0
      ? static_cast<int>(dshape.ndim()) + param.axis
      : param.axis);
  CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param.axis;

  const int channelCount = dshape[channelAxis];

  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  in_shape->at(batchnormrelu::kGamma) = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnormrelu::kBeta) = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnormrelu::kInMovingMean) = mxnet::TShape(Shape1(channelCount));  // kMovingMean
  in_shape->at(batchnormrelu::kInMovingVar) = mxnet::TShape(Shape1(channelCount));  // kMovingVar

  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  out_shape->push_back(Shape1(channelCount));  // kMean
  out_shape->push_back(Shape1(channelCount));  // kVar
  out_shape->push_back(dshape);                // kWorkspace
  return true;
}

static bool BatchNormWithReLUType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_type, std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_GE(in_type->size(), 1U);
  const int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  std::vector<std::string> args{"data", "gamma", "beta", "mean", "var"};
  CHECK_LE(in_type->size(), args.size());
  for (size_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  const size_t n_out = 4;
  out_type->clear();
  out_type->push_back(dtype);
  for (size_t i = 1; i < n_out; ++i) {
    out_type->push_back(dtype_param);
  }
  return true;
}

#if MXNET_USE_MKLDNN == 1
static inline bool SupportMKLDNNBNReLU(const NDArray &input, const BatchNormParam &param) {
  mxnet::TShape shape = input.shape();
  return SupportMKLDNN(input) && shape.ndim() == 4
      && param.axis == mxnet::op::batchnormrelu::DEFAULT_AXIS;
}

void BatchNormWithReLUComputeExCPU(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 5U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  bool fuse_relu = true;
  if (SupportMKLDNNBNReLU(inputs[0], param)) {
    CHECK_GT(outputs.size(), 3U);
    MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    MKLDNN_REAL_TYPE_SWITCH(inputs[0].dtype(), DTYPE, {
      MKLDNNBatchNormForward<DTYPE>(attrs, ctx, inputs, req, outputs, fuse_relu);
    });
    return;
  }
  LOG(FATAL) << "BatchNormWithReLU operator only supports MKL-DNN Backend.";
}

void BatchNormWithReLUGradComputeExCPU(const nnvm::NodeAttrs &attrs,
                                       const OpContext &ctx,
                                       const std::vector<NDArray> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<NDArray> &outputs) {
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  bool fuse_relu = true;
  if (SupportMKLDNNBNReLU(inputs[0], param)) {
      CHECK_EQ(inputs.size(), 9U);
      MKLDNN_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
      MKLDNNBatchNormBackward<float>(attrs, ctx, inputs, req, outputs, fuse_relu);
      return;
  }
  LOG(FATAL) << "BatchNormWithReLU operator only supports MKL-DNN Backend.";
}
#endif

static inline bool BatchNormWithReLUStorageType(const nnvm::NodeAttrs &attrs,
                                                const int dev_mask,
                                                DispatchMode *dispatch_mode,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);

  bool dispatched = false;
#if MXNET_USE_MKLDNN == 1
  if (!dispatched) {
    dispatched = MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode,
                                   in_attrs, out_attrs);
  }
  if (!MKLDNNEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  }
#else
  for (int& v : *in_attrs)
    if (v == - 1) v = kDefaultStorage;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
#endif
  if (!common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) && param.fix_gamma) {
    LOG(FATAL) << "fix_gamma=True is not supported for sparse ndarrays. Tracked at #11647";
  }
  return dispatched;
}

std::vector<nnvm::NodeEntry> BatchNormWithReLUGrad(const nnvm::ObjectPtr& n,
                                                   const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> out_data;
  out_data.reserve(n->num_outputs());
  for (size_t i = 0; i < n->num_outputs(); ++i)
    out_data.emplace_back(n, i, 0);
  std::vector<nnvm::NodeEntry> heads;
  heads.reserve(9);
  heads.emplace_back(ograds.at(0));
  heads.emplace_back(out_data.at(batchnormrelu::kMean));
  heads.emplace_back(out_data.at(batchnormrelu::kVar));
  heads.emplace_back(n->inputs.at(batchnormrelu::kData));
  heads.emplace_back(n->inputs.at(batchnormrelu::kGamma));
  heads.emplace_back(n->inputs.at(batchnormrelu::kBeta));
  heads.emplace_back(n->inputs.at(batchnormrelu::kInMovingMean));
  heads.emplace_back(n->inputs.at(batchnormrelu::kInMovingVar));
  heads.emplace_back(out_data.at(batchnormrelu::kWorkspace));

  nnvm::ObjectPtr gnode = nnvm::Node::Create();
  gnode->inputs = std::move(heads);
  gnode->control_deps.emplace_back(n);
  gnode->attrs = n->attrs;
  gnode->attrs.op = nnvm::Op::Get("_backward_contrib_BatchNormWithReLU");
  gnode->attrs.name = n->attrs.name + "_backward";
  // The input of batchnorm
  std::vector<nnvm::NodeEntry> in_grad;
  in_grad.reserve(5);
  for (size_t i = 0; i < 3; ++i)
    in_grad.emplace_back(gnode, i, 0);
  // attach no gradient node to forbid gradient on aux_state
  nnvm::ObjectPtr ng = nnvm::Node::Create();
  ng->attrs.op = Op::Get("_NoGradient");
  ng->attrs.name = "NoGradient";
  // the aux state of batchnorm
  for (size_t i = 3; i < 5; ++i)
    in_grad.emplace_back(ng);
  return in_grad;
}

NNVM_REGISTER_OP(_contrib_BatchNormWithReLU)
.describe(R"code(Batch normalization with ReLU fusion.

An extented operator of Batch normalization which can fuse ReLU activation。

)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(4)
.set_attr_parser(ParamParser<BatchNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "var", "workspace"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{3, 4};
})
.set_attr<mxnet::FInferShape>("FInferShape", BatchNormWithReLUShape)
.set_attr<nnvm::FInferType>("FInferType", BatchNormWithReLUType)
.set_attr<FInferStorageType>("FInferStorageType", BatchNormWithReLUStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormWithReLUComputeExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", BatchNormWithReLUGrad)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BatchNormParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::ObjectPtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == 3) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == 4) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_contrib_BatchNormWithReLU)
.set_num_inputs(9)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BatchNormWithReLUStorageType)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormWithReLUGradComputeExCPU)
#endif
.set_attr_parser(ParamParser<BatchNormParam>);

}  // namespace op
}  // namespace mxnet
