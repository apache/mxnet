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
 * \file dnnl_bn_relu.cc
 * \brief
 * \author Xinyu Chen
 */

#include <nnvm/op_attr_types.h>

#include "operator/elemwise_op_common.h"
#include "operator/nn/batch_norm-inl.h"
#include "operator/operator_common.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_batch_norm-inl.h"
#endif

namespace mxnet {
namespace op {

namespace batchnormrelu {

enum BatchNormWithReLUOpInputs {
  kData,
  kGamma,
  kBeta,
  kInMovingMean,
  kInMovingVar
};  // kGamma: weights, kBeta: biases
enum BatchNormWithReLUOpOutputs { kOut, kMean, kVar, kWorkspace };  // req, out_data
enum BatchNormWithReLUOpResource { kTempSpace };
enum BatchNormWithReLUOpAuxiliary { kMovingMean, kMovingVar };  // aux_states

/*! \brief Default channel axis if none specified in the params */
constexpr int DEFAULT_AXIS = 1;
}  // namespace batchnormrelu

static bool BatchNormWithReLUShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector* in_shape,
                                   mxnet::ShapeVector* out_shape) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 5U) << "Input:[data, gamma, beta, MovingMean, MovingVar]";
  CHECK_EQ(out_shape->size(), 4U);
  const mxnet::TShape& dshape = in_shape->at(batchnormrelu::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  const size_t channelAxis = static_cast<size_t>(
      param.axis < 0 ? static_cast<int>(dshape.ndim()) + param.axis : param.axis);
  CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param.axis;

  const int channelCount = dshape[channelAxis];

  in_shape->at(batchnormrelu::kGamma)        = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnormrelu::kBeta)         = mxnet::TShape(Shape1(channelCount));
  in_shape->at(batchnormrelu::kInMovingMean) = mxnet::TShape(Shape1(channelCount));  // kMovingMean
  in_shape->at(batchnormrelu::kInMovingVar)  = mxnet::TShape(Shape1(channelCount));  // kMovingVar

  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  out_shape->push_back(Shape1(channelCount));  // kMean
  out_shape->push_back(Shape1(channelCount));  // kVar
  out_shape->push_back(dshape);                // kWorkspace
  return true;
}

static bool BatchNormWithReLUType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_type,
                                  std::vector<int>* out_type) {
  using namespace mshadow;
  CHECK_GE(in_type->size(), 1U);
  const size_t n_out = 4;
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param;
  int dtype = (*in_type)[0];

  if (type_is_none(dtype)) {
    // Input type is undefined, we try backward inference
    if (out_type->size() == 0 || type_is_none((*out_type)[0])) {
      // Neither the input nor the output are defined,
      // types cannot be infered for this op
      return false;
    } else {
      // Input type is undefined but output type is: backward inference
      dtype         = (*out_type)[0];
      (*in_type)[0] = dtype;
      MSHADOW_REAL_TYPE_SWITCH_EX(
          dtype, DTypeX, AccRealX, { dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    }
  } else {
    // Input type is defined but output type is not: forward inference
    MSHADOW_REAL_TYPE_SWITCH_EX(
        dtype, DTypeX, AccRealX, { dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    out_type->clear();
    out_type->push_back(dtype);
    for (size_t i = 1; i < n_out; ++i) {
      out_type->push_back(dtype_param);
    }
  }
  std::vector<std::string> args{"data", "gamma", "beta", "mean", "var"};
  CHECK_LE(in_type->size(), args.size());
  for (size_t i = 1; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  return true;
}

#if MXNET_USE_ONEDNN == 1
// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_batch_normalization.html
static inline bool SupportDNNLBNReLU(const NDArray& input) {
  return SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(input) && !mxnet::op::batchnorm::disable_mkl;
}

void BatchNormWithReLUComputeExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 5U);
  if (SupportDNNLBNReLU(inputs[0])) {
    CHECK_GT(outputs.size(), 3U);
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLBatchNormForward</*fuse_relu*/ true>, attrs, ctx, inputs, req, outputs);
    return;
  }
  LOG(FATAL) << "BatchNormWithReLU operator only supports oneDNN Backend.";
}

#endif

static inline bool BatchNormWithReLUStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int>* in_attrs,
                                                std::vector<int>* out_attrs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);

  bool dispatched = false;
#if MXNET_USE_ONEDNN == 1
  if (!dispatched) {
    dispatched = DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
  }
  if (!DNNLEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  }
#else
  for (int& v : *in_attrs)
    if (v == -1)
      v = kDefaultStorage;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched =
        storage_type_assign(out_attrs, kDefaultStorage, dispatch_mode, DispatchMode::kFCompute);
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

NNVM_REGISTER_OP(_sg_onednn_batch_norm)
    .describe(R"code(Batch normalization with ReLU fusion.

An extented operator of Batch normalization which can fuse ReLU activation.

)code" ADD_FILELINE)
    .set_num_inputs(5)
    .set_num_outputs(4)
    .set_attr_parser(ParamParser<BatchNormParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"data", "gamma", "beta", "moving_mean", "moving_var"};
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "mean", "var", "workspace"};
        })
    .set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                        [](const NodeAttrs& attrs) {
                                          const BatchNormParam& param =
                                              nnvm::get<BatchNormParam>(attrs.parsed);
                                          return param.output_mean_var ? 3 : 1;
                                        })
    .set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                   [](const nnvm::NodeAttrs& attrs) {
                                     return std::vector<uint32_t>{3, 4};
                                   })
    .set_attr<mxnet::FInferShape>("FInferShape", BatchNormWithReLUShape)
    .set_attr<nnvm::FInferType>("FInferType", BatchNormWithReLUType)
    .set_attr<FInferStorageType>("FInferStorageType", BatchNormWithReLUStorageType)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", BatchNormWithReLUComputeExCPU)
#endif
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
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
          if (var->attrs.dict.find("__init__") != var->attrs.dict.end())
            return;
          if (index == 3) {
            var->attrs.dict["__init__"] = "[\"zero\", {}]";
          } else if (index == 4) {
            var->attrs.dict["__init__"] = "[\"one\", {}]";
          }
        });

}  // namespace op
}  // namespace mxnet
