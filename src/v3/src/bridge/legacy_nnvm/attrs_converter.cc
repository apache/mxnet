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
 * \file attrs.cc
 * \author Junru Shao
 */
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include "../../../../../src/operator/bilinear_sampler-inl.h"
#include "../../../../../src/operator/correlation-inl.h"
#include "../../../../../src/operator/crop-inl.h"
#include "../../../../../src/operator/grid_generator-inl.h"
#include "../../../../../src/operator/identity_attach_KL_sparse_reg-inl.h"
#include "../../../../../src/operator/image/crop-inl.h"
#include "../../../../../src/operator/image/image_random-inl.h"
#include "../../../../../src/operator/image/resize-inl.h"
#include "../../../../../src/operator/instance_norm-inl.h"
#include "../../../../../src/operator/l2_normalization-inl.h"
#include "../../../../../src/operator/leaky_relu-inl.h"
#include "../../../../../src/operator/loss_binary_op-inl.h"
#include "../../../../../src/operator/make_loss-inl.h"
#include "../../../../../src/operator/math_functions-inl.h"
#include "../../../../../src/operator/nn/activation-inl.h"
#include "../../../../../src/operator/nn/batch_norm-inl.h"
#include "../../../../../src/operator/nn/concat-inl.h"
#include "../../../../../src/operator/nn/convolution-inl.h"
#include "../../../../../src/operator/nn/ctc_loss-inl.h"
#include "../../../../../src/operator/nn/deconvolution-inl.h"
#include "../../../../../src/operator/nn/depthwise_convolution-inl.h"
#include "../../../../../src/operator/nn/dropout-inl.h"
#include "../../../../../src/operator/nn/fully_connected-inl.h"
#include "../../../../../src/operator/nn/group_norm-inl.h"
#include "../../../../../src/operator/nn/layer_norm-inl.h"
#include "../../../../../src/operator/nn/lrn-inl.h"
#include "../../../../../src/operator/nn/moments-inl.h"
#include "../../../../../src/operator/nn/pooling-inl.h"
#include "../../../../../src/operator/nn/sequence_mask-inl.h"
#include "../../../../../src/operator/nn/softmax-inl.h"
#include "../../../../../src/operator/nn/softmax_activation-inl.h"
#include "../../../../../src/operator/nn/upsampling-inl.h"
#include "../../../../../src/operator/operator_tune-inl.h"
#include "../../../../../src/operator/regression_output-inl.h"
#include "../../../../../src/operator/rnn-inl.h"
#include "../../../../../src/operator/roi_pooling-inl.h"
#include "../../../../../src/operator/sequence_last-inl.h"
#include "../../../../../src/operator/sequence_reverse-inl.h"
#include "../../../../../src/operator/softmax_output-inl.h"
#include "../../../../../src/operator/spatial_transformer-inl.h"
#include "../../../../../src/operator/svm_output-inl.h"
// #include "../../../../../src/operator/optimizer_op-inl.h"
#include "../../../../../src/operator/custom/custom-inl.h"
#include "../../../../../src/operator/custom/native_op-inl.h"
#include "../../../../../src/operator/custom/ndarray_op-inl.h"
#include "../../../../../src/operator/mkl_functions-inl.h"
#include "../../../../../src/operator/numpy/linalg/np_gesvd-inl.h"
#include "../../../../../src/operator/numpy/np_broadcast_reduce_op.h"
#include "../../../../../src/operator/numpy/np_cumsum-inl.h"
#include "../../../../../src/operator/numpy/np_dot-inl.h"
#include "../../../../../src/operator/numpy/np_init_op.h"
#include "../../../../../src/operator/numpy/np_matrix_op-inl.h"
#include "../../../../../src/operator/numpy/np_nonzero_op-inl.h"
#include "../../../../../src/operator/numpy/np_tensordot_op-inl.h"
#include "../../../../../src/operator/numpy/np_trace_op-inl.h"
#include "../../../../../src/operator/numpy/np_tril_op-inl.h"
#include "../../../../../src/operator/numpy/np_unique_op.h"
#include "../../../../../src/operator/numpy/np_window_op.h"
#include "../../../../../src/operator/numpy/random/dist_common.h"
#include "../../../../../src/operator/numpy/random/np_choice_op.h"
#include "../../../../../src/operator/numpy/random/np_multinomial_op.h"
#include "../../../../../src/operator/numpy/random/np_normal_op.h"
#include "../../../../../src/operator/numpy/random/np_uniform_op.h"
#include "../../../../../src/operator/pad-inl.h"
#include "../../../../../src/operator/random/multisample_op.h"
#include "../../../../../src/operator/random/pdf_op.h"
#include "../../../../../src/operator/random/sample_multinomial_op.h"
#include "../../../../../src/operator/random/sample_op.h"
#include "../../../../../src/operator/random/sampler.h"
#include "../../../../../src/operator/random/unique_sample_op.h"
#include "../../../../../src/operator/sequence_mask-inl.h"
#include "../../../../../src/operator/slice_channel-inl.h"
#include "../../../../../src/operator/special_functions-inl.h"
#include "../../../../../src/operator/swapaxis-inl.h"
#include "../../../../../src/operator/tensor/broadcast_reduce-inl.h"
#include "../../../../../src/operator/tensor/cast_storage-inl.h"
#include "../../../../../src/operator/tensor/diag_op-inl.h"
#include "../../../../../src/operator/tensor/dot-inl.h"
#include "../../../../../src/operator/tensor/elemwise_binary_op-inl.h"
#include "../../../../../src/operator/tensor/histogram-inl.h"
#include "../../../../../src/operator/tensor/la_op-inl.h"
#include "../../../../../src/operator/tensor/matrix_op-inl.h"
#include "../../../../../src/operator/tensor/ordering_op-inl.h"
#include "../../../../../src/operator/tensor/ravel.h"
#include "../../../../../src/operator/tensor/slice-inl.h"
#include "../../../../../src/operator/tensor/sparse_retain-inl.h"
#include "../../../../../src/operator/tensor/square_sum-inl.h"
#include "../../../../../src/operator/tensor/util/tensor_util-inl.h"
#undef Assign

#include "../../../include/bridge/legacy_nnvm.h"
#include "../../../include/op/attrs/legacy_nnvm_attrs.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {

using bridge::legacy_nnvm::NNVMCapsule;
using ir::Array;
using ir::Attrs;
using ir::Call;
using ir::CallNode;
using ir::Integer;
using ir::Op;

static Array<Integer> AsArray(const mxnet::TShape &from) {
  Array<Integer> result;
  for (const auto &item : from) {
    result.push_back(Integer(item));
  }
  return result;
}

static Array<Integer> AsArray(const mxnet::Tuple<int> &from) {
  Array<Integer> result;
  for (const auto &item : from) {
    result.push_back(Integer(item));
  }
  return result;
}

static Array<Integer> AsArray(const mxnet::Tuple<dmlc::optional<int>> &from) {
  Array<Integer> result;
  for (const auto &item : from) {
    result.push_back(Integer(item.has_value() ? item.value() : -1));
  }
  return result;
}

using FConvertAttrs = std::function<ir::Attrs(const nnvm::NodeAttrs &node_attrs)>;

NNVM_REGISTER_OP(abs).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(Activation)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ActivationParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyActivationAttrs>();
      static const char *ActivationActTypeValues[] = {"relu", "sigmoid", "softrelu", "softsign",
                                                      "tanh"};
      attrs->act_type = ActivationActTypeValues[param.act_type];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_arange).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RangeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyArangeAttrs>();
      attrs->start = static_cast<double>(param.start);
      attrs->stop = param.stop.has_value() ? param.stop.value() : -1;
      attrs->step = static_cast<double>(param.step);
      attrs->repeat = static_cast<int>(param.repeat);
      attrs->infer_range = static_cast<bool>(param.infer_range);
      attrs->ctx = param.ctx;
      static const char *_arangeDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                 "int64",   "int8",    "uint8"};
      attrs->dtype = _arangeDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(arccos).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(arccosh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(arcsin).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(arcsinh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(arctan).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(arctanh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(argmax).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxisParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyArgmaxAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(argmax_channel)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(argmin).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxisParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyArgminAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(argsort).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ArgSortParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyArgsortAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->is_ascend = static_cast<bool>(param.is_ascend);
      static const char *ArgsortDtypeValues[] = {"float16", "float32", "float64",
                                                 "int32",   "int64",   "uint8"};
      attrs->dtype = ArgsortDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(batch_dot).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DotParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyBatchDotAttrs>();
      attrs->transpose_a = static_cast<bool>(param.transpose_a);
      attrs->transpose_b = static_cast<bool>(param.transpose_b);
      static const char *Batch_dotForwardStypeValues[] = {"None", "csr", "default", "row_sparse"};
      attrs->forward_stype =
          Batch_dotForwardStypeValues[param.forward_stype.has_value() ? param.forward_stype.value()
                                                                      : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(BatchNorm).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::BatchNormParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyBatchNormAttrs>();
      attrs->eps = static_cast<double>(param.eps);
      attrs->momentum = static_cast<double>(param.momentum);
      attrs->fix_gamma = static_cast<bool>(param.fix_gamma);
      attrs->use_global_stats = static_cast<bool>(param.use_global_stats);
      attrs->output_mean_var = static_cast<bool>(param.output_mean_var);
      attrs->axis = static_cast<int>(param.axis);
      attrs->cudnn_off = static_cast<bool>(param.cudnn_off);
      attrs->min_calib_range =
          param.min_calib_range.has_value() ? param.min_calib_range.value() : -1;
      attrs->max_calib_range =
          param.max_calib_range.has_value() ? param.max_calib_range.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(batch_take)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_add)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_axis)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::BroadcastAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyBroadcastAxisAttrs>();
      attrs->axis = AsArray(param.axis);
      attrs->size = AsArray(param.size);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(broadcast_div)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_greater)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_greater_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_hypot)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_lesser)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_lesser_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::BroadcastLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyBroadcastLikeAttrs>();
      attrs->lhs_axes =
          param.lhs_axes.has_value() ? Array<Integer>() : AsArray(param.lhs_axes.value());
      attrs->rhs_axes =
          param.rhs_axes.has_value() ? Array<Integer>() : AsArray(param.rhs_axes.value());
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(broadcast_logical_and)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_logical_or)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_logical_xor)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_maximum)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_minimum)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_mod)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_mul)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_not_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_power)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_sub)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(broadcast_to)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::BroadcastToParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyBroadcastToAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(CTCLoss).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::CTCLossOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyCTCLossAttrs>();
      attrs->use_data_lengths = static_cast<bool>(param.use_data_lengths);
      attrs->use_label_lengths = static_cast<bool>(param.use_label_lengths);
      static const char *CTCLossBlankLabelValues[] = {"first", "last"};
      attrs->blank_label = CTCLossBlankLabelValues[param.blank_label];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(Cast).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::CastParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyCastAttrs>();
      static const char *CastDtypeValues[] = {"float16", "float32", "float64", "int32",
                                              "int64",   "int8",    "uint8"};
      attrs->dtype = CastDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(cbrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(ceil).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(clip).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ClipParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyClipAttrs>();
      attrs->a_min = static_cast<double>(param.a_min);
      attrs->a_max = static_cast<double>(param.a_max);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(Convolution)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ConvolutionParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyConvolutionAttrs>();
      attrs->kernel = AsArray(param.kernel);
      attrs->stride = AsArray(param.stride);
      attrs->dilate = AsArray(param.dilate);
      attrs->pad = AsArray(param.pad);
      attrs->num_filter = static_cast<int>(param.num_filter);
      attrs->num_group = static_cast<int>(param.num_group);
      attrs->workspace = static_cast<int64_t>(param.workspace);
      attrs->no_bias = static_cast<bool>(param.no_bias);
      static const char *ConvolutionCudnnTuneValues[] = {"None", "fastest", "limited_workspace",
                                                         "off"};
      attrs->cudnn_tune =
          ConvolutionCudnnTuneValues[param.cudnn_tune.has_value() ? param.cudnn_tune.value() : 0];
      attrs->cudnn_off = static_cast<bool>(param.cudnn_off);
      static const char *ConvolutionLayoutValues[] = {"None", "NCDHW", "NCHW",
                                                      "NCW",  "NDHWC", "NHWC"};
      attrs->layout = ConvolutionLayoutValues[param.layout.has_value() ? param.layout.value() : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_copy).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_copyto).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(cos).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(cosh).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(Deconvolution)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DeconvolutionParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyDeconvolutionAttrs>();
      attrs->kernel = AsArray(param.kernel);
      attrs->stride = AsArray(param.stride);
      attrs->dilate = AsArray(param.dilate);
      attrs->pad = AsArray(param.pad);
      attrs->adj = AsArray(param.adj);
      attrs->target_shape = AsArray(param.target_shape);
      attrs->num_filter = static_cast<int>(param.num_filter);
      attrs->num_group = static_cast<int>(param.num_group);
      attrs->workspace = static_cast<int64_t>(param.workspace);
      attrs->no_bias = static_cast<bool>(param.no_bias);
      static const char *DeconvolutionCudnnTuneValues[] = {"None", "fastest", "limited_workspace",
                                                           "off"};
      attrs->cudnn_tune =
          DeconvolutionCudnnTuneValues[param.cudnn_tune.has_value() ? param.cudnn_tune.value() : 0];
      attrs->cudnn_off = static_cast<bool>(param.cudnn_off);
      static const char *DeconvolutionLayoutValues[] = {"None", "NCDHW", "NCHW",
                                                        "NCW",  "NDHWC", "NHWC"};
      attrs->layout =
          DeconvolutionLayoutValues[param.layout.has_value() ? param.layout.value() : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(degrees).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(depth_to_space)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DepthToSpaceParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyDepthToSpaceAttrs>();
      attrs->block_size = static_cast<int>(param.block_size);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(diag).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DiagParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyDiagAttrs>();
      attrs->k = static_cast<int>(param.k);
      attrs->axis1 = static_cast<int>(param.axis1);
      attrs->axis2 = static_cast<int>(param.axis2);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_div_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyDivScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(dot).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DotParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyDotAttrs>();
      attrs->transpose_a = static_cast<bool>(param.transpose_a);
      attrs->transpose_b = static_cast<bool>(param.transpose_b);
      static const char *DotForwardStypeValues[] = {"None", "csr", "default", "row_sparse"};
      attrs->forward_stype =
          DotForwardStypeValues[param.forward_stype.has_value() ? param.forward_stype.value() : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(Dropout).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DropoutParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyDropoutAttrs>();
      attrs->p = static_cast<double>(param.p);
      static const char *DropoutModeValues[] = {"always", "training"};
      attrs->mode = DropoutModeValues[param.mode];
      attrs->axes = AsArray(param.axes);
      attrs->cudnn_off = param.cudnn_off.has_value() ? param.cudnn_off.value() : false;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(elemwise_add)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(elemwise_div)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(elemwise_mul)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(elemwise_sub)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(Embedding).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::EmbeddingParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyEmbeddingAttrs>();
      attrs->input_dim = static_cast<int>(param.input_dim);
      attrs->output_dim = static_cast<int>(param.output_dim);
      static const char *EmbeddingDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                   "int64",   "int8",    "uint8"};
      attrs->dtype = EmbeddingDtypeValues[param.dtype];
      attrs->sparse_grad = static_cast<bool>(param.sparse_grad);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_equal).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(erf).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(erfinv).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(exp).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(expand_dims)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ExpandDimParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyExpandDimsAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(expm1).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_eye).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::EyeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyEyeAttrs>();
      attrs->N = static_cast<int64_t>(param.N);
      attrs->M = static_cast<int64_t>(param.M);
      attrs->k = static_cast<int64_t>(param.k);
      attrs->ctx = param.ctx;
      static const char *_eyeDtypeValues[] = {"float16", "float32", "float64", "int32",
                                              "int64",   "int8",    "uint8"};
      attrs->dtype = _eyeDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(fix).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(Flatten).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(floor).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_full).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpWithScalarParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyFullAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_fullDtypeValues[] = {"float16", "float32", "float64", "int32",
                                               "int64",   "int8",    "uint8"};
      attrs->dtype = _fullDtypeValues[param.dtype];
      attrs->value = static_cast<double>(param.value);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(FullyConnected)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::FullyConnectedParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyFullyConnectedAttrs>();
      attrs->num_hidden = static_cast<int>(param.num_hidden);
      attrs->no_bias = static_cast<bool>(param.no_bias);
      attrs->flatten = static_cast<bool>(param.flatten);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(gamma).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(gammaln).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(gather_nd).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_grad_add).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_greater).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_greater_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_greater_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyGreaterEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_greater_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyGreaterScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(GroupNorm).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::GroupNormParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyGroupNormAttrs>();
      attrs->num_groups = static_cast<int>(param.num_groups);
      attrs->eps = static_cast<double>(param.eps);
      attrs->output_mean_var = static_cast<bool>(param.output_mean_var);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(hard_sigmoid)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::HardSigmoidParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyHardSigmoidAttrs>();
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->beta = static_cast<double>(param.beta);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_hypot).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(_hypot_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyHypotScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(LRN).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LRNParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLRNAttrs>();
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->beta = static_cast<double>(param.beta);
      attrs->knorm = static_cast<double>(param.knorm);
      attrs->nsize = static_cast<int>(param.nsize);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(LayerNorm).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LayerNormParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLayerNormAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->eps = static_cast<double>(param.eps);
      attrs->output_mean_var = static_cast<bool>(param.output_mean_var);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(LeakyReLU).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LeakyReLUParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLeakyReLUAttrs>();
      static const char *LeakyReLUActTypeValues[] = {"elu",   "gelu",  "leaky",
                                                     "prelu", "rrelu", "selu"};
      attrs->act_type = LeakyReLUActTypeValues[param.act_type];
      attrs->slope = static_cast<double>(param.slope);
      attrs->lower_bound = static_cast<double>(param.lower_bound);
      attrs->upper_bound = static_cast<double>(param.upper_bound);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_lesser).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(_lesser_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_lesser_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyLesserEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_lesser_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyLesserScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_det)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_extractdiag)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaDiagParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgExtractdiagAttrs>();
      attrs->offset = static_cast<int>(param.offset);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_extracttrian)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaTrianParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgExtracttrianAttrs>();
      attrs->offset = static_cast<int>(param.offset);
      attrs->lower = static_cast<bool>(param.lower);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_gelqf)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_gemm)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaMatrixMacParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgGemmAttrs>();
      attrs->transpose_a = static_cast<bool>(param.transpose_a);
      attrs->transpose_b = static_cast<bool>(param.transpose_b);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->beta = static_cast<double>(param.beta);
      attrs->axis = static_cast<int>(param.axis);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_gemm2)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaMatrixMultParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgGemm2Attrs>();
      attrs->transpose_a = static_cast<bool>(param.transpose_a);
      attrs->transpose_b = static_cast<bool>(param.transpose_b);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->axis = static_cast<int>(param.axis);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_inverse)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_makediag)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaDiagParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgMakediagAttrs>();
      attrs->offset = static_cast<int>(param.offset);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_maketrian)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaTrianParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgMaketrianAttrs>();
      attrs->offset = static_cast<int>(param.offset);
      attrs->lower = static_cast<bool>(param.lower);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_potrf)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_potri)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_slogdet)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_sumlogdiag)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_syevd)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_linalg_syrk)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaSyrkParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgSyrkAttrs>();
      attrs->transpose = static_cast<bool>(param.transpose);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_trmm)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaTriangMatrixMultParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgTrmmAttrs>();
      attrs->transpose = static_cast<bool>(param.transpose);
      attrs->rightside = static_cast<bool>(param.rightside);
      attrs->lower = static_cast<bool>(param.lower);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linalg_trsm)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::LaTriangMatrixMultParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinalgTrsmAttrs>();
      attrs->transpose = static_cast<bool>(param.transpose);
      attrs->rightside = static_cast<bool>(param.rightside);
      attrs->lower = static_cast<bool>(param.lower);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(LinearRegressionOutput)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RegressionOutputParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinearRegressionOutputAttrs>();
      attrs->grad_scale = static_cast<double>(param.grad_scale);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_linspace).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RangeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLinspaceAttrs>();
      attrs->start = static_cast<double>(param.start);
      attrs->stop = param.stop.has_value() ? param.stop.value() : -1;
      attrs->step = static_cast<double>(param.step);
      attrs->repeat = static_cast<int>(param.repeat);
      attrs->infer_range = static_cast<bool>(param.infer_range);
      attrs->ctx = param.ctx;
      static const char *_linspaceDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                   "int64",   "int8",    "uint8"};
      attrs->dtype = _linspaceDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(log).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(log10).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(log1p).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(log2).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(log_softmax)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SoftmaxParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLogSoftmaxAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->temperature = param.temperature.has_value() ? param.temperature.value() : -1;
      static const char *Log_softmaxDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = Log_softmaxDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->use_length = param.use_length.has_value() ? param.use_length.value() : false;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_logical_and)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_logical_and_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyLogicalAndScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(logical_not)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_logical_or)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_logical_or_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyLogicalOrScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_logical_xor)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_logical_xor_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyLogicalXorScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(LogisticRegressionOutput)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RegressionOutputParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyLogisticRegressionOutputAttrs>();
      attrs->grad_scale = static_cast<double>(param.grad_scale);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(MAERegressionOutput)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RegressionOutputParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyMAERegressionOutputAttrs>();
      attrs->grad_scale = static_cast<double>(param.grad_scale);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(make_loss).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(max).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyMaxAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_maximum).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_maximum_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyMaximumScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(mean).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyMeanAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(min).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyMinAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_minimum).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_minimum_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyMinimumScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_minus_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyMinusScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_mod).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(_mod_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyModScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(moments).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MomentsParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyMomentsAttrs>();
      attrs->axes = param.axes.has_value() ? Array<Integer>() : AsArray(param.axes.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_mul_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyMulScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(nanprod).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNanprodAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(nansum).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNansumAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(negative).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(norm).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NormParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNormAttrs>();
      attrs->ord = static_cast<int>(param.ord);
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *NormOutDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                 "int32", "int64",   "int8"};
      attrs->out_dtype =
          NormOutDtypeValues[param.out_dtype.has_value() ? param.out_dtype.value() : 0];
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_not_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_not_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNotEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_broadcast_to)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::BroadcastToParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpBroadcastToAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_copy).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_np_cumsum)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::CumsumParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpCumsumAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      static const char *_np_cumsumDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                    "int32", "int64",   "int8"};
      attrs->dtype = _np_cumsumDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_dot).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(_np__linalg_svd)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_np_max).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReduceAxesNoDTypeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpMaxAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->initial = param.initial.has_value() ? param.initial.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_min).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReduceAxesNoDTypeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpMinAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->initial = param.initial.has_value() ? param.initial.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_ones_like)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_np_prod).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpProdAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *_np_prodDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                  "int32", "int64",   "int8"};
      attrs->dtype = _np_prodDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->initial = param.initial.has_value() ? param.initial.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_reshape)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReshapeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpReshapeAttrs>();
      attrs->newshape = AsArray(param.newshape);
      attrs->order = param.order;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_roll).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyRollParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpRollAttrs>();
      attrs->shift = param.shift.has_value() ? Array<Integer>() : AsArray(param.shift.value());
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_squeeze)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SqueezeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpSqueezeAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_sum).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpSumAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *_np_sumDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                 "int32", "int64",   "int8"};
      attrs->dtype = _np_sumDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->initial = param.initial.has_value() ? param.initial.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_trace).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyTraceParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpTraceAttrs>();
      attrs->offset = static_cast<int>(param.offset);
      attrs->axis1 = static_cast<int>(param.axis1);
      attrs->axis2 = static_cast<int>(param.axis2);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_transpose)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyTransposeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpTransposeAttrs>();
      attrs->axes = AsArray(param.axes);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_np_zeros_like)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_absolute)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_add).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_add_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiAddScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_arange)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RangeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiArangeAttrs>();
      attrs->start = static_cast<double>(param.start);
      attrs->stop = param.stop.has_value() ? param.stop.value() : -1;
      attrs->step = static_cast<double>(param.step);
      attrs->repeat = static_cast<int>(param.repeat);
      attrs->infer_range = static_cast<bool>(param.infer_range);
      attrs->ctx = param.ctx;
      static const char *_npi_arangeDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                     "int64",   "int8",    "uint8"};
      attrs->dtype = _npi_arangeDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_arccos)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arccosh)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arcsin)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arcsinh)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arctan)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arctan2)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_arctan2_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiArctan2ScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_arctanh)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_argmax)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxisParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiArgmaxAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_around)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::AroundParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiAroundAttrs>();
      attrs->decimals = static_cast<int>(param.decimals);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_boolean_mask_assign_tensor)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_cbrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_ceil).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_copysign)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_copysign_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiCopysignScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_cos).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_cosh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_deg2rad)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_degrees)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_exp).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_expm1)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_fix).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_flip).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::FlipParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiFlipAttrs>();
      attrs->axis = AsArray(param.axis);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_floor)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_greater)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_greater_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_greater_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiGreaterEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_greater_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiGreaterScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_hypot)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_identity)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiIdentityAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_npi_identityDtypeValues[] = {"bool",  "float16", "float32", "float64",
                                                       "int32", "int64",   "int8",    "uint8"};
      attrs->dtype = _npi_identityDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_indices)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::IndicesOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiIndicesAttrs>();
      attrs->dimensions = AsArray(param.dimensions);
      static const char *_npi_indicesDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                      "int64",   "int8",    "uint8"};
      attrs->dtype = _npi_indicesDtypeValues[param.dtype];
      attrs->ctx = param.ctx;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_lcm).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_lcm_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiLcmScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_ldexp)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_ldexp_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiLdexpScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_less).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_less_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_less_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiLessEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_less_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiLessScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_log).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_log10)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_log1p)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_log2).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_logical_not)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_mean).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiMeanAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *_npi_meanDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                   "int32", "int64",   "int8"};
      attrs->dtype = _npi_meanDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->initial = param.initial.has_value() ? param.initial.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_mod).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_mod_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiModScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_multiply)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_multiply_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiMultiplyScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_negative)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_normal)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyNormalParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiNormalAttrs>();
      attrs->loc = param.loc.has_value() ? param.loc.value() : -1;
      attrs->scale = param.scale.has_value() ? param.scale.value() : -1;
      attrs->size = param.size.has_value() ? Array<Integer>() : AsArray(param.size.value());
      attrs->ctx = param.ctx;
      static const char *_npi_normalDtypeValues[] = {"float16", "float32", "float64"};
      attrs->dtype = _npi_normalDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_not_equal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_not_equal_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiNotEqualScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_ones).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiOnesAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_npi_onesDtypeValues[] = {"bool",  "float16", "float32", "float64",
                                                   "int32", "int64",   "int8",    "uint8"};
      attrs->dtype = _npi_onesDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_power)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_power_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiPowerScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rad2deg)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_radians)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_rarctan2_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRarctan2ScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rcopysign_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRcopysignScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_reciprocal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_rint).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_rldexp_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRldexpScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rmod_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRmodScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rpower_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRpowerScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rsubtract_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRsubtractScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_rtrue_divide_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiRtrueDivideScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_sign).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_sin).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_sinh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_sqrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_square)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_std).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyMomentsParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiStdAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *_npi_stdDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                  "int32", "int64",   "int8"};
      attrs->dtype = _npi_stdDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->ddof = static_cast<int>(param.ddof);
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_subtract)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_subtract_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiSubtractScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_tan).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(_npi_tanh).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npi_tensordot)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TensordotParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiTensordotAttrs>();
      attrs->a_axes_summed = AsArray(param.a_axes_summed);
      attrs->b_axes_summed = AsArray(param.b_axes_summed);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_tensordot_int_axes)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TensordotIntAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiTensordotIntAxesAttrs>();
      attrs->axes = static_cast<int>(param.axes);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_tril).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TrilParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiTrilAttrs>();
      attrs->k = static_cast<int>(param.k);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_true_divide)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_true_divide_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiTrueDivideScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_trunc)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npi_uniform)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyUniformParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiUniformAttrs>();
      attrs->low = param.low.has_value() ? param.low.value() : -1;
      attrs->high = param.high.has_value() ? param.high.value() : -1;
      attrs->size = param.size.has_value() ? Array<Integer>() : AsArray(param.size.value());
      attrs->ctx = param.ctx;
      static const char *_npi_uniformDtypeValues[] = {"float16", "float32", "float64"};
      attrs->dtype = _npi_uniformDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_unique)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyUniqueParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiUniqueAttrs>();
      attrs->return_index = static_cast<bool>(param.return_index);
      attrs->return_inverse = static_cast<bool>(param.return_inverse);
      attrs->return_counts = static_cast<bool>(param.return_counts);
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_var).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::NumpyMomentsParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiVarAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      static const char *_npi_varDtypeValues[] = {"None",  "float16", "float32", "float64",
                                                  "int32", "int64",   "int8"};
      attrs->dtype = _npi_varDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->ddof = static_cast<int>(param.ddof);
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npi_zeros)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyNpiZerosAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_npi_zerosDtypeValues[] = {"bool",  "float16", "float32", "float64",
                                                    "int32", "int64",   "int8",    "uint8"};
      attrs->dtype = _npi_zerosDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_npx_nonzero)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_npx_relu).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(_npx_sigmoid)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(one_hot).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::OneHotParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyOneHotAttrs>();
      attrs->depth = static_cast<int>(param.depth);
      attrs->on_value = static_cast<double>(param.on_value);
      attrs->off_value = static_cast<double>(param.off_value);
      static const char *One_hotDtypeValues[] = {"float16", "float32", "float64", "int32",
                                                 "int64",   "int8",    "uint8"};
      attrs->dtype = One_hotDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_onehot_encode)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_ones).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyOnesAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_onesDtypeValues[] = {"bool",  "float16", "float32", "float64",
                                               "int32", "int64",   "int8",    "uint8"};
      attrs->dtype = _onesDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(ones_like).set_attr<FConvertAttrs>("FConvertAttrs",
                                                    [](const nnvm::NodeAttrs &node_attrs) {
                                                      return ir::Attrs();
                                                    });

NNVM_REGISTER_OP(pick).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PickParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyPickAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->keepdims = static_cast<bool>(param.keepdims);
      static const char *PickModeValues[] = {"clip", "wrap"};
      attrs->mode = PickModeValues[param.mode];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_plus_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyPlusScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(Pooling).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PoolingParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyPoolingAttrs>();
      attrs->kernel = AsArray(param.kernel);
      static const char *PoolingPoolTypeValues[] = {"avg", "lp", "max", "sum"};
      attrs->pool_type = PoolingPoolTypeValues[param.pool_type];
      attrs->global_pool = static_cast<bool>(param.global_pool);
      attrs->cudnn_off = static_cast<bool>(param.cudnn_off);
      static const char *PoolingPoolingConventionValues[] = {"full", "same", "valid"};
      attrs->pooling_convention = PoolingPoolingConventionValues[param.pooling_convention];
      attrs->stride = AsArray(param.stride);
      attrs->pad = AsArray(param.pad);
      attrs->p_value = param.p_value.has_value() ? param.p_value.value() : -1;
      attrs->count_include_pad =
          param.count_include_pad.has_value() ? param.count_include_pad.value() : false;
      static const char *PoolingLayoutValues[] = {"None",  "NCDHW", "NCHW", "NCW",
                                                  "NDHWC", "NHWC",  "NWC"};
      attrs->layout = PoolingLayoutValues[param.layout.has_value() ? param.layout.value() : 0];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_power).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(_power_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyPowerScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(prod).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyProdAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(RNN).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RNNParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRNNAttrs>();
      attrs->state_size = static_cast<int>(param.state_size);
      attrs->num_layers = static_cast<int>(param.num_layers);
      attrs->bidirectional = static_cast<bool>(param.bidirectional);
      static const char *RNNModeValues[] = {"gru", "lstm", "rnn_relu", "rnn_tanh"};
      attrs->mode = RNNModeValues[param.mode];
      attrs->p = static_cast<double>(param.p);
      attrs->state_outputs = static_cast<bool>(param.state_outputs);
      attrs->projection_size =
          param.projection_size.has_value() ? param.projection_size.value() : -1;
      attrs->lstm_state_clip_min =
          param.lstm_state_clip_min.has_value() ? param.lstm_state_clip_min.value() : -1;
      attrs->lstm_state_clip_max =
          param.lstm_state_clip_max.has_value() ? param.lstm_state_clip_max.value() : -1;
      attrs->lstm_state_clip_nan = static_cast<bool>(param.lstm_state_clip_nan);
      attrs->use_sequence_length = static_cast<bool>(param.use_sequence_length);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(radians).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(_random_exponential)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleExponentialParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomExponentialAttrs>();
      attrs->lam = static_cast<double>(param.lam);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_exponentialDtypeValues[] = {"None", "float16", "float32",
                                                             "float64"};
      attrs->dtype = _random_exponentialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_exponential_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleExponentialLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomExponentialLikeAttrs>();
      attrs->lam = static_cast<double>(param.lam);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_gamma)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleGammaParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomGammaAttrs>();
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->beta = static_cast<double>(param.beta);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_gammaDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _random_gammaDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_gamma_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleGammaLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomGammaLikeAttrs>();
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->beta = static_cast<double>(param.beta);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_generalized_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleGenNegBinomialParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomGeneralizedNegativeBinomialAttrs>();
      attrs->mu = static_cast<double>(param.mu);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_generalized_negative_binomialDtypeValues[] = {
          "None", "float16", "float32", "float64"};
      attrs->dtype = _random_generalized_negative_binomialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_generalized_negative_binomial_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleGenNegBinomialLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomGeneralizedNegativeBinomialLikeAttrs>();
      attrs->mu = static_cast<double>(param.mu);
      attrs->alpha = static_cast<double>(param.alpha);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleNegBinomialParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomNegativeBinomialAttrs>();
      attrs->k = static_cast<int>(param.k);
      attrs->p = static_cast<double>(param.p);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_negative_binomialDtypeValues[] = {"None", "float16", "float32",
                                                                   "float64"};
      attrs->dtype = _random_negative_binomialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_negative_binomial_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleNegBinomialLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomNegativeBinomialLikeAttrs>();
      attrs->k = static_cast<int>(param.k);
      attrs->p = static_cast<double>(param.p);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_normal)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleNormalParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomNormalAttrs>();
      attrs->loc = static_cast<double>(param.loc);
      attrs->scale = static_cast<double>(param.scale);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_normalDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _random_normalDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_normal_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleNormalLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomNormalLikeAttrs>();
      attrs->loc = static_cast<double>(param.loc);
      attrs->scale = static_cast<double>(param.scale);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_dirichlet)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfDirichletAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_exponential)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfExponentialAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_gamma)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfGammaAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_generalized_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfGeneralizedNegativeBinomialAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfNegativeBinomialAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_normal)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfNormalAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_poisson)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfPoissonAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_pdf_uniform)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::PdfParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPdfUniformAttrs>();
      attrs->is_log = static_cast<bool>(param.is_log);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_poisson)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SamplePoissonParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPoissonAttrs>();
      attrs->lam = static_cast<double>(param.lam);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_poissonDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _random_poissonDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_poisson_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SamplePoissonLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomPoissonLikeAttrs>();
      attrs->lam = static_cast<double>(param.lam);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_randint)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleRandIntParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomRandintAttrs>();
      attrs->low = static_cast<int64_t>(param.low);
      attrs->high = static_cast<int64_t>(param.high);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_randintDtypeValues[] = {"None", "int32", "int64"};
      attrs->dtype = _random_randintDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_uniform)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleUniformParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomUniformAttrs>();
      attrs->low = static_cast<double>(param.low);
      attrs->high = static_cast<double>(param.high);
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_random_uniformDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _random_uniformDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_random_uniform_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleUniformLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRandomUniformLikeAttrs>();
      attrs->low = static_cast<double>(param.low);
      attrs->high = static_cast<double>(param.high);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_ravel_multi_index)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RavelParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRavelMultiIndexAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(rcbrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_rdiv_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyRdivScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(reciprocal)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(relu).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(repeat).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RepeatParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyRepeatAttrs>();
      attrs->repeats = static_cast<int>(param.repeats);
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(Reshape).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReshapeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyReshapeAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->reverse = static_cast<bool>(param.reverse);
      attrs->target_shape = AsArray(param.target_shape);
      attrs->keep_highest = static_cast<bool>(param.keep_highest);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(reshape_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReshapeLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyReshapeLikeAttrs>();
      attrs->lhs_begin = param.lhs_begin.has_value() ? param.lhs_begin.value() : -1;
      attrs->lhs_end = param.lhs_end.has_value() ? param.lhs_end.value() : -1;
      attrs->rhs_begin = param.rhs_begin.has_value() ? param.rhs_begin.value() : -1;
      attrs->rhs_end = param.rhs_end.has_value() ? param.rhs_end.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(reverse).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReverseParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyReverseAttrs>();
      attrs->axis = AsArray(param.axis);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(rint).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(_rminus_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyRminusScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_rmod_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyRmodScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(round).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_rpower_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyRpowerScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(rsqrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_sample_exponential)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleExponentialAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_exponentialDtypeValues[] = {"None", "float16", "float32",
                                                             "float64"};
      attrs->dtype = _sample_exponentialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_gamma)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleGammaAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_gammaDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _sample_gammaDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_generalized_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleGeneralizedNegativeBinomialAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_generalized_negative_binomialDtypeValues[] = {
          "None", "float16", "float32", "float64"};
      attrs->dtype = _sample_generalized_negative_binomialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_multinomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleMultinomialParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleMultinomialAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->get_prob = static_cast<bool>(param.get_prob);
      static const char *_sample_multinomialDtypeValues[] = {"float16", "float32", "float64",
                                                             "int32", "uint8"};
      attrs->dtype = _sample_multinomialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_negative_binomial)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleNegativeBinomialAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_negative_binomialDtypeValues[] = {"None", "float16", "float32",
                                                                   "float64"};
      attrs->dtype = _sample_negative_binomialDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_normal)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleNormalAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_normalDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _sample_normalDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_poisson)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySamplePoissonAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_poissonDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _sample_poissonDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_uniform)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::MultiSampleParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleUniformAttrs>();
      attrs->shape = AsArray(param.shape);
      static const char *_sample_uniformDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = _sample_uniformDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sample_unique_zipfian)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SampleUniqueZifpianParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySampleUniqueZipfianAttrs>();
      attrs->range_max = static_cast<int>(param.range_max);
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_scatter_elemwise_div)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_scatter_minus_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyScatterMinusScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(scatter_nd)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ScatterNDParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyScatterNdAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_scatter_plus_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      auto attrs = ir::make_node<v3::op::attrs::LegacyScatterPlusScalarAttrs>();
      attrs->scalar = std::stod(node_attrs.dict.at("scalar"));
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_scatter_set_nd)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ScatterNDParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyScatterSetNdAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sg_mkldnn_conv)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_sg_mkldnn_fully_connected)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(shape_array)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_shuffle).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(sigmoid).set_attr<FConvertAttrs>("FConvertAttrs",
                                                  [](const nnvm::NodeAttrs &node_attrs) {
                                                    return ir::Attrs();
                                                  });

NNVM_REGISTER_OP(sign).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(sin).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(sinh).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(size_array)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(slice).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SliceParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySliceAttrs>();
      attrs->begin = AsArray(param.begin);
      attrs->end = AsArray(param.end);
      attrs->step = AsArray(param.step);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_slice_assign)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SliceParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySliceAssignAttrs>();
      attrs->begin = AsArray(param.begin);
      attrs->end = AsArray(param.end);
      attrs->step = AsArray(param.step);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_slice_assign_scalar)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SliceAssignScalarParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySliceAssignScalarAttrs>();
      attrs->scalar = static_cast<double>(param.scalar);
      attrs->begin = AsArray(param.begin);
      attrs->end = AsArray(param.end);
      attrs->step = AsArray(param.step);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(slice_axis)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SliceAxisParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySliceAxisAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->begin = static_cast<int>(param.begin);
      attrs->end = param.end.has_value() ? param.end.value() : -1;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(slice_like)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SliceLikeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySliceLikeAttrs>();
      attrs->axes = AsArray(param.axes);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(softmax).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SoftmaxParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySoftmaxAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->temperature = param.temperature.has_value() ? param.temperature.value() : -1;
      static const char *SoftmaxDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = SoftmaxDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->use_length = param.use_length.has_value() ? param.use_length.value() : false;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(SoftmaxActivation)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SoftmaxActivationParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySoftmaxActivationAttrs>();
      static const char *SoftmaxActivationModeValues[] = {"channel", "instance"};
      attrs->mode = SoftmaxActivationModeValues[param.mode];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(softmax_cross_entropy)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(SoftmaxOutput)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SoftmaxOutputParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySoftmaxOutputAttrs>();
      attrs->grad_scale = static_cast<double>(param.grad_scale);
      attrs->ignore_label = static_cast<double>(param.ignore_label);
      attrs->multi_output = static_cast<bool>(param.multi_output);
      attrs->use_ignore = static_cast<bool>(param.use_ignore);
      attrs->preserve_shape = static_cast<bool>(param.preserve_shape);
      static const char *SoftmaxOutputNormalizationValues[] = {"batch", "null", "valid"};
      attrs->normalization = SoftmaxOutputNormalizationValues[param.normalization];
      attrs->out_grad = static_cast<bool>(param.out_grad);
      attrs->smooth_alpha = static_cast<double>(param.smooth_alpha);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(softmin).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SoftmaxParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySoftminAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      attrs->temperature = param.temperature.has_value() ? param.temperature.value() : -1;
      static const char *SoftminDtypeValues[] = {"None", "float16", "float32", "float64"};
      attrs->dtype = SoftminDtypeValues[param.dtype.has_value() ? param.dtype.value() : 0];
      attrs->use_length = param.use_length.has_value() ? param.use_length.value() : false;
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(softsign).set_attr<FConvertAttrs>("FConvertAttrs",
                                                   [](const nnvm::NodeAttrs &node_attrs) {
                                                     return ir::Attrs();
                                                   });

NNVM_REGISTER_OP(sort).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SortParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySortAttrs>();
      attrs->axis = param.axis.has_value() ? param.axis.value() : -1;
      attrs->is_ascend = static_cast<bool>(param.is_ascend);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(space_to_depth)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::DepthToSpaceParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySpaceToDepthAttrs>();
      attrs->block_size = static_cast<int>(param.block_size);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(_sparse_retain)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_split_v2).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SplitParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySplitV2Attrs>();
      attrs->indices = AsArray(param.indices);
      attrs->axis = static_cast<int>(param.axis);
      attrs->squeeze_axis = static_cast<bool>(param.squeeze_axis);
      attrs->sections = static_cast<int>(param.sections);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(sqrt).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(square).set_attr<FConvertAttrs>("FConvertAttrs",
                                                 [](const nnvm::NodeAttrs &node_attrs) {
                                                   return ir::Attrs();
                                                 });

NNVM_REGISTER_OP(_square_sum)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySquareSumAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(squeeze).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::SqueezeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySqueezeAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(sum).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::ReduceAxesParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacySumAttrs>();
      attrs->axis = param.axis.has_value() ? Array<Integer>() : AsArray(param.axis.value());
      attrs->keepdims = static_cast<bool>(param.keepdims);
      attrs->exclude = static_cast<bool>(param.exclude);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(take).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TakeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyTakeAttrs>();
      attrs->axis = static_cast<int>(param.axis);
      static const char *TakeModeValues[] = {"clip", "raise", "wrap"};
      attrs->mode = TakeModeValues[param.mode];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(tan).set_attr<FConvertAttrs>("FConvertAttrs",
                                              [](const nnvm::NodeAttrs &node_attrs) {
                                                return ir::Attrs();
                                              });

NNVM_REGISTER_OP(tanh).set_attr<FConvertAttrs>("FConvertAttrs",
                                               [](const nnvm::NodeAttrs &node_attrs) {
                                                 return ir::Attrs();
                                               });

NNVM_REGISTER_OP(tile).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TileParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyTileAttrs>();
      attrs->reps = AsArray(param.reps);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(transpose).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::TransposeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyTransposeAttrs>();
      attrs->axes = AsArray(param.axes);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(trunc).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_unravel_index)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::RavelParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyUnravelIndexAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(where).set_attr<FConvertAttrs>("FConvertAttrs",
                                                [](const nnvm::NodeAttrs &node_attrs) {
                                                  return ir::Attrs();
                                                });

NNVM_REGISTER_OP(_zeros).set_attr<FConvertAttrs>(
    "FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyZerosAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      static const char *_zerosDtypeValues[] = {"bool",  "float16", "float32", "float64",
                                                "int32", "int64",   "int8",    "uint8"};
      attrs->dtype = _zerosDtypeValues[param.dtype];
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

NNVM_REGISTER_OP(zeros_like)
    .set_attr<FConvertAttrs>("FConvertAttrs",
                             [](const nnvm::NodeAttrs &node_attrs) { return ir::Attrs(); });

NNVM_REGISTER_OP(_zeros_without_dtype)
    .set_attr<FConvertAttrs>("FConvertAttrs", [](const nnvm::NodeAttrs &node_attrs) {
      const auto &param = nnvm::get<mxnet::op::InitOpWithoutDTypeParam>(node_attrs.parsed);
      auto attrs = ir::make_node<v3::op::attrs::LegacyZerosWithoutDtypeAttrs>();
      attrs->shape = AsArray(param.shape);
      attrs->ctx = param.ctx;
      attrs->dtype = static_cast<int>(param.dtype);
      attrs->capsule = NNVMCapsule::make(node_attrs);
      return ir::Attrs(attrs);
    });

}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif
