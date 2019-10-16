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
 * \file legacy_nnvm_attrs.h
 * \author Junru Shao
 */
#pragma once
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <string>

#include "../../ir.h"

namespace mxnet {
namespace v3 {
namespace op {
namespace attrs {
// LegacyAbs
using LegacyAbsAttrs = ir::Attrs;
// LegacyActivation
class LegacyActivationAttrs : public ir::AttrsNode<LegacyActivationAttrs> {
 public:
  std::string act_type;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyActivationAttrs, "mxnet.v3.attrs.LegacyActivationAttrs") {
    MX_V3_ATTR_FIELD(act_type);
  }
};
// LegacyArange
class LegacyArangeAttrs : public ir::AttrsNode<LegacyArangeAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyArangeAttrs, "mxnet.v3.attrs.LegacyArangeAttrs") {
    MX_V3_ATTR_FIELD(start);
    MX_V3_ATTR_FIELD(stop);
    MX_V3_ATTR_FIELD(step);
    MX_V3_ATTR_FIELD(repeat);
    MX_V3_ATTR_FIELD(infer_range);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyArccos
using LegacyArccosAttrs = ir::Attrs;
// LegacyArccosh
using LegacyArccoshAttrs = ir::Attrs;
// LegacyArcsin
using LegacyArcsinAttrs = ir::Attrs;
// LegacyArcsinh
using LegacyArcsinhAttrs = ir::Attrs;
// LegacyArctan
using LegacyArctanAttrs = ir::Attrs;
// LegacyArctanh
using LegacyArctanhAttrs = ir::Attrs;
// LegacyArgmax
class LegacyArgmaxAttrs : public ir::AttrsNode<LegacyArgmaxAttrs> {
 public:
  int axis;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyArgmaxAttrs, "mxnet.v3.attrs.LegacyArgmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyArgmaxChannel
using LegacyArgmaxChannelAttrs = ir::Attrs;
// LegacyArgmin
class LegacyArgminAttrs : public ir::AttrsNode<LegacyArgminAttrs> {
 public:
  int axis;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyArgminAttrs, "mxnet.v3.attrs.LegacyArgminAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyArgsort
class LegacyArgsortAttrs : public ir::AttrsNode<LegacyArgsortAttrs> {
 public:
  int axis;
  bool is_ascend;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyArgsortAttrs, "mxnet.v3.attrs.LegacyArgsortAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(is_ascend);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyBatchDot
class LegacyBatchDotAttrs : public ir::AttrsNode<LegacyBatchDotAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  std::string forward_stype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBatchDotAttrs, "mxnet.v3.attrs.LegacyBatchDotAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(forward_stype);
  }
};
// LegacyBatchNorm
class LegacyBatchNormAttrs : public ir::AttrsNode<LegacyBatchNormAttrs> {
 public:
  double eps;
  double momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int axis;
  bool cudnn_off;
  double min_calib_range;
  double max_calib_range;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBatchNormAttrs, "mxnet.v3.attrs.LegacyBatchNormAttrs") {
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(momentum);
    MX_V3_ATTR_FIELD(fix_gamma);
    MX_V3_ATTR_FIELD(use_global_stats);
    MX_V3_ATTR_FIELD(output_mean_var);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(cudnn_off);
    MX_V3_ATTR_FIELD(min_calib_range);
    MX_V3_ATTR_FIELD(max_calib_range);
  }
};
// LegacyBatchTake
using LegacyBatchTakeAttrs = ir::Attrs;
// LegacyBilinearSampler
class LegacyBilinearSamplerAttrs : public ir::AttrsNode<LegacyBilinearSamplerAttrs> {
 public:
  bool cudnn_off;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBilinearSamplerAttrs, "mxnet.v3.attrs.LegacyBilinearSamplerAttrs") {
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// LegacyBroadcastAdd
using LegacyBroadcastAddAttrs = ir::Attrs;
// LegacyBroadcastAxis
class LegacyBroadcastAxisAttrs : public ir::AttrsNode<LegacyBroadcastAxisAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::Array<ir::Integer> size;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBroadcastAxisAttrs, "mxnet.v3.attrs.LegacyBroadcastAxisAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(size);
  }
};
// LegacyBroadcastDiv
using LegacyBroadcastDivAttrs = ir::Attrs;
// LegacyBroadcastEqual
using LegacyBroadcastEqualAttrs = ir::Attrs;
// LegacyBroadcastGreater
using LegacyBroadcastGreaterAttrs = ir::Attrs;
// LegacyBroadcastGreaterEqual
using LegacyBroadcastGreaterEqualAttrs = ir::Attrs;
// LegacyBroadcastHypot
using LegacyBroadcastHypotAttrs = ir::Attrs;
// LegacyBroadcastLesser
using LegacyBroadcastLesserAttrs = ir::Attrs;
// LegacyBroadcastLesserEqual
using LegacyBroadcastLesserEqualAttrs = ir::Attrs;
// LegacyBroadcastLike
class LegacyBroadcastLikeAttrs : public ir::AttrsNode<LegacyBroadcastLikeAttrs> {
 public:
  ir::Array<ir::Integer> lhs_axes;
  ir::Array<ir::Integer> rhs_axes;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBroadcastLikeAttrs, "mxnet.v3.attrs.LegacyBroadcastLikeAttrs") {
    MX_V3_ATTR_FIELD(lhs_axes);
    MX_V3_ATTR_FIELD(rhs_axes);
  }
};
// LegacyBroadcastLogicalAnd
using LegacyBroadcastLogicalAndAttrs = ir::Attrs;
// LegacyBroadcastLogicalOr
using LegacyBroadcastLogicalOrAttrs = ir::Attrs;
// LegacyBroadcastLogicalXor
using LegacyBroadcastLogicalXorAttrs = ir::Attrs;
// LegacyBroadcastMaximum
using LegacyBroadcastMaximumAttrs = ir::Attrs;
// LegacyBroadcastMinimum
using LegacyBroadcastMinimumAttrs = ir::Attrs;
// LegacyBroadcastMod
using LegacyBroadcastModAttrs = ir::Attrs;
// LegacyBroadcastMul
using LegacyBroadcastMulAttrs = ir::Attrs;
// LegacyBroadcastNotEqual
using LegacyBroadcastNotEqualAttrs = ir::Attrs;
// LegacyBroadcastPower
using LegacyBroadcastPowerAttrs = ir::Attrs;
// LegacyBroadcastSub
using LegacyBroadcastSubAttrs = ir::Attrs;
// LegacyBroadcastTo
class LegacyBroadcastToAttrs : public ir::AttrsNode<LegacyBroadcastToAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyBroadcastToAttrs, "mxnet.v3.attrs.LegacyBroadcastToAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyCTCLoss
class LegacyCTCLossAttrs : public ir::AttrsNode<LegacyCTCLossAttrs> {
 public:
  bool use_data_lengths;
  bool use_label_lengths;
  std::string blank_label;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyCTCLossAttrs, "mxnet.v3.attrs.LegacyCTCLossAttrs") {
    MX_V3_ATTR_FIELD(use_data_lengths);
    MX_V3_ATTR_FIELD(use_label_lengths);
    MX_V3_ATTR_FIELD(blank_label);
  }
};
// LegacyCast
class LegacyCastAttrs : public ir::AttrsNode<LegacyCastAttrs> {
 public:
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyCastAttrs, "mxnet.v3.attrs.LegacyCastAttrs") {
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyCbrt
using LegacyCbrtAttrs = ir::Attrs;
// LegacyCeil
using LegacyCeilAttrs = ir::Attrs;
// LegacyClip
class LegacyClipAttrs : public ir::AttrsNode<LegacyClipAttrs> {
 public:
  double a_min;
  double a_max;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyClipAttrs, "mxnet.v3.attrs.LegacyClipAttrs") {
    MX_V3_ATTR_FIELD(a_min);
    MX_V3_ATTR_FIELD(a_max);
  }
};
// LegacyConvolution
class LegacyConvolutionAttrs : public ir::AttrsNode<LegacyConvolutionAttrs> {
 public:
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> dilate;
  ir::Array<ir::Integer> pad;
  int num_filter;
  int num_group;
  int64_t workspace;
  bool no_bias;
  std::string cudnn_tune;
  bool cudnn_off;
  std::string layout;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyConvolutionAttrs, "mxnet.v3.attrs.LegacyConvolutionAttrs") {
    MX_V3_ATTR_FIELD(kernel);
    MX_V3_ATTR_FIELD(stride);
    MX_V3_ATTR_FIELD(dilate);
    MX_V3_ATTR_FIELD(pad);
    MX_V3_ATTR_FIELD(num_filter);
    MX_V3_ATTR_FIELD(num_group);
    MX_V3_ATTR_FIELD(workspace);
    MX_V3_ATTR_FIELD(no_bias);
    MX_V3_ATTR_FIELD(cudnn_tune);
    MX_V3_ATTR_FIELD(cudnn_off);
    MX_V3_ATTR_FIELD(layout);
  }
};
// LegacyCopy
using LegacyCopyAttrs = ir::Attrs;
// LegacyCopyto
using LegacyCopytoAttrs = ir::Attrs;
// LegacyCorrelation
class LegacyCorrelationAttrs : public ir::AttrsNode<LegacyCorrelationAttrs> {
 public:
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  int pad_size;
  bool is_multiply;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyCorrelationAttrs, "mxnet.v3.attrs.LegacyCorrelationAttrs") {
    MX_V3_ATTR_FIELD(kernel_size);
    MX_V3_ATTR_FIELD(max_displacement);
    MX_V3_ATTR_FIELD(stride1);
    MX_V3_ATTR_FIELD(stride2);
    MX_V3_ATTR_FIELD(pad_size);
    MX_V3_ATTR_FIELD(is_multiply);
  }
};
// LegacyCos
using LegacyCosAttrs = ir::Attrs;
// LegacyCosh
using LegacyCoshAttrs = ir::Attrs;
// LegacyDeconvolution
class LegacyDeconvolutionAttrs : public ir::AttrsNode<LegacyDeconvolutionAttrs> {
 public:
  ir::Array<ir::Integer> kernel;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> dilate;
  ir::Array<ir::Integer> pad;
  ir::Array<ir::Integer> adj;
  ir::Array<ir::Integer> target_shape;
  int num_filter;
  int num_group;
  int64_t workspace;
  bool no_bias;
  std::string cudnn_tune;
  bool cudnn_off;
  std::string layout;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDeconvolutionAttrs, "mxnet.v3.attrs.LegacyDeconvolutionAttrs") {
    MX_V3_ATTR_FIELD(kernel);
    MX_V3_ATTR_FIELD(stride);
    MX_V3_ATTR_FIELD(dilate);
    MX_V3_ATTR_FIELD(pad);
    MX_V3_ATTR_FIELD(adj);
    MX_V3_ATTR_FIELD(target_shape);
    MX_V3_ATTR_FIELD(num_filter);
    MX_V3_ATTR_FIELD(num_group);
    MX_V3_ATTR_FIELD(workspace);
    MX_V3_ATTR_FIELD(no_bias);
    MX_V3_ATTR_FIELD(cudnn_tune);
    MX_V3_ATTR_FIELD(cudnn_off);
    MX_V3_ATTR_FIELD(layout);
  }
};
// LegacyDegrees
using LegacyDegreesAttrs = ir::Attrs;
// LegacyDepthToSpace
class LegacyDepthToSpaceAttrs : public ir::AttrsNode<LegacyDepthToSpaceAttrs> {
 public:
  int block_size;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDepthToSpaceAttrs, "mxnet.v3.attrs.LegacyDepthToSpaceAttrs") {
    MX_V3_ATTR_FIELD(block_size);
  }
};
// LegacyDiag
class LegacyDiagAttrs : public ir::AttrsNode<LegacyDiagAttrs> {
 public:
  int k;
  int axis1;
  int axis2;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDiagAttrs, "mxnet.v3.attrs.LegacyDiagAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(axis1);
    MX_V3_ATTR_FIELD(axis2);
  }
};
// LegacyDivScalar
class LegacyDivScalarAttrs : public ir::AttrsNode<LegacyDivScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDivScalarAttrs, "mxnet.v3.attrs.LegacyDivScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyDot
class LegacyDotAttrs : public ir::AttrsNode<LegacyDotAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  std::string forward_stype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDotAttrs, "mxnet.v3.attrs.LegacyDotAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(forward_stype);
  }
};
// LegacyDropout
class LegacyDropoutAttrs : public ir::AttrsNode<LegacyDropoutAttrs> {
 public:
  double p;
  std::string mode;
  ir::Array<ir::Integer> axes;
  bool cudnn_off;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyDropoutAttrs, "mxnet.v3.attrs.LegacyDropoutAttrs") {
    MX_V3_ATTR_FIELD(p);
    MX_V3_ATTR_FIELD(mode);
    MX_V3_ATTR_FIELD(axes);
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// LegacyElemwiseAdd
using LegacyElemwiseAddAttrs = ir::Attrs;
// LegacyElemwiseDiv
using LegacyElemwiseDivAttrs = ir::Attrs;
// LegacyElemwiseMul
using LegacyElemwiseMulAttrs = ir::Attrs;
// LegacyElemwiseSub
using LegacyElemwiseSubAttrs = ir::Attrs;
// LegacyEmbedding
class LegacyEmbeddingAttrs : public ir::AttrsNode<LegacyEmbeddingAttrs> {
 public:
  int input_dim;
  int output_dim;
  std::string dtype;
  bool sparse_grad;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyEmbeddingAttrs, "mxnet.v3.attrs.LegacyEmbeddingAttrs") {
    MX_V3_ATTR_FIELD(input_dim);
    MX_V3_ATTR_FIELD(output_dim);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(sparse_grad);
  }
};
// LegacyEqual
using LegacyEqualAttrs = ir::Attrs;
// LegacyEqualScalar
class LegacyEqualScalarAttrs : public ir::AttrsNode<LegacyEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyEqualScalarAttrs, "mxnet.v3.attrs.LegacyEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyErf
using LegacyErfAttrs = ir::Attrs;
// LegacyErfinv
using LegacyErfinvAttrs = ir::Attrs;
// LegacyExp
using LegacyExpAttrs = ir::Attrs;
// LegacyExpandDims
class LegacyExpandDimsAttrs : public ir::AttrsNode<LegacyExpandDimsAttrs> {
 public:
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyExpandDimsAttrs, "mxnet.v3.attrs.LegacyExpandDimsAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyExpm1
using LegacyExpm1Attrs = ir::Attrs;
// LegacyEye
class LegacyEyeAttrs : public ir::AttrsNode<LegacyEyeAttrs> {
 public:
  int64_t N;
  int64_t M;
  int64_t k;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyEyeAttrs, "mxnet.v3.attrs.LegacyEyeAttrs") {
    MX_V3_ATTR_FIELD(N);
    MX_V3_ATTR_FIELD(M);
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyFix
using LegacyFixAttrs = ir::Attrs;
// LegacyFlatten
using LegacyFlattenAttrs = ir::Attrs;
// LegacyFloor
using LegacyFloorAttrs = ir::Attrs;
// LegacyFull
class LegacyFullAttrs : public ir::AttrsNode<LegacyFullAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  double value;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyFullAttrs, "mxnet.v3.attrs.LegacyFullAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(value);
  }
};
// LegacyFullyConnected
class LegacyFullyConnectedAttrs : public ir::AttrsNode<LegacyFullyConnectedAttrs> {
 public:
  int num_hidden;
  bool no_bias;
  bool flatten;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyFullyConnectedAttrs, "mxnet.v3.attrs.LegacyFullyConnectedAttrs") {
    MX_V3_ATTR_FIELD(num_hidden);
    MX_V3_ATTR_FIELD(no_bias);
    MX_V3_ATTR_FIELD(flatten);
  }
};
// LegacyGamma
using LegacyGammaAttrs = ir::Attrs;
// LegacyGammaln
using LegacyGammalnAttrs = ir::Attrs;
// LegacyGatherNd
using LegacyGatherNdAttrs = ir::Attrs;
// LegacyGradAdd
using LegacyGradAddAttrs = ir::Attrs;
// LegacyGreater
using LegacyGreaterAttrs = ir::Attrs;
// LegacyGreaterEqual
using LegacyGreaterEqualAttrs = ir::Attrs;
// LegacyGreaterEqualScalar
class LegacyGreaterEqualScalarAttrs : public ir::AttrsNode<LegacyGreaterEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyGreaterEqualScalarAttrs,
                      "mxnet.v3.attrs.LegacyGreaterEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyGreaterScalar
class LegacyGreaterScalarAttrs : public ir::AttrsNode<LegacyGreaterScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyGreaterScalarAttrs, "mxnet.v3.attrs.LegacyGreaterScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyGroupNorm
class LegacyGroupNormAttrs : public ir::AttrsNode<LegacyGroupNormAttrs> {
 public:
  int num_groups;
  double eps;
  bool output_mean_var;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyGroupNormAttrs, "mxnet.v3.attrs.LegacyGroupNormAttrs") {
    MX_V3_ATTR_FIELD(num_groups);
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(output_mean_var);
  }
};
// LegacyHardSigmoid
class LegacyHardSigmoidAttrs : public ir::AttrsNode<LegacyHardSigmoidAttrs> {
 public:
  double alpha;
  double beta;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyHardSigmoidAttrs, "mxnet.v3.attrs.LegacyHardSigmoidAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
  }
};
// LegacyHypot
using LegacyHypotAttrs = ir::Attrs;
// LegacyHypotScalar
class LegacyHypotScalarAttrs : public ir::AttrsNode<LegacyHypotScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyHypotScalarAttrs, "mxnet.v3.attrs.LegacyHypotScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyIdentityAttachKLSparseReg
class LegacyIdentityAttachKLSparseRegAttrs
    : public ir::AttrsNode<LegacyIdentityAttachKLSparseRegAttrs> {
 public:
  double sparseness_target;
  double penalty;
  double momentum;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyIdentityAttachKLSparseRegAttrs,
                      "mxnet.v3.attrs.LegacyIdentityAttachKLSparseRegAttrs") {
    MX_V3_ATTR_FIELD(sparseness_target);
    MX_V3_ATTR_FIELD(penalty);
    MX_V3_ATTR_FIELD(momentum);
  }
};
// LegacyIdentityWithAttrLikeRhs
using LegacyIdentityWithAttrLikeRhsAttrs = ir::Attrs;
// LegacyInstanceNorm
class LegacyInstanceNormAttrs : public ir::AttrsNode<LegacyInstanceNormAttrs> {
 public:
  double eps;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyInstanceNormAttrs, "mxnet.v3.attrs.LegacyInstanceNormAttrs") {
    MX_V3_ATTR_FIELD(eps);
  }
};
// LegacyL2Normalization
class LegacyL2NormalizationAttrs : public ir::AttrsNode<LegacyL2NormalizationAttrs> {
 public:
  double eps;
  std::string mode;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyL2NormalizationAttrs, "mxnet.v3.attrs.LegacyL2NormalizationAttrs") {
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(mode);
  }
};
// LegacyLRN
class LegacyLRNAttrs : public ir::AttrsNode<LegacyLRNAttrs> {
 public:
  double alpha;
  double beta;
  double knorm;
  int nsize;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLRNAttrs, "mxnet.v3.attrs.LegacyLRNAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(knorm);
    MX_V3_ATTR_FIELD(nsize);
  }
};
// LegacyLayerNorm
class LegacyLayerNormAttrs : public ir::AttrsNode<LegacyLayerNormAttrs> {
 public:
  int axis;
  double eps;
  bool output_mean_var;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLayerNormAttrs, "mxnet.v3.attrs.LegacyLayerNormAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(output_mean_var);
  }
};
// LegacyLeakyReLU
class LegacyLeakyReLUAttrs : public ir::AttrsNode<LegacyLeakyReLUAttrs> {
 public:
  std::string act_type;
  double slope;
  double lower_bound;
  double upper_bound;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLeakyReLUAttrs, "mxnet.v3.attrs.LegacyLeakyReLUAttrs") {
    MX_V3_ATTR_FIELD(act_type);
    MX_V3_ATTR_FIELD(slope);
    MX_V3_ATTR_FIELD(lower_bound);
    MX_V3_ATTR_FIELD(upper_bound);
  }
};
// LegacyLesser
using LegacyLesserAttrs = ir::Attrs;
// LegacyLesserEqual
using LegacyLesserEqualAttrs = ir::Attrs;
// LegacyLesserEqualScalar
class LegacyLesserEqualScalarAttrs : public ir::AttrsNode<LegacyLesserEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLesserEqualScalarAttrs, "mxnet.v3.attrs.LegacyLesserEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyLesserScalar
class LegacyLesserScalarAttrs : public ir::AttrsNode<LegacyLesserScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLesserScalarAttrs, "mxnet.v3.attrs.LegacyLesserScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyLinalgDet
using LegacyLinalgDetAttrs = ir::Attrs;
// LegacyLinalgExtractdiag
class LegacyLinalgExtractdiagAttrs : public ir::AttrsNode<LegacyLinalgExtractdiagAttrs> {
 public:
  int offset;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgExtractdiagAttrs, "mxnet.v3.attrs.LegacyLinalgExtractdiagAttrs") {
    MX_V3_ATTR_FIELD(offset);
  }
};
// LegacyLinalgExtracttrian
class LegacyLinalgExtracttrianAttrs : public ir::AttrsNode<LegacyLinalgExtracttrianAttrs> {
 public:
  int offset;
  bool lower;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgExtracttrianAttrs,
                      "mxnet.v3.attrs.LegacyLinalgExtracttrianAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(lower);
  }
};
// LegacyLinalgGelqf
using LegacyLinalgGelqfAttrs = ir::Attrs;
// LegacyLinalgGemm
class LegacyLinalgGemmAttrs : public ir::AttrsNode<LegacyLinalgGemmAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
  double beta;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgGemmAttrs, "mxnet.v3.attrs.LegacyLinalgGemmAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyLinalgGemm2
class LegacyLinalgGemm2Attrs : public ir::AttrsNode<LegacyLinalgGemm2Attrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgGemm2Attrs, "mxnet.v3.attrs.LegacyLinalgGemm2Attrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyLinalgInverse
using LegacyLinalgInverseAttrs = ir::Attrs;
// LegacyLinalgMakediag
class LegacyLinalgMakediagAttrs : public ir::AttrsNode<LegacyLinalgMakediagAttrs> {
 public:
  int offset;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgMakediagAttrs, "mxnet.v3.attrs.LegacyLinalgMakediagAttrs") {
    MX_V3_ATTR_FIELD(offset);
  }
};
// LegacyLinalgMaketrian
class LegacyLinalgMaketrianAttrs : public ir::AttrsNode<LegacyLinalgMaketrianAttrs> {
 public:
  int offset;
  bool lower;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgMaketrianAttrs, "mxnet.v3.attrs.LegacyLinalgMaketrianAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(lower);
  }
};
// LegacyLinalgPotrf
using LegacyLinalgPotrfAttrs = ir::Attrs;
// LegacyLinalgPotri
using LegacyLinalgPotriAttrs = ir::Attrs;
// LegacyLinalgSlogdet
using LegacyLinalgSlogdetAttrs = ir::Attrs;
// LegacyLinalgSumlogdiag
using LegacyLinalgSumlogdiagAttrs = ir::Attrs;
// LegacyLinalgSyevd
using LegacyLinalgSyevdAttrs = ir::Attrs;
// LegacyLinalgSyrk
class LegacyLinalgSyrkAttrs : public ir::AttrsNode<LegacyLinalgSyrkAttrs> {
 public:
  bool transpose;
  double alpha;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgSyrkAttrs, "mxnet.v3.attrs.LegacyLinalgSyrkAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// LegacyLinalgTrmm
class LegacyLinalgTrmmAttrs : public ir::AttrsNode<LegacyLinalgTrmmAttrs> {
 public:
  bool transpose;
  bool rightside;
  bool lower;
  double alpha;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgTrmmAttrs, "mxnet.v3.attrs.LegacyLinalgTrmmAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(rightside);
    MX_V3_ATTR_FIELD(lower);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// LegacyLinalgTrsm
class LegacyLinalgTrsmAttrs : public ir::AttrsNode<LegacyLinalgTrsmAttrs> {
 public:
  bool transpose;
  bool rightside;
  bool lower;
  double alpha;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinalgTrsmAttrs, "mxnet.v3.attrs.LegacyLinalgTrsmAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(rightside);
    MX_V3_ATTR_FIELD(lower);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// LegacyLinearRegressionOutput
class LegacyLinearRegressionOutputAttrs : public ir::AttrsNode<LegacyLinearRegressionOutputAttrs> {
 public:
  double grad_scale;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinearRegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyLinearRegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// LegacyLinspace
class LegacyLinspaceAttrs : public ir::AttrsNode<LegacyLinspaceAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLinspaceAttrs, "mxnet.v3.attrs.LegacyLinspaceAttrs") {
    MX_V3_ATTR_FIELD(start);
    MX_V3_ATTR_FIELD(stop);
    MX_V3_ATTR_FIELD(step);
    MX_V3_ATTR_FIELD(repeat);
    MX_V3_ATTR_FIELD(infer_range);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyLog
using LegacyLogAttrs = ir::Attrs;
// LegacyLog10
using LegacyLog10Attrs = ir::Attrs;
// LegacyLog1p
using LegacyLog1pAttrs = ir::Attrs;
// LegacyLog2
using LegacyLog2Attrs = ir::Attrs;
// LegacyLogSoftmax
class LegacyLogSoftmaxAttrs : public ir::AttrsNode<LegacyLogSoftmaxAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLogSoftmaxAttrs, "mxnet.v3.attrs.LegacyLogSoftmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// LegacyLogicalAnd
using LegacyLogicalAndAttrs = ir::Attrs;
// LegacyLogicalAndScalar
class LegacyLogicalAndScalarAttrs : public ir::AttrsNode<LegacyLogicalAndScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLogicalAndScalarAttrs, "mxnet.v3.attrs.LegacyLogicalAndScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyLogicalNot
using LegacyLogicalNotAttrs = ir::Attrs;
// LegacyLogicalOr
using LegacyLogicalOrAttrs = ir::Attrs;
// LegacyLogicalOrScalar
class LegacyLogicalOrScalarAttrs : public ir::AttrsNode<LegacyLogicalOrScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLogicalOrScalarAttrs, "mxnet.v3.attrs.LegacyLogicalOrScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyLogicalXor
using LegacyLogicalXorAttrs = ir::Attrs;
// LegacyLogicalXorScalar
class LegacyLogicalXorScalarAttrs : public ir::AttrsNode<LegacyLogicalXorScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLogicalXorScalarAttrs, "mxnet.v3.attrs.LegacyLogicalXorScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyLogisticRegressionOutput
class LegacyLogisticRegressionOutputAttrs
    : public ir::AttrsNode<LegacyLogisticRegressionOutputAttrs> {
 public:
  double grad_scale;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyLogisticRegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyLogisticRegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// LegacyMAERegressionOutput
class LegacyMAERegressionOutputAttrs : public ir::AttrsNode<LegacyMAERegressionOutputAttrs> {
 public:
  double grad_scale;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMAERegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyMAERegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// LegacyMakeLoss
using LegacyMakeLossAttrs = ir::Attrs;
// LegacyMax
class LegacyMaxAttrs : public ir::AttrsNode<LegacyMaxAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMaxAttrs, "mxnet.v3.attrs.LegacyMaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyMaximum
using LegacyMaximumAttrs = ir::Attrs;
// LegacyMaximumScalar
class LegacyMaximumScalarAttrs : public ir::AttrsNode<LegacyMaximumScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMaximumScalarAttrs, "mxnet.v3.attrs.LegacyMaximumScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyMean
class LegacyMeanAttrs : public ir::AttrsNode<LegacyMeanAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMeanAttrs, "mxnet.v3.attrs.LegacyMeanAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyMin
class LegacyMinAttrs : public ir::AttrsNode<LegacyMinAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMinAttrs, "mxnet.v3.attrs.LegacyMinAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyMinimum
using LegacyMinimumAttrs = ir::Attrs;
// LegacyMinimumScalar
class LegacyMinimumScalarAttrs : public ir::AttrsNode<LegacyMinimumScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMinimumScalarAttrs, "mxnet.v3.attrs.LegacyMinimumScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyMinusScalar
class LegacyMinusScalarAttrs : public ir::AttrsNode<LegacyMinusScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMinusScalarAttrs, "mxnet.v3.attrs.LegacyMinusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyMod
using LegacyModAttrs = ir::Attrs;
// LegacyModScalar
class LegacyModScalarAttrs : public ir::AttrsNode<LegacyModScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyModScalarAttrs, "mxnet.v3.attrs.LegacyModScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyMoments
class LegacyMomentsAttrs : public ir::AttrsNode<LegacyMomentsAttrs> {
 public:
  ir::Array<ir::Integer> axes;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMomentsAttrs, "mxnet.v3.attrs.LegacyMomentsAttrs") {
    MX_V3_ATTR_FIELD(axes);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyMulScalar
class LegacyMulScalarAttrs : public ir::AttrsNode<LegacyMulScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyMulScalarAttrs, "mxnet.v3.attrs.LegacyMulScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNanprod
class LegacyNanprodAttrs : public ir::AttrsNode<LegacyNanprodAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNanprodAttrs, "mxnet.v3.attrs.LegacyNanprodAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyNansum
class LegacyNansumAttrs : public ir::AttrsNode<LegacyNansumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNansumAttrs, "mxnet.v3.attrs.LegacyNansumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyNegative
using LegacyNegativeAttrs = ir::Attrs;
// LegacyNorm
class LegacyNormAttrs : public ir::AttrsNode<LegacyNormAttrs> {
 public:
  int ord;
  ir::Array<ir::Integer> axis;
  std::string out_dtype;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNormAttrs, "mxnet.v3.attrs.LegacyNormAttrs") {
    MX_V3_ATTR_FIELD(ord);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(out_dtype);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyNotEqual
using LegacyNotEqualAttrs = ir::Attrs;
// LegacyNotEqualScalar
class LegacyNotEqualScalarAttrs : public ir::AttrsNode<LegacyNotEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNotEqualScalarAttrs, "mxnet.v3.attrs.LegacyNotEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpBroadcastTo
class LegacyNpBroadcastToAttrs : public ir::AttrsNode<LegacyNpBroadcastToAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpBroadcastToAttrs, "mxnet.v3.attrs.LegacyNpBroadcastToAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyNpCopy
using LegacyNpCopyAttrs = ir::Attrs;
// LegacyNpCumsum
class LegacyNpCumsumAttrs : public ir::AttrsNode<LegacyNpCumsumAttrs> {
 public:
  int axis;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpCumsumAttrs, "mxnet.v3.attrs.LegacyNpCumsumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpDot
using LegacyNpDotAttrs = ir::Attrs;
// LegacyNpLinalgSvd
using LegacyNpLinalgSvdAttrs = ir::Attrs;
// LegacyNpMax
class LegacyNpMaxAttrs : public ir::AttrsNode<LegacyNpMaxAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  double initial;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpMaxAttrs, "mxnet.v3.attrs.LegacyNpMaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// LegacyNpMin
class LegacyNpMinAttrs : public ir::AttrsNode<LegacyNpMinAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  double initial;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpMinAttrs, "mxnet.v3.attrs.LegacyNpMinAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// LegacyNpOnesLike
using LegacyNpOnesLikeAttrs = ir::Attrs;
// LegacyNpProd
class LegacyNpProdAttrs : public ir::AttrsNode<LegacyNpProdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpProdAttrs, "mxnet.v3.attrs.LegacyNpProdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// LegacyNpReshape
class LegacyNpReshapeAttrs : public ir::AttrsNode<LegacyNpReshapeAttrs> {
 public:
  ir::Array<ir::Integer> newshape;
  std::string order;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpReshapeAttrs, "mxnet.v3.attrs.LegacyNpReshapeAttrs") {
    MX_V3_ATTR_FIELD(newshape);
    MX_V3_ATTR_FIELD(order);
  }
};
// LegacyNpRoll
class LegacyNpRollAttrs : public ir::AttrsNode<LegacyNpRollAttrs> {
 public:
  ir::Array<ir::Integer> shift;
  ir::Array<ir::Integer> axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpRollAttrs, "mxnet.v3.attrs.LegacyNpRollAttrs") {
    MX_V3_ATTR_FIELD(shift);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyNpSqueeze
class LegacyNpSqueezeAttrs : public ir::AttrsNode<LegacyNpSqueezeAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpSqueezeAttrs, "mxnet.v3.attrs.LegacyNpSqueezeAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyNpSum
class LegacyNpSumAttrs : public ir::AttrsNode<LegacyNpSumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpSumAttrs, "mxnet.v3.attrs.LegacyNpSumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// LegacyNpTrace
class LegacyNpTraceAttrs : public ir::AttrsNode<LegacyNpTraceAttrs> {
 public:
  int offset;
  int axis1;
  int axis2;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpTraceAttrs, "mxnet.v3.attrs.LegacyNpTraceAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(axis1);
    MX_V3_ATTR_FIELD(axis2);
  }
};
// LegacyNpTranspose
class LegacyNpTransposeAttrs : public ir::AttrsNode<LegacyNpTransposeAttrs> {
 public:
  ir::Array<ir::Integer> axes;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpTransposeAttrs, "mxnet.v3.attrs.LegacyNpTransposeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// LegacyNpZerosLike
using LegacyNpZerosLikeAttrs = ir::Attrs;
// LegacyNpiAbsolute
using LegacyNpiAbsoluteAttrs = ir::Attrs;
// LegacyNpiAdd
using LegacyNpiAddAttrs = ir::Attrs;
// LegacyNpiAddScalar
class LegacyNpiAddScalarAttrs : public ir::AttrsNode<LegacyNpiAddScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiAddScalarAttrs, "mxnet.v3.attrs.LegacyNpiAddScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiArange
class LegacyNpiArangeAttrs : public ir::AttrsNode<LegacyNpiArangeAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiArangeAttrs, "mxnet.v3.attrs.LegacyNpiArangeAttrs") {
    MX_V3_ATTR_FIELD(start);
    MX_V3_ATTR_FIELD(stop);
    MX_V3_ATTR_FIELD(step);
    MX_V3_ATTR_FIELD(repeat);
    MX_V3_ATTR_FIELD(infer_range);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpiArccos
using LegacyNpiArccosAttrs = ir::Attrs;
// LegacyNpiArccosh
using LegacyNpiArccoshAttrs = ir::Attrs;
// LegacyNpiArcsin
using LegacyNpiArcsinAttrs = ir::Attrs;
// LegacyNpiArcsinh
using LegacyNpiArcsinhAttrs = ir::Attrs;
// LegacyNpiArctan
using LegacyNpiArctanAttrs = ir::Attrs;
// LegacyNpiArctan2
using LegacyNpiArctan2Attrs = ir::Attrs;
// LegacyNpiArctan2Scalar
class LegacyNpiArctan2ScalarAttrs : public ir::AttrsNode<LegacyNpiArctan2ScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiArctan2ScalarAttrs, "mxnet.v3.attrs.LegacyNpiArctan2ScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiArctanh
using LegacyNpiArctanhAttrs = ir::Attrs;
// LegacyNpiArgmax
class LegacyNpiArgmaxAttrs : public ir::AttrsNode<LegacyNpiArgmaxAttrs> {
 public:
  int axis;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiArgmaxAttrs, "mxnet.v3.attrs.LegacyNpiArgmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyNpiAround
class LegacyNpiAroundAttrs : public ir::AttrsNode<LegacyNpiAroundAttrs> {
 public:
  int decimals;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiAroundAttrs, "mxnet.v3.attrs.LegacyNpiAroundAttrs") {
    MX_V3_ATTR_FIELD(decimals);
  }
};
// LegacyNpiBooleanMaskAssignScalar
class LegacyNpiBooleanMaskAssignScalarAttrs
    : public ir::AttrsNode<LegacyNpiBooleanMaskAssignScalarAttrs> {
 public:
  double value;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiBooleanMaskAssignScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiBooleanMaskAssignScalarAttrs") {
    MX_V3_ATTR_FIELD(value);
  }
};
// LegacyNpiBooleanMaskAssignTensor
using LegacyNpiBooleanMaskAssignTensorAttrs = ir::Attrs;
// LegacyNpiCbrt
using LegacyNpiCbrtAttrs = ir::Attrs;
// LegacyNpiCeil
using LegacyNpiCeilAttrs = ir::Attrs;
// LegacyNpiCopysign
using LegacyNpiCopysignAttrs = ir::Attrs;
// LegacyNpiCopysignScalar
class LegacyNpiCopysignScalarAttrs : public ir::AttrsNode<LegacyNpiCopysignScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiCopysignScalarAttrs, "mxnet.v3.attrs.LegacyNpiCopysignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiCos
using LegacyNpiCosAttrs = ir::Attrs;
// LegacyNpiCosh
using LegacyNpiCoshAttrs = ir::Attrs;
// LegacyNpiDeg2rad
using LegacyNpiDeg2radAttrs = ir::Attrs;
// LegacyNpiDegrees
using LegacyNpiDegreesAttrs = ir::Attrs;
// LegacyNpiEqual
using LegacyNpiEqualAttrs = ir::Attrs;
// LegacyNpiEqualScalar
class LegacyNpiEqualScalarAttrs : public ir::AttrsNode<LegacyNpiEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiEqualScalarAttrs, "mxnet.v3.attrs.LegacyNpiEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiExp
using LegacyNpiExpAttrs = ir::Attrs;
// LegacyNpiExpm1
using LegacyNpiExpm1Attrs = ir::Attrs;
// LegacyNpiFix
using LegacyNpiFixAttrs = ir::Attrs;
// LegacyNpiFlip
class LegacyNpiFlipAttrs : public ir::AttrsNode<LegacyNpiFlipAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiFlipAttrs, "mxnet.v3.attrs.LegacyNpiFlipAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyNpiFloor
using LegacyNpiFloorAttrs = ir::Attrs;
// LegacyNpiGreater
using LegacyNpiGreaterAttrs = ir::Attrs;
// LegacyNpiGreaterEqual
using LegacyNpiGreaterEqualAttrs = ir::Attrs;
// LegacyNpiGreaterEqualScalar
class LegacyNpiGreaterEqualScalarAttrs : public ir::AttrsNode<LegacyNpiGreaterEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiGreaterEqualScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiGreaterEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiGreaterScalar
class LegacyNpiGreaterScalarAttrs : public ir::AttrsNode<LegacyNpiGreaterScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiGreaterScalarAttrs, "mxnet.v3.attrs.LegacyNpiGreaterScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiHypot
using LegacyNpiHypotAttrs = ir::Attrs;
// LegacyNpiIdentity
class LegacyNpiIdentityAttrs : public ir::AttrsNode<LegacyNpiIdentityAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiIdentityAttrs, "mxnet.v3.attrs.LegacyNpiIdentityAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpiIndices
class LegacyNpiIndicesAttrs : public ir::AttrsNode<LegacyNpiIndicesAttrs> {
 public:
  ir::Array<ir::Integer> dimensions;
  std::string dtype;
  std::string ctx;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiIndicesAttrs, "mxnet.v3.attrs.LegacyNpiIndicesAttrs") {
    MX_V3_ATTR_FIELD(dimensions);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ctx);
  }
};
// LegacyNpiLcm
using LegacyNpiLcmAttrs = ir::Attrs;
// LegacyNpiLcmScalar
class LegacyNpiLcmScalarAttrs : public ir::AttrsNode<LegacyNpiLcmScalarAttrs> {
 public:
  int scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiLcmScalarAttrs, "mxnet.v3.attrs.LegacyNpiLcmScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiLdexp
using LegacyNpiLdexpAttrs = ir::Attrs;
// LegacyNpiLdexpScalar
class LegacyNpiLdexpScalarAttrs : public ir::AttrsNode<LegacyNpiLdexpScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiLdexpScalarAttrs, "mxnet.v3.attrs.LegacyNpiLdexpScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiLess
using LegacyNpiLessAttrs = ir::Attrs;
// LegacyNpiLessEqual
using LegacyNpiLessEqualAttrs = ir::Attrs;
// LegacyNpiLessEqualScalar
class LegacyNpiLessEqualScalarAttrs : public ir::AttrsNode<LegacyNpiLessEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiLessEqualScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiLessEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiLessScalar
class LegacyNpiLessScalarAttrs : public ir::AttrsNode<LegacyNpiLessScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiLessScalarAttrs, "mxnet.v3.attrs.LegacyNpiLessScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiLog
using LegacyNpiLogAttrs = ir::Attrs;
// LegacyNpiLog10
using LegacyNpiLog10Attrs = ir::Attrs;
// LegacyNpiLog1p
using LegacyNpiLog1pAttrs = ir::Attrs;
// LegacyNpiLog2
using LegacyNpiLog2Attrs = ir::Attrs;
// LegacyNpiLogicalNot
using LegacyNpiLogicalNotAttrs = ir::Attrs;
// LegacyNpiMean
class LegacyNpiMeanAttrs : public ir::AttrsNode<LegacyNpiMeanAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiMeanAttrs, "mxnet.v3.attrs.LegacyNpiMeanAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// LegacyNpiMod
using LegacyNpiModAttrs = ir::Attrs;
// LegacyNpiModScalar
class LegacyNpiModScalarAttrs : public ir::AttrsNode<LegacyNpiModScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiModScalarAttrs, "mxnet.v3.attrs.LegacyNpiModScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiMultiply
using LegacyNpiMultiplyAttrs = ir::Attrs;
// LegacyNpiMultiplyScalar
class LegacyNpiMultiplyScalarAttrs : public ir::AttrsNode<LegacyNpiMultiplyScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiMultiplyScalarAttrs, "mxnet.v3.attrs.LegacyNpiMultiplyScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiNegative
using LegacyNpiNegativeAttrs = ir::Attrs;
// LegacyNpiNormal
class LegacyNpiNormalAttrs : public ir::AttrsNode<LegacyNpiNormalAttrs> {
 public:
  double loc;
  double scale;
  ir::Array<ir::Integer> size;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiNormalAttrs, "mxnet.v3.attrs.LegacyNpiNormalAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
    MX_V3_ATTR_FIELD(size);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpiNotEqual
using LegacyNpiNotEqualAttrs = ir::Attrs;
// LegacyNpiNotEqualScalar
class LegacyNpiNotEqualScalarAttrs : public ir::AttrsNode<LegacyNpiNotEqualScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiNotEqualScalarAttrs, "mxnet.v3.attrs.LegacyNpiNotEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiOnes
class LegacyNpiOnesAttrs : public ir::AttrsNode<LegacyNpiOnesAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiOnesAttrs, "mxnet.v3.attrs.LegacyNpiOnesAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpiPower
using LegacyNpiPowerAttrs = ir::Attrs;
// LegacyNpiPowerScalar
class LegacyNpiPowerScalarAttrs : public ir::AttrsNode<LegacyNpiPowerScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiPowerScalarAttrs, "mxnet.v3.attrs.LegacyNpiPowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRad2deg
using LegacyNpiRad2degAttrs = ir::Attrs;
// LegacyNpiRadians
using LegacyNpiRadiansAttrs = ir::Attrs;
// LegacyNpiRarctan2Scalar
class LegacyNpiRarctan2ScalarAttrs : public ir::AttrsNode<LegacyNpiRarctan2ScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRarctan2ScalarAttrs, "mxnet.v3.attrs.LegacyNpiRarctan2ScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRcopysignScalar
class LegacyNpiRcopysignScalarAttrs : public ir::AttrsNode<LegacyNpiRcopysignScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRcopysignScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRcopysignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiReciprocal
using LegacyNpiReciprocalAttrs = ir::Attrs;
// LegacyNpiRint
using LegacyNpiRintAttrs = ir::Attrs;
// LegacyNpiRldexpScalar
class LegacyNpiRldexpScalarAttrs : public ir::AttrsNode<LegacyNpiRldexpScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRldexpScalarAttrs, "mxnet.v3.attrs.LegacyNpiRldexpScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRmodScalar
class LegacyNpiRmodScalarAttrs : public ir::AttrsNode<LegacyNpiRmodScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRmodScalarAttrs, "mxnet.v3.attrs.LegacyNpiRmodScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRpowerScalar
class LegacyNpiRpowerScalarAttrs : public ir::AttrsNode<LegacyNpiRpowerScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRpowerScalarAttrs, "mxnet.v3.attrs.LegacyNpiRpowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRsubtractScalar
class LegacyNpiRsubtractScalarAttrs : public ir::AttrsNode<LegacyNpiRsubtractScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRsubtractScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRsubtractScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiRtrueDivideScalar
class LegacyNpiRtrueDivideScalarAttrs : public ir::AttrsNode<LegacyNpiRtrueDivideScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiRtrueDivideScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRtrueDivideScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiSign
using LegacyNpiSignAttrs = ir::Attrs;
// LegacyNpiSin
using LegacyNpiSinAttrs = ir::Attrs;
// LegacyNpiSinh
using LegacyNpiSinhAttrs = ir::Attrs;
// LegacyNpiSqrt
using LegacyNpiSqrtAttrs = ir::Attrs;
// LegacyNpiSquare
using LegacyNpiSquareAttrs = ir::Attrs;
// LegacyNpiStd
class LegacyNpiStdAttrs : public ir::AttrsNode<LegacyNpiStdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  int ddof;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiStdAttrs, "mxnet.v3.attrs.LegacyNpiStdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ddof);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyNpiSubtract
using LegacyNpiSubtractAttrs = ir::Attrs;
// LegacyNpiSubtractScalar
class LegacyNpiSubtractScalarAttrs : public ir::AttrsNode<LegacyNpiSubtractScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiSubtractScalarAttrs, "mxnet.v3.attrs.LegacyNpiSubtractScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiTan
using LegacyNpiTanAttrs = ir::Attrs;
// LegacyNpiTanh
using LegacyNpiTanhAttrs = ir::Attrs;
// LegacyNpiTensordot
class LegacyNpiTensordotAttrs : public ir::AttrsNode<LegacyNpiTensordotAttrs> {
 public:
  ir::Array<ir::Integer> a_axes_summed;
  ir::Array<ir::Integer> b_axes_summed;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiTensordotAttrs, "mxnet.v3.attrs.LegacyNpiTensordotAttrs") {
    MX_V3_ATTR_FIELD(a_axes_summed);
    MX_V3_ATTR_FIELD(b_axes_summed);
  }
};
// LegacyNpiTensordotIntAxes
class LegacyNpiTensordotIntAxesAttrs : public ir::AttrsNode<LegacyNpiTensordotIntAxesAttrs> {
 public:
  int axes;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiTensordotIntAxesAttrs,
                      "mxnet.v3.attrs.LegacyNpiTensordotIntAxesAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// LegacyNpiTril
class LegacyNpiTrilAttrs : public ir::AttrsNode<LegacyNpiTrilAttrs> {
 public:
  int k;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiTrilAttrs, "mxnet.v3.attrs.LegacyNpiTrilAttrs") {
    MX_V3_ATTR_FIELD(k);
  }
};
// LegacyNpiTrueDivide
using LegacyNpiTrueDivideAttrs = ir::Attrs;
// LegacyNpiTrueDivideScalar
class LegacyNpiTrueDivideScalarAttrs : public ir::AttrsNode<LegacyNpiTrueDivideScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiTrueDivideScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiTrueDivideScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyNpiTrunc
using LegacyNpiTruncAttrs = ir::Attrs;
// LegacyNpiUniform
class LegacyNpiUniformAttrs : public ir::AttrsNode<LegacyNpiUniformAttrs> {
 public:
  double low;
  double high;
  ir::Array<ir::Integer> size;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiUniformAttrs, "mxnet.v3.attrs.LegacyNpiUniformAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(size);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpiUnique
class LegacyNpiUniqueAttrs : public ir::AttrsNode<LegacyNpiUniqueAttrs> {
 public:
  bool return_index;
  bool return_inverse;
  bool return_counts;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiUniqueAttrs, "mxnet.v3.attrs.LegacyNpiUniqueAttrs") {
    MX_V3_ATTR_FIELD(return_index);
    MX_V3_ATTR_FIELD(return_inverse);
    MX_V3_ATTR_FIELD(return_counts);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyNpiVar
class LegacyNpiVarAttrs : public ir::AttrsNode<LegacyNpiVarAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  int ddof;
  bool keepdims;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiVarAttrs, "mxnet.v3.attrs.LegacyNpiVarAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ddof);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// LegacyNpiZeros
class LegacyNpiZerosAttrs : public ir::AttrsNode<LegacyNpiZerosAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyNpiZerosAttrs, "mxnet.v3.attrs.LegacyNpiZerosAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyNpxNonzero
using LegacyNpxNonzeroAttrs = ir::Attrs;
// LegacyNpxRelu
using LegacyNpxReluAttrs = ir::Attrs;
// LegacyNpxSigmoid
using LegacyNpxSigmoidAttrs = ir::Attrs;
// LegacyOneHot
class LegacyOneHotAttrs : public ir::AttrsNode<LegacyOneHotAttrs> {
 public:
  int depth;
  double on_value;
  double off_value;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyOneHotAttrs, "mxnet.v3.attrs.LegacyOneHotAttrs") {
    MX_V3_ATTR_FIELD(depth);
    MX_V3_ATTR_FIELD(on_value);
    MX_V3_ATTR_FIELD(off_value);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyOnehotEncode
using LegacyOnehotEncodeAttrs = ir::Attrs;
// LegacyOnes
class LegacyOnesAttrs : public ir::AttrsNode<LegacyOnesAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyOnesAttrs, "mxnet.v3.attrs.LegacyOnesAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyOnesLike
using LegacyOnesLikeAttrs = ir::Attrs;
// LegacyPad
class LegacyPadAttrs : public ir::AttrsNode<LegacyPadAttrs> {
 public:
  std::string mode;
  ir::Array<ir::Integer> pad_width;
  double constant_value;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyPadAttrs, "mxnet.v3.attrs.LegacyPadAttrs") {
    MX_V3_ATTR_FIELD(mode);
    MX_V3_ATTR_FIELD(pad_width);
    MX_V3_ATTR_FIELD(constant_value);
  }
};
// LegacyPick
class LegacyPickAttrs : public ir::AttrsNode<LegacyPickAttrs> {
 public:
  int axis;
  bool keepdims;
  std::string mode;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyPickAttrs, "mxnet.v3.attrs.LegacyPickAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(mode);
  }
};
// LegacyPlusScalar
class LegacyPlusScalarAttrs : public ir::AttrsNode<LegacyPlusScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyPlusScalarAttrs, "mxnet.v3.attrs.LegacyPlusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyPooling
class LegacyPoolingAttrs : public ir::AttrsNode<LegacyPoolingAttrs> {
 public:
  ir::Array<ir::Integer> kernel;
  std::string pool_type;
  bool global_pool;
  bool cudnn_off;
  std::string pooling_convention;
  ir::Array<ir::Integer> stride;
  ir::Array<ir::Integer> pad;
  int p_value;
  bool count_include_pad;
  std::string layout;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyPoolingAttrs, "mxnet.v3.attrs.LegacyPoolingAttrs") {
    MX_V3_ATTR_FIELD(kernel);
    MX_V3_ATTR_FIELD(pool_type);
    MX_V3_ATTR_FIELD(global_pool);
    MX_V3_ATTR_FIELD(cudnn_off);
    MX_V3_ATTR_FIELD(pooling_convention);
    MX_V3_ATTR_FIELD(stride);
    MX_V3_ATTR_FIELD(pad);
    MX_V3_ATTR_FIELD(p_value);
    MX_V3_ATTR_FIELD(count_include_pad);
    MX_V3_ATTR_FIELD(layout);
  }
};
// LegacyPower
using LegacyPowerAttrs = ir::Attrs;
// LegacyPowerScalar
class LegacyPowerScalarAttrs : public ir::AttrsNode<LegacyPowerScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyPowerScalarAttrs, "mxnet.v3.attrs.LegacyPowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyProd
class LegacyProdAttrs : public ir::AttrsNode<LegacyProdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyProdAttrs, "mxnet.v3.attrs.LegacyProdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacyRNN
class LegacyRNNAttrs : public ir::AttrsNode<LegacyRNNAttrs> {
 public:
  int state_size;
  int num_layers;
  bool bidirectional;
  std::string mode;
  double p;
  bool state_outputs;
  int projection_size;
  double lstm_state_clip_min;
  double lstm_state_clip_max;
  bool lstm_state_clip_nan;
  bool use_sequence_length;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRNNAttrs, "mxnet.v3.attrs.LegacyRNNAttrs") {
    MX_V3_ATTR_FIELD(state_size);
    MX_V3_ATTR_FIELD(num_layers);
    MX_V3_ATTR_FIELD(bidirectional);
    MX_V3_ATTR_FIELD(mode);
    MX_V3_ATTR_FIELD(p);
    MX_V3_ATTR_FIELD(state_outputs);
    MX_V3_ATTR_FIELD(projection_size);
    MX_V3_ATTR_FIELD(lstm_state_clip_min);
    MX_V3_ATTR_FIELD(lstm_state_clip_max);
    MX_V3_ATTR_FIELD(lstm_state_clip_nan);
    MX_V3_ATTR_FIELD(use_sequence_length);
  }
};
// LegacyROIPooling
class LegacyROIPoolingAttrs : public ir::AttrsNode<LegacyROIPoolingAttrs> {
 public:
  ir::Array<ir::Integer> pooled_size;
  double spatial_scale;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyROIPoolingAttrs, "mxnet.v3.attrs.LegacyROIPoolingAttrs") {
    MX_V3_ATTR_FIELD(pooled_size);
    MX_V3_ATTR_FIELD(spatial_scale);
  }
};
// LegacyRadians
using LegacyRadiansAttrs = ir::Attrs;
// LegacyRandomExponential
class LegacyRandomExponentialAttrs : public ir::AttrsNode<LegacyRandomExponentialAttrs> {
 public:
  double lam;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomExponentialAttrs, "mxnet.v3.attrs.LegacyRandomExponentialAttrs") {
    MX_V3_ATTR_FIELD(lam);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomExponentialLike
class LegacyRandomExponentialLikeAttrs : public ir::AttrsNode<LegacyRandomExponentialLikeAttrs> {
 public:
  double lam;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomExponentialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomExponentialLikeAttrs") {
    MX_V3_ATTR_FIELD(lam);
  }
};
// LegacyRandomGamma
class LegacyRandomGammaAttrs : public ir::AttrsNode<LegacyRandomGammaAttrs> {
 public:
  double alpha;
  double beta;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomGammaAttrs, "mxnet.v3.attrs.LegacyRandomGammaAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomGammaLike
class LegacyRandomGammaLikeAttrs : public ir::AttrsNode<LegacyRandomGammaLikeAttrs> {
 public:
  double alpha;
  double beta;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomGammaLikeAttrs, "mxnet.v3.attrs.LegacyRandomGammaLikeAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
  }
};
// LegacyRandomGeneralizedNegativeBinomial
class LegacyRandomGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomGeneralizedNegativeBinomialAttrs> {
 public:
  double mu;
  double alpha;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(mu);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomGeneralizedNegativeBinomialLike
class LegacyRandomGeneralizedNegativeBinomialLikeAttrs
    : public ir::AttrsNode<LegacyRandomGeneralizedNegativeBinomialLikeAttrs> {
 public:
  double mu;
  double alpha;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomGeneralizedNegativeBinomialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomGeneralizedNegativeBinomialLikeAttrs") {
    MX_V3_ATTR_FIELD(mu);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// LegacyRandomNegativeBinomial
class LegacyRandomNegativeBinomialAttrs : public ir::AttrsNode<LegacyRandomNegativeBinomialAttrs> {
 public:
  int k;
  double p;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(p);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomNegativeBinomialLike
class LegacyRandomNegativeBinomialLikeAttrs
    : public ir::AttrsNode<LegacyRandomNegativeBinomialLikeAttrs> {
 public:
  int k;
  double p;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomNegativeBinomialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomNegativeBinomialLikeAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(p);
  }
};
// LegacyRandomNormal
class LegacyRandomNormalAttrs : public ir::AttrsNode<LegacyRandomNormalAttrs> {
 public:
  double loc;
  double scale;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomNormalAttrs, "mxnet.v3.attrs.LegacyRandomNormalAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomNormalLike
class LegacyRandomNormalLikeAttrs : public ir::AttrsNode<LegacyRandomNormalLikeAttrs> {
 public:
  double loc;
  double scale;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomNormalLikeAttrs, "mxnet.v3.attrs.LegacyRandomNormalLikeAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
  }
};
// LegacyRandomPdfDirichlet
class LegacyRandomPdfDirichletAttrs : public ir::AttrsNode<LegacyRandomPdfDirichletAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfDirichletAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfDirichletAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfExponential
class LegacyRandomPdfExponentialAttrs : public ir::AttrsNode<LegacyRandomPdfExponentialAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfExponentialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfExponentialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfGamma
class LegacyRandomPdfGammaAttrs : public ir::AttrsNode<LegacyRandomPdfGammaAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfGammaAttrs, "mxnet.v3.attrs.LegacyRandomPdfGammaAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfGeneralizedNegativeBinomial
class LegacyRandomPdfGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomPdfGeneralizedNegativeBinomialAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfNegativeBinomial
class LegacyRandomPdfNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomPdfNegativeBinomialAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfNormal
class LegacyRandomPdfNormalAttrs : public ir::AttrsNode<LegacyRandomPdfNormalAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfNormalAttrs, "mxnet.v3.attrs.LegacyRandomPdfNormalAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfPoisson
class LegacyRandomPdfPoissonAttrs : public ir::AttrsNode<LegacyRandomPdfPoissonAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfPoissonAttrs, "mxnet.v3.attrs.LegacyRandomPdfPoissonAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPdfUniform
class LegacyRandomPdfUniformAttrs : public ir::AttrsNode<LegacyRandomPdfUniformAttrs> {
 public:
  bool is_log;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfUniformAttrs, "mxnet.v3.attrs.LegacyRandomPdfUniformAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// LegacyRandomPoisson
class LegacyRandomPoissonAttrs : public ir::AttrsNode<LegacyRandomPoissonAttrs> {
 public:
  double lam;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPoissonAttrs, "mxnet.v3.attrs.LegacyRandomPoissonAttrs") {
    MX_V3_ATTR_FIELD(lam);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomPoissonLike
class LegacyRandomPoissonLikeAttrs : public ir::AttrsNode<LegacyRandomPoissonLikeAttrs> {
 public:
  double lam;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomPoissonLikeAttrs, "mxnet.v3.attrs.LegacyRandomPoissonLikeAttrs") {
    MX_V3_ATTR_FIELD(lam);
  }
};
// LegacyRandomRandint
class LegacyRandomRandintAttrs : public ir::AttrsNode<LegacyRandomRandintAttrs> {
 public:
  int64_t low;
  int64_t high;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomRandintAttrs, "mxnet.v3.attrs.LegacyRandomRandintAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomUniform
class LegacyRandomUniformAttrs : public ir::AttrsNode<LegacyRandomUniformAttrs> {
 public:
  double low;
  double high;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomUniformAttrs, "mxnet.v3.attrs.LegacyRandomUniformAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyRandomUniformLike
class LegacyRandomUniformLikeAttrs : public ir::AttrsNode<LegacyRandomUniformLikeAttrs> {
 public:
  double low;
  double high;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRandomUniformLikeAttrs, "mxnet.v3.attrs.LegacyRandomUniformLikeAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
  }
};
// LegacyRavelMultiIndex
class LegacyRavelMultiIndexAttrs : public ir::AttrsNode<LegacyRavelMultiIndexAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRavelMultiIndexAttrs, "mxnet.v3.attrs.LegacyRavelMultiIndexAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyRcbrt
using LegacyRcbrtAttrs = ir::Attrs;
// LegacyRdivScalar
class LegacyRdivScalarAttrs : public ir::AttrsNode<LegacyRdivScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRdivScalarAttrs, "mxnet.v3.attrs.LegacyRdivScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyReciprocal
using LegacyReciprocalAttrs = ir::Attrs;
// LegacyRelu
using LegacyReluAttrs = ir::Attrs;
// LegacyRepeat
class LegacyRepeatAttrs : public ir::AttrsNode<LegacyRepeatAttrs> {
 public:
  int repeats;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRepeatAttrs, "mxnet.v3.attrs.LegacyRepeatAttrs") {
    MX_V3_ATTR_FIELD(repeats);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyReshape
class LegacyReshapeAttrs : public ir::AttrsNode<LegacyReshapeAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  bool reverse;
  ir::Array<ir::Integer> target_shape;
  bool keep_highest;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyReshapeAttrs, "mxnet.v3.attrs.LegacyReshapeAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(reverse);
    MX_V3_ATTR_FIELD(target_shape);
    MX_V3_ATTR_FIELD(keep_highest);
  }
};
// LegacyReshapeLike
class LegacyReshapeLikeAttrs : public ir::AttrsNode<LegacyReshapeLikeAttrs> {
 public:
  int lhs_begin;
  int lhs_end;
  int rhs_begin;
  int rhs_end;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyReshapeLikeAttrs, "mxnet.v3.attrs.LegacyReshapeLikeAttrs") {
    MX_V3_ATTR_FIELD(lhs_begin);
    MX_V3_ATTR_FIELD(lhs_end);
    MX_V3_ATTR_FIELD(rhs_begin);
    MX_V3_ATTR_FIELD(rhs_end);
  }
};
// LegacyReverse
class LegacyReverseAttrs : public ir::AttrsNode<LegacyReverseAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyReverseAttrs, "mxnet.v3.attrs.LegacyReverseAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacyRint
using LegacyRintAttrs = ir::Attrs;
// LegacyRminusScalar
class LegacyRminusScalarAttrs : public ir::AttrsNode<LegacyRminusScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRminusScalarAttrs, "mxnet.v3.attrs.LegacyRminusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyRmodScalar
class LegacyRmodScalarAttrs : public ir::AttrsNode<LegacyRmodScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRmodScalarAttrs, "mxnet.v3.attrs.LegacyRmodScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyRound
using LegacyRoundAttrs = ir::Attrs;
// LegacyRpowerScalar
class LegacyRpowerScalarAttrs : public ir::AttrsNode<LegacyRpowerScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyRpowerScalarAttrs, "mxnet.v3.attrs.LegacyRpowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyRsqrt
using LegacyRsqrtAttrs = ir::Attrs;
// LegacySampleExponential
class LegacySampleExponentialAttrs : public ir::AttrsNode<LegacySampleExponentialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleExponentialAttrs, "mxnet.v3.attrs.LegacySampleExponentialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleGamma
class LegacySampleGammaAttrs : public ir::AttrsNode<LegacySampleGammaAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleGammaAttrs, "mxnet.v3.attrs.LegacySampleGammaAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleGeneralizedNegativeBinomial
class LegacySampleGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacySampleGeneralizedNegativeBinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacySampleGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleMultinomial
class LegacySampleMultinomialAttrs : public ir::AttrsNode<LegacySampleMultinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  bool get_prob;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleMultinomialAttrs, "mxnet.v3.attrs.LegacySampleMultinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(get_prob);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleNegativeBinomial
class LegacySampleNegativeBinomialAttrs : public ir::AttrsNode<LegacySampleNegativeBinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacySampleNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleNormal
class LegacySampleNormalAttrs : public ir::AttrsNode<LegacySampleNormalAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleNormalAttrs, "mxnet.v3.attrs.LegacySampleNormalAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySamplePoisson
class LegacySamplePoissonAttrs : public ir::AttrsNode<LegacySamplePoissonAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySamplePoissonAttrs, "mxnet.v3.attrs.LegacySamplePoissonAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleUniform
class LegacySampleUniformAttrs : public ir::AttrsNode<LegacySampleUniformAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleUniformAttrs, "mxnet.v3.attrs.LegacySampleUniformAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacySampleUniqueZipfian
class LegacySampleUniqueZipfianAttrs : public ir::AttrsNode<LegacySampleUniqueZipfianAttrs> {
 public:
  int range_max;
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySampleUniqueZipfianAttrs,
                      "mxnet.v3.attrs.LegacySampleUniqueZipfianAttrs") {
    MX_V3_ATTR_FIELD(range_max);
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyScatterElemwiseDiv
using LegacyScatterElemwiseDivAttrs = ir::Attrs;
// LegacyScatterMinusScalar
class LegacyScatterMinusScalarAttrs : public ir::AttrsNode<LegacyScatterMinusScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyScatterMinusScalarAttrs,
                      "mxnet.v3.attrs.LegacyScatterMinusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyScatterNd
class LegacyScatterNdAttrs : public ir::AttrsNode<LegacyScatterNdAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyScatterNdAttrs, "mxnet.v3.attrs.LegacyScatterNdAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyScatterPlusScalar
class LegacyScatterPlusScalarAttrs : public ir::AttrsNode<LegacyScatterPlusScalarAttrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyScatterPlusScalarAttrs, "mxnet.v3.attrs.LegacyScatterPlusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacyScatterSetNd
class LegacyScatterSetNdAttrs : public ir::AttrsNode<LegacyScatterSetNdAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyScatterSetNdAttrs, "mxnet.v3.attrs.LegacyScatterSetNdAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacySequenceLast
class LegacySequenceLastAttrs : public ir::AttrsNode<LegacySequenceLastAttrs> {
 public:
  bool use_sequence_length;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySequenceLastAttrs, "mxnet.v3.attrs.LegacySequenceLastAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacySequenceMask
class LegacySequenceMaskAttrs : public ir::AttrsNode<LegacySequenceMaskAttrs> {
 public:
  bool use_sequence_length;
  double value;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySequenceMaskAttrs, "mxnet.v3.attrs.LegacySequenceMaskAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(value);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacySequenceReverse
class LegacySequenceReverseAttrs : public ir::AttrsNode<LegacySequenceReverseAttrs> {
 public:
  bool use_sequence_length;
  int axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySequenceReverseAttrs, "mxnet.v3.attrs.LegacySequenceReverseAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacySetValue
class LegacySetValueAttrs : public ir::AttrsNode<LegacySetValueAttrs> {
 public:
  double src;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySetValueAttrs, "mxnet.v3.attrs.LegacySetValueAttrs") {
    MX_V3_ATTR_FIELD(src);
  }
};
// LegacySgMkldnnConv
using LegacySgMkldnnConvAttrs = ir::Attrs;
// LegacySgMkldnnFullyConnected
using LegacySgMkldnnFullyConnectedAttrs = ir::Attrs;
// LegacyShapeArray
using LegacyShapeArrayAttrs = ir::Attrs;
// LegacyShuffle
using LegacyShuffleAttrs = ir::Attrs;
// LegacySigmoid
using LegacySigmoidAttrs = ir::Attrs;
// LegacySign
using LegacySignAttrs = ir::Attrs;
// LegacySin
using LegacySinAttrs = ir::Attrs;
// LegacySinh
using LegacySinhAttrs = ir::Attrs;
// LegacySizeArray
using LegacySizeArrayAttrs = ir::Attrs;
// LegacySlice
class LegacySliceAttrs : public ir::AttrsNode<LegacySliceAttrs> {
 public:
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceAttrs, "mxnet.v3.attrs.LegacySliceAttrs") {
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// LegacySliceAssign
class LegacySliceAssignAttrs : public ir::AttrsNode<LegacySliceAssignAttrs> {
 public:
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceAssignAttrs, "mxnet.v3.attrs.LegacySliceAssignAttrs") {
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// LegacySliceAssignScalar
class LegacySliceAssignScalarAttrs : public ir::AttrsNode<LegacySliceAssignScalarAttrs> {
 public:
  double scalar;
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceAssignScalarAttrs, "mxnet.v3.attrs.LegacySliceAssignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// LegacySliceAxis
class LegacySliceAxisAttrs : public ir::AttrsNode<LegacySliceAxisAttrs> {
 public:
  int axis;
  int begin;
  int end;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceAxisAttrs, "mxnet.v3.attrs.LegacySliceAxisAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
  }
};
// LegacySliceChannel
class LegacySliceChannelAttrs : public ir::AttrsNode<LegacySliceChannelAttrs> {
 public:
  int num_outputs;
  int axis;
  bool squeeze_axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceChannelAttrs, "mxnet.v3.attrs.LegacySliceChannelAttrs") {
    MX_V3_ATTR_FIELD(num_outputs);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(squeeze_axis);
  }
};
// LegacySliceLike
class LegacySliceLikeAttrs : public ir::AttrsNode<LegacySliceLikeAttrs> {
 public:
  ir::Array<ir::Integer> axes;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySliceLikeAttrs, "mxnet.v3.attrs.LegacySliceLikeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// LegacySmoothL1
class LegacySmoothL1Attrs : public ir::AttrsNode<LegacySmoothL1Attrs> {
 public:
  double scalar;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySmoothL1Attrs, "mxnet.v3.attrs.LegacySmoothL1Attrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// LegacySoftmax
class LegacySoftmaxAttrs : public ir::AttrsNode<LegacySoftmaxAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySoftmaxAttrs, "mxnet.v3.attrs.LegacySoftmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// LegacySoftmaxActivation
class LegacySoftmaxActivationAttrs : public ir::AttrsNode<LegacySoftmaxActivationAttrs> {
 public:
  std::string mode;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySoftmaxActivationAttrs, "mxnet.v3.attrs.LegacySoftmaxActivationAttrs") {
    MX_V3_ATTR_FIELD(mode);
  }
};
// LegacySoftmaxCrossEntropy
using LegacySoftmaxCrossEntropyAttrs = ir::Attrs;
// LegacySoftmaxOutput
class LegacySoftmaxOutputAttrs : public ir::AttrsNode<LegacySoftmaxOutputAttrs> {
 public:
  double grad_scale;
  double ignore_label;
  bool multi_output;
  bool use_ignore;
  bool preserve_shape;
  std::string normalization;
  bool out_grad;
  double smooth_alpha;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySoftmaxOutputAttrs, "mxnet.v3.attrs.LegacySoftmaxOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
    MX_V3_ATTR_FIELD(ignore_label);
    MX_V3_ATTR_FIELD(multi_output);
    MX_V3_ATTR_FIELD(use_ignore);
    MX_V3_ATTR_FIELD(preserve_shape);
    MX_V3_ATTR_FIELD(normalization);
    MX_V3_ATTR_FIELD(out_grad);
    MX_V3_ATTR_FIELD(smooth_alpha);
  }
};
// LegacySoftmin
class LegacySoftminAttrs : public ir::AttrsNode<LegacySoftminAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySoftminAttrs, "mxnet.v3.attrs.LegacySoftminAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// LegacySoftsign
using LegacySoftsignAttrs = ir::Attrs;
// LegacySort
class LegacySortAttrs : public ir::AttrsNode<LegacySortAttrs> {
 public:
  int axis;
  bool is_ascend;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySortAttrs, "mxnet.v3.attrs.LegacySortAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(is_ascend);
  }
};
// LegacySpaceToDepth
class LegacySpaceToDepthAttrs : public ir::AttrsNode<LegacySpaceToDepthAttrs> {
 public:
  int block_size;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySpaceToDepthAttrs, "mxnet.v3.attrs.LegacySpaceToDepthAttrs") {
    MX_V3_ATTR_FIELD(block_size);
  }
};
// LegacySparseRetain
using LegacySparseRetainAttrs = ir::Attrs;
// LegacySpatialTransformer
class LegacySpatialTransformerAttrs : public ir::AttrsNode<LegacySpatialTransformerAttrs> {
 public:
  ir::Array<ir::Integer> target_shape;
  std::string transform_type;
  std::string sampler_type;
  bool cudnn_off;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySpatialTransformerAttrs,
                      "mxnet.v3.attrs.LegacySpatialTransformerAttrs") {
    MX_V3_ATTR_FIELD(target_shape);
    MX_V3_ATTR_FIELD(transform_type);
    MX_V3_ATTR_FIELD(sampler_type);
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// LegacySplitV2
class LegacySplitV2Attrs : public ir::AttrsNode<LegacySplitV2Attrs> {
 public:
  ir::Array<ir::Integer> indices;
  int axis;
  bool squeeze_axis;
  int sections;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySplitV2Attrs, "mxnet.v3.attrs.LegacySplitV2Attrs") {
    MX_V3_ATTR_FIELD(indices);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(squeeze_axis);
    MX_V3_ATTR_FIELD(sections);
  }
};
// LegacySqrt
using LegacySqrtAttrs = ir::Attrs;
// LegacySquare
using LegacySquareAttrs = ir::Attrs;
// LegacySquareSum
class LegacySquareSumAttrs : public ir::AttrsNode<LegacySquareSumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySquareSumAttrs, "mxnet.v3.attrs.LegacySquareSumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacySqueeze
class LegacySqueezeAttrs : public ir::AttrsNode<LegacySqueezeAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySqueezeAttrs, "mxnet.v3.attrs.LegacySqueezeAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// LegacySum
class LegacySumAttrs : public ir::AttrsNode<LegacySumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySumAttrs, "mxnet.v3.attrs.LegacySumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// LegacySwapAxis
class LegacySwapAxisAttrs : public ir::AttrsNode<LegacySwapAxisAttrs> {
 public:
  int dim1;
  int dim2;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacySwapAxisAttrs, "mxnet.v3.attrs.LegacySwapAxisAttrs") {
    MX_V3_ATTR_FIELD(dim1);
    MX_V3_ATTR_FIELD(dim2);
  }
};
// LegacyTake
class LegacyTakeAttrs : public ir::AttrsNode<LegacyTakeAttrs> {
 public:
  int axis;
  std::string mode;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyTakeAttrs, "mxnet.v3.attrs.LegacyTakeAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(mode);
  }
};
// LegacyTan
using LegacyTanAttrs = ir::Attrs;
// LegacyTanh
using LegacyTanhAttrs = ir::Attrs;
// LegacyTile
class LegacyTileAttrs : public ir::AttrsNode<LegacyTileAttrs> {
 public:
  ir::Array<ir::Integer> reps;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyTileAttrs, "mxnet.v3.attrs.LegacyTileAttrs") { MX_V3_ATTR_FIELD(reps); }
};
// LegacyTranspose
class LegacyTransposeAttrs : public ir::AttrsNode<LegacyTransposeAttrs> {
 public:
  ir::Array<ir::Integer> axes;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyTransposeAttrs, "mxnet.v3.attrs.LegacyTransposeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// LegacyTrunc
using LegacyTruncAttrs = ir::Attrs;
// LegacyUnravelIndex
class LegacyUnravelIndexAttrs : public ir::AttrsNode<LegacyUnravelIndexAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyUnravelIndexAttrs, "mxnet.v3.attrs.LegacyUnravelIndexAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// LegacyWhere
using LegacyWhereAttrs = ir::Attrs;
// LegacyZeros
class LegacyZerosAttrs : public ir::AttrsNode<LegacyZerosAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyZerosAttrs, "mxnet.v3.attrs.LegacyZerosAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// LegacyZerosLike
using LegacyZerosLikeAttrs = ir::Attrs;
// LegacyZerosWithoutDtype
class LegacyZerosWithoutDtypeAttrs : public ir::AttrsNode<LegacyZerosWithoutDtypeAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  int dtype;
  ir::NodeRef capsule{nullptr};

  MX_V3_DECLARE_ATTRS(LegacyZerosWithoutDtypeAttrs, "mxnet.v3.attrs.LegacyZerosWithoutDtypeAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};

}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif

