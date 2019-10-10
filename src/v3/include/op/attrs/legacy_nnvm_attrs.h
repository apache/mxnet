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
// _copyto
using LegacyCopytoAttrs = ir::Attrs;
// all_finite
class LegacyAllFiniteAttrs : public ir::AttrsNode<LegacyAllFiniteAttrs> {
 public:
  bool init_output;

  MX_V3_DECLARE_ATTRS(LegacyAllFiniteAttrs, "mxnet.v3.attrs.LegacyAllFiniteAttrs") {
    MX_V3_ATTR_FIELD(init_output);
  }
};
// _npi_deg2rad
using LegacyNpiDeg2radAttrs = ir::Attrs;
// _npi_rad2deg
using LegacyNpiRad2degAttrs = ir::Attrs;
// IdentityAttachKLSparseReg
class LegacyIdentityAttachKLSparseRegAttrs
    : public ir::AttrsNode<LegacyIdentityAttachKLSparseRegAttrs> {
 public:
  double sparseness_target;
  double penalty;
  double momentum;

  MX_V3_DECLARE_ATTRS(LegacyIdentityAttachKLSparseRegAttrs,
                      "mxnet.v3.attrs.LegacyIdentityAttachKLSparseRegAttrs") {
    MX_V3_ATTR_FIELD(sparseness_target);
    MX_V3_ATTR_FIELD(penalty);
    MX_V3_ATTR_FIELD(momentum);
  }
};
// LeakyReLU
class LegacyLeakyReLUAttrs : public ir::AttrsNode<LegacyLeakyReLUAttrs> {
 public:
  std::string act_type;
  double slope;
  double lower_bound;
  double upper_bound;

  MX_V3_DECLARE_ATTRS(LegacyLeakyReLUAttrs, "mxnet.v3.attrs.LegacyLeakyReLUAttrs") {
    MX_V3_ATTR_FIELD(act_type);
    MX_V3_ATTR_FIELD(slope);
    MX_V3_ATTR_FIELD(lower_bound);
    MX_V3_ATTR_FIELD(upper_bound);
  }
};
// softmax_cross_entropy
using LegacySoftmaxCrossEntropyAttrs = ir::Attrs;
// Activation
class LegacyActivationAttrs : public ir::AttrsNode<LegacyActivationAttrs> {
 public:
  std::string act_type;

  MX_V3_DECLARE_ATTRS(LegacyActivationAttrs, "mxnet.v3.attrs.LegacyActivationAttrs") {
    MX_V3_ATTR_FIELD(act_type);
  }
};
// BatchNorm
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
// Convolution
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
// CTCLoss
class LegacyCTCLossAttrs : public ir::AttrsNode<LegacyCTCLossAttrs> {
 public:
  bool use_data_lengths;
  bool use_label_lengths;
  std::string blank_label;

  MX_V3_DECLARE_ATTRS(LegacyCTCLossAttrs, "mxnet.v3.attrs.LegacyCTCLossAttrs") {
    MX_V3_ATTR_FIELD(use_data_lengths);
    MX_V3_ATTR_FIELD(use_label_lengths);
    MX_V3_ATTR_FIELD(blank_label);
  }
};
// Deconvolution
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
// Dropout
class LegacyDropoutAttrs : public ir::AttrsNode<LegacyDropoutAttrs> {
 public:
  double p;
  std::string mode;
  ir::Array<ir::Integer> axes;
  bool cudnn_off;

  MX_V3_DECLARE_ATTRS(LegacyDropoutAttrs, "mxnet.v3.attrs.LegacyDropoutAttrs") {
    MX_V3_ATTR_FIELD(p);
    MX_V3_ATTR_FIELD(mode);
    MX_V3_ATTR_FIELD(axes);
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// FullyConnected
class LegacyFullyConnectedAttrs : public ir::AttrsNode<LegacyFullyConnectedAttrs> {
 public:
  int num_hidden;
  bool no_bias;
  bool flatten;

  MX_V3_DECLARE_ATTRS(LegacyFullyConnectedAttrs, "mxnet.v3.attrs.LegacyFullyConnectedAttrs") {
    MX_V3_ATTR_FIELD(num_hidden);
    MX_V3_ATTR_FIELD(no_bias);
    MX_V3_ATTR_FIELD(flatten);
  }
};
// GroupNorm
class LegacyGroupNormAttrs : public ir::AttrsNode<LegacyGroupNormAttrs> {
 public:
  int num_groups;
  double eps;
  bool output_mean_var;

  MX_V3_DECLARE_ATTRS(LegacyGroupNormAttrs, "mxnet.v3.attrs.LegacyGroupNormAttrs") {
    MX_V3_ATTR_FIELD(num_groups);
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(output_mean_var);
  }
};
// LayerNorm
class LegacyLayerNormAttrs : public ir::AttrsNode<LegacyLayerNormAttrs> {
 public:
  int axis;
  double eps;
  bool output_mean_var;

  MX_V3_DECLARE_ATTRS(LegacyLayerNormAttrs, "mxnet.v3.attrs.LegacyLayerNormAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(output_mean_var);
  }
};
// log_softmax
class LegacyLogSoftmaxAttrs : public ir::AttrsNode<LegacyLogSoftmaxAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;

  MX_V3_DECLARE_ATTRS(LegacyLogSoftmaxAttrs, "mxnet.v3.attrs.LegacyLogSoftmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// LRN
class LegacyLRNAttrs : public ir::AttrsNode<LegacyLRNAttrs> {
 public:
  double alpha;
  double beta;
  double knorm;
  int nsize;

  MX_V3_DECLARE_ATTRS(LegacyLRNAttrs, "mxnet.v3.attrs.LegacyLRNAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(knorm);
    MX_V3_ATTR_FIELD(nsize);
  }
};
// moments
class LegacyMomentsAttrs : public ir::AttrsNode<LegacyMomentsAttrs> {
 public:
  ir::Array<ir::Integer> axes;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyMomentsAttrs, "mxnet.v3.attrs.LegacyMomentsAttrs") {
    MX_V3_ATTR_FIELD(axes);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// Pooling
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
// softmax
class LegacySoftmaxAttrs : public ir::AttrsNode<LegacySoftmaxAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;

  MX_V3_DECLARE_ATTRS(LegacySoftmaxAttrs, "mxnet.v3.attrs.LegacySoftmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// SoftmaxActivation
class LegacySoftmaxActivationAttrs : public ir::AttrsNode<LegacySoftmaxActivationAttrs> {
 public:
  std::string mode;

  MX_V3_DECLARE_ATTRS(LegacySoftmaxActivationAttrs, "mxnet.v3.attrs.LegacySoftmaxActivationAttrs") {
    MX_V3_ATTR_FIELD(mode);
  }
};
// softmin
class LegacySoftminAttrs : public ir::AttrsNode<LegacySoftminAttrs> {
 public:
  int axis;
  double temperature;
  std::string dtype;
  bool use_length;

  MX_V3_DECLARE_ATTRS(LegacySoftminAttrs, "mxnet.v3.attrs.LegacySoftminAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(temperature);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(use_length);
  }
};
// _np__linalg_svd
using LegacyNpLinalgSvdAttrs = ir::Attrs;
// _npi_boolean_mask_assign_scalar
class LegacyNpiBooleanMaskAssignScalarAttrs
    : public ir::AttrsNode<LegacyNpiBooleanMaskAssignScalarAttrs> {
 public:
  double value;

  MX_V3_DECLARE_ATTRS(LegacyNpiBooleanMaskAssignScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiBooleanMaskAssignScalarAttrs") {
    MX_V3_ATTR_FIELD(value);
  }
};
// _npi_boolean_mask_assign_tensor
using LegacyNpiBooleanMaskAssignTensorAttrs = ir::Attrs;
// _npi_argmax
class LegacyNpiArgmaxAttrs : public ir::AttrsNode<LegacyNpiArgmaxAttrs> {
 public:
  int axis;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyNpiArgmaxAttrs, "mxnet.v3.attrs.LegacyNpiArgmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// _np_sum
class LegacyNpSumAttrs : public ir::AttrsNode<LegacyNpSumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;

  MX_V3_DECLARE_ATTRS(LegacyNpSumAttrs, "mxnet.v3.attrs.LegacyNpSumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// _np_max
class LegacyNpMaxAttrs : public ir::AttrsNode<LegacyNpMaxAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  double initial;

  MX_V3_DECLARE_ATTRS(LegacyNpMaxAttrs, "mxnet.v3.attrs.LegacyNpMaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// _np_min
class LegacyNpMinAttrs : public ir::AttrsNode<LegacyNpMinAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  double initial;

  MX_V3_DECLARE_ATTRS(LegacyNpMinAttrs, "mxnet.v3.attrs.LegacyNpMinAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// _np_prod
class LegacyNpProdAttrs : public ir::AttrsNode<LegacyNpProdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;

  MX_V3_DECLARE_ATTRS(LegacyNpProdAttrs, "mxnet.v3.attrs.LegacyNpProdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// _npi_mean
class LegacyNpiMeanAttrs : public ir::AttrsNode<LegacyNpiMeanAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  bool keepdims;
  double initial;

  MX_V3_DECLARE_ATTRS(LegacyNpiMeanAttrs, "mxnet.v3.attrs.LegacyNpiMeanAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(initial);
  }
};
// _npi_std
class LegacyNpiStdAttrs : public ir::AttrsNode<LegacyNpiStdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  int ddof;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyNpiStdAttrs, "mxnet.v3.attrs.LegacyNpiStdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ddof);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// _npi_var
class LegacyNpiVarAttrs : public ir::AttrsNode<LegacyNpiVarAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  std::string dtype;
  int ddof;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyNpiVarAttrs, "mxnet.v3.attrs.LegacyNpiVarAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ddof);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// _np_broadcast_to
class LegacyNpBroadcastToAttrs : public ir::AttrsNode<LegacyNpBroadcastToAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyNpBroadcastToAttrs, "mxnet.v3.attrs.LegacyNpBroadcastToAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// _np_cumsum
class LegacyNpCumsumAttrs : public ir::AttrsNode<LegacyNpCumsumAttrs> {
 public:
  int axis;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpCumsumAttrs, "mxnet.v3.attrs.LegacyNpCumsumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _np_dot
using LegacyNpDotAttrs = ir::Attrs;
// _npi_add
using LegacyNpiAddAttrs = ir::Attrs;
// _npi_subtract
using LegacyNpiSubtractAttrs = ir::Attrs;
// _npi_multiply
using LegacyNpiMultiplyAttrs = ir::Attrs;
// _npi_mod
using LegacyNpiModAttrs = ir::Attrs;
// _npi_power
using LegacyNpiPowerAttrs = ir::Attrs;
// _npi_copysign
using LegacyNpiCopysignAttrs = ir::Attrs;
// _npi_lcm
using LegacyNpiLcmAttrs = ir::Attrs;
// _npi_add_scalar
class LegacyNpiAddScalarAttrs : public ir::AttrsNode<LegacyNpiAddScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiAddScalarAttrs, "mxnet.v3.attrs.LegacyNpiAddScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_subtract_scalar
class LegacyNpiSubtractScalarAttrs : public ir::AttrsNode<LegacyNpiSubtractScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiSubtractScalarAttrs, "mxnet.v3.attrs.LegacyNpiSubtractScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rsubtract_scalar
class LegacyNpiRsubtractScalarAttrs : public ir::AttrsNode<LegacyNpiRsubtractScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRsubtractScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRsubtractScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_multiply_scalar
class LegacyNpiMultiplyScalarAttrs : public ir::AttrsNode<LegacyNpiMultiplyScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiMultiplyScalarAttrs, "mxnet.v3.attrs.LegacyNpiMultiplyScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_mod_scalar
class LegacyNpiModScalarAttrs : public ir::AttrsNode<LegacyNpiModScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiModScalarAttrs, "mxnet.v3.attrs.LegacyNpiModScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rmod_scalar
class LegacyNpiRmodScalarAttrs : public ir::AttrsNode<LegacyNpiRmodScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRmodScalarAttrs, "mxnet.v3.attrs.LegacyNpiRmodScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_power_scalar
class LegacyNpiPowerScalarAttrs : public ir::AttrsNode<LegacyNpiPowerScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiPowerScalarAttrs, "mxnet.v3.attrs.LegacyNpiPowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rpower_scalar
class LegacyNpiRpowerScalarAttrs : public ir::AttrsNode<LegacyNpiRpowerScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRpowerScalarAttrs, "mxnet.v3.attrs.LegacyNpiRpowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_copysign_scalar
class LegacyNpiCopysignScalarAttrs : public ir::AttrsNode<LegacyNpiCopysignScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiCopysignScalarAttrs, "mxnet.v3.attrs.LegacyNpiCopysignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rcopysign_scalar
class LegacyNpiRcopysignScalarAttrs : public ir::AttrsNode<LegacyNpiRcopysignScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRcopysignScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRcopysignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_arctan2
using LegacyNpiArctan2Attrs = ir::Attrs;
// _npi_arctan2_scalar
class LegacyNpiArctan2ScalarAttrs : public ir::AttrsNode<LegacyNpiArctan2ScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiArctan2ScalarAttrs, "mxnet.v3.attrs.LegacyNpiArctan2ScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rarctan2_scalar
class LegacyNpiRarctan2ScalarAttrs : public ir::AttrsNode<LegacyNpiRarctan2ScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRarctan2ScalarAttrs, "mxnet.v3.attrs.LegacyNpiRarctan2ScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_hypot
using LegacyNpiHypotAttrs = ir::Attrs;
// _npi_lcm_scalar
class LegacyNpiLcmScalarAttrs : public ir::AttrsNode<LegacyNpiLcmScalarAttrs> {
 public:
  int scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiLcmScalarAttrs, "mxnet.v3.attrs.LegacyNpiLcmScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_ldexp
using LegacyNpiLdexpAttrs = ir::Attrs;
// _npi_ldexp_scalar
class LegacyNpiLdexpScalarAttrs : public ir::AttrsNode<LegacyNpiLdexpScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiLdexpScalarAttrs, "mxnet.v3.attrs.LegacyNpiLdexpScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rldexp_scalar
class LegacyNpiRldexpScalarAttrs : public ir::AttrsNode<LegacyNpiRldexpScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRldexpScalarAttrs, "mxnet.v3.attrs.LegacyNpiRldexpScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npx_relu
using LegacyNpxReluAttrs = ir::Attrs;
// _npx_sigmoid
using LegacyNpxSigmoidAttrs = ir::Attrs;
// _np_copy
using LegacyNpCopyAttrs = ir::Attrs;
// _npi_negative
using LegacyNpiNegativeAttrs = ir::Attrs;
// _npi_reciprocal
using LegacyNpiReciprocalAttrs = ir::Attrs;
// _npi_absolute
using LegacyNpiAbsoluteAttrs = ir::Attrs;
// _npi_sign
using LegacyNpiSignAttrs = ir::Attrs;
// _npi_rint
using LegacyNpiRintAttrs = ir::Attrs;
// _npi_ceil
using LegacyNpiCeilAttrs = ir::Attrs;
// _npi_floor
using LegacyNpiFloorAttrs = ir::Attrs;
// _npi_trunc
using LegacyNpiTruncAttrs = ir::Attrs;
// _npi_fix
using LegacyNpiFixAttrs = ir::Attrs;
// _npi_square
using LegacyNpiSquareAttrs = ir::Attrs;
// _npi_sqrt
using LegacyNpiSqrtAttrs = ir::Attrs;
// _npi_cbrt
using LegacyNpiCbrtAttrs = ir::Attrs;
// _npi_exp
using LegacyNpiExpAttrs = ir::Attrs;
// _npi_log
using LegacyNpiLogAttrs = ir::Attrs;
// _npi_log10
using LegacyNpiLog10Attrs = ir::Attrs;
// _npi_log2
using LegacyNpiLog2Attrs = ir::Attrs;
// _npi_log1p
using LegacyNpiLog1pAttrs = ir::Attrs;
// _npi_expm1
using LegacyNpiExpm1Attrs = ir::Attrs;
// _npi_logical_not
using LegacyNpiLogicalNotAttrs = ir::Attrs;
// _npi_sin
using LegacyNpiSinAttrs = ir::Attrs;
// _npi_cos
using LegacyNpiCosAttrs = ir::Attrs;
// _npi_tan
using LegacyNpiTanAttrs = ir::Attrs;
// _npi_arcsin
using LegacyNpiArcsinAttrs = ir::Attrs;
// _npi_arccos
using LegacyNpiArccosAttrs = ir::Attrs;
// _npi_arctan
using LegacyNpiArctanAttrs = ir::Attrs;
// _npi_degrees
using LegacyNpiDegreesAttrs = ir::Attrs;
// _npi_radians
using LegacyNpiRadiansAttrs = ir::Attrs;
// _npi_sinh
using LegacyNpiSinhAttrs = ir::Attrs;
// _npi_cosh
using LegacyNpiCoshAttrs = ir::Attrs;
// _npi_tanh
using LegacyNpiTanhAttrs = ir::Attrs;
// _npi_arcsinh
using LegacyNpiArcsinhAttrs = ir::Attrs;
// _npi_arccosh
using LegacyNpiArccoshAttrs = ir::Attrs;
// _npi_arctanh
using LegacyNpiArctanhAttrs = ir::Attrs;
// _npi_around
class LegacyNpiAroundAttrs : public ir::AttrsNode<LegacyNpiAroundAttrs> {
 public:
  int decimals;

  MX_V3_DECLARE_ATTRS(LegacyNpiAroundAttrs, "mxnet.v3.attrs.LegacyNpiAroundAttrs") {
    MX_V3_ATTR_FIELD(decimals);
  }
};
// _npi_zeros
class LegacyNpiZerosAttrs : public ir::AttrsNode<LegacyNpiZerosAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiZerosAttrs, "mxnet.v3.attrs.LegacyNpiZerosAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_ones
class LegacyNpiOnesAttrs : public ir::AttrsNode<LegacyNpiOnesAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiOnesAttrs, "mxnet.v3.attrs.LegacyNpiOnesAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_identity
class LegacyNpiIdentityAttrs : public ir::AttrsNode<LegacyNpiIdentityAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiIdentityAttrs, "mxnet.v3.attrs.LegacyNpiIdentityAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _np_zeros_like
using LegacyNpZerosLikeAttrs = ir::Attrs;
// _np_ones_like
using LegacyNpOnesLikeAttrs = ir::Attrs;
// _npi_arange
class LegacyNpiArangeAttrs : public ir::AttrsNode<LegacyNpiArangeAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;

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
// _npi_indices
class LegacyNpiIndicesAttrs : public ir::AttrsNode<LegacyNpiIndicesAttrs> {
 public:
  ir::Array<ir::Integer> dimensions;
  std::string dtype;
  std::string ctx;

  MX_V3_DECLARE_ATTRS(LegacyNpiIndicesAttrs, "mxnet.v3.attrs.LegacyNpiIndicesAttrs") {
    MX_V3_ATTR_FIELD(dimensions);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(ctx);
  }
};
// _np_transpose
class LegacyNpTransposeAttrs : public ir::AttrsNode<LegacyNpTransposeAttrs> {
 public:
  ir::Array<ir::Integer> axes;

  MX_V3_DECLARE_ATTRS(LegacyNpTransposeAttrs, "mxnet.v3.attrs.LegacyNpTransposeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// _np_reshape
class LegacyNpReshapeAttrs : public ir::AttrsNode<LegacyNpReshapeAttrs> {
 public:
  ir::Array<ir::Integer> newshape;
  std::string order;

  MX_V3_DECLARE_ATTRS(LegacyNpReshapeAttrs, "mxnet.v3.attrs.LegacyNpReshapeAttrs") {
    MX_V3_ATTR_FIELD(newshape);
    MX_V3_ATTR_FIELD(order);
  }
};
// _np_squeeze
class LegacyNpSqueezeAttrs : public ir::AttrsNode<LegacyNpSqueezeAttrs> {
 public:
  ir::Array<ir::Integer> axis;

  MX_V3_DECLARE_ATTRS(LegacyNpSqueezeAttrs, "mxnet.v3.attrs.LegacyNpSqueezeAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// _np_roll
class LegacyNpRollAttrs : public ir::AttrsNode<LegacyNpRollAttrs> {
 public:
  ir::Array<ir::Integer> shift;
  ir::Array<ir::Integer> axis;

  MX_V3_DECLARE_ATTRS(LegacyNpRollAttrs, "mxnet.v3.attrs.LegacyNpRollAttrs") {
    MX_V3_ATTR_FIELD(shift);
    MX_V3_ATTR_FIELD(axis);
  }
};
// _npi_flip
class LegacyNpiFlipAttrs : public ir::AttrsNode<LegacyNpiFlipAttrs> {
 public:
  ir::Array<ir::Integer> axis;

  MX_V3_DECLARE_ATTRS(LegacyNpiFlipAttrs, "mxnet.v3.attrs.LegacyNpiFlipAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// _npx_nonzero
using LegacyNpxNonzeroAttrs = ir::Attrs;
// _npi_tensordot
class LegacyNpiTensordotAttrs : public ir::AttrsNode<LegacyNpiTensordotAttrs> {
 public:
  ir::Array<ir::Integer> a_axes_summed;
  ir::Array<ir::Integer> b_axes_summed;

  MX_V3_DECLARE_ATTRS(LegacyNpiTensordotAttrs, "mxnet.v3.attrs.LegacyNpiTensordotAttrs") {
    MX_V3_ATTR_FIELD(a_axes_summed);
    MX_V3_ATTR_FIELD(b_axes_summed);
  }
};
// _npi_tensordot_int_axes
class LegacyNpiTensordotIntAxesAttrs : public ir::AttrsNode<LegacyNpiTensordotIntAxesAttrs> {
 public:
  int axes;

  MX_V3_DECLARE_ATTRS(LegacyNpiTensordotIntAxesAttrs,
                      "mxnet.v3.attrs.LegacyNpiTensordotIntAxesAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// _np_trace
class LegacyNpTraceAttrs : public ir::AttrsNode<LegacyNpTraceAttrs> {
 public:
  int offset;
  int axis1;
  int axis2;

  MX_V3_DECLARE_ATTRS(LegacyNpTraceAttrs, "mxnet.v3.attrs.LegacyNpTraceAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(axis1);
    MX_V3_ATTR_FIELD(axis2);
  }
};
// _npi_tril
class LegacyNpiTrilAttrs : public ir::AttrsNode<LegacyNpiTrilAttrs> {
 public:
  int k;

  MX_V3_DECLARE_ATTRS(LegacyNpiTrilAttrs, "mxnet.v3.attrs.LegacyNpiTrilAttrs") {
    MX_V3_ATTR_FIELD(k);
  }
};
// _npi_true_divide
using LegacyNpiTrueDivideAttrs = ir::Attrs;
// _npi_true_divide_scalar
class LegacyNpiTrueDivideScalarAttrs : public ir::AttrsNode<LegacyNpiTrueDivideScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiTrueDivideScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiTrueDivideScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_rtrue_divide_scalar
class LegacyNpiRtrueDivideScalarAttrs : public ir::AttrsNode<LegacyNpiRtrueDivideScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNpiRtrueDivideScalarAttrs,
                      "mxnet.v3.attrs.LegacyNpiRtrueDivideScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _npi_unique
class LegacyNpiUniqueAttrs : public ir::AttrsNode<LegacyNpiUniqueAttrs> {
 public:
  bool return_index;
  bool return_inverse;
  bool return_counts;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacyNpiUniqueAttrs, "mxnet.v3.attrs.LegacyNpiUniqueAttrs") {
    MX_V3_ATTR_FIELD(return_index);
    MX_V3_ATTR_FIELD(return_inverse);
    MX_V3_ATTR_FIELD(return_counts);
    MX_V3_ATTR_FIELD(axis);
  }
};
// _npi_hanning
class LegacyNpiHanningAttrs : public ir::AttrsNode<LegacyNpiHanningAttrs> {
 public:
  int M;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiHanningAttrs, "mxnet.v3.attrs.LegacyNpiHanningAttrs") {
    MX_V3_ATTR_FIELD(M);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_hamming
class LegacyNpiHammingAttrs : public ir::AttrsNode<LegacyNpiHammingAttrs> {
 public:
  int M;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiHammingAttrs, "mxnet.v3.attrs.LegacyNpiHammingAttrs") {
    MX_V3_ATTR_FIELD(M);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_blackman
class LegacyNpiBlackmanAttrs : public ir::AttrsNode<LegacyNpiBlackmanAttrs> {
 public:
  int M;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiBlackmanAttrs, "mxnet.v3.attrs.LegacyNpiBlackmanAttrs") {
    MX_V3_ATTR_FIELD(M);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_normal
class LegacyNpiNormalAttrs : public ir::AttrsNode<LegacyNpiNormalAttrs> {
 public:
  double loc;
  double scale;
  ir::Array<ir::Integer> size;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiNormalAttrs, "mxnet.v3.attrs.LegacyNpiNormalAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
    MX_V3_ATTR_FIELD(size);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _npi_uniform
class LegacyNpiUniformAttrs : public ir::AttrsNode<LegacyNpiUniformAttrs> {
 public:
  double low;
  double high;
  ir::Array<ir::Integer> size;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyNpiUniformAttrs, "mxnet.v3.attrs.LegacyNpiUniformAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(size);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// Pad
class LegacyPadAttrs : public ir::AttrsNode<LegacyPadAttrs> {
 public:
  std::string mode;
  ir::Array<ir::Integer> pad_width;
  double constant_value;

  MX_V3_DECLARE_ATTRS(LegacyPadAttrs, "mxnet.v3.attrs.LegacyPadAttrs") {
    MX_V3_ATTR_FIELD(mode);
    MX_V3_ATTR_FIELD(pad_width);
    MX_V3_ATTR_FIELD(constant_value);
  }
};
// Flatten
using LegacyFlattenAttrs = ir::Attrs;
// _sample_uniform
class LegacySampleUniformAttrs : public ir::AttrsNode<LegacySampleUniformAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleUniformAttrs, "mxnet.v3.attrs.LegacySampleUniformAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_normal
class LegacySampleNormalAttrs : public ir::AttrsNode<LegacySampleNormalAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleNormalAttrs, "mxnet.v3.attrs.LegacySampleNormalAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_gamma
class LegacySampleGammaAttrs : public ir::AttrsNode<LegacySampleGammaAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleGammaAttrs, "mxnet.v3.attrs.LegacySampleGammaAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_exponential
class LegacySampleExponentialAttrs : public ir::AttrsNode<LegacySampleExponentialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleExponentialAttrs, "mxnet.v3.attrs.LegacySampleExponentialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_poisson
class LegacySamplePoissonAttrs : public ir::AttrsNode<LegacySamplePoissonAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySamplePoissonAttrs, "mxnet.v3.attrs.LegacySamplePoissonAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_negative_binomial
class LegacySampleNegativeBinomialAttrs : public ir::AttrsNode<LegacySampleNegativeBinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacySampleNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _sample_generalized_negative_binomial
class LegacySampleGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacySampleGeneralizedNegativeBinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacySampleGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_pdf_uniform
class LegacyRandomPdfUniformAttrs : public ir::AttrsNode<LegacyRandomPdfUniformAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfUniformAttrs, "mxnet.v3.attrs.LegacyRandomPdfUniformAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_normal
class LegacyRandomPdfNormalAttrs : public ir::AttrsNode<LegacyRandomPdfNormalAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfNormalAttrs, "mxnet.v3.attrs.LegacyRandomPdfNormalAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_gamma
class LegacyRandomPdfGammaAttrs : public ir::AttrsNode<LegacyRandomPdfGammaAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfGammaAttrs, "mxnet.v3.attrs.LegacyRandomPdfGammaAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_exponential
class LegacyRandomPdfExponentialAttrs : public ir::AttrsNode<LegacyRandomPdfExponentialAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfExponentialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfExponentialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_poisson
class LegacyRandomPdfPoissonAttrs : public ir::AttrsNode<LegacyRandomPdfPoissonAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfPoissonAttrs, "mxnet.v3.attrs.LegacyRandomPdfPoissonAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_negative_binomial
class LegacyRandomPdfNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomPdfNegativeBinomialAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_generalized_negative_binomial
class LegacyRandomPdfGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomPdfGeneralizedNegativeBinomialAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _random_pdf_dirichlet
class LegacyRandomPdfDirichletAttrs : public ir::AttrsNode<LegacyRandomPdfDirichletAttrs> {
 public:
  bool is_log;

  MX_V3_DECLARE_ATTRS(LegacyRandomPdfDirichletAttrs,
                      "mxnet.v3.attrs.LegacyRandomPdfDirichletAttrs") {
    MX_V3_ATTR_FIELD(is_log);
  }
};
// _sample_multinomial
class LegacySampleMultinomialAttrs : public ir::AttrsNode<LegacySampleMultinomialAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  bool get_prob;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacySampleMultinomialAttrs, "mxnet.v3.attrs.LegacySampleMultinomialAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(get_prob);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_uniform
class LegacyRandomUniformAttrs : public ir::AttrsNode<LegacyRandomUniformAttrs> {
 public:
  double low;
  double high;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomUniformAttrs, "mxnet.v3.attrs.LegacyRandomUniformAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_normal
class LegacyRandomNormalAttrs : public ir::AttrsNode<LegacyRandomNormalAttrs> {
 public:
  double loc;
  double scale;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomNormalAttrs, "mxnet.v3.attrs.LegacyRandomNormalAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_gamma
class LegacyRandomGammaAttrs : public ir::AttrsNode<LegacyRandomGammaAttrs> {
 public:
  double alpha;
  double beta;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomGammaAttrs, "mxnet.v3.attrs.LegacyRandomGammaAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_exponential
class LegacyRandomExponentialAttrs : public ir::AttrsNode<LegacyRandomExponentialAttrs> {
 public:
  double lam;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomExponentialAttrs, "mxnet.v3.attrs.LegacyRandomExponentialAttrs") {
    MX_V3_ATTR_FIELD(lam);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_poisson
class LegacyRandomPoissonAttrs : public ir::AttrsNode<LegacyRandomPoissonAttrs> {
 public:
  double lam;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomPoissonAttrs, "mxnet.v3.attrs.LegacyRandomPoissonAttrs") {
    MX_V3_ATTR_FIELD(lam);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_negative_binomial
class LegacyRandomNegativeBinomialAttrs : public ir::AttrsNode<LegacyRandomNegativeBinomialAttrs> {
 public:
  int k;
  double p;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(p);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_generalized_negative_binomial
class LegacyRandomGeneralizedNegativeBinomialAttrs
    : public ir::AttrsNode<LegacyRandomGeneralizedNegativeBinomialAttrs> {
 public:
  double mu;
  double alpha;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomGeneralizedNegativeBinomialAttrs,
                      "mxnet.v3.attrs.LegacyRandomGeneralizedNegativeBinomialAttrs") {
    MX_V3_ATTR_FIELD(mu);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_randint
class LegacyRandomRandintAttrs : public ir::AttrsNode<LegacyRandomRandintAttrs> {
 public:
  int64_t low;
  int64_t high;
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyRandomRandintAttrs, "mxnet.v3.attrs.LegacyRandomRandintAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _random_uniform_like
class LegacyRandomUniformLikeAttrs : public ir::AttrsNode<LegacyRandomUniformLikeAttrs> {
 public:
  double low;
  double high;

  MX_V3_DECLARE_ATTRS(LegacyRandomUniformLikeAttrs, "mxnet.v3.attrs.LegacyRandomUniformLikeAttrs") {
    MX_V3_ATTR_FIELD(low);
    MX_V3_ATTR_FIELD(high);
  }
};
// _random_normal_like
class LegacyRandomNormalLikeAttrs : public ir::AttrsNode<LegacyRandomNormalLikeAttrs> {
 public:
  double loc;
  double scale;

  MX_V3_DECLARE_ATTRS(LegacyRandomNormalLikeAttrs, "mxnet.v3.attrs.LegacyRandomNormalLikeAttrs") {
    MX_V3_ATTR_FIELD(loc);
    MX_V3_ATTR_FIELD(scale);
  }
};
// _random_gamma_like
class LegacyRandomGammaLikeAttrs : public ir::AttrsNode<LegacyRandomGammaLikeAttrs> {
 public:
  double alpha;
  double beta;

  MX_V3_DECLARE_ATTRS(LegacyRandomGammaLikeAttrs, "mxnet.v3.attrs.LegacyRandomGammaLikeAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
  }
};
// _random_exponential_like
class LegacyRandomExponentialLikeAttrs : public ir::AttrsNode<LegacyRandomExponentialLikeAttrs> {
 public:
  double lam;

  MX_V3_DECLARE_ATTRS(LegacyRandomExponentialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomExponentialLikeAttrs") {
    MX_V3_ATTR_FIELD(lam);
  }
};
// _random_poisson_like
class LegacyRandomPoissonLikeAttrs : public ir::AttrsNode<LegacyRandomPoissonLikeAttrs> {
 public:
  double lam;

  MX_V3_DECLARE_ATTRS(LegacyRandomPoissonLikeAttrs, "mxnet.v3.attrs.LegacyRandomPoissonLikeAttrs") {
    MX_V3_ATTR_FIELD(lam);
  }
};
// _random_negative_binomial_like
class LegacyRandomNegativeBinomialLikeAttrs
    : public ir::AttrsNode<LegacyRandomNegativeBinomialLikeAttrs> {
 public:
  int k;
  double p;

  MX_V3_DECLARE_ATTRS(LegacyRandomNegativeBinomialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomNegativeBinomialLikeAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(p);
  }
};
// _random_generalized_negative_binomial_like
class LegacyRandomGeneralizedNegativeBinomialLikeAttrs
    : public ir::AttrsNode<LegacyRandomGeneralizedNegativeBinomialLikeAttrs> {
 public:
  double mu;
  double alpha;

  MX_V3_DECLARE_ATTRS(LegacyRandomGeneralizedNegativeBinomialLikeAttrs,
                      "mxnet.v3.attrs.LegacyRandomGeneralizedNegativeBinomialLikeAttrs") {
    MX_V3_ATTR_FIELD(mu);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// _shuffle
using LegacyShuffleAttrs = ir::Attrs;
// _sample_unique_zipfian
class LegacySampleUniqueZipfianAttrs : public ir::AttrsNode<LegacySampleUniqueZipfianAttrs> {
 public:
  int range_max;
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacySampleUniqueZipfianAttrs,
                      "mxnet.v3.attrs.LegacySampleUniqueZipfianAttrs") {
    MX_V3_ATTR_FIELD(range_max);
    MX_V3_ATTR_FIELD(shape);
  }
};
// LinearRegressionOutput
class LegacyLinearRegressionOutputAttrs : public ir::AttrsNode<LegacyLinearRegressionOutputAttrs> {
 public:
  double grad_scale;

  MX_V3_DECLARE_ATTRS(LegacyLinearRegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyLinearRegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// MAERegressionOutput
class LegacyMAERegressionOutputAttrs : public ir::AttrsNode<LegacyMAERegressionOutputAttrs> {
 public:
  double grad_scale;

  MX_V3_DECLARE_ATTRS(LegacyMAERegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyMAERegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// LogisticRegressionOutput
class LegacyLogisticRegressionOutputAttrs
    : public ir::AttrsNode<LegacyLogisticRegressionOutputAttrs> {
 public:
  double grad_scale;

  MX_V3_DECLARE_ATTRS(LegacyLogisticRegressionOutputAttrs,
                      "mxnet.v3.attrs.LegacyLogisticRegressionOutputAttrs") {
    MX_V3_ATTR_FIELD(grad_scale);
  }
};
// RNN
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
// ROIPooling
class LegacyROIPoolingAttrs : public ir::AttrsNode<LegacyROIPoolingAttrs> {
 public:
  ir::Array<ir::Integer> pooled_size;
  double spatial_scale;

  MX_V3_DECLARE_ATTRS(LegacyROIPoolingAttrs, "mxnet.v3.attrs.LegacyROIPoolingAttrs") {
    MX_V3_ATTR_FIELD(pooled_size);
    MX_V3_ATTR_FIELD(spatial_scale);
  }
};
// SequenceMask
class LegacySequenceMaskAttrs : public ir::AttrsNode<LegacySequenceMaskAttrs> {
 public:
  bool use_sequence_length;
  double value;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacySequenceMaskAttrs, "mxnet.v3.attrs.LegacySequenceMaskAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(value);
    MX_V3_ATTR_FIELD(axis);
  }
};
// SliceChannel
class LegacySliceChannelAttrs : public ir::AttrsNode<LegacySliceChannelAttrs> {
 public:
  int num_outputs;
  int axis;
  bool squeeze_axis;

  MX_V3_DECLARE_ATTRS(LegacySliceChannelAttrs, "mxnet.v3.attrs.LegacySliceChannelAttrs") {
    MX_V3_ATTR_FIELD(num_outputs);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(squeeze_axis);
  }
};
// SoftmaxOutput
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
// _sg_mkldnn_conv
using LegacySgMkldnnConvAttrs = ir::Attrs;
// _sg_mkldnn_fully_connected
using LegacySgMkldnnFullyConnectedAttrs = ir::Attrs;
// SwapAxis
class LegacySwapAxisAttrs : public ir::AttrsNode<LegacySwapAxisAttrs> {
 public:
  int dim1;
  int dim2;

  MX_V3_DECLARE_ATTRS(LegacySwapAxisAttrs, "mxnet.v3.attrs.LegacySwapAxisAttrs") {
    MX_V3_ATTR_FIELD(dim1);
    MX_V3_ATTR_FIELD(dim2);
  }
};
// max
class LegacyMaxAttrs : public ir::AttrsNode<LegacyMaxAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyMaxAttrs, "mxnet.v3.attrs.LegacyMaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// min
class LegacyMinAttrs : public ir::AttrsNode<LegacyMinAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyMinAttrs, "mxnet.v3.attrs.LegacyMinAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// norm
class LegacyNormAttrs : public ir::AttrsNode<LegacyNormAttrs> {
 public:
  int ord;
  ir::Array<ir::Integer> axis;
  std::string out_dtype;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyNormAttrs, "mxnet.v3.attrs.LegacyNormAttrs") {
    MX_V3_ATTR_FIELD(ord);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(out_dtype);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// argmax
class LegacyArgmaxAttrs : public ir::AttrsNode<LegacyArgmaxAttrs> {
 public:
  int axis;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyArgmaxAttrs, "mxnet.v3.attrs.LegacyArgmaxAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// argmin
class LegacyArgminAttrs : public ir::AttrsNode<LegacyArgminAttrs> {
 public:
  int axis;
  bool keepdims;

  MX_V3_DECLARE_ATTRS(LegacyArgminAttrs, "mxnet.v3.attrs.LegacyArgminAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
  }
};
// argmax_channel
using LegacyArgmaxChannelAttrs = ir::Attrs;
// pick
class LegacyPickAttrs : public ir::AttrsNode<LegacyPickAttrs> {
 public:
  int axis;
  bool keepdims;
  std::string mode;

  MX_V3_DECLARE_ATTRS(LegacyPickAttrs, "mxnet.v3.attrs.LegacyPickAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(mode);
  }
};
// broadcast_axis
class LegacyBroadcastAxisAttrs : public ir::AttrsNode<LegacyBroadcastAxisAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  ir::Array<ir::Integer> size;

  MX_V3_DECLARE_ATTRS(LegacyBroadcastAxisAttrs, "mxnet.v3.attrs.LegacyBroadcastAxisAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(size);
  }
};
// broadcast_to
class LegacyBroadcastToAttrs : public ir::AttrsNode<LegacyBroadcastToAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyBroadcastToAttrs, "mxnet.v3.attrs.LegacyBroadcastToAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// broadcast_like
class LegacyBroadcastLikeAttrs : public ir::AttrsNode<LegacyBroadcastLikeAttrs> {
 public:
  ir::Array<ir::Integer> lhs_axes;
  ir::Array<ir::Integer> rhs_axes;

  MX_V3_DECLARE_ATTRS(LegacyBroadcastLikeAttrs, "mxnet.v3.attrs.LegacyBroadcastLikeAttrs") {
    MX_V3_ATTR_FIELD(lhs_axes);
    MX_V3_ATTR_FIELD(rhs_axes);
  }
};
// prod
class LegacyProdAttrs : public ir::AttrsNode<LegacyProdAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyProdAttrs, "mxnet.v3.attrs.LegacyProdAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// nanprod
class LegacyNanprodAttrs : public ir::AttrsNode<LegacyNanprodAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyNanprodAttrs, "mxnet.v3.attrs.LegacyNanprodAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// sum
class LegacySumAttrs : public ir::AttrsNode<LegacySumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacySumAttrs, "mxnet.v3.attrs.LegacySumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// mean
class LegacyMeanAttrs : public ir::AttrsNode<LegacyMeanAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyMeanAttrs, "mxnet.v3.attrs.LegacyMeanAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// nansum
class LegacyNansumAttrs : public ir::AttrsNode<LegacyNansumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacyNansumAttrs, "mxnet.v3.attrs.LegacyNansumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// where
using LegacyWhereAttrs = ir::Attrs;
// diag
class LegacyDiagAttrs : public ir::AttrsNode<LegacyDiagAttrs> {
 public:
  int k;
  int axis1;
  int axis2;

  MX_V3_DECLARE_ATTRS(LegacyDiagAttrs, "mxnet.v3.attrs.LegacyDiagAttrs") {
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(axis1);
    MX_V3_ATTR_FIELD(axis2);
  }
};
// dot
class LegacyDotAttrs : public ir::AttrsNode<LegacyDotAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  std::string forward_stype;

  MX_V3_DECLARE_ATTRS(LegacyDotAttrs, "mxnet.v3.attrs.LegacyDotAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(forward_stype);
  }
};
// batch_dot
class LegacyBatchDotAttrs : public ir::AttrsNode<LegacyBatchDotAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  std::string forward_stype;

  MX_V3_DECLARE_ATTRS(LegacyBatchDotAttrs, "mxnet.v3.attrs.LegacyBatchDotAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(forward_stype);
  }
};
// broadcast_add
using LegacyBroadcastAddAttrs = ir::Attrs;
// broadcast_sub
using LegacyBroadcastSubAttrs = ir::Attrs;
// broadcast_mul
using LegacyBroadcastMulAttrs = ir::Attrs;
// broadcast_div
using LegacyBroadcastDivAttrs = ir::Attrs;
// broadcast_mod
using LegacyBroadcastModAttrs = ir::Attrs;
// broadcast_power
using LegacyBroadcastPowerAttrs = ir::Attrs;
// broadcast_maximum
using LegacyBroadcastMaximumAttrs = ir::Attrs;
// broadcast_minimum
using LegacyBroadcastMinimumAttrs = ir::Attrs;
// broadcast_hypot
using LegacyBroadcastHypotAttrs = ir::Attrs;
// broadcast_equal
using LegacyBroadcastEqualAttrs = ir::Attrs;
// broadcast_not_equal
using LegacyBroadcastNotEqualAttrs = ir::Attrs;
// broadcast_greater
using LegacyBroadcastGreaterAttrs = ir::Attrs;
// broadcast_greater_equal
using LegacyBroadcastGreaterEqualAttrs = ir::Attrs;
// broadcast_lesser
using LegacyBroadcastLesserAttrs = ir::Attrs;
// broadcast_lesser_equal
using LegacyBroadcastLesserEqualAttrs = ir::Attrs;
// broadcast_logical_and
using LegacyBroadcastLogicalAndAttrs = ir::Attrs;
// broadcast_logical_or
using LegacyBroadcastLogicalOrAttrs = ir::Attrs;
// broadcast_logical_xor
using LegacyBroadcastLogicalXorAttrs = ir::Attrs;
// elemwise_add
using LegacyElemwiseAddAttrs = ir::Attrs;
// _grad_add
using LegacyGradAddAttrs = ir::Attrs;
// elemwise_sub
using LegacyElemwiseSubAttrs = ir::Attrs;
// elemwise_mul
using LegacyElemwiseMulAttrs = ir::Attrs;
// elemwise_div
using LegacyElemwiseDivAttrs = ir::Attrs;
// _mod
using LegacyModAttrs = ir::Attrs;
// _power
using LegacyPowerAttrs = ir::Attrs;
// _maximum
using LegacyMaximumAttrs = ir::Attrs;
// _minimum
using LegacyMinimumAttrs = ir::Attrs;
// _hypot
using LegacyHypotAttrs = ir::Attrs;
// _equal
using LegacyEqualAttrs = ir::Attrs;
// _not_equal
using LegacyNotEqualAttrs = ir::Attrs;
// _greater
using LegacyGreaterAttrs = ir::Attrs;
// _greater_equal
using LegacyGreaterEqualAttrs = ir::Attrs;
// _lesser
using LegacyLesserAttrs = ir::Attrs;
// _lesser_equal
using LegacyLesserEqualAttrs = ir::Attrs;
// _logical_and
using LegacyLogicalAndAttrs = ir::Attrs;
// _logical_or
using LegacyLogicalOrAttrs = ir::Attrs;
// _logical_xor
using LegacyLogicalXorAttrs = ir::Attrs;
// _plus_scalar
class LegacyPlusScalarAttrs : public ir::AttrsNode<LegacyPlusScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyPlusScalarAttrs, "mxnet.v3.attrs.LegacyPlusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _minus_scalar
class LegacyMinusScalarAttrs : public ir::AttrsNode<LegacyMinusScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyMinusScalarAttrs, "mxnet.v3.attrs.LegacyMinusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _rminus_scalar
class LegacyRminusScalarAttrs : public ir::AttrsNode<LegacyRminusScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyRminusScalarAttrs, "mxnet.v3.attrs.LegacyRminusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _mul_scalar
class LegacyMulScalarAttrs : public ir::AttrsNode<LegacyMulScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyMulScalarAttrs, "mxnet.v3.attrs.LegacyMulScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _div_scalar
class LegacyDivScalarAttrs : public ir::AttrsNode<LegacyDivScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyDivScalarAttrs, "mxnet.v3.attrs.LegacyDivScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _rdiv_scalar
class LegacyRdivScalarAttrs : public ir::AttrsNode<LegacyRdivScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyRdivScalarAttrs, "mxnet.v3.attrs.LegacyRdivScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _mod_scalar
class LegacyModScalarAttrs : public ir::AttrsNode<LegacyModScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyModScalarAttrs, "mxnet.v3.attrs.LegacyModScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _rmod_scalar
class LegacyRmodScalarAttrs : public ir::AttrsNode<LegacyRmodScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyRmodScalarAttrs, "mxnet.v3.attrs.LegacyRmodScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _maximum_scalar
class LegacyMaximumScalarAttrs : public ir::AttrsNode<LegacyMaximumScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyMaximumScalarAttrs, "mxnet.v3.attrs.LegacyMaximumScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _minimum_scalar
class LegacyMinimumScalarAttrs : public ir::AttrsNode<LegacyMinimumScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyMinimumScalarAttrs, "mxnet.v3.attrs.LegacyMinimumScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _power_scalar
class LegacyPowerScalarAttrs : public ir::AttrsNode<LegacyPowerScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyPowerScalarAttrs, "mxnet.v3.attrs.LegacyPowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _rpower_scalar
class LegacyRpowerScalarAttrs : public ir::AttrsNode<LegacyRpowerScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyRpowerScalarAttrs, "mxnet.v3.attrs.LegacyRpowerScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _hypot_scalar
class LegacyHypotScalarAttrs : public ir::AttrsNode<LegacyHypotScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyHypotScalarAttrs, "mxnet.v3.attrs.LegacyHypotScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// smooth_l1
class LegacySmoothL1Attrs : public ir::AttrsNode<LegacySmoothL1Attrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacySmoothL1Attrs, "mxnet.v3.attrs.LegacySmoothL1Attrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _equal_scalar
class LegacyEqualScalarAttrs : public ir::AttrsNode<LegacyEqualScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyEqualScalarAttrs, "mxnet.v3.attrs.LegacyEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _not_equal_scalar
class LegacyNotEqualScalarAttrs : public ir::AttrsNode<LegacyNotEqualScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyNotEqualScalarAttrs, "mxnet.v3.attrs.LegacyNotEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _greater_scalar
class LegacyGreaterScalarAttrs : public ir::AttrsNode<LegacyGreaterScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyGreaterScalarAttrs, "mxnet.v3.attrs.LegacyGreaterScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _greater_equal_scalar
class LegacyGreaterEqualScalarAttrs : public ir::AttrsNode<LegacyGreaterEqualScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyGreaterEqualScalarAttrs,
                      "mxnet.v3.attrs.LegacyGreaterEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _lesser_scalar
class LegacyLesserScalarAttrs : public ir::AttrsNode<LegacyLesserScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyLesserScalarAttrs, "mxnet.v3.attrs.LegacyLesserScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _lesser_equal_scalar
class LegacyLesserEqualScalarAttrs : public ir::AttrsNode<LegacyLesserEqualScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyLesserEqualScalarAttrs, "mxnet.v3.attrs.LegacyLesserEqualScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _logical_and_scalar
class LegacyLogicalAndScalarAttrs : public ir::AttrsNode<LegacyLogicalAndScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyLogicalAndScalarAttrs, "mxnet.v3.attrs.LegacyLogicalAndScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _logical_or_scalar
class LegacyLogicalOrScalarAttrs : public ir::AttrsNode<LegacyLogicalOrScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyLogicalOrScalarAttrs, "mxnet.v3.attrs.LegacyLogicalOrScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _logical_xor_scalar
class LegacyLogicalXorScalarAttrs : public ir::AttrsNode<LegacyLogicalXorScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyLogicalXorScalarAttrs, "mxnet.v3.attrs.LegacyLogicalXorScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _scatter_elemwise_div
using LegacyScatterElemwiseDivAttrs = ir::Attrs;
// _scatter_plus_scalar
class LegacyScatterPlusScalarAttrs : public ir::AttrsNode<LegacyScatterPlusScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyScatterPlusScalarAttrs, "mxnet.v3.attrs.LegacyScatterPlusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// _scatter_minus_scalar
class LegacyScatterMinusScalarAttrs : public ir::AttrsNode<LegacyScatterMinusScalarAttrs> {
 public:
  double scalar;

  MX_V3_DECLARE_ATTRS(LegacyScatterMinusScalarAttrs,
                      "mxnet.v3.attrs.LegacyScatterMinusScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
  }
};
// relu
using LegacyReluAttrs = ir::Attrs;
// sigmoid
using LegacySigmoidAttrs = ir::Attrs;
// hard_sigmoid
class LegacyHardSigmoidAttrs : public ir::AttrsNode<LegacyHardSigmoidAttrs> {
 public:
  double alpha;
  double beta;

  MX_V3_DECLARE_ATTRS(LegacyHardSigmoidAttrs, "mxnet.v3.attrs.LegacyHardSigmoidAttrs") {
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
  }
};
// softsign
using LegacySoftsignAttrs = ir::Attrs;
// _copy
using LegacyCopyAttrs = ir::Attrs;
// make_loss
using LegacyMakeLossAttrs = ir::Attrs;
// _identity_with_attr_like_rhs
using LegacyIdentityWithAttrLikeRhsAttrs = ir::Attrs;
// reshape_like
class LegacyReshapeLikeAttrs : public ir::AttrsNode<LegacyReshapeLikeAttrs> {
 public:
  int lhs_begin;
  int lhs_end;
  int rhs_begin;
  int rhs_end;

  MX_V3_DECLARE_ATTRS(LegacyReshapeLikeAttrs, "mxnet.v3.attrs.LegacyReshapeLikeAttrs") {
    MX_V3_ATTR_FIELD(lhs_begin);
    MX_V3_ATTR_FIELD(lhs_end);
    MX_V3_ATTR_FIELD(rhs_begin);
    MX_V3_ATTR_FIELD(rhs_end);
  }
};
// shape_array
using LegacyShapeArrayAttrs = ir::Attrs;
// size_array
using LegacySizeArrayAttrs = ir::Attrs;
// Cast
class LegacyCastAttrs : public ir::AttrsNode<LegacyCastAttrs> {
 public:
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyCastAttrs, "mxnet.v3.attrs.LegacyCastAttrs") {
    MX_V3_ATTR_FIELD(dtype);
  }
};
// negative
using LegacyNegativeAttrs = ir::Attrs;
// abs
using LegacyAbsAttrs = ir::Attrs;
// sign
using LegacySignAttrs = ir::Attrs;
// round
using LegacyRoundAttrs = ir::Attrs;
// rint
using LegacyRintAttrs = ir::Attrs;
// ceil
using LegacyCeilAttrs = ir::Attrs;
// floor
using LegacyFloorAttrs = ir::Attrs;
// trunc
using LegacyTruncAttrs = ir::Attrs;
// fix
using LegacyFixAttrs = ir::Attrs;
// erf
using LegacyErfAttrs = ir::Attrs;
// erfinv
using LegacyErfinvAttrs = ir::Attrs;
// gamma
using LegacyGammaAttrs = ir::Attrs;
// gammaln
using LegacyGammalnAttrs = ir::Attrs;
// logical_not
using LegacyLogicalNotAttrs = ir::Attrs;
// exp
using LegacyExpAttrs = ir::Attrs;
// log
using LegacyLogAttrs = ir::Attrs;
// log10
using LegacyLog10Attrs = ir::Attrs;
// log2
using LegacyLog2Attrs = ir::Attrs;
// log1p
using LegacyLog1pAttrs = ir::Attrs;
// expm1
using LegacyExpm1Attrs = ir::Attrs;
// reciprocal
using LegacyReciprocalAttrs = ir::Attrs;
// square
using LegacySquareAttrs = ir::Attrs;
// sqrt
using LegacySqrtAttrs = ir::Attrs;
// rsqrt
using LegacyRsqrtAttrs = ir::Attrs;
// cbrt
using LegacyCbrtAttrs = ir::Attrs;
// rcbrt
using LegacyRcbrtAttrs = ir::Attrs;
// sin
using LegacySinAttrs = ir::Attrs;
// cos
using LegacyCosAttrs = ir::Attrs;
// tan
using LegacyTanAttrs = ir::Attrs;
// arcsin
using LegacyArcsinAttrs = ir::Attrs;
// arccos
using LegacyArccosAttrs = ir::Attrs;
// arctan
using LegacyArctanAttrs = ir::Attrs;
// degrees
using LegacyDegreesAttrs = ir::Attrs;
// radians
using LegacyRadiansAttrs = ir::Attrs;
// sinh
using LegacySinhAttrs = ir::Attrs;
// cosh
using LegacyCoshAttrs = ir::Attrs;
// tanh
using LegacyTanhAttrs = ir::Attrs;
// arcsinh
using LegacyArcsinhAttrs = ir::Attrs;
// arccosh
using LegacyArccoshAttrs = ir::Attrs;
// arctanh
using LegacyArctanhAttrs = ir::Attrs;
// Embedding
class LegacyEmbeddingAttrs : public ir::AttrsNode<LegacyEmbeddingAttrs> {
 public:
  int input_dim;
  int output_dim;
  std::string dtype;
  bool sparse_grad;

  MX_V3_DECLARE_ATTRS(LegacyEmbeddingAttrs, "mxnet.v3.attrs.LegacyEmbeddingAttrs") {
    MX_V3_ATTR_FIELD(input_dim);
    MX_V3_ATTR_FIELD(output_dim);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(sparse_grad);
  }
};
// take
class LegacyTakeAttrs : public ir::AttrsNode<LegacyTakeAttrs> {
 public:
  int axis;
  std::string mode;

  MX_V3_DECLARE_ATTRS(LegacyTakeAttrs, "mxnet.v3.attrs.LegacyTakeAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(mode);
  }
};
// batch_take
using LegacyBatchTakeAttrs = ir::Attrs;
// one_hot
class LegacyOneHotAttrs : public ir::AttrsNode<LegacyOneHotAttrs> {
 public:
  int depth;
  double on_value;
  double off_value;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyOneHotAttrs, "mxnet.v3.attrs.LegacyOneHotAttrs") {
    MX_V3_ATTR_FIELD(depth);
    MX_V3_ATTR_FIELD(on_value);
    MX_V3_ATTR_FIELD(off_value);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// gather_nd
using LegacyGatherNdAttrs = ir::Attrs;
// scatter_nd
class LegacyScatterNdAttrs : public ir::AttrsNode<LegacyScatterNdAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyScatterNdAttrs, "mxnet.v3.attrs.LegacyScatterNdAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// _scatter_set_nd
class LegacyScatterSetNdAttrs : public ir::AttrsNode<LegacyScatterSetNdAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyScatterSetNdAttrs, "mxnet.v3.attrs.LegacyScatterSetNdAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// _zeros_without_dtype
class LegacyZerosWithoutDtypeAttrs : public ir::AttrsNode<LegacyZerosWithoutDtypeAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  int dtype;

  MX_V3_DECLARE_ATTRS(LegacyZerosWithoutDtypeAttrs, "mxnet.v3.attrs.LegacyZerosWithoutDtypeAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _zeros
class LegacyZerosAttrs : public ir::AttrsNode<LegacyZerosAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyZerosAttrs, "mxnet.v3.attrs.LegacyZerosAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _eye
class LegacyEyeAttrs : public ir::AttrsNode<LegacyEyeAttrs> {
 public:
  int64_t N;
  int64_t M;
  int64_t k;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyEyeAttrs, "mxnet.v3.attrs.LegacyEyeAttrs") {
    MX_V3_ATTR_FIELD(N);
    MX_V3_ATTR_FIELD(M);
    MX_V3_ATTR_FIELD(k);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _ones
class LegacyOnesAttrs : public ir::AttrsNode<LegacyOnesAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyOnesAttrs, "mxnet.v3.attrs.LegacyOnesAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _full
class LegacyFullAttrs : public ir::AttrsNode<LegacyFullAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  std::string ctx;
  std::string dtype;
  double value;

  MX_V3_DECLARE_ATTRS(LegacyFullAttrs, "mxnet.v3.attrs.LegacyFullAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(ctx);
    MX_V3_ATTR_FIELD(dtype);
    MX_V3_ATTR_FIELD(value);
  }
};
// _arange
class LegacyArangeAttrs : public ir::AttrsNode<LegacyArangeAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;

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
// _linspace
class LegacyLinspaceAttrs : public ir::AttrsNode<LegacyLinspaceAttrs> {
 public:
  double start;
  double stop;
  double step;
  int repeat;
  bool infer_range;
  std::string ctx;
  std::string dtype;

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
// zeros_like
using LegacyZerosLikeAttrs = ir::Attrs;
// ones_like
using LegacyOnesLikeAttrs = ir::Attrs;
// _linalg_gemm
class LegacyLinalgGemmAttrs : public ir::AttrsNode<LegacyLinalgGemmAttrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
  double beta;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacyLinalgGemmAttrs, "mxnet.v3.attrs.LegacyLinalgGemmAttrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(beta);
    MX_V3_ATTR_FIELD(axis);
  }
};
// _linalg_gemm2
class LegacyLinalgGemm2Attrs : public ir::AttrsNode<LegacyLinalgGemm2Attrs> {
 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacyLinalgGemm2Attrs, "mxnet.v3.attrs.LegacyLinalgGemm2Attrs") {
    MX_V3_ATTR_FIELD(transpose_a);
    MX_V3_ATTR_FIELD(transpose_b);
    MX_V3_ATTR_FIELD(alpha);
    MX_V3_ATTR_FIELD(axis);
  }
};
// _linalg_potrf
using LegacyLinalgPotrfAttrs = ir::Attrs;
// _linalg_potri
using LegacyLinalgPotriAttrs = ir::Attrs;
// _linalg_trmm
class LegacyLinalgTrmmAttrs : public ir::AttrsNode<LegacyLinalgTrmmAttrs> {
 public:
  bool transpose;
  bool rightside;
  bool lower;
  double alpha;

  MX_V3_DECLARE_ATTRS(LegacyLinalgTrmmAttrs, "mxnet.v3.attrs.LegacyLinalgTrmmAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(rightside);
    MX_V3_ATTR_FIELD(lower);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// _linalg_trsm
class LegacyLinalgTrsmAttrs : public ir::AttrsNode<LegacyLinalgTrsmAttrs> {
 public:
  bool transpose;
  bool rightside;
  bool lower;
  double alpha;

  MX_V3_DECLARE_ATTRS(LegacyLinalgTrsmAttrs, "mxnet.v3.attrs.LegacyLinalgTrsmAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(rightside);
    MX_V3_ATTR_FIELD(lower);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// _linalg_sumlogdiag
using LegacyLinalgSumlogdiagAttrs = ir::Attrs;
// _linalg_extractdiag
class LegacyLinalgExtractdiagAttrs : public ir::AttrsNode<LegacyLinalgExtractdiagAttrs> {
 public:
  int offset;

  MX_V3_DECLARE_ATTRS(LegacyLinalgExtractdiagAttrs, "mxnet.v3.attrs.LegacyLinalgExtractdiagAttrs") {
    MX_V3_ATTR_FIELD(offset);
  }
};
// _linalg_makediag
class LegacyLinalgMakediagAttrs : public ir::AttrsNode<LegacyLinalgMakediagAttrs> {
 public:
  int offset;

  MX_V3_DECLARE_ATTRS(LegacyLinalgMakediagAttrs, "mxnet.v3.attrs.LegacyLinalgMakediagAttrs") {
    MX_V3_ATTR_FIELD(offset);
  }
};
// _linalg_extracttrian
class LegacyLinalgExtracttrianAttrs : public ir::AttrsNode<LegacyLinalgExtracttrianAttrs> {
 public:
  int offset;
  bool lower;

  MX_V3_DECLARE_ATTRS(LegacyLinalgExtracttrianAttrs,
                      "mxnet.v3.attrs.LegacyLinalgExtracttrianAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(lower);
  }
};
// _linalg_maketrian
class LegacyLinalgMaketrianAttrs : public ir::AttrsNode<LegacyLinalgMaketrianAttrs> {
 public:
  int offset;
  bool lower;

  MX_V3_DECLARE_ATTRS(LegacyLinalgMaketrianAttrs, "mxnet.v3.attrs.LegacyLinalgMaketrianAttrs") {
    MX_V3_ATTR_FIELD(offset);
    MX_V3_ATTR_FIELD(lower);
  }
};
// _linalg_syrk
class LegacyLinalgSyrkAttrs : public ir::AttrsNode<LegacyLinalgSyrkAttrs> {
 public:
  bool transpose;
  double alpha;

  MX_V3_DECLARE_ATTRS(LegacyLinalgSyrkAttrs, "mxnet.v3.attrs.LegacyLinalgSyrkAttrs") {
    MX_V3_ATTR_FIELD(transpose);
    MX_V3_ATTR_FIELD(alpha);
  }
};
// _linalg_gelqf
using LegacyLinalgGelqfAttrs = ir::Attrs;
// _linalg_syevd
using LegacyLinalgSyevdAttrs = ir::Attrs;
// _linalg_inverse
using LegacyLinalgInverseAttrs = ir::Attrs;
// _linalg_det
using LegacyLinalgDetAttrs = ir::Attrs;
// _linalg_slogdet
using LegacyLinalgSlogdetAttrs = ir::Attrs;
// Reshape
class LegacyReshapeAttrs : public ir::AttrsNode<LegacyReshapeAttrs> {
 public:
  ir::Array<ir::Integer> shape;
  bool reverse;
  ir::Array<ir::Integer> target_shape;
  bool keep_highest;

  MX_V3_DECLARE_ATTRS(LegacyReshapeAttrs, "mxnet.v3.attrs.LegacyReshapeAttrs") {
    MX_V3_ATTR_FIELD(shape);
    MX_V3_ATTR_FIELD(reverse);
    MX_V3_ATTR_FIELD(target_shape);
    MX_V3_ATTR_FIELD(keep_highest);
  }
};
// transpose
class LegacyTransposeAttrs : public ir::AttrsNode<LegacyTransposeAttrs> {
 public:
  ir::Array<ir::Integer> axes;

  MX_V3_DECLARE_ATTRS(LegacyTransposeAttrs, "mxnet.v3.attrs.LegacyTransposeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// expand_dims
class LegacyExpandDimsAttrs : public ir::AttrsNode<LegacyExpandDimsAttrs> {
 public:
  int axis;

  MX_V3_DECLARE_ATTRS(LegacyExpandDimsAttrs, "mxnet.v3.attrs.LegacyExpandDimsAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// slice
class LegacySliceAttrs : public ir::AttrsNode<LegacySliceAttrs> {
 public:
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;

  MX_V3_DECLARE_ATTRS(LegacySliceAttrs, "mxnet.v3.attrs.LegacySliceAttrs") {
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// _slice_assign
class LegacySliceAssignAttrs : public ir::AttrsNode<LegacySliceAssignAttrs> {
 public:
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;

  MX_V3_DECLARE_ATTRS(LegacySliceAssignAttrs, "mxnet.v3.attrs.LegacySliceAssignAttrs") {
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// _slice_assign_scalar
class LegacySliceAssignScalarAttrs : public ir::AttrsNode<LegacySliceAssignScalarAttrs> {
 public:
  double scalar;
  ir::Array<ir::Integer> begin;
  ir::Array<ir::Integer> end;
  ir::Array<ir::Integer> step;

  MX_V3_DECLARE_ATTRS(LegacySliceAssignScalarAttrs, "mxnet.v3.attrs.LegacySliceAssignScalarAttrs") {
    MX_V3_ATTR_FIELD(scalar);
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
    MX_V3_ATTR_FIELD(step);
  }
};
// slice_axis
class LegacySliceAxisAttrs : public ir::AttrsNode<LegacySliceAxisAttrs> {
 public:
  int axis;
  int begin;
  int end;

  MX_V3_DECLARE_ATTRS(LegacySliceAxisAttrs, "mxnet.v3.attrs.LegacySliceAxisAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(begin);
    MX_V3_ATTR_FIELD(end);
  }
};
// slice_like
class LegacySliceLikeAttrs : public ir::AttrsNode<LegacySliceLikeAttrs> {
 public:
  ir::Array<ir::Integer> axes;

  MX_V3_DECLARE_ATTRS(LegacySliceLikeAttrs, "mxnet.v3.attrs.LegacySliceLikeAttrs") {
    MX_V3_ATTR_FIELD(axes);
  }
};
// clip
class LegacyClipAttrs : public ir::AttrsNode<LegacyClipAttrs> {
 public:
  double a_min;
  double a_max;

  MX_V3_DECLARE_ATTRS(LegacyClipAttrs, "mxnet.v3.attrs.LegacyClipAttrs") {
    MX_V3_ATTR_FIELD(a_min);
    MX_V3_ATTR_FIELD(a_max);
  }
};
// repeat
class LegacyRepeatAttrs : public ir::AttrsNode<LegacyRepeatAttrs> {
 public:
  int repeats;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacyRepeatAttrs, "mxnet.v3.attrs.LegacyRepeatAttrs") {
    MX_V3_ATTR_FIELD(repeats);
    MX_V3_ATTR_FIELD(axis);
  }
};
// tile
class LegacyTileAttrs : public ir::AttrsNode<LegacyTileAttrs> {
 public:
  ir::Array<ir::Integer> reps;

  MX_V3_DECLARE_ATTRS(LegacyTileAttrs, "mxnet.v3.attrs.LegacyTileAttrs") { MX_V3_ATTR_FIELD(reps); }
};
// reverse
class LegacyReverseAttrs : public ir::AttrsNode<LegacyReverseAttrs> {
 public:
  ir::Array<ir::Integer> axis;

  MX_V3_DECLARE_ATTRS(LegacyReverseAttrs, "mxnet.v3.attrs.LegacyReverseAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// squeeze
class LegacySqueezeAttrs : public ir::AttrsNode<LegacySqueezeAttrs> {
 public:
  ir::Array<ir::Integer> axis;

  MX_V3_DECLARE_ATTRS(LegacySqueezeAttrs, "mxnet.v3.attrs.LegacySqueezeAttrs") {
    MX_V3_ATTR_FIELD(axis);
  }
};
// depth_to_space
class LegacyDepthToSpaceAttrs : public ir::AttrsNode<LegacyDepthToSpaceAttrs> {
 public:
  int block_size;

  MX_V3_DECLARE_ATTRS(LegacyDepthToSpaceAttrs, "mxnet.v3.attrs.LegacyDepthToSpaceAttrs") {
    MX_V3_ATTR_FIELD(block_size);
  }
};
// space_to_depth
class LegacySpaceToDepthAttrs : public ir::AttrsNode<LegacySpaceToDepthAttrs> {
 public:
  int block_size;

  MX_V3_DECLARE_ATTRS(LegacySpaceToDepthAttrs, "mxnet.v3.attrs.LegacySpaceToDepthAttrs") {
    MX_V3_ATTR_FIELD(block_size);
  }
};
// _split_v2
class LegacySplitV2Attrs : public ir::AttrsNode<LegacySplitV2Attrs> {
 public:
  ir::Array<ir::Integer> indices;
  int axis;
  bool squeeze_axis;
  int sections;

  MX_V3_DECLARE_ATTRS(LegacySplitV2Attrs, "mxnet.v3.attrs.LegacySplitV2Attrs") {
    MX_V3_ATTR_FIELD(indices);
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(squeeze_axis);
    MX_V3_ATTR_FIELD(sections);
  }
};
// sort
class LegacySortAttrs : public ir::AttrsNode<LegacySortAttrs> {
 public:
  int axis;
  bool is_ascend;

  MX_V3_DECLARE_ATTRS(LegacySortAttrs, "mxnet.v3.attrs.LegacySortAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(is_ascend);
  }
};
// argsort
class LegacyArgsortAttrs : public ir::AttrsNode<LegacyArgsortAttrs> {
 public:
  int axis;
  bool is_ascend;
  std::string dtype;

  MX_V3_DECLARE_ATTRS(LegacyArgsortAttrs, "mxnet.v3.attrs.LegacyArgsortAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(is_ascend);
    MX_V3_ATTR_FIELD(dtype);
  }
};
// _ravel_multi_index
class LegacyRavelMultiIndexAttrs : public ir::AttrsNode<LegacyRavelMultiIndexAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyRavelMultiIndexAttrs, "mxnet.v3.attrs.LegacyRavelMultiIndexAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// _unravel_index
class LegacyUnravelIndexAttrs : public ir::AttrsNode<LegacyUnravelIndexAttrs> {
 public:
  ir::Array<ir::Integer> shape;

  MX_V3_DECLARE_ATTRS(LegacyUnravelIndexAttrs, "mxnet.v3.attrs.LegacyUnravelIndexAttrs") {
    MX_V3_ATTR_FIELD(shape);
  }
};
// _sparse_retain
using LegacySparseRetainAttrs = ir::Attrs;
// _square_sum
class LegacySquareSumAttrs : public ir::AttrsNode<LegacySquareSumAttrs> {
 public:
  ir::Array<ir::Integer> axis;
  bool keepdims;
  bool exclude;

  MX_V3_DECLARE_ATTRS(LegacySquareSumAttrs, "mxnet.v3.attrs.LegacySquareSumAttrs") {
    MX_V3_ATTR_FIELD(axis);
    MX_V3_ATTR_FIELD(keepdims);
    MX_V3_ATTR_FIELD(exclude);
  }
};
// BilinearSampler
class LegacyBilinearSamplerAttrs : public ir::AttrsNode<LegacyBilinearSamplerAttrs> {
 public:
  bool cudnn_off;

  MX_V3_DECLARE_ATTRS(LegacyBilinearSamplerAttrs, "mxnet.v3.attrs.LegacyBilinearSamplerAttrs") {
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// Correlation
class LegacyCorrelationAttrs : public ir::AttrsNode<LegacyCorrelationAttrs> {
 public:
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  int pad_size;
  bool is_multiply;

  MX_V3_DECLARE_ATTRS(LegacyCorrelationAttrs, "mxnet.v3.attrs.LegacyCorrelationAttrs") {
    MX_V3_ATTR_FIELD(kernel_size);
    MX_V3_ATTR_FIELD(max_displacement);
    MX_V3_ATTR_FIELD(stride1);
    MX_V3_ATTR_FIELD(stride2);
    MX_V3_ATTR_FIELD(pad_size);
    MX_V3_ATTR_FIELD(is_multiply);
  }
};
// InstanceNorm
class LegacyInstanceNormAttrs : public ir::AttrsNode<LegacyInstanceNormAttrs> {
 public:
  double eps;

  MX_V3_DECLARE_ATTRS(LegacyInstanceNormAttrs, "mxnet.v3.attrs.LegacyInstanceNormAttrs") {
    MX_V3_ATTR_FIELD(eps);
  }
};
// L2Normalization
class LegacyL2NormalizationAttrs : public ir::AttrsNode<LegacyL2NormalizationAttrs> {
 public:
  double eps;
  std::string mode;

  MX_V3_DECLARE_ATTRS(LegacyL2NormalizationAttrs, "mxnet.v3.attrs.LegacyL2NormalizationAttrs") {
    MX_V3_ATTR_FIELD(eps);
    MX_V3_ATTR_FIELD(mode);
  }
};
// SequenceLast
class LegacySequenceLastAttrs : public ir::AttrsNode<LegacySequenceLastAttrs> {
 public:
  bool use_sequence_length;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacySequenceLastAttrs, "mxnet.v3.attrs.LegacySequenceLastAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(axis);
  }
};
// SequenceReverse
class LegacySequenceReverseAttrs : public ir::AttrsNode<LegacySequenceReverseAttrs> {
 public:
  bool use_sequence_length;
  int axis;

  MX_V3_DECLARE_ATTRS(LegacySequenceReverseAttrs, "mxnet.v3.attrs.LegacySequenceReverseAttrs") {
    MX_V3_ATTR_FIELD(use_sequence_length);
    MX_V3_ATTR_FIELD(axis);
  }
};
// SpatialTransformer
class LegacySpatialTransformerAttrs : public ir::AttrsNode<LegacySpatialTransformerAttrs> {
 public:
  ir::Array<ir::Integer> target_shape;
  std::string transform_type;
  std::string sampler_type;
  bool cudnn_off;

  MX_V3_DECLARE_ATTRS(LegacySpatialTransformerAttrs,
                      "mxnet.v3.attrs.LegacySpatialTransformerAttrs") {
    MX_V3_ATTR_FIELD(target_shape);
    MX_V3_ATTR_FIELD(transform_type);
    MX_V3_ATTR_FIELD(sampler_type);
    MX_V3_ATTR_FIELD(cudnn_off);
  }
};
// _set_value
class LegacySetValueAttrs : public ir::AttrsNode<LegacySetValueAttrs> {
 public:
  double src;

  MX_V3_DECLARE_ATTRS(LegacySetValueAttrs, "mxnet.v3.attrs.LegacySetValueAttrs") {
    MX_V3_ATTR_FIELD(src);
  }
};
// _onehot_encode
using LegacyOnehotEncodeAttrs = ir::Attrs;
}  // namespace attrs
}  // namespace op
}  // namespace v3
}  // namespace mxnet
#endif

