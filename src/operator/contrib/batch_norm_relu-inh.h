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
 * Copyright (c) 2017 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu, Chris Olivier, Da Zheng
 */
#ifndef MXNET_OPERATOR_NN_BATCH_NORM_RELU_INL_H_
#define MXNET_OPERATOR_NN_BATCH_NORM_RELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
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


/*! \brief Parameters for BatchNoramWithReLU operator */
struct BatchNormWithReLUParam : public dmlc::Parameter<BatchNormWithReLUParam> {
  double eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int axis;
  bool cudnn_off;

  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset

  DMLC_DECLARE_PARAMETER(BatchNormWithReLUParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be no less than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output the mean and inverse std ");
    DMLC_DECLARE_FIELD(axis).set_default(mxnet::op::batchnormrelu::DEFAULT_AXIS)
    .describe("Specify which shape axis the channel is specified");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Do not select CUDNN operator, if available");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized batch norm op to calculate primitive scale."
              "Note: this calib_range is to calib bn output.");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized batch norm op to calculate primitive scale."
              "Note: this calib_range is to calib bn output.");
  }

  bool operator==(const BatchNormWithReLUParam &other) const {
    bool flag = this->eps == other.eps && this->momentum == other.momentum &&
                this->fix_gamma == other.fix_gamma &&
                this->use_global_stats == other.use_global_stats &&
                this->output_mean_var == other.output_mean_var && this->axis == other.axis &&
                this->cudnn_off == other.cudnn_off &&
                this->min_calib_range.has_value() == other.min_calib_range.has_value() &&
                this->max_calib_range.has_value() == other.max_calib_range.has_value();
    if (this->min_calib_range.has_value() && other.min_calib_range.has_value() &&
        this->max_calib_range.has_value() && other.max_calib_range.has_value()) {
      flag = flag && this->min_calib_range.value() == other.min_calib_range.value() &&
             this->max_calib_range.value() == other.max_calib_range.value();
    }
    return flag;
  }
};

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::BatchNormWithReLUParam> {
  size_t operator()(const mxnet::op::BatchNormWithReLUParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.momentum);
    ret = dmlc::HashCombine(ret, val.fix_gamma);
    ret = dmlc::HashCombine(ret, val.use_global_stats);
    ret = dmlc::HashCombine(ret, val.output_mean_var);
    ret = dmlc::HashCombine(ret, val.axis);
    return ret;
  }
};
}  // namespace std

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif  // MXNET_OPERATOR_NN_BATCH_NORM_RELU_INL_H_
