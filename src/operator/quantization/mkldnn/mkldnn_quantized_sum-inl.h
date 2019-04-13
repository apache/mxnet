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
 * \file mkldnn_quantized_sum-inl.h
 * \brief
 * \author Rong Zhang
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_SUM_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_SUM_INL_H_
#if MXNET_USE_MKLDNN == 1

#include "../../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

struct RequantizeSumParam : public dmlc::Parameter<RequantizeSumParam> {
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;
  DMLC_DECLARE_PARAMETER(RequantizeSumParam) {
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int8 output data.");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int8 output data.");
  }
};

namespace quantized_sum_enum {
enum QuantizedSumOutputs { kOut, kMin, kMax };
enum QuantizedSumInputs { kDataA, kDataB, kAMin, kAMax, kBMin, kBMax};
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_SUM_INL_H_
