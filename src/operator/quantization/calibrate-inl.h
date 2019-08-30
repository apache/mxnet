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
 * \file calibraite-inl.h
 * \brief Implementation of calibrate operator
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_

#include <limits>
#include <vector>
#include "../mxnet_op.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

struct CalibrateEntropyParam : public dmlc::Parameter<CalibrateEntropyParam> {
  int num_quantized_bins;
  DMLC_DECLARE_PARAMETER(CalibrateEntropyParam) {
    DMLC_DECLARE_FIELD(num_quantized_bins)
      .set_default(255)
      .describe(
          "The number of quantized bins.");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_CALIBRATE_INL_H_
