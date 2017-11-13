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
* \file random_brightness-inl.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_RANDOM_BRIGHTNESS_INL_H_
#define MXNET_OPERATOR_IMAGE_RANDOM_BRIGHTNESS_INL_H_

#include <vector>
#include <mxnet/base.h>
#include <opencv2/core/mat.hpp>
#include "mxnet/op_attr_types.h"

namespace mxnet {
namespace op {
  struct RandomBrightnessParam : public dmlc::Parameter<RandomBrightnessParam> {
    float max_brightness;
    DMLC_DECLARE_PARAMETER(RandomBrightnessParam) {
      DMLC_DECLARE_FIELD(max_brightness)
      .set_default(0.0)
      .describe("Max Contrast.");
    }
  };


  template<typename xpu>
  static void RandomBrightness(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
    auto input = inputs[0];
    cv::Mat m;

    int hight = input.shape_[0];
    int weight = input.shape_[1];
    int channel = input.shape_[2];
    switch (input.type_flag_) {
      case mshadow::kFloat32: {
        typedef float DType;
        m = cv::Mat(hight, weight, CV_MAKETYPE(CV_32F, channel), input.dptr<DType>());
      }
      break;
      case mshadow::kFloat64: {
        typedef double DType;
        m = cv::Mat(hight, weight, CV_MAKETYPE(CV_64F, channel), input.dptr<DType>());
      }
      break;
      case mshadow::kFloat16: {
        typedef mshadow::half::half_t DType;

      }
      break;
      case mshadow::kUint8: {
        typedef uint8_t DType;
        m = cv::Mat(hight, weight, CV_MAKETYPE(CV_8U, channel), input.dptr<DType>());
      }
      break;
      case mshadow::kInt8: {
        typedef int8_t DType;
        m = cv::Mat(hight, weight, CV_MAKETYPE(CV_8S, channel), input.dptr<DType>());
      }
      break;
      case mshadow::kInt32: {
        typedef int32_t DType;
        m = cv::Mat(hight, weight, CV_MAKETYPE(CV_32S, channel), input.dptr<DType>());
      }
      break;
      case mshadow::kInt64: {
        typedef int64_t DType;
      }
      break;
      default:
        LOG(FATAL) << "Unknown type enum " << input.type_flag_;

    }
    std::default_random_engine generator;
    const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);
    float alpha_b = 1.0 + std::uniform_real_distribution<float>(-param.max_brightness, param.max_brightness)(generator);
    m.convertTo(m, -1, alpha_b, 0);
  }
} // namespace op
} // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_RANDOM_BRIGHTNESS_INL_H_
