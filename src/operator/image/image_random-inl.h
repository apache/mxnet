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
* \file image_random-inl.h
* \brief
* \author
*/
#ifndef MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_
#define MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_

#include <vector>
#include <mxnet/base.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "mxnet/op_attr_types.h"
#include "image_common.h"


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
  auto output = outputs[0];
  int hight = input.shape_[0];
  int weight = input.shape_[1];
  int channel = input.shape_[2];

  auto input_mat = mat_convert(input, hight, weight, channel);
  auto output_mat = mat_convert(output, hight, weight, channel);
  //input_mat.convertTo(output_mat, -1, 1/255.0, 0);
  std::default_random_engine generator;
  const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);
  float alpha_b = 1.0 + std::uniform_real_distribution<float>(-param.max_brightness, param.max_brightness)(generator);
  output_mat.convertTo(output_mat, -1, alpha_b, 0);
}


template<typename xpu>
static void RandomContrast(const nnvm::NodeAttrs &attrs,
  const OpContext &ctx,
  const std::vector<TBlob> &inputs,
  const std::vector<OpReqType> &req,
  const std::vector<TBlob> &outputs) {
  auto input = inputs[0];
  auto output = outputs[0];
  int hight = input.shape_[0];
  int weight = input.shape_[1];
  int channel = input.shape_[2];

  auto input_mat = mat_convert(input, hight, weight, channel);
  auto output_mat = mat_convert(output, hight, weight, channel);
  //input_mat.convertTo(output_mat, -1, 1/255.0, 0);
  std::default_random_engine generator;
  const RandomBrightnessParam &param = nnvm::get<RandomBrightnessParam>(attrs.parsed);
  float alpha_c = 1.0 + std::uniform_real_distribution<float>(-param.max_brightness, param.max_brightness)(generator);
  cv::Mat temp_;
  cv::cvtColor(input_mat, temp_,  CV_RGB2GRAY);
  float gray_mean = cv::mean(temp_)[0];
  input_mat.convertTo(output_mat, -1, alpha_c, (1 - alpha_c) * gray_mean);

}


} // namespace op
} // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_IMAGE_RANDOM_INL_H_
