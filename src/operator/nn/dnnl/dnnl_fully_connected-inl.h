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
 * \file dnnl_fully_connected-inl.h
 * \brief Common functions used by DNNL (Quantized) FullyConnected operator
 * \author Ciyong Chen
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_FULLY_CONNECTED_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "operator/nn/fully_connected-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

struct DNNLFCParam : public dmlc::Parameter<DNNLFCParam> {
  bool quantized;
  bool enable_float_output;
  bool with_eltwise;
  bool with_sum;
  bool first_quantization_pass;  // True for operator created during first quantization pass
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  dmlc::optional<bool> channel_wise_quantize;

  DMLC_DECLARE_PARAMETER(DNNLFCParam) {
    DMLC_DECLARE_FIELD(quantized).set_default(false).describe(
        "Whether it's a quantized FullyConnected operator");
    DMLC_DECLARE_FIELD(enable_float_output)
        .set_default(false)
        .describe("Whether to enable float32 output");
    DMLC_DECLARE_FIELD(with_eltwise)
        .set_default(false)
        .describe("Whether there's a post with_eltwise after FullyConnected operator");
    DMLC_DECLARE_FIELD(with_sum).set_default(false).describe("Add post sum");
    DMLC_DECLARE_FIELD(first_quantization_pass)
        .set_default(false)
        .describe("True for first quantization pass");
    DMLC_DECLARE_FIELD(min_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The minimum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized fullyconnected op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
        .set_default(dmlc::optional<float>())
        .describe(
            "The maximum scalar value in the form of float32 obtained "
            "through calibration. If present, it will be used to by "
            "quantized fullyconnected op to calculate primitive scale");
    DMLC_DECLARE_FIELD(channel_wise_quantize)
        .set_default(dmlc::optional<bool>())
        .describe("Whether support channel-wise-quantize for weight.");
  }
};

struct DNNLFCFullParam {
  FullyConnectedParam default_param;
  DNNLFCParam dnnl_param;
  DNNLPostEltwiseParam eltwise_param;
  float sum_scale                  = {1.0f};
  std::vector<float> output_scales = {0.0f};
};

static inline size_t GetInSumIndex(const DNNLFCFullParam& param) {
  assert(param.dnnl_param.with_sum);
  return fullc::kWeight + 1 + (param.default_param.no_bias ? 0 : 1);
}

class FCInputIndex {
 public:
  explicit FCInputIndex(const DNNLFCFullParam full_param) {
    auto& dnnl_param     = full_param.dnnl_param;
    const bool has_bias  = !full_param.default_param.no_bias;
    const bool quantized = dnnl_param.quantized;
    const bool sum_input_quantized =
        quantized && dnnl_param.with_sum && !dnnl_param.enable_float_output;
    const bool channel_wise = quantized && dnnl_param.channel_wise_quantize.has_value() &&
                              dnnl_param.channel_wise_quantize.value();

    // Calculate position of particular input in the input vector:
    int index = 0;
    data      = index++;
    weight    = index++;
    bias      = has_bias ? index++ : 0;
    sum       = dnnl_param.with_sum ? index++ : 0;
    num_base  = index;  // note number of base inputs

    data_min   = quantized ? index++ : 0;
    data_max   = quantized ? index++ : 0;
    weight_min = (quantized && !channel_wise) ? index++ : 0;
    weight_max = (quantized && !channel_wise) ? index++ : 0;
    bias_min   = (quantized && !channel_wise && has_bias) ? index++ : 0;
    bias_max   = (quantized && !channel_wise && has_bias) ? index++ : 0;
    sum_min    = sum_input_quantized ? index++ : 0;
    sum_max    = sum_input_quantized ? index++ : 0;
    num_total  = index;  // note number of total inputs
  }

  // Returns true if sum input exists
  bool IsSumExist() const {
    return sum;
  }

  // Returns true if bias input exists
  bool IsBiasExist() const {
    return bias;
  }

  // Returns true if sum input exists and it is float number
  bool IsSumInputFloat() const {
    return (sum && !sum_min);
  }
  int GetTotal() const {
    return num_total;
  }
  int GetBase() const {
    return num_base;
  }

  // Represent index of particular input in the input vector:
  int data;
  int weight;
  int bias;
  int sum;
  int data_min;
  int data_max;
  int weight_min;
  int weight_max;
  int bias_min;
  int bias_max;
  int sum_min;
  int sum_max;

 private:
  int num_base;   // Number of standard inputs
  int num_total;  // Number of total inputs: standard + additional needed for
                  // quantization
};

dnnl::inner_product_forward::primitive_desc GetFCFwdImpl(const DNNLFCFullParam& full_param,
                                                         const bool is_train,
                                                         const NDArray& data,
                                                         const NDArray& weight,
                                                         const NDArray* bias,
                                                         const dnnl::memory::desc& out_md);

class DNNLFullyConnectedForward {
 public:
  dnnl::inner_product_forward::primitive_desc fwd_pd;

  DNNLFullyConnectedForward(const DNNLFCFullParam& full_param,
                            const bool is_train,
                            const NDArray& data,
                            const NDArray& weight,
                            const NDArray* bias,
                            const dnnl::memory::desc& out_md)
      : fwd_pd(GetFCFwdImpl(full_param, is_train, data, weight, bias, out_md)) {
    fwd_ = std::make_shared<dnnl::inner_product_forward>(fwd_pd);
  }

  const dnnl::inner_product_forward& GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<dnnl::inner_product_forward> fwd_;
};

typedef ParamOpSign<FullyConnectedParam> DNNLFullyconSignature;

DNNLFullyConnectedForward& GetFCFwd(const FullyConnectedParam& param,
                                    const bool is_train,
                                    const NDArray& data,
                                    const NDArray& weight,
                                    const NDArray* bias,
                                    const dnnl::memory::desc& out_md);

void DNNLFCFlattenData(const FullyConnectedParam& param,
                       const NDArray& out_data,
                       NDArray* in_data,
                       dnnl::memory::desc* out_md);

void DNNLFCForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& in_data,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& out_data);

void DNNLFCForwardFullFeature(const DNNLFCFullParam& param,
                              const OpContext& ctx,
                              DNNLFullyConnectedForward* fwd,
                              const std::vector<NDArray>& in_data,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_FULLY_CONNECTED_INL_H_
