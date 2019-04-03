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
 * \file mkldnn_convolution-inl.h
 * \brief
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <vector>
#include <utility>
#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNConvParam : public dmlc::Parameter<MKLDNNConvParam> {
  bool with_bn;
  bool with_relu;
  bool with_sum;
  bool with_postsum_relu;
  bool quantized;
  bool weight_channelwise_scale;

  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset

  DMLC_DECLARE_PARAMETER(MKLDNNConvParam) {
    DMLC_DECLARE_FIELD(with_bn).set_default(false)
    .describe("Add post batchnorm.");
    DMLC_DECLARE_FIELD(with_relu).set_default(false)
    .describe("Add post relu");
    DMLC_DECLARE_FIELD(with_sum).set_default(false)
    .describe("Add post sum");
    DMLC_DECLARE_FIELD(with_postsum_relu).set_default(false)
    .describe("Add post relu after sum");
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("enable quantization");
    DMLC_DECLARE_FIELD(weight_channelwise_scale).set_default(true)
    .describe("Quantize weight with channel wise scales.");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized convolution op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized convolution op to calculate primitive scale");
  }
};

struct MKLDNNConvFullParam {
  ConvolutionParam conv_param;
  MKLDNNConvParam mkldnn_param;
  float sum_scale;
  std::vector<float> requantize_scales;
};

static inline bool IsOutputUInt8(const MKLDNNConvParam &mkldnn_param) {
  return ((!mkldnn_param.with_sum) && mkldnn_param.with_relu) ||
         mkldnn_param.with_postsum_relu;
}

mkldnn::convolution_forward::primitive_desc
GetConvFwdImpl(const MKLDNNConvFullParam &param, const bool is_train,
               const NDArray &data, const NDArray &weights, const NDArray *bias,
               const NDArray &output);

class MKLDNNConvForward {
 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const MKLDNNConvFullParam &param, const bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output)
      : fwd_pd(GetConvFwdImpl(param, is_train, data, weights, bias, output)) {}

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output);

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::convolution_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> weight_;
  std::shared_ptr<mkldnn::memory> bias_;
  std::shared_ptr<mkldnn::memory> out_;
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

MKLDNNConvForward &GetConvFwd(const ConvolutionParam &param,
                              const bool is_train, const NDArray &data,
                              const NDArray &weights, const NDArray *bias,
                              const NDArray &output);

void MKLDNNConvolutionForwardFullFeature(const MKLDNNConvFullParam &param,
                                         const OpContext &ctx,
                                         MKLDNNConvForward *fwd,
                                         const std::vector<NDArray> &in_data,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_
