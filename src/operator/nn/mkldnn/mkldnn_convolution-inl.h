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
  bool with_act;
  bool with_sum;
  bool with_postsum_act;
  bool quantized;

  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset

  DMLC_DECLARE_PARAMETER(MKLDNNConvParam) {
    DMLC_DECLARE_FIELD(with_bn).set_default(false)
    .describe("Add post batchnorm.");
    DMLC_DECLARE_FIELD(with_act).set_default(false)
    .describe("Add post activation");
    DMLC_DECLARE_FIELD(with_sum).set_default(false)
    .describe("Add post sum");
    DMLC_DECLARE_FIELD(with_postsum_act).set_default(false)
    .describe("Add post activation after sum");
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("enable quantization");
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
  float sum_scale = 1.f;
  std::vector<float> requantize_scales;
  MKLDNNPostEltwiseParam act_param;
  MKLDNNPostEltwiseParam postsum_act_param;
};

std::shared_ptr<mkldnn::convolution_forward::primitive_desc> GetConvFwdImpl(
    const ConvolutionParam &param, const bool is_train, const NDArray &data, const NDArray &weight,
    const NDArray *bias, const NDArray &output);

class MKLDNNConvForward {
 public:
  MKLDNNConvForward(const MKLDNNConvFullParam &param, const bool is_train, const NDArray &data,
                    const NDArray &weight, const NDArray *bias, const NDArray &output);

  const mkldnn::convolution_forward &GetFwd() const { return *fwd_; }

  const mkldnn::convolution_forward::primitive_desc &GetPd() const { return *pd_; }

 private:
  std::shared_ptr<mkldnn::convolution_forward> fwd_;
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> pd_;
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

MKLDNNConvForward &GetConvFwd(const MKLDNNConvFullParam &param, const bool is_train,
                              const NDArray &data, const NDArray &weight, const NDArray *bias,
                              const NDArray &output);

void MKLDNNConvolutionForwardFullFeature(const MKLDNNConvFullParam &param,
                                         const OpContext &ctx,
                                         MKLDNNConvForward *fwd,
                                         const std::vector<NDArray> &in_data,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &out_data);

void MKLDNNConvolutionForward(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data);

class MKLDNNConvBackward {
 public:
  MKLDNNConvBackward(const MKLDNNConvFullParam &param, const NDArray &data, const NDArray &weight,
                     const NDArray *bias, const NDArray &output);

  const mkldnn::convolution_backward_data &GetBwdData() const { return *bwd_data_; }

  const mkldnn::convolution_backward_weights &GetBwdWeights() const { return *bwd_weight_; }

  const mkldnn::convolution_backward_data::primitive_desc &GetDataPd() const {
    return *bwd_data_pd_;
  }

  const mkldnn::convolution_backward_weights::primitive_desc &GetWeightsPd() const {
    return *bwd_weight_pd_;
  }

 private:
  std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> bwd_data_pd_;
  std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc> bwd_weight_pd_;
  std::shared_ptr<mkldnn::convolution_backward_data> bwd_data_;
  std::shared_ptr<mkldnn::convolution_backward_weights> bwd_weight_;
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_
