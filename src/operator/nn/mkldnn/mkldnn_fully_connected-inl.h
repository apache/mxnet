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
 * \file mkldnn_fully_connected-inl.h
 * \brief
 * \author
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_

#if MXNET_USE_MKLDNN == 1

#include "../fully_connected-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet{
namespace op {

struct MKLDNNFCParam: public dmlc::Parameter<MKLDNNFCParam> {
  bool quantized;
  
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset

  DMLC_DECLARE_PARAMETER(MKLDNNFullyConnectedParam) {
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("enable quantization");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized fullyconnected op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized fullyconnected op to calculate primitive scale");
  }
};

struct MKLDNNFCFullParam {
  FullyConnectedParam fc_param;
  MKLDNNFCParam mkldnn_param;
  std::vector<float> output_scales;
  std::vector<float> requantize_scales;
}

mkldnn::inner_product_forward::primitive_desc GetIPFwd(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const mkldnn::memory::desc &out_md, const bool is_train, MKLDNNFCFullParam &param);

class MKLDNNFullyConnectForward {
 public:
  mkldnn::inner_product_forward::primitive_desc ipFwd_pd;

  MKLDNNFullyConnectForward(const MKLDNNFCFullParam &param, bool is_train,
                            const NDArray &data, const NDArray &weight,
                            const NDArray *bias,
                            const mkldnn::memory::desc &output)
      : ipFwd_pd(GetIPFwd(data, weight, bias, output, is_train, param)) {}


  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output);

  const mkldnn::inner_product_forward &GetIpFwd() const {
    return *ipFwd;
  }

 private:
  std::shared_ptr<mkldnn::inner_product_forward> ipFwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;
};

typedef ParamOpSign<FullyConnectedParam> MKLDNNFullyconSignature;

MKLDNNFullyConnectForward &GetFCFwd(
    const MKLDNNFCFullParam &param, const NDArray &data, const NDArray &weight,
    const NDArray *bias, const mkldnn::memory::desc &output,
    const bool is_train);

void MKLDNNFCForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_