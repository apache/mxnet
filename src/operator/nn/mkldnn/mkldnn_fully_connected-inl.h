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
 * \file mkldnn_fully_connected-inl.h
 * \brief Common functions used by MKLDNN (Quantized) FullyConnected operator
 * \author Ciyong Chen
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <vector>
#include <string>
#include "../fully_connected-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNFCParam: public dmlc::Parameter<MKLDNNFCParam> {
  bool quantized;
  bool enable_float_output;
  bool with_eltwise;
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  dmlc::optional<bool> channel_wise_quantize;

  DMLC_DECLARE_PARAMETER(MKLDNNFCParam) {
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("Whether it's a quantized FullyConnected operator");
    DMLC_DECLARE_FIELD(enable_float_output).set_default(false)
    .describe("Whether to enable float32 output");
    DMLC_DECLARE_FIELD(with_eltwise).set_default(false)
    .describe("Whether there's a post with_eltwise after FullyConnected operator");
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
    DMLC_DECLARE_FIELD(channel_wise_quantize)
    .set_default(dmlc::optional<bool>())
    .describe("Whether support channel-wise-quantize for weight.");
  }
};

struct MKLDNNFCFullParam {
  FullyConnectedParam default_param;
  MKLDNNFCParam mkldnn_param;
  MKLDNNPostEltwiseParam eltwise_param;
  std::vector<float> output_scales = {0.0f};
};

mkldnn::inner_product_forward::primitive_desc GetFCFwdImpl(
    const MKLDNNFCFullParam &full_param, const bool is_train,
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const mkldnn::memory::desc &out_md);

class MKLDNNFullyConnectedForward {
 public:
  mkldnn::inner_product_forward::primitive_desc fwd_pd;

  MKLDNNFullyConnectedForward(const MKLDNNFCFullParam &full_param, const bool is_train,
                              const NDArray &data, const NDArray &weight,
                              const NDArray *bias,
                              const mkldnn::memory::desc &out_md)
      : fwd_pd(GetFCFwdImpl(full_param, is_train, data, weight, bias, out_md)) {
          fwd_ = std::make_shared<mkldnn::inner_product_forward>(fwd_pd);
      }

  const mkldnn::inner_product_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::inner_product_forward> fwd_;
};

typedef ParamOpSign<FullyConnectedParam> MKLDNNFullyconSignature;

MKLDNNFullyConnectedForward &GetFCFwd(
    const FullyConnectedParam &param, const bool is_train,
    const NDArray &data, const NDArray &weight,
    const NDArray *bias, const mkldnn::memory::desc &out_md);

void MKLDNNFCFlattenData(const FullyConnectedParam &param,
                         const NDArray &out_data,
                         NDArray *in_data,
                         mkldnn::memory::desc *out_md);

void MKLDNNFCForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data);

void MKLDNNFCForwardFullFeature(const MKLDNNFCFullParam &param,
                                const OpContext &ctx,
                                MKLDNNFullyConnectedForward *fwd,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_FULLY_CONNECTED_INL_H_
