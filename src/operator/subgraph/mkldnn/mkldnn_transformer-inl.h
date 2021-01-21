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

#ifndef MXNET_OPERATOR_CONTRIB_MKLDNN_TRANSFORMER_INL_H_
#define MXNET_OPERATOR_CONTRIB_MKLDNN_TRANSFORMER_INL_H_

#include "../../mxnet_op.h"

namespace mxnet {
namespace op {

struct MKLDNNInterleavedMatMulParam : public dmlc::Parameter<MKLDNNInterleavedMatMulParam> {
  int heads;
  bool quantized;
  bool enable_float_output;
  bool with_mask;
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;
  DMLC_DECLARE_PARAMETER(MKLDNNInterleavedMatMulParam) {
    DMLC_DECLARE_FIELD(heads)
    .describe("Set number of heads");
    DMLC_DECLARE_FIELD(quantized).set_default(false)
    .describe("Whether it's a quantized InterleavedMatMul operator");
    DMLC_DECLARE_FIELD(enable_float_output).set_default(false)
    .describe("Whether to enable float32 output");
    DMLC_DECLARE_FIELD(with_mask).set_default(false)
    .describe("Whether to mask the output");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized InterleavedMatMul op to calculate primitive scale");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to by "
              "quantized InterleavedMatMul op to calculate primitive scale");
  }
};

class MKLDNNInterleavedMatMulSelfAttQKOp {
 public:
  explicit MKLDNNInterleavedMatMulSelfAttQKOp(const nnvm::NodeAttrs &attrs) :
    param_(nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn interleaved only supports "
                  "inference computation.";
  }

 private:
  bool initialized_{false};
  MKLDNNInterleavedMatMulParam param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::reorder> apply_mask_;
  std::shared_ptr<dnnl::memory> cached_data1_mem_;
  std::shared_ptr<dnnl::memory> cached_data2_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  std::shared_ptr<dnnl::memory> cached_mask_mem_;
  float cached_min_output_;
  float cached_max_output_;
};

class MKLDNNInterleavedMatMulSelfAttValAttOp {
 public:
  explicit MKLDNNInterleavedMatMulSelfAttValAttOp(const nnvm::NodeAttrs &attrs) :
    param_(nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn interleaved only supports "
                  "inference computation.";
  }

 private:
  bool initialized_{false};
  MKLDNNInterleavedMatMulParam param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_data1_mem_;
  std::shared_ptr<dnnl::memory> cached_data2_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  float cached_min_output_;
  float cached_max_output_;
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_MKLDNN_TRANSFORMER_INL_H_
