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
 * \file dnnl_dequantize-inl.h
 * \author Wenting Jiang, Xinyu Chen
 * \brief
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_DEQUANTIZE_INL_H_
#if MXNET_USE_ONEDNN == 1
#include <algorithm>
#include <string>
#include <vector>

#include "operator/nn/dnnl/dnnl_base-inl.h"

namespace mxnet {
namespace op {

class SgDNNLDequantizeOperator {
 public:
  explicit SgDNNLDequantizeOperator(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<DequantizeParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  bool initialized_{false};
  DequantizeParam param_;
  float cached_data_min_{0.f};
  float cached_data_max_{0.f};
  dnnl::memory::desc o_desc_;
  dnnl_args_map_t args_;
  std::shared_ptr<dnnl::reorder> fwd_pd_;
};

void SgDNNLDequantizeOperator::Forward(const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
  NDArray in_buffer = inputs[0];
  if (inputs[0].IsView() && inputs[0].IsDNNLData())
    in_buffer = inputs[0].Reorder2Default();
  auto i_mem     = in_buffer.GetDNNLData();
  float data_min = *inputs[1].data().dptr<float>();
  float data_max = *inputs[2].data().dptr<float>();

  if (initialized_ && (cached_data_min_ != data_min || cached_data_max_ != data_max))
    initialized_ = false;

  if (!initialized_) {
    cached_data_min_      = data_min;
    cached_data_max_      = data_max;
    float real_range      = MaxAbs(cached_data_min_, cached_data_max_);
    float quantized_range = 0.0;
    if (inputs[0].dtype() == mshadow::kUint8) {
      quantized_range = kUint8Range;
    } else if (inputs[0].dtype() == mshadow::kInt8) {
      quantized_range = kInt8Range;
      real_range      = MaxAbs(*inputs[1].data().dptr<float>(), *inputs[2].data().dptr<float>());
    } else {
      LOG(FATAL) << "dnnl dequantize op only supports int8 and uint8 as output type";
    }
    float scale = real_range / quantized_range;
    dnnl::primitive_attr attr;
    const int mask            = 0;
    std::vector<float> scales = {scale};
    attr.set_output_scales(mask, scales);
    dnnl::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
    auto i_desc             = i_mem->get_desc();
    size_t i_ndim           = in_buffer.shape().ndim();
    if (i_ndim == 4) {
      dnnl::memory::format_tag o_fmt = dnnl::memory::format_tag::nchw;
      dnnl::memory::dims o_dims(i_desc.data.dims, i_desc.data.dims + i_desc.data.ndims);
      o_desc_ = dnnl::memory::desc(o_dims, get_dnnl_type<float>(), o_fmt);
    } else {
      o_desc_                = i_desc;
      o_desc_.data.data_type = get_dnnl_type_t<float>();
    }
    auto reorder_pd = dnnl::reorder::primitive_desc(cpu_engine, i_desc, cpu_engine, o_desc_, attr);
    fwd_pd_         = std::make_shared<dnnl::reorder>(reorder_pd);
    initialized_    = true;
  }
  auto o_mem           = CreateDNNLMem(outputs[0], o_desc_, req[0]);
  args_[DNNL_ARG_FROM] = *i_mem;
  args_[DNNL_ARG_TO]   = *o_mem.second;
  DNNLStream::Get()->RegisterPrimArgs(*fwd_pd_, args_);
  CommitOutput(outputs[0], o_mem);
  DNNLStream::Get()->Submit();
}

static void SgDNNLDequantizeForward(const OpStatePtr& state_ptr,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  SgDNNLDequantizeOperator& op = state_ptr.get_state<SgDNNLDequantizeOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_DEQUANTIZE_INL_H_
