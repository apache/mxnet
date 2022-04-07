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
 * \file dnnl_quantize_asym-inl.h
 * \brief implementation of asymmetric quantize operation using DNNL
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_ASYM_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_ASYM_INL_H_
#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <vector>
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantize_asym-inl.h"

namespace mxnet {
namespace op {

class DNNLQuantizeAsymOp {
 public:
  explicit DNNLQuantizeAsymOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<QuantizeAsymParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  QuantizeAsymParam param_;
  bool initialized_{false};
  float cached_scale_{0.f};
  float cached_shift_{0.f};
  dnnl::memory::desc o_desc_;
  dnnl_args_map_t args_;
  std::shared_ptr<dnnl::reorder> fwd_pd_;
};

void DNNLQuantizeAsymOp::Forward(const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  using mshadow::red::limits::MaxValue;
  using mshadow::red::limits::MinValue;
  NDArray in_buffer = inputs[0];
  float scale       = 0.f;
  float shift       = 0.f;

  // Pass through quantized data
  if (inputs[0].dtype() == mshadow::kUint8) {
    *outputs[1].data().dptr<float>() = 1;
    *outputs[2].data().dptr<float>() = 0;
    if (req[0] != kWriteInplace) {
      const_cast<NDArray&>(outputs[0]).CopyFrom(*inputs[0].GetDNNLData());
      DNNLStream::Get()->Submit();
    }
  } else {
    in_buffer                 = inputs[0].Reorder2Default();
    const dnnl::memory* i_mem = in_buffer.GetDNNLData();
    float* in_ptr             = in_buffer.data().dptr<float>();
    const int nthreads        = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (inputs[0].dtype() == mshadow::kInt8) {
      *outputs[1].data().dptr<float>() = 1;
      *outputs[2].data().dptr<float>() = 128;
#pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(in_buffer.shape().Size()); ++i) {
        in_ptr[i] += 128.0f;
      }
    } else if (inputs[0].dtype() == mshadow::kFloat32) {
      if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
        scale =
            MaxValue<uint8_t>() / (param_.max_calib_range.value() - param_.min_calib_range.value());
        shift = MaxValue<uint8_t>() - param_.max_calib_range.value() * scale;
      } else {
        float data_min = mshadow::red::limits::MaxValue<float>();
        float data_max = mshadow::red::limits::MinValue<float>();
        std::vector<float> data_maxs(nthreads, data_max);
        std::vector<float> data_mins(nthreads, data_min);
#pragma omp parallel for num_threads(nthreads)
        for (index_t i = 0; i < static_cast<index_t>(in_buffer.shape().Size()); i++) {
          int tid = omp_get_thread_num();
          if (in_ptr[i] > data_maxs[tid])
            data_maxs[tid] = in_ptr[i];
          if (in_ptr[i] < data_mins[tid])
            data_mins[tid] = in_ptr[i];
        }
        for (index_t i = 0; i < nthreads; i++) {
          if (data_maxs[i] > data_max)
            data_max = data_maxs[i];
          if (data_mins[i] < data_min)
            data_min = data_mins[i];
        }
        scale = MaxValue<uint8_t>() / (data_max - data_min);
        shift = MaxValue<uint8_t>() - data_max * scale;
      }

      if (initialized_ && (cached_scale_ != scale || cached_shift_ != shift))
        initialized_ = false;
    }

    *outputs[1].data().dptr<float>() = scale;
    *outputs[2].data().dptr<float>() = shift;

    if (!initialized_) {
      cached_scale_ = scale;
      cached_shift_ = shift;
      dnnl::primitive_attr attr;
      attr.set_rnn_data_qparams(scale, shift);
      const dnnl::engine& cpu_engine   = mxnet::CpuEngine::Get()->get_engine();
      const dnnl::memory::desc& i_desc = i_mem->get_desc();
      o_desc_                          = i_desc;
      o_desc_.data.data_type           = get_dnnl_type_t(outputs[0].dtype());
      dnnl::reorder::primitive_desc reorder_pd(cpu_engine, i_desc, cpu_engine, o_desc_, attr);
      fwd_pd_      = std::make_shared<dnnl::reorder>(reorder_pd);
      initialized_ = true;
    }
    dnnl_output_t o_mem  = CreateDNNLMem(outputs[0], o_desc_, req[0]);
    args_[DNNL_ARG_FROM] = *i_mem;
    args_[DNNL_ARG_TO]   = *o_mem.second;
    DNNLStream::Get()->RegisterPrimArgs(*fwd_pd_, args_);
    CommitOutput(outputs[0], o_mem);
    DNNLStream::Get()->Submit();
  }
}

void DNNLQuantizeAsymForward(const OpStatePtr& state_ptr,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  if (inputs[0].shape().ndim() == 3 && inputs[0].dtype() == mshadow::kFloat32) {
    DNNLQuantizeAsymOp& op = state_ptr.get_state<DNNLQuantizeAsymOp>();
    op.Forward(ctx, inputs, req, outputs);
  } else {
    FallBackCompute(QuantizeAsymForward<cpu>, state_ptr, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZE_ASYM_INL_H_
