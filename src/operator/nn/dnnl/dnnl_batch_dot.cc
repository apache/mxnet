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
 * \file dnnl_batch_dot.cc
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_batch_dot-inl.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLDotParam);

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_matmul.html
bool SupportDNNLBatchDot(const std::vector<NDArray>& inputs) {
  return SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::lhs]) &&
         SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::rhs]);
}

DNNLBatchDotFwd& DNNLBatchDotFwd::GetCached(const DNNLDotParam& param,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<NDArray>& outputs) {
  using batch_dot_fwd_map = std::unordered_map<BatchDotSignature, DNNLBatchDotFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local batch_dot_fwd_map fwds;
#else
  static MX_THREAD_LOCAL batch_dot_fwd_map fwds;
#endif

  BatchDotSignature key(param);
  key.AddSign(inputs[DotIn::lhs]);
  key.AddSign(inputs[DotIn::rhs]);
  key.AddSign(outputs[DotOut::out]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLBatchDotFwd fwd(param, inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

dnnl::primitive_attr GetQuantizationAttributes(const DNNLDotParam& param,
                                               const std::vector<NDArray>& inputs,
                                               const std::vector<NDArray>& outputs) {
  dnnl::primitive_attr attr;
  float out_scale_ = 1.f;
  float lhs_scale_ = GetQuantizeScale(inputs[DotIn::lhs].dtype(),
                                      inputs[DotIn::lhs_min].data().dptr<float>()[0],
                                      inputs[DotIn::lhs_max].data().dptr<float>()[0]);
  float rhs_scale_ = GetQuantizeScale(inputs[DotIn::rhs].dtype(),
                                      inputs[DotIn::rhs_min].data().dptr<float>()[0],
                                      inputs[DotIn::rhs_max].data().dptr<float>()[0]);
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    // fused requantize => output is int
    out_scale_ = GetQuantizeScale(outputs[DotOut::out].dtype(),
                                  param.min_calib_range.value(),
                                  param.max_calib_range.value()) /
                 lhs_scale_ / rhs_scale_;
    attr.set_output_scales(0, {out_scale_});
  } else if (param.enabled_float_output.has_value()) {
    out_scale_ = 1.0 / lhs_scale_ / rhs_scale_;
    attr.set_output_scales(0, {out_scale_});
  }

  return attr;
}

DNNLBatchDotFwd::DNNLBatchDotFwd(const DNNLDotParam& param,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<NDArray>& outputs) {
  auto lhs_shape = inputs[DotIn::lhs].shape();
  auto ndim      = lhs_shape.ndim();
  auto bigDim    = lhs_shape[0];
  for (size_t i = 1; i < ndim - 2; ++i) {
    bigDim *= lhs_shape[i];
  }

  auto GetMemoryDesc = [&ndim, &bigDim](const NDArray& tensor, const bool transpose) {
    auto shape = tensor.shape();
    if (transpose) {
      return dnnl::memory::desc(dnnl::memory::dims{bigDim, shape[ndim - 1], shape[ndim - 2]},
                                get_dnnl_type(tensor.dtype()),
                                dnnl::memory::format_tag::acb);
    } else {
      return dnnl::memory::desc(dnnl::memory::dims{bigDim, shape[ndim - 2], shape[ndim - 1]},
                                get_dnnl_type(tensor.dtype()),
                                dnnl::memory::format_tag::any);
    }
  };

  dnnl::memory::desc data_md    = GetMemoryDesc(inputs[DotIn::lhs], param.transpose_a);
  dnnl::memory::desc weights_md = GetMemoryDesc(inputs[DotIn::rhs], param.transpose_b);
  dnnl::memory::desc out_md({bigDim, data_md.dims()[1], weights_md.dims()[2]},
                            get_dnnl_type(outputs[DotOut::out].dtype()),
                            dnnl::memory::format_tag::any);
  dnnl::matmul::desc fwd_desc(data_md, weights_md, out_md);
  if (param.quantized) {
    auto attrs = GetQuantizationAttributes(param, inputs, outputs);
    fwd_pd     = std::make_shared<batch_dot_fwd_pd_t>(
        fwd_desc, attrs, mxnet::CpuEngine::Get()->get_engine());

  } else {
    fwd_pd = std::make_shared<batch_dot_fwd_pd_t>(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  }

  fwd = std::make_shared<batch_dot_fwd_t>(*fwd_pd);
}

void DNNLBatchDotFwd::Execute(const OpContext& ctx,
                              const DNNLDotParam& param,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto lhs    = inputs[DotIn::lhs];
  auto rhs    = inputs[DotIn::rhs];
  // Created primitive descriptor assumes that both inputs are in default format
  if (lhs.IsDNNLData())
    lhs = lhs.Reorder2Default();
  if (rhs.IsDNNLData())
    rhs = rhs.Reorder2Default();

  auto lhs_mem =
      dnnl::memory(fwd_pd->src_desc(), engine, reinterpret_cast<void*>(lhs.data().dptr_));
  auto rhs_mem =
      dnnl::memory(fwd_pd->weights_desc(), engine, reinterpret_cast<void*>(rhs.data().dptr_));
  dnnl_output_t out_mem = CreateDNNLMem(
      outputs[DotOut::out], fwd_pd->dst_desc(), req[DotOut::out], &inputs[DotIn::lhs]);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, lhs_mem},
      {DNNL_ARG_WEIGHTS, rhs_mem},
      {DNNL_ARG_DST, *out_mem.second},
  };

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[0], out_mem);
  DNNLStream::Get()->Submit();

  if (param.quantized && !param.enabled_float_output.has_value()) {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    float min_output;
    float max_output;
    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      min_output = param.min_calib_range.value();
      max_output = param.max_calib_range.value();
    } else {
      if (inputs[DotIn::lhs].dtype() == mshadow::kInt8) {
        mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
            s,
            1,
            &min_output,
            &max_output,
            inputs[DotIn::rhs_min].data().dptr<float>(),
            inputs[DotIn::rhs_max].data().dptr<float>(),
            inputs[DotIn::lhs_min].data().dptr<float>(),
            inputs[DotIn::lhs_max].data().dptr<float>());
      } else {
        mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
            s,
            1,
            &min_output,
            &max_output,
            inputs[DotIn::rhs_min].data().dptr<float>(),
            inputs[DotIn::rhs_max].data().dptr<float>(),
            inputs[DotIn::lhs_min].data().dptr<float>(),
            inputs[DotIn::lhs_max].data().dptr<float>());
      }
    }

    float* min_output_ptr = outputs[DotOut::out_min].data().dptr<float>();
    float* max_output_ptr = outputs[DotOut::out_max].data().dptr<float>();
    *min_output_ptr       = min_output;
    *max_output_ptr       = max_output;
  }
}

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::DNNLDotParam> {
  size_t operator()(const mxnet::op::DNNLDotParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.transpose_a);
    ret        = dmlc::HashCombine(ret, val.transpose_b);
    ret        = dmlc::HashCombine(ret, val.quantized);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_USE_ONEDNN == 1
