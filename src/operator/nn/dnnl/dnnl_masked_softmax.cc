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
 * \file dnnl_maskedsoftmax.cc
 */

#include "dnnl_masked_softmax-inl.h"
#include "dnnl_softmax-inl.h"

#if MXNET_USE_ONEDNN == 1
namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_softmax.html
bool SupportDNNLMaskedSoftmax(const MaskedSoftmaxParam& param, const std::vector<NDArray>& inputs) {
  CHECK_EQ(inputs.size(), 2);
  const auto data = inputs[0];
  const auto mask = inputs[1];
  SoftmaxParam softmax_param;
  softmax_param.axis        = param.axis;
  softmax_param.dtype       = data.dtype();
  softmax_param.temperature = param.temperature;
  // threshold value selected experimentally basing on performance results - PR-21106
  constexpr size_t optimal_size_threshold = 2 << 13;
  return data.shape().Size() >= optimal_size_threshold && mask.dtype() == mshadow::kBool &&
         SupportDNNLSoftmax(softmax_param, data);
}

inline static dnnl::memory::dims GetOneDNNDims(const NDArray& arr) {
  return dnnl::memory::dims(arr.shape().begin(), arr.shape().end());
}

typedef ParamOpSign<MaskedSoftmaxParam> MaskedSoftmaxSignature;

DNNLMaskedSoftmaxFwd& DNNLMaskedSoftmaxFwd::GetCached(
    const MaskedSoftmaxParam& param,
    const DNNLMaskedSoftmaxFwd::Tensors& tensors) {
  using maskedsoftmax_fwd_map =
      std::unordered_map<MaskedSoftmaxSignature, DNNLMaskedSoftmaxFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local maskedsoftmax_fwd_map fwds;
#else
  static MX_THREAD_LOCAL maskedsoftmax_fwd_map fwds;
#endif
  MaskedSoftmaxSignature key(param);
  key.AddSign(tensors.input);
  key.AddSign(tensors.mask);
  key.AddSign(tensors.output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLMaskedSoftmaxFwd fwd(param, tensors);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

std::shared_ptr<Primitives> DNNLMaskedSoftmaxFwd::CreatePrimitives(const MaskedSoftmaxParam& param,
                                                                   const Tensors& tensors) {
  const dnnl::engine engine = CpuEngine::Get()->get_engine();

  auto prim = std::make_shared<Primitives>();

  auto mem_format =
      static_cast<dnnl::memory::format_tag>(GetDefaultFormat(tensors.input.shape().ndim()));
  dnnl::memory::dims input_dims = GetOneDNNDims(tensors.input);
  // output_desc is the same
  dnnl::memory::desc input_desc =
      dnnl::memory::desc(input_dims, dnnl::memory::data_type::f32, mem_format);

  dnnl::memory::dims mask_dims = GetOneDNNDims(tensors.mask);
  dnnl::memory::desc mask_desc =
      dnnl::memory::desc(mask_dims, dnnl::memory::data_type::s8, mem_format);

  // (mask - 1)
  auto minusone_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_scoring,
                                                   dnnl::algorithm::eltwise_linear,
                                                   mask_desc,
                                                   1.0f,    // multiply factor
                                                   -1.0f);  // minus one
  prim->minusone_pd  = dnnl::eltwise_forward::primitive_desc(minusone_desc, engine);
  prim->minusone     = dnnl::eltwise_forward(prim->minusone_pd);

  // (input + mask) / temperature
  auto mask_input_desc =
      dnnl::binary::desc(dnnl::algorithm::binary_add,
                         input_desc,
                         dnnl::memory::desc(mask_dims, dnnl::memory::data_type::f32, mem_format),
                         input_desc);
  if (param.temperature.has_value() && param.temperature.value() != 1.0f) {
    dnnl::post_ops binary_ops;
    binary_ops.append_eltwise(
        1.0f, dnnl::algorithm::eltwise_linear, 1 / param.temperature.value(), 0.0f);
    dnnl::primitive_attr binary_attr;
    binary_attr.set_post_ops(binary_ops);

    prim->mask_input_pd = dnnl::binary::primitive_desc(mask_input_desc, binary_attr, engine);
  } else {
    prim->mask_input_pd = dnnl::binary::primitive_desc(mask_input_desc, engine);
  }
  prim->mask_input = dnnl::binary(prim->mask_input_pd);

  // output * mask
  auto mask_output_desc =
      dnnl::binary::desc(dnnl::algorithm::binary_mul, input_desc, mask_desc, input_desc);
  prim->mask_output_pd = dnnl::binary::primitive_desc(mask_output_desc, engine);
  prim->mask_output    = dnnl::binary(prim->mask_output_pd);

  return prim;
}

void DNNLMaskedSoftmaxForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  const auto& param  = nnvm::get<MaskedSoftmaxParam>(attrs.parsed);
  const auto tensors = DNNLMaskedSoftmaxFwd::Tensors(inputs, outputs);
  const auto& fwd    = DNNLMaskedSoftmaxFwd::GetCached(param, tensors);
  fwd.Execute(tensors, req, param, ctx.is_train);
}

void DNNLMaskedSoftmaxFwd::Execute(const Tensors& tensors,
                                   const std::vector<OpReqType>& req,
                                   const MaskedSoftmaxParam& param,
                                   bool is_train) const {
  using namespace mxnet::op;
  using namespace mxnet::op::mxnet_op;
  using namespace dnnl;
  /*
   Three steps of masked softmax:
   1. out = (input + [(mask - 1) * inf]) / temperature
   2. out = softmax(out)
   3. out = out * mask
  */
  DNNLStream* stream    = DNNLStream::Get();
  auto engine           = CpuEngine::Get()->get_engine();
  const NDArray& input  = tensors.input;
  const NDArray& mask   = tensors.mask;
  const NDArray& output = tensors.output;

  // 1. A) out = mask - 1
  const memory::desc mask_desc    = this->primitives->minusone_pd.src_desc();
  dnnl::memory* mask_minusone_mem = TmpMemMgr::Get()->Alloc(mask_desc);
  dnnl::memory mask_mem =
      dnnl::memory(mask_desc, engine, reinterpret_cast<int8_t*>(mask.data().dptr<bool>()));
  stream->RegisterPrimArgs(this->primitives->minusone,
                           {
                               {DNNL_ARG_SRC, mask_mem},
                               {DNNL_ARG_DST, *mask_minusone_mem},
                           });

  // 1. B) out = out * inf
  const memory::desc converted_mask_desc = this->primitives->mask_input_pd.src1_desc();
  dnnl::memory* converted_mask_mem       = TmpMemMgr::Get()->Alloc(converted_mask_desc);
  dnnl::primitive_attr attr;
  attr.set_output_scales(0, {mshadow::red::limits::MaxValue<float>()});
  std::unordered_map<int, dnnl::memory> args(
      {{DNNL_ARG_FROM, *mask_minusone_mem}, {DNNL_ARG_TO, *converted_mask_mem}});
  stream->RegisterPrimArgs(dnnl::reorder(*mask_minusone_mem, *converted_mask_mem, attr), args);

  // prepare softmax primitive and memory
  SoftmaxParam p;
  p.axis = param.axis;

  auto softmax_tensors     = DNNLSoftmaxFwd::Tensors(output, output);
  auto softmax_op          = DNNLSoftmaxFwd::GetCached(p, softmax_tensors, is_train);
  auto softmax_op_dst_desc = softmax_op.softmax_pd->dst_desc();
  auto softmax_out_mem     = output.GetDNNLData(&softmax_op_dst_desc);
  const auto input_mem     = input.GetDNNLData();

  // 1. C) out = (input + out) / temperature
  stream->RegisterPrimArgs(this->primitives->mask_input,
                           {{DNNL_ARG_SRC_0, *input_mem},
                            {DNNL_ARG_SRC_1, *converted_mask_mem},
                            {DNNL_ARG_DST, *softmax_out_mem}});
  stream->Submit();

  // 2. out = softmax(out)
  softmax_op.Execute(softmax_tensors);

  // 3. out = out * mask
  stream->RegisterPrimArgs(this->primitives->mask_output,
                           {{DNNL_ARG_SRC_0, *softmax_out_mem},
                            {DNNL_ARG_SRC_1, mask_mem},
                            {DNNL_ARG_DST, *softmax_out_mem}});
  stream->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif
