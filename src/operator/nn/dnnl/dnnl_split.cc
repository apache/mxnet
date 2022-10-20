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
 * \file dnnl_split.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "../../tensor/matrix_op-inl.h"
#include "./dnnl_split-inl.h"

namespace mxnet {
namespace op {

void DNNLSplitForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  const SplitParam& param = dmlc::get<SplitParam>(attrs.parsed);
  const auto tensors      = DNNLSplitFwd::Tensors(inputs[0], outputs);

  const auto& ishape   = tensors.input.shape();
  const int split_axis = param.axis >= 0 ? param.axis : param.axis + ishape.ndim();
  const mxnet::TShape split_pts =
      (param.sections > 0) ? GetSplitIndices(tensors.input.shape(), split_axis, param.sections) :
                             param.indices;

  const auto& fwd = DNNLSplitFwd::GetCached(param, tensors, split_pts, split_axis);
  fwd.Execute(tensors, split_pts, split_axis, req);
}

DNNLSplitFwd::Tensors::Tensors(const NDArray& input, const std::vector<NDArray>& outputs)
    : input(input), outputs(outputs) {}

typedef ParamOpSign<SplitParam> DNNLSplitSignature;

DNNLSplitFwd& DNNLSplitFwd::GetCached(const SplitParam& param,
                                      const Tensors& tensors,
                                      const TShape& split_pts,
                                      const int split_axis) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSplitSignature, DNNLSplitFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSplitSignature, DNNLSplitFwd, OpHash> fwds;
#endif

  DNNLSplitSignature key(param);
  key.AddSign(tensors.input);
  key.AddSign(tensors.outputs);
  key.AddSign(split_pts);
  key.AddSign(split_axis);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSplitFwd fwd(tensors, split_pts, split_axis);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

DNNLSplitFwd::DNNLSplitFwd(const Tensors& tensors, const TShape& split_pts, const int split_axis) {
  const auto cpu_engine = CpuEngine::Get()->get_engine();
  const auto input      = tensors.input.Reorder2Default();
  const auto& ishape    = input.shape();
  const auto& dtype     = get_dnnl_type(input.dtype());
  const auto format_tag = static_cast<dnnl::memory::format_tag>(GetDefaultFormat(ishape.ndim()));

  strides = dnnl::memory::dims(ishape.ndim(), 1);
  // last dim stride = 1, start loop from the penultimate
  for (int i = ishape.ndim() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ishape[i + 1];
  }

  for (int i = 0; i < tensors.outputs.size(); ++i) {
    const auto& out = tensors.outputs[i];
    if (out.shape().Size() == 0) {
      continue;
    }
    dnnl::memory::dims dnnl_dims(ishape.begin(), ishape.end());
    // ending split point is always last dimension
    int end_split_pt      = (i + 1 >= split_pts.ndim()) ? ishape[split_axis] : split_pts[i + 1];
    dnnl_dims[split_axis] = end_split_pt - split_pts[i];

    auto in_mem_desc  = dnnl::memory::desc(dnnl_dims, dtype, strides);
    auto out_mem_desc = dnnl::memory::desc(dnnl_dims, dtype, format_tag);

    const auto split_pd = split_fwd_pd_t(cpu_engine, in_mem_desc, cpu_engine, out_mem_desc);
    split_pds.emplace_back(split_pd);
    split_fwds.emplace_back(split_fwd_t(split_pd));
  }
}

void DNNLSplitFwd::Execute(const Tensors& tensors,
                           const TShape& split_pts,
                           const int split_axis,
                           const std::vector<OpReqType>& req) const {
  const auto& cpu_engine = CpuEngine::Get()->get_engine();

  const auto& input_tensor = tensors.input.Reorder2Default();
  int out_idx = 0, primitive_idx = 0;
  int axis_offset      = strides[split_axis] * GetTypeSize(input_tensor.dtype());
  std::byte* input_ptr = reinterpret_cast<std::byte*>(input_tensor.data().dptr_);

  for (const auto& out : tensors.outputs) {
    if (out.shape().Size() == 0) {
      out_idx++;
      continue;
    }
    int offset  = split_pts[out_idx] * axis_offset;
    auto in_mem = dnnl::memory(split_pds[primitive_idx].src_desc(), cpu_engine, input_ptr + offset);

    auto out_mem = CreateDNNLMem(out, split_pds[primitive_idx].dst_desc(), req[out_idx]);
    DNNLStream::Get()->RegisterPrimArgs(split_fwds[primitive_idx],
                                        {{DNNL_ARG_SRC, in_mem}, {DNNL_ARG_DST, *out_mem.second}});

    CommitOutput(out, out_mem);
    ++out_idx;
    ++primitive_idx;
  }
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
