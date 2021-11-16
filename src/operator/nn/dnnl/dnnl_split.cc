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
 * \brief
 */

#if MXNET_USE_ONEDNN == 1

#include "../../tensor/matrix_op-inl.h"
#include "./dnnl_split-inl.h"

namespace mxnet {
namespace op {

bool SupportDNNLSplit(const SplitParam& param, const NDArray& input) {
  //   int in_ndim          = input.shape().ndim();
  //   int out_size         = output.shape().Size();
  //   int in_size          = input.shape().Size();
  //   bool param_supported = true;
    return (input.dtype() == mshadow::kFloat32 || input.dtype() == mshadow::kBfloat16);// &&
          //  (output.dtype() == mshadow::kFloat32 || output.dtype() == mshadow::kBfloat16);
  // return true;
}

void DNNLSplitForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp)
    return;
  CHECK_NE(req[0], kAddTo);
  const SplitParam& param = dmlc::get<SplitParam>(attrs.parsed);
  const bool is_train     = ctx.is_train;
  const auto tensors      = DNNLSplitFwd::Tensors(inputs[0], outputs);
  const auto fwd          = DNNLSplitFwd::GetCached(param, tensors, is_train);
  fwd.Execute(tensors);
}

DNNLSplitFwd::Tensors::Tensors(const NDArray& input, const std::vector<NDArray>& outputs)
    : input(input), outputs(outputs) {}

typedef ParamOpSign<SplitParam> DNNLSplitSignature;

DNNLSplitFwd DNNLSplitFwd::GetCached(const SplitParam& param,
                                     const Tensors& tensors,
                                     const bool is_train) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSplitSignature, DNNLSplitFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSplitSignature, DNNLSplitFwd, OpHash> fwds;
#endif

  DNNLSplitSignature key(param);
  key.AddSign(is_train);
  key.AddSign(tensors.input);
  key.AddSign(tensors.outputs);
  key.AddSign(param.indices);
  key.AddSign(param.squeeze_axis);
  key.AddSign(param.sections);
  DNNLSplitFwd fwd(param, tensors, is_train);
  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSplitFwd fwd(param, tensors, is_train);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
  return fwd;
}

DNNLSplitFwd::DNNLSplitFwd(const SplitParam& param, const Tensors& tensors, const bool is_train) {
    auto input_tensor        = tensors.input.Reorder2Default();
    // create X mem descriptors
    auto cpu_engine = CpuEngine::Get()->get_engine();
    const auto& ishape = tensors.input.shape();
    int real_axis      = param.axis;
    if (real_axis < 0) {
      real_axis += ishape.ndim();
    }

    const mxnet::TShape split_pts =
        (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
    // LOG(INFO) << split_pts;
    std::vector<int> strides(ishape.ndim());
    strides[ishape.ndim() - 1] = 1;
    for (int i = ishape.ndim() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * ishape[i + 1];
    }
    dnnl::memory::dims dnnl_strides(strides.begin(), strides.end());

    for (int i = 0; i < split_pts.ndim(); ++i) {
      auto section_shape       = ishape;
      int end_dim;
      if (i + 1 >= split_pts.ndim()) {
        end_dim = ishape[real_axis];
      } else {
        end_dim = split_pts[i + 1];
      }
      section_shape[real_axis] = end_dim - split_pts[i];
      if(section_shape[real_axis] == 0)
        continue;
      // LOG(INFO) << section_shape;
      dnnl::memory::dims dnnl_dims(section_shape.begin(), section_shape.end());
      auto in_mem_desc =
          dnnl::memory::desc(dnnl_dims, get_dnnl_type(tensors.input.dtype()), dnnl_strides);
      int offset = split_pts[i] * strides[real_axis] * GetTypeSize(tensors.input.dtype());
      auto in_mem =
          dnnl::memory(in_mem_desc, cpu_engine, reinterpret_cast<void*>(input_tensor.data().dptr_) + offset);

      auto out_mem             = tensors.outputs[i].GetDNNLData();
      const auto reorder_pd =
          dnnl::reorder::primitive_desc(cpu_engine, in_mem_desc, cpu_engine, out_mem->get_desc());
      dnnl_args_map_t reorder_args;
      reorder_args[DNNL_ARG_SRC] = in_mem;
      reorder_args[DNNL_ARG_DST] = *out_mem;
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd), reorder_args);
    DNNLStream::Get()->Submit();
    }
    // split indicies e.g. [0,1,3] - from 0 to 1 and from 1 to 3 --- 2 sections

    //   auto input_md         = input_mem->get_desc();
    //   const auto in_shape   = tensors.data.shape();
    //   const size_t in_ndim  = in_shape.ndim();
    //   const size_t out_ndim = tensors.out.shape().ndim();
    //   const auto out_dtype  = get_dnnl_type(tensors.out.dtype());
    //   dnnl::memory::desc out_md;

    //   split_pd  = std::make_shared<split_fwd_pd_t>(GetSplitFwdPd(input_md, out_md,
    //   reduction_alg)); split_fwd = std::make_shared<split_fwd_t>(*split_pd);
}

// split_fwd_pd_t DNNLSplitFwd::GetSplitFwdPd(const dnnl::memory::desc& input_md,
//                                            const dnnl::memory::desc& output_md) {
//   auto cpu_engine = CpuEngine::Get()->get_engine();
//   auto desc       = dnnl::reorder(input_md, output_md);
//   return split_fwd_pd_t(desc, cpu_engine);
// }

void DNNLSplitFwd::Execute(const Tensors& tensors) const {
  //   auto stream    = DNNLStream::Get();
  //   auto engine    = CpuEngine::Get()->get_engine();
  //   auto input_mem = tensors.data.GetDNNLData();
  //   if (tensors.out.shape().Size() == 1) {
  //     // scalar result
  //     auto out_mem = dnnl::memory(split_pd->dst_desc(), engine,
  //     tensors.out.data().dptr<float>()); stream->RegisterPrimArgs(*split_fwd, {{DNNL_ARG_SRC,
  //     *input_mem}, {DNNL_ARG_DST, out_mem}});
  //   } else {
  //     auto out_mem = tensors.out.GetDNNLData(split_pd->dst_desc());
  //     stream->RegisterPrimArgs(*split_fwd, {{DNNL_ARG_SRC, *input_mem}, {DNNL_ARG_DST,
  //     *out_mem}});
  //   }
  //   stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif