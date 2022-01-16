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
 * \file dnnl_stack.cc
 */

#include "./dnnl_base-inl.h"
#include "./dnnl_concat-inl.h"
#include "./dnnl_ops-inl.h"

#include "../../tensor/matrix_op-inl.h"

#if MXNET_USE_ONEDNN == 1
namespace mxnet {
namespace op {

bool SupportDNNLStack(const std::vector<NDArray>& inputs) {
  if (inputs[0].dtype() != mshadow::kFloat32 && inputs[0].dtype() != mshadow::kBfloat16) {
    return false;
  }

  int src_dtype = inputs[0].dtype();
  for (const auto& arr : inputs) {
    if (arr.dtype() != src_dtype) {
      return false;
    }
    // Do not support zero-size tensors.
    if (arr.shape().Size() == 0) {
      return false;
    }

    int ndim = arr.shape().ndim();
    if (ndim <= 0) {
      return false;
    }
  }
  return true;
}

void DNNLStackForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& in_data,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);

  // const value of artificial new dimension to
  // stack tensors on using oneDNN concat primitive
  constexpr int stacking_dim = 1;

  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
  const int axis          = CheckAxis(param.axis, out_data[0].shape().ndim());
  const TShape oshape     = out_data[0].shape();
  const int src_dtype     = in_data[0].dtype();
  const int dst_dtype     = out_data[0].dtype();
  const int mid_dim       = oshape[axis];
  int leading_dim         = 1;
  int trailing_dim        = 1;

  for (int i = 0; i < axis; ++i) {
    leading_dim *= oshape[i];
  }
  for (int i = axis + 1; i < oshape.ndim(); ++i) {
    trailing_dim *= oshape[i];
  }

  std::vector<dnnl::memory::desc> data_md;
  std::vector<dnnl::memory> data_mem;
  dnnl::memory::desc in_md({leading_dim, stacking_dim, trailing_dim},
                           get_dnnl_type(src_dtype),
                           dnnl::memory::format_tag::abc);
  dnnl::memory::desc out_md({leading_dim, mid_dim, trailing_dim},
                            get_dnnl_type(dst_dtype),
                            dnnl::memory::format_tag::any);

  const int num_in_data = in_data.size();
  data_md.reserve(num_in_data);
  data_mem.reserve(num_in_data);

  MSHADOW_TYPE_SWITCH(src_dtype, DType, {
    for (int i = 0; i < num_in_data; i++) {
      NDArray tmp = in_data[i].Reorder2Default();
      dnnl::memory tmp_mem(in_md, CpuEngine::Get()->get_engine(), tmp.data().dptr<DType>());
      data_mem.emplace_back(tmp_mem);
      data_md.emplace_back(in_md);
    }
  });

  auto& fwd = GetConcatForward(stacking_dim, in_data, data_md, axis);
  mxnet::dnnl_output_t out_mem =
      CreateDNNLMem(out_data[concat_enum::kOut], fwd.fwd_pd.dst_desc(), req[concat_enum::kOut]);

  std::unordered_map<int, dnnl::memory> net_args;
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  for (int i = 0; i < num_in_data; i++) {
    net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, data_mem[i]});
  }

  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
