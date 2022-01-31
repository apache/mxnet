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
 * \file dnnl_where.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_where-inl.h"
#include "src/operator/operator_common.h"

namespace mxnet {
namespace op {

bool SupportDNNLWhere(const std::vector<NDArray>& inputs) {
  static const std::set<int> supported_dtypes = {
      mshadow::kFloat32, mshadow::kBfloat16, mshadow::kInt8, mshadow::kUint8};
  for (int i = 0; i < inputs.size(); ++i) {
    if (!supported_dtypes.count(inputs[i].dtype()) || inputs[i].shape().Size() <= 0 ||
        inputs[i].shape().ndim() <= 0) {
      return false;
    }
  }
  return true;
}

void DNNLWhereForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  const auto tensors = DNNLWhereFwd::Tensors(inputs, outputs[0]);
  const auto fwd     = DNNLWhereFwd::GetCached(tensors);
  fwd.Execute(tensors, req, ctx);
}

DNNLWhereFwd::Tensors::Tensors(const std::vector<NDArray>& inputs, const NDArray& output)
    : condition(inputs[0]), left(inputs[1]), right(inputs[2]), output(output) {}

DNNLWhereFwd DNNLWhereFwd::GetCached(const Tensors& tensors) {
  using where_op_fwd_map = std::unordered_map<OpSignature, DNNLWhereFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local where_op_fwd_map fwds;
#else
  static MX_THREAD_LOCAL where_op_fwd_map fwds;
#endif

  OpSignature key;
  key.AddSign(tensors.condition);
  key.AddSign(tensors.left);
  key.AddSign(tensors.right);
  key.AddSign(tensors.output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLWhereFwd fwd(tensors);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

static mxnet::TShape GetBroadcastableShape(const mxnet::TShape& in_shape,
                                           const mxnet::TShape& out_shape) {
  if (in_shape == out_shape) {
    return in_shape;
  }

  mxnet::TShape broadcastable_in_shape(out_shape.ndim(), -1);
  const int lack_dims = out_shape.ndim() - in_shape.ndim();
  for (int i = 0; i < out_shape.ndim(); ++i) {
    int y = 1;
    if (i >= lack_dims) {
      y = in_shape[i - lack_dims];
    }
    broadcastable_in_shape[i] = y;
  }
  return broadcastable_in_shape;
}

DNNLWhereFwd::DNNLWhereFwd(const Tensors& tensors) {
  const auto cpu_engine = CpuEngine::Get()->get_engine();

  const auto cnd = tensors.condition;
  const auto lhs = tensors.left;
  const auto rhs = tensors.right;
  const auto out = tensors.output;

  const auto cnd_shape = GetBroadcastableShape(cnd.shape(), out.shape());
  const auto lhs_shape = GetBroadcastableShape(lhs.shape(), out.shape());
  const auto rhs_shape = GetBroadcastableShape(rhs.shape(), out.shape());

  const auto& cnd_dtype = get_dnnl_type(cnd.dtype());
  const auto& inp_dtype = get_dnnl_type(lhs.dtype());
  const auto& def_ft    = static_cast<dnnl::memory::format_tag>(GetDefaultFormat(lhs_shape.ndim()));

  const auto& cnd_dims    = dnnl::memory::dims(cnd_shape.begin(), cnd_shape.end());
  const auto& lhs_dims    = dnnl::memory::dims(lhs_shape.begin(), lhs_shape.end());
  const auto& rhs_dims    = dnnl::memory::dims(rhs_shape.begin(), rhs_shape.end());
  const auto& out_dims    = dnnl::memory::dims(out.shape().begin(), out.shape().end());
  const auto& scalar_dims = dnnl::memory::dims(cnd_shape.ndim(), 1);  // broadcastable scalar

  auto cnd_md    = dnnl::memory::desc(cnd_dims, cnd_dtype, def_ft);
  auto lhs_md    = dnnl::memory::desc(lhs_dims, inp_dtype, def_ft);
  auto rhs_md    = dnnl::memory::desc(rhs_dims, inp_dtype, def_ft);
  auto out_md    = dnnl::memory::desc(out_dims, inp_dtype, def_ft);
  auto scalar_md = dnnl::memory::desc(scalar_dims, cnd_dtype, def_ft);

  binary_eq_zero_pd = dnnl::binary::primitive_desc(
      dnnl::binary::desc(dnnl::algorithm::binary_ne, cnd_md, scalar_md, cnd_md), cpu_engine);
  binary_ne_zero_pd = dnnl::binary::primitive_desc(
      dnnl::binary::desc(dnnl::algorithm::binary_eq, cnd_md, scalar_md, cnd_md), cpu_engine);

  // if broadcast is needed output must be larger in size
  auto lmask_dim  = lhs_shape.Size() > cnd_shape.Size() ? lhs_dims : cnd_dims;
  auto lmask_md   = dnnl::memory::desc(lmask_dim, inp_dtype, def_ft);
  binary_mul_l_pd = dnnl::binary::primitive_desc(
      dnnl::binary::desc(dnnl::algorithm::binary_mul, lhs_md, cnd_md, lmask_md), cpu_engine);

  auto rmask_dim  = rhs_shape.Size() > cnd_shape.Size() ? rhs_dims : cnd_dims;
  auto rmask_md   = dnnl::memory::desc(rmask_dim, inp_dtype, def_ft);
  binary_mul_r_pd = dnnl::binary::primitive_desc(
      dnnl::binary::desc(dnnl::algorithm::binary_mul, rhs_md, cnd_md, rmask_md), cpu_engine);

  binary_sum_pd = dnnl::binary::primitive_desc(
      dnnl::binary::desc(dnnl::algorithm::binary_add, lmask_md, rmask_md, out_md), cpu_engine);

  binary_eq_zero = dnnl::binary(binary_eq_zero_pd);
  binary_ne_zero = dnnl::binary(binary_ne_zero_pd);
  binary_mul_l   = dnnl::binary(binary_mul_l_pd);
  binary_mul_r   = dnnl::binary(binary_mul_r_pd);
  binary_sum     = dnnl::binary(binary_sum_pd);
}

void DNNLWhereFwd::Execute(const Tensors& tensors,
                           const std::vector<OpReqType>& req,
                           const OpContext& ctx) const {
  const auto& cpu_engine = CpuEngine::Get()->get_engine();
  const auto& cpu_stream = ctx.get_stream<cpu>();

  const auto& cnd_tensor = tensors.condition.GetDNNLDataReorder(binary_eq_zero_pd.src0_desc());
  const auto& lhs_tensor = tensors.left.GetDNNLDataReorder(binary_mul_l_pd.src0_desc());
  const auto& rhs_tensor = tensors.right.GetDNNLDataReorder(binary_mul_r_pd.src0_desc());

  mxnet::dnnl_output_t out_mem = CreateDNNLMem(tensors.output, binary_sum_pd.dst_desc(), req[0]);

  const auto& ishape = tensors.left.shape();

  const int dtype_size =
      std::max(GetTypeSize(tensors.condition.dtype()), GetTypeSize(tensors.left.dtype()));

  // allocate temporary memory for 4 additional tensors
  mshadow::Tensor<cpu, 1> tmp_workspace = ctx.requested[0].get_space<cpu>(
      mshadow::Shape1(tensors.output.shape().Size() * 4 * dtype_size, cpu_stream);
  char* workspace_ptr   = reinterpret_cast<char*>(tmp_workspace.dptr_);
  const int offset_size = tensors.output.shape().Size() * dtype_size;

  dnnl::memory cnd_lhs(binary_eq_zero_pd.dst_desc(), cpu_engine, workspace_ptr);
  dnnl::memory cnd_rhs(binary_ne_zero_pd.dst_desc(), cpu_engine, workspace_ptr + offset_size);
  dnnl::memory masked_lhs(binary_mul_l_pd.dst_desc(), cpu_engine, workspace_ptr + 2 * offset_size);
  dnnl::memory masked_rhs(binary_mul_r_pd.dst_desc(), cpu_engine, workspace_ptr + 3 * offset_size);

  double zero{0};
  dnnl::memory zero_scalar(binary_ne_zero_pd.src1_desc(), cpu_engine, &zero);

  DNNLStream::Get()->RegisterPrimArgs(
      binary_eq_zero,
      {{DNNL_ARG_SRC_0, *cnd_tensor}, {DNNL_ARG_SRC_1, zero_scalar}, {DNNL_ARG_DST, cnd_lhs}});

  DNNLStream::Get()->RegisterPrimArgs(
      binary_ne_zero,
      {{DNNL_ARG_SRC_0, *cnd_tensor}, {DNNL_ARG_SRC_1, zero_scalar}, {DNNL_ARG_DST, cnd_rhs}});

  DNNLStream::Get()->RegisterPrimArgs(
      binary_mul_l,
      {{DNNL_ARG_SRC_0, *lhs_tensor}, {DNNL_ARG_SRC_1, cnd_lhs}, {DNNL_ARG_DST, masked_lhs}});

  DNNLStream::Get()->RegisterPrimArgs(
      binary_mul_r,
      {{DNNL_ARG_SRC_0, *rhs_tensor}, {DNNL_ARG_SRC_1, cnd_rhs}, {DNNL_ARG_DST, masked_rhs}});

  DNNLStream::Get()->RegisterPrimArgs(binary_sum,
                                      {{DNNL_ARG_SRC_0, masked_lhs},
                                       {DNNL_ARG_SRC_1, masked_rhs},
                                       {DNNL_ARG_DST, *out_mem.second}});

  CommitOutput(tensors.output, out_mem);
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif