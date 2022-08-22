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
 * \file dnnl_batch_norm-inl.h
 * \brief
 * \author Tao Lv
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_NORM_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <dnnl.hpp>

#include <utility>
#include <vector>

#include "dnnl_base-inl.h"
#include "operator/nn/batch_norm-inl.h"

namespace mxnet {
namespace op {

typedef dnnl::batch_normalization_forward::primitive_desc t_bn_f_pdesc;
typedef dnnl::batch_normalization_forward::desc t_bn_f_desc;
typedef dnnl::batch_normalization_backward::primitive_desc t_bn_b_pdesc;
typedef dnnl::batch_normalization_backward::desc t_bn_b_desc;

inline static dnnl::normalization_flags _GetFlags(const std::vector<NDArray>& in_data,
                                                  const std::vector<NDArray>& aux_states,
                                                  bool is_train_and_not_global_stats) {
  dnnl::normalization_flags flags = static_cast<dnnl::normalization_flags>(0U);
  if (in_data.size() == 3U) {
    flags |= dnnl::normalization_flags::use_scale_shift;
  }

  // aux_states[0]: inMean
  // aux_states[1]: inVariance
  if (aux_states.size() == 2U && !is_train_and_not_global_stats) {
    flags |= dnnl::normalization_flags::use_global_stats;
  }
  return flags;
}

inline static t_bn_f_pdesc _GetFwd(const dnnl::memory& data_mem,
                                   bool is_train,
                                   bool fuse_relu,
                                   float eps,
                                   dnnl::normalization_flags flags) {
  auto data_md = data_mem.get_desc();
  auto engine  = CpuEngine::Get()->get_engine();

  if (is_train) {
    t_bn_f_desc bnFwd_desc(dnnl::prop_kind::forward_training, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  }

  if (fuse_relu) {
    const float scale = 1.f;
    const float alpha = 0.f;
    const float beta  = 0.f;
    dnnl::post_ops post_ops;
    post_ops.append_eltwise(scale, dnnl::algorithm::eltwise_relu, alpha, beta);
    dnnl::primitive_attr attr;
    attr.set_post_ops(post_ops);
    t_bn_f_desc bnFwd_desc(dnnl::prop_kind::forward_inference, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, attr, engine);
  } else {
    t_bn_f_desc bnFwd_desc(dnnl::prop_kind::forward_inference, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  }
}

inline static t_bn_b_pdesc _GetBwd(const dnnl::memory& data_mem,
                                   const dnnl::memory& diff_mem,
                                   float eps,
                                   dnnl::normalization_flags flags) {
  auto data_md = data_mem.get_desc();
  auto diff_md = diff_mem.get_desc();
  auto engine  = CpuEngine::Get()->get_engine();

  t_bn_b_desc bnBwd_desc(dnnl::prop_kind::backward, diff_md, data_md, eps, flags);
  return t_bn_b_pdesc(bnBwd_desc, engine, _GetFwd(data_mem, true, false, eps, flags));
}

typedef ParamOpSign<BatchNormParam> DNNLBNSignature;

class DNNLBNForward {
  std::shared_ptr<const dnnl::memory> weight_m;
  std::shared_ptr<dnnl::batch_normalization_forward> fwd;
  bool is_train_and_not_global_stats;
  t_bn_f_pdesc pd;

 public:
  DNNLBNForward(const t_bn_f_pdesc& _pd, bool is_train_and_not_global_stats);

  const dnnl::memory& GetWeight() const;

  const t_bn_f_pdesc& GetPd() const;

  const dnnl::batch_normalization_forward& GetFwd() const;

  static DNNLBNForward& GetCached(const BatchNormParam& param,
                                  const OpContext& ctx,
                                  const dnnl::memory* data_mem,
                                  bool fuse_relu,
                                  dnnl::normalization_flags flags);
  void Execute(const OpContext& ctx,
               const BatchNormParam& param,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               bool fuse_relu);
};

template <bool fuse_relu>
void DNNLBatchNormForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);

  mxnet::TShape shape = inputs[batchnorm::kData].shape();
  const int real_axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  CHECK_LT(real_axis, shape.ndim());
  if (param.axis != 1 || shape.ndim() != 4) {
    // reshape to (N, C, 1, D)
    mxnet::TShape new_shape{
        static_cast<index_t>(shape.ProdShape(0, real_axis)),
        shape[real_axis],
        1,
        static_cast<index_t>(shape.ProdShape(real_axis + 1, static_cast<int>(shape.ndim())))};
    in_data[batchnorm::kData] = in_data[batchnorm::kData].Reshape(new_shape);
  }

  const std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  dnnl::normalization_flags flags =
      _GetFlags(in_data, aux_states, ctx.is_train && !param.use_global_stats);
  NDArray& data = in_data[batchnorm::kData];
  if (data.IsDNNLData() && data.IsView())
    data = data.Reorder2Default();
  auto data_mem      = data.GetDNNLData();
  DNNLBNForward& fwd = DNNLBNForward::GetCached(param, ctx, data_mem, fuse_relu, flags);
  fwd.Execute(ctx, param, inputs, req, outputs, fuse_relu);
}

class DNNLBNBackward {
  std::shared_ptr<dnnl::batch_normalization_backward> bwd;
  const std::shared_ptr<dnnl::memory> weight_m;
  const std::shared_ptr<dnnl::memory> gradw_m;

 public:
  const t_bn_b_pdesc pd;

  explicit DNNLBNBackward(const t_bn_b_pdesc& _pd);

  const dnnl::memory& GetWeight() const;

  const dnnl::memory& GetGradw() const;

  const dnnl::batch_normalization_backward& GetBwd() const;

  static DNNLBNBackward& GetCached(const BatchNormParam& param,
                                   const OpContext& ctx,
                                   const NDArray& in_data,
                                   const dnnl::memory& in_mem,
                                   const NDArray& diff_data,
                                   const dnnl::memory& diff_mem,
                                   dnnl::normalization_flags flags);

  void Execute(const BatchNormParam& param,
               const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);
};

inline void DNNLBatchNormBackward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  const BatchNormParam& param = nnvm::get<BatchNormParam>(attrs.parsed);
  std::vector<NDArray> out_grad(1);
  std::vector<NDArray> in_data(3);
  std::vector<NDArray> aux_states(2);
  out_grad[0]                        = inputs[0];
  in_data[batchnorm::kData]          = inputs[3];
  in_data[batchnorm::kGamma]         = inputs[4];
  in_data[batchnorm::kBeta]          = inputs[5];
  aux_states[batchnorm::kMovingMean] = inputs[6];
  aux_states[batchnorm::kMovingVar]  = inputs[7];
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  dnnl::normalization_flags flags =
      _GetFlags(in_data, aux_states, ctx.is_train && !param.use_global_stats);

  NDArray data = in_data[batchnorm::kData];
  NDArray diff = out_grad[batchnorm::kOut];

  mxnet::TShape shape = data.shape();
  const int real_axis = mxnet::op::batchnorm::GetRealAxis(shape, param.axis);
  CHECK_LT(real_axis, shape.ndim());
  if (param.axis != 1 || shape.ndim() != 4) {
    // reshape to (N, C, 1, D)
    mxnet::TShape new_shape{
        static_cast<index_t>(shape.ProdShape(0, real_axis)),
        shape[real_axis],
        1,
        static_cast<index_t>(shape.ProdShape(real_axis + 1, static_cast<int>(shape.ndim())))};
    data = data.Reshape(new_shape);
    diff = diff.Reshape(new_shape);
  }

  auto data_mem = data.GetDNNLData();
  auto diff_mem = diff.GetDNNLData();
  if (data.IsDefaultData()) {
    auto diff_desc = diff_mem->get_desc();
    data_mem       = data.GetDNNLDataReorder(&diff_desc);
  } else if (diff.IsDefaultData()) {
    auto data_desc = data_mem->get_desc();
    diff_mem       = diff.GetDNNLDataReorder(&data_desc);
  }
  DNNLBNBackward& bwd =
      DNNLBNBackward::GetCached(param, ctx, data, *data_mem, diff, *diff_mem, flags);
  bwd.Execute(param, ctx, inputs, req, outputs);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_NORM_INL_H_
