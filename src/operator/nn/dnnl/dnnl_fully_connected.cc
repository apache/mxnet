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
 * \file dnnl_fully_connected.cc
 * \brief DNNL FullyConnected operator
 * \author Da Zheng, Ciyong Chen
 */

#if MXNET_USE_ONEDNN == 1
#include "dnnl_fully_connected-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLFCParam);

dnnl::inner_product_forward::primitive_desc GetFCFwdImpl(const DNNLFCFullParam& full_param,
                                                         const bool is_train,
                                                         const NDArray& data,
                                                         const NDArray& weight,
                                                         const NDArray* bias,
                                                         const dnnl::memory::desc& out_md) {
  auto engine    = CpuEngine::Get()->get_engine();
  auto data_md   = GetMemDesc(data);
  auto weight_md = full_param.dnnl_param.quantized ?
                       GetFCWeightDesc(weight, data.shape()[0], mshadow::kInt8) :
                       GetFCWeightDesc(weight, data.shape()[0]);
  auto propagation =
      is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;

  dnnl::primitive_attr attr;
  dnnl::post_ops ops;
  if (full_param.dnnl_param.with_eltwise) {
    ops.append_eltwise(full_param.eltwise_param.scale,
                       full_param.eltwise_param.alg,
                       full_param.eltwise_param.alpha,
                       full_param.eltwise_param.beta);
  }
  if (full_param.dnnl_param.with_sum) {
    ops.append_sum(full_param.sum_scale);
  }
  attr.set_post_ops(ops);

  if (full_param.dnnl_param.quantized && full_param.output_scales.size()) {
    int mask = (full_param.output_scales.size() == 1) ? 0 : (1 << 1);
    attr.set_output_scales(mask, full_param.output_scales);
  }

  auto GetFCFwdPd = [&full_param, &attr, &engine](const dnnl::inner_product_forward::desc& desc) {
    try {
      return dnnl::inner_product_forward::primitive_desc(desc, attr, engine);
    } catch (dnnl::error& e) {
      if (e.status == dnnl_unimplemented && full_param.dnnl_param.quantized) {
        LOG(ERROR)
            << "AVX512-BW support or oneDNN v0.18 or later is required for INT8 fully_connected.";
      } else {
        LOG(ERROR) << e.message;
      }
      throw;
    }
  };

  if (bias) {
    if ((*bias).shape().ndim() != 1)
      LOG(FATAL) << "Unexpected shape for bias " << (*bias).shape();
    auto bias_md =
        full_param.dnnl_param.quantized ? GetMemDesc(*bias, mshadow::kInt32) : GetMemDesc(*bias);
    dnnl::inner_product_forward::desc desc(propagation, data_md, weight_md, bias_md, out_md);
    return GetFCFwdPd(desc);
  } else {
    dnnl::inner_product_forward::desc desc(propagation, data_md, weight_md, out_md);
    return GetFCFwdPd(desc);
  }
}

inline static dnnl::inner_product_backward_data::primitive_desc GetFCBwdData(
    const NDArray& data,
    const NDArray& weight,
    const NDArray& output,
    dnnl::inner_product_forward::primitive_desc fwd_pd) {
  auto data_md   = GetMemDesc(data);
  auto weight_md = GetFCWeightDesc(weight, data.shape()[0]);
  auto out_md    = GetMemDesc(output);
  auto engine    = CpuEngine::Get()->get_engine();
  dnnl::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return dnnl::inner_product_backward_data::primitive_desc(desc, engine, fwd_pd);
}

inline static dnnl::inner_product_backward_weights::primitive_desc GetFCBwdWeights(
    const NDArray& data,
    const NDArray& weight,
    const NDArray* bias,
    const NDArray& output,
    dnnl::inner_product_forward::primitive_desc fwd_pd) {
  auto data_md   = GetMemDesc(data);
  auto weight_md = GetFCWeightDesc(weight, data.shape()[0]);
  auto out_md    = GetMemDesc(output);
  auto engine    = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    dnnl::inner_product_backward_weights::desc desc(data_md, weight_md, bias_md, out_md);
    return dnnl::inner_product_backward_weights::primitive_desc(desc, engine, fwd_pd);
  } else {
    dnnl::inner_product_backward_weights::desc desc(data_md, weight_md, out_md);
    return dnnl::inner_product_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
}

DNNLFullyConnectedForward& GetFCFwd(const DNNLFCFullParam& param,
                                    const bool is_train,
                                    const NDArray& data,
                                    const NDArray& weight,
                                    const NDArray* bias,
                                    const dnnl::memory::desc& out_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLFullyconSignature, DNNLFullyConnectedForward, OpHash>
      fcFwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<DNNLFullyconSignature, DNNLFullyConnectedForward, OpHash>
          fcFwds;
#endif
  DNNLFullyconSignature key(param);
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(weight);
  if (bias)
    key.AddSign(*bias);

  auto it = fcFwds.find(key);
  if (it == fcFwds.end()) {
    DNNLFullyConnectedForward fcFwd(param, is_train, data, weight, bias, out_md);
    it = AddToCache(&fcFwds, key, fcFwd);
  }
  return it->second;
}

void DNNLFCFlattenData(const FullyConnectedParam& param,
                       const NDArray& out_data,
                       NDArray* in_data,
                       dnnl::memory::desc* out_md) {
  const mxnet::TShape ishape = in_data->shape();
  const mxnet::TShape oshape = out_data.shape();
  if (ishape.ndim() != 2) {
    if (!param.flatten) {
      *in_data = in_data->DNNLDataReshape(
          Shape2(ishape.ProdShape(0, ishape.ndim() - 1), ishape[ishape.ndim() - 1]));
      dnnl::memory::dims out_dims{static_cast<int>(oshape.ProdShape(0, oshape.ndim() - 1)),
                                  static_cast<int>(oshape[ishape.ndim() - 1])};
      *out_md = dnnl::memory::desc(
          out_dims, get_dnnl_type(out_data.dtype()), dnnl::memory::format_tag::any);
    } else {
      *in_data = in_data->DNNLDataReshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));
      dnnl::memory::dims out_dims{static_cast<int>(oshape[0]),
                                  static_cast<int>(oshape.ProdShape(1, oshape.ndim()))};
      *out_md = dnnl::memory::desc(
          out_dims, get_dnnl_type(out_data.dtype()), dnnl::memory::format_tag::any);
    }
  }
}

void DNNLFCForwardFullFeature(const DNNLFCFullParam& full_param,
                              const OpContext& ctx,
                              DNNLFullyConnectedForward* fwd,
                              const std::vector<NDArray>& in_data,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  NDArray weight    = in_data[fullc::kWeight];
  NDArray data      = in_data[fullc::kData];
  auto fwd_src_desc = fwd->fwd_pd.src_desc();
  auto data_mem     = data.GetDNNLDataReorder(&fwd_src_desc);
  const dnnl::memory* weight_mem;
  if (ctx.is_train) {
    if (weight.IsDNNLData()) {
      weight.Reorder2DefaultAsync();
    }
    weight_mem = GetWeights(weight, fwd->fwd_pd.weights_desc(), 1);
  } else {
    weight_mem = weight.GetDNNLData();
    if (weight_mem->get_desc() != fwd->fwd_pd.weights_desc()) {
      auto fwd_weight_desc = fwd->fwd_pd.weights_desc();
      weight.DNNLDataReorderAsync(&fwd_weight_desc);
      weight_mem = GetWeights(weight, fwd->fwd_pd.weights_desc(), 1);
    }
  }
  auto out_mem =
      CreateDNNLMem(out_data[fullc::kOut], fwd->fwd_pd.dst_desc(), req[fullc::kOut], &data);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, *data_mem},
      {DNNL_ARG_WEIGHTS, *weight_mem},
      {DNNL_ARG_DST, *out_mem.second},
  };
  if (!full_param.default_param.no_bias) {
    auto fwd_bias_desc  = fwd->fwd_pd.bias_desc();
    auto bias_mem       = in_data[fullc::kBias].GetDNNLDataReorder(&fwd_bias_desc);
    args[DNNL_ARG_BIAS] = *bias_mem;
  }
  DNNLStream::Get()->RegisterPrimArgs(fwd->GetFwd(), args);
  CommitOutput(out_data[fullc::kOut], out_mem);
  DNNLStream::Get()->Submit();
}

void DNNLFCForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& in_data,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& out_data) {
  DNNLFCFullParam full_param;
  full_param.default_param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  full_param.dnnl_param.Init(std::unordered_map<std::string, std::string>());
  DNNLFCForwardImpl(full_param, ctx, in_data, req, out_data);
}

void DNNLFCForwardImpl(const DNNLFCFullParam& full_param,
                       const OpContext& ctx,
                       const std::vector<NDArray>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& out_data) {
  NDArray data              = in_data[fullc::kData];
  dnnl::memory::desc out_md = GetMemDesc(out_data[fullc::kOut]);
  DNNLFCFlattenData(full_param.default_param, out_data[fullc::kOut], &data, &out_md);
  auto& fwd = GetFCFwd(full_param,
                       ctx.is_train,
                       data,
                       in_data[fullc::kWeight],
                       full_param.default_param.no_bias ? nullptr : &in_data[fullc::kBias],
                       out_md);
  std::vector<NDArray> new_inputs;
  if (full_param.default_param.no_bias)
    new_inputs = {data, in_data[fullc::kWeight]};
  else
    new_inputs = {data, in_data[fullc::kWeight], in_data[fullc::kBias]};
  DNNLFCForwardFullFeature(full_param, ctx, &fwd, new_inputs, req, out_data);
}

void DNNLFCBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const std::vector<NDArray>& in_grad = outputs;
  DNNLFCFullParam full_param;
  full_param.default_param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  full_param.dnnl_param.Init(std::unordered_map<std::string, std::string>());
  const FullyConnectedParam& param = full_param.default_param;
  const mxnet::TShape& ishape      = inputs[fullc::kData + 1].shape();
  const mxnet::TShape& oshape      = inputs[fullc::kOut].shape();

  NDArray weight = inputs[fullc::kWeight + 1];
  NDArray data   = inputs[fullc::kData + 1];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.DNNLDataReshape(
        Shape2(ishape.ProdShape(0, ishape.ndim() - 1), ishape[ishape.ndim() - 1]));
  else if (data.shape().ndim() != 2)
    data = data.DNNLDataReshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));
  NDArray out_grad = inputs[fullc::kOut];
  if (out_grad.shape().ndim() != 2 && !param.flatten)
    out_grad = out_grad.DNNLDataReshape(
        Shape2(oshape.ProdShape(0, oshape.ndim() - 1), oshape[oshape.ndim() - 1]));
  else if (out_grad.shape().ndim() != 2)
    out_grad = out_grad.DNNLDataReshape(Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())));

  dnnl::inner_product_forward::primitive_desc fwd_pd =
      GetFCFwdImpl(full_param,
                   ctx.is_train,
                   data,
                   weight,
                   param.no_bias ? nullptr : &in_grad[fullc::kBias],
                   GetMemDesc(out_grad));

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  if (req[fullc::kWeight]) {
    dnnl::inner_product_backward_weights::primitive_desc ipBwdWeights_pd = GetFCBwdWeights(
        data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias], out_grad, fwd_pd);
    auto ipBwdWeights_diff_dst_desc = ipBwdWeights_pd.diff_dst_desc();
    auto ipBwdWeights_src_desc      = ipBwdWeights_pd.src_desc();
    auto out_grad_mem               = out_grad.GetDNNLDataReorder(&ipBwdWeights_diff_dst_desc);
    auto data_mem                   = data.GetDNNLDataReorder(&ipBwdWeights_src_desc);
    auto in_grad_weight             = CreateDNNLWeightGrad(
        in_grad[fullc::kWeight], ipBwdWeights_pd.diff_weights_desc(), req[fullc::kWeight]);
    dnnl_args_map_t args = {
        {DNNL_ARG_DIFF_DST, *out_grad_mem},
        {DNNL_ARG_SRC, *data_mem},
        {DNNL_ARG_DIFF_WEIGHTS, *in_grad_weight.second},
    };

    dnnl_output_t in_grad_bias;
    if (!param.no_bias) {
      in_grad_bias =
          CreateDNNLMem(in_grad[fullc::kBias], ipBwdWeights_pd.diff_bias_desc(), req[fullc::kBias]);
      args[DNNL_ARG_DIFF_BIAS] = *in_grad_bias.second;
    }
    DNNLStream::Get()->RegisterPrimArgs(dnnl::inner_product_backward_weights(ipBwdWeights_pd),
                                        args);
    CommitOutput(in_grad[fullc::kWeight], in_grad_weight);
    if (!param.no_bias) {
      CommitOutput(in_grad[fullc::kBias], in_grad_bias);
    }
  }
  if (req[fullc::kData]) {
    dnnl::inner_product_backward_data::primitive_desc ipBwdData_pd =
        GetFCBwdData(data, weight, out_grad, fwd_pd);
    auto ipBwdData_diff_dst_desc = ipBwdData_pd.diff_dst_desc();
    auto ipBwdData_weight_desc   = ipBwdData_pd.weights_desc();
    auto out_grad_mem            = out_grad.GetDNNLDataReorder(&ipBwdData_diff_dst_desc);
    auto weight_mem              = weight.GetDNNLDataReorder(&ipBwdData_weight_desc);
    auto in_grad_mem =
        CreateDNNLMem(in_grad[fullc::kData], ipBwdData_pd.diff_src_desc(), req[fullc::kData]);
    dnnl_args_map_t args = {{DNNL_ARG_DIFF_DST, *out_grad_mem},
                            {DNNL_ARG_WEIGHTS, *weight_mem},
                            {DNNL_ARG_DIFF_SRC, *in_grad_mem.second}};

    DNNLStream::Get()->RegisterPrimArgs(dnnl::inner_product_backward_data(ipBwdData_pd), args);
    CommitOutput(in_grad[fullc::kData], in_grad_mem);
  }
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
