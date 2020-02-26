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
 * Copyright (c) 2018 by Contributors
 * \file mkldnn_fully_connected.cc
 * \brief MKLDNN FullyConnected operator
 * \author Da Zheng, Ciyong Chen
*/

#if MXNET_USE_MKLDNN == 1
#include "mkldnn_fully_connected-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNFCParam);

mkldnn::inner_product_forward::primitive_desc GetFCFwdImpl(
    const MKLDNNFCFullParam &full_param, const bool is_train,
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const mkldnn::memory::desc &out_md) {
  auto data_md = GetMemDesc(data);
  auto weight_md = full_param.mkldnn_param.quantized ?
    GetFCWeightDesc(weight, mshadow::kInt8) : GetFCWeightDesc(weight);
  auto engine = CpuEngine::Get()->get_engine();
  auto propagation =
    is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;

  mkldnn::primitive_attr attr;
  mkldnn::post_ops ops;
  if (full_param.mkldnn_param.with_eltwise) {
    ops.append_eltwise(full_param.eltwise_param.scale,
                       full_param.eltwise_param.alg,
                       full_param.eltwise_param.alpha,
                       full_param.eltwise_param.beta);
  }
  attr.set_post_ops(ops);

  if (full_param.mkldnn_param.quantized && full_param.output_scales.size()) {
    int mask = (full_param.output_scales.size() == 1) ? 0 : (1 << 1);
    attr.set_output_scales(mask, full_param.output_scales);
  }

  auto GetFCFwdPd = [&full_param, &attr,
                     &engine](const mkldnn::inner_product_forward::desc &desc) {
    try {
      return mkldnn::inner_product_forward::primitive_desc(desc, attr, engine);
    } catch (mkldnn::error &e) {
      if (e.status == mkldnn_unimplemented &&
          full_param.mkldnn_param.quantized) {
        LOG(ERROR) << "AVX512-BW support or MKLDNN v0.18 is required for INT8 fully_connected.";
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
       full_param.mkldnn_param.quantized ? GetMemDesc(*bias, mshadow::kInt32) : GetMemDesc(*bias);
    mkldnn::inner_product_forward::desc desc(propagation,
        data_md, weight_md, bias_md, out_md);
    return GetFCFwdPd(desc);
  } else {
    mkldnn::inner_product_forward::desc desc(propagation,
        data_md, weight_md, out_md);
    return GetFCFwdPd(desc);
  }
}

inline static mkldnn::inner_product_backward_data::primitive_desc GetFCBwdData(
    const NDArray &data, const NDArray &weight, const NDArray &output,
    mkldnn::inner_product_forward::primitive_desc fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetFCWeightDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return mkldnn::inner_product_backward_data::primitive_desc(desc, engine, fwd_pd);
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetFCBwdWeights(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, mkldnn::inner_product_forward::primitive_desc fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetFCWeightDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_backward_weights::desc desc(data_md,
        weight_md, bias_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        desc, engine, fwd_pd);
  } else {
    mkldnn::inner_product_backward_weights::desc desc(data_md,
        weight_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        desc, engine, fwd_pd);
  }
}

MKLDNNFullyConnectedForward &GetFCFwd(
    const FullyConnectedParam &param, const bool is_train,
    const NDArray &data, const NDArray &weight,
    const NDArray *bias, const mkldnn::memory::desc &out_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNFullyconSignature,
              MKLDNNFullyConnectedForward, OpHash> fcFwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNFullyconSignature,
              MKLDNNFullyConnectedForward, OpHash> fcFwds;
#endif
  MKLDNNFullyconSignature key(param);
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(weight);
  if (bias)
    key.AddSign(*bias);

  auto it = fcFwds.find(key);
  if (it == fcFwds.end()) {
    MKLDNNFCFullParam full_param;
    full_param.default_param = param;
    full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());
    MKLDNNFullyConnectedForward fcFwd(full_param, is_train, data, weight, bias, out_md);
    it = AddToCache(&fcFwds, key, fcFwd);
  }
  return it->second;
}

void MKLDNNFCFlattenData(const FullyConnectedParam &param,
                         const NDArray &out_data,
                         NDArray *in_data,
                         mkldnn::memory::desc *out_md) {
  const mxnet::TShape ishape = in_data->shape();
  const mxnet::TShape oshape = out_data.shape();
  if (ishape.ndim() != 2) {
    if (!param.flatten) {
      *in_data = in_data->MKLDNNDataReshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                                    ishape[ishape.ndim()-1]));
      mkldnn::memory::dims out_dims{static_cast<int>(oshape.ProdShape(0, oshape.ndim()-1)),
        static_cast<int>(oshape[ishape.ndim()-1])};
      *out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out_data.dtype()),
                                     mkldnn::memory::format_tag::any);
    } else {
      *in_data = in_data->MKLDNNDataReshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));
      mkldnn::memory::dims out_dims{static_cast<int>(oshape[0]),
        static_cast<int>(oshape.ProdShape(1, oshape.ndim()))};
      *out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out_data.dtype()),
                                     mkldnn::memory::format_tag::any);
    }
  }
}

void MKLDNNFCForwardFullFeature(const MKLDNNFCFullParam &full_param,
                                const OpContext &ctx,
                                MKLDNNFullyConnectedForward *fwd,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  NDArray weight = in_data[fullc::kWeight];
  NDArray data = in_data[fullc::kData];

  auto data_mem = data.GetMKLDNNDataReorder(fwd->fwd_pd.src_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    if (weight.IsMKLDNNData()) {
      weight.Reorder2DefaultAsync();
    }
    weight_mem = GetWeights(weight, fwd->fwd_pd.weights_desc(), 1);
  } else {
    weight_mem = weight.GetMKLDNNData();
    if (weight_mem->get_desc() != fwd->fwd_pd.weights_desc()) {
      weight.MKLDNNDataReorderAsync(fwd->fwd_pd.weights_desc());
      weight_mem = GetWeights(weight, fwd->fwd_pd.weights_desc(), 1);
    }
  }
  auto out_mem = CreateMKLDNNMem(out_data[fullc::kOut],
                                 fwd->fwd_pd.dst_desc(), req[fullc::kOut], &data);

  mkldnn_args_map_t args = {
      {MKLDNN_ARG_SRC, *data_mem},
      {MKLDNN_ARG_WEIGHTS, *weight_mem},
      {MKLDNN_ARG_DST, *out_mem.second},
  };
  if (!full_param.default_param.no_bias) {
    auto bias_mem = in_data[fullc::kBias].GetMKLDNNDataReorder(
        fwd->fwd_pd.bias_desc());
    args[MKLDNN_ARG_BIAS] = *bias_mem;
  }
  MKLDNNStream::Get()->RegisterPrimArgs(fwd->GetFwd(), args);
  CommitOutput(out_data[fullc::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNFCForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data) {
  MKLDNNFCFullParam full_param;
  full_param.default_param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());

  NDArray data = in_data[fullc::kData];
  mkldnn::memory::desc out_md = GetMemDesc(out_data[fullc::kOut]);
  MKLDNNFCFlattenData(full_param.default_param, out_data[fullc::kOut],
                      &data, &out_md);
  auto &fwd = GetFCFwd(full_param.default_param, ctx.is_train, data,
                       in_data[fullc::kWeight],
                       full_param.default_param.no_bias ? nullptr : &in_data[fullc::kBias],
                       out_md);
  std::vector<NDArray> new_inputs;
  if (full_param.default_param.no_bias)
    new_inputs = {data, in_data[fullc::kWeight]};
  else
    new_inputs = {data, in_data[fullc::kWeight], in_data[fullc::kBias]};
  MKLDNNFCForwardFullFeature(full_param, ctx, &fwd, new_inputs, req, out_data);
}

void MKLDNNFCBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  MKLDNNFCFullParam full_param;
  full_param.default_param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());
  const FullyConnectedParam& param = full_param.default_param;
  const mxnet::TShape& ishape = inputs[fullc::kData + 1].shape();
  const mxnet::TShape& oshape = inputs[fullc::kOut].shape();

  NDArray weight = inputs[fullc::kWeight + 1];
  NDArray data = inputs[fullc::kData + 1];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.MKLDNNDataReshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
  else if (data.shape().ndim() != 2)
    data = data.MKLDNNDataReshape(Shape2(ishape[0],
                                     ishape.ProdShape(1, ishape.ndim())));
  NDArray out_grad = inputs[fullc::kOut];
  if (out_grad.shape().ndim() != 2 && !param.flatten)
    out_grad = out_grad.MKLDNNDataReshape(Shape2(oshape.ProdShape(0, oshape.ndim()-1),
                                             oshape[oshape.ndim()-1]));
  else if (out_grad.shape().ndim() != 2)
    out_grad = out_grad.MKLDNNDataReshape(Shape2(oshape[0],
                                             oshape.ProdShape(1, oshape.ndim())));


  mkldnn::inner_product_forward::primitive_desc fwd_pd = GetFCFwdImpl(full_param, ctx.is_train,
      data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias], GetMemDesc(out_grad));

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  if (req[fullc::kWeight]) {
    mkldnn::inner_product_backward_weights::primitive_desc ipBwdWeights_pd
      = GetFCBwdWeights(data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias],
          out_grad, fwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdWeights_pd.diff_dst_desc());
    auto data_mem = data.GetMKLDNNDataReorder(ipBwdWeights_pd.src_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[fullc::kWeight],
                                                 ipBwdWeights_pd.diff_weights_desc(),
                                                 req[fullc::kWeight]);
    mkldnn_args_map_t args = {
      {MKLDNN_ARG_DIFF_DST, *out_grad_mem},
      {MKLDNN_ARG_SRC, *data_mem},
      {MKLDNN_ARG_DIFF_WEIGHTS, *in_grad_weight.second},
    };

    mkldnn_output_t in_grad_bias;
    if (!param.no_bias) {
      in_grad_bias = CreateMKLDNNMem(in_grad[fullc::kBias],
                                     ipBwdWeights_pd.diff_bias_desc(),
                                     req[fullc::kBias]);
      args[MKLDNN_ARG_DIFF_BIAS] = *in_grad_bias.second;
    }
    MKLDNNStream::Get()->RegisterPrimArgs(
        mkldnn::inner_product_backward_weights(ipBwdWeights_pd), args);
    CommitOutput(in_grad[fullc::kWeight], in_grad_weight);
    CommitOutput(in_grad[fullc::kBias], in_grad_bias);
  }
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd = GetFCBwdData(
        data, weight, out_grad, fwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdData_pd.diff_dst_desc());
    auto weight_mem = weight.GetMKLDNNDataReorder(ipBwdData_pd.weights_desc());
    auto in_grad_mem = CreateMKLDNNMem(in_grad[fullc::kData],
                                       ipBwdData_pd.diff_src_desc(),
                                       req[fullc::kData]);
    mkldnn_args_map_t args = {
      {MKLDNN_ARG_DIFF_DST, *out_grad_mem},
      {MKLDNN_ARG_WEIGHTS, *weight_mem},
      {MKLDNN_ARG_DIFF_SRC, *in_grad_mem.second}
    };

    MKLDNNStream::Get()->RegisterPrimArgs(mkldnn::inner_product_backward_data(ipBwdData_pd), args);
    CommitOutput(in_grad[fullc::kData], in_grad_mem);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
