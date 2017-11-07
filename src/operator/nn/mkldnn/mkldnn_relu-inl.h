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
 * \file mkldnn_relu-inl.h
 * \brief
 * \author Da Zheng
*/

#ifndef MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>

namespace mxnet {
namespace op {

template<typename Dtype>
void MKLDNNRelu_Forward(const OpContext &ctx, const NDArray &in_data,
    const OpReqType &req, const NDArray &out_data) {
  std::shared_ptr<const mkldnn::memory> input_mem = in_data.GetMKLDNNData();
  mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  auto cpu_engine = data_mpd.get_engine();
  Dtype alpha = 0;

  mkldnn::eltwise_forward::desc desc = ctx.is_train
    ? mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
        mkldnn::eltwise_relu, data_md, alpha)
    : mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_scoring,
        mkldnn::eltwise_relu, data_md, alpha);
  mkldnn::eltwise_forward::primitive_desc pdesc(desc, cpu_engine);

  std::shared_ptr<const mkldnn::memory> output_memory
    = const_cast<NDArray &>(out_data).CreateMKLDNNData(pdesc.dst_primitive_desc());
  MKLDNNStream &stream = MKLDNNStream::Instance();
  stream.RegisterPrim(mkldnn::eltwise_forward(pdesc, *input_mem, *output_memory));
  stream.Submit();
}

template<typename Dtype>
void MKLDNNRelu_Backward(const OpContext &ctx, const NDArray &out_grad,
                const NDArray &in_data, const OpReqType &req,
                const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }

  std::shared_ptr<const mkldnn::memory> diff_dst_memory = out_grad.GetMKLDNNData();
  std::shared_ptr<const mkldnn::memory> input_mem = in_data.GetMKLDNNData();
  mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  mkldnn::memory::desc diff_md = diff_dst_memory->get_primitive_desc().desc();
  auto cpu_engine = data_mpd.get_engine();
  Dtype alpha = 0;

  mkldnn::eltwise_forward::desc fw_desc(mkldnn::prop_kind::forward_training,
      mkldnn::eltwise_relu, data_md, alpha);
  mkldnn::eltwise_forward::primitive_desc fw_pdesc(fw_desc, cpu_engine);
  mkldnn::eltwise_backward::desc bw_desc(mkldnn::eltwise_relu, diff_md, data_md, alpha);
  mkldnn::eltwise_backward::primitive_desc bw_pdesc(bw_desc, cpu_engine, fw_pdesc);

  auto diff_src_memory = CreateMKLDNNMem(in_grad, bw_pdesc.diff_src_primitive_desc(), req);
  MKLDNNStream &stream = MKLDNNStream::Instance();
  stream.RegisterPrim(mkldnn::eltwise_backward(bw_pdesc, *input_mem,
        *diff_dst_memory, *diff_src_memory.second));
  CommitOutput(in_grad, diff_src_memory);
  stream.Submit();
}

}  // namespace op
}  // namespace mxnet

#endif
#endif  // MXNET_OPERATOR_MKL_MKLDNN_RELU_INL_H_
