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
 * \file dnnl_quantized_conv.cc
 * \brief
 * \author Wenting Jiang, Xinyu Chen
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/elemwise_op_common.h"
#include "operator/nn/convolution-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_convolution-inl.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

static void DNNLQuantizedConvForward(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<NDArray>& in_data,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  NDArray weight         = in_data[conv::kWeight];
  ConvolutionParam param = nnvm::get<ConvolutionParam>(attrs.parsed);
  DNNLConvFullParam full_param;
  full_param.conv_param = param;
  full_param.dnnl_param.Init(std::unordered_map<std::string, std::string>());
  auto& fwd         = GetConvFwd(full_param,
                         ctx.is_train,
                         in_data[conv::kData],
                         in_data[conv::kWeight],
                         param.no_bias ? nullptr : &in_data[conv::kBias],
                         out_data[conv::kOut]);
  auto fwd_src_desc = fwd.GetPd().src_desc();
  auto data_mem     = in_data[conv::kData].GetDNNLDataReorder(&fwd_src_desc);
  const dnnl::memory* weight_mem;
  // For inference, we want to reorder the weight array so we don't need to
  // reorder data every time.
  if (weight.IsDefaultData()) {
    // We also need to modify the layout on the original weight array.
    // Don't switch below sequence because naive engine will executes
    // pushAsync synchronously.
    auto fwd_weight_desc = fwd.GetPd().weights_desc();
    weight.DNNLDataReorderAsync(&fwd_weight_desc);
    weight_mem = GetWeights(weight, fwd_weight_desc, param.num_group);
  } else {
    weight_mem = weight.GetDNNLData();
  }
  auto out_mem = CreateDNNLMem(out_data[conv::kOut], fwd.GetPd().dst_desc(), req[conv::kOut]);
  dnnl_args_map_t net_args;
  if (!param.no_bias) {
    auto fwd_bias_desc           = fwd.GetPd().bias_desc();
    const dnnl::memory* bias_mem = in_data[conv::kBias].GetDNNLDataReorder(&fwd_bias_desc);
    net_args.insert({DNNL_ARG_BIAS, *bias_mem});
  }
  net_args.insert({DNNL_ARG_SRC, *data_mem});
  net_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
  net_args.insert({DNNL_ARG_DST, *out_mem.second});
  DNNLStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[conv::kOut], out_mem);
  DNNLStream::Get()->Submit();
  Stream<cpu>* s          = ctx.get_stream<cpu>();
  const size_t num_inputs = param.no_bias ? 2 : 3;
  mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
      s,
      1,
      out_data[1].data().dptr<float>(),
      out_data[2].data().dptr<float>(),
      in_data[num_inputs].data().dptr<float>(),
      in_data[num_inputs + 1].data().dptr<float>(),
      in_data[num_inputs + 2].data().dptr<float>(),
      in_data[num_inputs + 3].data().dptr<float>());
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedConvForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
