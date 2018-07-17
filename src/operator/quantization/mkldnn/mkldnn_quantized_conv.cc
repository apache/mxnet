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
 * \file mkldnn_quantized_conv.cc
 * \brief
 * \author Wenting Jiang, Xinyu Chen
*/

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_convolution-inl.h"
#include "../../nn/convolution-inl.h"
#include "../quantization_utils.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../elemwise_op_common.h"
namespace mxnet {
namespace op {

static void MKLDNNQuantizedConvForward(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<NDArray> &in_data,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<NDArray> &out_data) {
  CHECK_EQ(in_data[0].dtype(), mshadow::kUint8)
    << "mkldnn_quantized_conv op only supports uint8 as input type";
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  NDArray weight = in_data[conv::kWeight];
  MKLDNNConvForward &fwd = GetConvFwd(attrs, ctx.is_train,
      in_data[conv::kData], weight,
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);

  auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
  const mkldnn::memory *weight_mem;
  // For inference, we want to reorder the weight array so we don't need to
  // reorder data every time.
  if (weight.IsDefaultData()) {
    weight_mem = GetWeights(weight, fwd.fwd_pd.weights_primitive_desc(), param.num_group);
    // We also need to modify the layout on the original weight array. The
    // data conversion happens after the weight array is used.
    weight.MKLDNNDataReorderAsync(fwd.fwd_pd.weights_primitive_desc());
  } else {
    weight_mem = weight.GetMKLDNNData();
    CHECK(weight_mem->get_primitive_desc() == fwd.fwd_pd.weights_primitive_desc());
  }
  auto out_mem = CreateMKLDNNMem(out_data[conv::kOut], fwd.fwd_pd.dst_primitive_desc(),
                                 req[conv::kOut]);
  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias)
    bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(fwd.fwd_pd.bias_primitive_desc());
  fwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const size_t num_inputs = param.no_bias ? 2 : 3;
  mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
           out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
           in_data[num_inputs].data().dptr<float>(),
           in_data[num_inputs+1].data().dptr<float>(),
           in_data[num_inputs+2].data().dptr<float>(),
           in_data[num_inputs+3].data().dptr<float>());
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedConvForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
