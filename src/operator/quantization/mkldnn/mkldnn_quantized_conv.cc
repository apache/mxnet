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
 * \file mkldnn_quantized_onvolution.cc
 * \brief
 * \author Wenting Jiang
*/

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/convolution-inl.h"
#include "../quantization_utils.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../elemwise_op_common.h"
namespace mxnet {
namespace op {

static mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  }
  if (param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    }
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
                                             data_md, weight_md, bias_md, out_md, strides,
                                             dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, engine);
    }
  }
}

class MKLDNNConvForward {
  std::shared_ptr<mkldnn::convolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, data, weights, bias, output)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());

    if (this->weight == nullptr)
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.weights_primitive_desc(), weight.get_data_handle()));
    else
      this->weight->set_data_handle(weight.get_data_handle());

    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (bias != nullptr) {
      if (this->bias == nullptr)
        this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                fwd_pd.bias_primitive_desc(), bias->get_data_handle()));
      else
        this->bias->set_data_handle(bias->get_data_handle());
      if (this->fwd == nullptr)
        this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
            new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            mkldnn::primitive::at(*this->bias),
                                            *this->out));
    } else if (this->fwd == nullptr) {
      this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
          new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
    }
  }

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd;
  }
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

static inline MKLDNNConvForward &GetConvFwd(
    const nnvm::NodeAttrs& attrs,
    const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MKLDNNConvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvForward fwd(param, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNQuantizedConvForward(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  if (in_data[0].dtype() == mshadow::kUint8) {
    TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
    NDArray weight = in_data[conv::kWeight];
    MKLDNNConvForward &fwd = GetConvFwd(attrs, in_data[conv::kData], weight,
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
  } else {
    LOG(FATAL) << "mkldnn_quantized_conv op only supports uint8 as input type";
  }
}

NNVM_REGISTER_OP(_contrib_quantized_conv)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedConvForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
