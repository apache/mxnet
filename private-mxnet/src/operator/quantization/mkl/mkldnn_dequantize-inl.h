/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_quantized_conv-inl.h
* \brief
* \author young.jin.kim@intel.com
*         deepthi.karkada@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_DNN_MKLDNN_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_MKL_DNN_MKLDNN_DEQUANTIZE_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../mkl/mkldnn_base-inl.h"
#include "../dequantize-inl.h"

namespace mxnet {
namespace op {


template<typename SrcType, typename DstType>
void MKLDequantizeComputeKer(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<cpu> *s = ctx.get_stream<cpu>();

  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);

  // std::cout << "dequantize !!!!!" << std::endl;
  
  // check shapes
  int i_dim = inputs[0].ndim();
  int o_dim = outputs[0].ndim();

  CHECK_EQ(i_dim, o_dim);

  int total_len = 0;

  // std::vector<index_t> shape;
  memory::dims tensor_shape;

  for (size_t ii = 0; ii < i_dim; ++ii) {
    CHECK_EQ(inputs[0].size(ii), outputs[0].size(ii));
    total_len += inputs[0].size(ii);
  }
  tensor_shape.push_back(total_len);

  // find a exponents from the min-max values
  float quantized_range = MaxAbs(MaxValue<SrcType>(), MinValue<SrcType>());
  float real_range = MaxAbs(*inputs[1].dptr<DstType>(), *inputs[2].dptr<DstType>());

  //int exp_input = (int)log2(scale);
  int exp_input = param.shift_exponent;


  // std::cout << "dequantize done 22222!!!!!" << std::endl;

  quantization_set_round_mode(down);
  quantization_set_dfp_exp(exp_input, exp_fixed, 0, exp_fixed);


  //memory::format mfmt_any = memory::format::any;
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

  // std::cout << "dequantize done 444444!!!!!" << std::endl;

  auto i_mpd = memory::primitive_desc({tensor_shape, 
                                      (mkldnn::memory::data_type)data_type_enum<SrcType>::type, 
                                       memory::format::x}, 
                                       cpu_engine);
  auto o_mpd = memory::primitive_desc({tensor_shape, 
                                      (mkldnn::memory::data_type)data_type_enum<DstType>::type, 
                                      memory::format::x}, 
                                      cpu_engine);

  // std::cout << "dequantize done 5555555!!!!!" << std::endl;

  auto input = memory(i_mpd, inputs[0].dptr<SrcType>());
  auto output = memory(o_mpd, outputs[0].dptr<DstType>());

  // std::cout << "dequantize done 66666!!!!!" << std::endl;

  auto r = reorder(input, output);
  stream(stream::kind::lazy).submit({r}).wait();

  // std::cout << "dequantize done!!!!!" << std::endl;
  
}

void MKLDequantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {

  // std::cout << "quantize: " << param.out_type << std::endl;

  // auto out_type = param.out_type;
  // if (attrs.scalars.size() > 0)
  // {
  //   std::cout << "test param: " << attrs.scalars[attrs.scalars.size() - 1] << std::endl;
  //   out_type = attrs.scalars[attrs.scalars.size() - 1];
  // }

  // if (out_type == mshadow::kInt8) {
    MKLDequantizeComputeKer<uint8_t, float>(attrs,
                                         ctx,
                                         inputs,
                                         req,
                                         outputs
                                         );
  // } else if (out_type == mshadow::kUint8) {
  //   MKLQuantizeComputeKer<float, uint8_t>(attrs,
  //                                        ctx,
  //                                        inputs,
  //                                        req,
  //                                        outputs
  //                                        );
  // }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_DEQUANTIZE_INL_H_
