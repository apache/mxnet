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
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FULLY_CONNECTED_INL_H_

#include <vector>
#include "quantization_utils.h"
#include "../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {

namespace quantized_fc {
enum QuantilizedfcOpResource {kTempSpace};
}

struct QuantizedSumInitKernelWithBias {
  //  init sum data with bias for matrix b (n)
  MSHADOW_XINLINE static void Map(int i, int32_t *out,
                                  const int8_t *bias, const float *min_out,
                                  const float *max_out, const float *min_bias,
                                  const float *max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    float float_for_one_out_quant  =
      MaxAbs(*min_out, *max_out) / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
      MaxAbs(*min_bias, *max_bias) / static_cast<double>(MaxValue<T2>());
    if (float_for_one_out_quant != 0) {
      out[i] = bias[i] * float_for_one_bias_quant /
               float_for_one_out_quant;
    } else {
      LOG(INFO) << "WARNING: QuantizedBiasAddKernel float_for_one_out_quant is 0 !";
      out[i] = 0;
    }
  }
};
template<typename SrcType>
void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &out_data) {
#if MSHADOW_USE_MKL == 1
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  size_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_data.size(),  num_inputs * 3);
  CHECK_EQ(out_data.size(), 3U);
  const NDArray& data = in_data[0];
  const NDArray& weight = in_data[1];
  const NDArray& out = out_data[0];
  TShape dshape = data.shape();
  TShape wshape = weight.shape();
  TShape oshape = out.shape();
  auto output_temp = out.data().dptr<int32_t>();
  auto weight_temp = weight.data().dptr<SrcType>();
  auto data_temp = data.data().dptr<SrcType>();
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const float alpha = 1.0f;
  const float beta  = 1.0f;
  const CBLAS_OFFSET offsetc = CblasFixOffset;
  const MKL_INT8 oa = 0;
  const MKL_INT8 ob = 0;
  MKL_INT32 oc = 0;
  const int m = dshape[0], n = wshape[0], k = dshape.ProdShape(1, dshape.ndim());
  Stream<cpu> *s = ctx.get_stream<cpu>();
  //  cblas_gemm_s8u8s32 required first matrix must be uint8
  //  shift data from int8(from -128 to 127) to uint8 (from 0 to 255)
  int shift = 128;
  Tensor<cpu, 1, uint8_t> shiftdata =
    ctx.requested[quantized_fc::kTempSpace].get_space_typed<cpu, 1, uint8_t>(
      Shape1(m * k), s);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < m * k; ++i) {
    shiftdata.dptr_[i] = data_temp[i] + shift;
  }

  Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
      out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
      in_data[num_inputs].data().dptr<float>(), in_data[num_inputs+1].data().dptr<float>(),
      in_data[num_inputs+2].data().dptr<float>(), in_data[num_inputs+3].data().dptr<float>());
  if (!param.no_bias) {
    const NDArray& bias = in_data[2];
    Kernel<QuantizedSumInitKernelWithBias, cpu>::Launch(s, n, out.data().dptr<int32_t>(),
        bias.data().dptr<int8_t>(), out_data[1].data().dptr<float>(),
        out_data[2].data().dptr<float>(), in_data[7].data().dptr<float>(),
        in_data[8].data().dptr<float>());
  } else {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < m * n; ++i) {
      output_temp[i] = 0;
    }
  }
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      output_temp[i] -= shift * weight_temp[i * k + j];
    }
  }
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = n; i < m * n; ++i) {
    output_temp[i] = output_temp[i % n];
  }
  cblas_gemm_s8u8s32(CblasRowMajor,
                     CblasNoTrans,
                     CblasTrans,
                     offsetc,
                     m,
                     n,
                     k,
                     alpha,
                     shiftdata.dptr_,
                     k,
                     oa,
                     weight.data().dptr<SrcType>(),
                     k,
                     ob,
                     beta,
                     out.data().dptr<int32_t>(),
                     n,
                     &oc);
#else
  LOG(FATAL) << "s8u8s32 is only supported by MKL BLAS library";
#endif
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.set_attr<FComputeEx>("FComputeEx<cpu>",
    MKLDNNQuantizedFullyConnectedForward<int8_t>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZED_FULLY_CONNECTED_INL_H_
