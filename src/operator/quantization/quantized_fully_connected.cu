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
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cu
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include "./quantization_utils.h"
#include "../mxnet_op.h"
#include "../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {

#if CUDA_VERSION >= 8000
// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
struct QuantizedBiasAddKernel {
  MSHADOW_XINLINE static void Map(int i, size_t k, int32_t *out,
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
    out[i] = (out[i] * float_for_one_out_quant +
              bias[i%k] * float_for_one_bias_quant) /
             float_for_one_out_quant;
  }
};
#endif  // CUDA_VERSION >= 8000

template<typename SrcType, typename DstType, typename CmpType>
void QuantizedFullyConnectedForwardGPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
#if CUDA_VERSION >= 8000
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  size_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(),  num_inputs * 3);
  CHECK_EQ(outputs.size(), 3U);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);
  const TBlob& data   =  inputs[0];
  const TBlob& weight =  inputs[1];
  const TBlob& out    = outputs[0];
  TShape dshape = data.shape_;
  TShape wshape = weight.shape_;
  TShape oshape = out.shape_;
  // (m, n) * (k, n).T = (m, k)
  // A * B.T = C

  // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
  // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
  const int m = dshape[0], n = dshape.ProdShape(1, dshape.ndim()), k = wshape[0];
  CmpType alpha = 1.0f;
  CmpType beta  = 0.0f;
  const cudaDataType src_type = mshadow::DataType<SrcType>::kCudaFlag;
  const cudaDataType dst_type = mshadow::DataType<DstType>::kCudaFlag;
  const cudaDataType cmp_type = mshadow::DataType<CmpType>::kCudaFlag;
  CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           k,
                           m,
                           n,
                           &alpha,
                           weight.dptr_,
                           src_type,
                           n,
                           data.dptr_,
                           src_type,
                           n,
                           &beta,
                           out.dptr_,
                           dst_type,
                           k,
                           cmp_type,
                           CUBLAS_GEMM_DFALT));

  Kernel<QuantizationRangeForMultiplicationStruct, gpu>::Launch(s, 1,
    outputs[1].dptr<float>(), outputs[2].dptr<float>(),
     inputs[num_inputs].dptr<float>(),   inputs[num_inputs+1].dptr<float>(),
     inputs[num_inputs+2].dptr<float>(), inputs[num_inputs+3].dptr<float>());

  if (!param.no_bias) {
    const TBlob& bias = inputs[2];
    Kernel<QuantizedBiasAddKernel, gpu>::Launch(s, out.Size(),
        k, out.dptr<int32_t>(), bias.dptr<int8_t>(),
        outputs[1].dptr<float>(), outputs[2].dptr<float>(),
         inputs[7].dptr<float>(),  inputs[8].dptr<float>());
  }
#else
  LOG(FATAL) << "QuantizedFullyConnectedForwardGPU only supports CUDA >= 8.0";
#endif  // CUDA_VERSION >= 8000
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.set_attr<FCompute>("FCompute<gpu>", QuantizedFullyConnectedForwardGPU<int8_t, int32_t, int32_t>);

}  // namespace op
}  // namespace mxnet
