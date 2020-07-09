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

// get float_for_one_out_quant and float_for_one_bias_quant outside QuantizedBiasAddKernel
struct FloatForOneQuantBiasAddKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  const float *min_out,
                                  const float *max_out, 
                                  const float *min_bias,
                                  const float *max_bias,
                                  float *float_for_one_quant_tmp) {
    typedef int32_t T1;
    typedef int8_t  T2;
    
    float float_for_one_quant_out = FloatForOneQuantizedLevel<T1>(*min_out, *max_out, true);
    float float_for_one_quant_bias = FloatForOneQuantizedLevel<T2>(*min_bias, *max_bias, true);

    // the tmp space to store float_for_one_quant is 32 bits (1 float numbers)
    *float_for_one_quant_tmp = float_for_one_quant_bias / float_for_one_quant_out;
  }
};
#endif  // CUDA_VERSION >= 8000

#if defined(__CUDACC__)

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
// DType->output matrix type, BType->bias type
template <typename DType, typename BType>
__global__ void quantized_add_bias_kernel(DType* mat,
                                          BType* bias,
                                          size_t bias_length,
                                          const float *float_for_one_quant_tmp) {

const index_t row = blockIdx.x;
for (index_t i = threadIdx.x; i < bias_length; i += blockDim.x){
  int idx = row*bias_length + i;
  mat[idx] += bias[i] * (*float_for_one_quant_tmp);
}

}

template<typename DType, typename BType>
void QuantizedAddBias(Tensor<gpu, 1, BType> bias,
                      Tensor<gpu, 2, BType> data,
                      Tensor<gpu, 2, DType> out,
                      Stream<gpu>* s,
                      const float *float_for_one_quant_tmp) {
    
    int bias_len = bias.shape_[0];

    int nthreads_quant_addbias = 256;
    if(bias_len >= 512){
      nthreads_quant_addbias = 512;
    }else if(bias_len >= 256){
      nthreads_quant_addbias = 256;
    }else if(bias_len >= 128){
      nthreads_quant_addbias = 128;
    }else if(bias_len >= 64){
      nthreads_quant_addbias = 64;
    }else{
      nthreads_quant_addbias = 32;
    }

    quantized_add_bias_kernel<DType, BType><<<data.size(0),
                                  nthreads_quant_addbias,
                                  0,
                                  Stream<gpu>::GetStream(s)>>>(out.dptr_,
                                                                bias.dptr_,
                                                                bias_len,
                                                                float_for_one_quant_tmp);
}

#endif  // __CUDACC__

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
  mxnet::TShape dshape = data.shape_;
  mxnet::TShape wshape = weight.shape_;
  mxnet::TShape oshape = out.shape_;
  // (m, n) * (k, n).T = (m, k)
  // A * B.T = C

  Tensor<gpu, 2, SrcType> dataTensor;
  Tensor<gpu, 2, DstType> outTensor;

  if (!param.flatten) {
    dataTensor = FlattenAs2DHead<gpu, SrcType>(data, ctx);
    outTensor = FlattenAs2DHead<gpu, DstType>(out, ctx);
  } else {
    dataTensor = FlattenAs2DTail<gpu, SrcType>(data, ctx);
    outTensor = FlattenAs2DTail<gpu, DstType>(out, ctx);
  }

  Tensor<gpu, 2, SrcType> weightTensor = weight.get<gpu, 2, SrcType>(s);

  // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
  // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
 
  // A->dataTensor, B->weightTensor, C->outTensor
  CmpType alpha = 1.0f;
  CmpType beta  = 0.0f;
  const cudaDataType src_type = mshadow::DataType<SrcType>::kCudaFlag;
  const cudaDataType dst_type = mshadow::DataType<DstType>::kCudaFlag;
  const cudaDataType cmp_type = mshadow::DataType<CmpType>::kCudaFlag;
  CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           outTensor.size(1),
                           outTensor.size(0),
                           weightTensor.size(1),
                           &alpha,
                           weightTensor.dptr_,
                           src_type,
                           weightTensor.stride_,
                           dataTensor.dptr_,
                           src_type,
                           dataTensor.stride_,
                           &beta,
                           outTensor.dptr_,
                           dst_type,
                           outTensor.stride_,
                           cmp_type,
                           CUBLAS_GEMM_DFALT));

  // use min/max values of output and data to update the min/max values of weight
  Kernel<QuantizationRangeForS8S8MultiplicationStruct, gpu>::Launch(s, 1,
    outputs[1].dptr<float>(), outputs[2].dptr<float>(),
     inputs[num_inputs].dptr<float>(),   inputs[num_inputs+1].dptr<float>(),
     inputs[num_inputs+2].dptr<float>(), inputs[num_inputs+3].dptr<float>());

  if (!param.no_bias) {
    const TBlob& bias = inputs[2];

    Tensor<gpu, 1, SrcType> biasTensor = bias.get_with_shape<gpu, 1, SrcType>(Shape1(wshape[0]), s);
    CHECK_EQ(biasTensor.shape_[0], wshape[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";

    // Launch FloatForOneQuantBiasAddKernel
    // temporary storage for FloatForOneQuant values
    size_t temp_bytes = sizeof(float);
    Tensor<gpu, 1, float> FloatForOneQuant =
      ctx.requested[0].get_space_typed<gpu, 1, float>(
        Shape1(temp_bytes), s);

    Kernel<FloatForOneQuantBiasAddKernel, gpu>::Launch(s, 1,
      outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[7].dptr<float>(), inputs[8].dptr<float>(),
      FloatForOneQuant.dptr_);

    QuantizedAddBias<DstType, SrcType>(biasTensor, dataTensor, outTensor, s,
                  FloatForOneQuant.dptr_); 
  }
#else
  LOG(FATAL) << "QuantizedFullyConnectedForwardGPU only supports CUDA >= 8.0";
#endif  // CUDA_VERSION >= 8000
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.set_attr<FCompute>("FCompute<gpu>", QuantizedFullyConnectedForwardGPU<int8_t, int32_t, int32_t>);

}  // namespace op
}  // namespace mxnet