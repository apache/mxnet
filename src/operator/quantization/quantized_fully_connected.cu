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
#include "./quantized_fully_connected-inl.h"

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

// get quantizedtofloat scale
struct QuantizedToFloatScale {
  MSHADOW_XINLINE static void Map(int i,
                                  const float *min_out,
                                  const float *max_out, 
                                  float *quantized_to_float_scale) {
    typedef int32_t T1;

    float quantized_range = MinAbs(MinValue<T1>(), MaxValue<T1>());
    float real_range = MaxAbs(*min_out, *max_out);
    float scale = real_range / quantized_range;

    // the tmp space to store quantized_to_float_scale is 32 bits (1 float numbers)
    *quantized_to_float_scale = scale;
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
void QuantizedAddBias(BType* bias,
                      Tensor<gpu, 2, BType> data,
                      DType* out,
                      Stream<gpu>* s,
                      const float *float_for_one_quant_tmp,
                      const int bias_len) {
    
    int nthreads_quant_addbias = 256;

    if(bias_len >= 1024){
      nthreads_quant_addbias = 1024;
    }else if(bias_len >= 512){
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
                                  Stream<gpu>::GetStream(s)>>>(out,
                                                                bias,
                                                                bias_len,
                                                                float_for_one_quant_tmp);
}

// DType->output type, BType->bias type
template <typename DType, typename BType, typename DLoadType>
__global__ void quantized_add_bias_redequantize_kernel(int32_t* outCUBLAS,
                                          DType* out,
                                          BType* bias,
                                          size_t bias_length,
                                          const float *float_for_one_quant_tmp,
                                          const float *quantized_to_float_scale) {
                                          
  int row_num_each_thread = 4; // number of rows to deal with for each thread;

  int64_t* outCUBLASload = reinterpret_cast<int64_t*>(outCUBLAS);
  int16_t* biasload = reinterpret_cast<int16_t*>(bias);
  DLoadType* outload = reinterpret_cast<DLoadType*>(out);

  for (index_t i = threadIdx.x; i < bias_length / 2; i += blockDim.x){
    
    int16_t scratch_bias = *(biasload + i);
    int8_t* scratch_bias_aft_load = reinterpret_cast<int8_t*>(&scratch_bias);
  
  #pragma unroll
    for(int rw = 0; rw < row_num_each_thread; rw++){
      int idx = (blockIdx.x * row_num_each_thread + rw) * bias_length / 2 + i;

      int64_t scratch_outCUBLAS = *(outCUBLASload + idx);
      int32_t* scratch_outCUBLAS_aft_load = reinterpret_cast<int32_t*>(&scratch_outCUBLAS);

      DLoadType scratch_out = *(outload + idx);
      DType* scratch_out_aft_load = reinterpret_cast<DType*>(&scratch_out);
    
      scratch_out_aft_load[0] = ( scratch_outCUBLAS_aft_load[0] + scratch_bias_aft_load[0] * (*float_for_one_quant_tmp) )
        * (*quantized_to_float_scale);
      scratch_out_aft_load[1] = ( scratch_outCUBLAS_aft_load[1] + scratch_bias_aft_load[1] * (*float_for_one_quant_tmp) )
        * (*quantized_to_float_scale);

      *(outload + idx) = scratch_out;
    }     

  }

}

template<typename DType, typename BType>
void FusedQuantizedAddBiasAndReDequantize(BType* bias,
                      Tensor<gpu, 2, BType> data,
                      DType* out,
                      Tensor<gpu, 2, int32_t> outTensorCUBLAS,
                      Stream<gpu>* s,
                      const float *float_for_one_quant_tmp,
                      const int bias_len,
                      const float *quantized_to_float_scale) {
    
    int nthreads_quant_addbias = 256;

    if(bias_len % 512 == 0){
      nthreads_quant_addbias = 512;
    }else if(bias_len % 256 == 0){
      nthreads_quant_addbias = 256;
    }else if(bias_len % 128 == 0){
      nthreads_quant_addbias = 128;
    }else if(bias_len % 64 == 0){
      nthreads_quant_addbias = 64;
    }
    
    if(bias_len <= 32){
      nthreads_quant_addbias = 32;
    }

    int dltype;
    if(sizeof(DType) == 4){ // DType is fp32
      dltype = mshadow::kFloat64;
    }else if(sizeof(DType) == 2){ // Dtype is fp16
      dltype = mshadow::kFloat32;
    }else{
      LOG(FATAL) << "Unsupported float_out type.";
    }

    MXNET_LOAD_TYPE_SWITCH(dltype, DLoadType, {
    quantized_add_bias_redequantize_kernel<DType, BType, DLoadType><<<data.size(0) / 4,
                                  nthreads_quant_addbias,
                                  0,
                                  Stream<gpu>::GetStream(s)>>>(outTensorCUBLAS.dptr_,
                                                                out,
                                                                bias,
                                                                bias_len,
                                                                float_for_one_quant_tmp,
                                                                quantized_to_float_scale);
    });
}

template <typename DType, typename BType>
__global__ void fused_requantize_dequantize_kernel(int32_t* outCUBLAS,
                                          DType* out,
                                          const int num_elem,
                                          const float *quantized_to_float_scale) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < num_elem){
    out[tid] = outCUBLAS[tid] * (*quantized_to_float_scale);
    tid += blockDim.x * gridDim.x;
  }
}

template<typename DType, typename BType>
void FusedRequantizdAndDequantize(DType* out,
                      Tensor<gpu, 2, int32_t> outTensorCUBLAS,
                      Stream<gpu>* s,
                      const float *quantized_to_float_scale,
                      const int num_elem) {

    int nthreads = 256;

    if(num_elem >= 512){
      nthreads = 512;
    }else if(num_elem >= 256){
      nthreads = 256;
    }else if(num_elem >= 128){
      nthreads = 128;
    }else if(num_elem >= 64){
      nthreads = 64;
    }else{
      nthreads = 32;
    }
    
    fused_requantize_dequantize_kernel<DType, BType><<< (num_elem + nthreads - 1) / nthreads,
                                  nthreads,
                                  0,
                                  Stream<gpu>::GetStream(s)>>>(outTensorCUBLAS.dptr_,
                                                                out,
                                                                num_elem,
                                                                quantized_to_float_scale);
}

#endif  // __CUDACC__

template<typename SrcType>
void QuantizedFullyConnectedForwardGPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
#if CUDA_VERSION >= 8000
  typedef int32_t CmpType;

  const QuantizedFullyConnectedParam& param = nnvm::get<QuantizedFullyConnectedParam>(attrs.parsed);
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

  auto out_type = GetQuantizedFCFloatOutType(param);

  // allocate workspace for storaging both outTensorCUBLAS and FloatForOneQuant and QuantizedToFloatScaleFactor
  size_t workspace_size = sizeof(int32_t) * out.Size() + sizeof(float) + sizeof(float);
  auto workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  char* ptr = workspace.dptr_;

  Tensor<gpu, 2, SrcType> dataTensor;
  Tensor<gpu, 2, int32_t> outTensorCUBLAS;
  Tensor<gpu, 1, float> FloatForOneQuant;
  Tensor<gpu, 1, float> QuantizedToFloatScaleFactor;

  if(out_type == mshadow::kInt32){
    if (!param.flatten) {
      dataTensor = FlattenAs2DHead<gpu, SrcType>(data, ctx);
      outTensorCUBLAS = FlattenAs2DHead<gpu, int32_t>(out, ctx);
    } else {
      dataTensor = FlattenAs2DTail<gpu, SrcType>(data, ctx);
      outTensorCUBLAS = FlattenAs2DTail<gpu, int32_t>(out, ctx);
    }
    // workspace utilizing the first element for FloatForOneQuant
    FloatForOneQuant = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
    ptr += sizeof(float);
    // workspace: QuantizedToFloatScaleFactor
    QuantizedToFloatScaleFactor = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
  }else if(out_type == mshadow::kFloat32){
    Tensor<gpu, 2, float> outTensor;
    if (!param.flatten) {
      dataTensor = FlattenAs2DHead<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DHead<gpu, float>(out, ctx);
    } else {
      dataTensor = FlattenAs2DTail<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DTail<gpu, float>(out, ctx);
    }
    // workspace: temporary storage for output tensor, which is in int32 type for storaging CUBLAS output
    outTensorCUBLAS = Tensor<gpu, 2, int32_t>(reinterpret_cast<int32_t*>(ptr), outTensor.shape_, s);
    ptr += sizeof(int32_t) * out.Size();
    // workspace: FloatForOneQuant
    FloatForOneQuant = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
    ptr += sizeof(float);
    // workspace: QuantizedToFloatScaleFactor
    QuantizedToFloatScaleFactor = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
  }else if(out_type == mshadow::kFloat16){
    Tensor<gpu, 2, mshadow::half::half_t> outTensor;
    if (!param.flatten) {
      dataTensor = FlattenAs2DHead<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DHead<gpu, mshadow::half::half_t>(out, ctx);
    } else {
      dataTensor = FlattenAs2DTail<gpu, SrcType>(data, ctx);
      outTensor = FlattenAs2DTail<gpu, mshadow::half::half_t>(out, ctx);
    }
    // workspace: temporary storage for output tensor, which is in int32 type for storaging CUBLAS output
    outTensorCUBLAS = Tensor<gpu, 2, int32_t>(reinterpret_cast<int32_t*>(ptr), outTensor.shape_, s);
    ptr += sizeof(int32_t) * out.Size();
    // workspace: FloatForOneQuant
    FloatForOneQuant = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
    ptr += sizeof(float);
    // workspace: QuantizedToFloatScaleFactor
    QuantizedToFloatScaleFactor = Tensor<gpu, 1, float>(reinterpret_cast<float*>(ptr), Shape1(1), s);
  }else{
    LOG(FATAL) << "Unsupported float_out in params: " <<param.float_out;
  }

  Tensor<gpu, 2, SrcType> weightTensor = weight.get<gpu, 2, SrcType>(s);

  // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
  // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
 
  // A->dataTensor, B->weightTensor, C->outTensor
  CmpType alpha = 1.0f;
  CmpType beta  = 0.0f;
  const cudaDataType src_type = mshadow::DataType<SrcType>::kCudaFlag;
  const cudaDataType dst_type = mshadow::DataType<int32_t>::kCudaFlag;
  const cudaDataType cmp_type = mshadow::DataType<CmpType>::kCudaFlag;
  CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           outTensorCUBLAS.size(1),
                           outTensorCUBLAS.size(0),
                           weightTensor.size(1),
                           &alpha,
                           weightTensor.dptr_,
                           src_type,
                           weightTensor.stride_,
                           dataTensor.dptr_,
                           src_type,
                           dataTensor.stride_,
                           &beta,
                           outTensorCUBLAS.dptr_,
                           dst_type,
                           outTensorCUBLAS.stride_,
                           cmp_type,
                           CUBLAS_GEMM_DFALT));
  

  // use min/max values of weight and data to update the min/max values of output
  Kernel<QuantizationRangeForS8S8MultiplicationStruct, gpu>::Launch(s, 1,
    outputs[1].dptr<float>(), outputs[2].dptr<float>(),
     inputs[num_inputs].dptr<float>(),   inputs[num_inputs+1].dptr<float>(),
     inputs[num_inputs+2].dptr<float>(), inputs[num_inputs+3].dptr<float>());

  // Launch QuantizedToFloatScale
    Kernel<QuantizedToFloatScale, gpu>::Launch(s, 1,
      outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      QuantizedToFloatScaleFactor.dptr_);

  if (!param.no_bias) {
    const TBlob& bias = inputs[2];

    Tensor<gpu, 1, SrcType> biasTensor = bias.get_with_shape<gpu, 1, SrcType>(Shape1(wshape[0]), s);
    CHECK_EQ(biasTensor.shape_[0], wshape[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";

    // Launch FloatForOneQuantBiasAddKernel
    Kernel<FloatForOneQuantBiasAddKernel, gpu>::Launch(s, 1,
      outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[7].dptr<float>(), inputs[8].dptr<float>(),
      FloatForOneQuant.dptr_);

    // with_bias case without float out
    if(out_type == mshadow::kInt32){
      QuantizedAddBias<int32_t, SrcType>(bias.dptr<SrcType>(), dataTensor, out.dptr<int32_t>(), s,
                  FloatForOneQuant.dptr_, biasTensor.shape_[0]);
    }else if(out_type == mshadow::kFloat32){// with_bias case with float32 out
      // a kernel that fuse requantize, dequantize into add_bias
      FusedQuantizedAddBiasAndReDequantize<float, SrcType>(bias.dptr<SrcType>(), dataTensor,
                                          out.dptr<float>(), outTensorCUBLAS, s,
                                          FloatForOneQuant.dptr_, biasTensor.shape_[0],
                                          QuantizedToFloatScaleFactor.dptr_);
    }else if(out_type == mshadow::kFloat16){// with_bias case with float16 out
      // a kernel that fuse requantize, dequantize into add_bias
      FusedQuantizedAddBiasAndReDequantize<mshadow::half::half_t, SrcType>(bias.dptr<SrcType>(), dataTensor,
                                          out.dptr<mshadow::half::half_t>(), outTensorCUBLAS, s,
                                          FloatForOneQuant.dptr_, biasTensor.shape_[0],
                                          QuantizedToFloatScaleFactor.dptr_);
    }else{
      LOG(FATAL) << "Unsupported float_out in params: " <<param.float_out;
    }
  }else{
    // no_bias case with float out
    if(out_type == mshadow::kFloat32){// no_bias case with float32 out
      FusedRequantizdAndDequantize<float, SrcType>(out.dptr<float>(),
                                          outTensorCUBLAS, s,
                                          QuantizedToFloatScaleFactor.dptr_,
                                          out.Size());
    }else if(out_type == mshadow::kFloat16){// no_bias case with float16 out
      FusedRequantizdAndDequantize<mshadow::half::half_t, SrcType>(out.dptr<mshadow::half::half_t>(),
                                          outTensorCUBLAS, s,
                                          QuantizedToFloatScaleFactor.dptr_,
                                          out.Size());
    }else if(out_type != mshadow::kInt32){
      LOG(FATAL) << "Unsupported float_out in params: " <<param.float_out;
    }
  }
#else
  LOG(FATAL) << "QuantizedFullyConnectedForwardGPU only supports CUDA >= 8.0";
#endif  // CUDA_VERSION >= 8000
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.set_attr<FCompute>("FCompute<gpu>", QuantizedFullyConnectedForwardGPU<int8_t>);

}  // namespace op
}  // namespace mxnet