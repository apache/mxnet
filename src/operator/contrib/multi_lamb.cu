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
 *  Copyright (c) 2019 by Contributors
 * \file multi_lamb.cu
 * \brief vectorized lamb coefficient computed from sums of squared weights and grads
 * \author Moises Hernandez
 */

#include "./multi_lamb-inl.h"

namespace mxnet {
namespace op {

#define BLOCK_SIZE_LAMB 512
#define ILP_LAMB 4
    
template<bool has_mixed_precision, typename MPDType, typename DType>
__global__ void kernel_step1(const MultiLAMBKernelParam<DType, MPDType> kernel_params,
                             const float learning_rate, 
                             const float beta1, const float beta2, 
                             const float epsilon,
                             const float wd,
                             const int step,
                             const float clip_gradient,
                             const bool bias_correction, 
                             const float rescale_grad) {

  const size_t tensorID = blockIdx.x;
  const size_t posTensor = threadIdx.x;
  
  for(size_t i=posTensor; i<kernel_params.sizes[tensorID]; i+= blockDim.x){
    MPDType w = has_mixed_precision ? kernel_params.weights32[tensorID][i]:
                                      MPDType(kernel_params.weights[tensorID][i]);
    MPDType scaled_grad = static_cast<MPDType>(kernel_params.grads[tensorID][i])*rescale_grad;
    if (clip_gradient >= 0.0f)
      scaled_grad = max(min(scaled_grad, clip_gradient), -clip_gradient);

    MPDType mean = static_cast<MPDType>(beta1) * kernel_params.mean[tensorID][i] + 
      (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta1)) * scaled_grad;
    MPDType var = static_cast<MPDType>(beta2) * kernel_params.var[tensorID][i] + 
      (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta2)) * scaled_grad * scaled_grad;
    kernel_params.mean[tensorID][i]=mean;
    kernel_params.var[tensorID][i]=var;

    MPDType g;
    if(bias_correction){
      MPDType mean_hat = mean / (static_cast<MPDType>(1.0f) - pow(static_cast<MPDType>(beta1), static_cast<MPDType>(step)));
      MPDType var_hat = var / (static_cast<MPDType>(1.0f) - pow(static_cast<MPDType>(beta2), static_cast<MPDType>(step)));
      g = mean_hat / (sqrt(var_hat) + epsilon) + wd * w;
    }else{
      g = mean / (sqrt(var) + epsilon) + wd * w;
    }
    kernel_params.temp_g[tensorID][i]=g;
  }
}
    
template<bool has_mixed_precision, typename MPDType, typename DType>
__global__ void kernel_step2(const MultiLAMBKernelParam<DType, MPDType> kernel_params,
                             const float* sumSqWeigths,
                             const float* sumSqtemp_g,
                             const float learning_rate,
                             const float lower_bound, 
                             const float upper_bound,
                             const OpReqType req) {

  const size_t tensorID = blockIdx.x;
  const size_t posTensor = threadIdx.x;
  
  for(size_t i=posTensor; i<kernel_params.sizes[tensorID]; i+= blockDim.x){
    MPDType w = has_mixed_precision ? kernel_params.weights32[tensorID][i]:
                                      MPDType(kernel_params.weights[tensorID][i]);
    float r1 = sqrt(sumSqWeigths[tensorID]);
    float r2 = sqrt(sumSqtemp_g[tensorID]);
      
    r1 = min(max(r1, lower_bound), upper_bound);
      
    // calculate lamb_trust_ratio
    MPDType r;
    if (r1 == 0.0f || r2 == 0.0f)
      r = 1.0f;
    else
      r = r1/r2;
    MPDType lr_adjusted = learning_rate * r;
    w -= lr_adjusted * kernel_params.temp_g[tensorID][i];

    // update weights
    if (has_mixed_precision)
      kernel_params.weights32[tensorID][i] = w;
    KERNEL_ASSIGN(kernel_params.out_data[tensorID][i], req, w);
  }
}


template<typename MPDType, bool has_mixed_precision>
struct MultiLAMB_step1_kernelg {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                                  const float learning_rate, 
                                  const float beta1, const float beta2, 
                                  const float epsilon,
                                  const float wd,
                                  const int step,
                                  const float clip_gradient,
                                  const bool bias_correction, 
                                  const float rescale_grad) {
    using namespace mshadow_op;
    for (size_t index = 0; index < kernel_params.count; ++index) {
      if ((size_t)i < kernel_params.sizes[index]) {
        MPDType w = has_mixed_precision ? kernel_params.weights32[index][i]:
                                          MPDType(kernel_params.weights[index][i]);
        MPDType scaled_grad = static_cast<MPDType>(kernel_params.grads[index][i])*rescale_grad;
        if (clip_gradient >= 0.0f)
            scaled_grad = mshadow_op::clip::Map(scaled_grad, static_cast<MPDType>(clip_gradient));
  
        MPDType mean = static_cast<MPDType>(beta1) * kernel_params.mean[index][i] + 
          (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta1)) * scaled_grad;
        MPDType var = static_cast<MPDType>(beta2) * kernel_params.var[index][i] + 
          (static_cast<MPDType>(1.0f) - static_cast<MPDType>(beta2)) * scaled_grad * scaled_grad;
        kernel_params.mean[index][i]=mean;
        kernel_params.var[index][i]=var;
  
        MPDType g;
        if(bias_correction){
          MPDType mean_hat = mean / (static_cast<MPDType>(1.0f) - power::Map(static_cast<MPDType>(beta1), static_cast<MPDType>(step)));
          MPDType var_hat = var / (static_cast<MPDType>(1.0f) - power::Map(static_cast<MPDType>(beta2), static_cast<MPDType>(step)));
          g = mean_hat / (sqrt(var_hat) + epsilon) + wd * w;
        }else{
          g = mean / (sqrt(var) + epsilon) + wd * w;
        }
        kernel_params.temp_g[index][i]=g;
      }
    }
  }
};

template<typename MPDType, bool has_mixed_precision>
struct MultiLAMB_step2_kernelg {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, 
                                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                                  const float* sumSqWeigths,
                                  const float* sumSqtemp_g,
                                  const float learning_rate,
                                  const float lower_bound, 
                                  const float upper_bound,
                                  const OpReqType req) {
    for (size_t index = 0; index < kernel_params.count; ++index) {
      if ((size_t)i < kernel_params.sizes[index]) {
        MPDType w = has_mixed_precision ? kernel_params.weights32[index][i]:
                                            MPDType(kernel_params.weights[index][i]);
        float r1 = sqrt(sumSqWeigths[index]);
        float r2 = sqrt(sumSqtemp_g[index]);
      
        r1 = min(max(r1, lower_bound), upper_bound);
      
      
        // calculate lamb_trust_ratio
        MPDType r;
        if (r1 == 0.0f || r2 == 0.0f)
          r = 1.0f;
        else
          r = r1/r2;
          
        MPDType lr_adjusted = learning_rate * r;
        w -= lr_adjusted * kernel_params.temp_g[index][i];

        // update weights
        if (has_mixed_precision)
          kernel_params.weights32[index][i] = w;
        KERNEL_ASSIGN(kernel_params.out_data[index][i], req, w);
      }
    }
  }
};
    
    
  
template<typename MPDType, typename DType>
void call_kernel1(Stream<gpu>* s,
                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                  const MultiLAMBParam &param){
  
  size_t nblocks = kernel_params.count;
  bool has_mixed_precision = !std::is_same<DType, MPDType>::value;
  if(has_mixed_precision)
    kernel_step1<true><<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(
                      kernel_params,
                      param.learning_rate,
                      param.beta1, param.beta2,
                      param.epsilon, param.wd,
                      param.step, param.clip_gradient,
                      param.bias_correction,
                      param.rescale_grad);
  else
    kernel_step1<false><<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(
                      kernel_params,
                      param.learning_rate,
                      param.beta1, param.beta2,
                      param.epsilon, param.wd,
                      param.step, param.clip_gradient,
                      param.bias_correction,
                      param.rescale_grad);
    
  /*Kernel<MultiLAMB_step1_kernelg<MPDType, !std::is_same<DType, MPDType>::value>, gpu>::  
                                  Launch(s, kernel_params.max_size,
                                  kernel_params,
                                  param.learning_rate,
                                  param.beta1, param.beta2,
                                  param.epsilon,
                                  param.wd,
                                  param.step,
                                  param.clip_gradient,
                                  param.bias_correction,
                                  param.rescale_grad);*/
  }

template<typename MPDType, typename DType>
void call_kernel2(Stream<gpu>* s,
                  const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                  const MultiLAMBParam &param,
                  float* r1, float* r2,
                  const OpReqType req){

  size_t nblocks = kernel_params.count;
  bool has_mixed_precision = !std::is_same<DType, MPDType>::value;
  if(has_mixed_precision)
    kernel_step2<true><<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(
                      kernel_params,
                      r1, r2,
                      param.learning_rate,
                      param.lower_bound, param.upper_bound,
                      req);
  else
    kernel_step2<false><<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(
                      kernel_params,
                      r1, r2,
                      param.learning_rate,
                      param.lower_bound, param.upper_bound,
                      req);

  /*Kernel<MultiLAMB_step2_kernelg<MPDType, !std::is_same<DType, MPDType>::value>, gpu>::  
                                Launch(s, kernel_params.max_size,
                                kernel_params,
                                r1, r2,
                                param.learning_rate,
                                param.lower_bound, param.upper_bound,
                                req);*/
}


NNVM_REGISTER_OP(_multi_lamb_update)
.set_attr<FCompute>("FCompute<gpu>",  multiLAMBUpdate<gpu, false>);

NNVM_REGISTER_OP(_multi_mp_lamb_update)
.set_attr<FCompute>("FCompute<gpu>",  multiLAMBUpdate<gpu, true>);

}  // namespace op
}  // namespace mxnet
