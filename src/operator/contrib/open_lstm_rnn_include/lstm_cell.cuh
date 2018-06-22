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
 * Copyright (c) 2018 by Contributors
 * \file lstm_cell.cuh
 * \brief LSTM Cell CUDA Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_LSTM_CELL_CUH_
#define MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_LSTM_CELL_CUH_

//******************************************************************************
// Forward
//******************************************************************************

static __forceinline__ __device__ float __cu_sigmoidf(float i) {
  return 1.0 / (1.0 + expf(-i));
}

// Fused Implementation of LSTM Cell (Forward Pass)
// @param1  (input) i2h_workspace: Input  -> Hidden Temporary Workspace
// @param2  (input) h2h_workspace: Hidden -> Hidden Temporary Workspace
// [4 x num_hidden_units x batch_size]
// @param3  (input) i2h_bias: Input  -> Hidden Bias
// @param4  (input) h2h_bias: Hidden -> Hidden Bias
// [4 x num_hidden_units]
// @param5  (input) prev_cell_state: Previous Cell State
// [num_hidden_units x batch_size]
// @param6 (output) linear_gates: Linear Gates (used for Backward Pass)
// The idea is to replay computations in Forward Pass with minimum cost.
// [4 x num_hidden_units x batch_size]
// @param7 (output) curr_cell_state  : Current Cell   State
// @param8 (output) curr_hidden_state: Current Hidden State
// [num_hidden_units x batch_size]
// @param9  (param) num_hidden_units: Number of Hidden Units
// @param10 (param) batch_size: Batch Size
static __global__ void __cuda_fused_lstm_forward(
  const float * const __restrict__ i2h_workspace,
  const float * const __restrict__ h2h_workspace,
  const float * const __restrict__ i2h_bias,
  const float * const __restrict__ h2h_bias,
  const float * const __restrict__ prev_cell_state,
        float * const __restrict__ linear_gates,
        float * const __restrict__ curr_cell_state,
        float * const __restrict__ curr_hidden_state,
  const unsigned num_hidden_units, const unsigned batch_size) {
  const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
                 num_hidden_units_x_batch_size = num_hidden_units * batch_size;

  if (g_threadIdx >= num_hidden_units_x_batch_size)
    return;

  const unsigned hidden_idx = g_threadIdx / batch_size;

  float gate_input[4];

#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    gate_input[i] =
      i2h_workspace[i * num_hidden_units_x_batch_size + g_threadIdx] +
      h2h_workspace[i * num_hidden_units_x_batch_size + g_threadIdx] +
      i2h_bias[i * num_hidden_units + hidden_idx] +
      h2h_bias[i * num_hidden_units + hidden_idx];
    if (linear_gates != NULL)
      linear_gates[i * num_hidden_units_x_batch_size + g_threadIdx] =
        gate_input[i];
  }

  float  input_gate = __cu_sigmoidf(gate_input[0]);
  float forget_gate = __cu_sigmoidf(gate_input[1]);
  float  input_actv =          tanh(gate_input[2]);
  float output_gate = __cu_sigmoidf(gate_input[3]);

  curr_cell_state[g_threadIdx] = forget_gate * prev_cell_state[g_threadIdx] +
                                 input_gate * input_actv;
  curr_hidden_state[g_threadIdx] = output_gate *
                                   tanh(curr_cell_state[g_threadIdx]);
}

//******************************************************************************
// Backward
//******************************************************************************

enum class enumBackwardReduceAlgo {PURE_ATOMICS, _32_HIERARCHICAL,
  _64_HIERARCHICAL, _128_HIERARCHICAL};

// This is the most generic reduction subroutine.
static __forceinline__ __device__ void __cu_bias_grad_reduce_generic(
  float * const __restrict__ bias_grad,
  float  input_gate_input_grad,
  float forget_gate_input_grad,
  float  input_actv_input_grad,
  float output_gate_input_grad) {
  if ( input_gate_input_grad != 0)
    atomicAdd(&bias_grad[0 * gridDim.x + blockIdx.x],  input_gate_input_grad);
  if (forget_gate_input_grad != 0)
    atomicAdd(&bias_grad[1 * gridDim.x + blockIdx.x], forget_gate_input_grad);
  if ( input_actv_input_grad != 0)
    atomicAdd(&bias_grad[2 * gridDim.x + blockIdx.x],  input_actv_input_grad);
  if (output_gate_input_grad != 0)
    atomicAdd(&bias_grad[3 * gridDim.x + blockIdx.x], output_gate_input_grad);
}

template < unsigned block_size_T >
static __forceinline__ __device__ void __cu_bias_grad_reduce(
  volatile float * const __restrict__ svmem_fused_lstm_backward,
           float * const __restrict__ bias_grad,
           float  input_gate_input_grad,
           float forget_gate_input_grad,
           float  input_actv_input_grad,
           float output_gate_input_grad);

template <>
__forceinline__ __device__ void __cu_bias_grad_reduce < 32 > (
  volatile float * const __restrict__ svmem_fused_lstm_backward,
           float * const __restrict__ bias_grad,
           float  input_gate_input_grad,
           float forget_gate_input_grad,
           float  input_actv_input_grad,
           float output_gate_input_grad) {
#pragma unroll
  for (unsigned stride = 16; stride != 0; stride /= 2) {
     input_gate_input_grad +=  __shfl_down(input_gate_input_grad, stride);
    forget_gate_input_grad += __shfl_down(forget_gate_input_grad, stride);
     input_actv_input_grad +=  __shfl_down(input_actv_input_grad, stride);
    output_gate_input_grad += __shfl_down(output_gate_input_grad, stride);
  }
  if (threadIdx.x == 0) {
    atomicAdd(&bias_grad[0 * gridDim.x + blockIdx.x],  input_gate_input_grad);
    atomicAdd(&bias_grad[1 * gridDim.x + blockIdx.x], forget_gate_input_grad);
    atomicAdd(&bias_grad[2 * gridDim.x + blockIdx.x],  input_actv_input_grad);
    atomicAdd(&bias_grad[3 * gridDim.x + blockIdx.x], output_gate_input_grad);
  }
}

template <>
__forceinline__ __device__ void __cu_bias_grad_reduce < 64 > (
  volatile float * const __restrict__ svmem_fused_lstm_backward,
           float * const __restrict__ bias_grad,
           float  input_gate_input_grad,
           float forget_gate_input_grad,
           float  input_actv_input_grad,
           float output_gate_input_grad) {
  volatile float * svmem__input_gate_input_grad = svmem_fused_lstm_backward +
                                                  0 * blockDim.x;
  volatile float * svmem_forget_gate_input_grad = svmem_fused_lstm_backward +
                                                  1 * blockDim.x;
  volatile float * svmem__input_actv_input_grad = svmem_fused_lstm_backward +
                                                  2 * blockDim.x;
  volatile float * svmem_output_gate_input_grad = svmem_fused_lstm_backward +
                                                  3 * blockDim.x;

  svmem__input_gate_input_grad[threadIdx.x] =  input_gate_input_grad;
  svmem_forget_gate_input_grad[threadIdx.x] = forget_gate_input_grad;
  svmem__input_actv_input_grad[threadIdx.x] =  input_actv_input_grad;
  svmem_output_gate_input_grad[threadIdx.x] = output_gate_input_grad;
  __syncthreads();
  // Up to this point, shared memory is initialized with gate gradients.
  //======================================================================
  // Note that starting from this point, execution becomes warp-wide therefore
  // there is no more synchronization between threads.
  // 1st warp are responsible for reduction in input_gate, forget_gate.
  // 2nd warp are responsible for reduction in input_actv, output_gate.
  if (threadIdx.x < 32) {  // [0, 32)
     input_gate_input_grad += svmem__input_gate_input_grad[threadIdx.x + 32];
    forget_gate_input_grad += svmem_forget_gate_input_grad[threadIdx.x + 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2) {
       input_gate_input_grad +=  __shfl_down(input_gate_input_grad, stride);
      forget_gate_input_grad += __shfl_down(forget_gate_input_grad, stride);
    }
  } else {  // [32, 64)
     input_actv_input_grad += svmem__input_actv_input_grad[threadIdx.x - 32];
    output_gate_input_grad += svmem_output_gate_input_grad[threadIdx.x - 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2) {
       input_actv_input_grad +=  __shfl_down(input_actv_input_grad, stride);
      output_gate_input_grad += __shfl_down(output_gate_input_grad, stride);
    }
  }
  //======================================================================
  // Reduction is complete. Update global memory using atomic functions.
  if (threadIdx.x ==  0) {
    atomicAdd(&bias_grad[0 * gridDim.x + blockIdx.x],  input_gate_input_grad);
    atomicAdd(&bias_grad[1 * gridDim.x + blockIdx.x], forget_gate_input_grad);
  }
  if (threadIdx.x == 32) {
    atomicAdd(&bias_grad[2 * gridDim.x + blockIdx.x],  input_actv_input_grad);
    atomicAdd(&bias_grad[3 * gridDim.x + blockIdx.x], output_gate_input_grad);
  }
}

template <>
__forceinline__ __device__ void __cu_bias_grad_reduce < 128 > (
  volatile float * const __restrict__ svmem_fused_lstm_backward,
           float * const __restrict__ bias_grad,
           float  input_gate_input_grad,
           float forget_gate_input_grad,
           float  input_actv_input_grad,
           float output_gate_input_grad) {
  volatile float * svmem__input_gate_input_grad = svmem_fused_lstm_backward +
                                                  0 * blockDim.x;
  volatile float * svmem_forget_gate_input_grad = svmem_fused_lstm_backward +
                                                  1 * blockDim.x;
  volatile float * svmem__input_actv_input_grad = svmem_fused_lstm_backward +
                                                  2 * blockDim.x;
  volatile float * svmem_output_gate_input_grad = svmem_fused_lstm_backward +
                                                  3 * blockDim.x;

  svmem__input_gate_input_grad[threadIdx.x] =  input_gate_input_grad;
  svmem_forget_gate_input_grad[threadIdx.x] = forget_gate_input_grad;
  svmem__input_actv_input_grad[threadIdx.x] =  input_actv_input_grad;
  svmem_output_gate_input_grad[threadIdx.x] = output_gate_input_grad;
  __syncthreads();
  // Up to this point, shared memory is initialized with gate gradients.
  //======================================================================
  // apply interleaving parallel reduction
  // 1st 64 threads are responsible for reduction in  input_gate,  input_actv.
  // 2nd 64 threads are responsible for reduction in forget_gate, output_gate.
  if (threadIdx.x < 64) {  // [0, 64)
    svmem__input_gate_input_grad[threadIdx.x] =
       input_gate_input_grad =
       input_gate_input_grad + svmem__input_gate_input_grad[threadIdx.x + 64];
    svmem_forget_gate_input_grad[threadIdx.x] =
      forget_gate_input_grad =
      forget_gate_input_grad + svmem_forget_gate_input_grad[threadIdx.x + 64];
  } else {  // [64, 128)
    svmem__input_actv_input_grad[threadIdx.x] =
       input_actv_input_grad =
       input_actv_input_grad + svmem__input_actv_input_grad[threadIdx.x - 64];
    svmem_output_gate_input_grad[threadIdx.x] =
      output_gate_input_grad =
      output_gate_input_grad + svmem_output_gate_input_grad[threadIdx.x - 64];
  }
  __syncthreads();
  // Note that starting from this point, execution becomes warp-wide
  // and therefore there is no more synchronization between threads.
  // 1st warp ( 0 ~  31) are responsible for reduction in  input_gate.
  // 2nd warp (32 ~  63) are responsible for reduction in forget_gate.
  // 3rd warp (64 ~  95) are responsible for reduction in  input_actv.
  // 4th warp (96 ~ 127) are responsible for reduction in output_gate.
  if (threadIdx.x < 32) {  // [0, 32)
     input_gate_input_grad += svmem__input_gate_input_grad[threadIdx.x + 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2)
       input_gate_input_grad +=  __shfl_down(input_gate_input_grad, stride);
  } else if (threadIdx.x < 64) {  // [32, 64)
    forget_gate_input_grad += svmem_forget_gate_input_grad[threadIdx.x - 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2)
      forget_gate_input_grad += __shfl_down(forget_gate_input_grad, stride);
  } else if (threadIdx.x < 96) {  // [64, 96)
     input_actv_input_grad += svmem__input_actv_input_grad[threadIdx.x + 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2)
       input_actv_input_grad +=  __shfl_down(input_actv_input_grad, stride);
  } else {  // [96, 128)
    output_gate_input_grad += svmem_output_gate_input_grad[threadIdx.x - 32];
#pragma unroll
    for (unsigned stride = 16; stride != 0; stride /= 2)
      output_gate_input_grad += __shfl_down(output_gate_input_grad, stride);
  }
  //======================================================================
  // Reduction is complete. Update global memory using atomic functions.
  if (threadIdx.x ==  0)
    atomicAdd(&bias_grad[0 * gridDim.x + blockIdx.x],  input_gate_input_grad);
  if (threadIdx.x == 32)
    atomicAdd(&bias_grad[1 * gridDim.x + blockIdx.x], forget_gate_input_grad);
  if (threadIdx.x == 64)
    atomicAdd(&bias_grad[2 * gridDim.x + blockIdx.x],  input_actv_input_grad);
  if (threadIdx.x == 96)
    atomicAdd(&bias_grad[3 * gridDim.x + blockIdx.x], output_gate_input_grad);
}

// Fused Implementation of LSTM Cell (Backward Pass)
// This block should be launched with the following parameters:
// <<< dim3(num_hidden_units,
//          (batch_size - 1)/(128/64/32) + 1, 1),
//          (128/64/32), ... >>>
// where the choice of (128/64/32) should be
// figured out dynamically depending on batch size.
// @param1 (output) i2h_workspace: Input  -> Hidden Temporary Workspace
// @param2 (output) h2h_workspace: Hidden -> Hidden Temporary Workspace
// [4 x num_hidden_units x batch_size]
// @param3 (output) bias_grad: Bias Gradient
// [4 x num_hidden_units]
// We can only keep one copy of bias gradient because i2h_bias and h2h_bias
// always share the same gradient.
// @param4  (inout) cell_grad: Cell Gradient
// Gradient of current cell state will be computed and propagated back
// to the previous cell in the time sequence.
// @param5  (input) prev_cell_state: Previous Cell State
// [num_hidden_units x batch_size]
// @param6  (input) linear_gates: Linear Gates
// The idea is to replay computations in Forward Pass with minimum cost.
// [4 x num_hidden_units x batch_size]
// @param7  (input) curr_cell_state: Current  Cell State
// [num_hidden_units x batch_size]
// @param8  (input) i2h_grad_workspace: Input  ->
//   Hidden Gradient Temporary Workspace
// @param9  (input) h2h_grad_workspace: Hidden ->
//   Hidden Gradient Temporary Workspace
// [num_hidden_units x batch_size]
// @param10 (param) batch_size: Batch Size
// @param11 (param) algo: Backward Pass Reduction Algorithm
static __global__ void __cuda_fused_lstm_backward(
        float * const __restrict__ i2h_workspace,
        float * const __restrict__ h2h_workspace,
        float * const __restrict__ bias_grad,
        float * const __restrict__ cell_grad,
  const float * const __restrict__ prev_cell_state,
  const float * const __restrict__ linear_gates,
  const float * const __restrict__ curr_cell_state,
  const float * const __restrict__ i2h_grad_workspace,
  const float * const __restrict__ h2h_grad_workspace,
  const unsigned batch_size, const enumBackwardReduceAlgo algo) {
  extern __shared__ volatile float svmem_fused_lstm_backward[];

  const unsigned batch_idx = threadIdx.x + blockIdx.y * blockDim.x;

  float  input_gate_input_grad = 0;
  float forget_gate_input_grad = 0;
  float  input_actv_input_grad = 0;
  float output_gate_input_grad = 0;

  // If *this* thread is not padded for reduction purpose, ...
  if (batch_idx < batch_size) {
    const unsigned g_threadIdx = batch_idx + blockIdx.x * batch_size,
      num_hidden_units_x_batch_size = gridDim.x * batch_size;

    //==============================================================
    // Forward Pass Replay
    //==============================================================
    // replay the computation we just did for the forward pass
    float  input_gate = __cu_sigmoidf(linear_gates[
      0 * num_hidden_units_x_batch_size + g_threadIdx]);
    float forget_gate = __cu_sigmoidf(linear_gates[
      1 * num_hidden_units_x_batch_size + g_threadIdx]);
    float  input_actv =          tanh(linear_gates[
      2 * num_hidden_units_x_batch_size + g_threadIdx]);
    float output_gate = __cu_sigmoidf(linear_gates[
      3 * num_hidden_units_x_batch_size + g_threadIdx]);

    float curr_cell_state_actv = tanh(curr_cell_state[g_threadIdx]);

    //==============================================================
    // Backward Computation
    //==============================================================
    // In the gradient computation, we mainly make use of
    // the following two important properties:
    // 1. tanh'(x) = 1 - tanh(x) * tanh(x)
    // 2. sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // where the symbol ' denotes gradient.

    // curr_hidden_state[g_threadIdx] =
    //   output_gate * tanh(curr_cell_state[g_threadIdx]);
    float curr_hidden_state_grad = 0;
    curr_hidden_state_grad += i2h_grad_workspace == NULL ?
      0 : i2h_grad_workspace[g_threadIdx];
    curr_hidden_state_grad += h2h_grad_workspace == NULL ?
      0 : h2h_grad_workspace[g_threadIdx];

    float curr_cell_state_grad =
      curr_hidden_state_grad * output_gate *
        (1 - curr_cell_state_actv * curr_cell_state_actv) +
      cell_grad[g_threadIdx];
    float     output_gate_grad = curr_hidden_state_grad * curr_cell_state_actv;

    // curr_cell_state[g_threadIdx] =
    //   forget_gate * prev_cell_state[g_threadIdx] +
    //   input_gate * input_actv;
    float forget_gate_grad = curr_cell_state_grad *
                             prev_cell_state[g_threadIdx];
    float  input_gate_grad = curr_cell_state_grad * input_actv;
    float  input_actv_grad = curr_cell_state_grad * input_gate;

    cell_grad[g_threadIdx] = curr_cell_state_grad * forget_gate;

    /*
    float  input_gate = __cu_sigmoidf(gate_input[0]);
    float forget_gate = __cu_sigmoidf(gate_input[1]);
    float  input_actv =          tanh(gate_input[2]);
    float output_gate = __cu_sigmoidf(gate_input[3]);
     */
     input_gate_input_grad =  input_gate_grad *  input_gate * (1 -  input_gate);
    forget_gate_input_grad = forget_gate_grad * forget_gate * (1 - forget_gate);
     input_actv_input_grad =  input_actv_grad * (1 - input_actv * input_actv);
    output_gate_input_grad = output_gate_grad * output_gate * (1 - output_gate);

    /*
  #pragma unroll
    for (unsigned i = 0; i < 4; ++i)
    {
      gate_input[i] = 
        i2h_workspace[i * num_hidden_units_x_batch_size + g_threadIdx] + 
        h2h_workspace[i * num_hidden_units_x_batch_size + g_threadIdx] + 
        i2h_bias[i * num_hidden_units + hidden_idx] + 
        h2h_bias[i * num_hidden_units + hidden_idx];
      linear_gates[i * num_hidden_units_x_batch_size + g_threadIdx] = gate_input[i];
    }
     */
    i2h_workspace[0 * num_hidden_units_x_batch_size + g_threadIdx] =
    h2h_workspace[0 * num_hidden_units_x_batch_size + g_threadIdx] =
       input_gate_input_grad;
    i2h_workspace[1 * num_hidden_units_x_batch_size + g_threadIdx] =
    h2h_workspace[1 * num_hidden_units_x_batch_size + g_threadIdx] =
      forget_gate_input_grad;
    i2h_workspace[2 * num_hidden_units_x_batch_size + g_threadIdx] =
    h2h_workspace[2 * num_hidden_units_x_batch_size + g_threadIdx] =
       input_actv_input_grad;
    i2h_workspace[3 * num_hidden_units_x_batch_size + g_threadIdx] =
    h2h_workspace[3 * num_hidden_units_x_batch_size + g_threadIdx] =
      output_gate_input_grad;
  }
  //======================================================================
  // Bias Gradients Reduction
  //======================================================================
  // According to the technical report presented at GTC:
  // http://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf
  // The best performance is achieved when we use
  // a combination of CTA (block)-wide reduction and atomics.
  switch (algo) {
    case enumBackwardReduceAlgo:: _32_HIERARCHICAL:
      __cu_bias_grad_reduce < 32 >  (
        svmem_fused_lstm_backward, bias_grad,
        input_gate_input_grad, forget_gate_input_grad,
        input_actv_input_grad, output_gate_input_grad); break;
    case enumBackwardReduceAlgo:: _64_HIERARCHICAL:
      __cu_bias_grad_reduce < 64 >  (
        svmem_fused_lstm_backward, bias_grad,
        input_gate_input_grad, forget_gate_input_grad,
        input_actv_input_grad, output_gate_input_grad); break;
    case enumBackwardReduceAlgo::_128_HIERARCHICAL:
      __cu_bias_grad_reduce < 128 > (
        svmem_fused_lstm_backward, bias_grad,
        input_gate_input_grad, forget_gate_input_grad,
        input_actv_input_grad, output_gate_input_grad); break;
    default:
      __cu_bias_grad_reduce_generic(
        bias_grad,
        input_gate_input_grad, forget_gate_input_grad,
        input_actv_input_grad, output_gate_input_grad);
  }
}

#endif  // MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_LSTM_CELL_CUH_
