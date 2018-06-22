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
 * \file dropout.cuh
 * \brief Dropout CUDA Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_DROPOUT_CUH_
#define MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_DROPOUT_CUH_

//******************************************************************************
// Forward
//******************************************************************************

// Dropout Layer (Forward Pass)
// @param1 (output) hidden_state_o: Output Hidden State  (after Dropout)
// [num_hidden_units x batch_size]
// @param2  (input) hidden_state_i:  Input Hidden State (before Dropout)
// [num_hidden_units x batch_size]
// @param3  (input) uniform_rv: Uniform Random Variable
// [num_hidden_units x batch_size]
// @param4  (param) dp_prob: Dropout Probability
// @param4  (param) hidden_state_size:
// Number of Hidden Units x Batch Size x Number of Batches
static __global__ void __cuda_dropout_forward(
        float * const __restrict__ hidden_state_o,
  const float * const __restrict__ hidden_state_i,
  const float * const __restrict__ uniform_rv,
  const float dp_prob, const unsigned hidden_state_size) {
  const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (g_threadIdx >= hidden_state_size)
    return;

  if (uniform_rv[g_threadIdx] <= dp_prob)
    hidden_state_o[g_threadIdx] = 0;
  else
    // scale the kept ones by 1 / (1 - dp_prob)
    hidden_state_o[g_threadIdx] = hidden_state_i[g_threadIdx] / (1 - dp_prob);
}

//******************************************************************************
// Backward
//******************************************************************************

// Dropout Layer (Backward Pass)
// @param1  (input) hidden_state_o_grad:
//   Output Hidden State Gradient  (after Dropout)
// [num_hidden_units x batch_size]
// @param2 (output) hidden_state_i_grad:
//    Input Hidden State Gradient (before Dropout)
// [num_hidden_units x batch_size]
// @param3  (input) uniform_rv: Uniform Random Variable
// [num_hidden_units x batch_size]
// @param4  (param) dp_prob: Dropout Probability
// @param5  (param) hidden_state_size:
// Number of Hidden Units x Batch Size x Number of Batches
static __global__ void __cuda_dropout_backward(
  const float * const /* __restrict__ */ hidden_state_o_grad,
        float * const /* __restrict__ */ hidden_state_i_grad,
  const float * const /* __restrict__ */ uniform_rv,
  const float dp_prob, const unsigned hidden_state_size) {
  const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (g_threadIdx >= hidden_state_size)
    return;

  if (uniform_rv[g_threadIdx] <= dp_prob)
    hidden_state_i_grad[g_threadIdx] = 0;
  else
    // scale the kept ones by 1 / (1 - dp_prob)
    hidden_state_i_grad[g_threadIdx] = hidden_state_o_grad[g_threadIdx] /
                                         (1 - dp_prob);
}

#endif  // MXNET_OPERATOR_CONTRIB_OPEN_LSTM_RNN_INCLUDE_DROPOUT_CUH_

