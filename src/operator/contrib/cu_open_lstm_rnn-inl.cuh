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
 * \file open_lstm_rnn-inl.cuh
 * \brief LSTM RNN Open-Source CUDA Implementation
 * \author Bojian (Jack) Zheng, Gennady Pekhimenko, Jeremy Appleyard
 */
#ifndef MXNET_OPERATOR_CONTRIB_CU_OPEN_LSTM_RNN_INL_CUH_
#define MXNET_OPERATOR_CONTRIB_CU_OPEN_LSTM_RNN_INL_CUH_

#include <mxnet/storage.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include "./open_lstm_rnn-inl.h"
#include "./open_lstm_rnn_include/dropout.cuh"
#include "./open_lstm_rnn_include/lstm_cell.cuh"
#include "./open_lstm_rnn_include/cublas_matmul.h"
#include "./open_lstm_rnn_include/cublas_transpose.h"

#define RE_CAST(ptr) reinterpret_cast < float * > (ptr)

namespace mxnet {
namespace op {

class CUOpenLSTMRNNOp : public Operator {
 public:
  explicit CUOpenLSTMRNNOp(OpenLSTMRNNParam param) {
    this->param_ = param; initialized_ = false;
  }

  ~CUOpenLSTMRNNOp() {
    //=========================================================================
    // Free the allocated workspace memory.
    //=========================================================================
    Storage::Get()->Free(m_data_T_major);
    Storage::Get()->Free(m_data_T_major_grad);
    Storage::Get()->Free(m_cell);
    Storage::Get()->Free(m_hidden);
    Storage::Get()->Free(m_i2h_workspace);
    Storage::Get()->Free(m_h2h_workspace);
    Storage::Get()->Free(m_i2h_grad_workspace);
    Storage::Get()->Free(m_h2h_grad_workspace);
    Storage::Get()->Free(m_linear_gates);
    //=========================================================================
    // Destroy the cuBLAS handle.
    //=========================================================================
    CUBLAS_CALL(cublasDestroy(m_cublas_handle));
    //=========================================================================
    // Free the workers (cudaStream and cudaEvent).
    //=========================================================================
    for (unsigned layer_idx = 0; layer_idx < param_.num_layers; ++layer_idx) {
      CUDA_CALL(cudaStreamDestroy(m_stream_i2h[layer_idx]));
      m_stream_i2h[layer_idx] = NULL;
      CUDA_CALL(cudaStreamDestroy(m_stream_h2h[layer_idx]));
      m_stream_h2h[layer_idx] = NULL;
    }
    delete [] m_stream_i2h; m_stream_i2h = NULL;
    delete [] m_stream_h2h; m_stream_h2h = NULL;
    for (unsigned layer_idx = 0; layer_idx < param_.num_layers; ++layer_idx) {
      for (unsigned seq_idx = 0; seq_idx < param_.seq_len; ++seq_idx) {
        CUDA_CALL(cudaEventDestroy(m_event_i2h[layer_idx][seq_idx]));
        m_event_i2h[layer_idx][seq_idx] = NULL;
        CUDA_CALL(cudaEventDestroy(m_event_h2h[layer_idx][seq_idx]));
        m_event_h2h[layer_idx][seq_idx] = NULL;
      }
      delete [] m_event_i2h[layer_idx]; m_event_i2h[layer_idx] = NULL;
      delete [] m_event_h2h[layer_idx]; m_event_h2h[layer_idx] = NULL;
    }
    delete [] m_event_i2h; m_event_i2h = NULL;
    delete [] m_event_h2h; m_event_h2h = NULL;
    //=========================================================================
    // Destroy the cuRAND handle and associated workspace.
    //=========================================================================
    if (param_.i_dp_prob != 0 && param_.num_layers > 1) {
      CURAND_CALL(curandDestroyGenerator(m_rng));
      Storage::Get()->Free(m_i_dp_uniform_rv);
      Storage::Get()->Free(m_i_dp_workspace);
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;

    //=========================================================================
    // IO Data
    //=========================================================================
    std::size_t in_expected = 7, out_expected = param_.state_outputs ? 3 : 1;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 3, float> data        = in_data[open_lstm_rnn_enum::kData]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 3, float> init_hidden = in_data[open_lstm_rnn_enum::kInitHidden]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 3, float> init_cell   = in_data[open_lstm_rnn_enum::kInitCell]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 1, float> i2h_weight  = in_data[open_lstm_rnn_enum::ki2hWeight]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> i2h_bias    = in_data[open_lstm_rnn_enum::ki2hBias]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_weight  = in_data[open_lstm_rnn_enum::kh2hWeight]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_bias    = in_data[open_lstm_rnn_enum::kh2hBias]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 3, float> concat_hidden_states = out_data[open_lstm_rnn_enum::
                                                          kConcatHiddenStates]
                                        .get<gpu, 3, float>(s);
    CHECK_EQ(data       .CheckContiguous(), true);
    CHECK_EQ(init_hidden.CheckContiguous(), true);
    CHECK_EQ(init_cell  .CheckContiguous(), true);
    CHECK_EQ(i2h_weight .CheckContiguous(), true);
    CHECK_EQ(i2h_bias   .CheckContiguous(), true);
    CHECK_EQ(h2h_weight .CheckContiguous(), true);
    CHECK_EQ(h2h_bias   .CheckContiguous(), true);
    CHECK_EQ(concat_hidden_states.CheckContiguous(), true);
    float *ptr_final_hidden = NULL, *ptr_final_cell = NULL;
    if (param_.state_outputs) {
      Tensor<gpu, 3, float> final_hidden =
        out_data[open_lstm_rnn_enum::kFinalHidden].get<gpu, 3, float>(s);
      Tensor<gpu, 3, float> final_cell =
        out_data[open_lstm_rnn_enum::kFinalCell]  .get<gpu, 3, float>(s);
      CHECK_EQ(final_hidden.CheckContiguous(), true);
      CHECK_EQ(final_cell  .CheckContiguous(), true);
      ptr_final_hidden = final_hidden.dptr_;
      ptr_final_cell   = final_cell  .dptr_;
    }
    //=========================================================================
    // Initialization
    //=========================================================================
    if (!initialized_)
      Init(s, in_data, out_data);
    //=========================================================================
    // Forward Pass
    //=========================================================================
    // generate random variable if i_dp_prob is nonzero
    if (param_.i_dp_prob != 0 && param_.num_layers > 0 && ctx.is_train) {
      CURAND_CALL(curandSetStream(m_rng, m_stream_i2h[1]));
      CURAND_CALL(curandGenerateUniform(m_rng,
                                        RE_CAST(m_i_dp_uniform_rv.dptr),
                                        (param_.num_layers - 1) *
                                          param_.seq_len *
                                          m_num_hidden_units_x_batch_size));
    }
    CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_i2h[0]));
    transpose(m_cublas_handle,
              RE_CAST(m_data_T_major.dptr), data.dptr_,
              param_.batch_size,
              param_.seq_len * param_.embed_dim);
    // use ScheduleList to implement wavefront parallelism
    for (ScheduleList::iterator iter  = m_forward_schedule.begin();
                                iter != m_forward_schedule.end(); ++iter) {
      // obtain the precomputed schedule
      unsigned layer_begin = iter->m_layer_begin, layer_end = iter->m_layer_end,
        seq_begin = iter->m_seq_begin, seq_end = iter->m_seq_end;

      for (unsigned layer_idx = layer_begin; layer_idx < layer_end; ++layer_idx) {
        //=====================================================================
        // Input -> Hidden
        //=====================================================================
        // Comment: If you find it difficult to interpret the code due to
        // pointer operations, please kindly refer to the code in the
        // block comment section for equivalent implementation. Thanks.
        CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_i2h[layer_idx]));
        //=====================================================================
        // wait here until m_hidden is ready
        // i2h of next layer needs to wait for h2h of previous layer
        for (unsigned seq_idx = seq_begin; seq_idx < seq_end; ++seq_idx)
          if (layer_idx != 0)
            CUDA_CALL(cudaStreamWaitEvent(m_stream_i2h[layer_idx],
                                          m_event_h2h[layer_idx - 1][seq_idx], 0));
        //=====================================================================
        if (layer_idx == 0) {
          /*
          matmul_stridedbatched(m_cublas_handle,
            &m_i2h_workspace[seq_begin],
             m_l0_i2h_weight,
            &m_data_T_major[seq_begin],
            num_gates * m_num_hidden_units, m_batch_size, m_embed_dim,
            num_gates * m_num_hidden_units * m_batch_size, 0,
            m_embed_dim * m_batch_size, seq_end - seq_begin);
           */
          matmul_stridedbatched(m_cublas_handle,
                                RE_CAST(m_i2h_workspace.dptr) +
                                  seq_begin * 4 * m_num_hidden_units_x_batch_size,
                                i2h_weight.dptr_,
                                RE_CAST(m_data_T_major.dptr) +
                                  seq_begin * param_.embed_dim * param_.batch_size,
                                4 * param_.num_hidden_units,
                                param_.batch_size,
                                param_.embed_dim,
                                4 * m_num_hidden_units_x_batch_size,
                                0,
                                param_.embed_dim * param_.batch_size,
                                seq_end - seq_begin);
        } else {
          /*
          if (m_i_dp_prob != 0 && is_train)
          {
            m_i_dp_handle->forward(
              &m_i_dp_workspace [layer_idx - 1][seq_begin],
              &m_hidden[layer_idx - 1][seq_begin + 1],
              &m_i_dp_uniform_rv[layer_idx - 1][seq_begin],
              m_stream_i2h[layer_idx], seq_end - seq_begin);
          }
          matmul_stridedbatched(m_cublas_handle,
            &m_i2h_workspace[seq_begin],
            &m_lN_i2h_weight[layer_idx - 1],
            m_i_dp_prob != 0 && is_train ? 
              &m_i_dp_workspace[layer_idx - 1][seq_begin] :
              &m_hidden[layer_idx - 1][seq_begin + 1],
            num_gates * m_num_hidden_units, m_batch_size, m_num_hidden_units,
            num_gates * m_num_hidden_units * m_batch_size, 0,
              m_num_hidden_units * m_batch_size, 
            seq_end - seq_begin);
           */
          if (param_.i_dp_prob != 0 && ctx.is_train) {
            __cuda_dropout_forward
              <<< (m_num_hidden_units_x_batch_size * (seq_end - seq_begin) - 1) / 128 + 1,
                  128, 0, m_stream_i2h[layer_idx]
              >>> (RE_CAST(m_i_dp_workspace.dptr) +
                     (layer_idx - 1) * param_.seq_len * m_num_hidden_units_x_batch_size +
                                            seq_begin * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_hidden.dptr) +
                     (layer_idx - 1) * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                            (seq_begin + 1) * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_i_dp_uniform_rv.dptr) +
                     (layer_idx - 1) * param_.seq_len * m_num_hidden_units_x_batch_size +
                                            seq_begin * m_num_hidden_units_x_batch_size,
                   param_.i_dp_prob, m_num_hidden_units_x_batch_size * (seq_end - seq_begin));
          }  // i_dp_prob
          matmul_stridedbatched(m_cublas_handle,
                                RE_CAST(m_i2h_workspace.dptr) +
                                  seq_begin * 4 * m_num_hidden_units_x_batch_size,
                                i2h_weight.dptr_ +
                                  4 * param_.num_hidden_units * param_.embed_dim +
                                  (layer_idx - 1) * 4 * param_.num_hidden_units *
                                    param_.num_hidden_units,
                                param_.i_dp_prob != 0 && ctx.is_train ?
                                  RE_CAST(m_i_dp_workspace.dptr) +
                                    (layer_idx - 1) * param_.seq_len *
                                      m_num_hidden_units_x_batch_size +
                                    seq_begin * m_num_hidden_units_x_batch_size :
                                  RE_CAST(m_hidden.dptr) +
                                    (layer_idx - 1) * (param_.seq_len + 1) *
                                      m_num_hidden_units_x_batch_size +
                                    (seq_begin + 1) * m_num_hidden_units_x_batch_size,
                                  4 * param_.num_hidden_units,
                                  param_.batch_size,
                                  param_.num_hidden_units,
                                  4 * m_num_hidden_units_x_batch_size,
                                  0,
                                  m_num_hidden_units_x_batch_size,
                                  seq_end - seq_begin);
        }  // layer_idx == 0
        //=====================================================================
        // record that we are computing m_i2h_workspace
        for (unsigned seq_idx = seq_begin; seq_idx != seq_end; ++seq_idx)
          CUDA_CALL(cudaEventRecord(m_event_i2h[layer_idx][seq_idx],
                                    m_stream_i2h[layer_idx]));
        //=====================================================================
        // Hidden -> Hidden
        //=====================================================================
        CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_h2h[layer_idx]));
        for (unsigned seq_idx = seq_begin; seq_idx < seq_end; ++seq_idx) {
          if (seq_idx == 0) {
            /*
            transpose(m_cublas_handle,
              &m_hidden[layer_idx],
              &m_init_hidden[layer_idx],
              m_batch_size, m_num_hidden_units);
              */
            transpose(m_cublas_handle,
                      RE_CAST(m_hidden.dptr) +
                        layer_idx * (param_.seq_len + 1) *
                          m_num_hidden_units_x_batch_size,
                      init_hidden.dptr_ +
                        layer_idx * m_num_hidden_units_x_batch_size,
                      param_.batch_size, param_.num_hidden_units);
          }  // seq_idx
          /*
          matmul(m_cublas_handle,
            &m_h2h_workspace[layer_idx],
            &m_lX_h2h_weight[layer_idx],
            &m_hidden[layer_idx][seq_idx],
            num_gates * m_num_hidden_units,
            m_batch_size,
            m_num_hidden_units);
            */
          matmul(m_cublas_handle,
                 RE_CAST(m_h2h_workspace.dptr) +
                   layer_idx * 4 * m_num_hidden_units_x_batch_size,
                 h2h_weight.dptr_ +
                   layer_idx * 4 * param_.num_hidden_units * param_.num_hidden_units,
                 RE_CAST(m_hidden.dptr) +
                   layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                       seq_idx      * m_num_hidden_units_x_batch_size,
                 4 * param_.num_hidden_units,
                 param_.batch_size,
                 param_.num_hidden_units);
          if (seq_idx == 0) {
            /*
            transpose(m_cublas_handle,
              &m_cell[layer_idx],
              &m_init_cell[layer_idx],
              m_batch_size, m_num_hidden_units);
              */
            transpose(m_cublas_handle,
                      RE_CAST(m_cell.dptr) +
                        layer_idx * (param_.seq_len + 1) *
                          m_num_hidden_units_x_batch_size,
                      init_cell.dptr_ +
                        layer_idx * m_num_hidden_units_x_batch_size,
                      param_.batch_size,
                      param_.num_hidden_units);
          }
          //=====================================================================
          // wait here until the data in m_i2h_workspace is ready
          // h2h needs to wait for i2h
          CUDA_CALL(cudaStreamWaitEvent(m_stream_h2h[layer_idx],
                                        m_event_i2h[layer_idx][seq_idx], 0));
          /*
          if (layer_idx == 0)
          {
            forward(
              &m_i2h_workspace  [seq_idx],
              &m_h2h_workspace[layer_idx],
                m_l0_i2h_bias,
                m_lX_h2h_bias,
              &m_cell  [layer_idx][seq_idx],
              is_train ? &m_linear_gates[layer_idx][seq_idx] : nullptr,
              &m_cell  [layer_idx][seq_idx + 1],
              &m_hidden[layer_idx][seq_idx + 1],
              m_stream_h2h[layer_idx]);
          }
          else
          {
            forward(
              &m_i2h_workspace  [seq_idx],
              &m_h2h_workspace[layer_idx],
              &m_lN_i2h_bias[layer_idx - 1],
              &m_lX_h2h_bias[layer_idx],
              &m_cell  [layer_idx][seq_idx],
              is_train ? &m_linear_gates[layer_idx][seq_idx] : nullptr,
              &m_cell  [layer_idx][seq_idx + 1],
              &m_hidden[layer_idx][seq_idx + 1],
              m_stream_h2h[layer_idx]);
          }
            */
          __cuda_fused_lstm_forward
            <<< (m_num_hidden_units_x_batch_size - 1) / 128 + 1,
                128, 0, m_stream_h2h[layer_idx]
            >>> (RE_CAST(m_i2h_workspace.dptr) +
                   seq_idx * 4 * m_num_hidden_units_x_batch_size,
                 RE_CAST(m_h2h_workspace.dptr) +
                   layer_idx * 4 * m_num_hidden_units_x_batch_size,
                 i2h_bias.dptr_ + layer_idx * 4 * param_.num_hidden_units,
                 h2h_bias.dptr_ + layer_idx * 4 * param_.num_hidden_units,
                 RE_CAST(m_cell.dptr) +
                    layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                        seq_idx      * m_num_hidden_units_x_batch_size,
                 ctx.is_train ?
                   RE_CAST(m_linear_gates.dptr) +
                     layer_idx * param_.seq_len * 4 * m_num_hidden_units_x_batch_size +
                                        seq_idx * 4 * m_num_hidden_units_x_batch_size
                   : NULL,
                 RE_CAST(m_cell.dptr) +
                   layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                      (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                 RE_CAST(m_hidden.dptr) +
                   layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                      (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                 param_.num_hidden_units,
                 param_.batch_size);
          // record that we are computing m_hidden
          if (layer_idx != param_.num_layers - 1)
            CUDA_CALL(cudaEventRecord(m_event_h2h[layer_idx][seq_idx],
                                      m_stream_h2h[layer_idx]));
          // output final hidden and cell state if at the end of sequence
          /*
          if (param_.state_outputs && seq_idx == (param_.seq_len - 1))
          {
            transpose
            (
              transpose(m_cublas_handle,
                ptr_final_hidden[layer_idx],
                m_hidden[layer_idx][seq_idx + 1],
                param_.num_hidden_units, param_.batch_size);
              transpose(m_cublas_handle,
                ptr_final_cell  [layer_idx],
                m_hidden[layer_idx][seq_idx + 1],
                param_.num_hidden_units, param_.batch_size);
            )
          }
            */
          if (param_.state_outputs && seq_idx == (param_.seq_len - 1)) {
            transpose(m_cublas_handle,
                      ptr_final_hidden +
                        layer_idx * m_num_hidden_units_x_batch_size,
                      RE_CAST(m_hidden.dptr) +
                        layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                           (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                      param_.num_hidden_units,
                      param_.batch_size);
            transpose(m_cublas_handle,
                      ptr_final_cell +
                        layer_idx * m_num_hidden_units_x_batch_size,
                      RE_CAST(m_cell.dptr) +
                        layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                           (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                      param_.num_hidden_units,
                      param_.batch_size);
          }  // state_outputs
        }  // seq_idx
      }  // layer_idx
    }  // schedule
    /*
    transpose(m_cublas_handle,
      m_concat_hidden_states, &m_hidden[m_num_layers - 1][1],
      m_seq_len * m_num_hidden_units, m_batch_size);
     */
    transpose(m_cublas_handle,
              concat_hidden_states.dptr_,
              RE_CAST(m_hidden.dptr) +
                (param_.num_layers - 1) * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                                             1 * m_num_hidden_units_x_batch_size,
              param_.seq_len * param_.num_hidden_units,
              param_.batch_size);
    if (!param_.state_outputs) {
      CUDA_CALL(cudaStreamSynchronize(m_stream_h2h[param_.num_layers - 1]));
    } else {
      for (unsigned layer_idx = 0; layer_idx < param_.num_layers; ++layer_idx)
        CUDA_CALL(cudaStreamSynchronize(m_stream_h2h[layer_idx]));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;

    std::size_t in_expected = 7, out_expected = param_.state_outputs ? 3 : 1;
    //=========================================================================
    // IO Data
    //=========================================================================
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(req.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 3, float> data        = in_data[open_lstm_rnn_enum::kData]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 3, float> init_hidden = in_data[open_lstm_rnn_enum::kInitHidden]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 3, float> init_cell   = in_data[open_lstm_rnn_enum::kInitCell]
                                        .get<gpu, 3, float>(s);
    Tensor<gpu, 1, float> i2h_weight  = in_data[open_lstm_rnn_enum::ki2hWeight]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> i2h_bias    = in_data[open_lstm_rnn_enum::ki2hBias]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_weight  = in_data[open_lstm_rnn_enum::kh2hWeight]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_bias    = in_data[open_lstm_rnn_enum::kh2hBias]
                                        .get<gpu, 1, float>(s);
    Tensor<gpu, 3, float> data_grad       = in_grad[open_lstm_rnn_enum::kData]
                                            .get<gpu, 3, float>(s);
    Tensor<gpu, 1, float> i2h_weight_grad = in_grad[open_lstm_rnn_enum::ki2hWeight]
                                            .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> i2h_bias_grad   = in_grad[open_lstm_rnn_enum::ki2hBias]
                                            .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_weight_grad = in_grad[open_lstm_rnn_enum::kh2hWeight]
                                            .get<gpu, 1, float>(s);
    Tensor<gpu, 1, float> h2h_bias_grad   = in_grad[open_lstm_rnn_enum::kh2hBias]
                                            .get<gpu, 1, float>(s);
    Tensor<gpu, 3, float> concat_hidden_states_grad = out_grad[open_lstm_rnn_enum::
                                                               kConcatHiddenStates]
                                                      .get<gpu, 3, float>(s);
    CHECK_EQ(data       .CheckContiguous(), true);
    CHECK_EQ(init_hidden.CheckContiguous(), true);
    CHECK_EQ(init_cell  .CheckContiguous(), true);
    CHECK_EQ(i2h_weight .CheckContiguous(), true);
    CHECK_EQ(i2h_bias   .CheckContiguous(), true);
    CHECK_EQ(h2h_weight .CheckContiguous(), true);
    CHECK_EQ(h2h_bias   .CheckContiguous(), true);
    CHECK_EQ(data_grad      .CheckContiguous(), true);
    CHECK_EQ(i2h_weight_grad.CheckContiguous(), true);
    CHECK_EQ(i2h_bias_grad  .CheckContiguous(), true);
    CHECK_EQ(h2h_weight_grad.CheckContiguous(), true);
    CHECK_EQ(h2h_bias_grad  .CheckContiguous(), true);
    CHECK_EQ(concat_hidden_states_grad.CheckContiguous(), true);
    float *ptr_final_hidden_grad = NULL, *ptr_final_cell_grad = NULL;
    if (param_.state_outputs) {
      Tensor<gpu, 3, float> final_hidden_grad = out_grad[open_lstm_rnn_enum::kFinalHidden]
                                                .get<gpu, 3, float>(s);
      Tensor<gpu, 3, float> final_cell_grad   = out_grad[open_lstm_rnn_enum::kFinalCell]
                                                .get<gpu, 3, float>(s);
      CHECK_EQ(final_hidden_grad.CheckContiguous(), true);
      CHECK_EQ(final_cell_grad  .CheckContiguous(), true);
      ptr_final_hidden_grad = final_hidden_grad.dptr_;
      ptr_final_cell_grad   = final_cell_grad  .dptr_;
    }
    //=========================================================================
    // Preparation
    //=========================================================================
    unsigned block_size;
    switch (m_algo) {
      case enumBackwardReduceAlgo:: _32_HIERARCHICAL: block_size =  32; break;
      case enumBackwardReduceAlgo:: _64_HIERARCHICAL: block_size =  64; break;
      case enumBackwardReduceAlgo::_128_HIERARCHICAL: block_size = 128; break;
      default: block_size = param_.batch_size <= 1024 ? param_.batch_size : 128;
    }
    CUDA_CALL(cudaMemsetAsync(i2h_weight_grad.dptr_,
                              0,
                              i2h_weight_grad.shape_[0] * sizeof(float),
                              m_stream_i2h[param_.num_layers - 1]));
    CUDA_CALL(cudaMemsetAsync(h2h_weight_grad.dptr_,
                              0,
                              h2h_weight_grad.shape_[0] * sizeof(float),
                              m_stream_h2h[param_.num_layers - 1]));
    CUDA_CALL(cudaMemsetAsync(h2h_bias_grad  .dptr_,
                              0,
                              h2h_bias_grad  .shape_[0] * sizeof(float),
                              m_stream_h2h[param_.num_layers - 1]));
    // There is no need to clear the i2h_bias_grad, as we always directly
    // copy h2h_bias_grad to i2h_bias_grad.
    CUDA_CALL(cudaMemsetAsync(m_cell_grad.dptr,
                              0,
                              m_cell_grad.size,
                              m_stream_h2h[param_.num_layers - 1]));
    //=========================================================================
    // Backward Pass
    //=========================================================================
    CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_h2h[param_.num_layers - 1]));
    /*
    transpose(m_cublas_handle,
      m_i2h_grad_workspace, m_concat_hidden_states_grad,
      m_batch_size, m_seq_len * m_num_hidden_units);
     */
    transpose(m_cublas_handle,
              RE_CAST(m_i2h_grad_workspace.dptr),
              concat_hidden_states_grad.dptr_,
              param_.batch_size,
              param_.seq_len * param_.num_hidden_units);
    for (ScheduleList::iterator iter  = m_backward_schedule.begin();
                                iter != m_backward_schedule.end(); ++iter) {
      // obtain the precomputed schedule
      int layer_rbegin = param_.num_layers - 1 - iter->m_layer_begin,
          layer_rend   = param_.num_layers - 1 - iter->m_layer_end,
          seq_rbegin = param_.seq_len - 1 - iter->m_seq_begin,
          seq_rend   = param_.seq_len - 1 - iter->m_seq_end;
      for (int layer_idx = layer_rbegin; layer_idx > layer_rend; --layer_idx) {
        //=====================================================================
        // Hidden -> Hidden
        //=====================================================================
        CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_h2h[layer_idx]));
        for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx) {
          if (layer_idx != static_cast < int > (param_.num_layers - 1))
            // wait here until the data in m_i2h_grad_workspace is ready
            // h2h of previous layer needs to wait for i2h of next layer
            CUDA_CALL(cudaStreamWaitEvent(m_stream_h2h[layer_idx],
                                          m_event_i2h[layer_idx + 1][seq_idx], 0));
          if (seq_idx == static_cast < int > (param_.seq_len - 1)) {
            if (param_.state_outputs) {
              // Under the condition that final cell and hidden states
              // have gradients (e.g. training using stateful module),
              // those gradients must also propagate back through the network.
              transpose(m_cublas_handle,
                        RE_CAST(m_h2h_grad_workspace.dptr) +
                          layer_idx * m_num_hidden_units_x_batch_size,
                        ptr_final_hidden_grad +
                          layer_idx * m_num_hidden_units_x_batch_size,
                        param_.batch_size,
                        param_.num_hidden_units);
              transpose(m_cublas_handle,
                        RE_CAST(m_cell_grad.dptr) +
                          layer_idx * m_num_hidden_units_x_batch_size,
                        ptr_final_cell_grad +
                          layer_idx * m_num_hidden_units_x_batch_size,
                        param_.batch_size,
                        param_.num_hidden_units);
            }
            __cuda_fused_lstm_backward
              <<< dim3(param_.num_hidden_units,
                       (param_.batch_size - 1) / block_size + 1),
                  dim3(block_size),
                  (m_algo == enumBackwardReduceAlgo::PURE_ATOMICS ||
                   m_algo == enumBackwardReduceAlgo::_32_HIERARCHICAL) ?
                    0 : 4 * block_size * sizeof(float),
                  m_stream_h2h[layer_idx]
              >>> (RE_CAST(m_i2h_workspace.dptr) +
                       seq_idx * 4 * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_h2h_workspace.dptr) +
                     layer_idx * 4 * m_num_hidden_units_x_batch_size,
                   h2h_bias_grad.dptr_ +
                     layer_idx * 4 * param_.num_hidden_units,
                   RE_CAST(m_cell_grad.dptr) +
                     layer_idx * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_cell.dptr) +
                     layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                         seq_idx      * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_linear_gates.dptr) +
                     layer_idx *  param_.seq_len * 4  * m_num_hidden_units_x_batch_size +
                                         seq_idx * 4  * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_cell.dptr) +
                     layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                        (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_i2h_grad_workspace.dptr) +
                       seq_idx * m_num_hidden_units_x_batch_size,
                   param_.state_outputs ?
                     RE_CAST(m_h2h_grad_workspace.dptr) +
                       layer_idx * m_num_hidden_units_x_batch_size :
                       NULL,
                   param_.batch_size,
                   m_algo);
          } else {
            /*
            backward(
              &m_i2h_workspace  [seq_idx],
              &m_h2h_workspace[layer_idx],
              &m_bias_grad[layer_idx],
              &m_cell_grad[layer_idx],
              &m_cell[layer_idx][seq_idx],
              &m_linear_gates[layer_idx][seq_idx],
              &m_cell[layer_idx][seq_idx + 1],
              &m_i2h_grad_workspace  [seq_idx],
              &m_h2h_grad_workspace[layer_idx],
              m_stream_h2h[layer_idx], m_algo);
             */
            __cuda_fused_lstm_backward
              <<< dim3(param_.num_hidden_units,
                       (param_.batch_size - 1) / block_size + 1),
                  dim3(block_size),
                  (m_algo == enumBackwardReduceAlgo::PURE_ATOMICS ||
                   m_algo == enumBackwardReduceAlgo::_32_HIERARCHICAL) ?
                    0 : 4 * block_size * sizeof(float),
                  m_stream_h2h[layer_idx]
              >>> (RE_CAST(m_i2h_workspace.dptr) +
                       seq_idx * 4 * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_h2h_workspace.dptr) +
                     layer_idx * 4 * m_num_hidden_units_x_batch_size,
                   h2h_bias_grad.dptr_ +
                     layer_idx * 4 * param_.num_hidden_units,
                   RE_CAST(m_cell_grad.dptr) +
                     layer_idx * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_cell.dptr) +
                     layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                         seq_idx      * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_linear_gates.dptr) +
                     layer_idx *  param_.seq_len * 4  * m_num_hidden_units_x_batch_size +
                                         seq_idx * 4  * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_cell.dptr) +
                     layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                        (seq_idx + 1) * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_i2h_grad_workspace.dptr) +
                       seq_idx * m_num_hidden_units_x_batch_size,
                   RE_CAST(m_h2h_grad_workspace.dptr) +
                     layer_idx * m_num_hidden_units_x_batch_size,
                   param_.batch_size,
                   m_algo);
          }  // if (seq_idx == static_cast < int > (param_.seq_len - 1))
          // record that we are computing m_i2h_workspace
          CUDA_CALL(cudaEventRecord(m_event_h2h[layer_idx][seq_idx],
                                    m_stream_h2h[layer_idx]));
          /*
          matmul_ex(m_cublas_handle,
            &m_lX_h2h_weight_grad[layer_idx],
            &m_h2h_workspace[layer_idx],
            &m_hidden[layer_idx][seq_idx],
            CUBLAS_OP_N, CUBLAS_OP_T,
            num_gates * m_num_hidden_units,
              m_batch_size,
            m_num_hidden_units, m_batch_size,
            1.0, 1.0);
           */
          matmul_ex(m_cublas_handle,
                    h2h_weight_grad.dptr_ +
                      layer_idx * 4 * param_.num_hidden_units * param_.num_hidden_units,
                    RE_CAST(m_h2h_workspace.dptr) +
                      layer_idx * 4 * m_num_hidden_units_x_batch_size,
                    RE_CAST(m_hidden.dptr) +
                      layer_idx * (param_.seq_len + 1) * m_num_hidden_units_x_batch_size +
                                          seq_idx      * m_num_hidden_units_x_batch_size,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    4 * param_.num_hidden_units, param_.batch_size,
                        param_.num_hidden_units, param_.batch_size,
                    1.0, 1.0);
          if (seq_idx > 0) {
            /*
            matmul_ex(m_cublas_handle,
              &m_h2h_grad_workspace[layer_idx],
              &m_lX_h2h_weight[layer_idx],
              &m_h2h_workspace[layer_idx],
              CUBLAS_OP_T, CUBLAS_OP_N,
              num_gates * m_num_hidden_units,
                m_num_hidden_units,
              num_gates * m_num_hidden_units,
                m_batch_size,
              1.0, 0.0);
             */
            matmul_ex(m_cublas_handle,
              RE_CAST(m_h2h_grad_workspace.dptr) +
                layer_idx * m_num_hidden_units_x_batch_size,
              h2h_weight.dptr_ +
                layer_idx * 4 * param_.num_hidden_units * param_.num_hidden_units,
              RE_CAST(m_h2h_workspace.dptr) +
                layer_idx * 4 * m_num_hidden_units_x_batch_size,
              CUBLAS_OP_T, CUBLAS_OP_N,
              4 * param_.num_hidden_units, param_.num_hidden_units,
              4 * param_.num_hidden_units, param_.batch_size,
              1.0, 0.0);
          }  // if (seq_idx > 0)
        }  // seq_idx
        //=====================================================================
        // Input -> Hidden
        //=====================================================================
        CUBLAS_CALL(cublasSetStream(m_cublas_handle, m_stream_i2h[layer_idx]));
        // wait here until the data in m_i2h_workspace is ready
        // i2h needs to wait for h2h
        for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
          CUDA_CALL(cudaStreamWaitEvent(m_stream_i2h[layer_idx],
                                        m_event_h2h[layer_idx][seq_idx], 0));
        if (layer_idx > 0) {
          /*
          for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
          {
            // W_grad += matmul(Y, X.T)
            matmul_ex(m_cublas_handle,
              &m_lN_i2h_weight_grad[layer_idx - 1],
              &m_i2h_workspace[seq_idx],
              m_i_dp_prob != 0 ? 
                &m_i_dp_workspace[layer_idx - 1][seq_idx] :
                &m_hidden[layer_idx - 1][seq_idx + 1],
              CUBLAS_OP_N, CUBLAS_OP_T,
              num_gates * m_num_hidden_units,
                m_batch_size,
              m_num_hidden_units, m_batch_size,
              1.0, 1.0);
          }
          // X_grad = matmul(W.T, Y)
          matmul_stridedbatched_ex(m_cublas_handle,
            &m_i2h_grad_workspace[seq_rend + 1],
            &m_lN_i2h_weight[layer_idx - 1],
            &m_i2h_workspace[seq_rend + 1],
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_gates * m_num_hidden_units,
              m_num_hidden_units,
            num_gates * m_num_hidden_units,
              m_batch_size,
            m_num_hidden_units * m_batch_size,
            0, 
            num_gates * m_num_hidden_units *
              m_batch_size,
            seq_rbegin - seq_rend,
            1.0, 0.0);
          
          if (m_i_dp_prob != 0)
          {
            m_i_dp_handle->backward(
              &m_i2h_grad_workspace[seq_rend + 1],
              &m_i2h_grad_workspace[seq_rend + 1],
              &m_i_dp_uniform_rv[layer_idx - 1][seq_rend + 1],
              m_stream_i2h[layer_idx], seq_rbegin - seq_rend);
          }
           */
          for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
            matmul_ex(m_cublas_handle,
                      i2h_weight_grad.dptr_ +
                        4 * param_.num_hidden_units * param_.embed_dim +
                        (layer_idx - 1) * 4 * param_.num_hidden_units * param_.num_hidden_units,
                      RE_CAST(m_i2h_workspace.dptr) +
                        seq_idx * 4 * m_num_hidden_units_x_batch_size,
                      param_.i_dp_prob != 0 ?
                        RE_CAST(m_i_dp_workspace.dptr) +
                          (layer_idx - 1) * param_.seq_len * m_num_hidden_units_x_batch_size +
                                                   seq_idx * m_num_hidden_units_x_batch_size :
                        RE_CAST(m_hidden.dptr) +
                          (layer_idx - 1) * (param_.seq_len + 1) *
                            m_num_hidden_units_x_batch_size +
                                                   (seq_idx + 1) *
                            m_num_hidden_units_x_batch_size,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        4 * param_.num_hidden_units, param_.batch_size,
                            param_.num_hidden_units, param_.batch_size,
                        1.0, 1.0);
          matmul_stridedbatched_ex(m_cublas_handle,
                                   RE_CAST(m_i2h_grad_workspace.dptr) +
                                     (seq_rend + 1) * m_num_hidden_units_x_batch_size,
                                   i2h_weight.dptr_ +
                                     4 * param_.num_hidden_units * param_.embed_dim +
                                     (layer_idx - 1) * 4 * param_.num_hidden_units *
                                       param_.num_hidden_units,
                                   RE_CAST(m_i2h_workspace.dptr) +
                                     (seq_rend + 1) * 4 * m_num_hidden_units_x_batch_size,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   4 * param_.num_hidden_units, param_.num_hidden_units,
                                   4 * param_.num_hidden_units, param_.batch_size,
                                   m_num_hidden_units_x_batch_size,
                                   0,
                                   4 * m_num_hidden_units_x_batch_size,
                                   seq_rbegin - seq_rend,
                                   1.0, 0.0);
          if (param_.i_dp_prob != 0) {
            __cuda_dropout_backward
              <<< (m_num_hidden_units_x_batch_size * (seq_rbegin - seq_rend) - 1) / 128 + 1,
                  128,
                  0,
                  m_stream_i2h[layer_idx]
              >>> (
                RE_CAST(m_i2h_grad_workspace.dptr) +
                  (seq_rend + 1) * m_num_hidden_units_x_batch_size,
                RE_CAST(m_i2h_grad_workspace.dptr) +
                  (seq_rend + 1) * m_num_hidden_units_x_batch_size,
                RE_CAST(m_i_dp_uniform_rv.dptr) +
                  (layer_idx - 1) * param_.seq_len * m_num_hidden_units_x_batch_size +
                                    (seq_rend + 1) * m_num_hidden_units_x_batch_size,
                param_.i_dp_prob, m_num_hidden_units_x_batch_size * (seq_rbegin - seq_rend));
          }  // if (param_.i_dp_prob != 0)
          // record that we are computing m_i2h_grad_workspace
          for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
            CUDA_CALL(cudaEventRecord(m_event_i2h[layer_idx][seq_idx],
                                      m_stream_i2h[layer_idx]));
        } else {
          /*
          for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
          {
            // W_grad += matmul(Y, X.T)
            matmul_ex(m_cublas_handle,
               m_l0_i2h_weight_grad,
              &m_i2h_workspace[seq_idx],
              &m_data_T_major[seq_idx],
              CUBLAS_OP_N, CUBLAS_OP_T,
              num_gates * m_num_hidden_units,
                m_batch_size,
              m_embed_dim, m_batch_size,
              1.0, 1.0);
          }
          // X_grad = matmul(W.T, Y)
          matmul_stridedbatched_ex(m_cublas_handle,
            &m_data_T_major_grad[seq_rend + 1],
             m_l0_i2h_weight,
            &m_i2h_workspace[seq_rend + 1],
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_gates * m_num_hidden_units,
              m_embed_dim,
            num_gates * m_num_hidden_units,
              m_batch_size,
            m_embed_dim * m_batch_size,
            0, 
            num_gates * m_num_hidden_units * 
              m_batch_size,
            seq_rbegin - seq_rend,
            1.0, 0.0);
           */
          for (int seq_idx = seq_rbegin; seq_idx > seq_rend; --seq_idx)
            matmul_ex(m_cublas_handle,
                      i2h_weight_grad.dptr_,
                      RE_CAST(m_i2h_workspace.dptr) +
                        (seq_rend + 1) * 4 * m_num_hidden_units_x_batch_size,
                      RE_CAST(m_data_T_major.dptr) +
                        (seq_rend + 1) * param_.embed_dim * param_.batch_size,
                      CUBLAS_OP_N, CUBLAS_OP_T,
                      4 * param_.num_hidden_units, param_.batch_size,
                          param_.embed_dim       , param_.batch_size,
                      1.0, 1.0);
          matmul_stridedbatched_ex(m_cublas_handle,
                                   RE_CAST(m_data_T_major_grad.dptr) +
                                     (seq_rend + 1) * param_.embed_dim * param_.batch_size,
                                   i2h_weight.dptr_,
                                   RE_CAST(m_i2h_workspace.dptr) +
                                     (seq_rend + 1) * 4 * m_num_hidden_units_x_batch_size,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   4 * param_.num_hidden_units, param_.embed_dim,
                                   4 * param_.num_hidden_units, param_.batch_size,
                                   param_.embed_dim * param_.batch_size,
                                   0,
                                   4 * m_num_hidden_units_x_batch_size,
                                   seq_rbegin - seq_rend,
                                   1.0, 0.0);
        }  // if (layer_idx > 0)
      }  // layer_idx
    }  // schedule
    CUDA_CALL(cudaMemcpyAsync(i2h_bias_grad.dptr_,
                              h2h_bias_grad.dptr_,
                              param_.num_layers * 4 * param_.num_hidden_units * sizeof(float),
                              cudaMemcpyDeviceToDevice,
                              m_stream_h2h[0]));
    /*
    transpose(m_cublas_handle,
      m_data_B_major_grad, m_data_T_major_grad,
      m_seq_len * m_embed_dim, m_batch_size);
     */
    transpose(m_cublas_handle,
              data_grad.dptr_,
              RE_CAST(m_data_T_major_grad.dptr),
              param_.seq_len * param_.embed_dim,
              param_.batch_size);
    CUDA_CALL(cudaStreamSynchronize(m_stream_i2h[0]));
    CUDA_CALL(cudaStreamSynchronize(m_stream_h2h[0]));
  }

 private:
  void Init(mshadow::Stream<gpu> *s,
            const std::vector<TBlob> &in_data,
            const std::vector<TBlob> &out_data) {
    using namespace mshadow;

    std::size_t in_expected = 7, out_expected = param_.state_outputs ? 3 : 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);

    if (!initialized_) {
      initialized_ = true;
      Tensor<gpu, 3, float> data = in_data[open_lstm_rnn_enum::kData].get<gpu, 3, float>(s);
      //=======================================================================
      // infer the parameters from the input data
      param_.batch_size = data.shape_[0];
      param_.seq_len    = data.shape_[1];
      param_.embed_dim  = data.shape_[2];
      m_num_hidden_units_x_batch_size = param_.batch_size * param_.num_hidden_units;
      //=======================================================================
      // allocate workspace
      m_data_T_major       = Storage::Get()->Alloc(param_.seq_len * param_.embed_dim *
                                                     param_.batch_size * sizeof(float),
                                                   Context::GPU());
      m_data_T_major_grad  = Storage::Get()->Alloc(param_.seq_len * param_.embed_dim *
                                                     param_.batch_size * sizeof(float),
                                                   Context::GPU());
      m_cell               = Storage::Get()->Alloc(param_.num_layers * (param_.seq_len + 1) *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_hidden             = Storage::Get()->Alloc(param_.num_layers * (param_.seq_len + 1) *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_cell_grad          = Storage::Get()->Alloc(param_.num_layers *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_i2h_workspace      = Storage::Get()->Alloc(param_.seq_len    * 4 *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_h2h_workspace      = Storage::Get()->Alloc(param_.num_layers * 4 *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_i2h_grad_workspace = Storage::Get()->Alloc(param_.seq_len *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_h2h_grad_workspace = Storage::Get()->Alloc(param_.num_layers *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      m_linear_gates       = Storage::Get()->Alloc(param_.num_layers * param_.seq_len * 4 *
                                                     m_num_hidden_units_x_batch_size *
                                                     sizeof(float),
                                                   Context::GPU());
      //=======================================================================
      // initialize forward and backward compute schedule
       m_forward_schedule = ScheduleList(param_.num_layers, param_.seq_len);
      m_backward_schedule = ScheduleList(param_.num_layers, param_.seq_len);
      m_forward_schedule.init(1); m_backward_schedule.init(1);
      //=======================================================================
      // initialize workers (cudaStream and cudaEvent)
      CUBLAS_CALL(cublasCreate(&m_cublas_handle));
      m_stream_i2h = new cudaStream_t[param_.num_layers];
      m_stream_h2h = new cudaStream_t[param_.num_layers];
      for (unsigned layer_idx = 0; layer_idx < param_.num_layers; ++layer_idx) {
        CUDA_CALL(cudaStreamCreateWithPriority(&m_stream_i2h[layer_idx],
                                               cudaStreamNonBlocking,
                                               0));
        CUDA_CALL(cudaStreamCreateWithPriority(&m_stream_h2h[layer_idx],
                                               cudaStreamNonBlocking,
                                               -1));
      }
      m_event_i2h = new cudaEvent_t*[param_.num_layers];
      m_event_h2h = new cudaEvent_t*[param_.num_layers];
      for (unsigned layer_idx = 0; layer_idx < param_.num_layers; ++layer_idx) {
        m_event_i2h[layer_idx] = new cudaEvent_t[param_.seq_len];
        m_event_h2h[layer_idx] = new cudaEvent_t[param_.seq_len];
        for (unsigned seq_idx = 0; seq_idx < param_.seq_len; ++seq_idx) {
          CUDA_CALL(cudaEventCreateWithFlags(&m_event_i2h[layer_idx][seq_idx],
                                             cudaEventDisableTiming));
          CUDA_CALL(cudaEventCreateWithFlags(&m_event_h2h[layer_idx][seq_idx],
                                             cudaEventDisableTiming));
        }
      }
      //=======================================================================
      // determine the algorithm for backward pass
      if ((param_.batch_size % 128) <= 32) {
        m_algo = enumBackwardReduceAlgo:: _32_HIERARCHICAL;
      } else if ((param_.batch_size % 128) <= 64) {
        m_algo = enumBackwardReduceAlgo:: _64_HIERARCHICAL;
      } else {
        m_algo = enumBackwardReduceAlgo:: _128_HIERARCHICAL;
      }
      //=======================================================================
      // initialize input dropout random variable, if needed
      if (param_.i_dp_prob != 0 && param_.num_layers > 1) {
        CURAND_CALL(curandCreateGenerator(&m_rng,
                                          CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_rng,
                                                       time(nullptr)));
        m_i_dp_uniform_rv = Storage::Get()->Alloc((param_.num_layers - 1) * param_.seq_len *
                                                    m_num_hidden_units_x_batch_size *
                                                    sizeof(float),
                                                  Context::GPU());
        m_i_dp_workspace  = Storage::Get()->Alloc((param_.num_layers - 1) * param_.seq_len *
                                                    m_num_hidden_units_x_batch_size *
                                                    sizeof(float),
                                                  Context::GPU());
      }  // if (param_.i_dp_prob != 0 && param_.num_layers > 1)
    }  // if (!initialized_)
  }

  bool initialized_; OpenLSTMRNNParam param_;

  struct Schedule {
    unsigned m_layer_begin, m_layer_end, m_seq_begin, m_seq_end;
  };
  struct ScheduleList : public std::vector<Schedule> {
   private:
    unsigned m_num_layers, m_seq_len;

   public:
    ScheduleList() {}
    ScheduleList(const unsigned & num_layers, const unsigned & seq_len) :
      m_num_layers(num_layers), m_seq_len(seq_len)
    {}
    // Initialize the compute schedule, an essential part for wavefront parallelism.
    void init(const unsigned & schedule_time_stride) {
      for (int layer_begin = 0, layer_end = 0, seq_begin = 0, seq_end = 0; ;) {
        if (layer_end == 0) {
          layer_begin = 0; layer_end = 1; seq_begin = 0;
        } else {
          // move up and left
          ++layer_begin; ++layer_end;
          seq_begin -= schedule_time_stride;
          // over the top or off the left, reset to layer 0
          if (layer_end > static_cast < int > (m_num_layers) || seq_begin < 0) {
            seq_begin += (layer_begin + 1) * schedule_time_stride;
            layer_begin = 0; layer_end = 1;
          }
          while (seq_begin >= static_cast < int > (m_seq_len) &&
                 layer_end <= static_cast < int > (m_num_layers)) {
            ++layer_begin; ++layer_end;
            seq_begin -= schedule_time_stride;
          }
          // over the top or off the left -> DONE!
          if (layer_end > static_cast < int > (m_num_layers) || seq_begin < 0)
            break;
        }  // if (layer_end == 0)
        seq_end = seq_begin + schedule_time_stride;
        // prevent overflow
        if (seq_end > static_cast < int > (m_seq_len))
          seq_end = m_seq_len;
        //==============================================================
        // End of Scheduling
        //==============================================================
        Schedule schedule;
        schedule.m_layer_begin = layer_begin; schedule.m_layer_end = layer_end;
        schedule.m_seq_begin = seq_begin; schedule.m_seq_end = seq_end;
        this->push_back(schedule);
      }
    }
  } m_forward_schedule, m_backward_schedule;

  unsigned m_num_hidden_units_x_batch_size;

  Storage::Handle m_data_T_major, m_data_T_major_grad,
                  m_cell, m_hidden,
                  m_cell_grad,
                  m_i2h_workspace,
                  m_h2h_workspace,
                  m_i2h_grad_workspace,
                  m_h2h_grad_workspace,
                  m_linear_gates;

  curandGenerator_t m_rng;
  Storage::Handle m_i_dp_uniform_rv,
                  m_i_dp_workspace;

  cudaStream_t *m_stream_i2h, *m_stream_h2h;
  cudaEvent_t **m_event_i2h, **m_event_h2h;

  cublasHandle_t m_cublas_handle;

  enumBackwardReduceAlgo m_algo;
};  // CUOpenLSTMRNNOp

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CU_OPEN_LSTM_RNN_INL_CUH_
