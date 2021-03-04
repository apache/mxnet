/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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
*******************************************************************************/

/// @example cpu_rnn_inference_f32.cpp
/// @copybrief cpu_rnn_inference_f32_cpp
/// > Annotated version: @ref cpu_rnn_inference_f32_cpp

/// @page cpu_rnn_inference_f32_cpp RNN f32 inference example
/// This C++ API example demonstrates how to build GNMT model inference.
///
/// > Example code: @ref cpu_rnn_inference_f32.cpp
///
/// For the encoder we use:
///  - one primitive for the bidirectional layer of the encoder
///  - one primitive for all remaining unidirectional layers in the encoder
/// For the decoder we use:
///  - one primitive for the first iteration
///  - one primitive for all subsequent iterations in the decoder. Note that
///    in this example, this primitive computes the states in place.
///  - the attention mechanism is implemented separately as there is no support
///    for the context vectors in oneDNN yet

#include <assert.h>

#include <cstring>
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

using dim_t = dnnl::memory::dim;

const dim_t batch = 128;
const dim_t src_seq_length_max = 28;
const dim_t tgt_seq_length_max = 28;

const dim_t feature_size = 1024;

const dim_t enc_bidir_n_layers = 1;
const dim_t enc_unidir_n_layers = 7;
const dim_t dec_n_layers = 8;

const int lstm_n_gates = 4;
std::vector<float> weighted_src_layer(batch *feature_size, 1.0f);
std::vector<float> alignment_model(
        src_seq_length_max *batch *feature_size, 1.0f);
std::vector<float> alignments(src_seq_length_max *batch, 1.0f);
std::vector<float> exp_sums(batch, 1.0f);

void compute_weighted_annotations(float *weighted_annotations,
        dim_t src_seq_length_max, dim_t batch, dim_t feature_size,
        float *weights_annot, float *annotations) {
    // annotations(aka enc_dst_layer) is (t, n, 2c)
    // weights_annot is (2c, c)

    // annotation[i] = GEMM(weights_annot, enc_dst_layer[i]);
    dim_t num_weighted_annotations = src_seq_length_max * batch;
    dnnl_sgemm('N', 'N', num_weighted_annotations, feature_size, feature_size,
            1.f, annotations, feature_size, weights_annot, feature_size, 0.f,
            weighted_annotations, feature_size);
}

void compute_attention(float *context_vectors, dim_t src_seq_length_max,
        dim_t batch, dim_t feature_size, float *weights_src_layer,
        float *dec_src_layer, float *annotations, float *weighted_annotations,
        float *weights_alignments) {
    // dst_iter : (n, c) matrix
    // src_layer: (n, c) matrix
    // weighted_annotations (t, n, c)

    // weights_yi is (c, c)
    // weights_ai is (c, 1)
    // tmp[i] is (n, c)
    // a[i] is (n, 1)
    // p is (n, 1)

    // first we precompute the weighted_dec_src_layer
    dnnl_sgemm('N', 'N', batch, feature_size, feature_size, 1.f, dec_src_layer,
            feature_size, weights_src_layer, feature_size, 0.f,
            weighted_src_layer.data(), feature_size);

    // then we compute the alignment model
    float *alignment_model_ptr = alignment_model.data();

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (dim_t i = 0; i < src_seq_length_max; i++) {
        for (dim_t j = 0; j < batch * feature_size; j++)
            alignment_model_ptr[i * batch * feature_size + j] = tanhf(
                    weighted_src_layer[j]
                    + weighted_annotations[i * batch * feature_size + j]);
    }

    // gemv with alignments weights. the resulting alignments are in alignments
    dim_t num_weighted_annotations = src_seq_length_max * batch;
    dnnl_sgemm('N', 'N', num_weighted_annotations, 1, feature_size, 1.f,
            alignment_model_ptr, feature_size, weights_alignments, 1, 0.f,
            alignments.data(), 1);

    // softmax on alignments. the resulting context weights are in alignments
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
    for (dim_t i = 0; i < batch; i++)
        exp_sums[i] = 0.0f;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (dim_t i = 0; i < src_seq_length_max; i++) {
        for (dim_t j = 0; j < batch; j++) {
            alignments[i * batch + j] = expf(alignments[i * batch + j]);
            exp_sums[j] += alignments[i * batch + j];
        }
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (dim_t i = 0; i < src_seq_length_max; i++)
        for (dim_t j = 0; j < batch; j++)
            alignments[i * batch + j] /= exp_sums[j];

    // then we compute the context vectors
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (dim_t i = 0; i < batch; i++)
        for (dim_t j = 0; j < feature_size; j++)
            context_vectors[i * (feature_size + feature_size) + feature_size
                    + j]
                    = 0.0f;

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (dim_t i = 0; i < batch; i++)
        for (dim_t k = 0; k < src_seq_length_max; k++)
            for (dim_t j = 0; j < feature_size; j++)
                context_vectors[i * (feature_size + feature_size) + feature_size
                        + j]
                        += alignments[k * batch + i]
                        * annotations[j + feature_size * (i + batch * k)];
}

void copy_context(
        float *src_iter, dim_t n_layers, dim_t batch, dim_t feature_size) {
    // we copy the context from the first layer to all other layers
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (dim_t k = 1; k < n_layers; k++)
        for (dim_t j = 0; j < batch; j++)
            for (dim_t i = 0; i < feature_size; i++)
                src_iter[(k * batch + j) * (feature_size + feature_size)
                        + feature_size + i]
                        = src_iter[j * (feature_size + feature_size)
                                + feature_size + i];
}

void simple_net() {
    ///
    /// Initialize a CPU engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cpu_rnn_inference_f32.cpp Initialize engine and stream
    ///
    //[Initialize engine and stream]
    auto cpu_engine = engine(engine::kind::cpu, 0);
    stream s(cpu_engine);
    //[Initialize engine and stream]
    ///
    /// Declare encoder net and decoder net
    /// @snippet cpu_rnn_inference_f32.cpp declare net
    ///
    //[declare net]
    std::vector<primitive> encoder_net, decoder_net;
    std::vector<std::unordered_map<int, memory>> encoder_net_args,
            decoder_net_args;

    std::vector<float> net_src(batch * src_seq_length_max * feature_size, 1.0f);
    std::vector<float> net_dst(batch * tgt_seq_length_max * feature_size, 1.0f);
    //[declare net]
    ///
    /// **Encoder**
    ///
    ///
    /// Initialize Encoder Memory
    /// @snippet cpu_rnn_inference_f32.cpp Initialize encoder memory
    ///
    //[Initialize encoder memory]
    memory::dims enc_bidir_src_layer_tz
            = {src_seq_length_max, batch, feature_size};
    memory::dims enc_bidir_weights_layer_tz
            = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
    memory::dims enc_bidir_weights_iter_tz
            = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
    memory::dims enc_bidir_bias_tz
            = {enc_bidir_n_layers, 2, lstm_n_gates, feature_size};
    memory::dims enc_bidir_dst_layer_tz
            = {src_seq_length_max, batch, 2 * feature_size};
    //[Initialize encoder memory]

    ///
    ///
    /// Encoder: 1 bidirectional layer and 7 unidirectional layers
    ///

    std::vector<float> user_enc_bidir_wei_layer(
            enc_bidir_n_layers * 2 * feature_size * lstm_n_gates * feature_size,
            1.0f);
    std::vector<float> user_enc_bidir_wei_iter(
            enc_bidir_n_layers * 2 * feature_size * lstm_n_gates * feature_size,
            1.0f);
    std::vector<float> user_enc_bidir_bias(
            enc_bidir_n_layers * 2 * lstm_n_gates * feature_size, 1.0f);

    ///
    /// Create the memory for user data
    /// @snippet cpu_rnn_inference_f32.cpp data memory creation
    ///
    //[data memory creation]
    auto user_enc_bidir_src_layer_md = dnnl::memory::desc(
            {enc_bidir_src_layer_tz}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::tnc);

    auto user_enc_bidir_wei_layer_md = dnnl::memory::desc(
            {enc_bidir_weights_layer_tz}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);

    auto user_enc_bidir_wei_iter_md = dnnl::memory::desc(
            {enc_bidir_weights_iter_tz}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);

    auto user_enc_bidir_bias_md = dnnl::memory::desc({enc_bidir_bias_tz},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldgo);

    auto user_enc_bidir_src_layer_memory = dnnl::memory(
            user_enc_bidir_src_layer_md, cpu_engine, net_src.data());
    auto user_enc_bidir_wei_layer_memory
            = dnnl::memory(user_enc_bidir_wei_layer_md, cpu_engine,
                    user_enc_bidir_wei_layer.data());
    auto user_enc_bidir_wei_iter_memory
            = dnnl::memory(user_enc_bidir_wei_iter_md, cpu_engine,
                    user_enc_bidir_wei_iter.data());
    auto user_enc_bidir_bias_memory = dnnl::memory(
            user_enc_bidir_bias_md, cpu_engine, user_enc_bidir_bias.data());

    //[data memory creation]
    ///
    /// Create memory descriptors for RNN data w/o specified layout
    /// @snippet cpu_rnn_inference_f32.cpp memory desc for RNN data
    ///
    //[memory desc for RNN data]
    auto enc_bidir_wei_layer_md = memory::desc({enc_bidir_weights_layer_tz},
            memory::data_type::f32, memory::format_tag::any);

    auto enc_bidir_wei_iter_md = memory::desc({enc_bidir_weights_iter_tz},
            memory::data_type::f32, memory::format_tag::any);

    auto enc_bidir_dst_layer_md = memory::desc({enc_bidir_dst_layer_tz},
            memory::data_type::f32, memory::format_tag::any);

    //[memory desc for RNN data]
    ///
    /// Create bidirectional RNN
    /// @snippet cpu_rnn_inference_f32.cpp create rnn
    ///
    //[create rnn]

    lstm_forward::desc bi_layer_desc(prop_kind::forward_inference,
            rnn_direction::bidirectional_concat, user_enc_bidir_src_layer_md,
            memory::desc(), memory::desc(), enc_bidir_wei_layer_md,
            enc_bidir_wei_iter_md, user_enc_bidir_bias_md,
            enc_bidir_dst_layer_md, memory::desc(), memory::desc());

    auto enc_bidir_prim_desc
            = dnnl::lstm_forward::primitive_desc(bi_layer_desc, cpu_engine);
    //[create rnn]

    ///
    /// Create memory for input data and use reorders to reorder user data
    /// to internal representation
    /// @snippet cpu_rnn_inference_f32.cpp reorder input data
    ///
    //[reorder input data]
    auto enc_bidir_wei_layer_memory
            = memory(enc_bidir_prim_desc.weights_layer_desc(), cpu_engine);
    auto enc_bidir_wei_layer_reorder_pd = reorder::primitive_desc(
            user_enc_bidir_wei_layer_memory, enc_bidir_wei_layer_memory);
    reorder(enc_bidir_wei_layer_reorder_pd)
            .execute(s, user_enc_bidir_wei_layer_memory,
                    enc_bidir_wei_layer_memory);
    //[reorder input data]

    auto enc_bidir_wei_iter_memory
            = memory(enc_bidir_prim_desc.weights_iter_desc(), cpu_engine);
    auto enc_bidir_wei_iter_reorder_pd = reorder::primitive_desc(
            user_enc_bidir_wei_iter_memory, enc_bidir_wei_iter_memory);
    reorder(enc_bidir_wei_iter_reorder_pd)
            .execute(s, user_enc_bidir_wei_iter_memory,
                    enc_bidir_wei_iter_memory);

    auto enc_bidir_dst_layer_memory
            = dnnl::memory(enc_bidir_prim_desc.dst_layer_desc(), cpu_engine);

    ///
    /// Encoder : add the bidirectional rnn primitive with related arguments into encoder_net
    /// @snippet cpu_rnn_inference_f32.cpp push bi rnn to encoder net
    ///
    //[push bi rnn to encoder net]
    encoder_net.push_back(lstm_forward(enc_bidir_prim_desc));
    encoder_net_args.push_back(
            {{DNNL_ARG_SRC_LAYER, user_enc_bidir_src_layer_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, enc_bidir_wei_layer_memory},
                    {DNNL_ARG_WEIGHTS_ITER, enc_bidir_wei_iter_memory},
                    {DNNL_ARG_BIAS, user_enc_bidir_bias_memory},
                    {DNNL_ARG_DST_LAYER, enc_bidir_dst_layer_memory}});
    //[push bi rnn to encoder net]

    ///
    /// Encoder: unidirectional layers
    ///
    ///
    /// First unidirectinal layer scales 2 * feature_size output of bidirectional
    /// layer to feature_size output
    /// @snippet cpu_rnn_inference_f32.cpp first uni layer
    ///
    //[first uni layer]
    std::vector<float> user_enc_uni_first_wei_layer(
            1 * 1 * 2 * feature_size * lstm_n_gates * feature_size, 1.0f);
    std::vector<float> user_enc_uni_first_wei_iter(
            1 * 1 * feature_size * lstm_n_gates * feature_size, 1.0f);
    std::vector<float> user_enc_uni_first_bias(
            1 * 1 * lstm_n_gates * feature_size, 1.0f);
    //[first uni layer]
    memory::dims user_enc_uni_first_wei_layer_dims
            = {1, 1, 2 * feature_size, lstm_n_gates, feature_size};
    memory::dims user_enc_uni_first_wei_iter_dims
            = {1, 1, feature_size, lstm_n_gates, feature_size};
    memory::dims user_enc_uni_first_bias_dims
            = {1, 1, lstm_n_gates, feature_size};
    memory::dims enc_uni_first_dst_layer_dims
            = {src_seq_length_max, batch, feature_size};
    auto user_enc_uni_first_wei_layer_md = dnnl::memory::desc(
            {user_enc_uni_first_wei_layer_dims}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);
    auto user_enc_uni_first_wei_iter_md = dnnl::memory::desc(
            {user_enc_uni_first_wei_iter_dims}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);
    auto user_enc_uni_first_bias_md = dnnl::memory::desc(
            {user_enc_uni_first_bias_dims}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldgo);
    auto user_enc_uni_first_wei_layer_memory
            = dnnl::memory(user_enc_uni_first_wei_layer_md, cpu_engine,
                    user_enc_uni_first_wei_layer.data());
    auto user_enc_uni_first_wei_iter_memory
            = dnnl::memory(user_enc_uni_first_wei_iter_md, cpu_engine,
                    user_enc_uni_first_wei_iter.data());
    auto user_enc_uni_first_bias_memory
            = dnnl::memory(user_enc_uni_first_bias_md, cpu_engine,
                    user_enc_uni_first_bias.data());

    auto enc_uni_first_wei_layer_md
            = memory::desc({user_enc_uni_first_wei_layer_dims},
                    memory::data_type::f32, memory::format_tag::any);
    auto enc_uni_first_wei_iter_md
            = memory::desc({user_enc_uni_first_wei_iter_dims},
                    memory::data_type::f32, memory::format_tag::any);
    auto enc_uni_first_dst_layer_md
            = memory::desc({enc_uni_first_dst_layer_dims},
                    memory::data_type::f32, memory::format_tag::any);

    // TODO: add support for residual connections
    // should it be a set residual in op_desc or a field to set manually?
    // should be an integer to specify at which layer to start
    ///
    /// Encoder : Create unidirection RNN for first cell
    /// @snippet cpu_rnn_inference_f32.cpp create uni first
    ///
    //[create uni first]
    lstm_forward::desc enc_uni_first_layer_desc(prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right, enc_bidir_dst_layer_md,
            memory::desc(), memory::desc(), enc_uni_first_wei_layer_md,
            enc_uni_first_wei_iter_md, user_enc_uni_first_bias_md,
            enc_uni_first_dst_layer_md, memory::desc(), memory::desc());

    auto enc_uni_first_prim_desc = dnnl::lstm_forward::primitive_desc(
            enc_uni_first_layer_desc, cpu_engine);

    //[create uni first]
    auto enc_uni_first_wei_layer_memory
            = memory(enc_uni_first_prim_desc.weights_layer_desc(), cpu_engine);
    auto enc_uni_first_wei_layer_reorder_pd
            = reorder::primitive_desc(user_enc_uni_first_wei_layer_memory,
                    enc_uni_first_wei_layer_memory);
    reorder(enc_uni_first_wei_layer_reorder_pd)
            .execute(s, user_enc_uni_first_wei_layer_memory,
                    enc_uni_first_wei_layer_memory);

    auto enc_uni_first_wei_iter_memory
            = memory(enc_uni_first_prim_desc.weights_iter_desc(), cpu_engine);
    auto enc_uni_first_wei_iter_reorder_pd = reorder::primitive_desc(
            user_enc_uni_first_wei_iter_memory, enc_uni_first_wei_iter_memory);
    reorder(enc_uni_first_wei_iter_reorder_pd)
            .execute(s, user_enc_uni_first_wei_iter_memory,
                    enc_uni_first_wei_iter_memory);

    auto enc_uni_first_dst_layer_memory = dnnl::memory(
            enc_uni_first_prim_desc.dst_layer_desc(), cpu_engine);

    /// Encoder : add the first unidirectional rnn primitive with related
    /// arguments into encoder_net
    ///
    /// @snippet cpu_rnn_inference_f32.cpp push first uni rnn to encoder net
    ///
    //[push first uni rnn to encoder net]
    // TODO: add a reorder when they will be available
    encoder_net.push_back(lstm_forward(enc_uni_first_prim_desc));
    encoder_net_args.push_back(
            {{DNNL_ARG_SRC_LAYER, enc_bidir_dst_layer_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, enc_uni_first_wei_layer_memory},
                    {DNNL_ARG_WEIGHTS_ITER, enc_uni_first_wei_iter_memory},
                    {DNNL_ARG_BIAS, user_enc_uni_first_bias_memory},
                    {DNNL_ARG_DST_LAYER, enc_uni_first_dst_layer_memory}});
    //[push first uni rnn to encoder net]

    ///
    /// Encoder : Remaining unidirectional layers
    /// @snippet cpu_rnn_inference_f32.cpp remaining uni layers
    ///
    //[remaining uni layers]
    std::vector<float> user_enc_uni_wei_layer((enc_unidir_n_layers - 1) * 1
                    * feature_size * lstm_n_gates * feature_size,
            1.0f);
    std::vector<float> user_enc_uni_wei_iter((enc_unidir_n_layers - 1) * 1
                    * feature_size * lstm_n_gates * feature_size,
            1.0f);
    std::vector<float> user_enc_uni_bias(
            (enc_unidir_n_layers - 1) * 1 * lstm_n_gates * feature_size, 1.0f);
    //[remaining uni layers]
    memory::dims user_enc_uni_wei_layer_dims = {(enc_unidir_n_layers - 1), 1,
            feature_size, lstm_n_gates, feature_size};
    memory::dims user_enc_uni_wei_iter_dims = {(enc_unidir_n_layers - 1), 1,
            feature_size, lstm_n_gates, feature_size};
    memory::dims user_enc_uni_bias_dims
            = {(enc_unidir_n_layers - 1), 1, lstm_n_gates, feature_size};
    memory::dims enc_dst_layer_dims = {src_seq_length_max, batch, feature_size};
    auto user_enc_uni_wei_layer_md = dnnl::memory::desc(
            {user_enc_uni_wei_layer_dims}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);
    auto user_enc_uni_wei_iter_md = dnnl::memory::desc(
            {user_enc_uni_wei_iter_dims}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ldigo);
    auto user_enc_uni_bias_md = dnnl::memory::desc({user_enc_uni_bias_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldgo);
    auto user_enc_uni_wei_layer_memory = dnnl::memory(user_enc_uni_wei_layer_md,
            cpu_engine, user_enc_uni_wei_layer.data());
    auto user_enc_uni_wei_iter_memory = dnnl::memory(
            user_enc_uni_wei_iter_md, cpu_engine, user_enc_uni_wei_iter.data());
    auto user_enc_uni_bias_memory = dnnl::memory(
            user_enc_uni_bias_md, cpu_engine, user_enc_uni_bias.data());

    auto enc_uni_wei_layer_md = memory::desc({user_enc_uni_wei_layer_dims},
            memory::data_type::f32, memory::format_tag::any);
    auto enc_uni_wei_iter_md = memory::desc({user_enc_uni_wei_iter_dims},
            memory::data_type::f32, memory::format_tag::any);
    auto enc_dst_layer_md = memory::desc({enc_dst_layer_dims},
            memory::data_type::f32, memory::format_tag::any);

    // TODO: add support for residual connections
    // should it be a set residual in op_desc or a field to set manually?
    // should be an integer to specify at which layer to start
    ///
    /// Encoder : Create unidirection RNN cell
    /// @snippet cpu_rnn_inference_f32.cpp create uni rnn
    ///
    //[create uni rnn]
    lstm_forward::desc enc_uni_layer_desc(prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right,
            enc_uni_first_dst_layer_md, memory::desc(), memory::desc(),
            enc_uni_wei_layer_md, enc_uni_wei_iter_md, user_enc_uni_bias_md,
            enc_dst_layer_md, memory::desc(), memory::desc());
    auto enc_uni_prim_desc = dnnl::lstm_forward::primitive_desc(
            enc_uni_layer_desc, cpu_engine);
    //[create uni rnn]

    auto enc_uni_wei_layer_memory
            = memory(enc_uni_prim_desc.weights_layer_desc(), cpu_engine);
    auto enc_uni_wei_layer_reorder_pd = reorder::primitive_desc(
            user_enc_uni_wei_layer_memory, enc_uni_wei_layer_memory);
    reorder(enc_uni_wei_layer_reorder_pd)
            .execute(
                    s, user_enc_uni_wei_layer_memory, enc_uni_wei_layer_memory);

    auto enc_uni_wei_iter_memory
            = memory(enc_uni_prim_desc.weights_iter_desc(), cpu_engine);
    auto enc_uni_wei_iter_reorder_pd = reorder::primitive_desc(
            user_enc_uni_wei_iter_memory, enc_uni_wei_iter_memory);
    reorder(enc_uni_wei_iter_reorder_pd)
            .execute(s, user_enc_uni_wei_iter_memory, enc_uni_wei_iter_memory);

    auto enc_dst_layer_memory
            = dnnl::memory(enc_uni_prim_desc.dst_layer_desc(), cpu_engine);

    // TODO: add a reorder when they will be available
    ///
    /// Encoder : add the unidirectional rnn primitive with related arguments into encoder_net
    /// @snippet cpu_rnn_inference_f32.cpp push uni rnn to encoder net
    ///
    //[push uni rnn to encoder net]
    encoder_net.push_back(lstm_forward(enc_uni_prim_desc));
    encoder_net_args.push_back(
            {{DNNL_ARG_SRC_LAYER, enc_uni_first_dst_layer_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, enc_uni_wei_layer_memory},
                    {DNNL_ARG_WEIGHTS_ITER, enc_uni_wei_iter_memory},
                    {DNNL_ARG_BIAS, user_enc_uni_bias_memory},
                    {DNNL_ARG_DST_LAYER, enc_dst_layer_memory}});
    //[push uni rnn to encoder net]
    ///
    /// **Decoder with attention mechanism**
    ///
    ///
    /// Decoder : declare memory dimensions
    /// @snippet cpu_rnn_inference_f32.cpp dec mem dim
    ///
    //[dec mem dim]
    std::vector<float> user_dec_wei_layer(
            dec_n_layers * 1 * feature_size * lstm_n_gates * feature_size,
            1.0f);
    std::vector<float> user_dec_wei_iter(dec_n_layers * 1
                    * (feature_size + feature_size) * lstm_n_gates
                    * feature_size,
            1.0f);
    std::vector<float> user_dec_bias(
            dec_n_layers * 1 * lstm_n_gates * feature_size, 1.0f);
    std::vector<float> user_dec_dst(
            tgt_seq_length_max * batch * feature_size, 1.0f);
    std::vector<float> user_weights_attention_src_layer(
            feature_size * feature_size, 1.0f);
    std::vector<float> user_weights_annotation(
            feature_size * feature_size, 1.0f);
    std::vector<float> user_weights_alignments(feature_size, 1.0f);

    memory::dims user_dec_wei_layer_dims
            = {dec_n_layers, 1, feature_size, lstm_n_gates, feature_size};
    memory::dims user_dec_wei_iter_dims = {dec_n_layers, 1,
            feature_size + feature_size, lstm_n_gates, feature_size};
    memory::dims user_dec_bias_dims
            = {dec_n_layers, 1, lstm_n_gates, feature_size};

    memory::dims dec_src_layer_dims = {1, batch, feature_size};
    memory::dims dec_dst_layer_dims = {1, batch, feature_size};
    memory::dims dec_dst_iter_c_dims = {dec_n_layers, 1, batch, feature_size};
    //[dec mem dim]

    /// We will use the same memory for dec_src_iter and dec_dst_iter
    /// However, dec_src_iter has a context vector but not
    /// dec_dst_iter.
    /// To resolve this we will create one memory that holds the
    /// context vector as well as the both the hidden and cell states.
    /// The dst_iter will be a sub-memory of this memory.
    /// Note that the cell state will be padded by
    /// feature_size values. However, we do not compute or
    /// access those.
    /// @snippet cpu_rnn_inference_f32.cpp noctx mem dim
    //[noctx mem dim]
    memory::dims dec_dst_iter_dims
            = {dec_n_layers, 1, batch, feature_size + feature_size};
    memory::dims dec_dst_iter_noctx_dims
            = {dec_n_layers, 1, batch, feature_size};
    //[noctx mem dim]

    ///
    /// Decoder : create memory description
    /// @snippet cpu_rnn_inference_f32.cpp dec mem desc
    ///
    //[dec mem desc]
    auto user_dec_wei_layer_md = dnnl::memory::desc({user_dec_wei_layer_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldigo);
    auto user_dec_wei_iter_md = dnnl::memory::desc({user_dec_wei_iter_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldigo);
    auto user_dec_bias_md = dnnl::memory::desc({user_dec_bias_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldgo);
    auto dec_dst_layer_md = dnnl::memory::desc({dec_dst_layer_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::tnc);
    auto dec_src_layer_md = dnnl::memory::desc({dec_src_layer_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::tnc);
    auto dec_dst_iter_md = dnnl::memory::desc({dec_dst_iter_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldnc);
    auto dec_dst_iter_c_md = dnnl::memory::desc({dec_dst_iter_c_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldnc);
    //[dec mem desc]
    ///
    /// Decoder : Create memory
    /// @snippet cpu_rnn_inference_f32.cpp create dec memory
    ///
    //[create dec memory]
    auto user_dec_wei_layer_memory = dnnl::memory(
            user_dec_wei_layer_md, cpu_engine, user_dec_wei_layer.data());
    auto user_dec_wei_iter_memory = dnnl::memory(
            user_dec_wei_iter_md, cpu_engine, user_dec_wei_iter.data());
    auto user_dec_bias_memory
            = dnnl::memory(user_dec_bias_md, cpu_engine, user_dec_bias.data());
    auto user_dec_dst_layer_memory
            = dnnl::memory(dec_dst_layer_md, cpu_engine, user_dec_dst.data());
    auto dec_src_layer_memory = dnnl::memory(dec_src_layer_md, cpu_engine);
    auto dec_dst_iter_c_memory = dnnl::memory(dec_dst_iter_c_md, cpu_engine);
    //[create dec memory]

    auto dec_wei_layer_md = dnnl::memory::desc({user_dec_wei_layer_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto dec_wei_iter_md = dnnl::memory::desc({user_dec_wei_iter_dims},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    // As mentioned above, we create a view without context out of the
    // memory with context.
    ///
    /// Decoder : As mentioned above, we create a view without context out of the memory with context.
    /// @snippet cpu_rnn_inference_f32.cpp create noctx mem
    ///
    //[create noctx mem]
    auto dec_dst_iter_memory = dnnl::memory(dec_dst_iter_md, cpu_engine);
    auto dec_dst_iter_noctx_md = dec_dst_iter_md.submemory_desc(
            dec_dst_iter_noctx_dims, {0, 0, 0, 0, 0});
    //[create noctx mem]

    // TODO: add support for residual connections
    // should it be a set residual in op_desc or a field to set manually?
    // should be an integer to specify at which layer to start
    ///
    /// Decoder : Create RNN decoder cell
    /// @snippet cpu_rnn_inference_f32.cpp create dec rnn
    ///
    //[create dec rnn]
    lstm_forward::desc dec_ctx_desc(prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right, dec_src_layer_md,
            dec_dst_iter_md, dec_dst_iter_c_md, dec_wei_layer_md,
            dec_wei_iter_md, user_dec_bias_md, dec_dst_layer_md,
            dec_dst_iter_noctx_md, dec_dst_iter_c_md);
    auto dec_ctx_prim_desc
            = dnnl::lstm_forward::primitive_desc(dec_ctx_desc, cpu_engine);
    //[create dec rnn]

    ///
    /// Decoder : reorder weight memory
    /// @snippet cpu_rnn_inference_f32.cpp reorder weight memory
    ///
    //[reorder weight memory]
    auto dec_wei_layer_memory
            = memory(dec_ctx_prim_desc.weights_layer_desc(), cpu_engine);
    auto dec_wei_layer_reorder_pd = reorder::primitive_desc(
            user_dec_wei_layer_memory, dec_wei_layer_memory);
    reorder(dec_wei_layer_reorder_pd)
            .execute(s, user_dec_wei_layer_memory, dec_wei_layer_memory);

    auto dec_wei_iter_memory
            = memory(dec_ctx_prim_desc.weights_iter_desc(), cpu_engine);
    auto dec_wei_iter_reorder_pd = reorder::primitive_desc(
            user_dec_wei_iter_memory, dec_wei_iter_memory);
    reorder(dec_wei_iter_reorder_pd)
            .execute(s, user_dec_wei_iter_memory, dec_wei_iter_memory);
    //[reorder weight memory]

    ///
    /// Decoder : add the rnn primitive with related arguments into decoder_net
    /// @snippet cpu_rnn_inference_f32.cpp push rnn to decoder net
    ///
    //[push rnn to decoder net]
    // TODO: add a reorder when they will be available
    decoder_net.push_back(lstm_forward(dec_ctx_prim_desc));
    decoder_net_args.push_back({{DNNL_ARG_SRC_LAYER, dec_src_layer_memory},
            {DNNL_ARG_SRC_ITER, dec_dst_iter_memory},
            {DNNL_ARG_SRC_ITER_C, dec_dst_iter_c_memory},
            {DNNL_ARG_WEIGHTS_LAYER, dec_wei_layer_memory},
            {DNNL_ARG_WEIGHTS_ITER, dec_wei_iter_memory},
            {DNNL_ARG_BIAS, user_dec_bias_memory},
            {DNNL_ARG_DST_LAYER, user_dec_dst_layer_memory},
            {DNNL_ARG_DST_ITER, dec_dst_iter_memory},
            {DNNL_ARG_DST_ITER_C, dec_dst_iter_c_memory}});
    //[push rnn to decoder net]
    // allocating temporary buffer for attention mechanism
    std::vector<float> weighted_annotations(
            src_seq_length_max * batch * feature_size, 1.0f);

    ///
    /// **Execution**
    ///
    auto execute = [&]() {
        assert(encoder_net.size() == encoder_net_args.size()
                && "something is missing");
        ///
        /// run encoder (1 stream)
        /// @snippet cpu_rnn_inference_f32.cpp run enc
        ///
        //[run enc]
        for (size_t p = 0; p < encoder_net.size(); ++p)
            encoder_net.at(p).execute(s, encoder_net_args.at(p));
        //[run enc]

        ///
        /// we compute the weighted annotations once before the decoder
        /// @snippet cpu_rnn_inference_f32.cpp weight ano
        ///
        //[weight ano]
        compute_weighted_annotations(weighted_annotations.data(),
                src_seq_length_max, batch, feature_size,
                user_weights_annotation.data(),
                (float *)enc_dst_layer_memory.get_data_handle());
        //[weight ano]

        ///
        /// We initialize src_layer to the embedding of the end of
        /// sequence character, which are assumed to be 0 here
        /// @snippet cpu_rnn_inference_f32.cpp init src_layer
        ///
        //[init src_layer]
        memset(dec_src_layer_memory.get_data_handle(), 0,
                dec_src_layer_memory.get_desc().get_size());
        //[init src_layer]
        ///
        /// From now on, src points to the output of the last iteration
        ///
        for (dim_t i = 0; i < tgt_seq_length_max; i++) {
            float *src_att_layer_handle
                    = (float *)dec_src_layer_memory.get_data_handle();
            float *src_att_iter_handle
                    = (float *)dec_dst_iter_memory.get_data_handle();

            ///
            /// Compute attention context vector into the first layer src_iter
            /// @snippet cpu_rnn_inference_f32.cpp att ctx
            ///
            //[att ctx]
            compute_attention(src_att_iter_handle, src_seq_length_max, batch,
                    feature_size, user_weights_attention_src_layer.data(),
                    src_att_layer_handle,
                    (float *)enc_bidir_dst_layer_memory.get_data_handle(),
                    weighted_annotations.data(),
                    user_weights_alignments.data());
            //[att ctx]

            ///
            /// copy the context vectors to all layers of src_iter
            /// @snippet cpu_rnn_inference_f32.cpp cp ctx
            ///
            //[cp ctx]
            copy_context(
                    src_att_iter_handle, dec_n_layers, batch, feature_size);
            //[cp ctx]

            assert(decoder_net.size() == decoder_net_args.size()
                    && "something is missing");
            ///
            /// run the decoder iteration
            /// @snippet cpu_rnn_inference_f32.cpp run dec iter
            ///
            //[run dec iter]
            for (size_t p = 0; p < decoder_net.size(); ++p)
                decoder_net.at(p).execute(s, decoder_net_args.at(p));
            //[run dec iter]

            ///
            /// Move the handle on the src/dst layer to the next iteration
            /// @snippet cpu_rnn_inference_f32.cpp set handle
            ///
            //[set handle]
            auto dst_layer_handle
                    = (float *)user_dec_dst_layer_memory.get_data_handle();
            dec_src_layer_memory.set_data_handle(dst_layer_handle);
            user_dec_dst_layer_memory.set_data_handle(
                    dst_layer_handle + batch * feature_size);
            //[set handle]
        }
    };
    /// @page cpu_rnn_inference_f32_cpp
    ///
    std::cout << "Parameters:" << std::endl
              << " batch = " << batch << std::endl
              << " feature size = " << feature_size << std::endl
              << " maximum source sequence length = " << src_seq_length_max
              << std::endl
              << " maximum target sequence length = " << tgt_seq_length_max
              << std::endl
              << " number of layers of the bidirectional encoder = "
              << enc_bidir_n_layers << std::endl
              << " number of layers of the unidirectional encoder = "
              << enc_unidir_n_layers << std::endl
              << " number of layers of the decoder = " << dec_n_layers
              << std::endl;

    execute();
    s.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, simple_net);
}
