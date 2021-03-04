/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

/// @example cpu_cnn_training_f32.c
/// @copybrief cpu_cnn_training_f32_c

/// @page cpu_cnn_training_f32_c CNN f32 training example
/// This C API example demonstrates how to build an AlexNet model training.
/// The example implements a few layers from AlexNet model.
///
/// @include cpu_cnn_training_f32.c

// Required for posix_memalign
#define _POSIX_C_SOURCE 200112L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkldnn.h"

#define BATCH 1
#define IC 3
#define OC 96
#define CONV_IH 227
#define CONV_IW 227
#define CONV_OH 55
#define CONV_OW 55
#define CONV_STRIDE 4
#define CONV_PAD 0
#define POOL_OH 27
#define POOL_OW 27
#define POOL_STRIDE 2
#define POOL_PAD 0

#define CHECK(f) \
    do { \
        mkldnn_status_t s = f; \
        if (s != mkldnn_success) { \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, \
                    s); \
            exit(2); \
        } \
    } while (0)

#define CHECK_TRUE(expr) \
    do { \
        int e_ = expr; \
        if (!e_) { \
            printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
            exit(2); \
        } \
    } while (0)

static size_t product(mkldnn_dim_t *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

static void init_net_data(float *data, uint32_t dim, const mkldnn_dim_t *dims) {
    if (dim == 1) {
        for (mkldnn_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = (float)(i % 1637);
        }
    } else if (dim == 4) {
        for (mkldnn_dim_t in = 0; in < dims[0]; ++in)
            for (mkldnn_dim_t ic = 0; ic < dims[1]; ++ic)
                for (mkldnn_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (mkldnn_dim_t iw = 0; iw < dims[3]; ++iw) {
                        mkldnn_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = (float)(indx % 1637);
                    }
    }
}

typedef struct {
    int nargs;
    mkldnn_exec_arg_t *args;
} args_t;

static void prepare_arg_node(args_t *node, int nargs) {
    node->args = (mkldnn_exec_arg_t *)malloc(sizeof(mkldnn_exec_arg_t) * nargs);
    node->nargs = nargs;
}
static void free_arg_node(args_t *node) {
    free(node->args);
}

static void set_arg(
        mkldnn_exec_arg_t *arg, int arg_idx, mkldnn_memory_t memory) {
    arg->arg = arg_idx;
    arg->memory = memory;
}

static void init_data_memory(uint32_t dim, const mkldnn_dim_t *dims,
        mkldnn_format_tag_t user_fmt, mkldnn_data_type_t data_type,
        mkldnn_engine_t engine, float *data, mkldnn_memory_t *memory) {
    mkldnn_memory_desc_t user_md;
    CHECK(mkldnn_memory_desc_init_by_tag(
            &user_md, dim, dims, mkldnn_f32, user_fmt));
    CHECK(mkldnn_memory_create(memory, &user_md, engine, data));
}

mkldnn_status_t prepare_reorder(mkldnn_memory_t *user_memory, /// in
        const mkldnn_memory_desc_t *prim_memory_md, /// in
        mkldnn_engine_t prim_engine, /// in: primitive's engine
        int dir_is_user_to_prim, /// in: user -> prim or prim -> user
        mkldnn_memory_t *prim_memory, /// out: primitive's memory created
        mkldnn_primitive_t *reorder, /// out: reorder primitive created
        uint32_t *net_index, /// primitive index in net (inc if reorder created)
        mkldnn_primitive_t *net, args_t *net_args) {
    const mkldnn_memory_desc_t *user_memory_md;
    mkldnn_memory_get_memory_desc(*user_memory, &user_memory_md);

    mkldnn_engine_t user_mem_engine;
    mkldnn_memory_get_engine(*user_memory, &user_mem_engine);

    if (!mkldnn_memory_desc_equal(user_memory_md, prim_memory_md)) {
        CHECK(mkldnn_memory_create(prim_memory, prim_memory_md, prim_engine,
                MKLDNN_MEMORY_ALLOCATE));

        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                    user_memory_md, user_mem_engine, prim_memory_md,
                    prim_engine, NULL));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                    prim_memory_md, prim_engine, user_memory_md,
                    user_mem_engine, NULL));
        }
        CHECK(mkldnn_primitive_create(reorder, reorder_pd));
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));

        net[*net_index] = *reorder;
        prepare_arg_node(&net_args[*net_index], 2);
        set_arg(&net_args[*net_index].args[0], MKLDNN_ARG_FROM,
                dir_is_user_to_prim ? *user_memory : *prim_memory);
        set_arg(&net_args[*net_index].args[1], MKLDNN_ARG_TO,
                dir_is_user_to_prim ? *prim_memory : *user_memory);
        (*net_index)++;
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net() {
    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0)); // idx

    // build a simple net
    uint32_t n_fwd = 0, n_bwd = 0;
    mkldnn_primitive_t net_fwd[10], net_bwd[10];
    args_t net_fwd_args[10], net_bwd_args[10];

    mkldnn_dim_t net_src_sizes[4] = {BATCH, IC, CONV_IH, CONV_IW};
    mkldnn_dim_t net_dst_sizes[4] = {BATCH, OC, POOL_OH, POOL_OW};

    float *net_src = (float *)malloc(product(net_src_sizes, 4) * sizeof(float));
    float *net_dst = (float *)malloc(product(net_dst_sizes, 4) * sizeof(float));

    init_net_data(net_src, 4, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, 4) * sizeof(float));

    //----------------------------------------------------------------------
    //----------------- Forward Stream -------------------------------------
    // AlexNet: conv
    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
    // {BATCH, OC, CONV_OH, CONV_OW}
    // strides: {CONV_STRIDE, CONV_STRIDE}
    mkldnn_dim_t *conv_user_src_sizes = net_src_sizes;
    mkldnn_dim_t conv_user_weights_sizes[4] = {OC, IC, 11, 11};
    mkldnn_dim_t conv_bias_sizes[4] = {OC};
    mkldnn_dim_t conv_user_dst_sizes[4] = {BATCH, OC, CONV_OH, CONV_OW};
    mkldnn_dim_t conv_strides[2] = {CONV_STRIDE, CONV_STRIDE};
    mkldnn_dim_t conv_padding[2] = {CONV_PAD, CONV_PAD};

    float *conv_src = net_src;
    float *conv_weights = (float *)malloc(
            product(conv_user_weights_sizes, 4) * sizeof(float));
    float *conv_bias
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));

    init_net_data(conv_weights, 4, conv_user_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    // create memory for user data
    mkldnn_memory_t conv_user_src_memory, conv_user_weights_memory,
            conv_user_bias_memory;
    init_data_memory(4, conv_user_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
            conv_src, &conv_user_src_memory);
    init_data_memory(4, conv_user_weights_sizes, mkldnn_oihw, mkldnn_f32,
            engine, conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine,
            conv_bias, &conv_user_bias_memory);

    // create a convolution
    mkldnn_primitive_desc_t conv_pd;

    {
        // create data descriptors for convolution w/ no specified format
        mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
                conv_dst_md;
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_src_md, 4,
                conv_user_src_sizes, mkldnn_f32, mkldnn_format_tag_any));
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_weights_md, 4,
                conv_user_weights_sizes, mkldnn_f32, mkldnn_format_tag_any));
        CHECK(mkldnn_memory_desc_init_by_tag(
                &conv_bias_md, 1, conv_bias_sizes, mkldnn_f32, mkldnn_x));
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_dst_md, 4,
                conv_user_dst_sizes, mkldnn_f32, mkldnn_format_tag_any));

        mkldnn_convolution_desc_t conv_any_desc;
        CHECK(mkldnn_convolution_forward_desc_init(&conv_any_desc,
                mkldnn_forward, mkldnn_convolution_direct, &conv_src_md,
                &conv_weights_md, &conv_bias_md, &conv_dst_md, conv_strides,
                conv_padding, conv_padding));

        CHECK(mkldnn_primitive_desc_create(
                &conv_pd, &conv_any_desc, NULL, engine, NULL));
    }

    mkldnn_memory_t conv_internal_src_memory, conv_internal_weights_memory,
            conv_internal_dst_memory;

    // create memory for dst data, we don't need to reorder it to user data
    const mkldnn_memory_desc_t *conv_dst_md
            = mkldnn_primitive_desc_query_md(conv_pd, mkldnn_query_dst_md, 0);
    CHECK(mkldnn_memory_create(&conv_internal_dst_memory, conv_dst_md, engine,
            MKLDNN_MEMORY_ALLOCATE));

    // create reorder primitives between user data and convolution srcs
    // if required
    mkldnn_primitive_t conv_reorder_src, conv_reorder_weights;

    const mkldnn_memory_desc_t *conv_src_md
            = mkldnn_primitive_desc_query_md(conv_pd, mkldnn_query_src_md, 0);
    CHECK(prepare_reorder(&conv_user_src_memory, conv_src_md, engine, 1,
            &conv_internal_src_memory, &conv_reorder_src, &n_fwd, net_fwd,
            net_fwd_args));

    const mkldnn_memory_desc_t *conv_weights_md
            = mkldnn_primitive_desc_query_md(
                    conv_pd, mkldnn_query_weights_md, 0);
    CHECK(prepare_reorder(&conv_user_weights_memory, conv_weights_md, engine, 1,
            &conv_internal_weights_memory, &conv_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));

    mkldnn_memory_t conv_src_memory = conv_internal_src_memory
            ? conv_internal_src_memory
            : conv_user_src_memory;
    mkldnn_memory_t conv_weights_memory = conv_internal_weights_memory
            ? conv_internal_weights_memory
            : conv_user_weights_memory;

    // finally create a convolution primitive
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, conv_pd));
    net_fwd[n_fwd] = conv;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], MKLDNN_ARG_SRC, conv_src_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], MKLDNN_ARG_WEIGHTS,
            conv_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], MKLDNN_ARG_BIAS,
            conv_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], MKLDNN_ARG_DST,
            conv_internal_dst_memory);
    n_fwd++;

    // AlexNet: relu
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}

    float negative_slope = 1.0f;

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const mkldnn_memory_desc_t *relu_src_md = conv_dst_md;

    // create a relu primitive descriptor
    mkldnn_eltwise_desc_t relu_desc;
    CHECK(mkldnn_eltwise_forward_desc_init(&relu_desc, mkldnn_forward,
            mkldnn_eltwise_relu, relu_src_md, negative_slope, 0));

    mkldnn_primitive_desc_t relu_pd;
    CHECK(mkldnn_primitive_desc_create(
            &relu_pd, &relu_desc, NULL, engine, NULL));

    // create relu dst memory
    mkldnn_memory_t relu_dst_memory;
    const mkldnn_memory_desc_t *relu_dst_md
            = mkldnn_primitive_desc_query_md(relu_pd, mkldnn_query_dst_md, 0);
    CHECK(mkldnn_memory_create(
            &relu_dst_memory, relu_dst_md, engine, MKLDNN_MEMORY_ALLOCATE));

    // finally create a relu primitive
    mkldnn_primitive_t relu;
    CHECK(mkldnn_primitive_create(&relu, relu_pd));
    net_fwd[n_fwd] = relu;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], MKLDNN_ARG_SRC,
            conv_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], MKLDNN_ARG_DST, relu_dst_memory);
    n_fwd++;

    // AlexNet: lrn
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
    // local size: 5
    // alpha: 0.0001
    // beta: 0.75
    // k: 1.0
    uint32_t local_size = 5;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float k = 1.0f;

    // create lrn src memory descriptor using dst memory descriptor
    //  from previous primitive
    const mkldnn_memory_desc_t *lrn_src_md = relu_dst_md;

    // create a lrn primitive descriptor
    mkldnn_lrn_desc_t lrn_desc;
    CHECK(mkldnn_lrn_forward_desc_init(&lrn_desc, mkldnn_forward,
            mkldnn_lrn_across_channels, lrn_src_md, local_size, alpha, beta,
            k));

    mkldnn_primitive_desc_t lrn_pd;
    CHECK(mkldnn_primitive_desc_create(&lrn_pd, &lrn_desc, NULL, engine, NULL));

    // create primitives for lrn dst and workspace memory
    mkldnn_memory_t lrn_dst_memory, lrn_ws_memory;

    const mkldnn_memory_desc_t *lrn_dst_md
            = mkldnn_primitive_desc_query_md(lrn_pd, mkldnn_query_dst_md, 0);
    CHECK(mkldnn_memory_create(
            &lrn_dst_memory, lrn_dst_md, engine, MKLDNN_MEMORY_ALLOCATE));

    // create workspace only in training and only for forward primitive
    // query lrn_pd for workspace, this memory will be shared with forward lrn
    const mkldnn_memory_desc_t *lrn_ws_md = mkldnn_primitive_desc_query_md(
            lrn_pd, mkldnn_query_workspace_md, 0);
    CHECK(mkldnn_memory_create(
            &lrn_ws_memory, lrn_ws_md, engine, MKLDNN_MEMORY_ALLOCATE));

    // finally create a lrn primitive
    mkldnn_primitive_t lrn;
    CHECK(mkldnn_primitive_create(&lrn, lrn_pd));
    net_fwd[n_fwd] = lrn;
    prepare_arg_node(&net_fwd_args[n_fwd], 3);
    set_arg(&net_fwd_args[n_fwd].args[0], MKLDNN_ARG_SRC, relu_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], MKLDNN_ARG_DST, lrn_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], MKLDNN_ARG_WORKSPACE, lrn_ws_memory);
    n_fwd++;

    // AlexNet: pool
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
    // kernel: {3, 3}
    // strides: {POOL_STRIDE, POOL_STRIDE}
    mkldnn_dim_t *pool_dst_sizes = net_dst_sizes;
    mkldnn_dim_t pool_kernel[2] = {3, 3};
    mkldnn_dim_t pool_strides[2] = {POOL_STRIDE, POOL_STRIDE};
    mkldnn_dim_t pool_padding[2] = {POOL_PAD, POOL_PAD};

    // create memory for user dst data
    mkldnn_memory_t pool_user_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
            net_dst, &pool_user_dst_memory);

    // create a pooling primitive descriptor
    mkldnn_primitive_desc_t pool_pd;

    {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive
        const mkldnn_memory_desc_t *pool_src_md = lrn_dst_md;

        // create descriptors for dst pooling data
        mkldnn_memory_desc_t pool_dst_md;
        CHECK(mkldnn_memory_desc_init_by_tag(&pool_dst_md, 4, pool_dst_sizes,
                mkldnn_f32, mkldnn_format_tag_any));

        mkldnn_pooling_desc_t pool_desc;
        CHECK(mkldnn_pooling_forward_desc_init(&pool_desc, mkldnn_forward,
                mkldnn_pooling_max, pool_src_md, &pool_dst_md, pool_strides,
                pool_kernel, pool_padding, pool_padding));

        CHECK(mkldnn_primitive_desc_create(
                &pool_pd, &pool_desc, NULL, engine, NULL));
    }

    // create memory for workspace
    mkldnn_memory_t pool_ws_memory;
    const mkldnn_memory_desc_t *pool_ws_md = mkldnn_primitive_desc_query_md(
            pool_pd, mkldnn_query_workspace_md, 0);
    CHECK(mkldnn_memory_create(
            &pool_ws_memory, pool_ws_md, engine, MKLDNN_MEMORY_ALLOCATE));

    // create reorder primitives between pooling dsts and user format dst
    // if required
    mkldnn_primitive_t pool_reorder_dst;
    mkldnn_memory_t pool_internal_dst_memory;
    const mkldnn_memory_desc_t *pool_dst_md
            = mkldnn_primitive_desc_query_md(pool_pd, mkldnn_query_dst_md, 0);
    n_fwd += 1; // tentative workaround: preserve space for pooling that should
            // happen before the reorder
    CHECK(prepare_reorder(&pool_user_dst_memory, pool_dst_md, engine, 0,
            &pool_internal_dst_memory, &pool_reorder_dst, &n_fwd, net_fwd,
            net_fwd_args));
    n_fwd -= pool_reorder_dst ? 2 : 1;

    mkldnn_memory_t pool_dst_memory = pool_internal_dst_memory
            ? pool_internal_dst_memory
            : pool_user_dst_memory;

    // finally create a pooling primitive
    mkldnn_primitive_t pool;
    CHECK(mkldnn_primitive_create(&pool, pool_pd));
    net_fwd[n_fwd] = pool;
    prepare_arg_node(&net_fwd_args[n_fwd], 3);
    set_arg(&net_fwd_args[n_fwd].args[0], MKLDNN_ARG_SRC, lrn_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], MKLDNN_ARG_DST, pool_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], MKLDNN_ARG_WORKSPACE, pool_ws_memory);
    n_fwd++;

    if (pool_reorder_dst) n_fwd += 1;

    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    //-----------------------------------------------------------------------

    // ... user diff_data ...
    float *net_diff_dst
            = (float *)malloc(product(pool_dst_sizes, 4) * sizeof(float));

    init_net_data(net_diff_dst, 4, pool_dst_sizes);

    // create memory for user diff dst data
    mkldnn_memory_t pool_user_diff_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
            net_diff_dst, &pool_user_diff_dst_memory);

    // Pooling Backward
    // pooling diff src memory descriptor
    const mkldnn_memory_desc_t *pool_diff_src_md = lrn_dst_md;

    // pooling diff dst memory descriptor
    const mkldnn_memory_desc_t *pool_diff_dst_md = pool_dst_md;

    // create backward pooling descriptor
    mkldnn_pooling_desc_t pool_bwd_desc;
    CHECK(mkldnn_pooling_backward_desc_init(&pool_bwd_desc, mkldnn_pooling_max,
            pool_diff_src_md, pool_diff_dst_md, pool_strides, pool_kernel,
            pool_padding, pool_padding));

    // backward primitive descriptor needs to hint forward descriptor
    mkldnn_primitive_desc_t pool_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(
            &pool_bwd_pd, &pool_bwd_desc, NULL, engine, pool_pd));

    // create reorder primitive between user diff dst and pool diff dst
    // if required
    mkldnn_memory_t pool_diff_dst_memory, pool_internal_diff_dst_memory;
    mkldnn_primitive_t pool_reorder_diff_dst;
    CHECK(prepare_reorder(&pool_user_diff_dst_memory, pool_diff_dst_md, engine,
            1, &pool_internal_diff_dst_memory, &pool_reorder_diff_dst, &n_bwd,
            net_bwd, net_bwd_args));

    pool_diff_dst_memory = pool_internal_diff_dst_memory
            ? pool_internal_diff_dst_memory
            : pool_user_diff_dst_memory;

    // create memory for pool diff src data
    mkldnn_memory_t pool_diff_src_memory;
    CHECK(mkldnn_memory_create(&pool_diff_src_memory, pool_diff_src_md, engine,
            MKLDNN_MEMORY_ALLOCATE));

    // finally create backward pooling primitive
    mkldnn_primitive_t pool_bwd;
    CHECK(mkldnn_primitive_create(&pool_bwd, pool_bwd_pd));
    net_bwd[n_bwd] = pool_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 3);
    set_arg(&net_bwd_args[n_bwd].args[0], MKLDNN_ARG_DIFF_DST,
            pool_diff_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], MKLDNN_ARG_WORKSPACE, pool_ws_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], MKLDNN_ARG_DIFF_SRC,
            pool_diff_src_memory);
    n_bwd++;

    // Backward lrn
    const mkldnn_memory_desc_t *lrn_diff_dst_md = pool_diff_src_md;

    // create backward lrn descriptor
    mkldnn_lrn_desc_t lrn_bwd_desc;
    CHECK(mkldnn_lrn_backward_desc_init(&lrn_bwd_desc,
            mkldnn_lrn_across_channels, lrn_src_md, lrn_diff_dst_md, local_size,
            alpha, beta, k));

    mkldnn_primitive_desc_t lrn_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(
            &lrn_bwd_pd, &lrn_bwd_desc, NULL, engine, lrn_pd));

    // create memory for lrn diff src
    mkldnn_memory_t lrn_diff_src_memory;
    const mkldnn_memory_desc_t *lrn_diff_src_md
            = mkldnn_primitive_desc_query_md(
                    lrn_bwd_pd, mkldnn_query_diff_src_md, 0);
    CHECK(mkldnn_memory_create(&lrn_diff_src_memory, lrn_diff_src_md, engine,
            MKLDNN_MEMORY_ALLOCATE));

    // finally create backward lrn primitive
    mkldnn_primitive_t lrn_bwd;
    CHECK(mkldnn_primitive_create(&lrn_bwd, lrn_bwd_pd));
    net_bwd[n_bwd] = lrn_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 4);
    set_arg(&net_bwd_args[n_bwd].args[0], MKLDNN_ARG_SRC, relu_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], MKLDNN_ARG_DIFF_DST,
            pool_diff_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], MKLDNN_ARG_WORKSPACE, lrn_ws_memory);
    set_arg(&net_bwd_args[n_bwd].args[3], MKLDNN_ARG_DIFF_SRC,
            lrn_diff_src_memory);
    n_bwd++;

    // Backward relu
    const mkldnn_memory_desc_t *relu_diff_dst_md = lrn_diff_src_md;

    // create backward relu descriptor
    mkldnn_eltwise_desc_t relu_bwd_desc;
    CHECK(mkldnn_eltwise_backward_desc_init(&relu_bwd_desc, mkldnn_eltwise_relu,
            relu_diff_dst_md, relu_src_md, negative_slope, 0));

    mkldnn_primitive_desc_t relu_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(
            &relu_bwd_pd, &relu_bwd_desc, NULL, engine, relu_pd));

    // create memory for relu diff src
    mkldnn_memory_t relu_diff_src_memory;
    const mkldnn_memory_desc_t *relu_diff_src_md
            = mkldnn_primitive_desc_query_md(
                    relu_bwd_pd, mkldnn_query_diff_src_md, 0);
    CHECK(mkldnn_memory_create(&relu_diff_src_memory, relu_diff_src_md, engine,
            MKLDNN_MEMORY_ALLOCATE));

    // finally create backward relu primitive
    mkldnn_primitive_t relu_bwd;
    CHECK(mkldnn_primitive_create(&relu_bwd, relu_bwd_pd));
    net_bwd[n_bwd] = relu_bwd;
    prepare_arg_node(&net_bwd_args[n_bwd], 3);
    set_arg(&net_bwd_args[n_bwd].args[0], MKLDNN_ARG_SRC,
            conv_internal_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], MKLDNN_ARG_DIFF_DST,
            lrn_diff_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], MKLDNN_ARG_DIFF_SRC,
            relu_diff_src_memory);
    n_bwd++;

    // Backward convolution with respect to weights
    float *conv_diff_bias_buffer
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));
    float *conv_user_diff_weights_buffer = (float *)malloc(
            product(conv_user_weights_sizes, 4) * sizeof(float));

    // initialize memory for diff weights in user format
    mkldnn_memory_t conv_user_diff_weights_memory;
    init_data_memory(4, conv_user_weights_sizes, mkldnn_oihw, mkldnn_f32,
            engine, conv_user_diff_weights_buffer,
            &conv_user_diff_weights_memory);

    // create backward convolution primitive descriptor
    mkldnn_primitive_desc_t conv_bwd_weights_pd;

    {
        // memory descriptors should be in format `any` to allow backward
        // convolution for
        // weights to chose the format it prefers for best performance
        mkldnn_memory_desc_t conv_diff_src_md, conv_diff_weights_md,
                conv_diff_bias_md, conv_diff_dst_md;
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_diff_src_md, 4,
                conv_user_src_sizes, mkldnn_f32, mkldnn_format_tag_any));
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_diff_weights_md, 4,
                conv_user_weights_sizes, mkldnn_f32, mkldnn_format_tag_any));
        CHECK(mkldnn_memory_desc_init_by_tag(
                &conv_diff_bias_md, 1, conv_bias_sizes, mkldnn_f32, mkldnn_x));
        CHECK(mkldnn_memory_desc_init_by_tag(&conv_diff_dst_md, 4,
                conv_user_dst_sizes, mkldnn_f32, mkldnn_format_tag_any));

        // create backward convolution descriptor
        mkldnn_convolution_desc_t conv_bwd_weights_desc;
        CHECK(mkldnn_convolution_backward_weights_desc_init(
                &conv_bwd_weights_desc, mkldnn_convolution_direct,
                &conv_diff_src_md, &conv_diff_weights_md, &conv_diff_bias_md,
                &conv_diff_dst_md, conv_strides, conv_padding, conv_padding));

        CHECK(mkldnn_primitive_desc_create(&conv_bwd_weights_pd,
                &conv_bwd_weights_desc, NULL, engine, conv_pd));
    }

    // for best performance convolution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward convolution
    // for src and dst respectively
    // create reorder primitives for src from forward convolution to the
    // format chosen by backward convolution
    mkldnn_primitive_t conv_bwd_reorder_src;
    mkldnn_memory_t conv_bwd_internal_src_memory;
    const mkldnn_memory_desc_t *conv_diff_src_md
            = mkldnn_primitive_desc_query_md(
                    conv_bwd_weights_pd, mkldnn_query_src_md, 0);
    CHECK(prepare_reorder(&conv_src_memory, conv_diff_src_md, engine, 1,
            &conv_bwd_internal_src_memory, &conv_bwd_reorder_src, &n_bwd,
            net_bwd, net_bwd_args));

    mkldnn_memory_t conv_bwd_weights_src_memory = conv_bwd_internal_src_memory
            ? conv_bwd_internal_src_memory
            : conv_src_memory;

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by conv_diff_weights
    mkldnn_primitive_t conv_reorder_diff_dst;
    mkldnn_memory_t conv_internal_diff_dst_memory;
    const mkldnn_memory_desc_t *conv_diff_dst_md
            = mkldnn_primitive_desc_query_md(
                    conv_bwd_weights_pd, mkldnn_query_diff_dst_md, 0);

    CHECK(prepare_reorder(&relu_diff_src_memory, conv_diff_dst_md, engine, 1,
            &conv_internal_diff_dst_memory, &conv_reorder_diff_dst, &n_bwd,
            net_bwd, net_bwd_args));

    mkldnn_memory_t conv_diff_dst_memory = conv_internal_diff_dst_memory
            ? conv_internal_diff_dst_memory
            : relu_diff_src_memory;

    // create reorder primitives for conv diff weights memory
    mkldnn_primitive_t conv_reorder_diff_weights;
    mkldnn_memory_t conv_internal_diff_weights_memory;
    const mkldnn_memory_desc_t *conv_diff_weights_md
            = mkldnn_primitive_desc_query_md(
                    conv_bwd_weights_pd, mkldnn_query_diff_weights_md, 0);
    n_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
            // that should happen before the reorder

    CHECK(prepare_reorder(&conv_user_diff_weights_memory, conv_diff_weights_md,
            engine, 0, &conv_internal_diff_weights_memory,
            &conv_reorder_diff_weights, &n_bwd, net_bwd, net_bwd_args));
    n_bwd -= conv_reorder_diff_weights ? 2 : 1;

    mkldnn_memory_t conv_diff_weights_memory = conv_internal_diff_weights_memory
            ? conv_internal_diff_weights_memory
            : conv_user_diff_weights_memory;

    // create memory for diff bias memory
    mkldnn_memory_t conv_diff_bias_memory;
    const mkldnn_memory_desc_t *conv_diff_bias_md
            = mkldnn_primitive_desc_query_md(
                    conv_bwd_weights_pd, mkldnn_query_diff_weights_md, 1);
    CHECK(mkldnn_memory_create(
            &conv_diff_bias_memory, conv_diff_bias_md, engine, NULL));
    CHECK(mkldnn_memory_set_data_handle(
            conv_diff_bias_memory, conv_diff_bias_buffer));

    // finally created backward convolution weights primitive
    mkldnn_primitive_t conv_bwd_weights;
    CHECK(mkldnn_primitive_create(&conv_bwd_weights, conv_bwd_weights_pd));
    net_bwd[n_bwd] = conv_bwd_weights;
    prepare_arg_node(&net_bwd_args[n_bwd], 4);
    set_arg(&net_bwd_args[n_bwd].args[0], MKLDNN_ARG_SRC,
            conv_bwd_weights_src_memory);
    set_arg(&net_bwd_args[n_bwd].args[1], MKLDNN_ARG_DIFF_DST,
            conv_diff_dst_memory);
    set_arg(&net_bwd_args[n_bwd].args[2], MKLDNN_ARG_DIFF_WEIGHTS,
            conv_diff_weights_memory);
    set_arg(&net_bwd_args[n_bwd].args[3], MKLDNN_ARG_DIFF_BIAS,
            conv_diff_bias_memory);
    n_bwd++;

    if (conv_reorder_diff_weights) n_bwd += 1;

    // output from backward stream
    void *net_diff_weights = NULL;
    void *net_diff_bias = NULL;

    int n_iter = 10; // number of iterations for training.
    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, engine, mkldnn_stream_default_flags));
    // Execute the net
    for (int i = 0; i < n_iter; i++) {
        for (uint32_t i = 0; i < n_fwd; ++i)
            CHECK(mkldnn_primitive_execute(net_fwd[i], stream,
                    net_fwd_args[i].nargs, net_fwd_args[i].args));

        // Update net_diff_dst
        void *net_output = NULL; // output from forward stream:
        CHECK(mkldnn_memory_get_data_handle(pool_user_dst_memory, &net_output));
        // ...user updates net_diff_dst using net_output...
        // some user defined func update_diff_dst(net_diff_dst, net_output)

        // Backward pass
        for (uint32_t i = 0; i < n_bwd; ++i)
            CHECK(mkldnn_primitive_execute(net_bwd[i], stream,
                    net_bwd_args[i].nargs, net_bwd_args[i].args));

        // ... update weights ...
        CHECK(mkldnn_memory_get_data_handle(
                conv_user_diff_weights_memory, &net_diff_weights));
        CHECK(mkldnn_memory_get_data_handle(
                conv_diff_bias_memory, &net_diff_bias));
        // ...user updates weights and bias using diff weights and bias...
        // some user defined func update_weights(conv_user_weights_memory,
        // conv_bias_memory,
        //      net_diff_weights, net_diff_bias);
    }
    CHECK(mkldnn_stream_wait(stream));

    mkldnn_stream_destroy(stream);

    // clean up nets
    for (uint32_t i = 0; i < n_fwd; ++i)
        free_arg_node(&net_fwd_args[i]);
    for (uint32_t i = 0; i < n_bwd; ++i)
        free_arg_node(&net_bwd_args[i]);

    // Cleanup forward
    CHECK(mkldnn_primitive_desc_destroy(pool_pd));
    CHECK(mkldnn_primitive_desc_destroy(lrn_pd));
    CHECK(mkldnn_primitive_desc_destroy(relu_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv_pd));

    free(net_src);
    free(net_dst);

    mkldnn_memory_destroy(conv_user_src_memory);
    mkldnn_memory_destroy(conv_user_weights_memory);
    mkldnn_memory_destroy(conv_user_bias_memory);
    mkldnn_memory_destroy(conv_internal_src_memory);
    mkldnn_memory_destroy(conv_internal_weights_memory);
    mkldnn_memory_destroy(conv_internal_dst_memory);
    mkldnn_primitive_destroy(conv_reorder_src);
    mkldnn_primitive_destroy(conv_reorder_weights);
    mkldnn_primitive_destroy(conv);

    free(conv_weights);
    free(conv_bias);

    mkldnn_memory_destroy(relu_dst_memory);
    mkldnn_primitive_destroy(relu);

    mkldnn_memory_destroy(lrn_ws_memory);
    mkldnn_memory_destroy(lrn_dst_memory);
    mkldnn_primitive_destroy(lrn);

    mkldnn_memory_destroy(pool_user_dst_memory);
    mkldnn_memory_destroy(pool_internal_dst_memory);
    mkldnn_memory_destroy(pool_ws_memory);
    mkldnn_primitive_destroy(pool_reorder_dst);
    mkldnn_primitive_destroy(pool);

    // Cleanup backward
    CHECK(mkldnn_primitive_desc_destroy(pool_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(lrn_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(relu_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv_bwd_weights_pd));

    mkldnn_memory_destroy(pool_user_diff_dst_memory);
    mkldnn_memory_destroy(pool_diff_src_memory);
    mkldnn_memory_destroy(pool_internal_diff_dst_memory);
    mkldnn_primitive_destroy(pool_reorder_diff_dst);
    mkldnn_primitive_destroy(pool_bwd);

    free(net_diff_dst);

    mkldnn_memory_destroy(lrn_diff_src_memory);
    mkldnn_primitive_destroy(lrn_bwd);

    mkldnn_memory_destroy(relu_diff_src_memory);
    mkldnn_primitive_destroy(relu_bwd);

    mkldnn_memory_destroy(conv_user_diff_weights_memory);
    mkldnn_memory_destroy(conv_diff_bias_memory);
    mkldnn_memory_destroy(conv_bwd_internal_src_memory);
    mkldnn_primitive_destroy(conv_bwd_reorder_src);
    mkldnn_memory_destroy(conv_internal_diff_dst_memory);
    mkldnn_primitive_destroy(conv_reorder_diff_dst);
    mkldnn_memory_destroy(conv_internal_diff_weights_memory);
    mkldnn_primitive_destroy(conv_reorder_diff_weights);
    mkldnn_primitive_destroy(conv_bwd_weights);

    free(conv_diff_bias_buffer);
    free(conv_user_diff_weights_buffer);

    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

int main(int argc, char **argv) {
    mkldnn_status_t result = simple_net();
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
