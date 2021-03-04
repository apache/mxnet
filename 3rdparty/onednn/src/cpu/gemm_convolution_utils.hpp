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

#ifndef CPU_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_GEMM_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

enum conv_gemm_loop_order_t { gemm_loop_rlb, gemm_loop_lrb, gemm_loop_lbr };
struct conv_gemm_conf_t {
    prop_kind_t prop_kind;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, id, ow, oh, od;
    int l_pad, t_pad, f_pad;
    int kh, kw, kd;
    int stride_h, stride_w, stride_d;
    int dilate_h, dilate_w, dilate_d;
    bool with_bias;
    bool is_nspc;

    int is, os, ks;
    int ic_block, oc_block;

    int nthr;
    ptrdiff_t im2col_sz;
    bool need_wei_reduction;
    bool signed_input;
    int oh_block;
    int ow_block;
    int os_block, os_nb_block;
    bool outer_threading;
    conv_gemm_loop_order_t loop_order;
    int nthr_oc;
};

namespace jit_gemm_convolution_utils {
template <typename data_type_t>
void im2col_3d(const conv_gemm_conf_t &jcp, const data_type_t *im,
        data_type_t *col, int od, int spatial_step, int spatial_block);

template <typename T>
void transpose_dt(const conv_gemm_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr);

template <typename im_dt, typename col_dt>
void im2col_dt_3d(const conv_gemm_conf_t &jcp, const im_dt *__restrict im,
        col_dt *__restrict col, int od);

template <typename data_type_t>
void im2col(const conv_gemm_conf_t &jcp, const data_type_t *__restrict im,
        data_type_t *__restrict col, int ss, int sb, int cs, int cb);

template <typename im_dt, typename col_dt>
void im2col_dt(const conv_gemm_conf_t &jcp, const im_dt *__restrict im,
        im_dt *__restrict imtr, col_dt *__restrict col, int hs, int hb, int ws,
        int wb);

template <typename T>
void col2im_dt(
        const conv_gemm_conf_t &jcp, const T *__restrict col, T *__restrict im);
void col2im_3d(const conv_gemm_conf_t &jcp, const float *col, float *im, int od,
        int spatial_step, int spatial_block);
void col2im(const conv_gemm_conf_t &jcp, const float *col, float *im,
        int spatial_step, int spatial_block);

status_t init_conf(conv_gemm_conf_t &jcp,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int max_threads);

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb, int &ithr_g,
        int &nthr_g, int &ithr_mb, int &nthr_mb);
void bwd_weights_reduction_par_ncsp(int ithr, int nthr,
        const conv_gemm_conf_t &jcp, const float *weights_reduce_ws,
        float *weights);
void bwd_weights_reduction_par_nspc(int ithr, int nthr, size_t g_start,
        size_t g_end, const conv_gemm_conf_t &jcp,
        const float *weights_reduce_base, float *diff_weights);
} // namespace jit_gemm_convolution_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
