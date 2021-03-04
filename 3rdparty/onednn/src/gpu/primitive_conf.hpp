/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_PRIMITIVE_CONF_HPP
#define GPU_PRIMITIVE_CONF_HPP

#include <stdint.h>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_storage.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_eltwise_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

#define MAX_NDIMS 6

struct memory_desc_info_t {
    // Max 2 levels of blocking
    static const int nlevels = 2;

    int ndims;
    data_type_t data_type;

    int offset0;
    int dims[MAX_NDIMS];
    int padded_dims[MAX_NDIMS];
    int blocks[MAX_NDIMS][nlevels + 1];
    int strides[MAX_NDIMS][nlevels + 1];

    static memory_desc_info_t create(const memory_desc_wrapper &mdw) {
        auto md_info = memory_desc_info_t();

        md_info.ndims = mdw.ndims();
        md_info.data_type = mdw.data_type();
        md_info.offset0 = mdw.offset0();

        auto &blk = mdw.blocking_desc();
        dim_t blk_stride
                = utils::array_product(blk.inner_blks, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); ++d) {
            utils::array_set(md_info.blocks[d], 1, nlevels + 1);
            utils::array_set(md_info.strides[d], 0, nlevels + 1);
        }

        for (int d = 0; d < mdw.ndims(); ++d) {
            md_info.dims[d] = mdw.dims()[d];
            md_info.padded_dims[d] = mdw.padded_dims()[d];
            md_info.strides[d][0] = blk.strides[d];
        }

        int levels[MAX_NDIMS] = {0};
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            int d = blk.inner_idxs[iblk];
            ++levels[d];

            md_info.blocks[d][levels[d]] = blk.inner_blks[iblk];
            blk_stride /= blk.inner_blks[iblk];
            md_info.strides[d][levels[d]] = blk_stride;
        }
        return md_info;
    }
};

struct attr_info_t {
    static attr_info_t create(const primitive_attr_t *attr) {
        const auto &po = attr->post_ops_;

        attr_info_t attr_info;

        attr_info.all_post_ops.copy_from(po);

        int binary_idx = po.find(primitive_kind::binary);
        attr_info.with_binary = (binary_idx != -1);

        // Eltwise
        attr_info.eltwise_idx = po.find(primitive_kind::eltwise);
        attr_info.with_eltwise = (attr_info.eltwise_idx != -1);

        if (attr_info.with_eltwise) {
            auto &eltwise = po.entry_[attr_info.eltwise_idx].eltwise;
            attr_info.eltwise_alg = eltwise.alg;
            attr_info.eltwise_scale = eltwise.scale;
            attr_info.eltwise_alpha = eltwise.alpha;
            attr_info.eltwise_beta = eltwise.beta;
        } else {
            attr_info.eltwise_alg = alg_kind::undef;
            attr_info.eltwise_scale = 1.0f;
            attr_info.eltwise_alpha = 1.0f;
            attr_info.eltwise_beta = 0.0f;
        }

        // Sum
        attr_info.sum_idx = po.find(primitive_kind::sum);
        attr_info.sum_scale = (attr_info.sum_idx != -1
                        ? po.entry_[attr_info.sum_idx].sum.scale
                        : 0.0f);
        attr_info.sum_data_type = (attr_info.sum_idx != -1)
                ? po.entry_[attr_info.sum_idx].sum.dt
                : dnnl_data_type_undef;
        attr_info.with_sum
                = (attr_info.sum_idx != -1) && (attr_info.sum_scale != 0.0f);

        // Output scales
        attr_info.with_oscales = !attr->output_scales_.has_default_values();

        const auto &scales_mask = attr->output_scales_.mask_;
        attr_info.with_common_oscales
                = attr_info.with_oscales && (scales_mask == 0);
        attr_info.common_oscales = (attr_info.with_common_oscales
                        ? attr->output_scales_.scales_[0]
                        : 1.0f);

        attr_info.with_per_oc_oscales
                = attr_info.with_oscales && (scales_mask == (1 << 1));

        attr_info.with_runtime_oscales = attr_info.with_per_oc_oscales
                && !attr->output_scales_.defined();

        const auto &src0_scales = attr->scales_.get(DNNL_ARG_SRC_0);
        attr_info.with_src0_scale = !src0_scales.has_default_values();
        attr_info.src0_scale = *src0_scales.scales_;
        assert(src0_scales.mask_ == 0);

        const auto &src1_scales = attr->scales_.get(DNNL_ARG_SRC_1);
        attr_info.with_src1_scale = !src1_scales.has_default_values();
        attr_info.src1_scale = *src1_scales.scales_;
        assert(src1_scales.mask_ == 0);

        // zero points
        const auto &zp = attr->zero_points_;
        attr_info.with_src_zpoints = !zp.has_default_values(DNNL_ARG_SRC);
        attr_info.with_dst_zpoints = !zp.has_default_values(DNNL_ARG_DST);

        attr_info.with_per_ic_src_zpoints = attr_info.with_src_zpoints
                && !zp.defined(DNNL_ARG_SRC) && !zp.common(DNNL_ARG_SRC);
        attr_info.common_src_zpoint
                = attr_info.with_src_zpoints && zp.defined(DNNL_ARG_SRC)
                ? *zp.get(DNNL_ARG_SRC)
                : 0;

        attr_info.with_per_oc_dst_zpoints = attr_info.with_dst_zpoints
                && !zp.defined(DNNL_ARG_DST) && !zp.common(DNNL_ARG_DST);
        attr_info.common_dst_zpoint
                = attr_info.with_dst_zpoints && zp.defined(DNNL_ARG_DST)
                ? *zp.get(DNNL_ARG_DST)
                : 0;

        attr_info.initialized = true;
        return attr_info;
    }

    bool initialized = false;

    post_ops_t all_post_ops;

    bool with_binary;
    bool with_eltwise;
    int eltwise_idx;
    alg_kind_t eltwise_alg;
    float eltwise_scale;
    float eltwise_alpha;
    float eltwise_beta;

    bool with_sum;
    int sum_idx;
    float sum_scale;
    data_type_t sum_data_type;

    bool with_oscales;
    bool with_common_oscales;
    float common_oscales;
    bool with_per_oc_oscales;
    bool with_runtime_oscales;

    bool with_src0_scale;
    float src0_scale;

    bool with_src1_scale;
    float src1_scale;

    bool with_src_zpoints;
    bool with_dst_zpoints;
    bool with_per_ic_src_zpoints;
    bool with_per_oc_dst_zpoints;
    int common_src_zpoint;
    int common_dst_zpoint;
};

struct offsets_t {
    int src_off[4][MAX_NDIMS];
    int wei_off[4][MAX_NDIMS];
    int dst_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
};

struct rnn_offsets_t {
    int src_layer_off[4][MAX_NDIMS];
    int src_iter_off[4][MAX_NDIMS];
    int src_iter_c_off[4][MAX_NDIMS];
    int weights_layer_off[4][MAX_NDIMS];
    int weights_iter_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
    int dst_layer_off[4][MAX_NDIMS];
    int dst_iter_off[4][MAX_NDIMS];
    int dst_iter_c_off[4][MAX_NDIMS];
    int diff_src_layer_off[4][MAX_NDIMS];
    int diff_src_iter_off[4][MAX_NDIMS];
    int diff_src_iter_c_off[4][MAX_NDIMS];
    int diff_weights_layer_off[4][MAX_NDIMS];
    int diff_weights_iter_off[4][MAX_NDIMS];
    int diff_bias_off[4][MAX_NDIMS];
    int diff_dst_layer_off[4][MAX_NDIMS];
    int diff_dst_iter_off[4][MAX_NDIMS];
    int diff_dst_iter_c_off[4][MAX_NDIMS];
    int ws_off[4][MAX_NDIMS];
};

// Convolution
enum conv_version_t {
    ver_unused,
    ver_1stconv,
    ver_16mb16c,
    ver_8ow16c,
    ver_nhwc,
    ver_mb_block,
    ver_ow_block
};

struct conv_conf_t {
    prop_kind_t prop_kind;

    int ndims;
    int mb;
    int ngroups, ic, oc;
    int ngroups_without_padding, oc_without_padding, ic_without_padding;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;

    int sp_block;
    int od_block, oh_block, ow_block;
    int id_block, ih_block, iw_block;
    int oc_block, ic_block, nchunk;
    int omb;
    int odb, ohb, owb;
    int icb;
    int ocb;
    int osp_chunk, mb_chunk, mb_block, slm_ic;
    size_t wei_slm_size, src_slm_size;
    int sub_group_size;
    size_t gws_d[3], lws_d[3];
    compute::dispatch_t dispatch;

    bool with_bias, with_groups;

    attr_info_t attr_info;

    bool is_depthwise;
    bool is_nhwc;
    int ver;
    format_tag_t src_tag, dst_tag, wei_tag;
    bool is_nchw;
    bool is_src_nchw, is_src_nhwc;
    bool is_dst_nhwc;

    int tile_size;
    int wino_m;
    int wino_r;
    int wino_ih, wino_oh;
    int wino_iw, wino_ow;
    int wino_ic;
    int wino_oc;
    int wino_ic_block;
    int wino_oc_block;
    size_t U_gws_d[3], U_lws_d[3];
    size_t V_gws_d[3], V_lws_d[3];
    size_t M_gws_d[3], M_lws_d[3];

    data_type_t src_data_type;
    data_type_t weights_data_type;
    data_type_t bias_data_type;
    data_type_t dst_data_type;
    data_type_t acc_data_type;
};

// Pooling
struct pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int dd, dh, dw;
    int f_pad, t_pad, l_pad;
    data_type_t src_dt;
    data_type_t dst_dt;
    alg_kind_t alg;
    bool is_training, is_backward;
    bool use_mb_c_block, use_only_c_block;
    int chunks_per_c_block, chunks_per_mb_block;
    int vect_dt_n;
    int nvect;
    compute::dispatch_t dispatch;
    int sub_group_size;

    attr_info_t attr_info;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

// Inner Product
struct inner_product_conf_t {
    int ndims;
    int src_ndims, wei_ndims, dst_ndims;
    int mb, oc, ic, ic_total;
    int id, ih, iw, od, oh, ow;
    int kd, kh, kw;
    bool with_bias, has_spatial;
    bool is_forward, is_backward_data, is_backward_weights;
    compute::dispatch_t dispatch;
    bool reorder_dst = false;

    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;

    attr_info_t attr_info;
};

// RNN
struct rnn_conf_t {
    int cell_kind;
    int activation_kind;
    int direction_kind;
    bool with_bias;
    bool with_src_iter;
    bool with_src_iter_c;
    bool with_dst_iter;
    bool with_dst_iter_c;
    bool is_lbr;
    bool is_vanilla_gru;
    bool is_fwd;
    bool copy_bias;
    bool is_int8;
    bool is_testmode;
    bool is_training;
    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;
    data_type_t aux_dt;
    data_type_t input_dt;
    data_type_t output_dt;
    data_type_t diff_dt;

    int n_layer;
    int n_dir;
    int n_iter;
    int n_iter_scratch_gates;
    int n_gates;
    int n_bias;
    int n_states;
    int n_weights_input;
    int n_weights_state;
    int batch;
    int slc;
    int sic;
    int dhc;
    int dlc;
    int wic;
    int n_parts_weights_iter, n_parts_weights_layer;
    int src_layer_ndims;
    int src_iter_ndims;
    int src_iter_c_ndims;
    int weights_layer_ndims;
    int weights_iter_ndims;
    int dst_layer_ndims;
    int dst_iter_ndims;
    int dst_iter_c_ndims;
    int bias_ndims;
    int diff_src_layer_ndims;
    int diff_src_iter_ndims;
    int diff_src_iter_c_ndims;
    int diff_weights_layer_ndims;
    int diff_weights_iter_ndims;
    int diff_dst_layer_ndims;
    int diff_dst_iter_ndims;
    int diff_dst_iter_c_ndims;
    int diff_bias_ndims;
    int states_ws_ld, gates_ws_ld, diff_states_ws_ld, scratch_gates_ld;

    int wei_qparam_mask;

    size_t ws_gates_offset;
    size_t ws_states_offset;
    size_t ws_diff_states_offset;
    size_t ws_grid_comp_offset;
    size_t scratch_cell_offset;
    size_t ws_dhG1_offset;
    size_t ws_h_state_offset;
    size_t ws_c_state_offset;
    size_t ws_bias_offset;
    size_t scratch_gates_offset;
    size_t scratchpad_size;
    size_t workspace_size;
};

struct rnn_reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;
    compute::dispatch_t dispatch;
    int block[3];
    int sub_group_size;
    int mask;
    size_t scales_count;
};

// Batch Normalization
struct bnorm_conf_t {
    data_type_t data_type;

    int ndims;
    int mb, ic, mb_block, ic_block;
    int reduce_dim_idx, reduce_dim;
    int id, ih, iw;
    int nn, sp, sp_tail, vect_size;
    int stat_sp_nblocks, stat_sp_tail, stat_sp_block;
    int reduce_stat_nblocks;
    bool with_relu, use_16mb_unroll, use_nhwc;
    int stat_ic;
    bool is_forward, is_backward;
    bool use_scaleshift, save_stats, is_training;
    bool fuse_norm_relu, calculate_stats, calculate_diff_stats;
    bool diff_scaleshift;
    float relu_negative_slope, eps;

    compute::dispatch_t dispatch_calc_stat;
    compute::dispatch_t dispatch_reduce_stat;
    compute::dispatch_t dispatch;
};

// Layer Normalization
struct lnorm_conf_t {
    data_type_t data_type;

    bool is_fwd;
    int ndims;
    int norm_axis;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t stat_md_info;

    bool use_scaleshift;
    bool calculate_stats;
    bool save_stats;
    float eps;

    compute::dispatch_t dispatch_scaleshift;
    compute::dispatch_t dispatch;
};

// Binary
struct binary_conf_t {
    int ndims, nvect;
    bool use_unroll_16b, src0_unroll_16b;
    bool is_ncX_layout;
    data_type_t src0_data_type;
    data_type_t src1_data_type;
    data_type_t dst_data_type;
    bool is_mul;
    bool is_add;
    bool is_max;
    bool is_min;
    bool is_div;
    bool is_tensor_op;
    compute::dispatch_t dispatch;
    int dim0[MAX_NDIMS];
    int bcast_dims[MAX_NDIMS];
    bool is_dense;
    bool is_same_md;
    bool same_src_dt;
    bool with_binary_post_op;

    memory_desc_info_t src0_md_info;
    memory_desc_info_t src1_md_info;
    memory_desc_info_t dst_md_info;

    attr_info_t attr_info;
};

// Reorder
struct reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool scale_quant, with_sum_ab, with_sum_a;
    bool use_ref_impl, use_dense_vect;
    bool vectorize_last_dim;
    int ndims;
    size_t nelems;

    compute::dispatch_t dispatch;

    int sub_group_size;
    int scale_mask;
    size_t scales_num;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

// Concat
struct concat_conf_t {
    dim_t dst_extern_dim_size;
    dim_t src_extern_dim_sizes[16];
    dim_t offset[16];
    dim_t inner_axis;
    int block;
    int n;
    int simd;
    int data_type_size;
    size_t gws_d[3], lws_d[3];
};

// Elementwise
struct eltwise_conf_t {
    int ndims;
    bool with_zero_padding;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    compute::dispatch_t dispatch;
    memory_desc_info_t data_md_info;
    memory_desc_info_t data_diff_md_info;

    attr_info_t attr_info;
};

// Shuffle
struct shuffle_conf_t {
    data_type_t data_type;
    int axis;
    int axis_size;
    int group_size;
    int transpose_row;
    int transpose_col;
    size_t outer_size;
    size_t inner_size;
    size_t dim;
    int ndims;
    size_t gws_d[3];
};

inline void set_default_pool_conf(pool_conf_t &conf,
        const pooling_v2_desc_t &desc, const memory_desc_t &src_md,
        const memory_desc_t &dst_md, const primitive_attr_t &attr) {
    const memory_desc_wrapper src_mdw(src_md);
    const memory_desc_wrapper dst_mdw(dst_md);

    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    int ndims = src_mdw.ndims();
    conf.ndims = ndims;

    conf.mb = src_dims[0];

    conf.c = src_dims[1];
    conf.id = (ndims == 5) ? src_dims[2] : 1;
    conf.ih = (ndims == 3) ? 1 : src_dims[ndims - 2];
    conf.iw = src_dims[ndims - 1];
    conf.od = (ndims == 5) ? dst_dims[2] : 1;
    conf.oh = (ndims == 3) ? 1 : dst_dims[ndims - 2];
    conf.ow = dst_dims[ndims - 1];

    conf.stride_d = (ndims == 5) ? desc.strides[0] : 1;
    conf.stride_h = (ndims == 3) ? 1 : desc.strides[ndims - 4];
    conf.stride_w = desc.strides[ndims - 3];
    conf.kd = (ndims == 5) ? desc.kernel[0] : 1;
    conf.kh = (ndims == 3) ? 1 : desc.kernel[ndims - 4];
    conf.kw = desc.kernel[ndims - 3];

    if (desc.primitive_kind != dnnl_pooling_v2) {
        conf.dd = conf.dh = conf.dw = 0;
    } else {
        conf.dd = (ndims == 5) ? desc.dilation[0] : 0;
        conf.dh = (ndims == 3) ? 0 : desc.dilation[ndims - 4];
        conf.dw = desc.dilation[ndims - 3];
    }

    conf.f_pad = (ndims == 5) ? desc.padding[0][0] : 0;
    conf.t_pad = (ndims == 3) ? 0 : desc.padding[0][ndims - 4];
    conf.l_pad = desc.padding[0][ndims - 3];

    conf.alg = desc.alg_kind;

    conf.src_dt = src_mdw.data_type();
    conf.dst_dt = dst_mdw.data_type();

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    conf.is_training = desc.prop_kind == prop_kind::forward_training;
    conf.is_backward = desc.prop_kind == prop_kind::backward_data;

    conf.attr_info = attr_info_t::create(&attr);
}

inline void set_default_conf(conv_conf_t &conf, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr) {

    const memory_desc_wrapper src_mdw(&src_md);
    const memory_desc_wrapper weights_mdw(&weights_md);
    const memory_desc_wrapper dst_mdw(&dst_md);
    const memory_desc_wrapper bias_mdw(&bias_md);

    const bool with_groups = weights_mdw.ndims() == src_mdw.ndims() + 1;
    int ndims = src_mdw.ndims();

    conf = utils::zero<decltype(conf)>();
    conf.with_groups = with_groups;
    conf.ndims = ndims;
    conf.prop_kind = cd.prop_kind;
    conf.ngroups = with_groups ? weights_mdw.dims()[0] : 1;
    conf.mb = src_mdw.dims()[0];
    conf.oc_without_padding = dst_mdw.dims()[1] / conf.ngroups;
    conf.ic_without_padding = src_mdw.dims()[1] / conf.ngroups;
    conf.id = (ndims == 5) ? src_mdw.dims()[2] : 1;
    conf.ih = (ndims == 3) ? 1 : src_mdw.dims()[ndims - 2];
    conf.iw = src_mdw.dims()[ndims - 1];
    conf.od = (ndims == 5) ? dst_mdw.dims()[2] : 1;
    conf.oh = (ndims == 3) ? 1 : dst_mdw.dims()[ndims - 2];
    conf.ow = dst_mdw.dims()[ndims - 1];
    conf.kd = (ndims == 5) ? weights_mdw.dims()[with_groups + 2] : 1;
    conf.kh = (ndims == 3) ? 1 : weights_mdw.dims()[with_groups + ndims - 2];
    conf.kw = weights_mdw.dims()[with_groups + ndims - 1];

    conf.is_depthwise = conf.with_groups && conf.oc_without_padding == 1
            && conf.ic_without_padding == 1;
    conf.oc = dst_mdw.dims()[1] / conf.ngroups;
    conf.ic = src_mdw.dims()[1] / conf.ngroups;

    conf.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    conf.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    conf.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    conf.b_pad = (ndims == 3) ? 0 : cd.padding[1][ndims - 4];
    conf.l_pad = cd.padding[0][ndims - 3];
    conf.r_pad = cd.padding[1][ndims - 3];
    conf.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    conf.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    conf.stride_w = cd.strides[ndims - 3];
    conf.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    conf.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    conf.dilate_w = cd.dilates[ndims - 3];

    conf.with_bias = bias_mdw.format_kind() != format_kind::undef;

    conf.src_data_type = src_mdw.data_type();
    conf.weights_data_type = weights_mdw.data_type();
    conf.dst_data_type = dst_mdw.data_type();
    conf.acc_data_type = cd.accum_data_type;
    conf.bias_data_type
            = conf.with_bias ? bias_mdw.data_type() : data_type::f32;

    conf.attr_info = attr_info_t::create(&attr);
}

inline void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        const int block = block_dims[d];

        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < md.ndims()) ? block : 1);
        kernel_ctx.define_int(utils::format("%s_S%d", str, d),
                (d < md.ndims()) ? strides_compat[0][d] : 0);
        kernel_ctx.define_int(utils::format("%s_SB%d", str, d),
                (d < md.ndims()) ? strides_compat[1][d] : 0);
    }

    kernel_ctx.define_int(utils::format("%s_OFFSET_PAD", str), md.md_->offset0);
}

inline void set_offsets(const memory_desc_wrapper &md, int offs[4][MAX_NDIMS]) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);
    const dims_t &dims = md.dims();

    for (int d = 0; d < md.ndims(); ++d) {
        const int block = block_dims[d];

        offs[0][d] = block;
        offs[1][d] = strides_compat[0][d];
        offs[2][d] = strides_compat[1][d];
        offs[3][d] = dims[d];
    }
}

inline void def_offsets(const int offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str, const int ndims) {

    for (int d = 0; d < MAX_NDIMS; d++) {
        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < ndims) ? offs[0][d] : 1);
        kernel_ctx.define_int(
                utils::format("%s_S%d", str, d), (d < ndims) ? offs[1][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_SB%d", str, d), (d < ndims) ? offs[2][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_D%d", str, d), (d < ndims) ? offs[3][d] : 0);
    }
}

inline void def_data_type(
        compute::kernel_ctx_t &kernel_ctx, data_type_t dt, const char *str) {
    switch (dt) {
        case data_type::bf16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=ushort -D%s_DT_BF16", str, str));
            break;
        case data_type::f16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=half -D%s_DT_F16", str, str));
            break;
        case data_type::f32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=float -D%s_DT_F32", str, str));
            break;
        case data_type::s8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=char -D%s_DT_S8", str, str));
            break;
        case data_type::u8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=uchar -D%s_DT_U8", str, str));
            break;
        case data_type::s32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=int -D%s_DT_S32", str, str));
            break;
        default: assert(!"unsupported data type"); break;
    }
}

inline void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_info_t &md_info, const char *prefix) {
    def_data_type(kernel_ctx, md_info.data_type, prefix);

    kernel_ctx.define_int(utils::format("%s_OFFSET0", prefix), md_info.offset0);
    kernel_ctx.define_int(utils::format("%s_NDIMS", prefix), md_info.ndims);

    kernel_ctx.define_int(utils::format("%s_NLEVELS", prefix), md_info.nlevels);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        int dim = (d < md_info.ndims) ? md_info.dims[d] : 1;
        int padded_dim = (d < md_info.ndims) ? md_info.padded_dims[d] : 1;
        kernel_ctx.define_int(utils::format("%s_D%d", prefix, d), dim);
        kernel_ctx.define_int(utils::format("%s_PD%d", prefix, d), padded_dim);

        for (int l = 0; l < md_info.nlevels + 1; ++l) {
            int block = (d < md_info.ndims) ? md_info.blocks[d][l] : 1;
            int stride = (d < md_info.ndims) ? md_info.strides[d][l] : 0;
            kernel_ctx.define_int(
                    utils::format("%s_B%d_%d", prefix, d, l), block);
            kernel_ctx.define_int(
                    utils::format("%s_S%d_%d", prefix, d, l), stride);
        }
    }
}

inline void def_binary_alg_kinds(compute::kernel_ctx_t &kernel_ctx) {
    kernel_ctx.define_int("BINARY_ADD", alg_kind::binary_add);
    kernel_ctx.define_int("BINARY_MUL", alg_kind::binary_mul);
    kernel_ctx.define_int("BINARY_MIN", alg_kind::binary_min);
    kernel_ctx.define_int("BINARY_MAX", alg_kind::binary_max);
    kernel_ctx.define_int("BINARY_DIV", alg_kind::binary_div);
}

inline void def_eltwise_alg_kinds(compute::kernel_ctx_t &kernel_ctx) {
    kernel_ctx.define_int("RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("LINEAR", alg_kind::eltwise_linear);
    kernel_ctx.define_int("BOUNDED_RELU", alg_kind::eltwise_bounded_relu);
    kernel_ctx.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
    kernel_ctx.define_int("LOGISTIC", alg_kind::eltwise_logistic);
    kernel_ctx.define_int("TANH", alg_kind::eltwise_tanh);
    kernel_ctx.define_int("ELU", alg_kind::eltwise_elu);
    kernel_ctx.define_int("SQUARE", alg_kind::eltwise_square);
    kernel_ctx.define_int("SQRT", alg_kind::eltwise_sqrt);
    kernel_ctx.define_int("ABS", alg_kind::eltwise_abs);
    kernel_ctx.define_int("EXP", alg_kind::eltwise_exp);
    kernel_ctx.define_int("GELU_TANH", alg_kind::eltwise_gelu_tanh);
    kernel_ctx.define_int("SWISH", alg_kind::eltwise_swish);
    kernel_ctx.define_int("LOG", alg_kind::eltwise_log);
    kernel_ctx.define_int("CLIP", alg_kind::eltwise_clip);
    kernel_ctx.define_int("POW", alg_kind::eltwise_pow);
    kernel_ctx.define_int("GELU_ERF", alg_kind::eltwise_gelu_erf);
    kernel_ctx.define_int("ROUND", alg_kind::eltwise_round);

    kernel_ctx.define_int("RELU_DST", alg_kind::eltwise_relu_use_dst_for_bwd);
    kernel_ctx.define_int(
            "LOGISTIC_DST", alg_kind::eltwise_logistic_use_dst_for_bwd);
    kernel_ctx.define_int("TANH_DST", alg_kind::eltwise_tanh_use_dst_for_bwd);
    kernel_ctx.define_int("ELU_DST", alg_kind::eltwise_elu_use_dst_for_bwd);
    kernel_ctx.define_int("SQRT_DST", alg_kind::eltwise_sqrt_use_dst_for_bwd);
    kernel_ctx.define_int("EXP_DST", alg_kind::eltwise_exp_use_dst_for_bwd);
}

inline bool post_ops_with_binary_ok(
        const primitive_attr_t *attr, const data_type_t dst_dt) {
    const auto &p = attr->post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };
    auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };

    bool is_po_ok = true;
    for (int po_idx = 0; po_idx < p.len(); ++po_idx) {
        is_po_ok &= is_eltwise(po_idx) | is_sum(po_idx) | is_binary(po_idx);

        if (is_sum(po_idx)) {
            if (p.entry_[po_idx].sum.dt != dnnl_data_type_undef
                    && types::data_type_size(p.entry_[po_idx].sum.dt)
                            != types::data_type_size(dst_dt))
                return false;
        }
    }

    if (p.len() > 10) is_po_ok = false;

    return is_po_ok;
}

inline void def_post_ops_cfg(
        compute::kernel_ctx_t &kernel_ctx, const post_ops_t &all_post_ops) {
    const int po_nop_id = 0;
    const int po_binary_id = 1;
    const int po_eltwise_id = 2;
    const int po_sum_id = 3;

    kernel_ctx.define_int("PO_BINARY", po_binary_id);
    kernel_ctx.define_int("PO_ELTWISE", po_eltwise_id);
    kernel_ctx.define_int("PO_SUM", po_sum_id);

    std::string po_kernel_args = "-DPOST_OP_ARGS=\"";
    int nof_supported_post_ops = 0;

    auto add_po_defines = [&](const std::string &bin_arg_name,
                                  const post_ops_t::entry_t &e, int idx) {
        if (e.is_binary()) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_binary_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", e.binary.alg);

            const memory_desc_wrapper src1_mdw(e.binary.src1_desc);
            const auto mdi = memory_desc_info_t::create(src1_mdw);
            def_memory_desc_info(kernel_ctx, mdi, bin_arg_name.c_str());
            if (mdi.data_type == data_type::bf16) {
                kernel_ctx.define_int(
                        "PO_" + std::to_string(idx) + "_BIN_ARG_DT_IS_BF16", 1);
            } else {
                kernel_ctx.define_int(
                        "PO_" + std::to_string(idx) + "_BIN_ARG_DT_IS_BF16", 0);
            }
        } else {
            dnnl_memory_desc_t empty_mem_desc;
            dnnl_dims_t empty_dims = {1, 1, 1, 1};
            dnnl_memory_desc_init_by_tag(&empty_mem_desc, 4, empty_dims,
                    data_type_t::dnnl_s8, format_tag_t::dnnl_nchw);
            const memory_desc_wrapper src1_mdw(empty_mem_desc);
            const auto mdi = memory_desc_info_t::create(src1_mdw);
            def_memory_desc_info(kernel_ctx, mdi, bin_arg_name.c_str());
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_BIN_ARG_DT_IS_BF16", 0);
        }
        if (e.is_eltwise(false)) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_eltwise_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", e.eltwise.alg);
        }
        if (e.is_sum(false)) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_sum_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", alg_kind::undef);
        }
        if (!(e.is_binary() || e.is_eltwise(false) || e.is_sum(false))) {
            // empty post op
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_nop_id);
            // *_ALG need to be set but it's unused when kind is NOP
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", alg_kind::undef);
            --nof_supported_post_ops;
        }
        po_kernel_args += ", const __global PO_" + std::to_string(idx)
                + "_BIN_ARG_DATA_T *po_" + std::to_string(idx) + "_binary_arg";
        po_kernel_args
                += ", float po_" + std::to_string(idx) + "_eltwise_alpha";
        po_kernel_args += ", float po_" + std::to_string(idx) + "_eltwise_beta";
        po_kernel_args
                += ", float po_" + std::to_string(idx) + "_eltwise_scale";
        po_kernel_args += ", float po_" + std::to_string(idx) + "_sum_scale";
    };

    for (int idx = 0; idx < all_post_ops.len();
            ++idx, ++nof_supported_post_ops) {
        const std::string bin_arg_name
                = "PO_" + std::to_string(idx) + "_BIN_ARG";
        add_po_defines(bin_arg_name, all_post_ops.entry_[idx], idx);
    }
    post_ops_t::entry_t empty_po = post_ops_t::entry_t();
    for (int idx = all_post_ops.len(); idx < 10;
            ++idx, ++nof_supported_post_ops) {
        const std::string bin_arg_name
                = "PO_" + std::to_string(idx) + "_BIN_ARG";
        add_po_defines(bin_arg_name, empty_po, idx);
    }

    kernel_ctx.define_int("POST_OP_CHAIN_LENGTH", nof_supported_post_ops);
    if (all_post_ops.len() > 0) {
        // due to C macro limitations on which post op service is build always
        // load bf16 convertion functions
        kernel_ctx.define_int("POST_OP_USING_BF16", 1);
    }
    po_kernel_args += "\"";
    kernel_ctx.add_option(po_kernel_args);
}

inline int append_post_ops_to_arg_list(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &all_post_ops) {
    auto set_arg_entry = [&](const post_ops_t::entry_t &e, int po_idx) {
        if (e.is_binary()) {
            auto &binary_arg = CTX_IN_STORAGE(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx) | DNNL_ARG_SRC_1);
            arg_list.set(post_op_idx++, binary_arg);
        } else {
            arg_list.set(post_op_idx++, memory_storage_t::empty_storage());
        }

        if (e.is_eltwise()) {
            arg_list.set(post_op_idx++, e.eltwise.alpha);
            arg_list.set(post_op_idx++, e.eltwise.beta);
            arg_list.set(post_op_idx++, e.eltwise.scale);
        } else {
            arg_list.set(post_op_idx++, 1.0f); // _eltwise_alpha
            arg_list.set(post_op_idx++, 0.0f); // _eltwise_beta
            arg_list.set(post_op_idx++, 1.0f); // _eltwise_scale
        }

        if (e.is_sum(false)) {
            arg_list.set(post_op_idx++, e.sum.scale);
        } else {
            arg_list.set(post_op_idx++, 1.0f);
        }
    };

    for (int idx = 0; idx < all_post_ops.len(); ++idx) {
        set_arg_entry(all_post_ops.entry_[idx], idx);
    }
    post_ops_t::entry_t empty_po = post_ops_t::entry_t();
    for (int idx = all_post_ops.len(); idx < 10; ++idx) {
        set_arg_entry(empty_po, 0);
    }
    return post_op_idx;
}

inline bool post_ops_preserves_zeroes(
        const exec_ctx_t &ctx, const post_ops_t &all_post_ops) {
    bool preserve_zeroes = true;
    for (int idx = 0; idx < all_post_ops.len(); ++idx) {
        const post_ops_t::entry_t &po_entry = all_post_ops.entry_[idx];
        if (po_entry.is_binary()) {
            // only binary mul is preserving zeroes
            preserve_zeroes &= po_entry.binary.alg
                    == dnnl::impl::alg_kind_t::dnnl_binary_mul;
        }
        if (po_entry.is_eltwise(false)) {
            preserve_zeroes &= gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                    po_entry.eltwise.alg, po_entry.eltwise.alpha,
                    po_entry.eltwise.beta);
        }
    }
    return preserve_zeroes;
}

inline void def_attr_info(
        compute::kernel_ctx_t &kernel_ctx, const attr_info_t &attr_info) {
    assert(attr_info.initialized);

    kernel_ctx.define_int("WITH_POST_OP", attr_info.all_post_ops.len() > 0);

    kernel_ctx.define_int("WITH_ELTWISE", attr_info.with_eltwise);
    kernel_ctx.define_int("ELTWISE_IDX", attr_info.eltwise_idx);
    kernel_ctx.define_int("ELTWISE_ALG", attr_info.eltwise_alg);
    kernel_ctx.define_int("ELTWISE_ALPHA0", attr_info.eltwise_alpha == 0.0f);

    kernel_ctx.define_int("WITH_SUM", attr_info.with_sum);
    kernel_ctx.define_int("SUM_IDX", attr_info.sum_idx);
    kernel_ctx.define_int("SUM_SCALE", attr_info.sum_scale);
    kernel_ctx.define_int("SUM_SCALE1", attr_info.sum_scale == 1.0f);

    kernel_ctx.define_int("WITH_SRC0_SCALE", attr_info.with_src0_scale);
    kernel_ctx.define_int("WITH_SRC1_SCALE", attr_info.with_src1_scale);

    kernel_ctx.define_int("WITH_SCALES", attr_info.with_oscales);
    kernel_ctx.define_int("SCALES_PER_OC", attr_info.with_per_oc_oscales);
    kernel_ctx.define_int("SCALES_COMMON", attr_info.with_common_oscales);

    kernel_ctx.define_int("WITH_SRC_ZPOINTS", attr_info.with_src_zpoints);
    kernel_ctx.define_int("WITH_DST_ZPOINTS", attr_info.with_dst_zpoints);
    kernel_ctx.define_int("SRC_ZPOINT_COMMON", attr_info.common_src_zpoint);
    kernel_ctx.define_int("DST_ZPOINT_COMMON", attr_info.common_dst_zpoint);
    kernel_ctx.define_int(
            "WITH_SRC_ZPOINTS_PER_IC", attr_info.with_per_ic_src_zpoints);
    kernel_ctx.define_int(
            "WITH_DST_ZPOINTS_PER_OC", attr_info.with_per_oc_dst_zpoints);

    def_binary_alg_kinds(kernel_ctx);
    def_eltwise_alg_kinds(kernel_ctx);

    def_post_ops_cfg(kernel_ctx, attr_info.all_post_ops);
}

inline void def_dispatch(compute::kernel_ctx_t &kernel_ctx,
        const compute::dispatch_t &dispatch) {
    dispatch.def_kernel_macros(kernel_ctx);
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
