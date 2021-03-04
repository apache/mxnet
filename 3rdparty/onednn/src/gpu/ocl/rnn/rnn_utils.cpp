/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/ocl/rnn/rnn_utils.hpp"

#include "common/c_types_map.hpp"
#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
#define AOC array_offset_calculator

using namespace dnnl::impl::utils;
using namespace prop_kind;
using namespace data_type;

bool rnn_utils::is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

void rnn_utils::init_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_kind == dnnl_lbr_gru;
    rnn.is_vanilla_gru = rd.cell_kind == dnnl_vanilla_gru;

    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (everyone_is(bf16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        rnn.dt_conf = all_bf16;
    else if (everyone_is(f16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        rnn.dt_conf = all_f16;
    else if (dst_layer_d.data_type() == u8) {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }
    rnn.is_int8 = !one_of(rnn.dt_conf, all_f32, all_f16, all_bf16);

    rnn.aux_data_type
            = rnn.dt_conf == all_f16 ? data_type::f16 : data_type::f32;
    rnn.diff_data_type = data_type::f32;

    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dhc = weights_layer_d.dims()[4];
    rnn.dlc = dst_layer_d.dims()[2];

    rnn.gates_ld = rnn.dhc * rnn.n_gates;
    rnn.gates_nld = rnn.mb;
    rnn.states_nld = rnn.mb;

    // Set the correct number of weights parts
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    //there are two parts for VANILLA GRU weights iteration
    rnn.n_parts_weights_iter = rnn.is_vanilla_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = rnn.is_vanilla_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = rnn.is_vanilla_gru ? 1 : 0;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    bool is_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::lbr_gru);

    // Decide if to merge gemm across iterations or layers
    auto src_layer_ld = src_layer_d.blocking_desc().strides[1];
    auto dst_layer_ld = dst_layer_d.blocking_desc().strides[1];
    auto src_layer_is_trivial_stride
            = src_layer_d.blocking_desc().strides[0] == (src_layer_ld * rnn.mb);
    auto dst_layer_is_trivial_stride
            = dst_layer_d.blocking_desc().strides[0] == (dst_layer_ld * rnn.mb);

    rnn.merge_gemm_layer = ((rnn.is_fwd && src_layer_is_trivial_stride)
                                   || ((rd.prop_kind == prop_kind::backward)
                                           && dst_layer_is_trivial_stride))
            && (((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd) || rnn.is_int8);
    rnn.merge_gemm_iter
            = dst_layer_is_trivial_stride && !(rnn.is_fwd || is_gru);

    // Decide to copy bias
    rnn.copy_bias = rnn.is_int8;

    rnn.use_workspace = rnn.is_training;

    switch (rnn.dt_conf) {
        case all_f32:
        case f32u8f32f32:
            rnn.input_data_type = f32;
            rnn.dst_data_type = f32;
            rnn.output_data_type = f32;
            break;
        case all_f16:
            rnn.input_data_type = f16;
            rnn.dst_data_type = f16;
            rnn.output_data_type = f16;
            break;
        case u8u8u8u8:
            rnn.input_data_type = u8;
            rnn.dst_data_type = u8;
            rnn.output_data_type = u8;
            break;
        case u8u8u8f32:
            rnn.input_data_type = u8;
            rnn.dst_data_type = f32;
            rnn.output_data_type = u8;
            break;
        case f32u8f32u8:
            rnn.input_data_type = f32;
            rnn.dst_data_type = u8;
            rnn.output_data_type = f32;
            break;
        case all_bf16:
            rnn.input_data_type = bf16;
            rnn.dst_data_type = bf16;
            rnn.output_data_type = bf16;
            break;
        default: assert(!"unimplemented");
    }
}

void rnn_utils::init_test_mode(conf_t &rnn, const primitive_attr_t &attr) {
    rnn.is_testmode = attr.rnn_tparams_.test_mode_;
    rnn.tm_ngates = attr.rnn_tparams_.ngates_;
    rnn.tm_cscale = attr.rnn_tparams_.cscale_;
}

void rnn_utils::set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    //Set leading dimensions for input weights arrays depending on input format
    auto set_dims = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
        ld = 0;
        nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = (int)md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = (int)md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
    }

    int sizeof_states_dt
            = rnn.dt_conf == all_f32 ? sizeof(cl_float) : sizeof(cl_half);
    int aux_elsz = rnn.aux_data_type == data_type::f16 ? sizeof(cl_half)
                                                       : sizeof(float);
    rnn.ws_states_elsz = rnn.dt_conf == all_f32
            ? sizeof(cl_float)
            : rnn.dt_conf == all_f16 || rnn.dt_conf == all_bf16
                    ? sizeof(cl_half)
                    : rnn.dt_conf == u8u8u8u8 ? sizeof(int8_t)
                                              : sizeof(int32_t);

    // Different size required for forward and backward pass
    rnn.scratch_gates_elsz = (!rnn.is_fwd && rnn.dt_conf == all_bf16)
            ? sizeof(cl_half)
            : aux_elsz;

    // Set workspace sizes to store:
    // states to copmute a pass
    // diff states to copmute bwd pass (training only)
    // intermediate results from the gates
    rnn.states_ws_ld = get_good_ld(
            nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc)), sizeof_states_dt);
    rnn.gates_ws_ld = get_good_ld(rnn.gates_ld,
            rnn.dt_conf == all_f16 ? sizeof(cl_half) : sizeof(cl_float));
    rnn.diff_states_ws_ld = get_good_ld(
            nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc)), sizeof(cl_float));
    rnn.scratch_gates_ld = get_good_ld(rnn.gates_ld, rnn.scratch_gates_elsz);

    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;

    rnn.ws_states_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.states_ws_ld * rnn.ws_states_elsz;
    // we do not need a good ld for iter_c as it is not involved in GEMM
    // for now reverting it back to what it was originally
    // TODO: seprate diff_c_offsets from diff-states & seprate h- and c- off
    rnn.ws_c_states_size = is_lstm ? (size_t)(rnn.n_layer + 1) * rnn.n_dir
                    * (rnn.n_iter + 1) * rnn.mb * rnn.states_ws_ld * aux_elsz
                                   : (size_t)0;
    rnn.ws_diff_states_size = rnn.is_training ? (size_t)(rnn.n_layer + 1)
                    * rnn.n_dir * (rnn.n_states + 1) * (rnn.n_iter + 1) * rnn.mb
                    * rnn.diff_states_ws_ld * aux_elsz
                                              : (size_t)0;
    rnn.ws_gates_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.mb
            * rnn.gates_ws_ld * aux_elsz;
    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;
    rnn.scratch_gates_size = (size_t)rnn.n_iter_scratch_gates * rnn.gates_nld
            * rnn.scratch_gates_ld * rnn.scratch_gates_elsz;
    rnn.ws_dhG1_size
            = (rd.cell_kind == alg_kind::vanilla_gru && rnn.is_training)
            ? (size_t)rnn.gates_nld * rnn.diff_states_ws_ld * sizeof(float)
            : 0;
    rnn.ws_bias_size
            = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc * aux_elsz;

    // For intermediate step in post-gemm fwd lbr gru
    rnn.scratch_cell_size = rnn.is_lbr
            ? (size_t)rnn.gates_nld * rnn.scratch_gates_ld
                    * rnn.scratch_gates_elsz
            : (rd.cell_kind == alg_kind::vanilla_gru && rnn.is_training
                            ? (size_t)rnn.states_nld * rnn.states_ws_ld
                                    * rnn.ws_states_elsz
                            : 0);

    // Used for storing the intermediate value from fwd pass in training lbr gru
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dhc * aux_elsz;
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * aux_elsz;
}

int rnn_utils::get_good_ld(int dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    int ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_c_states_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &scratch_cell_offset, size_t &ws_dhG1_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratchpad_size, size_t &workspace_size) {

    const size_t page_size = 4096;
    size_t current_offset;

    // Mandatory workspaces: go to workspace if use_workspace, scratchpad
    // otherwise assumes the workspace base pointer is page aligned

    current_offset = 0;
    ws_gates_offset = current_offset;
    current_offset += rnn.ws_gates_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_states_offset = current_offset;
    current_offset += rnn.ws_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_c_states_offset = current_offset;
    current_offset += rnn.ws_c_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_grid_comp_offset = current_offset;
    current_offset += rnn.ws_grid_comp_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_diff_states_offset = current_offset;
    current_offset += rnn.ws_diff_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_dhG1_offset = current_offset;
    current_offset += rnn.ws_dhG1_size;

    workspace_size = rnn.use_workspace ? current_offset : 0;

    // Optional scratchpads
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    current_offset = utils::rnd_up(current_offset, page_size);
    scratch_gates_offset = current_offset;
    current_offset += rnn.scratch_gates_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    scratch_cell_offset = current_offset;
    current_offset += rnn.scratch_cell_size;

    if (rnn.copy_bias) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_bias_offset = current_offset;
        current_offset += rnn.ws_bias_size;
    }
    scratchpad_size = current_offset;
}

void rnn_utils::get_scratchpad_and_workspace_sizes(
        const conf_t &rnn, size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_diff_states_offset, ws_grid_comp_offset, scratch_cell_offset,
            ws_dhG1_offset, ws_bias_offset, sratch_gates_offset;
    set_offsets(rnn, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, scratch_cell_offset,
            ws_dhG1_offset, ws_bias_offset, sratch_gates_offset,
            scratchpad_size, workspace_size);
}

void rnn_utils::set_offsets_fwd_gemm(const conf_t &rnn, int dir, int lay,
        data_type_t src_t, size_t *wei_layer_off_ptr,
        const size_t &ws_states_offset_, size_t &grid_ws_lay_offset,
        size_t &grid_wei_lay_offset, size_t &grid_ws_iter_offset) {
    // Function overloaded. This function is called by grid execution
    int n_layer = rnn.n_layer;
    int n_dir = rnn.n_dir;

    AOC<size_t, 3> off_weights_lay(
            wei_layer_off_ptr, n_layer, n_dir, rnn.n_parts_weights_layer);

    grid_wei_lay_offset = off_weights_lay(lay, dir, 0);
    grid_ws_lay_offset = (cl_ulong)(ws_states_offset_
            + OFF4(lay, n_layer + 1, dir, n_dir, 1, rnn.n_iter + 1, 0,
                      rnn.mb * rnn.states_ws_ld)
                    * types::data_type_size(src_t));
    grid_ws_iter_offset = (cl_ulong)(ws_states_offset_
            + OFF4(lay + 1, rnn.n_layers + 1, dir, rnn.n_dir, 0, rnn.n_iter + 1,
                      0, rnn.mb * rnn.states_ws_ld)
                    * types::data_type_size(src_t));
    UNUSED(n_layer);
}

void rnn_utils::set_offsets_fwd_gemm(const conf_t &rnn, int iter, int dir,
        int lay, data_type_t src_t, size_t *wei_iter_off_ptr,
        const size_t &ws_states_offset_, size_t &cell_ws_iter_offset,
        size_t &cell_ws_lay_offset, size_t &cell_scratch_offset,
        size_t &cell_wei_iter_offset) {
    int n_layers = rnn.n_layer;
    int batch = rnn.mb;
    int n_iter = rnn.n_iter;
    int n_dir = rnn.n_dir;

    if (wei_iter_off_ptr) {
        AOC<size_t, 3> off_weights_iter(wei_iter_off_ptr, rnn.n_layer,
                rnn.n_dir, rnn.n_parts_weights_iter);
        cell_wei_iter_offset = off_weights_iter(lay, dir, 0);
    }

    cell_scratch_offset = (rnn.merge_gemm_iter || rnn.merge_gemm_layer)
            ? (cl_ulong)(
                    OFF2(iter, n_iter, 0, rnn.gates_nld * rnn.scratch_gates_ld)
                    * rnn.scratch_gates_elsz)
            : (size_t)0;
    cell_ws_iter_offset = (cl_ulong)(ws_states_offset_
            + OFF4(lay + 1, n_layers + 1, dir, n_dir, iter, n_iter + 1, 0,
                      batch * rnn.states_ws_ld)
                    * types::data_type_size(src_t));
    cell_ws_lay_offset = (cl_ulong)(ws_states_offset_
            + OFF4(lay, n_layers + 1, dir, n_dir, iter + 1, n_iter + 1, 0,
                      batch * rnn.states_ws_ld)
                    * types::data_type_size(src_t));
    UNUSED(n_layers);
}

void rnn_utils::set_gru_offsets_part2(const conf_t &rnn, int iter, int dir,
        int lay, data_type_t src_t, size_t *wei_iter_off_ptr,
        const size_t &ws_states_offset_, size_t &cell_wei_iter_offset,
        size_t &cell_scratch_offset, size_t &cell_ws_iter_offset) {

    AOC<size_t, 3> off_weights_iter(
            wei_iter_off_ptr, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_iter);
    cell_wei_iter_offset = off_weights_iter(lay, dir, 1);
    cell_scratch_offset += 2 * rnn.dhc * rnn.scratch_gates_elsz;
    cell_ws_iter_offset = (cl_ulong)(ws_states_offset_
            + OFF4(lay + 1, rnn.n_layers + 1, dir, rnn.n_dir, iter + 1,
                      rnn.n_iter + 1, 0, rnn.mb * rnn.states_ws_ld)
                    * types::data_type_size(src_t));
}

void rnn_utils::set_offsets_bwd_gemm(const conf_t &rnn, int iter, int dir,
        int lay, const size_t &ws_diff_states_off_,
        size_t &cell_diff_wei_iter_off, size_t &cell_diff_wei_lay_off,
        size_t &cell_diff_ws_lay_off) {
    // Function overloaded. This function is called by grid execution and it
    // then calls set_offsets_bwd_gemm which is otherwise called in cell exec
    cl_ulong dummy_var;
    set_offsets_bwd_gemm(rnn, iter, dir, lay, ws_diff_states_off_,
            cell_diff_wei_iter_off, cell_diff_wei_lay_off, cell_diff_ws_lay_off,
            dummy_var);
}

void rnn_utils::set_offsets_bwd_gemm(const conf_t &rnn, int iter, int dir,
        int lay, const size_t &ws_diff_states_off_,
        size_t &cell_diff_wei_iter_off, size_t &cell_diff_wei_lay_off,
        size_t &cell_diff_ws_lay_off, size_t &cell_diff_ws_iter_off,
        size_t &cell_diff_wei_iter_off2) {

    set_offsets_bwd_gemm(rnn, iter, dir, lay, ws_diff_states_off_,
            cell_diff_wei_iter_off, cell_diff_wei_lay_off, cell_diff_ws_lay_off,
            cell_diff_ws_iter_off);
    cell_diff_wei_iter_off2
            = cell_diff_wei_iter_off + 2 * rnn.dhc * sizeof(float);
}

void rnn_utils::set_offsets_bwd_gemm(const conf_t &rnn, int iter, int dir,
        int lay, const size_t &ws_diff_states_off_,
        size_t &cell_diff_wei_iter_off, size_t &cell_diff_wei_lay_off,
        size_t &cell_diff_ws_lay_off, size_t &cell_diff_ws_iter_off) {
    int n_layers = rnn.n_layer;
    int batch = rnn.mb;
    int n_iter = rnn.n_iter;
    int n_dir = rnn.n_dir;
    int n_states = rnn.n_states;

    cell_diff_ws_iter_off = ws_diff_states_off_
            + OFF5(lay, n_layers + 1, dir, n_dir, 0, n_states + 1, iter,
                      n_iter + 1, 0, rnn.states_nld * rnn.diff_states_ws_ld)
                    * sizeof(float);
    cell_diff_ws_lay_off = ws_diff_states_off_
            + OFF5(lay, n_layers + 1, dir, n_dir, n_states, n_states + 1, iter,
                      n_iter + 1, 0, rnn.states_nld * rnn.diff_states_ws_ld)
                    * sizeof(float);
    cell_diff_wei_lay_off
            = OFF3(lay, n_layers, dir, n_dir, 0,
                      rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ld)
            * sizeof(float);
    cell_diff_wei_iter_off
            = OFF3(lay, n_layers, dir, n_dir, 0,
                      rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ld)
            * sizeof(float);
    UNUSED(n_layers);
    UNUSED(batch);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;
    using namespace format_tag;

    if (tag == ldigo) {
        strides[2] = rnn_utils::get_good_ld((int)strides[2],
                (int)types::data_type_size(weights_md.data_type));
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[4] = rnn_utils::get_good_ld((int)strides[4],
                (int)types::data_type_size(weights_md.data_type));
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t rnn_utils::set_expected_desc(
        conf_t &rnn, memory_desc_t &weights_md, bool is_iter) {
    using namespace format_tag;
    CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));

    // Adjust strides for good leading dimension in GEMM
    CHECK(set_good_strides(weights_md, rnn.is_fwd ? ldigo : ldgoi));

    // set we need extra memory
    if (rnn.is_fwd && !one_of(rnn.dt_conf, all_f32, all_f16, all_bf16)) {
        weights_md.extra.flags = memory_extra_flags::gpu_rnn_u8s8_compensation;
        weights_md.extra.compensation_mask = 27; // ldigo 11011;
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
