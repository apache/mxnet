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
#include <algorithm>
#include <bitset>
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {

std::vector<const void *> prepare_binary_args(
        const post_ops_t &post_ops, const exec_ctx_t &ctx) {
    std::vector<const void *> post_ops_binary_rhs_arg_vec;
    post_ops_binary_rhs_arg_vec.reserve(post_ops.entry_.size());

    unsigned idx = 0;
    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_binary()) {
            post_ops_binary_rhs_arg_vec.emplace_back(CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
        }
        ++idx;
    }

    post_ops_binary_rhs_arg_vec.shrink_to_fit();

    return post_ops_binary_rhs_arg_vec;
}

static broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d,
        bool use_per_oc_spatial_strategy = true) {
    const int ndims = rhs_arg_md.ndims;
    const auto output_dims = injector_utils::make_output_dims(dst_d);

    bool all_ones = true;
    std::bitset<5> mask(0);
    for (int d = 0; d < ndims; d++) {
        const auto &rhs_arg_dim = rhs_arg_md.dims[d];

        if (rhs_arg_dim != 1) all_ones = false;

        if (output_dims[d] != rhs_arg_md.dims[d] || output_dims[d] == 1)
            mask.set(d);
    }

    if (all_ones)
        return broadcasting_strategy_t::scalar;
    else if (mask.none())
        return broadcasting_strategy_t::no_broadcast;

    const auto &mb_rhs = rhs_arg_md.dims[0];
    const bool broadcast_per_mb = !mask.test(0);
    const bool broadcast_per_oc = !mask.test(1);

    if (broadcast_per_mb && broadcast_per_oc && mb_rhs != 1) {
        return broadcasting_strategy_t::unsupported;
    } else if (broadcast_per_oc) {
        if (use_per_oc_spatial_strategy && dst_d.is_blocking_desc()) {
            const auto &strides = dst_d.blocking_desc().strides;

            //per_oc_spatial basically used in nchw data format
            return dst_d.is_plain() && strides[1] != 1
                            && strides[0] >= strides[1]
                            && IMPLICATION(ndims >= 3, strides[1] >= strides[2])
                    ? broadcasting_strategy_t::per_oc_spatial
                    : broadcasting_strategy_t::per_oc;
        } else {
            return broadcasting_strategy_t::per_oc;
        }
    }
    return broadcasting_strategy_t::unsupported;
}

bool binary_args_broadcast_supported(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return bcast_type == broadcasting_strategy_t::unsupported;
                }
                return false;
            });
}

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen) {
    const auto dims = injector_utils::make_output_dims(dst_d);
    const int vmm_l_len = vlen / 4;

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return utils::one_of(bcast_type,
                                   broadcasting_strategy_t::per_oc,
                                   broadcasting_strategy_t::per_oc_spatial)
                            && (dims[1] % vmm_l_len != 0);
                }
                return false;
            });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
                if (entry.is_binary()) {
                    const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
                    return rhs_arg_d.matches_tag(tag);
                }
                return true;
            });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                            == broadcasting_strategy_t::per_oc_spatial;
                }
                return false;
            });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> predicate) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    if (bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                                    == broadcasting_strategy_t::per_oc_spatial)
                        return predicate(
                                memory_desc_wrapper(entry.binary.src1_desc));
                }
                return true;
            });
}

template <cpu_isa_t isa>
jit_uni_binary_injector_t<isa>::jit_uni_binary_injector_t(
        jit_generator *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , use_per_oc_spatial_strategy_(static_params.use_per_oc_spatial_strategy) {}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_elem_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(out_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    }
    return true;
}

template <cpu_isa_t isa>
int jit_uni_binary_injector_t<isa>::adjust_temp_vmm_hint(
        int user_hint, int start_idx, int end_idx, int max_vmm_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_vmm_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        const bool max_vmm_idx_in_vector_range
                = max_vmm_idx >= start_idx && max_vmm_idx <= end_idx;

        if (max_vmm_idx_in_vector_range || user_hint_exceeded_limit
                || user_hint == max_vmm_idx)
            return 0;
        else
            return max_vmm_idx;
    }

    return user_hint;
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::push_vmm(const Vmm &vmm) const {
    host_->sub(host_->rsp, injector_utils::vmm_size_t<Vmm>::bytes);
    host_->uni_vmovups(host_->ptr[host_->rsp], vmm);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::pop_vmm(const Vmm &vmm) const {
    host_->uni_vmovups(vmm, host_->ptr[host_->rsp]);
    host_->add(host_->rsp, injector_utils::vmm_size_t<Vmm>::bytes);
}

template <cpu_isa_t isa>
std::pair<bool, int> jit_uni_binary_injector_t<isa>::should_preserve_vmm(
        int curr_idx, int vmm_hint, int max_vmm_idx,
        bool dt_helper_vmm_needed) const {
    if (dt_helper_vmm_needed && vmm_hint == curr_idx) {
        if (curr_idx == 0)
            return std::make_pair(true, max_vmm_idx);
        else
            return std::make_pair(true, 0);
    }
    return std::make_pair(false, vmm_hint);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    static constexpr int max_vmm_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const bool dt_helper_vmm_needed = rhs_arg_data_type != data_type::f32
            || scalar_f32_non_avx512(rhs_arg_data_type);

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t register_guard {host_,
            (rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak::Reg64>(
                                    {rhs_arg_static_params_.rhs_addr_reg,
                                            rhs_arg_static_params_
                                                    .rhs_helper_reg})
                            : std::initializer_list<Xbyak::Reg64>()),
            (rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                            ? std::initializer_list<Xbyak::Xmm>({Vmm(vmm_hint)})
                            : std::initializer_list<Xbyak::Xmm>())};

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, use_per_oc_spatial_strategy_);
    bool vmm0_was_preserved = false;
    static const Vmm zero_vmm(0);
    Xbyak::Address rhs_arg_addr(0);

    // Phase 3 Apply binary post-op over all vmms.
    for (const auto vmm_idx : vmm_idxs) {
        if (vmm_idx == start_idx
                || rhs_arg_params_differ(vmm_idx, vmm_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy);
        }

        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const Vmm dst_vmm(vmm_idx);

        if (vmm_preservation_needed) {
            const Vmm vmm_to_preserve(local_vmm_preservation.second);
            push_vmm(vmm_to_preserve);
            inject_binary(post_op, dst_vmm, rhs_arg_addr);
            pop_vmm(vmm_to_preserve);
            // in case all Vmm are occupied, Vmm(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_vmm(zero_vmm);
            vmm0_was_preserved = true;
        } else
            inject_binary(post_op, dst_vmm, rhs_arg_addr);
    }
    // ...and restored afterwards
    if (vmm0_was_preserved) pop_vmm(zero_vmm);
}

template <cpu_isa_t isa>
Xbyak::Address jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);

    host_->mov(rhs_addr_reg, host_->ptr[param1_ + abi_param_offset]);
    host_->mov(rhs_addr_reg,
            host_->ptr[rhs_addr_reg + rhs_arg_idx * rhs_arg_ptr_size]);

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar: return host_->ptr_b[rhs_addr_reg];
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_out_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_out_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_out_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return host_->ptr[rhs_addr_reg];
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? host_->ptr_b[rhs_addr_reg]
                    : host_->ptr[rhs_addr_reg];
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return host_->ptr[rhs_addr_reg];
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, Xbyak::Operand> &vmm_idx_to_elem_operand_off,
        int vmm_idx, const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, it_operand_off->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(tmp_reg, it_operand_off->second);
            host_->sal(tmp_reg, shift_val);
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, Xbyak::Address> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const Xbyak::Reg64 &addr_reg, const Xbyak::Reg64 &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, it_off_addr->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(tmp_reg, it_off_addr->second);
            host_->sal(tmp_reg, shift_val);
            host_->add(addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, int> &vmm_idx_to_elem_val_off, int vmm_idx,
        const Xbyak::Reg64 &addr_reg, std::size_t elem_size_bytes) const {

    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end())
        host_->add(addr_reg, it_off_val->second * elem_size_bytes);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const Xbyak::Address &rhs_addr) const {
    const auto &alg = post_op.binary.alg;
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;

    if (rhs_arg_data_type != data_type::f32
            || scalar_f32_non_avx512(rhs_arg_data_type)) {
        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (rhs_addr.isBroadcast())
            execute_broadcast(
                    rhs_arg_data_type, tmp_vmm, remove_bcast_bit(rhs_addr));
        else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr);

        if (rhs_arg_data_type != data_type::bf16
                && rhs_arg_data_type != data_type::f32)
            cvt_to_f32(tmp_vmm);

        execute_binary(alg, dst, tmp_vmm);
    } else
        execute_binary(alg, dst, rhs_addr);
}

template <cpu_isa_t isa>
bool jit_uni_binary_injector_t<isa>::scalar_f32_non_avx512(
        const data_type_t &data_type) const {
    return data_type == data_type::f32; // && data_type::f32;
}

template <>
bool jit_uni_binary_injector_t<avx512_common>::scalar_f32_non_avx512(
        const data_type_t &data_type) const {
    return false;
}

template <>
bool jit_uni_binary_injector_t<avx512_core>::scalar_f32_non_avx512(
        const data_type_t &data_type) const {
    return false;
}

template <>
bool jit_uni_binary_injector_t<avx512_core_bf16>::scalar_f32_non_avx512(
        const data_type_t &data_type) const {
    return false;
}

template <cpu_isa_t isa>
Xbyak::Address jit_uni_binary_injector_t<isa>::remove_bcast_bit(
        const Xbyak::Address &rhs_addr) const {
    return Xbyak::Address(rhs_addr.getBit(), false, rhs_addr.getRegExp());
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::cvt_to_f32(const Vmm &tmp_vmm) const {
    host_->vcvtdq2ps(tmp_vmm, tmp_vmm);
}

template <>
void jit_uni_binary_injector_t<sse41>::cvt_to_f32(const Vmm &tmp_vmm) const {
    static_assert(
            std::is_same<Vmm, Xbyak::Xmm>::value, "Vmm type should match Xmm");
    host_->cvtdq2ps(tmp_vmm, tmp_vmm);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32: host_->uni_vbroadcastss(tmp_vmm, rhs_addr); break;
        case data_type::s32: host_->uni_vpbroadcastd(tmp_vmm, rhs_addr); break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8(data_type, tmp_vmm, rhs_addr);
            break;
        case data_type::bf16:
            if (std::is_same<Vmm, Xbyak::Zmm>::value) {
                host_->vpbroadcastw(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_s8u8(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {
    const Xbyak::Xmm xmm(tmp_vmm.getIdx());
    switch (data_type) {
        case data_type::s8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovsxbd(tmp_vmm, xmm);
            break;
        case data_type::u8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovzxbd(tmp_vmm, xmm);
            break;
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<avx>::execute_broadcast_s8u8(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak::Reg8 tmp_reg8 = Xbyak::Reg8(tmp_reg64_idx);
        const Xbyak::Reg32 tmp_reg32 = Xbyak::Reg32(tmp_reg64_idx);
        const auto tmp_xmm = Xbyak::Xmm(tmp_vmm.getIdx());
        host_->mov(tmp_reg8, rhs_addr);
        host_->vmovd(tmp_xmm, tmp_reg32);
        host_->vpunpcklbw(tmp_xmm, tmp_xmm, tmp_xmm);
        host_->vpshuflw(tmp_xmm, tmp_xmm, 0);
        if (data_type == data_type::s8)
            host_->vpmovsxbd(tmp_xmm, tmp_xmm);
        else
            host_->vpmovzxbd(tmp_xmm, tmp_xmm);

        host_->vinsertf128(tmp_vmm, tmp_vmm, tmp_xmm, 1);
    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<sse41>::execute_broadcast_s8u8(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const Xbyak::Address &rhs_addr) const {

    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak::Reg8 tmp_reg8 = Xbyak::Reg8(tmp_reg64_idx);
        host_->mov(tmp_reg8, rhs_addr);
        const Xbyak::Reg32 tmp_reg32 = Xbyak::Reg32(tmp_reg64_idx);
        host_->movd(tmp_vmm, tmp_reg32);
        host_->punpcklbw(tmp_vmm, tmp_vmm);
        host_->pshuflw(tmp_vmm, tmp_vmm, 0);
        if (data_type == data_type::s8)
            host_->pmovsxbd(tmp_vmm, tmp_vmm);
        else
            host_->pmovzxbd(tmp_vmm, tmp_vmm);
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const data_type_t &data_type,
        const Vmm &tmp_vmm, const Xbyak::Address &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32: host_->uni_vmovups(tmp_vmm, rhs_addr); break;
        case data_type::s8: host_->uni_vpmovsxbd(tmp_vmm, rhs_addr); break;
        case data_type::u8: host_->uni_vpmovzxbd(tmp_vmm, rhs_addr); break;
        case data_type::bf16:
            if (std::is_same<Vmm, Xbyak::Zmm>::value) {
                host_->vpmovzxwd(tmp_vmm, rhs_addr);
                host_->vpslld(tmp_vmm, tmp_vmm, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
template <typename T>
void jit_uni_binary_injector_t<isa>::execute_binary(
        alg_kind_t binary_alg, const Vmm &dst, const T &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add: host_->uni_vaddps(dst, dst, rhs); break;
        case alg_kind::binary_mul: host_->uni_vmulps(dst, dst, rhs); break;
        case alg_kind::binary_max: host_->uni_vmaxps(dst, dst, rhs); break;
        case alg_kind::binary_min: host_->uni_vminps(dst, dst, rhs); break;
        case alg_kind::binary_div: host_->uni_vdivps(dst, dst, rhs); break;
        default: assert(!"unsupported algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<avx512_core_bf16>;
template class jit_uni_binary_injector_t<avx512_core>;
template class jit_uni_binary_injector_t<avx512_common>;
template class jit_uni_binary_injector_t<avx2>;
template class jit_uni_binary_injector_t<avx>;
template class jit_uni_binary_injector_t<sse41>;

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
