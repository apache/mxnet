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

#ifndef CPU_X64_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_X64_JIT_UNI_BINARY_INJECTOR_HPP

#include <array>
#include <cassert>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace binary_injector {

/*
 * Extracts pointers to tensors passed by user as binary postops rhs (right-hand-side)
 * arguments (arg1 from binary postop) from execution context. Those pointers are placed
 * in vector in order of binary post-op appearance inside post_ops_t structure. Returned vector
 * usually is passed to kernel during execution phase in runtime params.
 */
std::vector<const void *> prepare_binary_args(
        const post_ops_t &post_ops, const dnnl::impl::exec_ctx_t &ctx);
bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops);

enum class broadcasting_strategy_t {
    // [n, c, d, h, w]
    scalar, // [1, 1, 1, 1, 1]
    per_oc, // [1, c, 1, 1, 1]
    per_oc_spatial, // [1, c, 1, 1, 1] specific case for binary kernel nchw format
    no_broadcast, // [n, c, d, h, w]
    unsupported
};

bool binary_args_broadcast_supported(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);

bool binary_args_tail_supported(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d, int vlen);

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> predicate);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param rhs_dt_helper_vmm_idx - index of vmm helper used when loading data for
 * calculations. Treated as hint from user. If inside compute_vector_range hint
 * turns out to be invalid, it will be overwriten by register preserving logic inside
 * binary injector.
 * @param rhs_addr_reg - gpr register, used as the currently processed address of
 * rhs tensor slice. Data of rhs(arg1) for the binary operation is loaded from address
 * stored inside rhs_addr_reg.
 * @param rhs_helper_reg - gpr register used as helper for calculations during data
 * loading phase.
 * @param preserve_gpr_helpers - determines whether gpr registers specified above
 * should be preserved (pushed to stack and poped back afterwords) between
 * compute_vector_range calls.
 * @param preserve_vmm_helper - determines whether vmm helper register specified
 * above should be preserved between compute_vector_range calls.
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations)
 */
struct rhs_arg_static_params_t {
    mutable std::size_t rhs_dt_helper_vmm_idx;
    Xbyak::Reg64 rhs_addr_reg;
    Xbyak::Reg64 rhs_helper_reg;
    bool preserve_gpr_helpers;
    bool preserve_vmm_helper;
    std::size_t abi_param_offset;
    memory_desc_wrapper dst_d;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator.
 * @param use_per_oc_spatial_strategy - flag for enabling broadcast strategy
 * "per_oc_spatial_strategy". That strategy is used only in binary kernel with nchw
 * format.
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak::Reg64 &param1,
            bool use_per_oc_spatial_strategy,
            const rhs_arg_static_params_t &rhs_arg_static_params)
        : param1(param1)
        , use_per_oc_spatial_strategy(use_per_oc_spatial_strategy)
        , rhs_arg_static_params(rhs_arg_static_params) {}

    static_params_t(const Xbyak::Reg64 &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params)
        : static_params_t(param1, true, rhs_arg_static_params) {}

    Xbyak::Reg64 param1;
    bool use_per_oc_spatial_strategy;
    rhs_arg_static_params_t rhs_arg_static_params;
};

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call.
 * Contains configurable std::maps where key is vmm index and value is
 * offset in elements. The offset value identifies tensor slice in particular
 * vmm. This is utilized by broadcasting mechanism. Offset, depending on the
 * implementation particular kernels, can be passed as value (usually during
 * unrolling), inside operand, under memory address.
 *
 * @param vmm_idx_to_out_elem_off_addr - vmm mapped to offset in elements stored under
 * memory address intended to use in no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_addr - vmm mapped to offset in elements passed as raw
 * value intended to use in no_broadcast strategy
 * @param vmm_idx_to_out_elem_off_addr - vmm mapped to offset in elements inside operand
 * intended to use in no_broadcast strategy
 * @param vmm_idx_to_oc_elem_off_addr - vmm mapped to output channel offset in elements
 * stored under memory address intended to use in per_oc broadcast strategies.
 * @param vmm_idx_to_oc_elem_off_val - vmm mapped to  output channel offset in elements
 * passed as raw value intended to use in per_oc broadcast strategies.
 * @param vmm_idx_to_oc_off_oprnd - vmm mapped to output channel offset in elements inside
 * operand intended to use in per_oc broadcast strategies.
 */

struct rhs_arg_dynamic_params_t {
    std::map<int, Xbyak::Address> vmm_idx_to_out_elem_off_addr;
    std::map<int, int> vmm_idx_to_out_elem_off_val;
    std::map<int, Xbyak::Operand> vmm_idx_to_out_off_oprnd;

    std::map<int, Xbyak::Address> vmm_idx_to_oc_elem_off_addr;
    std::map<int, int> vmm_idx_to_oc_elem_off_val;
    std::map<int, Xbyak::Operand> vmm_idx_to_oc_off_oprnd;
};

/*
 * Main mechanism responsbile for injecting binary postops supporting various
 * isa: sse41, avx, avx2, avx512 with core, bf16 extensions as well as data
 * types: f32, bf16, s32, u8, s8.
 */
template <cpu_isa_t isa>
class jit_uni_binary_injector_t {
public:
    jit_uni_binary_injector_t(
            jit_generator *host, const static_params_t &static_params);

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. Function loads appropriate
     * slice of rhs tensor for computations based on internally determined
     * broadcast strategy and information about stored data in particular vmm
     * described inside rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes. Function loads
     * appropriate slice of rhs tensor for computations based on internally
     * determined broadcast strategy and information about stored data in particular
     * vmm described inside rhs_arg_params.
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular vmm described inside rhs_arg_params.
     */
    void compute_vector(size_t idx, std::size_t rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

private:
    /*
     * Determines if hint passed by user is valid (is inside range
     * <start_idx, end_idx>). If not it returns new vmm idx value that will be
     * used as temporary vmm in future computations.
     */
    int adjust_temp_vmm_hint(
            int user_hint, int start_idx, int end_idx, int max_vmm_idx) const;
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Vmm(vmm_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns
     * ptr to it.
     */
    Xbyak::Address prepare_rhs_arg_addr(std::size_t vmm_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, Vmm dst,
            const Xbyak::Address &rhs_addr) const;
    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_offset_from_operand(
            const std::map<int, Xbyak::Operand> &vmm_idx_to_elem_addr_off,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes) const;
    void append_offset_under_mem_addr(
            const std::map<int, Xbyak::Address> &vmm_idx_to_elem_addr_off,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            const Xbyak::Reg64 &tmp_reg, std::size_t elem_size_bytes) const;
    void append_value_offset(const std::map<int, int> &vmm_idx_to_elem_val_off,
            int vmm_idx, const Xbyak::Reg64 &addr_reg,
            std::size_t elem_size_bytes) const;

    template <typename T>
    void execute_binary(
            alg_kind_t binary_alg, const Vmm &dst, const T &rhs) const;
    /*
     * Used in scalar broadcast strategy, broadcasting single value of given
     * data type over entire vector Vmm register.
     */
    void execute_broadcast(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    void execute_broadcast_s8u8(const data_type_t &data_type,
            const Vmm &tmp_reg, const Xbyak::Address &rhs_addr) const;
    void load_rhs(const data_type_t &data_type, const Vmm &tmp_reg,
            const Xbyak::Address &rhs_addr) const;
    bool scalar_f32_non_avx512(const data_type_t &data_type) const;
    void cvt_to_f32(const Vmm &tmp_reg) const;
    void push_vmm(const Vmm &vmm) const;
    void pop_vmm(const Vmm &vmm) const;
    /*
     * Returns pair consisting of flag indication preservation is needed for vmm
     * index in second member that should be used as temporary vmm inside inject
     * binary.
     */
    std::pair<bool, int> should_preserve_vmm(int curr_idx, int vmm_hint,
            int max_vmm_idx, bool dt_helper_vmm_needed) const;
    /*
     * Used in isa != avx512 where m32bcst is not supported, replaces ptr_b
     * with ptr.
     */
    Xbyak::Address remove_bcast_bit(const Xbyak::Address &rhs_addr) const;

    jit_generator *host_;
    const rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak::Reg64 param1_;
    const bool use_per_oc_spatial_strategy_;
};

} // namespace binary_injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
