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
#ifndef CPU_X64_JIT_UNI_POSTOPS_INJECTOR_HPP
#define CPU_X64_JIT_UNI_POSTOPS_INJECTOR_HPP

#include <functional>
#include <map>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include <initializer_list>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

/*
 * Allows specifying custom injector function for given post-op type - one
 * function per primitive. There are post-ops type (example: sum) that don't
 * have specialized injector. They heavily rely on kernel specific intrnals,
 * which makes the generalization unreasonable. As so user can prepare internal
 * kernel lambda and pass it explicitly to injector.
 */
using lambda_jit_injectors_t
        = std::map<dnnl_primitive_kind_t, std::function<void()>>;

/*
 * Main mechanism of handling various post-ops types. It utilizes internally
 * specialized injectors to generate post-ops code to host primitive. Random
 * order of post-ops is supported.
 */
template <cpu_isa_t isa>
class jit_uni_postops_injector_t {
public:
    /*
     * @param host <required> - user primitive where post-ops generated code is
     * injected
     * @param post_ops <required> - struct representing requested post-ops chain
     * @binary_static_params <reguired> - static params needed for binary_injector.
     * see: jit_uni_binary_injector.hpp for more info.
     * @param eltwise_static_params <optional> - allows user specify non default
     * params for eltwise_injector
     * @param lambda_jit_injectors <optional> - allows user specify custom injector
     * function for given post-op type
     */
    jit_uni_postops_injector_t(jit_generator *host, const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params);
    jit_uni_postops_injector_t(jit_generator *host, const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const lambda_jit_injectors_t &lambda_jit_injectors);
    jit_uni_postops_injector_t(jit_generator *host, const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const eltwise_injector::static_params_t &eltwise_static_params);
    jit_uni_postops_injector_t(jit_generator *host, const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const eltwise_injector::static_params_t &eltwise_static_params,
            const lambda_jit_injectors_t &lambda_jit_injectors);

    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * ordered set of vector registers' indexes.
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes.
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * a single vector register index.
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector(size_t idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params);
    /*
     * Thin wrapper for eltwise injector specific function
     */
    void prepare_table(bool gen_table = true);
    void set_lambda_injector(lambda_jit_injectors_t::key_type,
            const lambda_jit_injectors_t::mapped_type &jit_injector);

private:
    post_ops_t post_ops_;
    jit_generator *host_;
    std::map<dnnl::impl::alg_kind_t, jit_uni_eltwise_injector_f32<isa>>
            alg_to_eltwise_injector_;
    std::unique_ptr<binary_injector::jit_uni_binary_injector_t<isa>>
            binary_injector_;
    lambda_jit_injectors_t lambda_jit_injectors_;
};

enum post_op_type { sum = 0, eltwise, binary, convolution };

bool post_ops_ok(std::initializer_list<post_op_type> accepted_post_op_types,
        const size_t vlen, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d, bool sum_first_only = true);

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif