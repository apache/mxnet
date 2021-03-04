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

#ifndef CPU_X64_JIT_UNI_SHUFFLE_HPP
#define CPU_X64_JIT_UNI_SHUFFLE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_shuffle_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_shuffle_base_kernel_t;

template <cpu_isa_t isa, int data_type_size>
struct jit_uni_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_shuffle_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;

            const bool set_default_formats
                    = IMPLICATION(!is_fwd(), set_default_formats_common());

            dat_tag_ = memory_desc_matches_one_of_tag(*data_md(), nChw16c);

            const data_type_t data_type = data_md()->data_type;

            // Currently supporting only group=3 (FWD, BWD) for nChw16c format
            const bool ok = mayiuse(isa)
                    && data_type_size == types::data_type_size(data_type)
                    && platform::has_data_type_support(data_type)
                    && attr()->has_default_values() && set_default_formats
                    && dat_tag_ == nChw16c && group_size() == 3 && axis() == 1;

            if (!ok) return status::unimplemented;

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    jit_uni_shuffle_t(const pd_t *apd);

    ~jit_uni_shuffle_t();

    using data_t = typename typesize_traits<data_type_size>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace format_tag;
        execute_<nChw16c>(ctx);
        return status::success;
    }

private:
    template <format_tag_t tag>
    void execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_shuffle_base_kernel_t<isa>> kernel_;
    int *input_off_;
    static constexpr int blk_size = 16;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
