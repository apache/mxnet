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

#ifndef CPU_X64_JIT_UNI_I8I8_BINARY_HPP
#define CPU_X64_JIT_UNI_I8I8_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct i8i8_binary_kernel_t;

template <data_type_t src0_type, data_type_t src1_type>
struct jit_uni_i8i8_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni:i8i8", jit_uni_i8i8_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const bool ok = src_md(0)->data_type == src0_type
                    && src_md(1)->data_type == src1_type
                    && dst_md(0)->data_type == src0_type
                    && set_default_params()
                            == status::success /* should precede comparison */
                    && !has_zero_dim_memory() && is_applicable()
                    && memory_desc_wrapper(src_md(0))
                            == memory_desc_wrapper(dst_md(0))
                    && attr()->has_default_values(sm::post_ops | sm::scales)
                    && post_ops_ok(attr(), src_md(0))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());
            if (!ok) return status::unimplemented;

            return status::success;
        };

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }

        bool is_applicable() {
            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            // full tensor operation
            if (src0_d.similar_to(src1_d, true, false, 0)) return true;

            // broadcast operation
            const auto ndims = src0_d.ndims();
            bool ok = ndims >= 2;
            // only supported case for now is NxCxDxHxW:{N,1}xCx1x1x1
            const auto &bcast_dims = broadcast_dims();
            ok = ok && IMPLICATION(bcast_dims[0] == 0, bcast_dims[1] == 0);
            for (int d = 2; d < ndims; ++d)
                ok = ok && bcast_dims[d] == 1;
            if (!ok) return false;

            const auto &bd = src0_d.blocking_desc();
            return bd.strides[1] == 1 && bd.inner_nblks == 0;
        }
    };

    jit_uni_i8i8_binary_t(const pd_t *apd);
    ~jit_uni_i8i8_binary_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    static bool post_ops_ok(
            const primitive_attr_t *attr, const memory_desc_wrapper &d);
    std::unique_ptr<i8i8_binary_kernel_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
