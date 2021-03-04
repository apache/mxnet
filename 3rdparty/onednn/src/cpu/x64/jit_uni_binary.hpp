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

#ifndef CPU_X64_JIT_UNI_BINARY_HPP
#define CPU_X64_JIT_UNI_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct binary_kernel_t;

template <data_type_t src_type>
struct jit_uni_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;
            memory_desc_wrapper dst_md_(dst_md());
            memory_desc_wrapper src0_md_(src_md(0));
            memory_desc_wrapper src1_md_(src_md(1));
            const auto &po = attr()->post_ops_;
            const int elt_idx = po.find(primitive_kind::eltwise);

            const bool ok = IMPLICATION(src_type == bf16, mayiuse(avx512_core))
                    && utils::everyone_is(src_type, src_md(0)->data_type,
                            src_md(1)->data_type)
                    && set_default_params() == status::success
                    && !has_zero_dim_memory() && src0_md_ == dst_md_
                    && is_applicable()
                    && attr()->has_default_values(sm::post_ops)
                    && post_ops_ok(attr(), src_md(0))
                    && (elt_idx == -1
                            || IMPLICATION(!dst_md_.is_dense(),
                                    cpu_eltwise_fwd_pd_t::
                                            eltwise_preserves_zero(
                                                    po.entry_[elt_idx]
                                                            .eltwise)))
                    && IMPLICATION(!mayiuse(avx2),
                            src0_md_.consistent_with(src1_md_)
                                    || src0_md_.is_plain());

            if (!ok) return status::unimplemented;

            return status::success;
        }

    private:
        // alg_preserves_zero returns true if operation preserves zero in case
        // of both inputs contain zero.
        bool alg_preserves_zero() const {
            using namespace utils;
            using namespace alg_kind;
            return utils::one_of(desc()->alg_kind, binary_add, binary_max,
                    binary_min, binary_mul);
        }

        bool is_applicable() {
            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            // check density first to avoid same non-dense src0 and src1 to pass
            // the next check
            bool ok = src0_d.is_dense(true) && src1_d.is_dense(true)
                    && dst_d.is_dense(true);
            if (!ok) return false;

            const bool has_padding = utils::one_of(true,
                    src0_d.nelems(true) != src0_d.nelems(false),
                    src1_d.nelems(true) != src1_d.nelems(false),
                    dst_d.nelems(true) != dst_d.nelems(false));
            ok = IMPLICATION(has_padding, alg_preserves_zero());
            if (!ok) return false;

            // full tensor operation
            if (src0_d == src1_d) return true;

            // broadcast operation
            const auto ndims = src0_d.ndims();
            ok = ndims >= 2;
            // supported case: NxCxDxHxW:{NxCx1x1x1,1xCx1x1x1,1x1x1x1x1}
            const auto &bcast_dims = broadcast_dims();
            ok = ok && IMPLICATION(bcast_dims[0] == 0, bcast_dims[1] == 0);
            for (int d = 2; d < ndims; ++d)
                ok = ok && bcast_dims[d] == 1;
            if (!ok) return false;

            if (src0_d.is_plain() && src1_d.is_plain()) {
                const auto &bd0 = src0_d.blocking_desc();
                const auto &bd1 = src1_d.blocking_desc();
                return bd0.strides[0] >= bd0.strides[1]
                        && IMPLICATION(bd0.strides[1] > 1,
                                bd0.strides[1] >= bd0.strides[2])
                        && bd1.strides[0] >= bd1.strides[1];
            }

            const bool point_bcast = src1_d.nelems() == 1;
            // check blocking_desc consistency
            const auto valid_bd = [&](const memory_desc_wrapper &mdw) {
                int blksize = 8;
                if (mayiuse(avx512_core)) blksize = 16;
                const auto &bd = mdw.blocking_desc();
                bool point_bcast_is_ok = true;
                // this is zero pad guard; the problem appears for blocked
                // formats when last C block has tails and `add` operation does
                // not preserve zero in padded area. To be considered when
                // implementing binary injector.
                if (point_bcast)
                    point_bcast_is_ok = src0_d.dims()[1] % blksize == 0;

                return bd.inner_nblks == 1 && bd.inner_blks[0] == blksize
                        && bd.inner_idxs[0] == 1 && point_bcast_is_ok;
            };

            return valid_bd(src0_d) && valid_bd(src1_d);
        }
    };

    jit_uni_binary_t(const pd_t *apd);
    ~jit_uni_binary_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    static bool post_ops_ok(
            const primitive_attr_t *attr, const memory_desc_wrapper &d);

    std::unique_ptr<binary_kernel_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
