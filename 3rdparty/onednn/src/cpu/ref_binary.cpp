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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src0_type, data_type_t src1_type, data_type_t dst_type>
status_t ref_binary_t<src0_type, src1_type, dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const src0_data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const src1_data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto alg = pd()->desc()->alg_kind;

    // 0:src0 1:src1
    constexpr int nargs = 2;
    scales_t scales[nargs];
    int args[nargs] = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};

    if (nstl::is_integral<src0_data_t>::value)
        CHECK(scales[0].copy_from(pd()->attr()->scales_.get(args[0])));

    if (nstl::is_integral<src0_data_t>::value)
        CHECK(scales[1].copy_from(pd()->attr()->scales_.get(args[1])));

    bool do_scale_src0 = !scales[0].has_default_values();
    bool do_scale_src1 = !scales[1].has_default_values();

    const auto nelems_A = src0_d.nelems();
    const auto ndims = pd()->ndims();

    parallel_nd(nelems_A, [&](dim_t i) {
        dims_t l_dims; // single decomposition for all physical offsets
        utils::l_dims_by_l_offset(l_dims, i, src0_d.dims(), ndims);
        auto off_A = src0_d.off_v(l_dims);
        auto off_C = dst_d.off_v(l_dims);

        int mask = utils::get_dims_mask(src0_d.dims(), src1_d.dims(), ndims);
        utils::apply_mask_on_dims(l_dims, ndims, mask);
        auto off_B = src1_d.off_v(l_dims);

        float x_f = (float)src0[off_A];
        float y_f = (float)src1[off_B];
        float dst_f = (float)dst[off_C];

        if (do_scale_src0) x_f *= scales[0].scales_[0];
        if (do_scale_src1) y_f *= scales[1].scales_[0];

        float acc = compute_binary_scalar(alg, x_f, y_f);

        ref_post_ops_t::args_t args;
        args.dst_val = dst_f;
        args.ctx = &ctx;
        args.l_offset = i;
        args.dst_md = pd()->dst_md();
        ref_post_ops->execute(acc, args);

        dst[off_C] = cpu::saturate_and_round<dst_data_t>(acc);
    });

    return status::success;
}

using namespace data_type;

template struct ref_binary_t<f32>;
template struct ref_binary_t<bf16>;
template struct ref_binary_t<s8, u8, s8>;
template struct ref_binary_t<s8, s8, s8>;
template struct ref_binary_t<u8, s8, u8>;
template struct ref_binary_t<u8, u8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
