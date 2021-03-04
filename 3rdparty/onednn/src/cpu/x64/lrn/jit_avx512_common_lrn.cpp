/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/lrn/jit_avx512_common_lrn.hpp"
#include "cpu/x64/lrn/lrn_executor_factory.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr int vsize = 16;

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;

template <data_type_t d_type>
status_t jit_avx512_common_lrn_fwd_t<d_type>::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    const bool ok = true && mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && is_fwd()
            && !has_zero_dim_memory() && everyone_is(d_type, data_d.data_type())
            && data_d.ndims() == 4 && attr()->has_default_values();
    if (!ok) return unimplemented;

    const auto fmt_tag
            = data_d.matches_one_of_tag(format_tag::nhwc, format_tag::nChw16c);

    const bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size >= 1 && desc()->local_size <= 16
            && (desc()->lrn_beta == 0.75 || desc()->lrn_beta == 1.0)
            && data_d.matches_tag(fmt_tag)
            && IMPLICATION(fmt_tag == format_tag::nChw16c,
                    data_d.dims()[1] % vsize == 0 && desc()->local_size == 5);

    if (!args_ok_across) return unimplemented;

    if (desc()->prop_kind == forward_training) {
        dims_t ws_dims = {MB(), C(), H(), 2 * W()};
        dnnl_memory_desc_init_by_tag(&ws_md_, 4, ws_dims, d_type, fmt_tag);
    }

    return success;
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_fwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , lrn_executor_(lrn::lrn_executor_factory_t::create_executor<d_type,
              typename jit_avx512_common_lrn_fwd_t<d_type>::pd_t>(
              pd(), lrn::direction::forward)) {}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::~jit_avx512_common_lrn_fwd_t() = default;

template struct jit_avx512_common_lrn_fwd_t<f32>;
template struct jit_avx512_common_lrn_fwd_t<bf16>;

template <data_type_t d_type>
status_t jit_avx512_common_lrn_bwd_t<d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    const bool ok = true && mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && !is_fwd()
            && utils::everyone_is(d_type, data_d.data_type())
            && set_default_formats_common() && !has_zero_dim_memory()
            && data_d.ndims() == 4 && attr()->has_default_values();
    if (!ok) return unimplemented;

    const dims_t ws_dims = {MB(), C(), H(), 2 * W()};
    const auto fmt_tag
            = data_d.matches_one_of_tag(format_tag::nhwc, format_tag::nChw16c);
    dnnl_memory_desc_init_by_tag(&ws_md_, 4, ws_dims, d_type, fmt_tag);
    if (!compare_ws(hint_fwd_pd_)) return unimplemented;

    const bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size >= 1 && desc()->local_size <= 16
            && (desc()->lrn_beta == 0.75 || desc()->lrn_beta == 1.0)
            && data_d.matches_tag(fmt_tag)
            && IMPLICATION(fmt_tag == format_tag::nChw16c,
                    data_d.dims()[1] % vsize == 0 && desc()->local_size == 5);

    return args_ok_across ? success : unimplemented;
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::jit_avx512_common_lrn_bwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , lrn_executor_(lrn::lrn_executor_factory_t::create_executor<d_type,
              typename jit_avx512_common_lrn_bwd_t<d_type>::pd_t>(
              pd(), lrn::direction::backward)) {}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::~jit_avx512_common_lrn_bwd_t() = default;

template struct jit_avx512_common_lrn_bwd_t<f32>;
template struct jit_avx512_common_lrn_bwd_t<bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
