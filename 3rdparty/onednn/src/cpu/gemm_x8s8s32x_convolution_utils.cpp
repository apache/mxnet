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

#include <memory>

#include "common/math_utils.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#endif

#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

template <typename dst_data_t>
struct ref_pp_ker_t : pp_ker_t {
    ref_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp) {
        if (do_eltwise_)
            ref_eltwise_.reset(new ref_eltwise_scalar_fwd_t(eltwise_));
    }

    using acc_data_t = pp_ker_t::acc_data_t;

    void operator()(void *dst, const acc_data_t *acc, const char *bias,
            const float *scales, float nslope, float sum_scale,
            float signed_scale, int g, size_t start, size_t end) const override;

private:
    std::unique_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_;
};

template <typename dst_data_t>
void ref_pp_ker_t<dst_data_t>::operator()(void *void_dst, const acc_data_t *acc,
        const char *bias, const float *scales, float nslope, float sum_scale,
        float signed_scale, int g, size_t start, size_t end) const {
    if (end <= start) return;

    assert(data_traits<dst_data_t>::data_type == dst_data_type_);
    dst_data_t *dst = (dst_data_t *)void_dst;

    const size_t first_oc = start % OC_;
    const size_t last_oc = (end - 1) % OC_;
    const size_t first_os = start / OC_;
    const size_t last_os = (end - 1) / OC_;
    for (size_t os = first_os; os <= last_os; os++) {
        const size_t start_oc = (os == first_os) ? first_oc : 0;
        const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
        for (size_t oc = start_oc; oc <= end_oc; oc++) {
            const size_t acc_off = os * jcp_.oc + oc;
            const size_t dst_off = os * dst_os_stride_ + oc;

            float d = (float)(acc[acc_off]);
            if (jcp_.signed_input) d *= signed_scale;

            if (do_bias_)
                d += math::get_bias(bias, g * jcp_.oc + oc, bias_data_type_);

            d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
            if (do_sum_) d += sum_scale * dst[dst_off];
            if (do_eltwise_) d = ref_eltwise_->compute_scalar(d);
            dst[dst_off] = qz_a1b0<float, dst_data_t>()(d);
        }
    }
}

// Interface section

pp_ker_t::pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
    : jcp_(jcp), OC_(jcp_.oc), OS_(jcp_.os) {
    const auto dst_md = memory_desc_wrapper(pd->dst_md());

    dst_os_stride_ = dst_md.blocking_desc().strides[pd->ndims() - 1];
    dst_data_type_ = dst_md.data_type();

    scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));

    auto &post_ops = pd->attr()->post_ops_;

    do_signed_scaling_ = jcp_.signed_input;

    do_sum_ = post_ops.contain(primitive_kind::sum, 0);
    do_bias_ = pd->with_bias();
    bias_data_type_ = pd->desc()->bias_desc.data_type;

    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    do_eltwise_ = eltwise_ind != -1;
    if (do_eltwise_) eltwise_ = post_ops.entry_[eltwise_ind].eltwise;
}

pp_ker_t *pp_ker_t::create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
#if DNNL_X64
    auto *res
            = x64::gemm_x8s8s32x_convolution_utils::jit_pp_ker_create(pd, jcp);
    if (res) return res;
#endif

    switch (pd->dst_md()->data_type) {
        case data_type::f32: return new ref_pp_ker_t<float>(pd, jcp);
        case data_type::s32: return new ref_pp_ker_t<int32_t>(pd, jcp);
        case data_type::s8: return new ref_pp_ker_t<int8_t>(pd, jcp);
        case data_type::u8: return new ref_pp_ker_t<uint8_t>(pd, jcp);
        default: assert(!"unexpected data type");
    }
    return nullptr;
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
