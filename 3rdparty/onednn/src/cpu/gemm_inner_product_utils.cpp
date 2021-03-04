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

#include <memory>

#include "common/math_utils.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/jit_gemm_inner_product_utils.hpp"
#endif

#include "cpu/gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace inner_product_utils {

template <data_type_t acc_type, data_type_t dst_type>
struct ref_pp_kernel_t : public pp_kernel_t<acc_type, dst_type> {
    ref_pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum)
        : pp_kernel_t<acc_type, dst_type>(
                OC, MB, dst_mb_stride, attr, bias_dt, skip_sum) {
        if (this->do_eltwise_)
            ref_eltwise_.reset(new ref_eltwise_scalar_fwd_t(this->eltwise_.alg,
                    this->eltwise_.alpha, this->eltwise_.beta,
                    this->eltwise_.scale));
    }

    using acc_data_t = typename prec_traits<acc_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end, size_t runtime_oc,
            dim_t dst_mb_stride, const float *dst_zero_points) const override;

private:
    std::unique_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_;
};

template <data_type_t acc_type, data_type_t dst_type>
void ref_pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end, size_t runtime_oc, dim_t dst_mb_stride,
        const float *dst_zero_points) const {
    using math::get_bias;

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;
    const bool acc_is_dst = dst == (dst_data_t *)acc;

    size_t oc = start % OC;
    if (this->has_trivial_mb_stride()) {
        dst = dst + start;
        acc = acc + start;
    } else {
        const dim_t offt = (start / OC) * dst_mb_stride + oc;
        dst = dst + offt;
        // if dst and acc point to same address (inplace), then strides
        // must be similar, else assume acc buffer is dense.
        acc = acc + (acc_is_dst ? offt : start);
    }

    while (start < end) {
        float d = (float)*acc;
        if (this->do_bias()) d += get_bias(bias, oc, this->bias_data_type_);
        if (this->do_scale_) d *= scales[oc * this->scale_idx_mult_];
        if (this->do_sum_) d += this->sum_scale_ * (*dst);
        if (this->do_eltwise_) d = ref_eltwise_->compute_scalar(d);
        if (this->do_dst_zero_points_) d += dst_zero_points[0];
        *dst = qz_a1b0<float, dst_data_t>()(d);
        oc = (oc == OC - 1) ? 0 : oc + 1;
        if (oc == 0) {
            if (!this->has_trivial_mb_stride()) {
                dst = dst + dst_mb_stride - OC;
                // if dst and acc point to same address (inplace), then strides
                // must be similar, else assume acc buffer is dense.
                if (acc_is_dst) acc = acc + dst_mb_stride - OC;
            }
        }
        ++dst;
        ++acc;
        ++start;
    }
}

// Interface section

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type>::pp_kernel_t(size_t OC, size_t MB,
        dim_t dst_mb_stride, const primitive_attr_t *attr, data_type_t bias_dt,
        bool skip_sum)
    : OC_(OC)
    , MB_(MB)
    , dst_mb_stride_(dst_mb_stride)
    , bias_data_type_(bias_dt) {
    do_scale_ = !attr->output_scales_.has_default_values();
    if (do_scale_) scale_idx_mult_ = (attr->output_scales_.mask_ == (1 << 1));

    auto &p = attr->post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    do_eltwise_ = eltwise_ind != -1;
    if (do_eltwise_) eltwise_ = p.entry_[eltwise_ind].eltwise;

    const int sum_ind = p.find(primitive_kind::sum);
    do_sum_ = sum_ind != -1 && !skip_sum;
    if (do_sum_) sum_scale_ = p.entry_[sum_ind].sum.scale;

    if (do_bias())
        bias_data_type_size_ = types::data_type_size(bias_data_type_);

    if (!attr->zero_points_.has_default_values(DNNL_ARG_DST))
        do_dst_zero_points_ = true;
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type> *pp_kernel_t<acc_type, dst_type>::create(
        size_t OC, size_t MB, dim_t dst_mb_stride, const primitive_attr_t *attr,
        data_type_t bias_dt, bool skip_sum) {
#if DNNL_X64
    auto *res = x64::inner_product_utils::jit_pp_kernel_create<acc_type,
            dst_type>(OC, MB, dst_mb_stride, attr, bias_dt, skip_sum);
    if (res) return res;
#endif

    return new ref_pp_kernel_t<acc_type, dst_type>(
            OC, MB, dst_mb_stride, attr, bias_dt, skip_sum);
}

using namespace data_type;
template struct pp_kernel_t<f32, f32>;
template struct pp_kernel_t<s32, f32>;
template struct pp_kernel_t<s32, s32>;
template struct pp_kernel_t<s32, s8>;
template struct pp_kernel_t<s32, u8>;
template struct pp_kernel_t<f32, bf16>;

} // namespace inner_product_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
