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

#ifndef CPU_GEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_GEMM_INNER_PRODUCT_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace inner_product_utils {

template <data_type_t acc_type, data_type_t dst_type>
struct pp_kernel_t {
    static pp_kernel_t *create(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum);
    static pp_kernel_t *create(
            const cpu_inner_product_fwd_pd_t *pd, bool skip_sum) {
        return create(pd->OC(), pd->MB(), pd->OC(), pd->attr(),
                pd->desc()->bias_desc.data_type, skip_sum);
    }

    virtual ~pp_kernel_t() = default;

    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    // mb kernel only supports single-threaded execution where performance
    // degradation is larger
    bool sequential_kernel() const { return mb_blk_kernel_; }

    virtual void operator()(dst_data_t *dst, const acc_data_t *acc,
            const char *bias, const float *scales, size_t start, size_t end,
            size_t runtime_oc, dim_t dst_mb_stride,
            const float *dst_zero_points) const = 0;

    virtual status_t create_kernel() { return status::success; }

protected:
    pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum);

    size_t OC_;
    size_t MB_;
    dim_t dst_mb_stride_;
    data_type_t bias_data_type_;
    size_t bias_data_type_size_ = 0;
    bool do_scale_ = false;
    size_t scale_idx_mult_ = 0;
    bool do_eltwise_ = false;
    post_ops_t::entry_t::eltwise_t eltwise_;
    bool do_sum_ = false;
    bool do_dst_zero_points_ = false;
    float sum_scale_ = 0.f;
    bool mb_blk_kernel_ = false;

    bool has_trivial_mb_stride() const {
        return (!runtime_oc()) && (OC_ == (size_t)dst_mb_stride_);
    }
    bool do_bias() const { return bias_data_type_ != data_type::undef; }
    bool runtime_oc() const { return OC_ == (size_t)DNNL_RUNTIME_DIM_VAL; }
    bool runtime_mb() const { return MB_ == (size_t)DNNL_RUNTIME_DIM_VAL; }
};

} // namespace inner_product_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
