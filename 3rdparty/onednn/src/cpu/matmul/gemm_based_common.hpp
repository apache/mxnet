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

#ifndef CPU_MATMUL_GEMM_BASED_COMMON_HPP
#define CPU_MATMUL_GEMM_BASED_COMMON_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {
namespace gemm_based {

struct params_t {
    // indicates if an auxiliary array for intermediate computations is not
    // required
    bool dst_is_acc_;

    // indicates if output scales from attributes are applied
    // by gemm (alpha parameter) or post-op kernel (pp_kernel_)
    bool gemm_applies_output_scales_ = false;

    // sum post-op scaling factor that is fused into gemm
    float gemm_beta_ = 0.f;

    // indicates if a special post processing kernel
    // should be invoked after gemm
    bool has_pp_kernel_ = false;

    // an attribute for post processing kernel
    primitive_attr_t pp_attr_;

    // auxiliary functions

    // returns gemm alpha parameter (a single value for now)
    float get_gemm_alpha(const float *primitive_scales) const {
        return gemm_applies_output_scales_ ? primitive_scales[0] : 1.f;
    }

    // returns scaling factors for post processing kernel
    const float *get_post_processing_scales(
            const float *primitive_scales) const {
        return gemm_applies_output_scales_ ? pp_attr_.output_scales_.scales_
                                           : primitive_scales;
    }
};

inline bool check_gemm_compatible_formats(const matmul_pd_t &pd) {

    const memory_desc_wrapper dst_d(pd.dst_md());
    const int ndims = dst_d.ndims();

    auto check_input_format = [=](const memory_desc_t *md) {
        memory_desc_wrapper mdw(md);

        if (!mdw.is_plain()) return false;

        const dims_t &strides = mdw.blocking_desc().strides;

        // for GeMM atleast one of the two innermost axes must be contiguous
        return utils::one_of(1, strides[ndims - 1], strides[ndims - 2]);
    };

    bool ok = check_input_format(pd.src_md())
            && check_input_format(pd.weights_md()) && dst_d.is_plain()
            && dst_d.blocking_desc().strides[ndims - 1] == 1;

    return ok;
}

inline void book_acc_scratchpad(
        matmul_pd_t &pd, const params_t &params, size_t sizeof_acc_data) {
    bool has_runtime_dims = false;
    for (auto d : pd.dst_md()->dims) {
        if (d == DNNL_RUNTIME_DIM_VAL) {
            has_runtime_dims = true;
            break;
        }
    }

    if (!params.dst_is_acc_ && !has_runtime_dims) {
        auto scratchpad = pd.scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                nstl::min(pd.batch(), (dim_t)dnnl_get_max_threads()) * pd.M()
                        * pd.N(),
                sizeof_acc_data);
    }
}

} // namespace gemm_based
} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
