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

#ifndef COMMON_GEMM_UTILS_HPP
#define COMMON_GEMM_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

static inline status_t check_gemm_input(char transa, char transb, int m, int n,
        int k, int lda, int ldb, int ldc, float alpha, float beta) {
    using namespace status;
    bool consistency = true && utils::one_of(transa, 'T', 't', 'N', 'n')
            && utils::one_of(transb, 'T', 't', 'N', 'n') && m >= 0 && n >= 0
            && k >= 0;
    if (!consistency) return invalid_arguments;
    bool isTransA = utils::one_of(transa, 'T', 't');
    bool isTransB = utils::one_of(transb, 'T', 't');
    int nrowA = isTransA ? k : m;
    int nrowB = isTransB ? n : k;
    consistency = true && lda >= nstl::max(1, nrowA)
            && ldb >= nstl::max(1, nrowB) && ldc >= nstl::max(1, m);
    if (!consistency) return invalid_arguments;

    return success;
}

static inline status_t check_gemm_x8x8s32_input(char offsetc, char transa,
        char transb, int m, int n, int k, int lda, int ldb, int ldc,
        float alpha, float beta) {
    using namespace status;
    if (!utils::one_of(offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return invalid_arguments;
    return check_gemm_input(
            transa, transb, m, n, k, lda, ldb, ldc, alpha, beta);
}

static inline status_t create_gemm_memory_desc(memory_desc_t *m_desc,
        const gemm_desc_t *desc, int index, data_type_t data_type) {
    using namespace status;
    int dims[2] = {0};
    switch (index) {
        case 0:
            dims[0] = desc->m;
            dims[1] = desc->k;
            dims[desc->transa] = desc->lda;
            break;
        case 1:
            dims[0] = desc->k;
            dims[1] = desc->n;
            dims[desc->transb] = desc->ldb;
            break;
        case 2:
            dims[0] = desc->m;
            dims[1] = desc->n;
            break;
    }
    dims_t dims_flat = {dims[0] * dims[1]};
    return dnnl_memory_desc_init_by_tag(
            m_desc, 1, dims_flat, data_type, format_tag::x);
}

} // namespace impl
} // namespace dnnl

#endif
