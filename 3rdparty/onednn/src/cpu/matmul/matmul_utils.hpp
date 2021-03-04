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

#ifndef CPU_MATMUL_UTILS_HPP
#define CPU_MATMUL_UTILS_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace matmul {

struct matmul_helper_t {
    using mdw_t = const memory_desc_wrapper;

    matmul_helper_t(mdw_t &src_md, mdw_t &weights_md, mdw_t &dst_md)
        : src_md_(src_md), weights_md_(weights_md), dst_md_(dst_md) {}

    int ndims() const { return dst_md_.ndims(); }
    bool batched() const { return ndims() > 2; }

    dim_t batch() const {
        return utils::array_product(dst_md_.dims(), ndims() - 2);
    };
    dim_t M() const { return dst_md_.dims()[ndims() - 2]; }
    dim_t N() const { return dst_md_.dims()[ndims() - 1]; }
    dim_t K() const { return src_md_.dims()[ndims() - 1]; }

    char transA() const {
        const auto &strides = &src_md_.blocking_desc().strides[ndims() - 2];
        return (strides[1] == 1 && src_md_.dims()[ndims() - 2] > 1) ? 'N' : 'T';
    }

    char transB() const {
        const auto &strides = &weights_md_.blocking_desc().strides[ndims() - 2];
        return (strides[1] == 1 && weights_md_.dims()[ndims() - 2] > 1) ? 'N'
                                                                        : 'T';
    }

    dim_t lda() const {
        const auto &strides = &src_md_.blocking_desc().strides[ndims() - 2];
        return strides[transA() == 'N' ? 0 : 1];
    }

    dim_t ldb() const {
        const auto &strides = &weights_md_.blocking_desc().strides[ndims() - 2];
        return strides[transB() == 'N' ? 0 : 1];
    }

    dim_t ldc() const { return dst_md_.blocking_desc().strides[ndims() - 2]; }

private:
    mdw_t src_md_;
    mdw_t weights_md_;
    mdw_t dst_md_;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
