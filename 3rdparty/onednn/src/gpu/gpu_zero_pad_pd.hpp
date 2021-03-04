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

#ifndef GPU_GPU_ZERO_PAD_PD_HPP
#define GPU_GPU_ZERO_PAD_PD_HPP

#include "common/primitive_desc.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_zero_pad_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::zero_pad;
    typedef gpu_zero_pad_pd_t hint_class;

    gpu_zero_pad_pd_t(const zero_pad_desc_t *adesc,
            const primitive_attr_t *attr, const hint_class *hint_fwd_pd)
        : primitive_desc_t(base_pkind), desc_(*adesc) {}

    const zero_pad_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

private:
    zero_pad_desc_t desc_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
