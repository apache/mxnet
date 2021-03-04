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

#ifndef GPU_GPU_BATCH_NORMALIZATION_PD_HPP
#define GPU_GPU_BATCH_NORMALIZATION_PD_HPP

#include <assert.h>

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
template <typename pd_t>
inline void gpu_init_default_ws(pd_t *self, memory_desc_t &ws_md) {
    auto mdw = memory_desc_wrapper(self->src_md(0));
    ws_md = *mdw.md_;
    ws_md.data_type = data_type::s32;
}
} // namespace

struct gpu_batch_normalization_fwd_pd_t : public batch_normalization_fwd_pd_t {
    using batch_normalization_fwd_pd_t::batch_normalization_fwd_pd_t;

protected:
    void init_default_ws(size_t bits_per_element) override {
        UNUSED(bits_per_element);
        gpu_init_default_ws(this, ws_md_);
    }
};

struct gpu_batch_normalization_bwd_pd_t : public batch_normalization_bwd_pd_t {
    using batch_normalization_bwd_pd_t::batch_normalization_bwd_pd_t;

    void init_default_ws(size_t bits_per_element) override {
        UNUSED(bits_per_element);
        gpu_init_default_ws(this, ws_md_);
    }
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
