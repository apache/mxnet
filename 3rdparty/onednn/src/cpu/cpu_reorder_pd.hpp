/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_CPU_REORDER_PD_HPP
#define CPU_CPU_REORDER_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/reorder_pd.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_reorder_pd_t : public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

    status_t init(
            engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
        const auto &post_ops = attr()->post_ops_;
        bool args_ok = IMPLICATION(post_ops.len() != 0,
                post_ops.len() == 1
                        && post_ops.entry_[0].kind == primitive_kind::sum);
        return args_ok ? status::success : status::unimplemented;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
