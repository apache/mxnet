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

#ifndef GPU_JIT_NGEN_TYPE_BRIDGE_HPP
#define GPU_JIT_NGEN_TYPE_BRIDGE_HPP

#include "common/c_types_map.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline ngen::DataType convert_dnnl_type_to_ngen(data_type_t dt) {
    using namespace ngen;

    DataType dt_out = DataType::invalid;

    switch (dt) {
        case data_type::f16: dt_out = DataType::hf; break;
        case data_type::f32: dt_out = DataType::f; break;
        case data_type::s32: dt_out = DataType::d; break;
        case data_type::s8: dt_out = DataType::b; break;
        case data_type::u8: dt_out = DataType::ub; break;
        default: assert(!"Unknown datatype");
    }

    return dt_out;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
