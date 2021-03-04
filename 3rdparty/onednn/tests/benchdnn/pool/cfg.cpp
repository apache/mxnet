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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"

#include "pool/pool.hpp"

namespace pool {

/* cfgs definition
 * arrays: SRC, UNUSED, UNUSED, DST
 * params: {data_type, min, max, f_min, f_max, eps}
 */

// though integers are expected, eps is needed to cover division error
const dt_conf_t conf_entry_f32
        = {dnnl_f32, -FLT_MAX, FLT_MAX, -2048, 2048, 3e-7};
const dt_conf_t conf_entry_s32 = {dnnl_s32, INT_MIN, INT_MAX, -2048, 2048, 0.};
const dt_conf_t conf_entry_s8
        = {dnnl_s8, INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, 0.};
const dt_conf_t conf_entry_u8 = {dnnl_u8, 0, UINT8_MAX, 0, UINT8_MAX, 0.};

const float16_t flt16_max = dnnl::impl::nstl::numeric_limits<float16_t>::max();
const dt_conf_t conf_entry_f16
        = {dnnl_f16, -flt16_max, flt16_max, -32, 32, 1e-3};

#define BFLT16_MAX 3.38953138925153547590470800371487866880e+38F
/* Although integers are expected, eps is needed to cover
 * for the division error */
const dt_conf_t conf_entry_bf16
        = {dnnl_bf16, -BFLT16_MAX, BFLT16_MAX, -32, 32, 5e-2};
#undef BFLT16_MAX

// Configurations with same SRC and DST datatypes
const _dt_conf_t conf_f32 = {conf_entry_f32, {}, {}, conf_entry_f32};
const _dt_conf_t conf_s32 = {conf_entry_s32, {}, {}, conf_entry_s32};
const _dt_conf_t conf_f16 = {conf_entry_f16, {}, {}, conf_entry_f16};
const _dt_conf_t conf_bf16 = {conf_entry_bf16, {}, {}, conf_entry_bf16};
const _dt_conf_t conf_s8 = {conf_entry_s8, {}, {}, conf_entry_s8};
const _dt_conf_t conf_u8 = {conf_entry_u8, {}, {}, conf_entry_u8};

// Configurations with different SRC and DST datatypes
const _dt_conf_t conf_s8f32 {conf_entry_s8, {}, {}, conf_entry_f32};
const _dt_conf_t conf_f32s8 {conf_entry_f32, {}, {}, conf_entry_s8};
const _dt_conf_t conf_u8f32 {conf_entry_u8, {}, {}, conf_entry_f32};
const _dt_conf_t conf_f32u8 {conf_entry_f32, {}, {}, conf_entry_u8};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg) \
    if (!strcasecmp(STRINGIFY(cfg), str)) return CONCAT2(conf_, cfg)
    CASE(f32);
    CASE(s32);
    CASE(f16);
    CASE(bf16);
    CASE(s8);
    CASE(u8);
    CASE(s8f32);
    CASE(f32s8);
    CASE(u8f32);
    CASE(f32u8);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return (const dt_conf_t *)1;
}

std::ostream &operator<<(std::ostream &s, const dt_conf_t *cfg) {
#define CASE(_cfg) \
    if (cfg == CONCAT2(conf_, _cfg)) return s << STRINGIFY(_cfg)
    CASE(f32);
    CASE(s32);
    CASE(f16);
    CASE(bf16);
    CASE(s8);
    CASE(u8);
    CASE(s8f32);
    CASE(f32s8);
    CASE(u8f32);
    CASE(f32u8);
#undef CASE
    SAFE_V(FAIL);
    return s;
}

} // namespace pool
