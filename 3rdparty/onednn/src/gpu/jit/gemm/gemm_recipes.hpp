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

#ifndef GPU_JIT_GEMM_RECIPES_HPP
#define GPU_JIT_GEMM_RECIPES_HPP

#include "gpu/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace {

struct gemm_recipe_t {
    ngen::HW hw;
    char precisions[4];
    char layouts[4];
    struct {
        int a = 1, b = 1;
    } crosspacks;
    int unrollM, unrollN;
    const char *strategyString;
};

// clang-format off
const gemm_recipe_t gemm_recipes[] = {
    {ngen::HW::Gen9, "SSS", "NNN", {}, 8,  4,  "ab8x2 ab16x2 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 8,  8,  "ab8x2 ab16x2 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 16, 8,  "ab4x2 ab16x2 ab acb nmk"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 16, 16, "ab8 ab16 ab acb"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 8,  "ab4x2 ab16x2 ab acb nmk"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 12, "ab1x2 ab16 ab acb"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 16, "ab1x2 ab16 ab acb"},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 64, 8,  "ab1x2 ab16 ab acb nmk"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 8,  4,  "as8x2 ab16x2 as cb1 wg 8x1 acb nmk"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 8,  8,  "ab2x2 as8x2 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 8,  "ab4x2 ab4x2 ab cb1 wg 8x1 acb nmk"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 16, "ab4x2 ab16 ab acb"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 32, "ab1x2 ab8 ab acb"},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 32, 16, "ab4 ab4x2 ab acb"},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 8,  4,  "as8x2 ab32 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 8,  8,  "as8x2 ab32 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 8,  "ab16 as4x2 as cb1 wg 8x1 acb nmk"},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 16, "as1x2 ab16 ab ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "SSS", "TTN", {}, 16, 32, "as4 ab4 as k8 da cs"},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 8,  8,  "ab16x2 as16x2 ab l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 16, 16, "ab8x2 ab32x2 ab l4 acb"},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 32, 16, "ab8x2 ab32/8 ab l4 acb nmk"},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 32, 32, "ab4x2 as8 ab k16 l4 acb"},
    {ngen::HW::Gen9, "HHH", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 l4 acb"},
    {ngen::HW::Gen9, "HHH", "NTN", {}, 32, 32, "ab2x2 ab2x2 ab k4 l4 acb"},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 8,  8,  "as32x2 as16x2 ab l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 16, 16, "as2x2 ab32 ab l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 32, 16, "as8 ab16 ab l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 32, 32, "as2 as8 ab l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "HHH", "TTN", {}, 32, 16, "as8 ab4 ab k8 ra8 l4 cs"},
    {ngen::HW::Gen9, "HHH", "TTN", {}, 32, 32, "as8 ab2 ab k8 ra8 l4 cs"},
    {ngen::HW::Gen9, "OOI", "NNN", {}, 32, 16, "ab4/2x2 as2x2 as l4 cb1 wg 8x1 acb nmk"},
    {ngen::HW::Gen9, "OOI", "NTN", {}, 32, 16, "ab2 ab1x2 as l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen9, "OOI", "TNN", {}, 16, 16, "as8 as8 as l4 cab1 k32 wg 2x4 acb"},
    {ngen::HW::Gen9, "OOI", "TTN", {}, 16, 32, "as2x2 ab8/2x2 as l4 ca1 wg 1x8 acb"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 8,  4,  "ab16 ab32/16x2 ab ca1 wg 2x8 int"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 8,  8,  "ab32 ab32 ab ca1 wg 2x8 vnc"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 16, 8,  "ab2 ab32 ab ca1 wg 2x8 int"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 8,  "ab2 ab32 ab ca1 wg 2x8 int"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 12, "ab4x2 ab16/8 ab k32 int"},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 16, "ab4 ab8 ab cb1 wg 8x2 int nmk"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 8,  4,  "ab4 ab16 ab cab1 wg 4x4 int"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 8,  8,  "ab4 ab16 ab cab1 wg 4x4 vnc"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 8,  "ab4x2 ab8 ab cb1 wg 8x2 int nmk"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 16, "ab4 ab4x2 ab vnc nmk"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 32, "ab4x2 ab2x2 ab k8 int ns64"},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 int ns64"},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 8,  4,  "ab16 ab32 ab ca1 wg 2x8 int"},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 8,  8,  "ab32 ab32 ab ca1 wg 2x8 vnc"},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 16, 8,  "ab16 ab32/16 ab ca1 wg 2x8 int"},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 16, 16, "ab8 ab8 ab k16 cab1 wg 4x4 vnc"},
    {ngen::HW::Gen12LP, "SSS", "TTN", {}, 12, 32, "ab16/8 ab4x2 as k32 int"},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32, 16, "ab4x2 ab32/8 ab k64 l4 int"},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32, 32, "ab2x2 as8x2 ab k16 l4 vnc"},
    {ngen::HW::Gen12LP, "HHH", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 l4 int"},
    {ngen::HW::Gen12LP, "HHH", "NTN", {}, 32, 32, "ab2x2 ab2x2 ab k4 l4 vnc"},
    {ngen::HW::Gen12LP, "HHH", "TNN", {}, 32, 16, "ab4 ab4 ab k8 vnc cab1 wg 4x4"},
    {ngen::HW::Gen12LP, "HHH", "TNN", {}, 32, 32, "as4 as8 ab k8 ra4 l4 vnc"},
    {ngen::HW::Gen12LP, "HHH", "TTN", {}, 32, 16, "as8 ab4x2 ab k16 ra8 l4 int"},
    {ngen::HW::Gen12LP, "HHH", "TTN", {}, 32, 32, "as8 ab2x2 ab k16 ra8 l4 vnc"},
    {ngen::HW::Gen12LP, "OOI", "NNN", {}, 32, 16, "sb4 sb8 sb l4 int k32 cab1 wg 4x4"},
    {ngen::HW::Gen12LP, "OOI", "NTN", {}, 16, 32, "sb8 sb4 sb l4 int k16 cab1 wg 4x4"},
    {ngen::HW::Gen12LP, "OOI", "TNN", {}, 16, 16, "sb8x2 sb8x2 sb l4 vnc k32 cab1 wg 4x4"},
    {ngen::HW::Gen12LP, "OOI", "TTN", {}, 16, 32, "sb8 sb4 sb l4 int k32 cab1 wg 4x4 fn nmk"},
};
// clang-format on

const int gemm_recipe_count = sizeof(gemm_recipes) / sizeof(gemm_recipes[0]);

} // anonymous namespace
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
