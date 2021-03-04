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

#include <utility>

#include "dnnl_test_common.hpp"
#include "test_reorder_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using f32_f32 = std::pair<float, float>;

using tag = memory::format_tag;

using cfg_f32 = test_simple_params<f32_f32>;

using reorder_simple_test_f32_f32 = reorder_simple_test<f32_f32>;

TEST_P(reorder_simple_test_f32_f32, CPU_GPU) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "GPU engines not found.");

    engine eng_cpu(engine::kind::cpu, 0);
    engine eng_gpu(engine::kind::gpu, 0);

    Test(eng_cpu, eng_gpu);
}

TEST_P(reorder_simple_test_f32_f32, GPU_CPU) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "GPU engines not found.");

    engine eng_gpu(engine::kind::gpu, 0);
    engine eng_cpu(engine::kind::cpu, 0);

    Test(eng_gpu, eng_cpu);
}

TEST_P(reorder_simple_test_f32_f32, GPU_GPU) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "GPU engines not found.");

    engine eng_gpu1(engine::kind::gpu, 0);
    engine eng_gpu2(engine::kind::gpu, 0);

    // Reorder within one engine
    Test(eng_gpu1, eng_gpu1);

    // Different GPU engines
    Test(eng_gpu1, eng_gpu2);
}

INSTANTIATE_TEST_SUITE_P(Data, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {tag::nchw, tag::nhwc, {32, 48, 5, 4}},
                cfg_f32 {tag::oihw, tag::IOhw16i16o, {32, 48, 2, 3}},
                cfg_f32 {tag::oihw, tag::OIhw16o16i, {32, 32, 1, 1}},
                cfg_f32 {tag::hwigo, tag::gIOhw16i16o, {2, 64, 32, 1, 3}},
                cfg_f32 {tag::goihw, tag::gOIhw16o16i, {2, 32, 64, 2, 3}},
                cfg_f32 {tag::OIhw16o16i, tag::IOhw16i16o, {32, 48, 2, 3}},
                cfg_f32 {tag::gOIhw16o16i, tag::gIOhw16i16o, {2, 64, 32, 3, 2}},
                cfg_f32 {tag::oidhw, tag::OIdhw16i16o, {64, 32, 3, 9, 5}},
                cfg_f32 {tag::goidhw, tag::gOIdhw16i16o, {2, 32, 64, 4, 1, 7}},
                cfg_f32 {tag::nchw, tag::nhwc, {32, 48, 5, 4}},
                cfg_f32 {tag::nchw, tag::NChw16n16c, {64, 32, 5, 6}},
                cfg_f32 {tag::nChw16c, tag::NChw16n16c, {32, 48, 6, 9}}));

} // namespace dnnl
