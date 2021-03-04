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

#include <map>
#include <set>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "src/cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {

using namespace impl::cpu::x64;

const std::set<cpu_isa_t> cpu_isa_all
        = {sse41, avx, avx2, avx512_mic, avx512_mic_4ops, avx512_core,
                avx512_core_vnni, avx512_core_bf16, avx512_core_bf16_amx_int8,
                avx512_core_bf16_amx_bf16, avx512_core_amx, isa_all};

struct isa_compat_info_t {
    cpu_isa_t this_isa;
    std::set<cpu_isa_t> cpu_isa_compatible;
};

// This mostly duplicates isa_traits, but the idea is to *not* rely on that
// information...
static std::map<cpu_isa, isa_compat_info_t> isa_compatibility_table = {
        {cpu_isa::sse41, {sse41, {sse41}}},
        {cpu_isa::avx, {avx, {sse41, avx}}},
        {cpu_isa::avx2, {avx2, {sse41, avx, avx2}}},
        {cpu_isa::avx512_mic, {avx512_mic, {sse41, avx, avx2, avx512_mic}}},
        {cpu_isa::avx512_mic_4ops,
                {avx512_mic_4ops,
                        {sse41, avx, avx2, avx512_mic, avx512_mic_4ops}}},
        {cpu_isa::avx512_core, {avx512_core, {sse41, avx, avx2, avx512_core}}},
        {cpu_isa::avx512_core_vnni,
                {avx512_core_vnni,
                        {sse41, avx, avx2, avx512_core, avx512_core_vnni}}},
        {cpu_isa::avx512_core_bf16,
                {avx512_core_bf16,
                        {sse41, avx, avx2, avx512_core, avx512_core_vnni,
                                avx512_core_bf16}}},
        {cpu_isa::avx512_core_amx,
                {avx512_core_amx,
                        {sse41, avx, avx2, avx512_core, avx512_core_vnni,
                                avx512_core_bf16, avx512_core_bf16_amx_int8,
                                avx512_core_bf16_amx_bf16, avx512_core_amx}}},
        {cpu_isa::all,
                {isa_all,
                        {sse41, avx, avx2, avx512_mic, avx512_mic_4ops,
                                avx512_core, avx512_core_vnni, avx512_core_bf16,
                                isa_all}}},
};

class isa_test_t : public ::testing::TestWithParam<cpu_isa> {
protected:
    void SetUp() override {
        auto isa = ::testing::TestWithParam<cpu_isa>::GetParam();

        // soft version of mayiuse that allows resetting the max_cpu_isa
        auto test_mayiuse = [](cpu_isa_t isa) { return mayiuse(isa, true); };

        status st = set_max_cpu_isa(isa);
        // status::unimplemented if the feature was disabled at compile time
        if (st == status::unimplemented) return;

        ASSERT_TRUE(st == status::success);

        auto info = isa_compatibility_table[isa];
        for (auto cur_isa : cpu_isa_all) {
            if (info.cpu_isa_compatible.find(cur_isa)
                    != info.cpu_isa_compatible.end())
                ASSERT_TRUE(
                        !test_mayiuse(info.this_isa) || test_mayiuse(cur_isa));
            else
                ASSERT_TRUE(!test_mayiuse(cur_isa));
        }
    }
};

TEST_P(isa_test_t, TestISA) {}
INSTANTIATE_TEST_SUITE_P(TestISACompatibility, isa_test_t,
        ::testing::Values(cpu_isa::sse41, cpu_isa::avx, cpu_isa::avx2,
                cpu_isa::avx512_mic, cpu_isa::avx512_mic_4ops,
                cpu_isa::avx512_core, cpu_isa::avx512_core_vnni,
                cpu_isa::avx512_core_bf16, cpu_isa::avx512_core_amx,
                cpu_isa::all));

} // namespace dnnl
