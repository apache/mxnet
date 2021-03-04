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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

namespace dnnl {

class submemory_test_cpp_t
    : public ::testing::TestWithParam<dnnl_engine_kind_t> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(submemory_test_cpp_t, SubmemoryMemoryInteraction) {
    auto engine_kind = static_cast<engine::kind>(GetParam());

    SKIP_IF(engine::get_count(engine_kind) == 0,
            "Engine kind is not supported");

    engine eng(engine_kind, 0);

    const memory::dim dst_offset = 1;
    const memory::dim copy_size = 1;

    float src_buf[1] = {35};
    auto src = test::make_memory(
            {{1}, memory::data_type::f32, memory::format_tag::a}, eng);
    {
        auto mapped_src_ptr = map_memory<float>(src);
        std::copy(src_buf, src_buf + sizeof(src_buf) / sizeof(src_buf[0]),
                static_cast<float *>(mapped_src_ptr));
    }

    float dst_buf[2] = {1, 0};
    auto dst = test::make_memory(
            {{2}, memory::data_type::f32, memory::format_tag::a}, eng);
    {
        auto mapped_dst_ptr = map_memory<float>(dst);
        std::copy(dst_buf, dst_buf + sizeof(dst_buf) / sizeof(dst_buf[0]),
                static_cast<float *>(mapped_dst_ptr));
    }

    memory::desc dst_submemory
            = dst.get_desc().submemory_desc({copy_size}, {dst_offset});

    reorder::primitive_desc reorder_pd(
            eng, src.get_desc(), eng, dst_submemory, primitive_attr());
    reorder reorder_prim(reorder_pd);

    stream strm(eng);
    reorder_prim.execute(strm, src, dst);
    strm.wait();

    dst_buf[dst_offset] = src_buf[0];
    {
        auto mapped_dst_ptr = map_memory<float>(dst);
        for (size_t i = 0; i < sizeof(dst_buf) / sizeof(dst_buf[0]); ++i)
            ASSERT_EQ(mapped_dst_ptr[i], dst_buf[i]) << "at position " << i;
    }
}

namespace {
struct print_to_string_param_name_t {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(info.param);
    }
};

auto all_engine_kinds = ::testing::Values(dnnl_cpu, dnnl_gpu);

} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, submemory_test_cpp_t, all_engine_kinds,
        print_to_string_param_name_t());

} // namespace dnnl
