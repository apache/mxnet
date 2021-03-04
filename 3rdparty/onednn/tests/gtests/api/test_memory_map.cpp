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

class memory_map_test_c_t
    : public ::testing::TestWithParam<dnnl_engine_kind_t> {
protected:
    void SetUp() override {
        auto engine_kind = GetParam();
        if (dnnl_engine_get_count(engine_kind) == 0) return;

        DNNL_CHECK(dnnl_engine_create(&engine, engine_kind, 0));
        DNNL_CHECK(
                dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));
    }

    void TearDown() override {
        if (engine) { DNNL_CHECK(dnnl_engine_destroy(engine)); }
        if (stream) { DNNL_CHECK(dnnl_stream_destroy(stream)); }
    }

    dnnl_engine_t engine = nullptr;
    dnnl_stream_t stream = nullptr;
};

class memory_map_test_cpp_t
    : public ::testing::TestWithParam<dnnl_engine_kind_t> {};

TEST_P(memory_map_test_c_t, MapNullMemory) {
    SKIP_IF(!engine, "Engine kind is not supported.");

    int ndims = 4;
    dnnl_dims_t dims = {2, 3, 4, 5};
    dnnl_memory_desc_t mem_d;
    dnnl_memory_t mem;

    DNNL_CHECK(dnnl_memory_desc_init_by_tag(
            &mem_d, ndims, dims, dnnl_f32, dnnl_nchw));
    DNNL_CHECK(dnnl_memory_create(&mem, &mem_d, engine, nullptr));

    void *mapped_ptr;
    DNNL_CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
    ASSERT_EQ(mapped_ptr, nullptr);

    DNNL_CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
    DNNL_CHECK(dnnl_memory_destroy(mem));
}

HANDLE_EXCEPTIONS_FOR_TEST_P(memory_map_test_c_t, Map) {
    SKIP_IF(!engine, "Engine kind is not supported.");

    const int ndims = 1;
    const dnnl_dim_t N = 15;
    const dnnl_dims_t dims = {N};

    dnnl_memory_desc_t mem_d;
    DNNL_CHECK(dnnl_memory_desc_init_by_tag(
            &mem_d, ndims, dims, dnnl_f32, dnnl_x));

    // Create and fill mem_ref to use as a reference
    dnnl_memory_t mem_ref;
    DNNL_CHECK(
            dnnl_memory_create(&mem_ref, &mem_d, engine, DNNL_MEMORY_ALLOCATE));

    float buffer_ref[N];
    std::iota(buffer_ref, buffer_ref + N, 1);

    void *mapped_ptr_ref;
    DNNL_CHECK(dnnl_memory_map_data(mem_ref, &mapped_ptr_ref));
    float *mapped_ptr_ref_f32 = static_cast<float *>(mapped_ptr_ref);
    std::copy(buffer_ref, buffer_ref + N, mapped_ptr_ref_f32);
    DNNL_CHECK(dnnl_memory_unmap_data(mem_ref, mapped_ptr_ref));

    // Create memory for the tested engine
    dnnl_memory_t mem;
    DNNL_CHECK(dnnl_memory_create(&mem, &mem_d, engine, DNNL_MEMORY_ALLOCATE));

    // Reorder mem_ref to memory
    dnnl_primitive_desc_t reorder_pd;
    DNNL_CHECK(dnnl_reorder_primitive_desc_create(
            &reorder_pd, &mem_d, engine, &mem_d, engine, nullptr));

    dnnl_primitive_t reorder;
    DNNL_CHECK(dnnl_primitive_create(&reorder, reorder_pd));

    dnnl_exec_arg_t reorder_args[2]
            = {{DNNL_ARG_SRC, mem_ref}, {DNNL_ARG_DST, mem}};
    DNNL_CHECK(dnnl_primitive_execute(reorder, stream, 2, reorder_args));
    DNNL_CHECK(dnnl_stream_wait(stream));

    // Validate the results
    void *mapped_ptr;
    DNNL_CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
    float *mapped_ptr_f32 = static_cast<float *>(mapped_ptr);
    for (size_t i = 0; i < N; i++) {
        ASSERT_EQ(mapped_ptr_f32[i], buffer_ref[i]);
    }
    DNNL_CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));

    // Clean up
    DNNL_CHECK(dnnl_primitive_destroy(reorder));
    DNNL_CHECK(dnnl_primitive_desc_destroy(reorder_pd));

    DNNL_CHECK(dnnl_memory_destroy(mem));
    DNNL_CHECK(dnnl_memory_destroy(mem_ref));
}

HANDLE_EXCEPTIONS_FOR_TEST_P(memory_map_test_cpp_t, Map) {
    auto engine_kind = static_cast<engine::kind>(GetParam());

    SKIP_IF(engine::get_count(engine_kind) == 0,
            "Engine kind is not supported");

    engine eng(engine_kind, 0);

    const dnnl::memory::dim N = 7;
    memory::desc mem_d({N}, memory::data_type::f32, memory::format_tag::x);

    auto mem_ref = test::make_memory(mem_d, eng);

    float buffer_ref[N];
    std::iota(buffer_ref, buffer_ref + N, 1);

    float *mapped_ptr_ref = mem_ref.map_data<float>();
    std::copy(buffer_ref, buffer_ref + N, mapped_ptr_ref);
    mem_ref.unmap_data(mapped_ptr_ref);

    auto mem = test::make_memory(mem_d, eng);

    reorder::primitive_desc reorder_pd(
            eng, mem_d, eng, mem_d, primitive_attr());
    reorder reorder_prim(reorder_pd);

    stream strm(eng);
    reorder_prim.execute(strm, mem_ref, mem);
    strm.wait();

    float *mapped_ptr = mem.map_data<float>();
    for (size_t i = 0; i < N; i++) {
        ASSERT_EQ(mapped_ptr[i], buffer_ref[i]);
    }
    mem.unmap_data(mapped_ptr);
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

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_map_test_c_t, all_engine_kinds,
        print_to_string_param_name_t());

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_map_test_cpp_t,
        all_engine_kinds, print_to_string_param_name_t());

} // namespace dnnl
