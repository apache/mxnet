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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#include <limits>
#include <new>

#if DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <CL/sycl.hpp>
#endif

namespace dnnl {

bool is_sycl_engine(dnnl_engine_kind_t eng_kind) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (eng_kind == dnnl_cpu) return true;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (eng_kind == dnnl_gpu) return true;
#endif
    return false;
}

class memory_test_c : public ::testing::TestWithParam<dnnl_engine_kind_t> {
protected:
    void SetUp() override {
        eng_kind = GetParam();

        if (dnnl_engine_get_count(eng_kind) == 0) return;

        DNNL_CHECK(dnnl_engine_create(&engine, eng_kind, 0));
    }

    void TearDown() override {
        if (engine) { DNNL_CHECK(dnnl_engine_destroy(engine)); }
    }

    dnnl_engine_kind_t eng_kind;
    dnnl_engine_t engine = nullptr;
};

class memory_test_cpp : public ::testing::TestWithParam<dnnl_engine_kind_t> {};

TEST_P(memory_test_c, OutOfMemory) {
    SKIP_IF(!engine, "Engine is not found.");
    SKIP_IF(is_sycl_engine(eng_kind), "Do not test C API with SYCL.");

    dnnl_dim_t sz = std::numeric_limits<memory::dim>::max();
    dnnl_dims_t dims = {sz};
    dnnl_memory_desc_t md;
    DNNL_CHECK(dnnl_memory_desc_init_by_tag(&md, 1, dims, dnnl_u8, dnnl_x));

    dnnl_memory_t mem;
    dnnl_status_t s
            = dnnl_memory_create(&mem, &md, engine, DNNL_MEMORY_ALLOCATE);
    ASSERT_EQ(s, dnnl_out_of_memory);
}

TEST_P(memory_test_cpp, OutOfMemory) {
    dnnl_engine_kind_t eng_kind_c = GetParam();
    engine::kind eng_kind = static_cast<engine::kind>(eng_kind_c);
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");

    engine eng(eng_kind, 0);

    bool is_sycl = is_sycl_engine(eng_kind_c);

    auto sz = std::numeric_limits<memory::dim>::max();
#if DNNL_WITH_SYCL
    if (is_sycl) {
        auto dev = sycl_interop::get_device(eng);
        auto max_alloc_size
                = dev.get_info<cl::sycl::info::device::max_mem_alloc_size>();
        sz = (max_alloc_size < sz) ? max_alloc_size + 1 : sz;
    }
#endif

    auto dt = memory::data_type::u8;
    auto tag = memory::format_tag::x;
    memory::desc md({sz}, dt, tag);
    try {
        auto mem = test::make_memory(md, eng);
        ASSERT_NE(mem.get_data_handle(), nullptr);
    } catch (const dnnl::error &e) {
        ASSERT_EQ(e.status, dnnl_out_of_memory);
        return;
    } catch (const std::bad_alloc &) {
        // Expect bad_alloc only with SYCL.
        if (is_sycl) return;
        throw;
    }

    // XXX: SYCL does not always throw, even when allocating
    //  > max_mem_alloc_size bytes.
    if (!is_sycl) FAIL() << "Expected exception.";
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(info.param);
    }
};

auto all_engine_kinds = ::testing::Values(dnnl_cpu, dnnl_gpu);

} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_test_c, all_engine_kinds,
        PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_test_cpp, all_engine_kinds,
        PrintToStringParamName());

} // namespace dnnl
