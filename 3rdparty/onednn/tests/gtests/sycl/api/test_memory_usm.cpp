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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdint>
#include <vector>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace dnnl {

#ifdef DNNL_SYCL_DPCPP

class fill_kernel;

class sycl_memory_usm_test : public ::testing::TestWithParam<engine::kind> {
protected:
    static void fill_data(void *usm_ptr, memory::dim n, const engine &eng) {
        auto alloc_kind = cl::sycl::get_pointer_type(
                usm_ptr, sycl_interop::get_context(eng));
        if (alloc_kind == cl::sycl::usm::alloc::host
                && alloc_kind == cl::sycl::usm::alloc::shared) {
            for (int i = 0; i < n; i++)
                ((float *)usm_ptr)[i] = float(i);
        } else {
            std::vector<float> host_ptr(n);
            for (int i = 0; i < n; i++)
                host_ptr[i] = float(i);

            auto q = sycl_interop::get_queue(stream(eng));
            q.memcpy(usm_ptr, host_ptr.data(), n * sizeof(float)).wait();
        }
    }
};

TEST_P(sycl_memory_usm_test, Constructor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    void *ptr = cl::sycl::malloc_shared(sizeof(float) * n,
            sycl_interop::get_device(eng), sycl_interop::get_context(eng));
    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, ptr);

    ASSERT_EQ(ptr, mem.get_data_handle());

    {
        for (int i = 0; i < n; i++) {
            ((float *)ptr)[i] = float(i);
        }
    }

    {
        float *ptr_f32 = (float *)mem.get_data_handle();
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(ptr_f32[i], float(i));
        }
    }

    cl::sycl::free(ptr, sycl_interop::get_context(eng));
}

TEST_P(sycl_memory_usm_test, ConstructorNone) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, DNNL_MEMORY_NONE);

    ASSERT_EQ(nullptr, mem.get_data_handle());
}

TEST_P(sycl_memory_usm_test, ConstructorAllocate) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, DNNL_MEMORY_ALLOCATE);

    void *ptr = mem.get_data_handle();
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

TEST_P(sycl_memory_usm_test, DefaultConstructor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm);

    void *ptr = mem.get_data_handle();
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(static_cast<dnnl_engine_kind_t>(info.param));
    }
};
} // namespace

INSTANTIATE_TEST_SUITE_P(Simple, sycl_memory_usm_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

#endif

} // namespace dnnl
