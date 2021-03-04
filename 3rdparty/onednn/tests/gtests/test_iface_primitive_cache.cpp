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

#include "oneapi/dnnl/dnnl.hpp"
#include "src/common/primitive_cache.hpp"

int get_primitive_cache_size() {
    int result = 0;
    auto status = dnnl::impl::get_primitive_cache_size(&result);
    if (status != dnnl::impl::status::success) return -1;
    return result;
}

namespace dnnl {

void fill_primitive_cache(int n) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(get_test_engine_kind(), 0);
    for (int i = 0; i < n; i++) {
        // fill primitive cache with n primitives
        auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, {{i, 1, 1, 1}, dt::f32, tag::nchw},
                0.f, 0.f);
        auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
        auto relu = eltwise_forward(relu_pd);
    }
}

TEST(primitive_cache_test, TestDefaultCapacity) {
    auto default_capacity = get_primitive_cache_capacity();
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    ASSERT_EQ(default_capacity, 1024);
#else
    ASSERT_EQ(default_capacity, 0);
#endif
}

#ifndef DNNL_DISABLE_PRIMITIVE_CACHE

TEST(primitive_cache_test, TestInitState) {
    ASSERT_EQ(get_primitive_cache_size(), 0);
}

TEST(primitive_cache_test, TestSetCapacity) {
    set_primitive_cache_capacity(18);
    ASSERT_EQ(get_primitive_cache_capacity(), 18);
}

TEST(primitive_cache_test, TestClearCache) {
    set_primitive_cache_capacity(8);
    fill_primitive_cache(8);
    ASSERT_EQ(get_primitive_cache_size(), 8);

    set_primitive_cache_capacity(0);
    ASSERT_EQ(get_primitive_cache_size(), 0);
}

TEST(primitive_cache_test, TestEviction) {
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(22);
    fill_primitive_cache(30);
    ASSERT_EQ(get_primitive_cache_size(), 22);
}

TEST(primitive_cache_test, TestSizeLessCapacity) {
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(15);
    fill_primitive_cache(12);
    set_primitive_cache_capacity(13);
    ASSERT_EQ(get_primitive_cache_size(), 12);
}

TEST(primitive_cache_test, TestSizeGreaterCapacity) {
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(15);
    fill_primitive_cache(12);
    set_primitive_cache_capacity(10);
    ASSERT_EQ(get_primitive_cache_size(), 10);
}

TEST(primitive_cache_test, TestCacheHit) {
    set_primitive_cache_capacity(0);
    set_primitive_cache_capacity(2);
    fill_primitive_cache(1);
    fill_primitive_cache(1);
    ASSERT_EQ(get_primitive_cache_size(), 1);
}
#endif

} // namespace dnnl
