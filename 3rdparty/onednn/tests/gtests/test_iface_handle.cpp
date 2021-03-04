/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

namespace dnnl {

class handle_test_t : public ::testing::Test {
public:
    engine e;

protected:
    void SetUp() override { e = get_test_engine(); }
};

TEST_F(handle_test_t, TestHandleConstructorsAndOperators) {
    // The initial state is 0
    convolution_forward::primitive_desc pd;
    ASSERT_TRUE((bool)pd == false);
    ASSERT_TRUE((dnnl_primitive_desc_t)pd == nullptr);

    // Dummy descriptor just to be able to create a pd
    auto d = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct,
            {{1, 16, 7, 7}, memory::data_type::f32, memory::format_tag::any},
            {{16, 16, 1, 1}, memory::data_type::f32, memory::format_tag::any},
            {{1, 16, 7, 7}, memory::data_type::f32, memory::format_tag::any},
            {1, 1}, {0, 0}, {0, 0});
    pd = convolution_forward::primitive_desc(d, e);

    // Copy from pd to pd1
    auto pd1 = pd;
    ASSERT_TRUE(pd1 == pd);

    // This should set pd's handle to 0
    pd1 = std::move(pd);
    ASSERT_TRUE(pd1 != pd);
    ASSERT_TRUE((dnnl_primitive_desc_t)pd == nullptr);
}

} // namespace dnnl
