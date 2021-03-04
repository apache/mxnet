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
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {

class pd_test_t : public ::testing::Test {
protected:
    engine e = get_test_engine();
    memory::desc dat_md {
            {16, 16, 16, 16}, memory::data_type::f32, memory::format_tag::nhwc};
    memory::desc wht_md {
            {16, 16, 1, 1}, memory::data_type::f32, memory::format_tag::oihw};
};

TEST_F(pd_test_t, ConvTestNotEmpty) {
    bool no_exception = true;
    bool is_empty = false;

    try {
        auto pd = convolution_forward::primitive_desc {
                {prop_kind::forward_inference, algorithm::convolution_direct,
                        dat_md, wht_md, dat_md, {1, 1}, {0, 0}, {0, 0}},
                e, false};
        is_empty = pd.get(true) == nullptr; // not reached if !allow_empty
    } catch (error &) { no_exception = false; }

    ASSERT_TRUE(no_exception);
    ASSERT_TRUE(!is_empty);
}

TEST_F(pd_test_t, ConvTestEmpty) {
    auto attrs = primitive_attr {};
    attrs.set_output_scales(0, {2.0});

    for (bool allow_empty : {true, false}) {
        bool no_exception = true;
        bool is_empty = false;

        try {
            auto pd = convolution_forward::primitive_desc {
                    {prop_kind::forward_inference,
                            algorithm::convolution_direct, dat_md, wht_md,
                            dat_md, {1, 1}, {0, 0}, {0, 0}},
                    attrs, e, allow_empty};
            is_empty = pd.get(true) == nullptr; // not reached if !allow_empty
        } catch (error &) { no_exception = false; }

        ASSERT_TRUE(no_exception == allow_empty);
        ASSERT_TRUE(is_empty == allow_empty);
    }
}

} // namespace dnnl
