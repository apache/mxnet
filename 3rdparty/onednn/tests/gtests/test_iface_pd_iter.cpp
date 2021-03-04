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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {

const dnnl_status_t ok = dnnl_success;

class pd_iter_test_t : public ::testing::Test {
protected:
    dnnl_engine_t engine;
    void SetUp() override {
        auto engine_kind
                = static_cast<dnnl_engine_kind_t>(get_test_engine_kind());
        ASSERT_EQ(dnnl_engine_create(&engine, engine_kind, 0), ok);
    }
    void TearDown() override { dnnl_engine_destroy(engine); }
};

TEST_F(pd_iter_test_t, TestReLUImpls) {
    dnnl_memory_desc_t dense_md;
    dnnl_dims_t dims = {4, 16, 16, 16};
    ASSERT_EQ(dnnl_memory_desc_init_by_tag(
                      &dense_md, 4, dims, dnnl_f32, dnnl_nchw),
            ok);

    dnnl_eltwise_desc_t ed;
    ASSERT_EQ(dnnl_eltwise_forward_desc_init(&ed, dnnl_forward_inference,
                      dnnl_eltwise_relu, &dense_md, 0., 0.),
            ok);

    dnnl_primitive_desc_iterator_t it;
    dnnl_status_t rc;

    ASSERT_EQ(rc = dnnl_primitive_desc_iterator_create(
                      &it, &ed, nullptr, engine, nullptr),
            ok); /* there should be at least one impl */

    dnnl_primitive_desc_t pd;
    ASSERT_NE(pd = dnnl_primitive_desc_iterator_fetch(it), nullptr);
    dnnl_primitive_desc_destroy(pd);

    while ((rc = dnnl_primitive_desc_iterator_next(it)) == ok) {
        ASSERT_NE(pd = dnnl_primitive_desc_iterator_fetch(it), nullptr);
        dnnl_primitive_desc_destroy(pd);
    }

    ASSERT_EQ(rc, dnnl_iterator_ends);
    dnnl_primitive_desc_iterator_destroy(it);
}

TEST(pd_next_impl, TestEltwiseImpl) {
    auto eng = get_test_engine();
    memory::desc md(
            {8, 32, 4, 4}, memory::data_type::f32, memory::format_tag::nChw8c);

    eltwise_forward::desc ed(
            prop_kind::forward_training, algorithm::eltwise_relu, md, 0, 0);
    eltwise_forward::primitive_desc epd(ed, eng);

    std::string impl0(epd.impl_info_str());
    eltwise_forward e0(epd);

    while (epd.next_impl()) {
        std::string impl1(epd.impl_info_str());
        eltwise_forward e1(epd);
        ASSERT_NE(impl0, impl1);
        impl0 = impl1;
    }
}

} // namespace dnnl
