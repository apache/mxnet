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
#include "src/common/dnnl_thread.hpp"

#if DNNL_X64
#include "src/cpu/x64/cpu_isa_traits.hpp"
#endif

namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class wino_conv_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    struct input_data_t {
        data_type dat_dt;
        data_type wei_dt;
        bool wino_supported;
        bool backward_supported;
    } input_f32, input_f16, input_int8;

    void SetUp() override {
        input_f32.dat_dt = data_type::f32;
        input_f32.wei_dt = data_type::f32;

        input_f16.dat_dt = data_type::f16;
        input_f16.wei_dt = data_type::f16;
        input_f16.backward_supported = false;

        input_int8.dat_dt = data_type::u8;
        input_int8.wei_dt = data_type::s8;
        input_int8.backward_supported = false;

#if DNNL_X64
        input_f32.wino_supported
                = (get_test_engine_kind() == engine::kind::cpu
                          && impl::cpu::x64::mayiuse(
                                  impl::cpu::x64::avx512_common))
                || get_test_engine_kind() == engine::kind::gpu;
        input_f16.wino_supported = false;
        input_int8.wino_supported = get_test_engine_kind() == engine::kind::cpu
                && impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core);
        input_f32.backward_supported
                = get_test_engine_kind() == engine::kind::cpu
                && impl::dnnl_thr_syncable();

#else
        input_f32.wino_supported = false;
        input_f16.wino_supported = false;
        input_int8.wino_supported = false;
#endif // DNNL_X64
    }
};

TEST_F(wino_conv_test_t, TestSmallPadding) {
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        memory::desc src_md {{1, 16, 7, 7}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 3, 3}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 7, 7}, input.dat_dt, tag::any};
        auto fwd_op_desc = convolution_forward::desc(prop_kind::forward,
                algorithm::convolution_winograd, src_md, wei_md, dst_md, {1, 1},
                {1, 1}, {1, 1});

        if (input.wino_supported) {
            EXPECT_NO_THROW(
                    convolution_forward::primitive_desc(fwd_op_desc, eng));
            if (input.backward_supported) {
                auto bwdd_op_desc = convolution_backward_data::desc(
                        algorithm::convolution_winograd, src_md, wei_md, dst_md,
                        {1, 1}, {1, 1}, {1, 1});
                auto bwdw_op_desc = convolution_backward_weights::desc(
                        algorithm::convolution_winograd, src_md, wei_md, dst_md,
                        {1, 1}, {1, 1}, {1, 1});
                EXPECT_NO_THROW(convolution_backward_data::primitive_desc(
                        bwdd_op_desc, eng,
                        convolution_forward::primitive_desc(fwd_op_desc, eng)));
                EXPECT_NO_THROW(convolution_backward_weights::primitive_desc(
                        bwdw_op_desc, eng,
                        convolution_forward::primitive_desc(fwd_op_desc, eng)));
            }
        } else {
            EXPECT_ANY_THROW(
                    convolution_forward::primitive_desc(fwd_op_desc, eng));
        }
    }
}

TEST_F(wino_conv_test_t, TestLargePadding) {
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        memory::desc src_md {{1, 16, 7, 7}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 3, 3}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 9, 9}, input.dat_dt, tag::any};
        auto fwd_op_desc = convolution_forward::desc(prop_kind::forward,
                algorithm::convolution_winograd, src_md, wei_md, dst_md, {1, 1},
                {2, 2}, {2, 2});

        EXPECT_ANY_THROW(convolution_forward::primitive_desc(fwd_op_desc, eng));
    }
}

TEST_F(wino_conv_test_t, TestUnsupportedKernel) {
    for (const auto &input : {input_f32, input_f16, input_int8}) {
        memory::desc src_md {{1, 16, 5, 5}, input.dat_dt, tag::any};
        memory::desc wei_md {{32, 16, 2, 2}, input.wei_dt, tag::any};
        memory::desc dst_md {{1, 32, 6, 6}, input.dat_dt, tag::any};
        auto fwd_op_desc = convolution_forward::desc(prop_kind::forward,
                algorithm::convolution_winograd, src_md, wei_md, dst_md, {1, 1},
                {1, 1}, {1, 1});
        EXPECT_ANY_THROW(convolution_forward::primitive_desc(fwd_op_desc, eng));
    }
}

} // namespace dnnl
