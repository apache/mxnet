/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include <numeric>
#include <utility>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "test_reorder_common.hpp"

namespace dnnl {

using f32_f32 = std::pair<float, float>;
using s32_s32 = std::pair<int32_t, int32_t>;
using s8_s8 = std::pair<int8_t, int8_t>;

using cfg_f32 = test_simple_params<f32_f32>;
using cfg_s32 = test_simple_params<s32_s32>;
using cfg_s8 = test_simple_params<s8_s8>;

using reorder_simple_test_f32_f32 = reorder_simple_test<f32_f32>;
using reorder_simple_test_s32_s32 = reorder_simple_test<s32_s32>;
using reorder_simple_test_s8_s8 = reorder_simple_test<s8_s8>;

using fmt = memory::format_tag;

TEST_P(reorder_simple_test_s32_s32, TestsReorder) {
    Test();
}
TEST_P(reorder_simple_test_f32_f32, TestsReorder) {
    Test();
}
TEST_P(reorder_simple_test_s8_s8, TestsReorder) {
    Test();
}

INSTANTIATE_TEST_SUITE_P(CornerCases, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::nchw, fmt::nc, {2, 16, 8, 8}, true,
                                  dnnl_invalid_arguments},
                cfg_f32 {fmt::nchw, fmt::nchw, {0, 16, 8, 8}},
                cfg_f32 {fmt::nchw, fmt::nChw8c, {0, 5, 8, 8}},
                cfg_f32 {fmt::nchw, fmt::nChw16c, {0, 5, 8, 8}},
                cfg_f32 {fmt::OIhw8o8i, fmt::oihw, {13, 0, 3, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::OIhw8o8i, {0, 32, 3, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::oihw, {16, 31, 0, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::OIhw16o16i, {32, 16, 3, 0}}));

CPU_INSTANTIATE_TEST_SUITE_P(PaddedData, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::nchw, fmt::nChw8c, {2, 28, 3, 4}},
                cfg_f32 {fmt::nChw8c, fmt::nchw, {2, 28, 3, 4}},
                cfg_f32 {fmt::chwn, fmt::nChw8c, {2, 28, 3, 4}},
                cfg_f32 {fmt::nChw8c, fmt::chwn, {2, 28, 3, 4}},
                cfg_f32 {fmt::nhwc, fmt::nChw8c, {3, 28, 3, 4}},
                cfg_f32 {fmt::nChw8c, fmt::nhwc, {3, 28, 3, 4}},

                cfg_f32 {fmt::nchw, fmt::nChw16c, {2, 28, 3, 4}},
                cfg_f32 {fmt::nChw16c, fmt::nchw, {2, 28, 3, 4}},
                cfg_f32 {fmt::chwn, fmt::nChw16c, {2, 28, 3, 4}},
                cfg_f32 {fmt::nChw16c, fmt::chwn, {2, 28, 3, 4}},
                cfg_f32 {fmt::nhwc, fmt::nChw16c, {3, 28, 3, 4}},
                cfg_f32 {fmt::nChw16c, fmt::nhwc, {3, 28, 3, 4}},

                cfg_f32 {fmt::ncdhw, fmt::nCdhw16c, {2, 28, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw16c, fmt::ncdhw, {2, 28, 2, 3, 4}},
                // cfg_f32{fmt::cdhwn, fmt::nCdhw16c, {2, 28, 2, 3, 4}},
                // cfg_f32{fmt::nCdhw16c, fmt::cdhwn, {2, 28, 2, 3, 4}},
                cfg_f32 {fmt::ndhwc, fmt::nCdhw16c, {3, 28, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw16c, fmt::ndhwc, {3, 28, 2, 3, 4}}));

CPU_INSTANTIATE_TEST_SUITE_P(Data_3d, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::ncdhw, fmt::nCdhw16c, {2, 32, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw16c, fmt::ncdhw, {2, 32, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw8c, fmt::ncdhw, {2, 32, 2, 3, 4}},
                cfg_f32 {fmt::ndhwc, fmt::nCdhw16c, {3, 32, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw16c, fmt::ndhwc, {3, 32, 2, 3, 4}},
                cfg_f32 {fmt::ndhwc, fmt::nCdhw8c, {3, 32, 2, 3, 4}},
                cfg_f32 {fmt::nCdhw8c, fmt::ndhwc, {3, 32, 2, 3, 4}}));

CPU_INSTANTIATE_TEST_SUITE_P(PaddedWeights, reorder_simple_test_f32_f32,
        ::testing::Values(
                // Oi(d)hw16o
                cfg_f32 {fmt::oihw, fmt::Oihw16o, {17, 23, 2, 3}},
                cfg_f32 {fmt::Oihw16o, fmt::oihw, {17, 23, 2, 3}},
                cfg_f32 {fmt::oidhw, fmt::Oidhw16o, {17, 23, 2, 2, 3}},
                cfg_f32 {fmt::Oidhw16o, fmt::oidhw, {17, 23, 2, 2, 3}},
                // OIhw16i16o
                cfg_f32 {fmt::oihw, fmt::OIhw16i16o, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::oihw, {17, 23, 2, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw16o16i, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::oihw, {17, 23, 2, 3}},
                cfg_f32 {fmt::hwio, fmt::OIhw16i16o, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::hwio, {17, 23, 2, 3}},
                // OIhw16o16i
                cfg_f32 {fmt::oihw, fmt::OIhw16o16i, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::oihw, {17, 23, 2, 3}},
                // IOhw16o16i
                cfg_f32 {fmt::oihw, fmt::IOhw16o16i, {17, 23, 2, 3}},
                cfg_f32 {fmt::IOhw16o16i, fmt::oihw, {17, 23, 2, 3}},
                // gOdhwi16o
                cfg_f32 {fmt::goidhw, fmt::gOdhwi16o, {2, 17, 23, 2, 2, 3}},
                cfg_f32 {fmt::gOdhwi16o, fmt::goidhw, {2, 17, 23, 3, 2, 3}},
                // OIdhw16o16i
                cfg_f32 {fmt::oidhw, fmt::OIdhw16o16i, {17, 23, 2, 3, 3}},
                cfg_f32 {fmt::OIdhw16o16i, fmt::oidhw, {17, 23, 2, 3, 3}},
                // IOdhw16o16i
                cfg_f32 {fmt::oidhw, fmt::IOdhw16o16i, {17, 23, 2, 3, 3}},
                cfg_f32 {fmt::IOdhw16o16i, fmt::oidhw, {17, 23, 2, 3, 3}},
                // gOIdhw16i16o
                cfg_f32 {fmt::goidhw, fmt::gOIdhw16i16o, {2, 17, 23, 2, 2, 3}},
                cfg_f32 {fmt::gOIdhw16i16o, fmt::goidhw, {2, 17, 23, 3, 2, 3}},
                // gOIdhw16o16i
                cfg_f32 {fmt::goidhw, fmt::gOIdhw16o16i, {2, 17, 23, 2, 2, 3}},
                cfg_f32 {fmt::gOIdhw16o16i, fmt::goidhw, {2, 17, 23, 3, 2, 3}},
                // gIOdhw16o16i
                cfg_f32 {fmt::goidhw, fmt::gIOdhw16o16i, {2, 17, 23, 2, 2, 3}},
                cfg_f32 {fmt::gIOdhw16o16i, fmt::goidhw, {2, 17, 23, 3, 2, 3}},
                // Oihw16o
                cfg_f32 {fmt::oihw, fmt::Oihw16o, {17, 23, 2, 3}},
                cfg_f32 {fmt::Oihw16o, fmt::oihw, {17, 23, 2, 3}},
                // OIhw8i8o
                cfg_f32 {fmt::oihw, fmt::OIhw8i8o, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::oihw, {17, 23, 2, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw8o8i, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw8o8i, fmt::oihw, {17, 23, 2, 3}},
                cfg_f32 {fmt::hwio, fmt::OIhw8i8o, {17, 23, 2, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::hwio, {17, 23, 2, 3}}));

CPU_INSTANTIATE_TEST_SUITE_P(Weights_3d, reorder_simple_test_f32_f32,
        ::testing::Values(
                cfg_f32 {fmt::oidhw, fmt::OIdhw8i8o, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::OIdhw8i8o, fmt::oidhw, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::oidhw, fmt::OIdhw8o8i, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::OIdhw8o8i, fmt::oidhw, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::oidhw, fmt::OIdhw8o4i, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::OIdhw8o4i, fmt::oidhw, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::dhwio, fmt::OIdhw8i8o, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::OIdhw8i8o, fmt::dhwio, {16, 24, 2, 3, 3}},
                cfg_f32 {fmt::goidhw, fmt::gOdhwi8o, {2, 16, 24, 2, 2, 3}},
                cfg_f32 {fmt::gOdhwi8o, fmt::goidhw, {2, 16, 24, 3, 2, 3}},
                cfg_f32 {fmt::goidhw, fmt::gOIdhw8i8o, {2, 16, 24, 2, 2, 3}},
                cfg_f32 {fmt::gOIdhw8i8o, fmt::goidhw, {2, 16, 24, 3, 2, 3}},
                cfg_f32 {fmt::goidhw, fmt::gOIdhw8o8i, {2, 16, 24, 2, 2, 3}},
                cfg_f32 {fmt::gOIdhw8o8i, fmt::goidhw, {2, 16, 24, 3, 2, 3}},
                cfg_f32 {fmt::giodhw, fmt::gOIdhw8o8i, {2, 16, 24, 2, 2, 3}},
                cfg_f32 {fmt::gOIdhw8o4i, fmt::goidhw, {2, 16, 24, 3, 2, 3}},
                cfg_f32 {fmt::giodhw, fmt::gOIdhw8o4i, {2, 16, 24, 2, 2, 3}},
                cfg_f32 {fmt::goidhw, fmt::giodhw, {2, 16, 24, 3, 2, 3}},
                cfg_f32 {fmt::iodhw, fmt::OIdhw8o8i, {16, 24, 2, 2, 3}},
                // OIdhw16i16o and IOdhw16o16i
                cfg_f32 {fmt::oidhw, fmt::OIdhw16i16o, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::OIdhw16i16o, fmt::oidhw, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::dhwio, fmt::OIdhw16i16o, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::OIdhw16i16o, fmt::dhwio, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::oidhw, fmt::IOdhw16o16i, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::IOdhw16o16i, fmt::oidhw, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::dhwio, fmt::IOdhw16o16i, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::IOdhw16o16i, fmt::dhwio, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::OIdhw16i16o, fmt::IOdhw16o16i, {64, 48, 2, 3, 4}},
                cfg_f32 {fmt::IOdhw16o16i, fmt::OIdhw16i16o, {64, 48, 2, 3, 4}},
                // gOIdhw16i16o and gIOdhw16o16i
                cfg_f32 {fmt::goidhw, fmt::gOIdhw16i16o, {2, 64, 96, 2, 3, 4}},
                cfg_f32 {fmt::gOIdhw16i16o, fmt::goidhw, {2, 64, 96, 2, 3, 4}},
                cfg_f32 {fmt::goidhw, fmt::gIOdhw16o16i, {2, 64, 96, 2, 3, 4}},
                cfg_f32 {fmt::gIOdhw16o16i, fmt::goidhw, {2, 64, 96, 2, 3, 4}},
                cfg_f32 {fmt::gOIdhw16i16o, fmt::gIOdhw16o16i,
                        {2, 64, 96, 2, 3, 4}},
                cfg_f32 {fmt::gIOdhw16o16i, fmt::gOIdhw16i16o,
                        {2, 64, 96, 2, 3, 4}}));

CPU_INSTANTIATE_TEST_SUITE_P(Data, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::nchw, fmt::nchw, {10, 10, 13, 13}},
                cfg_f32 {fmt::nchw, fmt::nhwc, {10, 10, 10, 10}},
                cfg_f32 {fmt::nhwc, fmt::nchw, {10, 10, 10, 10}},
                cfg_f32 {fmt::nchw, fmt::chwn, {28, 3, 10, 10}},
                cfg_f32 {fmt::chwn, fmt::nchw, {28, 3, 10, 10}},
                cfg_f32 {fmt::nhwc, fmt::nhwc, {10, 10, 13, 13}},
                cfg_f32 {fmt::nchw, fmt::nChw8c, {2, 32, 4, 4}},
                cfg_f32 {fmt::nChw8c, fmt::nchw, {2, 32, 4, 4}},
                cfg_f32 {fmt::chwn, fmt::nChw8c, {28, 96, 10, 10}},
                cfg_f32 {fmt::nChw8c, fmt::chwn, {28, 96, 10, 10}},
                cfg_f32 {fmt::nhwc, fmt::nChw8c, {3, 64, 16, 16}},
                cfg_f32 {fmt::nChw8c, fmt::nhwc, {3, 64, 16, 16}},
                cfg_f32 {fmt::nChw8c, fmt::nChw16c, {10, 96, 27, 27}},
                cfg_f32 {fmt::nChw16c, fmt::nChw8c, {10, 96, 27, 27}},
                cfg_f32 {fmt::nchw, fmt::nChw16c, {2, 64, 4, 4}},
                cfg_f32 {fmt::nChw16c, fmt::nchw, {2, 64, 4, 4}},
                cfg_f32 {fmt::chwn, fmt::nChw16c, {28, 96, 10, 10}},
                cfg_f32 {fmt::nChw16c, fmt::chwn, {28, 96, 10, 10}},
                cfg_f32 {fmt::nhwc, fmt::nChw16c, {2, 64, 4, 4}},
                cfg_f32 {fmt::nChw16c, fmt::nhwc, {2, 64, 4, 4}},
                cfg_f32 {fmt::abcd, fmt::abdc, {10, 10, 10, 10}}));

CPU_INSTANTIATE_TEST_SUITE_P(Weights_0, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::hwio, fmt::oihw, {32, 32, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::hwio, {32, 32, 3, 3}},
                cfg_f32 {fmt::hwio, fmt::Ohwi8o, {32, 32, 3, 3}},
                cfg_f32 {fmt::Ohwi8o, fmt::hwio, {32, 32, 3, 3}},
                cfg_f32 {fmt::hwio, fmt::Ohwi16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::Ohwi16o, fmt::hwio, {64, 64, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw8i8o, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::oihw, {32, 32, 3, 3}},
                cfg_f32 {fmt::ihwo, fmt::OIhw8i8o, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::ihwo, {32, 32, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw8o8i, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8o8i, fmt::oihw, {32, 32, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw8o4i, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8o4i, fmt::oihw, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::OIhw8o8i, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8o8i, fmt::OIhw8i8o, {32, 32, 3, 3}},
                cfg_f32 {fmt::hwio, fmt::OIhw8i8o, {32, 32, 3, 3}},
                cfg_f32 {fmt::OIhw8i8o, fmt::hwio, {32, 32, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::hwigo, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::hwigo, fmt::goihw, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw8i8o, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::gOIhw8i8o, fmt::goihw, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw8o8i, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::gOIhw8o8i, fmt::goihw, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw8o4i, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::gOIhw8o4i, fmt::goihw, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::gOIhw8i8o, fmt::gOIhw8o8i, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::gOIhw8o8i, fmt::gOIhw8i8o, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw16i16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::oihw, {64, 64, 3, 3}},
                cfg_f32 {fmt::ihwo, fmt::OIhw16i16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::ihwo, {64, 64, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw16o16i, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::oihw, {64, 64, 3, 3}},
                cfg_f32 {fmt::hwio, fmt::OIhw16i16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::hwio, {64, 64, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw16i16o, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOIhw16i16o, fmt::goihw, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOIhw16o16i, fmt::goihw, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::OIhw16o16i, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::OIhw16i16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::gOIhw16i16o, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOIhw16o16i, fmt::gOIhw16i16o, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::Oihw16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::Oihw16o, fmt::oihw, {64, 64, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOihw16o, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOihw16o, fmt::goihw, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::Ohwi16o, fmt::Oihw16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::Oihw16o, fmt::Ohwi16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::gOhwi16o, fmt::gOihw16o, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOihw16o, fmt::gOhwi16o, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::Goihw8g, {16, 16, 16, 3, 3}},
                cfg_f32 {fmt::Goihw8g, fmt::goihw, {16, 16, 16, 3, 3}}));

CPU_INSTANTIATE_TEST_SUITE_P(Weights_1, reorder_simple_test_f32_f32,
        ::testing::Values(
                cfg_f32 {fmt::goihw, fmt::Goihw16g, {32, 32, 32, 3, 3}},
                cfg_f32 {fmt::Goihw16g, fmt::goihw, {32, 32, 32, 3, 3}},
                cfg_f32 {fmt::oihw, fmt::iohw, {32, 32, 3, 3}},
                cfg_f32 {fmt::iohw, fmt::oihw, {32, 32, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::giohw, {2, 32, 32, 3, 3}},
                cfg_f32 {fmt::giohw, fmt::goihw, {2, 32, 32, 3, 3}}));

CPU_INSTANTIATE_TEST_SUITE_P(Weights_IOhw16o16i, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::oihw, fmt::IOhw16o16i, {64, 64, 3, 3}},
                cfg_f32 {fmt::IOhw16o16i, fmt::oihw, {64, 64, 3, 3}},
                cfg_f32 {fmt::OIhw16i16o, fmt::IOhw16o16i, {64, 64, 3, 3}},
                cfg_f32 {fmt::IOhw16o16i, fmt::OIhw16i16o, {64, 64, 3, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw16o16i, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gIOhw16o16i, fmt::goihw, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gOIhw16i16o, fmt::gIOhw16o16i, {2, 64, 64, 3, 3}},
                cfg_f32 {fmt::gIOhw16o16i, fmt::gOIhw16i16o,
                        {2, 64, 64, 3, 3}}));

CPU_INSTANTIATE_TEST_SUITE_P(Simple, reorder_simple_test_s32_s32,
        ::testing::Values(cfg_s32 {fmt::nchw, fmt::nChw16c, {2, 64, 4, 4}},
                cfg_s32 {fmt::nChw16c, fmt::nchw, {2, 64, 4, 4}}));

CPU_INSTANTIATE_TEST_SUITE_P(Simple, reorder_simple_test_s8_s8,
        ::testing::Values(cfg_s8 {fmt::oihw, fmt::OIhw4i16o4i, {64, 64, 3, 3}},
                cfg_s8 {fmt::OIhw4i16o4i, fmt::oihw, {64, 64, 3, 3}},
                cfg_s8 {fmt::goihw, fmt::gOIhw4i16o4i, {2, 64, 64, 3, 3}},
                cfg_s8 {fmt::gOIhw4i16o4i, fmt::goihw, {2, 64, 64, 3, 3}}));

GPU_INSTANTIATE_TEST_SUITE_P(Data, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::nchw, fmt::nhwc, {2, 48, 5, 4}},
                cfg_f32 {fmt::nchw, fmt::NChw16n16c, {64, 32, 5, 6}},
                cfg_f32 {fmt::nChw16c, fmt::NChw16n16c, {32, 48, 6, 9}},
                cfg_f32 {fmt::ncdhw, fmt::ndhwc, {2, 48, 2, 5, 4}},
                cfg_f32 {fmt::ncdhw, fmt::NCdhw16n16c, {32, 32, 2, 5, 6}},
                cfg_f32 {fmt::nCdhw16c, fmt::NCdhw16n16c, {32, 48, 2, 6, 9}}));

GPU_INSTANTIATE_TEST_SUITE_P(Data_1D, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::ncw, fmt::nCw16c, {2, 48, 7}},
                cfg_f32 {fmt::nCw16c, fmt::ncw, {2, 48, 7}},
                cfg_f32 {fmt::ncw, fmt::NCw16n16c, {32, 48, 7}},
                cfg_f32 {fmt::NCw16n16c, fmt::ncw, {32, 48, 7}},
                cfg_f32 {fmt::nCw16c, fmt::NCw16n16c, {32, 48, 7}},
                cfg_f32 {fmt::NCw16n16c, fmt::nCw16c, {32, 48, 7}}));

GPU_INSTANTIATE_TEST_SUITE_P(PaddedData, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::nchw, fmt::nChw8c, {2, 28, 5, 4}},
                cfg_f32 {fmt::nChw8c, fmt::nchw, {2, 28, 5, 4}},
                cfg_f32 {fmt::nchw, fmt::nChw16c, {2, 28, 5, 4}},
                cfg_f32 {fmt::nChw16c, fmt::nchw, {2, 28, 5, 4}}));

GPU_INSTANTIATE_TEST_SUITE_P(Weights, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::oihw, fmt::IOhw16i16o, {32, 48, 2, 3}},
                cfg_f32 {fmt::oihw, fmt::OIhw16o16i, {32, 32, 2, 2}},
                cfg_f32 {fmt::hwigo, fmt::gIOhw16i16o, {2, 64, 32, 2, 3}},
                cfg_f32 {fmt::goihw, fmt::gOIhw16o16i, {2, 32, 64, 2, 3}},
                cfg_f32 {fmt::OIhw16o16i, fmt::IOhw16i16o, {32, 48, 2, 3}},
                cfg_f32 {fmt::gOIhw16o16i, fmt::gIOhw16i16o, {2, 64, 32, 3, 2}},
                cfg_f32 {fmt::oidhw, fmt::OIdhw16i16o, {64, 32, 3, 9, 5}},
                cfg_f32 {
                        fmt::goidhw, fmt::gOIdhw16i16o, {2, 32, 64, 2, 2, 7}}));

GPU_INSTANTIATE_TEST_SUITE_P(weights_1D, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::oiw, fmt::OIw8i16o2i, {32, 48, 7}},
                cfg_f32 {fmt::OIw8i16o2i, fmt::oiw, {32, 48, 7}},
                cfg_f32 {fmt::goiw, fmt::gOIw8i16o2i, {8, 32, 48, 7}},
                cfg_f32 {fmt::gOIw8i16o2i, fmt::goiw, {8, 32, 48, 7}},
                cfg_f32 {fmt::oiw, fmt::OIw16i16o, {32, 48, 7}},
                cfg_f32 {fmt::OIw16i16o, fmt::oiw, {32, 48, 7}},
                cfg_f32 {fmt::oiw, fmt::OIw8o4i, {32, 48, 7}},
                cfg_f32 {fmt::OIw8o4i, fmt::oiw, {32, 48, 7}},
                cfg_f32 {fmt::goiw, fmt::gOIw16i16o, {8, 32, 48, 7}},
                cfg_f32 {fmt::gOIw16i16o, fmt::goiw, {8, 32, 48, 7}},
                cfg_f32 {fmt::goiw, fmt::gOIw8o4i, {8, 32, 48, 7}},
                cfg_f32 {fmt::gOIw8o4i, fmt::goiw, {8, 32, 48, 7}},
                cfg_f32 {fmt::oiw, fmt::Oiw16o, {32, 48, 7}},
                cfg_f32 {fmt::Oiw16o, fmt::oiw, {32, 48, 7}},
                cfg_f32 {fmt::goiw, fmt::gOiw16o, {8, 32, 48, 7}},
                cfg_f32 {fmt::gOiw16o, fmt::goiw, {8, 32, 48, 7}},
                cfg_f32 {fmt::oiw, fmt::IOw16i16o, {32, 48, 7}},
                cfg_f32 {fmt::IOw16i16o, fmt::oiw, {32, 48, 7}},
                cfg_f32 {fmt::goiw, fmt::gIOw16i16o, {8, 32, 48, 7}},
                cfg_f32 {fmt::gIOw16i16o, fmt::goiw, {8, 32, 48, 7}}));

GPU_INSTANTIATE_TEST_SUITE_P(PaddedWeights, reorder_simple_test_f32_f32,
        ::testing::Values(cfg_f32 {fmt::oihw, fmt::IOhw16i16o, {17, 23, 2, 1}},
                cfg_f32 {fmt::goihw, fmt::gOIhw16o16i, {2, 17, 23, 1, 2}}));

} // namespace dnnl
