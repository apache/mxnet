/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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
#include "math_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "test_convolution_eltwise_forward_common.hpp"
#include "gtest/gtest.h"

namespace dnnl {

using convolution_test = convolution_eltwise_test<float, float, float, float>;

TEST_P(convolution_test, TestConvolutionEltwise) {}

#define EXPAND_FORMATS(src, weights, bias, dst) \
    { \
        dnnl::memory::format_tag::src, dnnl::memory::format_tag::weights, \
                dnnl::memory::format_tag::bias, dnnl::memory::format_tag::dst \
    }

#define CONCAT_WITH_UNDERSCORE_(a, b) a##_##b
#define CONCAT_WITH_UNDERSCORE(a, b) CONCAT_WITH_UNDERSCORE_(a, b)

#define CPU_INST_TEST_CASE_(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, convolution_test, ::testing::Values(__VA_ARGS__))

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE( \
                    CONCAT_WITH_UNDERSCORE(Convolution, str), eltwise), \
            __VA_ARGS__)

#define INST_TEST_CASE_(str, ...) \
    INSTANTIATE_TEST_SUITE_P( \
            str, convolution_test, ::testing::Values(__VA_ARGS__))

#define INST_TEST_CASE(str, ...) \
    INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE( \
                    CONCAT_WITH_UNDERSCORE(Convolution, str), eltwise), \
            __VA_ARGS__)

#define EXPAND_ARGS(args) args

#define PARAMS(...) \
    EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_relu, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_elu, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_tanh, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_square, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_abs, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_linear, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV( \
                    algorithm::eltwise_bounded_relu, __VA_ARGS__)), \
            EXPAND_ARGS( \
                    PARAMS_CONV(algorithm::eltwise_soft_relu, __VA_ARGS__)), \
            EXPAND_ARGS( \
                    PARAMS_CONV(algorithm::eltwise_logistic, __VA_ARGS__)), \
            EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_swish, __VA_ARGS__))
//  Not testing due to not scaled output
//  EXPAND_ARGS(PARAMS_CONV(algorithm::eltwise_exp, __VA_ARGS__))
#define ELTWISE_ALPHA 0.5f
#define ELTWISE_BETA 1.5f

#define PARAMS_CONV(alg, src, weights, bias, dst, ...) \
    test_convolution_eltwise_params_t { \
        alg, dnnl::algorithm::convolution_direct, ELTWISE_ALPHA, ELTWISE_BETA, \
                EXPAND_FORMATS(src, weights, bias, dst), \
                /* empty attributes */ {}, { \
            __VA_ARGS__ \
        } \
    }

INST_TEST_CASE(SimpleSmall,
        PARAMS(nchw, oihw, x, nchw, 2, 1, 32, 13, 13, 48, 11, 11, 3, 3, 0, 0, 1,
                1),
        PARAMS(nchw, oihw, x, nchw, 2, 1, 16, 13, 13, 48, 13, 13, 1, 1, 0, 0, 1,
                1),
        PARAMS(nchw, goihw, x, nchw, 2, 64, 64, 16, 16, 64, 16, 16, 3, 3, 0, 0,
                1, 1),
        PARAMS(nchw, goihw, x, nchw, 2, 32, 32, 9, 9, 32, 9, 9, 1, 1, 0, 0, 1,
                1));

CPU_INST_TEST_CASE(SimpleSmall_Blocked,
        PARAMS(nChw8c, Goihw8g, x, nChw8c, 1, 48, 48, 20, 20, 48, 20, 20, 3, 3,
                1, 1, 1, 1),
        PARAMS(nChw8c, OIhw8i8o, x, nChw8c, 1, 1, 48, 20, 20, 48, 20, 20, 1, 1,
                0, 0, 1, 1),
        PARAMS(nChw8c, OIhw8i8o, x, nChw8c, 1, 1, 48, 20, 20, 48, 20, 20, 3, 3,
                0, 0, 1, 1));

CPU_INST_TEST_CASE(SimpleSmall_Blocked_Tail,
        PARAMS(nChw8c, Goihw8g, x, nChw8c, 1, 47, 47, 20, 20, 47, 20, 20, 3, 3,
                1, 1, 1, 1),
        PARAMS(nChw8c, OIhw8i8o, x, nChw8c, 1, 1, 47, 20, 20, 47, 20, 20, 1, 1,
                0, 0, 1, 1),
        PARAMS(nChw8c, OIhw8i8o, x, nChw8c, 1, 1, 47, 20, 20, 47, 20, 20, 3, 3,
                0, 0, 1, 1));

INST_TEST_CASE(SimpleSmall_Blocked16,
        PARAMS(nChw16c, Goihw16g, x, nChw16c, 1, 48, 48, 20, 20, 48, 20, 20, 3,
                3, 1, 1, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 1, 1, 48, 20, 20, 48, 20, 20, 1,
                1, 0, 0, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 1, 1, 48, 20, 20, 48, 20, 20, 3,
                3, 0, 0, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 2, 1, 32, 32, 32, 32, 32, 32, 3,
                3, 0, 0, 1, 1));

CPU_INST_TEST_CASE(SimpleSmall_Blocked16_Tail,
        PARAMS(nChw16c, Goihw16g, x, nChw16c, 1, 47, 47, 20, 20, 47, 20, 20, 3,
                3, 1, 1, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 1, 1, 47, 20, 20, 47, 20, 20, 1,
                1, 0, 0, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 1, 1, 47, 20, 20, 47, 20, 20, 3,
                3, 0, 0, 1, 1),
        PARAMS(nChw16c, OIhw16i16o, x, nChw16c, 2, 1, 32, 32, 32, 32, 32, 32, 3,
                3, 0, 0, 1, 1));

} // namespace dnnl
