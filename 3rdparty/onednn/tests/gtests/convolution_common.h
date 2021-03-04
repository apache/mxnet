#if 0
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
#endif

#include "oneapi/dnnl/dnnl.hpp"

#define EXPAND_FORMATS(src, weights, bias, dst) \
    { \
        dnnl::memory::format_tag::src, dnnl::memory::format_tag::weights, \
                dnnl::memory::format_tag::bias, dnnl::memory::format_tag::dst \
    }

#define ALGORITHM dnnl::algorithm::convolution_direct

#ifdef DIRECTION_FORWARD
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_GPU_BLOCKED16x16 IOhw16i16o
#define FMT_WEIGHTS_GPU_BLOCKED16 OIhw16i16o
#if defined(FP32)
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#elif defined(S16S16S32)
#define FMT_WEIGHTS_BLOCKED16 OIhw8i16o2i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw8i16o2i
#elif defined(U8S8)
#define FMT_WEIGHTS_BLOCKED16 OIhw4i16o4i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw4i16o4i
#endif
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define TEST_CASE_NAME_PREFIX Forward
#elif defined DIRECTION_BACKWARD_DATA
#define FMT_WEIGHTS_BLOCKED OIhw8o8i
#define FMT_WEIGHTS_BLOCKED_G gOIhw8o8i
#define FMT_WEIGHTS_GPU_BLOCKED16x16 OIhw16o16i
#define FMT_WEIGHTS_GPU_BLOCKED16 OIhw16o16i
#if defined(FP32)
#define FMT_WEIGHTS_BLOCKED16 OIhw16o16i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16o16i
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i IOhw16o16i
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i gIOhw16o16i
#elif defined(S16S16S32)
#define FMT_WEIGHTS_BLOCKED16 OIhw8o16i2o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw8o16i2o
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i FMT_WEIGHTS_BLOCKED16_G
#endif
#define TEST_CASE_NAME_PREFIX BackwardData
#elif defined DIRECTION_BACKWARD_WEIGHTS
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i FMT_WEIGHTS_BLOCKED16_G
#define FMT_WEIGHTS_GPU_BLOCKED16x16 IOhw16i16o
#define FMT_WEIGHTS_GPU_BLOCKED16 IOhw16i16o
#define TEST_CASE_NAME_PREFIX BackwardWeights
#endif

#define FMT_BIAS x
#define FMT_NO_BIAS undef
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c

#define CONCAT_WITH_UNDERSCORE_(a, b) a##_##b
#define CONCAT_WITH_UNDERSCORE(a, b) CONCAT_WITH_UNDERSCORE_(a, b)

#define CPU_INST_TEST_CASE_(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, convolution_test, ::testing::Values(__VA_ARGS__))
#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)

#define GPU_INST_TEST_CASE_(str, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            str, convolution_test, ::testing::Values(__VA_ARGS__))
#define GPU_INST_TEST_CASE(str, ...) \
    GPU_INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)

#define INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE(str, __VA_ARGS__); \
    GPU_INST_TEST_CASE(str, __VA_ARGS__)

#define PARAMS(src, weights, bias, dst, ...) \
    test_convolution_params_t { \
        ALGORITHM, EXPAND_FORMATS(src, weights, bias, dst), \
                /* empty attributes */ {}, { \
            __VA_ARGS__ \
        } \
    }

#define PARAMS_EXPECT_FAIL(src, weights, bias, dst, code, ...) \
    test_convolution_params_t { \
        ALGORITHM, EXPAND_FORMATS(src, weights, bias, dst), \
                /* empty attributes */ {}, {__VA_ARGS__}, true, code \
    }

#define PARAMS_ATTR(src, weights, bias, dst, scale, policy, ...) \
    test_convolution_params_t { \
        ALGORITHM, EXPAND_FORMATS(src, weights, bias, dst), \
                {scale, test_convolution_attr_t::scale_t::policy}, { \
            __VA_ARGS__ \
        } \
    }

#ifdef TEST_PARAM_ATTR
#include "convolution_attr.h"
#else
#include "convolution_simple.h"
#endif
