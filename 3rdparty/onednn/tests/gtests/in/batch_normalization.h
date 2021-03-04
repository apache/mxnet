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

#define EPS 1e-5f

#define EXPAND_ARGS(args) args
#define EXPAND_SIZES_3D(...) \
    { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb, c, h, w) \
    { mb, c, 1, h, w }
#define EXPAND_FORMATS(data, diff) \
    { memory::format_tag::data, memory::format_tag::diff }

#define PARAMS(data, diff, mb, c, h, w, eps, ef, st) \
    test_bnorm_params_t { \
        EXPAND_FORMATS(data, diff), EXPAND_SIZES_2D(mb, c, h, w), eps, 4, ef, \
                st \
    }

#define PARAMS_3D(data, diff, mb, c, d, h, w, eps, ef, st) \
    test_bnorm_params_t { \
        EXPAND_FORMATS(data, diff), EXPAND_SIZES_3D(mb, c, d, h, w), eps, 5, \
                ef, st \
    }

#define PARAMS_N_3D(...) \
    EXPAND_ARGS(PARAMS_3D(ncdhw, ncdhw, __VA_ARGS__, false, dnnl_success))
#define PARAMS_B8_3D(...) \
    EXPAND_ARGS(PARAMS_3D(nCdhw8c, nCdhw8c, __VA_ARGS__, false, dnnl_success))
#define PARAMS_B16_3D(...) \
    EXPAND_ARGS(PARAMS_3D(nCdhw16c, nCdhw16c, __VA_ARGS__, false, dnnl_success))
#define PARAMS_N(...) \
    EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__, false, dnnl_success))
#define PARAMS_NHWC(...) \
    EXPAND_ARGS(PARAMS(nhwc, nhwc, __VA_ARGS__, false, dnnl_success))
#define PARAMS_NC(...) \
    EXPAND_ARGS(PARAMS(nc, nc, __VA_ARGS__, false, dnnl_success))
#define PARAMS_B8(...) \
    EXPAND_ARGS(PARAMS(nChw8c, nChw8c, __VA_ARGS__, false, dnnl_success))
#define PARAMS_B16(...) \
    EXPAND_ARGS(PARAMS(nChw16c, nChw16c, __VA_ARGS__, false, dnnl_success))
#define PARAMS_EF(...) EXPAND_ARGS(PARAMS(nchw, nchw, __VA_ARGS__))

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, bnorm_test, ::testing::Values(__VA_ARGS__));

#define GPU_INST_TEST_CASE(str, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            str, bnorm_test, ::testing::Values(__VA_ARGS__));

#define INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE(str, __VA_ARGS__); \
    GPU_INST_TEST_CASE(str, __VA_ARGS__)

CPU_INST_TEST_CASE(SimpleZeroDim, PARAMS_N(0, 27, 9, 10, EPS),
        PARAMS_N(1, 0, 10, 9, EPS), PARAMS_N(4, 20, 0, 12, EPS));

INST_TEST_CASE(SimpleExpectedFails,
        PARAMS_EF(-1, 27, 9, 10, EPS, true, dnnl_invalid_arguments),
        PARAMS_EF(1, -12, 10, 9, EPS, true, dnnl_invalid_arguments),
        PARAMS_EF(4, 20, -12, 12, EPS, true, dnnl_invalid_arguments));

CPU_INST_TEST_CASE(Simple_nChw16c_padded, PARAMS_B16(1, 27, 9, 10, EPS),
        PARAMS_B16(1, 12, 10, 9, EPS), PARAMS_B16(4, 20, 12, 12, EPS),
        PARAMS_B16(4, 9, 16, 16, EPS));

INST_TEST_CASE(Simple_nCdhw16c_padded, PARAMS_B16_3D(2, 12, 16, 8, 20, EPS),
        PARAMS_B16_3D(2, 9, 16, 8, 20, EPS),
        PARAMS_B16_3D(2, 23, 10, 8, 4, EPS),
        PARAMS_B16_3D(2, 27, 10, 8, 4, EPS));

INST_TEST_CASE(Simple_nChw8c_padded, PARAMS_B8(1, 27, 9, 10, EPS),
        PARAMS_B8(1, 12, 10, 9, EPS), PARAMS_B8(4, 20, 12, 12, EPS),
        PARAMS_B8(4, 7, 16, 16, EPS));

INST_TEST_CASE(Simple_nCdhw16c, PARAMS_B16_3D(2, 32, 4, 4, 4, EPS),
        PARAMS_B16_3D(2, 32, 4, 4, 4, EPS), PARAMS_B16_3D(2, 32, 8, 8, 8, EPS),
        PARAMS_B16_3D(2, 32, 8, 8, 8, EPS),
        PARAMS_B16_3D(2, 32, 16, 8, 20, EPS),
        PARAMS_B16_3D(2, 32, 16, 8, 20, EPS),
        PARAMS_B16_3D(2, 32, 10, 8, 4, EPS),
        PARAMS_B16_3D(2, 32, 10, 8, 4, EPS));

CPU_INST_TEST_CASE(Simple_nCdhw8c, PARAMS_B8_3D(2, 32, 4, 4, 4, EPS),
        PARAMS_B8_3D(2, 32, 4, 4, 4, EPS), PARAMS_B8_3D(2, 32, 8, 8, 8, EPS),
        PARAMS_B8_3D(2, 32, 8, 8, 8, EPS), PARAMS_B8_3D(2, 32, 16, 8, 20, EPS),
        PARAMS_B8_3D(2, 32, 16, 8, 20, EPS), PARAMS_B8_3D(2, 32, 10, 8, 4, EPS),
        PARAMS_B8_3D(2, 32, 10, 8, 4, EPS));

CPU_INST_TEST_CASE(Simple_NC, PARAMS_NC(2, 8, 1, 1, EPS),
        PARAMS_NC(2, 10, 1, 1, EPS), PARAMS_NC(2, 8, 1, 1, EPS),
        PARAMS_NC(2, 10, 1, 1, EPS));

CPU_INST_TEST_CASE(Simple_NCDHW, PARAMS_N_3D(2, 8, 1, 1, 1, EPS),
        PARAMS_N_3D(2, 10, 1, 1, 1, EPS), PARAMS_N_3D(2, 8, 4, 4, 4, EPS),
        PARAMS_N_3D(2, 10, 4, 4, 4, EPS));

CPU_INST_TEST_CASE(Simple_NCHW, PARAMS_N(2, 8, 1, 1, EPS),
        PARAMS_N(2, 10, 1, 1, EPS), PARAMS_N(2, 8, 4, 4, EPS),
        PARAMS_N(2, 10, 4, 4, EPS));

CPU_INST_TEST_CASE(Simple_NHWC, PARAMS_NHWC(2, 8, 1, 1, EPS),
        PARAMS_NHWC(2, 10, 1, 1, EPS), PARAMS_NHWC(2, 8, 4, 4, EPS),
        PARAMS_NHWC(2, 10, 4, 4, EPS));

CPU_INST_TEST_CASE(Simple_Blocked, PARAMS_B8(2, 8, 1, 1, EPS),
        PARAMS_B8(2, 8, 4, 4, EPS), PARAMS_B8(2, 8, 6, 6, EPS),
        PARAMS_B8(2, 16, 4, 4, EPS), PARAMS_B8(2, 16, 4, 4, EPS),
        PARAMS_B8(2, 16, 8, 8, EPS), PARAMS_B8(2, 16, 8, 8, EPS),
        PARAMS_B8(2, 16, 16, 8, EPS), PARAMS_B8(2, 16, 16, 8, EPS),
        PARAMS_B8(2, 16, 10, 8, EPS), PARAMS_B8(2, 16, 10, 8, EPS),
        PARAMS_B16(2, 16, 4, 4, EPS), PARAMS_B16(2, 16, 4, 4, EPS),
        PARAMS_B16(2, 16, 8, 8, EPS), PARAMS_B16(2, 16, 8, 8, EPS),
        PARAMS_B16(2, 16, 16, 8, EPS), PARAMS_B16(2, 16, 16, 8, EPS),
        PARAMS_B16(2, 16, 10, 8, EPS), PARAMS_B16(2, 16, 10, 8, EPS));

CPU_INST_TEST_CASE(GoogleNet_NCHW, PARAMS_N(2, 64, 112, 112, EPS),
        PARAMS_N(2, 64, 56, 56, EPS), PARAMS_N(2, 192, 56, 56, EPS),
        PARAMS_N(2, 96, 28, 28, EPS), PARAMS_N(2, 16, 28, 28, EPS),
        PARAMS_N(2, 64, 28, 28, EPS), PARAMS_N(2, 128, 28, 28, EPS),
        PARAMS_N(2, 32, 28, 28, EPS), PARAMS_N(2, 96, 28, 28, EPS),
        PARAMS_N(2, 96, 14, 14, EPS), PARAMS_N(2, 16, 14, 14, EPS),
        PARAMS_N(2, 192, 14, 14, EPS), PARAMS_N(2, 208, 14, 14, EPS),
        PARAMS_N(2, 48, 14, 14, EPS), PARAMS_N(2, 64, 14, 14, EPS),
        PARAMS_N(2, 112, 14, 14, EPS), PARAMS_N(2, 24, 14, 14, EPS),
        PARAMS_N(2, 160, 14, 14, EPS), PARAMS_N(2, 224, 14, 14, EPS),
        PARAMS_N(2, 128, 4, 4, EPS), PARAMS_N(2, 128, 14, 14, EPS),
        PARAMS_N(2, 512, 14, 14, EPS), PARAMS_N(2, 256, 14, 14, EPS),
        PARAMS_N(2, 144, 14, 14, EPS), PARAMS_N(2, 32, 14, 14, EPS),
        PARAMS_N(2, 228, 14, 14, EPS), PARAMS_N(2, 528, 14, 14, EPS),
        PARAMS_N(2, 320, 14, 14, EPS), PARAMS_N(2, 160, 7, 7, EPS),
        PARAMS_N(2, 32, 7, 7, EPS), PARAMS_N(2, 256, 7, 7, EPS),
        PARAMS_N(2, 320, 7, 7, EPS), PARAMS_N(2, 128, 7, 7, EPS),
        PARAMS_N(2, 192, 7, 7, EPS), PARAMS_N(2, 48, 7, 7, EPS),
        PARAMS_N(2, 384, 7, 7, EPS));

CPU_INST_TEST_CASE(GoogleNet_Blocked_8, PARAMS_B8(2, 64, 112, 112, EPS),
        PARAMS_B8(2, 64, 56, 56, EPS), PARAMS_B8(2, 192, 56, 56, EPS),
        PARAMS_B8(2, 96, 28, 28, EPS), PARAMS_B8(2, 16, 28, 28, EPS),
        PARAMS_B8(2, 64, 28, 28, EPS), PARAMS_B8(2, 128, 28, 28, EPS),
        PARAMS_B8(2, 32, 28, 28, EPS), PARAMS_B8(2, 96, 28, 28, EPS),
        PARAMS_B8(2, 96, 14, 14, EPS), PARAMS_B8(2, 16, 14, 14, EPS),
        PARAMS_B8(2, 192, 14, 14, EPS), PARAMS_B8(2, 208, 14, 14, EPS),
        PARAMS_B8(2, 48, 14, 14, EPS), PARAMS_B8(2, 64, 14, 14, EPS),
        PARAMS_B8(2, 112, 14, 14, EPS), PARAMS_B8(2, 24, 14, 14, EPS),
        PARAMS_B8(2, 160, 14, 14, EPS), PARAMS_B8(2, 224, 14, 14, EPS),
        PARAMS_B8(2, 128, 4, 4, EPS), PARAMS_B8(2, 128, 14, 14, EPS),
        PARAMS_B8(2, 512, 14, 14, EPS), PARAMS_B8(2, 256, 14, 14, EPS),
        PARAMS_B8(2, 144, 14, 14, EPS), PARAMS_B8(2, 32, 14, 14, EPS),
        PARAMS_B8(2, 528, 14, 14, EPS), PARAMS_B8(2, 320, 14, 14, EPS),
        PARAMS_B8(2, 160, 7, 7, EPS), PARAMS_B8(2, 32, 7, 7, EPS),
        PARAMS_B8(2, 256, 7, 7, EPS), PARAMS_B8(2, 320, 7, 7, EPS),
        PARAMS_B8(2, 128, 7, 7, EPS), PARAMS_B8(2, 192, 7, 7, EPS),
        PARAMS_B8(2, 48, 7, 7, EPS), PARAMS_B8(2, 384, 7, 7, EPS));

CPU_INST_TEST_CASE(GoogleNet_Blocked_16, PARAMS_B16(2, 64, 112, 112, EPS),
        PARAMS_B16(2, 64, 56, 56, EPS), PARAMS_B16(2, 192, 56, 56, EPS),
        PARAMS_B16(2, 96, 28, 28, EPS), PARAMS_B16(2, 16, 28, 28, EPS),
        PARAMS_B16(2, 64, 28, 28, EPS), PARAMS_B16(2, 128, 28, 28, EPS),
        PARAMS_B16(2, 32, 28, 28, EPS), PARAMS_B16(2, 96, 28, 28, EPS),
        PARAMS_B16(2, 96, 14, 14, EPS), PARAMS_B16(2, 16, 14, 14, EPS),
        PARAMS_B16(2, 192, 14, 14, EPS), PARAMS_B16(2, 208, 14, 14, EPS),
        PARAMS_B16(2, 48, 14, 14, EPS), PARAMS_B16(2, 64, 14, 14, EPS),
        PARAMS_B16(2, 112, 14, 14, EPS),
        //PARAMS_B16(2, 24, 14, 14, EPS),
        PARAMS_B16(2, 160, 14, 14, EPS), PARAMS_B16(2, 224, 14, 14, EPS),
        PARAMS_B16(2, 128, 4, 4, EPS), PARAMS_B16(2, 128, 14, 14, EPS),
        PARAMS_B16(2, 512, 14, 14, EPS), PARAMS_B16(2, 256, 14, 14, EPS),
        PARAMS_B16(2, 144, 14, 14, EPS), PARAMS_B16(2, 32, 14, 14, EPS),
        PARAMS_B16(2, 528, 14, 14, EPS), PARAMS_B16(2, 320, 14, 14, EPS),
        PARAMS_B16(2, 160, 7, 7, EPS), PARAMS_B16(2, 32, 7, 7, EPS),
        PARAMS_B16(2, 256, 7, 7, EPS), PARAMS_B16(2, 320, 7, 7, EPS),
        PARAMS_B16(2, 128, 7, 7, EPS), PARAMS_B16(2, 192, 7, 7, EPS),
        PARAMS_B16(2, 48, 7, 7, EPS), PARAMS_B16(2, 384, 7, 7, EPS));

GPU_INST_TEST_CASE(Simple,
        PARAMS(nchw, nchw, 32, 32, 1, 1, EPS, false, dnnl_success),
        PARAMS(nchw, nchw, 32, 32, 4, 5, EPS, false, dnnl_success),
        PARAMS_3D(ncdhw, ncdhw, 32, 32, 2, 4, 2, EPS, false, dnnl_success),
        PARAMS(NChw16n16c, NChw16n16c, 32, 32, 1, 1, EPS, false, dnnl_success),
        PARAMS(NChw16n16c, NChw16n16c, 32, 32, 4, 5, EPS, false, dnnl_success),
        PARAMS_3D(NCdhw16n16c, NCdhw16n16c, 32, 32, 2, 4, 2, EPS, false,
                dnnl_success),
        PARAMS(nChw16c, nChw16c, 32, 32, 1, 1, EPS, false, dnnl_success),
        PARAMS(nChw16c, nChw16c, 32, 32, 4, 5, EPS, false, dnnl_success),
        PARAMS(nChw16c, nChw16c, 25, 32, 1, 1, EPS, false, dnnl_success),
        PARAMS(nChw16c, nChw16c, 25, 32, 4, 5, EPS, false, dnnl_success),
        PARAMS_3D(
                nCdhw16c, nCdhw16c, 25, 32, 2, 4, 2, EPS, false, dnnl_success));
