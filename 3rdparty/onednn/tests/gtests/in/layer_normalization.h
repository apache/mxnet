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

#define EXPAND_FORMATS(data, stat, diff) \
    memory::format_tag::data, memory::format_tag::stat, memory::format_tag::diff

#define PARAMS_NC(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(ab, a, ab), __VA_ARGS__, EPS, false, dnnl_success \
    }

#define PARAMS_TNC(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(abc, ab, abc), __VA_ARGS__, EPS, false, dnnl_success \
    }

#define PARAMS_TNC_CROSS_CASE(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(abc, ba, abc), __VA_ARGS__, EPS, false, dnnl_success \
    }

#define PARAMS_NTC(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(bac, ba, bac), __VA_ARGS__, EPS, false, dnnl_success \
    }

#define PARAMS_LDSNC(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(abcde, abcd, abcde), __VA_ARGS__, EPS, false, \
                dnnl_success \
    }

#define PARAMS_LDSNC_CROSS_CASE(...) \
    test_lnorm_params_t { \
        EXPAND_FORMATS(abcde, acdb, abcde), __VA_ARGS__, EPS, false, \
                dnnl_success \
    }

#define PARAMS_EF(...) \
    test_lnorm_params_t { EXPAND_FORMATS(abc, ab, abc), __VA_ARGS__ }

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, lnorm_test_t, ::testing::Values(__VA_ARGS__));

CPU_INST_TEST_CASE(SimpleExpectedFails,
        PARAMS_EF({-1, 27, 9}, EPS, true, dnnl_invalid_arguments),
        PARAMS_EF({1, -12, 10}, EPS, true, dnnl_invalid_arguments),
        PARAMS_EF({4, 20, -12}, EPS, true, dnnl_invalid_arguments));

CPU_INST_TEST_CASE(SimpleZeroDim, PARAMS_NC({0, 9}), PARAMS_NC({1, 0}));

CPU_INST_TEST_CASE(
        Simple_NC, PARAMS_NC({1, 100}), PARAMS_NC({20, 8}), PARAMS_NC({2, 10}));

CPU_INST_TEST_CASE(Simple_TNC, PARAMS_TNC({6, 32, 8}), PARAMS_TNC({2, 10, 4}),
        PARAMS_TNC({2, 8, 16}));

CPU_INST_TEST_CASE(CrossCase_TNC, PARAMS_TNC_CROSS_CASE({6, 32, 8}),
        PARAMS_TNC_CROSS_CASE({2, 10, 4}), PARAMS_TNC_CROSS_CASE({2, 8, 16}));

CPU_INST_TEST_CASE(Simple_NTC, PARAMS_TNC({64, 32, 8}), PARAMS_TNC({32, 10, 4}),
        PARAMS_TNC({12, 8, 16}));

CPU_INST_TEST_CASE(Simple_LDSNC, PARAMS_LDSNC({6, 2, 2, 32, 8}),
        PARAMS_LDSNC({2, 2, 2, 10, 4}), PARAMS_LDSNC({2, 2, 2, 8, 16}));

CPU_INST_TEST_CASE(CrossCase_LDSNC, PARAMS_LDSNC_CROSS_CASE({6, 2, 2, 32, 8}),
        PARAMS_LDSNC_CROSS_CASE({2, 2, 2, 10, 4}),
        PARAMS_LDSNC_CROSS_CASE({2, 2, 2, 8, 16}));
