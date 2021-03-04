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

#if defined(FP16) || defined(FP32) || defined(F16F16F32) || defined(BF16BF16F32)
INST_TEST_CASE(TestGEMM,
        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, {}, {}, true,
                dnnl_invalid_arguments},

        test_params {'N', 'N', 1, 1, 1, 1.0, 0.0, 4, 4, 4},
        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'N', 'n', 31, 21, 11, 2.0, 1.5, 61, 51, 81},
        test_params {'n', 'T', 31, 21, 11, 2.0, 1.5, 61, 51, 81},
        test_params {'T', 'N', 31, 21, 11, 2.0, 1.5, 61, 51, 81},
        test_params {'t', 't', 31, 21, 11, 2.0, 1.5, 61, 51, 81},
        test_params {'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100},
        test_params {'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100},
        test_params {'t', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100},
        test_params {'t', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2},
        test_params {'t', 't', 2, 2, 10000, 1.0, 2.0, 2, 10000, 2},

        make_test_params_with_offset(
                {1, 2, 3}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f, 100, 100, 100),
        make_test_params_with_offset(
                {30, 20, 10}, 'n', 't', 100, 2, 100, 1.0f, 2.0f, 100, 100, 100),

        test_params {'n', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000},
        test_params {'n', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000},
        test_params {'t', 'n', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000},
        test_params {'t', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000},
        test_params {'n', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000},
        test_params {'n', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000},
        test_params {'t', 't', 2000, 2000, 2000, 1.0, 0.0, 2000, 2000, 2000},
        test_params {'t', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000});

CPU_INST_TEST_CASE(TestGEMV,
        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 0.0f, 1000, 1, 1},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 0.0f, 2000, 3000, 3000},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 0.0f, 2000, 1, 1},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 0.0f, 1, 3000, 3000},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 0.0f, 1000, 1000, 1},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 0.0f, 2000, 2000, 3000},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 0.0f, 2000, 1000, 1},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 0.0f, 1, 2000, 3000},

        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 0.0f, 1010, 1, 30},
        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 0.0f, 1010, 20, 1},
        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 0.0f, 1010, 20, 30},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 0.0f, 2010, 3010, 3010},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 0.0f, 2010, 20, 30},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 0.0f, 20, 3010, 3010},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 0.0f, 1010, 1010, 20},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 0.0f, 2010, 2010, 3010},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 0.0f, 2010, 1010, 20},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 0.0f, 20, 2010, 3010},

        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1, 1},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 1.0f, 2000, 3000, 3000},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1, 1},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 1.0f, 1, 3000, 3000},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1000, 1},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 1.0f, 2000, 2000, 3000},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1000, 1},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 1.0f, 1, 2000, 3000});

/**
 * These cases are used to test the small-N avx-512 sgemm TN kernels.
 * Note: The kernels assume a column major layout while the external 
 * APIs assume row major layout, so the M/N and transA/transB values 
 * are swapped.
 */
CPU_INST_TEST_CASE(TestGEMM_smalln,
        test_params {'n', 't', 5, 512, 512, 1.0f, 1.0f, 512, 512, 512},
        test_params {'n', 't', 5, 512, 1536, 1.0f, 1.0f, 1536, 1536, 512},
        test_params {'n', 't', 5, 512, 2048, 1.0f, 1.0f, 2048, 2048, 512},
        test_params {'n', 't', 5, 2048, 512, 1.0f, 1.0f, 512, 512, 2048},
        test_params {'n', 't', 7, 512, 512, 0.0f, 1.0f, 512, 512, 512},
        test_params {'n', 't', 7, 512, 1536, 1.0f, 0.0f, 1536, 1536, 512},
        test_params {'n', 't', 7, 512, 2048, 0.5f, 0.5f, 2048, 2048, 512},
        test_params {'n', 't', 7, 2048, 512, 1.0f, 1.0f, 512, 512, 2048},
        test_params {'n', 't', 4, 512, 512, 1.0f, 1.0f, 512, 512, 512},
        test_params {'n', 't', 4, 512, 1536, 1.0f, 1.0f, 1536, 1536, 512},
        test_params {'n', 't', 4, 512, 2048, 1.0f, 1.0f, 2048, 2048, 512},
        test_params {'n', 't', 4, 2048, 512, 1.0f, 1.0f, 512, 512, 2048},
        test_params {'n', 't', 8, 512, 512, 1.0f, 1.0f, 512, 512, 512},
        test_params {'n', 't', 8, 512, 1536, 1.0f, 1.0f, 1536, 1536, 512},
        test_params {'n', 't', 8, 512, 2048, 1.0f, 1.0f, 2048, 2048, 512},
        test_params {'n', 't', 8, 2048, 512, 1.0f, 1.0f, 512, 512, 2048});

#if defined(FP32) || defined(BF16BF16F32)
INST_TEST_CASE(TestGEMM_packed,
        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, {}, {false, true},
                true, dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, {}, {true, false},
                true, dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, {}, {true, true},
                true, dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, {}, {true, true},
                true, dnnl_invalid_arguments},

        make_test_params_pack(
                {true, false}, 'N', 'n', 31, 21, 11, 1.0f, 1.5f, 61, 51, 81),
        make_test_params_pack(
                {false, true}, 'n', 'T', 31, 21, 11, 1.0f, 1.5f, 61, 51, 81),
        make_test_params_pack(
                {true, false}, 'T', 'N', 31, 21, 11, 1.0f, 1.5f, 61, 51, 81),
        make_test_params_pack(
                {true, true}, 't', 't', 31, 21, 11, 1.0f, 1.5f, 61, 51, 81),
        make_test_params_pack({false, true}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f,
                100, 100, 100),
        make_test_params_pack(
                {true, true}, 'n', 't', 100, 2, 100, 1.0f, 2.0f, 100, 100, 100),
        make_test_params_pack(
                {true, true}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f, 10000, 2, 2),
        make_test_params_pack(
                {true, true}, 'n', 'n', 100, 1, 100, 1.0f, 2.0f, 100, 100, 100),
        make_test_params_pack({true, false}, 'n', 'n', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100),
        make_test_params_pack({false, true}, 'n', 'n', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100),

        make_test_params_pack({true, false}, 'n', 'n', 3000, 3000, 3000, 1.0f,
                2.0f, 3000, 3000, 3000),
        make_test_params_pack({true, false}, 't', 'n', 3000, 3000, 3000, 1.0f,
                0.0f, 3000, 3000, 3000),
        make_test_params_pack({true, false}, 'n', 't', 3000, 3000, 3000, 1.0f,
                1.0f, 3000, 3000, 3000),
        make_test_params_pack({true, false}, 't', 't', 3000, 3000, 3000, 1.0f,
                2.0f, 3000, 3000, 3000),

        make_test_params_pack({false, true}, 'n', 'n', 200, 20000, 2000, 1.0f,
                2.0f, 2000, 20000, 20000),
        make_test_params_pack({false, true}, 'n', 'n', 2000, 2000, 2000, 1.0f,
                2.0f, 2000, 2000, 2000),
        make_test_params_pack({true, true}, 'n', 'n', 2000, 5000, 2000, 1.0f,
                2.0f, 2000, 5000, 5000),
        make_test_params_pack({true, true}, 'n', 'n', 5000, 100, 2000, 1.0f,
                2.0f, 2000, 100, 100),
        make_test_params_pack({false, true}, 't', 'n', 2000, 2000, 2000, 1.0f,
                0.0f, 2000, 2000, 2000),
        make_test_params_pack({false, true}, 't', 'n', 2000, 5000, 2000, 1.0f,
                2.0f, 2000, 5000, 5000),
        make_test_params_pack({false, true}, 't', 'n', 5000, 100, 2000, 1.0f,
                2.0f, 5000, 100, 100),
        make_test_params_pack({false, true}, 'n', 't', 2000, 2000, 2000, 1.0f,
                1.0f, 2000, 2000, 2000),
        make_test_params_pack({false, true}, 't', 't', 2000, 2000, 2000, 1.0f,
                2.0f, 2000, 2000, 2000),
        make_test_params_pack({true, true}, 't', 't', 2000, 5000, 2000, 1.0f,
                2.0f, 2000, 2000, 5000),
        make_test_params_pack({true, true}, 't', 't', 5000, 100, 2000, 1.0f,
                2.0f, 5000, 2000, 100),

        make_test_params_pack({true, false}, 'n', 'n', 150, 150, 8000, 1.0f,
                3.0f, 8000, 150, 150),
        make_test_params_pack({true, true}, 'n', 't', 200, 200, 8000, 1.0f,
                3.0f, 8000, 8000, 200),
        make_test_params_pack({false, true}, 't', 'n', 200, 300, 8000, 1.0f,
                3.0f, 200, 300, 300));
#endif

#elif defined(BF16BF16BF16)

INST_TEST_CASE(TestGEMM,
        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, {}, {}, true,
                dnnl_invalid_arguments},

        test_params {'N', 'N', 1, 1, 1, 1.0, 0.0, 4, 4, 4},
        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80},
        test_params {'N', 'n', 31, 21, 11, 2.5, 1.5, 61, 51, 81},
        test_params {'n', 'T', 31, 21, 11, 2.5, 1.5, 61, 51, 81},
        test_params {'T', 'N', 31, 21, 11, 2.5, 1.5, 61, 51, 81},
        test_params {'t', 't', 31, 21, 11, 2.5, 1.5, 61, 51, 81},
        test_params {'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100},
        test_params {'n', 't', 100, 2, 58, 1.0, 2.0, 100, 100, 100},
        test_params {'t', 'n', 2, 100, 61, 1.0, 2.0, 100, 100, 100},
        test_params {'t', 't', 2, 100, 60, 1.0, 2.0, 100, 100, 100},
        test_params {'n', 'n', 2, 2, 11, 1.0, -1.0, 20, 2, 2},
        test_params {'t', 't', 2, 2, 11, 1.0, -1.0, 2, 20, 2},

        make_test_params_with_offset(
                {1, 2, 3}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f, 100, 100, 100),
        make_test_params_with_offset(
                {30, 20, 10}, 'n', 't', 100, 2, 100, 1.0f, 2.0f, 100, 100, 100),

        test_params {'n', 'n', 2000, 2000, 20, 1.0, 0.0, 20, 2000, 2000},
        test_params {'n', 'n', 3000, 3000, 30, 1.0, 0.0, 30, 3000, 3000},
        test_params {'t', 'n', 2000, 2000, 20, 1.0, 0.0, 2000, 2000, 2000},
        test_params {'t', 'n', 3000, 3000, 30, 1.0, 0.0, 3000, 3000, 3000},
        test_params {'n', 't', 2000, 2000, 20, 1.0, 0.0, 20, 20, 2000},
        test_params {'n', 't', 3000, 3000, 30, 1.0, 0.0, 30, 30, 3000},
        test_params {'t', 't', 2000, 2000, 20, 1.0, 0.0, 2000, 20, 2000},
        test_params {'t', 't', 3000, 3000, 30, 1.0, 0.0, 3000, 30, 3000});

#else
constexpr test_igemm_params fix_use_oc = {'F', false, false, true};
constexpr test_igemm_params col_use_oc = {'C', false, false, true};
constexpr test_igemm_params row_use_oc = {'R', false, false, true};

constexpr test_igemm_params fix_use_all_offsets = {'F', true, true, true};
constexpr test_igemm_params col_use_all_offsets = {'C', true, true, true};
constexpr test_igemm_params row_use_all_offsets = {'R', true, true, true};

constexpr test_igemm_params fix_no_offsets = {'F', false, false, false};
constexpr test_igemm_params col_no_offsets = {'C', false, false, false};
constexpr test_igemm_params row_no_offsets = {'R', false, false, false};

INST_TEST_CASE(TestGEMM_expected_failures,
        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, {}, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, {}, {}, true,
                dnnl_invalid_arguments},

        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, fix_use_oc, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, fix_use_oc, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, fix_use_oc, {}, true,
                dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, fix_use_oc, {}, true,
                dnnl_invalid_arguments},

        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, fix_use_all_offsets,
                {}, true, dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, fix_use_all_offsets,
                {}, true, dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, fix_use_all_offsets,
                {}, true, dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, fix_use_all_offsets,
                {}, true, dnnl_invalid_arguments},

        test_params {'t', 'n', 3, 2, 1, 1.0, 0.0, 2, 5, 8, {}, {true, true},
                true, dnnl_invalid_arguments},
        test_params {'n', 'n', 3, 2, 2, 1.0, 0.0, 1, 5, 8, {}, {false, true},
                true, dnnl_invalid_arguments},
        test_params {'n', 't', 3, 2, 2, 1.0, 0.0, 3, 1, 8, {}, {true, false},
                true, dnnl_invalid_arguments},
        test_params {'n', 'd', 3, 2, 1, 1.0, 0.0, 3, 3, 3, {}, {false, true},
                true, dnnl_invalid_arguments});

INST_TEST_CASE(TestGEMM_general_cases_fix_offset,
        test_params {'N', 'n', 30, 20, 10, 1.0, 0.0, 60, 50, 80, fix_use_oc},
        test_params {'n', 'T', 30, 20, 10, 1.0, 0.0, 60, 50, 80, fix_use_oc},
        test_params {'T', 'N', 30, 20, 10, 1.0, 0.0, 60, 50, 80, fix_use_oc},
        test_params {'t', 't', 30, 20, 10, 1.0, 0.0, 60, 50, 80, fix_use_oc},
        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_use_oc},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_use_oc},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_use_oc},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, fix_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, fix_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, fix_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, fix_use_oc},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, fix_use_oc},

        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2,
                fix_use_all_offsets},

        test_params {
                'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_no_offsets},
        test_params {
                'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, fix_no_offsets},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, fix_no_offsets},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, fix_no_offsets},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, fix_no_offsets},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, fix_no_offsets},
        test_params {
                'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, fix_no_offsets});

INST_TEST_CASE(TestGEMM_general_cases_col_offset,
        test_params {'N', 'n', 30, 20, 10, 1.0, 0.0, 60, 50, 80, col_use_oc},
        test_params {'n', 'T', 30, 20, 10, 1.0, 0.0, 60, 50, 80, col_use_oc},
        test_params {'T', 'N', 30, 20, 10, 1.0, 0.0, 60, 50, 80, col_use_oc},
        test_params {'t', 't', 30, 20, 10, 1.0, 0.0, 60, 50, 80, col_use_oc},
        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_use_oc},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_use_oc},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_use_oc},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, col_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, col_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, col_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, col_use_oc},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, col_use_oc},

        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                col_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                col_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                col_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                col_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100,
                col_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100,
                col_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                col_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                col_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2,
                col_use_all_offsets},

        test_params {
                'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_no_offsets},
        test_params {
                'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, col_no_offsets},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, col_no_offsets},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, col_no_offsets},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, col_no_offsets},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, col_no_offsets},
        test_params {
                'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, col_no_offsets});

INST_TEST_CASE(TestGEMM_general_cases_row_offset,
        test_params {'N', 'n', 30, 20, 10, 1.0, 0.0, 60, 50, 80, row_use_oc},
        test_params {'n', 'T', 30, 20, 10, 1.0, 0.0, 60, 50, 80, row_use_oc},
        test_params {'T', 'N', 30, 20, 10, 1.0, 0.0, 60, 50, 80, row_use_oc},
        test_params {'t', 't', 30, 20, 10, 1.0, 0.0, 60, 50, 80, row_use_oc},
        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_use_oc},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_use_oc},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_use_oc},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, row_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, row_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, row_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, row_use_oc},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, row_use_oc},

        test_params {'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                row_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                row_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                row_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80,
                row_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100,
                row_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100,
                row_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                row_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100,
                row_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2,
                row_use_all_offsets},

        test_params {
                'N', 'n', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_no_offsets},
        test_params {
                'n', 'T', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.0, 1.0, 60, 50, 80, row_no_offsets},
        test_params {
                'n', 'n', 100, 100, 2, 1.0, 2.0, 100, 100, 100, row_no_offsets},
        test_params {
                'n', 't', 100, 2, 100, 1.0, 2.0, 100, 100, 100, row_no_offsets},
        test_params {
                't', 'n', 2, 100, 100, 1.0, 2.0, 100, 100, 100, row_no_offsets},
        test_params {
                't', 't', 2, 100, 100, 1.0, 2.0, 100, 100, 100, row_no_offsets},
        test_params {
                'n', 'n', 2, 2, 10000, 1.0, 2.0, 10000, 2, 2, row_no_offsets});

CPU_INST_TEST_CASE(TestGEMM_fractional_scales_fix_offset,
        /* alpha and beta have non-zero fractional part */
        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, fix_use_oc},
        test_params {
                'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, fix_use_oc},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, fix_use_oc},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, fix_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, fix_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, fix_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, fix_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, fix_use_oc},
        test_params {
                'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2, fix_use_oc},

        test_params {'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                fix_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80,
                fix_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                fix_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                fix_use_all_offsets},

        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, fix_no_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                fix_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, fix_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, fix_no_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                fix_no_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                fix_no_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                fix_no_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                fix_no_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                fix_no_offsets});

CPU_INST_TEST_CASE(TestGEMM_fractional_scales_col_offset,
        /* alpha and beta have non-zero fractional part */
        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, col_use_oc},
        test_params {
                'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, col_use_oc},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, col_use_oc},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, col_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, col_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, col_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, col_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, col_use_oc},
        test_params {
                'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2, col_use_oc},

        test_params {'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80,
                col_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                col_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80,
                col_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80,
                col_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                col_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                col_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                col_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                col_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                col_use_all_offsets},

        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, col_no_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                col_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, col_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, col_no_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                col_no_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                col_no_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                col_no_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                col_no_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                col_no_offsets});

CPU_INST_TEST_CASE(TestGEMM_fractional_scales_row_offset,
        /* alpha and beta have non-zero fractional part */
        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, row_use_oc},
        test_params {
                'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120, row_use_oc},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, row_use_oc},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, row_use_oc},
        test_params {
                'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100, row_use_oc},
        test_params {
                'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100, row_use_oc},
        test_params {
                't', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100, row_use_oc},
        test_params {
                't', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100, row_use_oc},
        test_params {
                'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2, row_use_oc},

        test_params {'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80,
                row_use_all_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                row_use_all_offsets},
        test_params {'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80,
                row_use_all_offsets},
        test_params {'t', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80,
                row_use_all_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                row_use_all_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                row_use_all_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                row_use_all_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                row_use_all_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                row_use_all_offsets},

        test_params {
                'n', 'T', 30, 20, 10, 2.33f, 1.66f, 60, 50, 80, row_no_offsets},
        test_params {'n', 'T', 30, 20, 10, 2.19f, 1.99f, 120, 120, 120,
                row_no_offsets},
        test_params {
                'T', 'N', 30, 20, 10, 2.01f, 1.01f, 60, 50, 80, row_no_offsets},
        test_params {
                't', 't', 30, 20, 10, 2.99f, 1.19f, 60, 50, 80, row_no_offsets},
        test_params {'n', 'n', 100, 100, 2, 1.33f, 2.33f, 100, 100, 100,
                row_no_offsets},
        test_params {'n', 't', 100, 2, 100, 1.19f, 2.99f, 100, 100, 100,
                row_no_offsets},
        test_params {'t', 'n', 2, 100, 100, 1.01f, 2.01f, 100, 100, 100,
                row_no_offsets},
        test_params {'t', 't', 2, 100, 100, 1.99f, 2.19f, 100, 100, 100,
                row_no_offsets},
        test_params {'n', 'n', 2, 2, 10000, 1.66f, 2.33f, 10000, 2, 2,
                row_no_offsets});

CPU_INST_TEST_CASE(TestGEMV,
        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 0.0f, 1000, 1, 1,
                fix_no_offsets},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 0.0f, 2000, 3000, 3000,
                fix_no_offsets},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 0.0f, 2000, 1, 1,
                fix_no_offsets},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 0.0f, 1, 3000, 3000,
                fix_no_offsets},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 0.0f, 1000, 1000, 1,
                fix_no_offsets},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 0.0f, 2000, 2000, 3000,
                fix_no_offsets},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 0.0f, 2000, 1000, 1,
                fix_no_offsets},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 0.0f, 1, 2000, 3000,
                fix_no_offsets},

        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1, 1,
                fix_no_offsets},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 1.0f, 2000, 3000, 3000,
                fix_no_offsets},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1, 1,
                fix_no_offsets},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 1.0f, 1, 3000, 3000,
                fix_no_offsets},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1000, 1,
                fix_no_offsets},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 1.0f, 2000, 2000, 3000,
                fix_no_offsets},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1000, 1,
                fix_no_offsets},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 1.0f, 1, 2000, 3000,
                fix_no_offsets},

        test_params {'n', 'n', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1, 1,
                {'F', true, false, false}},
        test_params {'n', 'n', 1, 3000, 2000, 1.0f, 1.0f, 2000, 3000, 3000,
                {'F', true, true, false}},
        test_params {'t', 'n', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1, 1,
                {'F', false, true, false}},
        test_params {'t', 'n', 1, 3000, 2000, 1.0f, 1.0f, 1, 3000, 3000,
                {'F', true, false, true}},
        test_params {'n', 't', 2000, 1, 1000, 1.0f, 1.0f, 1000, 1000, 1,
                {'F', false, true, true}},
        test_params {'n', 't', 1, 3000, 2000, 1.0f, 1.0f, 2000, 2000, 3000,
                {'F', true, true, false}},
        test_params {'t', 't', 2000, 1, 1000, 1.0f, 1.0f, 2000, 1000, 1,
                {'F', true, false, false}},
        test_params {'t', 't', 1, 3000, 2000, 1.0f, 1.0f, 1, 2000, 3000,
                {'F', false, true, false}});

CPU_INST_TEST_CASE(TestGEMV_kblocking,
        test_params {
                't', 'n', 20, 1, 7000, 1.0f, 0.0f, 20, 1, 500, fix_no_offsets},
        test_params {'t', 't', 50, 1, 7000, 1.0f, 0.0f, 50, 7000, 500,
                fix_no_offsets},
        test_params {'t', 'n', 400, 1, 7000, 1.0f, 0.0f, 400, 1, 500,
                fix_no_offsets},
        test_params {'t', 't', 500, 1, 7000, 1.0f, 0.0f, 500, 7000, 500,
                fix_no_offsets},
        test_params {
                't', 'n', 20, 1, 7000, 1.0f, 1.0f, 20, 1, 500, fix_no_offsets},
        test_params {'t', 't', 50, 1, 7000, 1.0f, 1.0f, 50, 7000, 500,
                fix_no_offsets},
        test_params {'t', 'n', 500, 1, 7000, 1.0f, 1.0f, 500, 1, 500,
                fix_no_offsets},
        test_params {'t', 't', 500, 1, 7000, 1.0f, 1.0f, 500, 7000, 500,
                fix_no_offsets},

        test_params {'n', 'n', 1, 40, 7000, 1.0f, 0.0f, 7000, 40, 500,
                fix_no_offsets},
        test_params {'t', 'n', 1, 10, 7000, 1.0f, 0.0f, 7000, 10, 10,
                fix_no_offsets},
        test_params {'n', 'n', 1, 400, 7000, 1.0f, 0.0f, 7000, 400, 500,
                fix_no_offsets},
        test_params {'t', 'n', 1, 100, 7000, 1.0f, 0.0f, 7000, 100, 500,
                fix_no_offsets},
        test_params {'n', 'n', 1, 40, 7000, 1.0f, 1.0f, 7000, 40, 500,
                fix_no_offsets},
        test_params {'t', 'n', 1, 10, 7000, 1.0f, 1.0f, 7000, 10, 500,
                fix_no_offsets},
        test_params {'n', 'n', 1, 400, 7000, 1.0f, 1.0f, 7000, 400, 500,
                fix_no_offsets},
        test_params {'t', 'n', 1, 550, 7000, 1.0f, 1.0f, 7000, 550, 550,
                fix_no_offsets});

CPU_INST_TEST_CASE(TestGEMM_packed,
        make_test_params_pack({false, true}, 'N', 'n', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_use_oc),
        make_test_params_pack({true, false}, 'n', 'T', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_use_oc),
        make_test_params_pack({true, true}, 'T', 'N', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_use_oc),
        make_test_params_pack({false, true}, 't', 't', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_use_oc),

        make_test_params_pack({false, true}, 'N', 'n', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_no_offsets),
        make_test_params_pack({true, false}, 'n', 'T', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_no_offsets),
        make_test_params_pack({true, true}, 'T', 'N', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_no_offsets),
        make_test_params_pack({false, true}, 't', 't', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, fix_no_offsets),

        make_test_params_pack({false, true}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f,
                100, 100, 100, fix_use_oc),
        make_test_params_pack({true, false}, 'n', 't', 100, 2, 100, 1.0f, 2.0f,
                100, 100, 100, fix_use_oc),
        make_test_params_pack({true, true}, 't', 'n', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, fix_use_oc),
        make_test_params_pack({false, true}, 't', 't', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, fix_use_oc),
        make_test_params_pack({true, false}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f,
                10000, 2, 2, fix_use_oc),

        make_test_params_pack({false, true}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f,
                100, 100, 100, row_use_oc),
        make_test_params_pack({true, false}, 'n', 't', 100, 2, 100, 1.0f, 2.0f,
                100, 100, 100, row_use_oc),
        make_test_params_pack({true, true}, 't', 'n', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, row_use_oc),
        make_test_params_pack({false, true}, 't', 't', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, row_use_oc),

        make_test_params_pack({false, true}, 'n', 'n', 100, 100, 2, 1.0f, 2.0f,
                100, 100, 100, row_no_offsets),
        make_test_params_pack({true, false}, 'n', 't', 100, 1, 100, 1.0f, 2.0f,
                100, 100, 100, row_no_offsets),
        make_test_params_pack({true, true}, 't', 'n', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100, row_no_offsets),
        make_test_params_pack({false, true}, 't', 't', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100, row_no_offsets),

        make_test_params_pack({false, true}, 'N', 'n', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, row_use_oc),
        make_test_params_pack({true, false}, 'n', 'T', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, row_use_oc),
        make_test_params_pack({true, true}, 'T', 'N', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, row_use_oc),
        make_test_params_pack({false, true}, 't', 't', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, row_use_oc),
        make_test_params_pack({true, true}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f,
                10000, 2, 2, row_use_oc),

        make_test_params_pack({true, false}, 'n', 't', 100, 2, 100, 1.0f, 2.0f,
                100, 100, 100, col_use_oc),
        make_test_params_pack({true, true}, 't', 'n', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, col_use_oc),
        make_test_params_pack({false, true}, 't', 't', 2, 100, 100, 1.0f, 2.0f,
                100, 100, 100, col_use_oc),
        make_test_params_pack({true, true}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f,
                10000, 2, 2, col_use_oc),

        make_test_params_pack({true, false}, 'n', 't', 100, 1, 100, 1.0f, 2.0f,
                100, 100, 100, col_no_offsets),
        make_test_params_pack({true, true}, 't', 'n', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100, col_no_offsets),
        make_test_params_pack({false, true}, 't', 't', 1, 100, 100, 1.0f, 2.0f,
                100, 100, 100, col_no_offsets),
        make_test_params_pack({true, true}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f,
                10000, 2, 2, col_no_offsets),

        make_test_params_pack({false, true}, 'N', 'n', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, col_use_oc),
        make_test_params_pack({true, false}, 'n', 'T', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, col_use_oc),
        make_test_params_pack({true, true}, 'T', 'N', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, col_use_oc),
        make_test_params_pack({false, true}, 't', 't', 30, 20, 10, 1.0f, 1.0f,
                60, 50, 80, col_use_oc),
        make_test_params_pack({true, true}, 'n', 'n', 2, 2, 10000, 1.0f, 2.0f,
                10000, 2, 2, col_use_oc),

        make_test_params_pack({false, true}, 'N', 'n', 200, 1, 200, 1.0f, 1.0f,
                200, 200, 200, fix_no_offsets),
        make_test_params_pack({true, false}, 't', 'N', 200, 1, 200, 1.0f, 0.0f,
                200, 200, 200, fix_no_offsets),
        make_test_params_pack({true, true}, 'T', 'N', 1, 200, 200, 1.0f, 1.0f,
                1, 200, 200, fix_no_offsets),
        make_test_params_pack({false, true}, 'n', 'T', 1, 200, 200, 1.0f, 0.0f,
                200, 200, 200, fix_no_offsets));

CPU_INST_TEST_CASE(TestGEMM_heavy,
        test_params {'n', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'t', 'n', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'n', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'t', 't', 3000, 3000, 3000, 1.0, 0.0, 3000, 3000, 3000,
                fix_use_oc},

        test_params {'n', 'n', 3000, 3000, 3000, 2.19f, 1.99f, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'t', 'n', 3000, 3000, 3000, 2.99f, 1.19f, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'n', 't', 3000, 3000, 3000, 1.19f, 2.99f, 3000, 3000, 3000,
                fix_use_oc},
        test_params {'t', 't', 3000, 3000, 3000, 1.99f, 2.19f, 3000, 3000, 3000,
                fix_use_oc});

CPU_INST_TEST_CASE(TestGEMM_packed_heavy,
        make_test_params_pack({false, true}, 'n', 'n', 3000, 3000, 3000, 1.0f,
                0.0f, 3000, 3000, 3000, fix_use_oc),
        make_test_params_pack({true, false}, 't', 'n', 3000, 3000, 3000, 1.0f,
                0.0f, 3000, 3000, 3000, fix_use_oc),
        make_test_params_pack({true, true}, 'n', 't', 3000, 3000, 3000, 1.0f,
                0.0f, 3000, 3000, 3000, row_use_oc),
        make_test_params_pack({true, true}, 't', 't', 3000, 3000, 3000, 1.0f,
                0.0f, 3000, 3000, 3000, row_use_oc),

        make_test_params_pack({true, true}, 'n', 'n', 2000, 5000, 2000, 1.0f,
                1.35f, 2000, 5000, 5000, col_use_oc),
        make_test_params_pack({false, true}, 't', 'n', 2000, 5000, 2000, 1.0f,
                1.77f, 2000, 5000, 5000, col_use_oc),

        make_test_params_pack({false, true}, 'n', 'n', 200, 20000, 2000, 1.0f,
                2.0f, 2000, 20000, 20000, fix_use_oc),
        make_test_params_pack({true, true}, 'n', 'n', 200, 20000, 2000, 1.0f,
                2.0f, 2000, 20000, 20000, row_use_oc),
        make_test_params_pack({true, false}, 'n', 'n', 200, 20000, 2000, 1.0f,
                2.0f, 2000, 20000, 20000, col_use_oc),

        make_test_params_pack({true, true}, 'n', 'n', 5000, 100, 2000, 1.0f,
                2.0f, 2000, 100, 100, row_use_oc),
        make_test_params_pack({false, true}, 't', 'n', 5000, 100, 2000, 1.0f,
                2.0f, 5000, 100, 100, col_use_oc),

        make_test_params_pack({true, false}, 'n', 'n', 150, 150, 8000, 1.0f,
                1.7f, 8000, 150, 150, fix_use_oc),
        make_test_params_pack({true, true}, 'n', 't', 200, 200, 8000, 1.0f,
                3.0f, 8000, 8000, 200, row_use_oc),
        make_test_params_pack({false, true}, 't', 'n', 200, 300, 8000, 1.0f,
                0.0f, 200, 300, 300, col_use_oc));

#endif
