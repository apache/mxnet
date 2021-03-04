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

#include <vector>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

namespace dnnl {

TEST(test_parallel, Test) {
    impl::parallel(0, [&](int ithr, int nthr) {
        ASSERT_LE(0, ithr);
        ASSERT_LT(ithr, nthr);
        ASSERT_LE(nthr, dnnl_get_max_threads());
    });
}

using data_t = ptrdiff_t;

struct nd_params_t {
    std::vector<ptrdiff_t> dims;
};
using np_t = nd_params_t;

class test_nd_t : public ::testing::TestWithParam<nd_params_t> {
protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        size = 1;
        for (auto &d : p.dims)
            size *= d;
        data.resize((size_t)size);
    }

    void CheckID() {
        for (ptrdiff_t i = 0; i < size; ++i)
            ASSERT_EQ(data[i], i);
    }

    nd_params_t p;
    ptrdiff_t size;
    std::vector<data_t> data;
};

class test_for_nd_t : public test_nd_t {
protected:
    void emit_for_nd(int ithr, int nthr) {
        switch ((int)p.dims.size()) {
            case 1:
                impl::for_nd(ithr, nthr, p.dims[0], [&](ptrdiff_t d0) {
                    ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                    data[d0] = d0;
                });
                break;
            case 2:
                impl::for_nd(ithr, nthr, p.dims[0], p.dims[1],
                        [&](ptrdiff_t d0, ptrdiff_t d1) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            const ptrdiff_t idx = d0 * p.dims[1] + d1;
                            data[idx] = idx;
                        });
                break;
            case 3:
                impl::for_nd(ithr, nthr, p.dims[0], p.dims[1], p.dims[2],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            const ptrdiff_t idx
                                    = (d0 * p.dims[1] + d1) * p.dims[2] + d2;
                            data[idx] = idx;
                        });
                break;
            case 4:
                impl::for_nd(ithr, nthr, p.dims[0], p.dims[1], p.dims[2],
                        p.dims[3],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            const ptrdiff_t idx
                                    = ((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                            * p.dims[3]
                                    + d3;
                            data[idx] = idx;
                        });
                break;
            case 5:
                impl::for_nd(ithr, nthr, p.dims[0], p.dims[1], p.dims[2],
                        p.dims[3], p.dims[4],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3, ptrdiff_t d4) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            ASSERT_TRUE(0 <= d4 && d4 < p.dims[4]);
                            const ptrdiff_t idx
                                    = (((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                                      * p.dims[3]
                                              + d3)
                                            * p.dims[4]
                                    + d4;
                            data[idx] = idx;
                        });
                break;
            case 6:
                impl::for_nd(ithr, nthr, p.dims[0], p.dims[1], p.dims[2],
                        p.dims[3], p.dims[4], p.dims[5],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3, ptrdiff_t d4, ptrdiff_t d5) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            ASSERT_TRUE(0 <= d4 && d4 < p.dims[4]);
                            ASSERT_TRUE(0 <= d5 && d5 < p.dims[5]);
                            const ptrdiff_t idx
                                    = ((((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                                       * p.dims[3]
                                               + d3) * p.dims[4]
                                              + d4)
                                            * p.dims[5]
                                    + d5;
                            data[idx] = idx;
                        });
                break;
            default: ASSERT_TRUE(false);
        }
    }
};

TEST_P(test_for_nd_t, Sequential) {
    emit_for_nd(0, 1);
    CheckID();
}

TEST_P(test_for_nd_t, Parallel) {
    impl::parallel(0, [&](int ithr, int nthr) { emit_for_nd(ithr, nthr); });
    CheckID();
}

CPU_INSTANTIATE_TEST_SUITE_P(Case0, test_for_nd_t,
        ::testing::Values(np_t {{0}}, np_t {{1}}, np_t {{100}}, np_t {{0, 0}},
                np_t {{1, 2}}, np_t {{10, 10}}, np_t {{0, 1, 0}},
                np_t {{1, 2, 1}}, np_t {{4, 4, 10}}, np_t {{0, 3, 0, 1}},
                np_t {{1, 1, 2, 1}}, np_t {{4, 4, 5, 2}},
                np_t {{3, 0, 3, 0, 1}}, np_t {{2, 1, 1, 2, 1}},
                np_t {{4, 1, 4, 5, 2}}, np_t {{4, 3, 0, 3, 0, 1}},
                np_t {{2, 1, 3, 1, 2, 1}}, np_t {{4, 1, 4, 3, 2, 2}}));

class test_for_nd_with_diff_types_t : public test_nd_t {};
TEST_P(test_for_nd_with_diff_types_t, Test) {
    ASSERT_EQ(p.dims.size(), 2u);

    const int D0 = (int)p.dims[0];

    ASSERT_TRUE(p.dims[1] >= 0);
    const unsigned D1 = (unsigned)p.dims[1];

    impl::for_nd(0, 1, D0, D1, [&](int d0, unsigned d1) {
        ASSERT_TRUE(0 <= d0 && d0 < D0);
        ASSERT_TRUE(0u <= d1 && d1 < D1);
        const ptrdiff_t idx = (ptrdiff_t)d0 * (ptrdiff_t)D1 + (ptrdiff_t)d1;
        data[idx] = idx;
    });

    CheckID();
}
CPU_INSTANTIATE_TEST_SUITE_P(
        Case0, test_for_nd_with_diff_types_t, ::testing::Values(np_t {{4, 9}}));

class test_parallel_nd_t : public test_nd_t {
protected:
    void emit_parallel_nd() {
        switch ((int)p.dims.size()) {
            case 1:
                impl::parallel_nd(p.dims[0], [&](ptrdiff_t d0) {
                    ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                    data[d0] = d0;
                });
                break;
            case 2:
                impl::parallel_nd(
                        p.dims[0], p.dims[1], [&](ptrdiff_t d0, ptrdiff_t d1) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            const ptrdiff_t idx = d0 * p.dims[1] + d1;
                            data[idx] = idx;
                        });
                break;
            case 3:
                impl::parallel_nd(p.dims[0], p.dims[1], p.dims[2],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            const ptrdiff_t idx
                                    = (d0 * p.dims[1] + d1) * p.dims[2] + d2;
                            data[idx] = idx;
                        });
                break;
            case 4:
                impl::parallel_nd(p.dims[0], p.dims[1], p.dims[2], p.dims[3],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            const ptrdiff_t idx
                                    = ((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                            * p.dims[3]
                                    + d3;
                            data[idx] = idx;
                        });
                break;
            case 5:
                impl::parallel_nd(p.dims[0], p.dims[1], p.dims[2], p.dims[3],
                        p.dims[4],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3, ptrdiff_t d4) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            ASSERT_TRUE(0 <= d4 && d4 < p.dims[4]);
                            const ptrdiff_t idx
                                    = (((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                                      * p.dims[3]
                                              + d3)
                                            * p.dims[4]
                                    + d4;
                            data[idx] = idx;
                        });
                break;
            case 6:
                impl::parallel_nd(p.dims[0], p.dims[1], p.dims[2], p.dims[3],
                        p.dims[4], p.dims[5],
                        [&](ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2,
                                ptrdiff_t d3, ptrdiff_t d4, ptrdiff_t d5) {
                            ASSERT_TRUE(0 <= d0 && d0 < p.dims[0]);
                            ASSERT_TRUE(0 <= d1 && d1 < p.dims[1]);
                            ASSERT_TRUE(0 <= d2 && d2 < p.dims[2]);
                            ASSERT_TRUE(0 <= d3 && d3 < p.dims[3]);
                            ASSERT_TRUE(0 <= d4 && d4 < p.dims[4]);
                            ASSERT_TRUE(0 <= d5 && d5 < p.dims[5]);
                            const ptrdiff_t idx
                                    = ((((d0 * p.dims[1] + d1) * p.dims[2] + d2)
                                                       * p.dims[3]
                                               + d3) * p.dims[4]
                                              + d4)
                                            * p.dims[5]
                                    + d5;
                            data[idx] = idx;
                        });
                break;
            default: ASSERT_TRUE(false);
        }
    }
};

TEST_P(test_parallel_nd_t, Test) {
    emit_parallel_nd();
    CheckID();
}

CPU_INSTANTIATE_TEST_SUITE_P(Case, test_parallel_nd_t,
        ::testing::Values(np_t {{0}}, np_t {{1}}, np_t {{100}}, np_t {{0, 0}},
                np_t {{1, 2}}, np_t {{10, 10}}, np_t {{0, 1, 0}},
                np_t {{1, 2, 1}}, np_t {{4, 4, 10}}, np_t {{0, 3, 0, 1}},
                np_t {{1, 1, 2, 1}}, np_t {{4, 4, 5, 2}},
                np_t {{3, 0, 3, 0, 1}}, np_t {{2, 1, 1, 2, 1}},
                np_t {{4, 1, 4, 5, 2}}, np_t {{4, 3, 0, 3, 0, 1}},
                np_t {{2, 1, 3, 1, 2, 1}}, np_t {{4, 1, 4, 3, 2, 2}}));

} // namespace dnnl
