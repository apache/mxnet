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

#include <cstring>
#include <memory>
#include <vector>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#define DEBUG_TEST_MEMORY_DESC_OPS_CPP 0

namespace dnnl {
namespace memory_desc_ops {

namespace debug {
#if DEBUG_TEST_MEMORY_DESC_OPS_CPP
template <typename T>
void print_vec(const char *str, const T *vec, int size) {
    printf("%s", str);
    for (int d = 0; d < size; ++d)
        printf("%d ", (int)vec[d]);
    printf("\n");
}
void print_md(const char *str, const dnnl_memory_desc_t &md) {
    const auto &o_bd = md.format_desc.blocking;

    printf("%s\n", str);

    print_vec("\tdims : ", md.dims, md.ndims);
    print_vec("\tpdims: ", md.padded_dims, md.ndims);
    print_vec("\toffs : ", md.padded_offsets, md.ndims);
    print_vec("\tstrs : ", o_bd.strides, md.ndims);

    printf("\t\tnblks : %d\n", o_bd.inner_nblks);
    print_vec("\t\tidxs  : ", o_bd.inner_idxs, o_bd.inner_nblks);
    print_vec("\t\tblks  : ", o_bd.inner_blks, o_bd.inner_nblks);
}
#else // DEBUG_TEST_MEMORY_DESC_OPS_CPP
template <typename T>
void print_vec(const char *, const T *, int) {}
void print_md(const char *, const dnnl_memory_desc_t &) {}
#endif // DEBUG_TEST_MEMORY_DESC_OPS_CPP
void print_md(const char *str, const memory::desc &md) {
    print_md(str, md.data);
}
} // namespace debug

// A proxy to memory::desc with fixed data type (f32)
struct memory_desc_proxy_t {
    memory::desc md;
    memory_desc_proxy_t() = default;
    memory_desc_proxy_t(const memory::desc &md) : md(md) {}

    memory_desc_proxy_t(const memory::dims &dims, memory::format_tag tag)
        : md(dims, memory::data_type::f32, tag) {}
    memory_desc_proxy_t(const memory::dims &dims, const memory::dims &strides)
        : md(dims, memory::data_type::f32, strides) {}

    memory_desc_proxy_t(const memory::dims &dims, const memory::dims &strides,
            const memory::dims &padded_dims)
        : md(dims, memory::data_type::f32, strides) {
        for (int d = 0; d < md.data.ndims; ++d)
            md.data.padded_dims[d] = padded_dims[d];
    }
    memory_desc_proxy_t(const memory::dims &dims, memory::format_kind fmt_kind)
        : md(dims, memory::data_type::f32, memory::format_tag::any) {
        md.data.format_kind = static_cast<dnnl_format_kind_t>(fmt_kind);
    }
};

enum test_direction_t { BI_DIRECTION = 0 /* default */, UNI_DIRECTION = 1 };

namespace properties {

using fmt = dnnl::memory::format_tag;

TEST(memory_desc_properties_test, TestMemoryDescSize) {
    auto md1_simple = memory_desc_proxy_t {{1, 1, 1, 1}, {1, 1, 1, 1}}.md;
    auto md1_strided = memory_desc_proxy_t {{1, 1, 1, 1}, {8, 4, 2, 1}}.md;
    auto md2_blocked = memory_desc_proxy_t {{1, 4, 1, 1}, fmt::nChw8c}.md;

    ASSERT_EQ(md1_simple, md1_strided);
    ASSERT_NE(md2_blocked, md1_simple);
    ASSERT_NE(md2_blocked, md1_strided);

    ASSERT_EQ(md1_simple.get_size(), 1 * sizeof(float));
    ASSERT_EQ(md1_strided.get_size(), 1 * sizeof(float));
    ASSERT_EQ(md2_blocked.get_size(), 8 * sizeof(float));
}

} // namespace properties

namespace reshape {

struct params_t {
    memory_desc_proxy_t in;
    memory_desc_proxy_t out;
    test_direction_t test_direction;
    dnnl_status_t expected_status;
};

class reshape_test_t : public ::testing::TestWithParam<params_t> {
protected:
    void Test(const memory::desc &in_md, const memory::desc &out_md) {
        memory::desc get_out_md = in_md.reshape(out_md.dims());

        debug::print_md("in_md", in_md);
        debug::print_md("out_md", get_out_md);
        debug::print_md("expect_out_md", out_md);

        ASSERT_EQ(get_out_md, out_md);
    }
};
TEST_P(reshape_test_t, TestsReshape) {
    params_t p = ::testing::TestWithParam<decltype(p)>::GetParam();
    catch_expected_failures([=]() { Test(p.in.md, p.out.md); },
            p.expected_status != dnnl_success, p.expected_status);
    if (p.test_direction == UNI_DIRECTION) return;
    catch_expected_failures([=]() { Test(p.out.md, p.in.md); },
            p.expected_status != dnnl_success, p.expected_status);
}

using fmt = dnnl::memory::format_tag;

// clang-format off
auto cases_expect_to_fail = ::testing::Values(
        // volume mismatch
        params_t {{{2, 2, 1, 1}, fmt::abcd}, {{2, 2, 2, 1, 1}, fmt::abcde}, BI_DIRECTION, dnnl_invalid_arguments},
        // volume mismatch
        params_t {{{2, 1}, {1, 1}}, {{2, 1, 2}, {2, 2, 1}}, BI_DIRECTION, dnnl_invalid_arguments},
        // volume mismatch
        params_t {{{6, 2}, fmt::ab}, {{6}, fmt::a}, BI_DIRECTION, dnnl_invalid_arguments},
        // joining axes are not contiguous in memory (`cdab` would be oK)
        params_t {{{2, 3, 0, 2}, fmt::cdba}, {{6, 0, 2}, fmt::bca}, UNI_DIRECTION, dnnl_invalid_arguments},
        // joining axes are not contiguous in memory
        params_t {{{6, 2}, fmt::ba}, {{12}, fmt::a}, UNI_DIRECTION, dnnl_invalid_arguments},
        // joining axes are not contiguous in memory (strides {2, 1} would be oK)
        params_t {{{6, 2}, {3, 1}}, {{12}, fmt::a}, UNI_DIRECTION, dnnl_invalid_arguments},
        // removing an axis of size `1` that has padding is not allowed
        params_t {{{6, 1, 2}, {4, 2, 1}, {6, 2, 2}}, {{6, 2}, fmt::any}, UNI_DIRECTION, dnnl_invalid_arguments},
        // joining axes where one has padding is not allowed
        params_t {{{6, 2, 2}, {6, 2, 1}, {6, 3, 2}}, {{6, 4}, fmt::any}, UNI_DIRECTION, dnnl_invalid_arguments},
        // splitting an axis that has padding is not allowed
        params_t {{{6}, {1}, {12}}, {{2, 3}, fmt::any}, UNI_DIRECTION, dnnl_invalid_arguments},
        // joining axes are not contiguous (partially, due to the blocking)
        params_t {{{2, 8, 3, 4}, fmt::aBcd8b}, {{2, 8 * 3 * 4}, fmt::ab}, UNI_DIRECTION, dnnl_invalid_arguments},
        // nothing can be done with zero memory desc
        params_t {{}, {}, UNI_DIRECTION, dnnl_invalid_arguments},
        // invalid format kind
        params_t {{{2, 2, 1, 1}, memory::format_kind::wino}, {{2, 2}, fmt::any}, UNI_DIRECTION, dnnl_invalid_arguments},
        // invalid format kind
        params_t {{{2, 2, 1, 1}, memory::format_kind::undef}, {{2, 2}, fmt::any}, UNI_DIRECTION, dnnl_invalid_arguments},
        // run-time dims are not supported
        params_t {{{DNNL_RUNTIME_DIM_VAL}, {1}}, {{DNNL_RUNTIME_DIM_VAL}, {1}}, UNI_DIRECTION, dnnl_invalid_arguments}
        );

auto cases_zero_dim = ::testing::Values(
        params_t {{{2, 0, 2}, fmt::abc}, {{2, 0, 2, 1}, fmt::abcd}},
        params_t {{{2, 0, 2}, fmt::abc}, {{2, 0, 1, 2, 1}, fmt::abcde}},
        params_t {{{2, 1, 0, 2}, fmt::abcd}, {{2, 0, 2, 1}, fmt::abcd}},
        params_t {{{31, 1, 0, 2}, fmt::Abcd16a}, {{1, 31, 0, 2, 1}, fmt::aBcde16b}},
        params_t {{{2, 3, 0, 2}, {6, 2, 2, 1}}, {{6, 0, 2}, {2, 2, 1}}}
        );

auto cases_generic = ::testing::Values(
        // add and/or remove axes of size `1`
        params_t {{{2, 1}, {2, 2}}, {{2}, {2}}},
        params_t {{{2, 1}, {2, 2}}, {{2, 1, 1}, {2, 2, 1}}},
        params_t {{{2, 1}, {2, 2}}, {{2, 1, 1}, {2, 1, 2}}},
        params_t {{{2, 1}, {2, 2}}, {{2, 1, 1}, {2, 2, 2}}},
        params_t {{{2, 2}, fmt::ab}, {{2, 2, 1}, fmt::abc}},
        params_t {{{2, 1}, fmt::ab}, {{1, 2, 1, 1}, fmt::abcd}},
        params_t {{{1, 2, 1}, fmt::abc}, {{2}, fmt::a}},
        params_t {{{3, 4, 5, 6}, fmt::ABcd16b16a}, {{1, 3, 4, 5, 6}, fmt::aBCde16c16b}},
        // UNI_DIRECTION due to ambiguity of adding 1, where there is already another axes of size 1
        params_t {{{2, 1, 1}, {2, 1, 1}, {2, 2, 1}}, {{2, 1}, {2, 1}, {2, 2}}, UNI_DIRECTION},
        params_t {{{2, 1, 1}, {2, 2, 1}, {2, 1, 2}}, {{2, 1}, {2, 1}, {2, 2}}},
        // split and join axes (as test_direction == BI_DIRECTION)
        params_t {{{6, 2}, fmt::ab}, {{3, 2, 2}, fmt::abc}},
        params_t {{{6, 2}, fmt::ab}, {{2, 3, 2}, fmt::abc}},
        params_t {{{6, 2}, fmt::ba}, {{2, 3, 2}, /* fmt::cab: */ {3, 1, 6}}},
        params_t {{{6, 2}, {4, 1}, {6, 4}}, {{2, 3, 2}, {12, 4, 1}, {2, 3, 4}}},
        params_t {{{1, 15, 12}, fmt::aBc8b}, {{1, 15, 3, 4}, fmt::aBcd8b}},
        params_t {{{1, 15, 12}, fmt::aBc8b}, {{1, 15, 2, 3, 2}, fmt::aBcde8b}},
        // combined cases
        params_t {{{15, 3, 4}, fmt::abc}, {{3, 5, 6, 1, 2}, fmt::abcde}},
        params_t {{{15, 3, 4}, fmt::bca}, {{3, 5, 6, 1, 2}, /* fmt::cdeab */ {5, 1, 30, 30, 15}}}
        );
// clang-format on

INSTANTIATE_TEST_SUITE_P(TestReshapeEF, reshape_test_t, cases_expect_to_fail);
INSTANTIATE_TEST_SUITE_P(TestReshapeZeroDim, reshape_test_t, cases_zero_dim);
INSTANTIATE_TEST_SUITE_P(TestReshapeOK, reshape_test_t, cases_generic);

} // namespace reshape

namespace permute_axes {

struct params_t {
    memory_desc_proxy_t in;
    memory_desc_proxy_t out;
    std::vector<int> perm;
    test_direction_t test_direction;
    dnnl_status_t expected_status;
};

class permute_axes_test_t : public ::testing::TestWithParam<params_t> {
protected:
    void Test(const memory::desc &in_md, const memory::desc &out_md,
            const std::vector<int> &perm) {
        memory::desc get_out_md = in_md.permute_axes(perm);

        debug::print_md("in_md", in_md);
        debug::print_vec("perm : ", perm.data(), (int)perm.size());
        debug::print_md("out_md", get_out_md);
        debug::print_md("expect_out_md", out_md);

        ASSERT_EQ(get_out_md, out_md);
    }
};
TEST_P(permute_axes_test_t, TestsPermuteAxes) {
    params_t p = ::testing::TestWithParam<decltype(p)>::GetParam();
    catch_expected_failures([=]() { Test(p.in.md, p.out.md, p.perm); },
            p.expected_status != dnnl_success, p.expected_status);
    if (p.test_direction == UNI_DIRECTION) return;

    std::vector<int> inv_perm(p.perm.size());
    for (int i = 0; i < (int)p.perm.size(); ++i)
        inv_perm[p.perm[i]] = i;
    catch_expected_failures([=]() { Test(p.out.md, p.in.md, inv_perm); },
            p.expected_status != dnnl_success, p.expected_status);
}

using fmt = dnnl::memory::format_tag;

// clang-format off
auto cases_expect_to_fail = ::testing::Values(
        // incorrect permutation
        params_t {{{2, 2, 1, 1}, fmt::abcd}, {{2, 2, 1, 1}, fmt::abcd}, {0, 1, 2, 2}, UNI_DIRECTION, dnnl_invalid_arguments},
        // incorrect permutation
        params_t {{{2, 2, 1, 1}, fmt::abcd}, {{2, 2, 1, 1}, fmt::abcd}, {0, 1, 2, 4}, UNI_DIRECTION, dnnl_invalid_arguments},
        // incorrect permutation
        params_t {{{2, 2, 1, 1}, fmt::abcd}, {{2, 2, 1, 1}, fmt::abcd}, {0, 1, 2, -1}, UNI_DIRECTION, dnnl_invalid_arguments},
        // nothing can be done with zero memory desc
        params_t {{}, {}, {}, UNI_DIRECTION, dnnl_invalid_arguments},
        // invalid format kind
        params_t {{{2, 2, 1, 1}, memory::format_kind::wino}, {{2, 2, 1, 1}, fmt::any}, {1, 2, 3, 4}, UNI_DIRECTION, dnnl_invalid_arguments},
        // invalid format kind
        params_t {{{2, 2, 1, 1}, memory::format_kind::undef}, {{2, 2, 1, 1}, fmt::any}, {1, 2, 3, 4}, UNI_DIRECTION, dnnl_invalid_arguments},
        // run-time dims are not supported
        params_t {{{DNNL_RUNTIME_DIM_VAL}, {1}}, {{DNNL_RUNTIME_DIM_VAL}, {1}}, {0}, UNI_DIRECTION, dnnl_invalid_arguments}
        );

auto cases_generic = ::testing::Values(
        params_t {{{2, 1}, fmt::ab}, {{2, 1}, fmt::ab}, {0, 1}},
        params_t {{{2, 1}, fmt::ab}, {{1, 2}, fmt::ba}, {1, 0}},
        params_t {{{2, 1}, fmt::ba}, {{1, 2}, fmt::ab}, {1, 0}},
        params_t {{{2, 3}, {4, 1}, {2, 4}}, {{3, 2}, {1, 4}, {4, 2}}, {1, 0}},
        params_t {{{3, 2}, {2, 30}}, {{2, 3}, {30, 2}}, {1, 0}},
        params_t {{{2, 3, 4, 5}, fmt::acdb}, {{2, 4, 5, 3}, fmt::abcd}, {0, 3, 1, 2}},
        params_t {{{2, 3, 4, 5}, fmt::cdba}, {{4, 5, 3, 2}, fmt::abcd}, {3, 2, 0, 1}},
        params_t {{{2, 15, 3, 4}, fmt::ABcd16b16a}, {{15, 2, 3, 4}, fmt::BAcd16a16b}, {1, 0, 2, 3}},
        params_t {{{3, 2, 15, 3, 4, 5}, fmt::aBCdef16b16c}, {{3, 15, 2, 3, 4, 5}, fmt::aCBdef16c16b}, {0, 2, 1, 3, 4, 5}}
        );
// clang-format on

INSTANTIATE_TEST_SUITE_P(
        TestPermuteAxesEF, permute_axes_test_t, cases_expect_to_fail);
INSTANTIATE_TEST_SUITE_P(TestPermuteAxesOK, permute_axes_test_t, cases_generic);

} // namespace permute_axes

} // namespace memory_desc_ops
} // namespace dnnl
