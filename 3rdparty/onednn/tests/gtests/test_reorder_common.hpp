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

#ifndef TEST_REORDER_COMMON_HPP
#define TEST_REORDER_COMMON_HPP

#include <memory>
#include <numeric>
#include <utility>
#include <type_traits>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

template <typename data_i_t, typename data_o_t>
inline void check_reorder(const memory::desc &md_i, const memory::desc &md_o,
        memory &src, memory &dst) {
    auto src_data = map_memory<data_i_t>(src);
    auto dst_data = map_memory<data_o_t>(dst);

    const auto ndims = md_i.data.ndims;
    const auto *dims = md_i.data.dims;
    const size_t nelems = std::accumulate(
            dims, dims + ndims, size_t(1), std::multiplies<size_t>());

    const dnnl::impl::memory_desc_wrapper mdw_i(md_i.data);
    const dnnl::impl::memory_desc_wrapper mdw_o(md_o.data);
    for (size_t i = 0; i < nelems; ++i) {
        data_i_t s_raw = src_data[mdw_i.off_l(i, false)];
        data_o_t s = static_cast<data_o_t>(s_raw);
        data_o_t d = dst_data[mdw_o.off_l(i, false)];
        ASSERT_EQ(s, d) << "mismatch at position " << i;
    }
}

template <typename reorder_types>
struct test_simple_params {
    memory::format_tag fmt_i;
    memory::format_tag fmt_o;
    memory::dims dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename reorder_types>
class reorder_simple_test
    : public ::testing::TestWithParam<test_simple_params<reorder_types>> {
protected:
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
    void Test() {
        test_simple_params<reorder_types> p
                = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() {
                    engine eng = get_test_engine();
                    RunTest(eng, eng);
                },
                p.expect_to_fail, p.expected_status);
    }
#endif

    void Test(engine &eng_i, engine &eng_o) {
        test_simple_params<reorder_types> p
                = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([&]() { RunTest(eng_i, eng_o); },
                p.expect_to_fail, p.expected_status);
    }

    void RunTest(engine &eng_i, engine &eng_o) {
        using data_i_t = typename reorder_types::first_type;
        using data_o_t = typename reorder_types::second_type;

        test_simple_params<reorder_types> p
                = ::testing::TestWithParam<decltype(p)>::GetParam();

        const size_t nelems = std::accumulate(p.dims.begin(), p.dims.end(),
                size_t(1), std::multiplies<size_t>());

        memory::data_type prec_i = data_traits<data_i_t>::data_type;
        memory::data_type prec_o = data_traits<data_o_t>::data_type;
        auto md_i = memory::desc(p.dims, prec_i, p.fmt_i);
        auto md_o = memory::desc(p.dims, prec_o, p.fmt_o);

        auto src = test::make_memory(md_i, eng_i);
        auto dst = test::make_memory(md_o, eng_o);

        /* initialize input data */
        const dnnl::impl::memory_desc_wrapper mdw_i(md_i.data);
        {
            auto src_data = map_memory<data_i_t>(src);
            for (size_t i = 0; i < nelems; ++i)
                src_data[mdw_i.off_l(i, false)] = data_i_t(i);
        }

        reorder::primitive_desc r_pd(
                eng_i, md_i, eng_o, md_o, primitive_attr());
        // test construction from a C pd
        r_pd = reorder::primitive_desc(r_pd.get());

        ASSERT_TRUE(r_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == r_pd.src_desc());
        ASSERT_TRUE(r_pd.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == r_pd.dst_desc());

        auto r = reorder(r_pd);
        auto strm = make_stream(r_pd.get_engine());
        r.execute(strm, src, dst);
        strm.wait();

        check_reorder<data_i_t, data_o_t>(md_i, md_o, src, dst);
        check_zero_tail<data_o_t>(0, dst);
    }
};

} // namespace dnnl

#endif
