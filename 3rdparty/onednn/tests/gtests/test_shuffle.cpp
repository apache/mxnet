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

#include <cmath>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

struct shuffle_test_params_t {
    prop_kind aprop_kind;
    memory::format_tag data_format;
    memory::dims dims;
    int axis;
    memory::dim group_size;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
void check_shuffle(const shuffle_test_params_t &p, const memory &input,
        const memory &output, memory::dim ROW) {
    auto in_ptr = map_memory<data_t>(input);
    auto out_ptr = map_memory<data_t>(output);

    const memory::desc in_d = input.get_desc();
    const memory::desc out_d = output.get_desc();

    auto dims = in_d.data.dims;
    auto ndims = in_d.data.ndims;

    const dnnl::impl::memory_desc_wrapper input_mdw(in_d.data);
    const dnnl::impl::memory_desc_wrapper output_mdw(out_d.data);

    const int axis = p.axis;
    memory::dim inner_size = 1, outer_size = 1;
    const memory::dim axis_size = dims[axis];
    const memory::dim padded_axis = in_d.data.padded_dims[axis];

    auto rev_transpose = [=](memory::dim a) {
        memory::dim COL = axis_size / ROW;
        memory::dim row = a / COL;
        memory::dim col = a % COL;
        return ROW * col + row;
    };

    for (int i = 0; i < axis; ++i)
        outer_size *= (size_t)dims[i];
    for (int i = axis + 1; i < ndims; ++i)
        inner_size *= (size_t)dims[i];
    const memory::dim dim = padded_axis * inner_size;

    dnnl::impl::parallel_nd(outer_size, axis_size, inner_size,
            [&](memory::dim ou, memory::dim a, memory::dim in) {
                if (is_current_test_failed()) return;

                data_t refout = in_ptr[input_mdw.off_l(
                        ou * dim + rev_transpose(a) * inner_size + in, true)];
                data_t out = out_ptr[output_mdw.off_l(
                        ou * dim + a * inner_size + in, true)];
                ASSERT_NEAR(out, refout, 0);
            });
}

template <typename data_t>
class shuffle_test_t : public ::testing::TestWithParam<shuffle_test_params_t> {
private:
    shuffle_forward::primitive_desc shuffle_fwd_prim_desc;
    shuffle_test_params_t p;
    memory::dims padR;
    engine eng;
    stream strm;
    memory::data_type data_type;

protected:
    void SetUp() override {
        data_type = data_traits<data_t>::data_type;
        SKIP_IF(data_type == memory::data_type::f16
                        && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support f16 data type.");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);
        data_type = data_traits<data_t>::data_type;

        bool is_training = p.aprop_kind == prop_kind::forward_training;

        Forward();
        if (is_training) Backward();
    }

    void Forward() {
        memory::desc src_desc(p.dims, data_type, p.data_format);
        memory::desc dst_desc(p.dims, data_type, p.data_format);

        auto shuffle_desc = shuffle_forward::desc(
                p.aprop_kind, src_desc, p.axis, p.group_size);
        shuffle_fwd_prim_desc
                = shuffle_forward::primitive_desc(shuffle_desc, eng);
        shuffle_fwd_prim_desc = shuffle_forward::primitive_desc(
                shuffle_fwd_prim_desc.get()); // test construction from a C pd

        ASSERT_TRUE(
                shuffle_fwd_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == shuffle_fwd_prim_desc.src_desc());
        ASSERT_TRUE(
                shuffle_fwd_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == shuffle_fwd_prim_desc.dst_desc());

        test_memory src(src_desc, eng);
        test_memory dst(dst_desc, eng);

        fill_data<data_t>(src.get_size() / sizeof(data_t), src.get());
        check_zero_tail<data_t>(1, src.get());
        check_zero_tail<data_t>(1, dst.get());

        shuffle_forward(shuffle_fwd_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src.get()}, {DNNL_ARG_DST, dst.get()}});
        strm.wait();

        check_shuffle<data_t>(p, src.get(), dst.get(), p.group_size);
    }

    void Backward() {
        memory::desc diff_dst_desc(p.dims, data_type, p.data_format);
        memory::desc diff_src_desc(p.dims, data_type, p.data_format);

        auto shuffle_desc
                = shuffle_backward::desc(diff_dst_desc, p.axis, p.group_size);

        test_memory diff_dst(diff_dst_desc, eng);
        test_memory diff_src(diff_src_desc, eng);

        auto shuffle_prim_desc = shuffle_backward::primitive_desc(
                shuffle_desc, eng, shuffle_fwd_prim_desc);
        shuffle_prim_desc = shuffle_backward::primitive_desc(
                shuffle_prim_desc.get()); // test construction from a C pd

        ASSERT_TRUE(shuffle_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == shuffle_prim_desc.diff_src_desc());
        ASSERT_TRUE(shuffle_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == shuffle_prim_desc.diff_dst_desc());

        fill_data<data_t>(diff_dst.get_size() / sizeof(data_t), diff_dst.get());

        check_zero_tail<data_t>(1, diff_dst.get());
        check_zero_tail<data_t>(1, diff_src.get());

        // Execute
        shuffle_backward(shuffle_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, diff_dst.get()},
                                {DNNL_ARG_DIFF_SRC, diff_src.get()}});
        strm.wait();

        const int axis_size = diff_dst_desc.data.dims[p.axis];
        check_shuffle<data_t>(
                p, diff_dst.get(), diff_src.get(), axis_size / p.group_size);
    }
};

using shuffle_test_float = shuffle_test_t<float>;
using shuffle_test_float16 = shuffle_test_t<float16_t>;
using shuffle_test_s8 = shuffle_test_t<int8_t>;
using shuffle_test_u8 = shuffle_test_t<uint8_t>;

#define INST_TEST_CASE(test) \
    TEST_P(test, TestsShuffle) {} \
    INSTANTIATE_TEST_SUITE_P(TestShuffle_nChw16c, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 16, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 64, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 32, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 16, 4, 4}, 1, \
                            2})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_nChw16c_Tail, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 24, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 66, 4, 4}, 1, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 34, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw16c, {2, 12, 10, 10}, 1, \
                            2})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_NCHW, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 10, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 10, 4, 4}, 1, 5})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_NCDHW, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 10, 2, 4, 4}, 2, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 10, 2, 4, 4}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 10, 2, 4, 4}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 10, 2, 4, 4}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 12, 1, 7, 7}, 1, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 12, 2, 7, 7}, 1, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 12, 3, 7, 7}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncdhw, {2, 12, 1, 7, 7}, 1, \
                            4})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffleNHWC, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nhwc, {2, 10, 4, 4}, 3, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nhwc, {2, 10, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nhwc, {2, 10, 4, 4}, 1, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nhwc, {2, 10, 4, 4}, 1, 2})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_nChw8c, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {2, 16, 4, 4}, 2, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {2, 16, 4, 4}, 2, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {2, 16, 4, 4}, 1, 8}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {2, 16, 4, 4}, 1, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {1, 8, 1, 1}, 1, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nChw8c, {1, 8, 1, 1}, 1, 2})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_nCdhw16c, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {2, 16, 2, 4, 4}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {2, 16, 2, 4, 4}, 3, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {2, 16, 2, 4, 4}, 1, \
                            8}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {2, 16, 2, 4, 4}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 16, 2, 1, 1}, 1, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 16, 2, 1, 1}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 16, 2, 1, 1}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 16, 2, 1, 1}, 1, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 32, 1, 5, 5}, 1, \
                            4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 32, 1, 5, 5}, 1, \
                            8}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 32, 1, 5, 5}, 1, \
                            2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nCdhw16c, {1, 32, 1, 15, 15}, \
                            3, 5})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_OIHW, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::oihw, {2, 16, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::oihw, {2, 64, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::oihw, {2, 32, 4, 4}, 2, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::oihw, {2, 16, 4, 4}, 1, 2})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_NC, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nc, {10, 8}, 1, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nc, {10, 8}, 1, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nc, {2, 32}, 0, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nc, {10, 32}, 0, 5})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_NCW, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncw, {10, 8, 5}, 1, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncw, {10, 8, 5}, 1, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncw, {2, 32, 5}, 0, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::ncw, {10, 32, 5}, 0, 5})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffle_X, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::x, {10}, 0, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::x, {8}, 0, 4}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::x, {2}, 0, 2}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::x, {10}, 0, 5})); \
\
    INSTANTIATE_TEST_SUITE_P(TestShuffleEF_NCHW, test, \
            ::testing::Values( \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 15, 4, 4}, 1, 2, \
                            true, dnnl_invalid_arguments}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 64, 7, 7}, 2, 2, \
                            true, dnnl_invalid_arguments}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 32, 11, 11}, 2, 2, \
                            true, dnnl_invalid_arguments}, \
                    shuffle_test_params_t {prop_kind::forward_training, \
                            memory::format_tag::nchw, {2, 16, 4, 4}, 4, 2, \
                            true, dnnl_invalid_arguments}));

INST_TEST_CASE(shuffle_test_float)
INST_TEST_CASE(shuffle_test_float16)
INST_TEST_CASE(shuffle_test_s8)
INST_TEST_CASE(shuffle_test_u8)

} // namespace dnnl
