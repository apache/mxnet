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

#ifndef TEST_CONVOLUTION_BACKWARD_DATA_COMMON_H
#define TEST_CONVOLUTION_BACKWARD_DATA_COMMON_H

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

template <typename data_t_diff_dst, typename data_t_wei, typename data_t_acc,
        typename data_t_diff_src>
void compute_ref_conv_bwd_data(const test_convolution_sizes_t &c,
        const memory &diff_src, const memory &weights, const memory &diff_dst) {
    auto diff_dst_data = map_memory<data_t_diff_dst>(diff_dst);
    auto weights_data = map_memory<data_t_wei>(weights);
    auto diff_src_data = map_memory<data_t_diff_src>(diff_src);

    const memory::desc diff_src_d = diff_src.get_desc();
    const memory::desc weights_d = weights.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();

    auto padded_ic = diff_src_d.data.padded_dims[1];
    auto padded_oc = diff_dst_d.data.padded_dims[1];

    const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);

    dnnl::impl::parallel_nd(c.mb, c.ng, c.ic / c.ng, c.ih, c.iw,
            [&](memory::dim mb, memory::dim g, memory::dim ic, memory::dim ih,
                    memory::dim iw) {
                memory::dim sidx = mb * padded_ic * c.ih * c.iw
                        + g * padded_ic / c.ng * c.ih * c.iw + ic * c.ih * c.iw
                        + ih * c.iw + iw;
                data_t_acc a = data_t_acc(0);
                for (memory::dim oc = 0; oc < c.oc / c.ng; oc++) {
                    for (memory::dim kh = 0; kh < c.kh; kh++) {
                        for (memory::dim kw = 0; kw < c.kw; kw++) {
                            if (iw + c.padw < kw * (1 + c.dilw)
                                    || ih + c.padh < kh * (1 + c.dilh))
                                continue;
                            memory::dim ow = iw - kw * (1 + c.dilw) + c.padw;
                            memory::dim oh = ih - kh * (1 + c.dilh) + c.padh;
                            if (ow % c.strw != 0 || oh % c.strh != 0) continue;
                            ow /= c.strw;
                            oh /= c.strh;
                            if (oh < c.oh && ow < c.ow) {
                                memory::dim didx = mb * padded_oc * c.oh * c.ow
                                        + g * padded_oc / c.ng * c.oh * c.ow
                                        + oc * c.oh * c.ow + oh * c.ow + ow;
                                memory::dim widx = g * padded_oc / c.ng
                                                * padded_ic / c.ng * c.kh * c.kw
                                        + oc * padded_ic / c.ng * c.kh * c.kw
                                        + ic * c.kh * c.kw + kh * c.kw + kw;

                                a += (data_t_acc)(
                                        diff_dst_data[diff_dst_mdw.off_l(
                                                didx, true)]
                                        * weights_data[weights_mdw.off_l(
                                                widx, true)]);
                            }
                        }
                    }
                }
                diff_src_data[diff_src_mdw.off_l(sidx, true)]
                        = (data_t_diff_src)a;
            });
}

template <typename data_t_diff_dst, typename data_t_wei, typename data_t_acc,
        typename data_t_diff_src>
class convolution_backward_data_test
    : public ::testing::TestWithParam<test_convolution_params_t> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<
                test_convolution_params_t>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<
                test_convolution_params_t>::GetParam();
        ASSERT_EQ(p.aalgorithm, algorithm::convolution_direct);
        auto eng = get_test_engine();
        auto strm = stream(eng);
        auto data_type_diff_src = data_traits<data_t_diff_src>::data_type;
        auto data_type_diff_dst = data_traits<data_t_diff_dst>::data_type;
        auto data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw},
                data_type_diff_src, p.formats.src_format);
        auto c_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_wei, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type_wei,
                        p.formats.weights_format);
        auto c_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow},
                data_type_diff_dst, p.formats.dst_format);
        auto c_src_desc_f = create_md({cd.mb, cd.ic, cd.ih, cd.iw},
                data_type_diff_dst, p.formats.src_format);
        auto c_dst_desc_f = create_md({cd.mb, cd.oc, cd.oh, cd.ow},
                data_type_diff_src, p.formats.dst_format);

        auto c_diff_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_diff_dst = test_memory(c_dst_desc, eng);

        memory::dims padR = {
                right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
                right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)};

        // Only true for dense format
        fill_data<data_t_wei>(
                c_weights.get_size() / sizeof(data_t_wei), c_weights.get());
        fill_data<data_t_diff_dst>(
                c_diff_dst.get_size() / sizeof(data_t_diff_dst),
                c_diff_dst.get());
        fill_data<data_t_diff_src>(
                c_diff_src.get_size() / sizeof(data_t_diff_src),
                c_diff_src.get());
        check_zero_tail<data_t_diff_dst>(1, c_diff_dst.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_diff_src>(1, c_diff_src.get());

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                p.aalgorithm, c_src_desc_f, c_weights_desc, c_dst_desc_f,
                {cd.strh, cd.strw}, {cd.dilh, cd.dilw}, {cd.padh, cd.padw},
                padR);
        auto conv_primitive_desc
                = convolution_forward::primitive_desc(conv_desc, eng);

        auto conv_bwd_data_desc = convolution_backward_data::desc(p.aalgorithm,
                c_src_desc, c_weights_desc, c_dst_desc, {cd.strh, cd.strw},
                {cd.dilh, cd.dilw}, {cd.padh, cd.padw}, padR);
        auto conv_bwd_data_primitive_desc
                = convolution_backward_data::primitive_desc(
                        conv_bwd_data_desc, eng, conv_primitive_desc);
        conv_bwd_data_primitive_desc
                = convolution_backward_data::primitive_desc(
                        conv_bwd_data_primitive_desc
                                .get()); // test construction from a C pd

        ASSERT_TRUE(conv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == conv_bwd_data_primitive_desc.diff_src_desc());
        ASSERT_TRUE(conv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == conv_bwd_data_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(conv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == conv_bwd_data_primitive_desc.weights_desc());

        convolution_backward_data(conv_bwd_data_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, c_diff_dst.get()},
                                {DNNL_ARG_WEIGHTS, c_weights.get()},
                                {DNNL_ARG_DIFF_SRC, c_diff_src.get()}});
        strm.wait();

        auto ref_memory = test::make_memory(c_src_desc, eng);
        compute_ref_conv_bwd_data<data_t_diff_dst, data_t_wei, data_t_acc,
                data_t_diff_src>(
                cd, ref_memory, c_weights.get(), c_diff_dst.get());
        check_zero_tail<data_t_diff_src>(1, ref_memory);

        compare_data<data_t_diff_src>(ref_memory, c_diff_src.get());
        check_zero_tail<data_t_diff_src>(0, c_diff_src.get());
    }
};

} // namespace dnnl
#endif
