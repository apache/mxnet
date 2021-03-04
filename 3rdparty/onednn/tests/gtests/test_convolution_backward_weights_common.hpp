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

#ifndef TEST_CONVOLUTION_BACKWARD_WEIGHTS_COMMON_H
#define TEST_CONVOLUTION_BACKWARD_WEIGHTS_COMMON_H

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

template <typename data_t_src, typename data_t_diff_dst,
        typename data_t_diff_bias>
void compute_ref_conv_bwd_bias(const test_convolution_sizes_t &c,
        const memory &diff_dst, const memory &diff_bias) {
    auto diff_bias_data = map_memory<data_t_diff_bias>(diff_bias);
    auto diff_dst_data = map_memory<data_t_diff_dst>(diff_dst);

    const memory::desc bias_d = diff_bias.get_desc();
    const memory::desc dst_d = diff_dst.get_desc();
    const dnnl::impl::memory_desc_wrapper diff_bias_mdw(bias_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(dst_d.data);

    auto padded_oc = dst_d.data.padded_dims[1];

    dnnl::impl::parallel_nd(
            c.ng, c.oc / c.ng, [&](memory::dim g, memory::dim oc) {
                memory::dim bidx = g * padded_oc / c.ng + oc;
                diff_bias_data[diff_bias_mdw.off_l(bidx, true)] = 0.0;
                for (memory::dim mb = 0; mb < c.mb; ++mb) {
                    for (memory::dim oh = 0; oh < c.oh; ++oh) {
                        for (memory::dim ow = 0; ow < c.ow; ++ow) {
                            memory::dim oidx = mb * padded_oc * c.oh * c.ow
                                    + g * padded_oc / c.ng * c.oh * c.ow
                                    + oc * c.oh * c.ow + oh * c.ow + ow;
                            diff_bias_data[diff_bias_mdw.off_l(bidx, true)]
                                    += diff_dst_data[diff_dst_mdw.off_l(
                                            oidx, true)];
                        }
                    }
                }
            });
}

template <typename data_t_src, typename data_t_diff_dst,
        typename data_t_diff_weights>
void compute_ref_conv_bwd_weights(const test_convolution_sizes_t &c,
        const memory &src, const memory &diff_dst, const memory &diff_weights) {
    auto src_data = map_memory<data_t_src>(src);
    auto diff_weights_data = map_memory<data_t_diff_weights>(diff_weights);
    auto diff_dst_data = map_memory<data_t_diff_dst>(diff_dst);

    const memory::desc src_d = src.get_desc();
    const memory::desc weights_d = diff_weights.get_desc();
    const memory::desc dst_d = diff_dst.get_desc();
    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper diff_weights_mdw(weights_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(dst_d.data);

    auto padded_ic = src_d.data.padded_dims[1];
    auto padded_oc = dst_d.data.padded_dims[1];

    dnnl::impl::parallel_nd(c.ng, c.oc / c.ng, c.ic / c.ng, c.kh, c.kw,
            [&](memory::dim g, memory::dim oc, memory::dim ic, memory::dim kh,
                    memory::dim kw) {
                memory::dim widx
                        = g * padded_oc / c.ng * padded_ic / c.ng * c.kh * c.kw
                        + oc * padded_ic / c.ng * c.kh * c.kw + ic * c.kh * c.kw
                        + kh * c.kw + kw;
                diff_weights_data[diff_weights_mdw.off_l(widx, true)] = 0.0;
                for (memory::dim mb = 0; mb < c.mb; ++mb) {
                    for (memory::dim oh = 0; oh < c.oh; ++oh) {
                        for (memory::dim ow = 0; ow < c.ow; ++ow) {
                            if (ow * c.strw + kw * (1 + c.dilw) < c.padw
                                    || oh * c.strh + kh * (1 + c.dilh) < c.padh
                                    || ow * c.strw + kw * (1 + c.dilw)
                                            >= c.iw + c.padw
                                    || oh * c.strh + kh * (1 + c.dilh)
                                            >= c.ih + c.padh)
                                continue;

                            memory::dim ih
                                    = oh * c.strh - c.padh + kh * (1 + c.dilh);
                            memory::dim iw
                                    = ow * c.strw - c.padw + kw * (1 + c.dilw);
                            memory::dim sidx = mb * padded_ic * c.ih * c.iw
                                    + g * padded_ic / c.ng * c.ih * c.iw
                                    + ic * c.ih * c.iw + ih * c.iw + iw;
                            memory::dim didx = mb * padded_oc * c.oh * c.ow
                                    + g * padded_oc / c.ng * c.oh * c.ow
                                    + oc * c.oh * c.ow + oh * c.ow + ow;

                            diff_weights_data[diff_weights_mdw.off_l(
                                    widx, true)]
                                    += src_data[src_mdw.off_l(sidx, true)]
                                    * diff_dst_data[diff_dst_mdw.off_l(
                                            didx, true)];
                        }
                    }
                }
            });
}

template <typename data_t_src, typename data_t_diff_dst,
        typename data_t_diff_weights, typename data_t_diff_bias>
class convolution_backward_weights_test
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
        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_diff_dst
                = data_traits<data_t_diff_dst>::data_type;
        memory::data_type data_type_diff_weights
                = data_traits<data_t_diff_weights>::data_type;
        memory::data_type data_type_diff_bias
                = data_traits<data_t_diff_bias>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        bool with_bias = p.formats.bias_format != memory::format_tag::undef;

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw}, data_type_src,
                p.formats.src_format);
        auto c_diff_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_diff_weights, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw},
                        data_type_diff_weights, p.formats.weights_format);
        auto c_diff_bias_desc = create_md(
                {cd.oc}, data_type_diff_bias, p.formats.bias_format);
        auto c_diff_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow},
                data_type_diff_dst, p.formats.dst_format);
        auto c_weights_desc_f = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_diff_dst, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type_diff_dst,
                        p.formats.weights_format);
        auto c_dst_desc_f = create_md({cd.mb, cd.oc, cd.oh, cd.ow},
                data_type_diff_weights, p.formats.dst_format);
        auto c_src = test_memory(c_src_desc, eng);
        auto c_diff_weights = test_memory(c_diff_weights_desc, eng);
        auto c_diff_bias = test_memory(c_diff_bias_desc, eng);
        auto c_diff_dst = test_memory(c_diff_dst_desc, eng);
        auto weights_primitive_desc_f = test_memory(c_weights_desc_f, eng);
        auto dst_primitive_desc_f = test_memory(c_dst_desc_f, eng);
        fill_data<data_t_diff_dst>(
                c_diff_dst.get_size() / sizeof(data_t_diff_dst),
                c_diff_dst.get());
        fill_data<data_t_src>(
                c_src.get_size() / sizeof(data_t_src), c_src.get());
        fill_data<data_t_diff_weights>(
                c_diff_weights.get_size() / sizeof(data_t_diff_weights),
                c_diff_weights.get());

        check_zero_tail<data_t_diff_dst>(1, c_diff_dst.get());
        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_diff_weights>(1, c_diff_weights.get());

        memory::dims padR = {
                right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
                right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)};

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                p.aalgorithm, c_src_desc, c_weights_desc_f, c_diff_bias_desc,
                c_dst_desc_f, {cd.strh, cd.strw}, {cd.dilh, cd.dilw},
                {cd.padh, cd.padw}, padR);

        auto conv_bwd_weights_desc = convolution_backward_weights::desc(
                p.aalgorithm, c_src_desc, c_diff_weights_desc, c_diff_bias_desc,
                c_diff_dst_desc, {cd.strh, cd.strw}, {cd.dilh, cd.dilw},
                {cd.padh, cd.padw}, padR);

        auto conv_primitive_desc
                = convolution_forward::primitive_desc(conv_desc, eng);

        auto conv_bwd_weights_primitive_desc
                = convolution_backward_weights::primitive_desc(
                        conv_bwd_weights_desc, eng, conv_primitive_desc);
        conv_bwd_weights_primitive_desc
                = convolution_backward_weights::primitive_desc(
                        conv_bwd_weights_primitive_desc
                                .get()); // test construction from a C pd

        ASSERT_TRUE(conv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_SRC)
                == conv_bwd_weights_primitive_desc.src_desc());
        ASSERT_TRUE(conv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == conv_bwd_weights_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(conv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS)
                == conv_bwd_weights_primitive_desc.diff_weights_desc());
        ASSERT_TRUE(conv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_BIAS)
                == conv_bwd_weights_primitive_desc.diff_bias_desc());

        convolution_backward_weights(conv_bwd_weights_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, c_diff_dst.get()},
                                {DNNL_ARG_SRC, c_src.get()},
                                {DNNL_ARG_DIFF_WEIGHTS, c_diff_weights.get()},
                                {DNNL_ARG_DIFF_BIAS, c_diff_bias.get()}});
        strm.wait();

        auto ref_diff_weights = test::make_memory(c_diff_weights_desc, eng);
        auto ref_diff_bias = test::make_memory(c_diff_bias_desc, eng);

        compute_ref_conv_bwd_weights<data_t_src, data_t_diff_dst,
                data_t_diff_weights>(
                cd, c_src.get(), c_diff_dst.get(), ref_diff_weights);
        check_zero_tail<data_t_diff_weights>(1, ref_diff_weights);
        compare_data<data_t_diff_weights>(
                ref_diff_weights, c_diff_weights.get());
        check_zero_tail<data_t_diff_weights>(1, c_diff_weights.get());

        if (with_bias) {
            compute_ref_conv_bwd_bias<data_t_src, data_t_diff_dst,
                    data_t_diff_bias>(cd, c_diff_dst.get(), ref_diff_bias);

            compare_data<data_t_diff_bias>(ref_diff_bias, c_diff_bias.get());
        }
    }
};

} // namespace dnnl
#endif
