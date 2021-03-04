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

#ifndef TEST_CONVOLUTION_FORWARD_COMMON_H
#define TEST_CONVOLUTION_FORWARD_COMMON_H

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include <stdint.h>
#include "oneapi/dnnl/dnnl.hpp"

#include <math.h>

namespace dnnl {

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
void compute_ref_conv_fwd(const test_convolution_sizes_t &c,
        const test_convolution_attr_t &attr, const memory::desc &src_d,
        const memory::desc &weights_d, const memory::desc &bias_d,
        const memory::desc &dst_d, const memory &src, const memory &weights,
        const memory &bias, const memory &dst) {
    const bool w_bias = bias_d.data.ndims != 0;
    auto src_data = map_memory<data_t_src>(src);
    auto weights_data = map_memory<data_t_wei>(weights);

    auto bias_data = w_bias ? map_memory<data_t_dst>(bias) : nullptr;
    auto dst_data = map_memory<data_t_dst>(dst);

    auto padded_ic = src_d.data.padded_dims[1];
    auto padded_oc = dst_d.data.padded_dims[1];

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.data);

    dnnl::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
            [&](memory::dim n, memory::dim g, memory::dim oc, memory::dim oh,
                    memory::dim ow) {
                data_t_acc a = 0;
                for (memory::dim ic = 0; ic < c.ic / c.ng; ic++) {
                    for (memory::dim kh = 0; kh < c.kh; kh++) {
                        for (memory::dim kw = 0; kw < c.kw; kw++) {
                            memory::dim iw
                                    = ow * c.strw - c.padw + kw * (1 + c.dilw);
                            memory::dim ih
                                    = oh * c.strh - c.padh + kh * (1 + c.dilh);
                            if (iw < 0 || iw >= c.iw) continue;
                            if (ih < 0 || ih >= c.ih) continue;
                            memory::dim iidx = n * padded_ic * c.ih * c.iw
                                    + g * padded_ic / c.ng * c.ih * c.iw
                                    + ic * c.ih * c.iw + ih * c.iw + iw;
                            memory::dim widx = g * padded_oc / c.ng * padded_ic
                                            / c.ng * c.kh * c.kw
                                    + oc * padded_ic / c.ng * c.kh * c.kw
                                    + ic * c.kh * c.kw + kh * c.kw + kw;
                            a += ((data_t_acc)src_data[src_mdw.off_l(
                                         iidx, true)])
                                    * weights_data[weights_mdw.off_l(
                                            widx, true)];
                        }
                    }
                }

                float a_fp = (float)a;

                a_fp += (float)(bias_data ? bias_data[bias_mdw.off_l(
                                        g * c.oc / c.ng + oc, true)]
                                          : 0);

                if (attr.oscale.is_def()) {
                    const auto &s = attr.oscale;
                    using P = test_convolution_attr_t::scale_t;
                    if (s.policy == P::policy_t::COMMON) { a_fp *= s.scale; }
                }

                a_fp = out_round<data_t_dst>(a_fp);

                memory::dim oidx = n * padded_oc * c.oh * c.ow
                        + g * padded_oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                        + oh * c.ow + ow;
                dst_data[dst_mdw.off_l(oidx, true)] = (data_t_dst)a_fp;
            });
}

template <typename data_t_src, typename data_t_wei, typename data_t_acc,
        typename data_t_dst>
class convolution_forward_test
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
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;

        test_convolution_sizes_t cd = p.sizes;

        test_convolution_attr_t attr = p.attr;
        attr.dnnl_attr_recreate();

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format_tag::undef;

        auto c_src_desc = create_md({cd.mb, cd.ic, cd.ih, cd.iw}, data_type_src,
                p.formats.src_format);
        auto c_weights_desc = cd.ng > 1
                ? create_md({cd.ng, cd.oc / cd.ng, cd.ic / cd.ng, cd.kh, cd.kw},
                        data_type_wei, p.formats.weights_format)
                : create_md({cd.oc, cd.ic, cd.kh, cd.kw}, data_type_wei,
                        p.formats.weights_format);
        auto c_dst_desc = create_md({cd.mb, cd.oc, cd.oh, cd.ow}, data_type_dst,
                p.formats.dst_format);
        auto c_bias_desc = with_bias
                ? create_md({cd.oc}, data_type_dst, p.formats.bias_format)
                : create_md({}, data_type_dst, p.formats.bias_format);

        auto c_src = test_memory(c_src_desc, eng);
        auto c_weights = test_memory(c_weights_desc, eng);
        auto c_bias = test_memory(c_bias_desc, eng);
        auto c_dst = test_memory(c_dst_desc, eng);

        // Only true for dense format
        fill_data<data_t_dst>(
                c_dst.get_size() / sizeof(data_t_dst), c_dst.get());
        fill_data<data_t_src>(
                c_src.get_size() / sizeof(data_t_src), c_src.get());
        fill_data<data_t_wei>(
                c_weights.get_size() / sizeof(data_t_wei), c_weights.get());
        if (with_bias) {
            fill_data<data_t_dst>(
                    c_bias.get_size() / sizeof(data_t_dst), c_bias.get());
        }
        check_zero_tail<data_t_src>(1, c_src.get());
        check_zero_tail<data_t_wei>(1, c_weights.get());
        check_zero_tail<data_t_dst>(1, c_dst.get());

        memory::dims padR = {
                right_padding(cd.ih, cd.oh, cd.kh, cd.padh, cd.strh, cd.dilh),
                right_padding(cd.iw, cd.ow, cd.kw, cd.padw, cd.strw, cd.dilw)};

        auto conv_desc = with_bias
                ? convolution_forward::desc(aprop_kind, p.aalgorithm,
                        c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                        {cd.strh, cd.strw}, {cd.dilh, cd.dilw},
                        {cd.padh, cd.padw}, padR)
                : convolution_forward::desc(aprop_kind, p.aalgorithm,
                        c_src_desc, c_weights_desc, c_dst_desc,
                        {cd.strh, cd.strw}, {cd.dilh, cd.dilw},
                        {cd.padh, cd.padw}, padR);

        auto conv_primitive_desc = convolution_forward::primitive_desc(
                conv_desc, attr.mkl_attr, eng);
        conv_primitive_desc = convolution_forward::primitive_desc(
                conv_primitive_desc.get()); // test construction from a C pd

        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == conv_primitive_desc.src_desc());
        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == conv_primitive_desc.dst_desc());
        ASSERT_TRUE(conv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == conv_primitive_desc.weights_desc());
        ASSERT_TRUE(
                conv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_BIAS)
                == conv_primitive_desc.bias_desc());

        convolution_forward(conv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, c_src.get()},
                                {DNNL_ARG_WEIGHTS, c_weights.get()},
                                {DNNL_ARG_BIAS, c_bias.get()},
                                {DNNL_ARG_DST, c_dst.get()}});
        strm.wait();

        auto ref_memory = test::make_memory(c_dst_desc, eng);
        compute_ref_conv_fwd<data_t_src, data_t_wei, data_t_acc, data_t_dst>(cd,
                attr, c_src_desc, c_weights_desc, c_bias_desc, c_dst_desc,
                c_src.get(), c_weights.get(), c_bias.get(), ref_memory);
        check_zero_tail<data_t_dst>(1, ref_memory);

        compare_data<data_t_dst>(ref_memory, c_dst.get());
        check_zero_tail<data_t_dst>(0, c_dst.get());
    }
};

} // namespace dnnl
#endif
