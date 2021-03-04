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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

struct test_inner_product_descr_t {
    memory::dim mb;
    memory::dim ic;
    memory::dim oc;
    memory::dim kd, kh, kw;
};

template <typename data_t>
void compute_ref_inner_product_bwd_bias(const test_inner_product_descr_t &ipd,
        const memory &diff_dst, const memory &diff_bias) {
    auto diff_bias_data = map_memory<data_t>(diff_bias);
    auto diff_dst_data = map_memory<data_t>(diff_dst);

    const memory::desc diff_bias_d = diff_bias.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();
    const dnnl::impl::memory_desc_wrapper diff_bias_mdw(diff_bias_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);

    dnnl::impl::parallel_nd(ipd.oc, [&](memory::dim oc) {
        data_t *db = &diff_bias_data[diff_bias_mdw.off_l(oc, true)];
        *db = data_t(0);
        for (memory::dim n = 0; n < ipd.mb; ++n) {
            *db += diff_dst_data[diff_dst_mdw.off_l(n * ipd.oc + oc, true)];
        }
    });
}

template <typename data_t>
void compute_ref_inner_product_bwd_weights(int ndims,
        const test_inner_product_descr_t &ipd, const memory &src,
        const memory &diff_dst, const memory &diff_weights) {
    auto src_data = map_memory<data_t>(src);
    auto diff_weights_data = map_memory<data_t>(diff_weights);
    auto diff_dst_data = map_memory<data_t>(diff_dst);

    const memory::desc src_d = src.get_desc();
    const memory::desc diff_weights_d = diff_weights.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();
    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper diff_weights_mdw(diff_weights_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);

    auto padded_ic = src_d.data.padded_dims[1];

    bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
    if (ndims == 5) has_spatial = has_spatial || ipd.kd > 1;
    dnnl::impl::parallel_nd(
            ipd.oc, ipd.ic, [&](memory::dim oc, memory::dim ic) {
                if (has_spatial) {
                    for_(memory::dim kd = 0; kd < ipd.kd; ++kd)
                    for_(memory::dim kh = 0; kh < ipd.kh; ++kh)
                    for (memory::dim kw = 0; kw < ipd.kw; ++kw) {
                        memory::dim dwidx
                                = oc * padded_ic * ipd.kd * ipd.kh * ipd.kw
                                + ic * ipd.kd * ipd.kh * ipd.kw
                                + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                        data_t *dw = &diff_weights_data[diff_weights_mdw.off_l(
                                dwidx, true)];
                        *dw = data_t(0);
                        for (memory::dim n = 0; n < ipd.mb; ++n) {
                            memory::dim ddidx = n * ipd.oc + oc;
                            memory::dim sidx
                                    = n * padded_ic * ipd.kd * ipd.kh * ipd.kw
                                    + ic * ipd.kd * ipd.kh * ipd.kw
                                    + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                            *dw += diff_dst_data[diff_dst_mdw.off_l(
                                           ddidx, true)]
                                    * src_data[src_mdw.off_l(sidx, true)];
                        }
                    }
                } else {
                    memory::dim dwidx = oc * ipd.ic + ic;
                    data_t *dw = &diff_weights_data[diff_weights_mdw.off_l(
                            dwidx, true)];
                    *dw = data_t(0);
                    for (memory::dim n = 0; n < ipd.mb; ++n) {
                        memory::dim ddidx = n * ipd.oc + oc;
                        memory::dim sidx = n * ipd.ic + ic;
                        *dw += diff_dst_data[diff_dst_mdw.off_l(ddidx, true)]
                                * src_data[src_mdw.off_l(sidx, true)];
                    }
                }
            });
}

struct inprod_test_params_t {
    memory::format_tag src_format;
    memory::format_tag diff_weights_format;
    memory::format_tag diff_bias_format;
    memory::format_tag diff_dst_format;
    int ndims;
    test_inner_product_descr_t test_ipd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
class inner_product_test_bwd_weights_t
    : public ::testing::TestWithParam<inprod_test_params_t> {
protected:
    void SetUp() override {
        auto p = ::testing::TestWithParam<inprod_test_params_t>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<inprod_test_params_t>::GetParam();
        test_inner_product_descr_t ipd = p.test_ipd;

        bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
        if (p.ndims == 5) has_spatial = has_spatial || ipd.kd > 1;

        bool with_bias = p.diff_bias_format != memory::format_tag::undef;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, dnnl::memory::data_type::f32);

        memory::dims src_dims = {ipd.mb, ipd.ic},
                     diff_wei_dims = {ipd.oc, ipd.ic};
        if (has_spatial) {
            if (p.ndims == 5) {
                src_dims.push_back(ipd.kd);
                diff_wei_dims.push_back(ipd.kd);
            }
            if (p.ndims >= 4) {
                src_dims.push_back(ipd.kh);
                diff_wei_dims.push_back(ipd.kh);
            }
            if (p.ndims >= 3) {
                src_dims.push_back(ipd.kw);
                diff_wei_dims.push_back(ipd.kw);
            }
        }
        auto ip_src_desc = create_md(src_dims, data_type, p.src_format);
        auto ip_diff_weights_desc
                = create_md(diff_wei_dims, data_type, p.diff_weights_format);
        auto ip_diff_dst_desc
                = create_md({ipd.mb, ipd.oc}, data_type, p.diff_dst_format);
        auto ip_diff_bias_desc = with_bias
                ? create_md({ipd.oc}, data_type, p.diff_bias_format)
                : create_md({}, data_type, p.diff_bias_format);

        // Create inner product forward (hint for backward)
        auto ip_fwd_desc = inner_product_forward::desc(prop_kind::forward,
                ip_src_desc, ip_diff_weights_desc, ip_diff_dst_desc);
        auto ip_fwd_pdesc
                = inner_product_forward::primitive_desc(ip_fwd_desc, eng);

        // Create inner product backward
        auto ip_desc = with_bias
                ? inner_product_backward_weights::desc(ip_src_desc,
                        ip_diff_weights_desc, ip_diff_bias_desc,
                        ip_diff_dst_desc)
                : inner_product_backward_weights::desc(
                        ip_src_desc, ip_diff_weights_desc, ip_diff_dst_desc);

        auto ip_primitive_desc = inner_product_backward_weights::primitive_desc(
                ip_desc, eng, ip_fwd_pdesc);
        ip_primitive_desc = inner_product_backward_weights::primitive_desc(
                ip_primitive_desc.get()); // test construction from a C pd

        auto ip_src = test::make_memory(ip_primitive_desc.src_desc(), eng);
        auto ip_diff_dst
                = test::make_memory(ip_primitive_desc.diff_dst_desc(), eng);
        auto ip_diff_weights
                = test::make_memory(ip_primitive_desc.diff_weights_desc(), eng);
        auto diff_weights_ref
                = test::make_memory(ip_primitive_desc.diff_weights_desc(), eng);
        auto ip_diff_bias
                = test::make_memory(ip_primitive_desc.diff_bias_desc(), eng);
        auto diff_bias_ref
                = test::make_memory(ip_primitive_desc.diff_bias_desc(), eng);

        fill_data<data_t>(
                ip_src.get_desc().get_size() / sizeof(data_t), ip_src);
        fill_data<data_t>(ip_diff_dst.get_desc().get_size() / sizeof(data_t),
                ip_diff_dst);

        check_zero_tail<data_t>(1, ip_src);
        check_zero_tail<data_t>(1, ip_diff_dst);

        ASSERT_TRUE(ip_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == ip_primitive_desc.src_desc());
        ASSERT_TRUE(ip_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == ip_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(ip_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS)
                == ip_primitive_desc.diff_weights_desc());
        ASSERT_TRUE(ip_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_BIAS)
                == ip_primitive_desc.diff_bias_desc());

        inner_product_backward_weights(ip_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, ip_diff_dst},
                                {DNNL_ARG_SRC, ip_src},
                                {DNNL_ARG_DIFF_WEIGHTS, ip_diff_weights},
                                {DNNL_ARG_DIFF_BIAS, ip_diff_bias}});
        strm.wait();

        compute_ref_inner_product_bwd_weights<data_t>(
                p.ndims, ipd, ip_src, ip_diff_dst, diff_weights_ref);
        check_zero_tail<data_t>(1, diff_weights_ref);

        compare_data<data_t>(diff_weights_ref, ip_diff_weights);

        check_zero_tail<data_t>(0, ip_diff_weights);

        if (with_bias) {
            compute_ref_inner_product_bwd_bias<data_t>(
                    ipd, ip_diff_dst, diff_bias_ref);
            compare_data<data_t>(diff_bias_ref, ip_diff_bias);
        }
    }
};

using inner_product_test_float = inner_product_test_bwd_weights_t<float>;
using inprod_test_params_float = inprod_test_params_t;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb, ic, oc, kh, kw) \
    4, { mb, ic, oc, 1, kh, kw }
#define EXPAND_SIZES_1D(mb, ic, oc, kw) \
    3, { mb, ic, oc, 1, 1, kw }

TEST_P(inner_product_test_float, TestsInnerProduct) {}

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeightsZeroDim,
        inner_product_test_float,
        ::testing::Values(inprod_test_params_float {memory::format_tag::any,
                memory::format_tag::any, memory::format_tag::any,
                memory::format_tag::any, EXPAND_SIZES_2D(0, 32, 48, 6, 6)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeightsEF,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 0, 48, 6, 6), true,
                        dnnl_invalid_arguments},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(-1, 32, 48, 6, 6), true,
                        dnnl_invalid_arguments},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, -1, 48, 6, 6), true,
                        dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeightsNoBias_padded,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 17, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 10, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 17, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 5, 15, 3, 3)}));

GPU_INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeights_padded,
        inner_product_test_float,
        ::testing::Values(inprod_test_params_float {memory::format_tag::nChw16c,
                                  memory::format_tag::aBcd16b,
                                  memory::format_tag::x, memory::format_tag::nc,
                                  EXPAND_SIZES_2D(2, 17, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 10, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 17, 5, 3, 3)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 5, 15, 3, 3)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeightsNoBias,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::undef,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::undef,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 1024, 48, 2, 2)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::hwio, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nwc,
                        memory::format_tag::wio, memory::format_tag::undef,
                        memory::format_tag::nc, EXPAND_SIZES_1D(2, 32, 48, 6)},
                inprod_test_params_float {memory::format_tag::ncw,
                        memory::format_tag::oiw, memory::format_tag::undef,
                        memory::format_tag::nc, EXPAND_SIZES_1D(2, 32, 48, 6)},
                inprod_test_params_float {memory::format_tag::ncw,
                        memory::format_tag::wio, memory::format_tag::undef,
                        memory::format_tag::nc, EXPAND_SIZES_1D(2, 32, 48, 6)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::hwio, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::oihw, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nchw,
                        memory::format_tag::oihw, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1000, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::aBcd16b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1000, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::any, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1000, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1000, 6, 6)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1152, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::undef,
                        memory::format_tag::nc, EXPAND_SIZES_2D(2, 2, 4, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::io, memory::format_tag::undef,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 8, 16, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeights,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 32, 1024, 2, 2)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::hwio, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::oihw, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nchw,
                        memory::format_tag::oihw, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1000, 6, 6)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1152, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::x,
                        memory::format_tag::nc, EXPAND_SIZES_2D(2, 2, 4, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::io, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 8, 16, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardWeights3D,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_3D(2, 32, 48, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        memory::format_tag::any,
                        EXPAND_SIZES_3D(2, 32, 1024, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::ncdhw,
                        memory::format_tag::oidhw, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 32, 48, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::nCdhw8c,
                        memory::format_tag::aBcde8b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 32, 48, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::nCdhw16c,
                        memory::format_tag::aBcde16b, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 32, 1000, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::ndhwc,
                        memory::format_tag::dhwio, memory::format_tag::x,
                        memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 16, 48, 3, 3, 3)}));
} // namespace dnnl
