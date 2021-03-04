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
void compute_ref_inner_product_bwd_data(int ndims,
        const test_inner_product_descr_t &ipd, const memory &diff_dst,
        const memory &weights, const memory &diff_src) {
    auto diff_dst_data = map_memory<data_t>(diff_dst);
    auto weights_data = map_memory<data_t>(weights);
    auto diff_src_data = map_memory<data_t>(diff_src);

    const memory::desc diff_dst_d = diff_dst.get_desc();
    const memory::desc weights_d = weights.get_desc();
    const memory::desc diff_src_d = diff_src.get_desc();
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);
    const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);

    bool has_spatial = ipd.kh > 1 || ipd.kw > 1;
    if (ndims == 5) has_spatial = has_spatial || ipd.kd > 1;
    auto padded_ic = diff_src_d.data.padded_dims[1];

    dnnl::impl::parallel_nd(ipd.mb, ipd.ic, [&](memory::dim n, memory::dim ic) {
        if (has_spatial) {
            for_(memory::dim kd = 0; kd < ipd.kd; ++kd)
            for_(memory::dim kh = 0; kh < ipd.kh; ++kh)
            for (memory::dim kw = 0; kw < ipd.kw; ++kw) {
                memory::dim dsidx = n * padded_ic * ipd.kd * ipd.kh * ipd.kw
                        + ic * ipd.kd * ipd.kh * ipd.kw + kd * ipd.kh * ipd.kw
                        + kh * ipd.kw + kw;
                data_t *ds = &diff_src_data[diff_src_mdw.off_l(dsidx, true)];
                *ds = data_t(0);
                for (memory::dim oc = 0; oc < ipd.oc; ++oc) {
                    memory::dim ddidx = n * ipd.oc + oc;
                    memory::dim widx = oc * padded_ic * ipd.kd * ipd.kh * ipd.kw
                            + ic * ipd.kd * ipd.kh * ipd.kw
                            + kd * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                    *ds += diff_dst_data[diff_dst_mdw.off_l(ddidx, true)]
                            * weights_data[weights_mdw.off_l(widx, true)];
                }
            }
        } else {
            memory::dim dsidx = n * ipd.ic + ic;
            data_t *ds = &diff_src_data[diff_src_mdw.off_l(dsidx, true)];
            *ds = data_t(0);
            for (memory::dim oc = 0; oc < ipd.oc; ++oc) {
                memory::dim ddidx = n * ipd.oc + oc;
                memory::dim widx = oc * ipd.ic + ic;
                *ds += diff_dst_data[diff_dst_mdw.off_l(ddidx, true)]
                        * weights_data[weights_mdw.off_l(widx, true)];
            }
        }
    });
}

struct inprod_test_params_t {
    memory::format_tag diff_src_format;
    memory::format_tag weights_format;
    memory::format_tag diff_dst_format;
    int ndims;
    test_inner_product_descr_t test_ipd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
class inner_product_test_bwd_data_t
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

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        memory::data_type data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, dnnl::memory::data_type::f32);

        memory::dims diff_src_dims = {ipd.mb, ipd.ic},
                     wei_dims = {ipd.oc, ipd.ic};
        if (has_spatial) {
            if (p.ndims == 5) {
                diff_src_dims.push_back(ipd.kd);
                wei_dims.push_back(ipd.kd);
            }
            if (p.ndims >= 4) {
                diff_src_dims.push_back(ipd.kh);
                wei_dims.push_back(ipd.kh);
            }
            if (p.ndims >= 3) {
                diff_src_dims.push_back(ipd.kw);
                wei_dims.push_back(ipd.kw);
            }
        }
        auto ip_diff_src_desc
                = create_md(diff_src_dims, data_type, p.diff_src_format);
        auto ip_weights_desc = create_md(wei_dims, data_type, p.weights_format);
        auto ip_diff_dst_desc
                = create_md({ipd.mb, ipd.oc}, data_type, p.diff_dst_format);

        // Create inner product forward (hint for backward)
        auto ip_fwd_desc = inner_product_forward::desc(prop_kind::forward,
                ip_diff_src_desc, ip_weights_desc, ip_diff_dst_desc);
        auto ip_fwd_pdesc
                = inner_product_forward::primitive_desc(ip_fwd_desc, eng);

        // Create inner product backward
        auto ip_desc = inner_product_backward_data::desc(
                ip_diff_src_desc, ip_weights_desc, ip_diff_dst_desc);

        auto ip_primitive_desc = inner_product_backward_data::primitive_desc(
                ip_desc, eng, ip_fwd_pdesc);
        ip_primitive_desc = inner_product_backward_data::primitive_desc(
                ip_primitive_desc.get()); // test construction from a C pd

        auto ip_diff_src
                = test::make_memory(ip_primitive_desc.diff_src_desc(), eng);
        auto ip_weights
                = test::make_memory(ip_primitive_desc.weights_desc(), eng);
        auto ip_diff_dst
                = test::make_memory(ip_primitive_desc.diff_dst_desc(), eng);
        auto diff_src_ref
                = test::make_memory(ip_primitive_desc.diff_src_desc(), eng);

        fill_data<data_t>(ip_diff_dst.get_desc().get_size() / sizeof(data_t),
                ip_diff_dst);
        fill_data<data_t>(
                ip_weights.get_desc().get_size() / sizeof(data_t), ip_weights);

        check_zero_tail<data_t>(1, ip_diff_dst);
        check_zero_tail<data_t>(1, ip_weights);

        ASSERT_TRUE(ip_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == ip_primitive_desc.diff_src_desc());
        ASSERT_TRUE(ip_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == ip_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(
                ip_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == ip_primitive_desc.weights_desc());

        inner_product_backward_data(ip_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, ip_diff_dst},
                                {DNNL_ARG_WEIGHTS, ip_weights},
                                {DNNL_ARG_DIFF_SRC, ip_diff_src}});
        strm.wait();

        compute_ref_inner_product_bwd_data<data_t>(
                p.ndims == 5, ipd, ip_diff_dst, ip_weights, diff_src_ref);
        check_zero_tail<data_t>(1, diff_src_ref);
        compare_data<data_t>(diff_src_ref, ip_diff_src);
        check_zero_tail<data_t>(0, ip_diff_src);
    }
};

using inner_product_test_float = inner_product_test_bwd_data_t<float>;
using inprod_test_params_float = inprod_test_params_t;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb, ic, oc, kh, kw) \
    4, { mb, ic, oc, 1, kh, kw }
#define EXPAND_SIZES_1D(mb, ic, oc, kw) \
    3, { mb, ic, oc, 1, 1, kw }

TEST_P(inner_product_test_float, TestsInnerProduct) {}

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardZeroDim,
        inner_product_test_float,
        ::testing::Values(inprod_test_params_float {memory::format_tag::any,
                memory::format_tag::any, memory::format_tag::any,
                EXPAND_SIZES_2D(0, 32, 48, 6, 6)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardDataEF,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 0, 48, 6, 6), true,
                        dnnl_invalid_arguments},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_2D(-1, 32, 48, 6, 6), true,
                        dnnl_invalid_arguments},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_2D(2, -1, 48, 6, 6), true,
                        dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardData_nCdhw8c,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::nCdhw8c,
                        memory::format_tag::aBcde8b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 9, 4, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::nCdhw8c,
                        memory::format_tag::aBcde8b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 17, 16, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::nCdhw8c,
                        memory::format_tag::aBcde8b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 29, 7, 2, 2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardData_nCdhw16c,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::nCdhw16c,
                        memory::format_tag::aBcde16b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 9, 4, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::nCdhw16c,
                        memory::format_tag::aBcde16b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 17, 16, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::nCdhw16c,
                        memory::format_tag::aBcde16b, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 29, 7, 2, 2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardData_padded,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 9, 4, 2, 2)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 17, 16, 2, 2)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 29, 7, 2, 2)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 5, 4, 2, 2)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 14, 16, 2, 2)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 33, 7, 2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardData, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_2D(2, 1024, 48, 2, 2)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::hwio, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nwc,
                        memory::format_tag::wio, memory::format_tag::nc,
                        EXPAND_SIZES_1D(2, 32, 48, 5)},
                inprod_test_params_float {memory::format_tag::ncw,
                        memory::format_tag::oiw, memory::format_tag::nc,
                        EXPAND_SIZES_1D(2, 32, 48, 5)},
                inprod_test_params_float {memory::format_tag::ncw,
                        memory::format_tag::wio, memory::format_tag::nc,
                        EXPAND_SIZES_1D(2, 32, 48, 5)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::hwio, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nhwc,
                        memory::format_tag::oihw, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nchw,
                        memory::format_tag::oihw, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::any, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw8c,
                        memory::format_tag::aBcd8b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nChw16c,
                        memory::format_tag::aBcd16b, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 48, 6, 6)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 32, 1152, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::oi, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 2, 4, 1, 1)},
                inprod_test_params_float {memory::format_tag::nc,
                        memory::format_tag::io, memory::format_tag::nc,
                        EXPAND_SIZES_2D(2, 8, 16, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestInnerProductBackwardData3D,
        inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_3D(2, 32, 48, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::any,
                        memory::format_tag::any, memory::format_tag::any,
                        EXPAND_SIZES_3D(2, 1024, 48, 2, 2, 2)},
                inprod_test_params_float {memory::format_tag::ncdhw,
                        memory::format_tag::oidhw, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 32, 48, 6, 6, 6)},
                inprod_test_params_float {memory::format_tag::ndhwc,
                        memory::format_tag::dhwio, memory::format_tag::nc,
                        EXPAND_SIZES_3D(2, 16, 48, 3, 3, 3)}));

} // namespace dnnl
