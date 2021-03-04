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

struct test_pool_desc_t {
    memory::dim mb, c;
    memory::dim id, ih, iw;
    memory::dim od, oh, ow;
    memory::dim kd, kh, kw;
    memory::dim dd, dh, dw;
    memory::dim padf, padt, padl;
    memory::dim strd, strh, strw;
};

struct pool_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag src_format;
    memory::format_tag dst_format;
    int ndims;
    test_pool_desc_t test_pd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
void check_pool_fwd(const pool_test_params_t &p, const memory &src,
        const memory &dst, const memory &ws) {
    auto src_data = map_memory<data_t>(src);
    auto dst_data = map_memory<data_t>(dst);
    auto ws_data_ptr = map_memory<unsigned char>(ws);

    auto ws_data = [&](size_t idx) -> int {
        auto w = (const unsigned char *)ws_data_ptr;
        if (w == nullptr) return -1;
        if (ws.get_desc().data.data_type == dnnl_u8)
            return (int)w[idx];
        else
            return ((const int *)w)[idx];
    };

    const memory::desc src_d = src.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const memory::desc ws_d = ws.get_desc();

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);
    const dnnl::impl::memory_desc_wrapper ws_mdw(ws_d.data);

    auto pd = p.test_pd;
    size_t padded_c = src_d.data.padded_dims[1];

    dnnl::impl::parallel_nd(pd.mb, pd.c, pd.od, pd.oh, pd.ow,
            [&](memory::dim n, memory::dim c, memory::dim od, memory::dim oh,
                    memory::dim ow) {
                if (is_current_test_failed()) return;

                memory::dim oidx = n * padded_c * pd.od * pd.oh * pd.ow
                        + c * pd.od * pd.oh * pd.ow + od * pd.oh * pd.ow
                        + oh * pd.ow + ow;
                data_t out = dst_data[dst_mdw.off_l(oidx, true)];
                int out_index = -1;
                if (p.aalgorithm == algorithm::pooling_max
                        && p.aprop_kind == prop_kind::forward_training) {
                    out_index = ws_data(ws_mdw.off_l(oidx, true));
                }
                // match implementation for pooling_max: padding
                // is done with lowest value and not zero, it
                // affects the case when kernel slips into
                // the padding area entirely
                typename acc_t<data_t>::type acc_ref
                        = (p.aalgorithm == algorithm::pooling_max)
                        ? std::numeric_limits<data_t>::lowest()
                        : data_t(0);
                int out_ref_index = 0;
                bool is_initialized = false;
                int num_summands = 0;

                for_(memory::dim kd = 0; kd < pd.kd; ++kd)
                for_(memory::dim kh = 0; kh < pd.kh; ++kh)
                for (memory::dim kw = 0; kw < pd.kw; ++kw) {
                    const memory::dim id
                            = od * pd.strd - pd.padf + kd * (pd.dd + 1);
                    const memory::dim ih
                            = oh * pd.strh - pd.padt + kh * (pd.dh + 1);
                    const memory::dim iw
                            = ow * pd.strw - pd.padl + kw * (pd.dw + 1);

                    if (id < 0 || id >= pd.id) continue;
                    if (ih < 0 || ih >= pd.ih) continue;
                    if (iw < 0 || iw >= pd.iw) continue;

                    size_t iidx = (size_t)n * padded_c * pd.id * pd.ih * pd.iw
                            + (size_t)c * pd.id * pd.ih * pd.iw
                            + (size_t)id * pd.ih * pd.iw + (size_t)ih * pd.iw
                            + iw;

                    data_t d = src_data[src_mdw.off_l(iidx, true)];
                    if (p.aalgorithm == algorithm::pooling_max) {
                        if (!is_initialized) {
                            acc_ref = d;
                            out_ref_index = (int)(kd * pd.kw * pd.kh
                                    + kh * pd.kw + kw);
                            is_initialized = true;
                        } else {
                            if (acc_ref < d) {
                                acc_ref = d;
                                out_ref_index = (int)(kd * pd.kw * pd.kh
                                        + kh * pd.kw + kw);
                            }
                        }
                    } else if (p.aalgorithm
                                    == algorithm::pooling_avg_include_padding
                            || p.aalgorithm
                                    == algorithm::pooling_avg_exclude_padding) {
                        acc_ref += d;
                        num_summands++;
                    }
                }

                if (p.aalgorithm == algorithm::pooling_avg_include_padding) {
                    num_summands = pd.kw * pd.kh * pd.kd;
                }

                if ((p.aalgorithm == algorithm::pooling_avg_include_padding
                            || p.aalgorithm
                                    == algorithm::pooling_avg_exclude_padding)
                        && num_summands) {
                    acc_ref = out_round<data_t>((float)acc_ref / num_summands);
                }

                const data_t out_ref = (data_t)acc_ref;
                ASSERT_NEAR(out, out_ref, 1e-6);
                if (p.aalgorithm == algorithm::pooling_max
                        && p.aprop_kind == prop_kind::forward_training) {
                    ASSERT_EQ(out_index, out_ref_index)
                            << " n = " << n << " c = " << c << " od = " << od
                            << " oh = " << oh << " ow = " << ow;
                }
            });
}

template <typename data_t>
class pooling_test_t : public ::testing::TestWithParam<pool_test_params_t> {
    pool_test_params_t p;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    template <typename prim_desc>
    void check_prim_desc(prim_desc pool_prim_desc) {
        ASSERT_TRUE(pool_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == pool_prim_desc.src_desc());
        ASSERT_TRUE(pool_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == pool_prim_desc.dst_desc());
        ASSERT_TRUE(
                pool_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == pool_prim_desc.workspace_desc());
    }

    void Test() {
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_pool_desc_t pd = p.test_pd;
        auto p_src_desc = (p.ndims == 5)
                ? create_md({pd.mb, pd.c, pd.id, pd.ih, pd.iw}, data_type,
                        p.src_format)
                : create_md(
                        {pd.mb, pd.c, pd.ih, pd.iw}, data_type, p.src_format);
        auto p_dst_desc = (p.ndims == 5)
                ? create_md({pd.mb, pd.c, pd.od, pd.oh, pd.ow}, data_type,
                        p.dst_format)
                : create_md(
                        {pd.mb, pd.c, pd.oh, pd.ow}, data_type, p.dst_format);

        auto p_src = test::make_memory(p_src_desc, eng);
        auto p_dst = test::make_memory(p_dst_desc, eng);

        fill_data<data_t>(
                p_src.get_desc().get_size() / sizeof(data_t), p_src, 1., true);
        fill_data<data_t>(
                p_dst.get_desc().get_size() / sizeof(data_t), p_dst, 1., true);
        check_zero_tail<data_t>(1, p_src);
        check_zero_tail<data_t>(1, p_dst);

        memory::dims strides, ker, dilation, pad_l, pad_r;
        if (p.ndims == 5) {
            strides = memory::dims({pd.strd, pd.strh, pd.strw});
            ker = memory::dims({pd.kd, pd.kh, pd.kw});
            dilation = memory::dims({pd.dd, pd.dh, pd.dw});
            pad_l = memory::dims({pd.padf, pd.padt, pd.padl});
            pad_r = memory::dims({right_padding(pd.id, pd.od, pd.kd, pd.padf,
                                          pd.strd, pd.dd),
                    right_padding(pd.ih, pd.oh, pd.kh, pd.padt, pd.strh, pd.dh),
                    right_padding(
                            pd.iw, pd.ow, pd.kw, pd.padl, pd.strw, pd.dw)});
        } else {
            strides = memory::dims({pd.strh, pd.strw});
            ker = memory::dims({pd.kh, pd.kw});
            dilation = memory::dims({pd.dh, pd.dw});
            pad_l = memory::dims({pd.padt, pd.padl});
            pad_r = memory::dims({right_padding(pd.ih, pd.oh, pd.kh, pd.padt,
                                          pd.strh, pd.dh),
                    right_padding(
                            pd.iw, pd.ow, pd.kw, pd.padl, pd.strw, pd.dw)});
        }

        memory workspace;

        if (pd.dd == 0 && pd.dh == 0 && pd.dw == 0) {
            auto pool_desc = pooling_forward::desc(p.aprop_kind, p.aalgorithm,
                    p_src_desc, p_dst_desc, strides, ker, pad_l, pad_r);
            auto pool_prim_desc
                    = pooling_forward::primitive_desc(pool_desc, eng);
            // test construction from a C pd
            pool_prim_desc
                    = pooling_forward::primitive_desc(pool_prim_desc.get());

            check_prim_desc<pooling_forward::primitive_desc>(pool_prim_desc);

            auto workspace_desc = pool_prim_desc.workspace_desc();
            workspace = test::make_memory(workspace_desc, eng);

            pooling_forward(pool_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_SRC, p_src}, {DNNL_ARG_DST, p_dst},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        } else {
            auto pool_desc = pooling_v2_forward::desc(p.aprop_kind,
                    p.aalgorithm, p_src_desc, p_dst_desc, strides, ker,
                    dilation, pad_l, pad_r);
            auto pool_prim_desc
                    = pooling_v2_forward::primitive_desc(pool_desc, eng);
            // test construction from a C pd
            pool_prim_desc
                    = pooling_v2_forward::primitive_desc(pool_prim_desc.get());

            check_prim_desc<pooling_v2_forward::primitive_desc>(pool_prim_desc);

            auto workspace_desc = pool_prim_desc.workspace_desc();
            workspace = test::make_memory(workspace_desc, eng);

            pooling_v2_forward(pool_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_SRC, p_src}, {DNNL_ARG_DST, p_dst},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        }
        strm.wait();

        check_pool_fwd<data_t>(p, p_src, p_dst, workspace);
        check_zero_tail<data_t>(0, p_dst);
    }
};

using pooling_test_float = pooling_test_t<float>;
using pooling_test_s8 = pooling_test_t<int8_t>;
using pooling_test_u8 = pooling_test_t<uint8_t>;
using pooling_test_s32 = pooling_test_t<int32_t>;
using pool_test_params_float = pool_test_params_t;

//#define EXPAND_NDIMS(ndims, ...) ndims, {__VA_ARGS__}

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D( \
        mb, ic, ih, iw, oh, ow, kh, kw, dh, dw, padt, padl, strh, strw) \
    4, { \
        mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, dh, dw, 0, padt, padl, 1, \
                strh, strw \
    }

#define GPU_INST_TEST_CASE(test, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardSlipsToPadding, test, \
            ::testing::Values( \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_max, \
                            memory::format_tag::NChw16n16c, \
                            memory::format_tag::NChw16n16c, \
                            EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, \
                                    0, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_exclude_padding, \
                            memory::format_tag::NChw16n16c, \
                            memory::format_tag::NChw16n16c, \
                            EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, \
                                    0, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_include_padding, \
                            memory::format_tag::NChw16n16c, \
                            memory::format_tag::NChw16n16c, \
                            EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, \
                                    0, 1, 1, 1, 1)})); \
    GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForward_gpu_3D, test, \
            ::testing::Values( \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_max, memory::format_tag::ncdhw, \
                            memory::format_tag::ncdhw, \
                            EXPAND_SIZES_3D(4, 16, 10, 10, 10, 10, 10, 10, 2, \
                                    2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_exclude_padding, \
                            memory::format_tag::ncdhw, \
                            memory::format_tag::ncdhw, \
                            EXPAND_SIZES_3D(4, 16, 10, 10, 10, 10, 10, 10, 2, \
                                    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_include_padding, \
                            memory::format_tag::ncdhw, \
                            memory::format_tag::ncdhw, \
                            EXPAND_SIZES_3D(4, 16, 10, 10, 10, 10, 10, 10, 2, \
                                    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_max, \
                            memory::format_tag::NCdhw16n16c, \
                            memory::format_tag::NCdhw16n16c, \
                            EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_exclude_padding, \
                            memory::format_tag::NCdhw16n16c, \
                            memory::format_tag::NCdhw16n16c, \
                            EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_include_padding, \
                            memory::format_tag::NCdhw16n16c, \
                            memory::format_tag::NCdhw16n16c, \
                            EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_max, \
                            memory::format_tag::nCdhw16c, \
                            memory::format_tag::nCdhw16c, \
                            EXPAND_SIZES_3D(13, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_exclude_padding, \
                            memory::format_tag::nCdhw16c, \
                            memory::format_tag::nCdhw16c, \
                            EXPAND_SIZES_3D(13, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}, \
                    pool_test_params_t {prop_kind::forward_inference, \
                            algorithm::pooling_avg_include_padding, \
                            memory::format_tag::nCdhw16c, \
                            memory::format_tag::nCdhw16c, \
                            EXPAND_SIZES_3D(13, 32, 14, 14, 14, 14, 14, 14, 3, \
                                    3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)}));

TEST_P(pooling_test_s8, TestsPooling) {}

INSTANTIATE_TEST_SUITE_P(TestPoolingAlexnetForwardS8, pooling_test_s8,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(1, 256, 13, 13, 6, 6, 3, 3, 1, 1, 0, 0,
                                2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxS8, pooling_test_s8,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardAvgS8, pooling_test_s8,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 3, 3, 0,
                                0, 2, 2)}));

GPU_INST_TEST_CASE(pooling_test_s8);

TEST_P(pooling_test_u8, TestsPooling) {}

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxU8, pooling_test_u8,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 3, 3, 0,
                                0, 2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardAvgU8, pooling_test_u8,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)}));

GPU_INST_TEST_CASE(pooling_test_u8);

TEST_P(pooling_test_s32, TestsPooling) {}

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAlexnetForwardS32, pooling_test_s32,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                1, 96, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(1, 256, 27, 27, 13, 13, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(1, 256, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxS32, pooling_test_s32,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardAvgS32, pooling_test_s32,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 128, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 64, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 96, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)}));

TEST_P(pooling_test_float, TestsPooling) {}

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardZeroDim, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 0, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nhwc,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                0, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 0, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardEF, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, -4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                -1, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::eltwise_square, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw16c_with_padded, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 17, 6, 6, 7, 7, 2, 2, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 23, 60, 60, 31, 31, 3, 4, 1, 0, 0, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 0, 0, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 1, 0, 0, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 25, 60, 60, 31, 31, 2, 4, 1, 1, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw8c_with_padded, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 5, 6, 6, 7, 7, 2, 2, 0, 0, 1, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 9, 60, 60, 31, 31, 3, 4, 0, 0, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 0, 0, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 0, 0, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 25, 60, 60, 31, 31, 2, 4, 1, 1, 1, 1, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxKernelSlipsToPadding,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10,
                                5, 5)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw16c, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 4, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 2, 4,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 4, 2,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw8c, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 4, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 2, 4,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 4, 2,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ndhwc, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::ndhwc,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 4, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ndhwc,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 2, 4,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 4, 2,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ncdhw, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::ncdhw,
                        memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 2, 4, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ncdhw,
                        memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 2, 4,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 31, 31, 3, 4, 2,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3Dunet_ncdhw, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ncdhw,
                        memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ncdhw,
                        memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ncdhw,
                        memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3Dunet_ndhwc, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ndhwc,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ndhwc,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::ndhwc,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3Dunet_blocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_test_params_t {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMax, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 2, 2, 3, 3, 3, 3, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxNHWC, pooling_test_float,
        ::testing::Values(pool_test_params_float {prop_kind::forward_training,
                algorithm::pooling_max, memory::format_tag::nhwc,
                memory::format_tag::nhwc,
                EXPAND_SIZES_2D(2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxBlocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1, 1, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxBlockedPerf,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardAvgBlockedPerf,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxBlocked16, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1, 1, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardMaxBlocked16Perf,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingForwardAvgBlocked16Perf,
        pooling_test_float,
        ::testing::Values(pool_test_params_float {prop_kind::forward_training,
                                  algorithm::pooling_avg_include_padding,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3,
                                          0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 2, 2, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAlexnetForwardMaxNCHW,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAlexnetForwardMaxBlocked,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAlexnetForwardMaxBlocked16,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxBlockedStride1, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 11, 11, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(2, 16, 13, 13, 11, 11, 3, 3, 2, 2, 0, 0,
                                1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxCIFAR10NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgCIFAR10NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 15, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxCIFAR10Blocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgCIFAR10Blocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxCIFAR10Blocked16, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgCIFAR10Blocked16, pooling_test_float,
        ::testing::Values(pool_test_params_float {prop_kind::forward_training,
                                  algorithm::pooling_avg_include_padding,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(2, 32, 16, 16, 8, 8, 3, 3, 0,
                                          0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxGoogleNetV1NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxGoogleNetV1Blocked,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxGoogleNetV1Blocked16,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxResnet50NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nchw,
                        memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxResnet50Blocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingMaxResnet50Blocked16,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgGoogleNetV1NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgGoogleNetV1Blocked,
        pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgGoogleNetV1Blocked16,
        pooling_test_float,
        ::testing::Values(pool_test_params_float {prop_kind::forward_training,
                                  algorithm::pooling_avg_include_padding,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(2, 512, 14, 14, 4, 4, 5, 5, 0,
                                          0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 528, 14, 14, 4, 4, 5, 5, 0, 0, 0, 0, 3, 3)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 1024, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgResnet50NCHW, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgResnet50Blocked, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAvgResnet50Blocked16,
        pooling_test_float,
        ::testing::Values(pool_test_params_float {prop_kind::forward_training,
                                  algorithm::pooling_avg_include_padding,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(2, 512, 7, 7, 1, 1, 7, 7, 0,
                                          0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_training,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 0, 0, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAsymmPadding, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 1, 1, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 1, 1, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 2, 2, 0, 1, 1, 2)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 1, 1, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 1, 1, 0, 1, 1, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 2, 2, 0, 1, 1, 2)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 2,
                                1, 1, 2, 2)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAsymmDilation, pooling_test_float,
        ::testing::Values(
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)}

                ,
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_max, memory::format_tag::nChw8c,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_test_params_float {prop_kind::forward_inference,
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)}));

GPU_INST_TEST_CASE(pooling_test_float);

} // namespace dnnl
