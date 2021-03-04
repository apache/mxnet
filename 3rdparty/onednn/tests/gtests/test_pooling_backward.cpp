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

struct test_pool_bwd_desc_t {
    memory::dim mb, c;
    memory::dim id, ih, iw;
    memory::dim od, oh, ow;
    memory::dim kd, kh, kw;
    memory::dim dd, dh, dw;
    memory::dim padf, padt, padl;
    memory::dim strd, strh, strw;
};

struct pool_bwd_test_params_t {
    algorithm aalgorithm;
    memory::format_tag diff_src_format;
    memory::format_tag diff_dst_format;
    int ndims;
    test_pool_bwd_desc_t test_pd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
void check_pool_fwd(
        const pool_bwd_test_params_t &p, const memory &src, const memory &dst) {
    auto src_data = map_memory<data_t>(src);
    auto dst_data = map_memory<data_t>(dst);

    const memory::desc src_d = src.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    auto pd = p.test_pd;
    auto padded_c = src_d.data.padded_dims[1];

    dnnl::impl::parallel_nd(pd.mb, pd.c, pd.od, pd.oh, pd.ow,
            [&](memory::dim n, memory::dim c, memory::dim od, memory::dim oh,
                    memory::dim ow) {
                if (is_current_test_failed()) return;

                memory::dim oidx = n * padded_c * pd.od * pd.oh * pd.ow
                        + c * pd.od * pd.oh * pd.ow + od * pd.oh * pd.ow
                        + oh * pd.ow + ow;
                data_t out = dst_data[dst_mdw.off_l(oidx, true)];

                // match implementation for pooling_max: padding
                // is done with lowest value and not zero, it
                // affects the case when kernel slips into
                // the padding area entirely
                data_t out_ref = (p.aalgorithm == algorithm::pooling_max)
                        ? std::numeric_limits<data_t>::lowest()
                        : data_t(0);
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
                            out_ref = d;
                            is_initialized = true;
                        } else {
                            if (out_ref < d) out_ref = d;
                        }
                    } else if (p.aalgorithm
                                    == algorithm::pooling_avg_include_padding
                            || p.aalgorithm
                                    == algorithm::pooling_avg_exclude_padding) {
                        out_ref += d;
                        num_summands++;
                    }
                }
                if (p.aalgorithm == algorithm::pooling_avg_include_padding)
                    num_summands = pd.kd * pd.kh * pd.kw;
                if ((p.aalgorithm == algorithm::pooling_avg_include_padding
                            || p.aalgorithm
                                    == algorithm::pooling_avg_exclude_padding)
                        && num_summands) {
                    out_ref /= num_summands;
                }
                ASSERT_NEAR(out, out_ref, 1e-6f);
            });
}

template <typename data_t>
void check_pool_bwd(const pool_bwd_test_params_t &p, const memory &diff_src,
        const memory &diff_dst, const memory &ws) {
    auto diff_src_data = map_memory<data_t>(diff_src);
    auto diff_dst_data = map_memory<data_t>(diff_dst);

    auto ws_data_ptr = map_memory<unsigned char>(ws);

    auto ws_data = [&](size_t idx) -> int {
        auto w = (const unsigned char *)ws_data_ptr;
        if (w == nullptr) return -1;
        if (ws.get_desc().data.data_type == dnnl_u8)
            return (int)w[idx];
        else
            return ((const int *)w)[idx];
    };

    const memory::desc diff_src_d = diff_src.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();
    const memory::desc ws_d = ws.get_desc();

    const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);
    const dnnl::impl::memory_desc_wrapper ws_mdw(ws_d.data);

    auto pd = p.test_pd;
    if (pd.mb * pd.c * pd.id * pd.ih * pd.iw == 0) return;

    std::vector<data_t> ref_diff_src_vec(pd.mb * pd.c * pd.id * pd.ih * pd.iw);
    data_t *ref_diff_src = &ref_diff_src_vec[0];

    dnnl::impl::parallel_nd(pd.mb * pd.c * pd.id * pd.ih * pd.iw,
            [&](memory::dim i) { ref_diff_src[i] = 0.; });

    dnnl::impl::parallel_nd(pd.mb, pd.c, [&](memory::dim n, memory::dim c) {
        for_(memory::dim od = 0; od < pd.od; od++)
        for_(memory::dim oh = 0; oh < pd.oh; oh++)
        for (memory::dim ow = 0; ow < pd.ow; ow++) {
            memory::dim oidx = n * pd.c * pd.od * pd.oh * pd.ow
                    + c * pd.od * pd.oh * pd.ow + od * pd.oh * pd.ow
                    + oh * pd.ow + ow;
            data_t diff_dst = diff_dst_data[diff_dst_mdw.off_l(oidx, true)];
            for_(memory::dim kd = 0; kd < pd.kd; kd++)
            for_(memory::dim kh = 0; kh < pd.kh; kh++)
            for (memory::dim kw = 0; kw < pd.kw; kw++) {
                memory::dim iw = ow * pd.strw - pd.padl + kw * (pd.dw + 1);
                memory::dim ih = oh * pd.strh - pd.padt + kh * (pd.dh + 1);
                memory::dim id = od * pd.strd - pd.padf + kd * (pd.dd + 1);
                if (iw < 0 || iw >= pd.iw) continue;
                if (ih < 0 || ih >= pd.ih) continue;
                if (id < 0 || id >= pd.id) continue;
                memory::dim iidx = n * pd.c * pd.id * pd.ih * pd.iw
                        + c * pd.id * pd.ih * pd.iw + id * pd.ih * pd.iw
                        + ih * pd.iw + iw;
                if (p.aalgorithm == algorithm::pooling_max) {
                    memory::dim kw_max
                            = ws_data(ws_mdw.off_l(oidx, true)) % pd.kw;
                    memory::dim kh_max
                            = (ws_data(ws_mdw.off_l(oidx, true)) / pd.kw)
                            % pd.kh;
                    memory::dim kd_max
                            = (ws_data(ws_mdw.off_l(oidx, true)) / pd.kw)
                            / pd.kh;
                    if (kh == kh_max && kw == kw_max && kd == kd_max)
                        ref_diff_src[iidx] += diff_dst;
                } else {
                    auto id_start = od * pd.strd - pd.padf;
                    auto ih_start = oh * pd.strh - pd.padt;
                    auto iw_start = ow * pd.strw - pd.padl;
                    auto id_end = od * pd.strd - pd.padf + (pd.kd - 1) * pd.dd
                            + pd.kd;
                    auto ih_end = oh * pd.strh - pd.padt + (pd.kh - 1) * pd.dh
                            + pd.kh;
                    auto iw_end = ow * pd.strw - pd.padl + (pd.kw - 1) * pd.dw
                            + pd.kw;

                    auto id_start_excluded = id_start < 0
                            ? (0 - id_start - 1) / (pd.dd + 1) + 1
                            : 0;
                    auto ih_start_excluded = ih_start < 0
                            ? (0 - ih_start - 1) / (pd.dh + 1) + 1
                            : 0;
                    auto iw_start_excluded = iw_start < 0
                            ? (0 - iw_start - 1) / (pd.dw + 1) + 1
                            : 0;
                    auto id_end_excluded = id_end > pd.id
                            ? (id_end - pd.id - 1) / (pd.dd + 1) + 1
                            : 0;
                    auto ih_end_excluded = ih_end > pd.ih
                            ? (ih_end - pd.ih - 1) / (pd.dh + 1) + 1
                            : 0;
                    auto iw_end_excluded = iw_end > pd.iw
                            ? (iw_end - pd.iw - 1) / (pd.dw + 1) + 1
                            : 0;

                    auto num_summands
                            = (p.aalgorithm
                                      != algorithm::pooling_avg_exclude_padding)
                            ? pd.kw * pd.kh * pd.kd
                            : (pd.kd - id_start_excluded - id_end_excluded)
                                    * (pd.kh - ih_start_excluded
                                            - ih_end_excluded)
                                    * (pd.kw - iw_start_excluded
                                            - iw_end_excluded);

                    ref_diff_src[iidx] += diff_dst / num_summands;
                }
            }
        }
    });

    dnnl::impl::parallel_nd(
            pd.mb * pd.c * pd.id * pd.ih * pd.iw, [&](memory::dim i) {
                if (is_current_test_failed()) return;

                ASSERT_NEAR(ref_diff_src[i],
                        diff_src_data[diff_src_mdw.off_l(i, true)], 1e-5f);
            });
}

template <typename data_t>
class pooling_bwd_test_t
    : public ::testing::TestWithParam<pool_bwd_test_params_t> {
private:
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    memory workspace;
    union prim_desc_union {
        pooling_forward::primitive_desc pool_prim_desc;
        pooling_v2_forward::primitive_desc pool_v2_prim_desc;
        prim_desc_union() {
            new (&pool_v2_prim_desc) pooling_v2_forward::primitive_desc();
        }
        ~prim_desc_union() { pool_v2_prim_desc.~primitive_desc(); }
    } prim_desc;
    pool_bwd_test_params_t p;
    memory::dims strides, ker, dilation, pad_l, pad_r;
    engine eng;
    stream strm;
    memory::data_type data_type;
    bool is_not_dilated;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        test_pool_bwd_desc_t pd = p.test_pd;

        eng = get_test_engine();
        strm = make_stream(eng);
        data_type = data_traits<data_t>::data_type;
        ASSERT_EQ(data_type, dnnl::memory::data_type::f32);

        if (p.ndims == 5) {
            src_desc.reset(new memory::desc({pd.mb, pd.c, pd.id, pd.ih, pd.iw},
                    data_type, p.diff_src_format));
            dst_desc.reset(new memory::desc({pd.mb, pd.c, pd.od, pd.oh, pd.ow},
                    data_type, p.diff_dst_format));
        } else {
            src_desc.reset(new memory::desc(
                    {pd.mb, pd.c, pd.ih, pd.iw}, data_type, p.diff_src_format));
            dst_desc.reset(new memory::desc(
                    {pd.mb, pd.c, pd.oh, pd.ow}, data_type, p.diff_dst_format));
        }

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

        is_not_dilated = pd.dd == 0 && pd.dh == 0 && pd.dw == 0;

        Forward();
        Backward();
    }

    template <typename prim_desc>
    void check_prim_desc(prim_desc pool_bwd_prim_desc) {
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == pool_bwd_prim_desc.diff_src_desc());
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == pool_bwd_prim_desc.diff_dst_desc());
        ASSERT_TRUE(pool_bwd_prim_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == pool_bwd_prim_desc.workspace_desc());
    }

    void Forward() {
        auto src = test::make_memory(*src_desc, eng);
        auto dst = test::make_memory(*dst_desc, eng);

        fill_data<data_t>(src.get_desc().get_size() / sizeof(data_t), src);
        fill_data<data_t>(dst.get_desc().get_size() / sizeof(data_t), dst);
        check_zero_tail<data_t>(1, src);
        check_zero_tail<data_t>(1, dst);

        if (is_not_dilated) {
            auto pool_desc = pooling_forward::desc(prop_kind::forward_training,
                    p.aalgorithm, *src_desc, *dst_desc, strides, ker, pad_l,
                    pad_r);
            prim_desc.pool_prim_desc
                    = pooling_forward::primitive_desc(pool_desc, eng);

            auto p_workspace_desc = prim_desc.pool_prim_desc.workspace_desc();
            workspace = test::make_memory(p_workspace_desc, eng);

            pooling_forward(prim_desc.pool_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        } else {
            auto pool_desc = pooling_v2_forward::desc(
                    prop_kind::forward_training, p.aalgorithm, *src_desc,
                    *dst_desc, strides, ker, dilation, pad_l, pad_r);
            prim_desc.pool_v2_prim_desc
                    = pooling_v2_forward::primitive_desc(pool_desc, eng);

            auto p_workspace_desc
                    = prim_desc.pool_v2_prim_desc.workspace_desc();
            workspace = test::make_memory(p_workspace_desc, eng);

            pooling_v2_forward(prim_desc.pool_v2_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        }
        strm.wait();

        check_zero_tail<data_t>(0, dst);
        check_pool_fwd<data_t>(p, src, dst);
    }

    void Backward() {
        auto diff_src = test::make_memory(*src_desc, eng);
        auto diff_dst = test::make_memory(*dst_desc, eng);

        fill_data<data_t>(
                diff_dst.get_desc().get_size() / sizeof(data_t), diff_dst);
        fill_data<data_t>(
                diff_src.get_desc().get_size() / sizeof(data_t), diff_src);
        check_zero_tail<data_t>(1, diff_dst);
        check_zero_tail<data_t>(1, diff_src);

        if (is_not_dilated) {
            auto pool_bwd_desc = pooling_backward::desc(p.aalgorithm, *src_desc,
                    *dst_desc, strides, ker, pad_l, pad_r);
            auto pool_bwd_prim_desc = pooling_backward::primitive_desc(
                    pool_bwd_desc, eng, prim_desc.pool_prim_desc);
            pool_bwd_prim_desc = pooling_backward::primitive_desc(
                    pool_bwd_prim_desc.get()); // test construction from a C pd

            check_prim_desc(pool_bwd_prim_desc);

            pooling_backward(pool_bwd_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_DIFF_DST, diff_dst},
                                    {DNNL_ARG_DIFF_SRC, diff_src},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        } else {
            auto pool_bwd_desc = pooling_v2_backward::desc(p.aalgorithm,
                    *src_desc, *dst_desc, strides, ker, dilation, pad_l, pad_r);
            auto pool_bwd_prim_desc = pooling_v2_backward::primitive_desc(
                    pool_bwd_desc, eng, prim_desc.pool_v2_prim_desc);
            pool_bwd_prim_desc = pooling_v2_backward::primitive_desc(
                    pool_bwd_prim_desc.get()); // test construction from a C pd

            check_prim_desc(pool_bwd_prim_desc);

            pooling_v2_backward(pool_bwd_prim_desc)
                    .execute(strm,
                            {{DNNL_ARG_DIFF_DST, diff_dst},
                                    {DNNL_ARG_DIFF_SRC, diff_src},
                                    {DNNL_ARG_WORKSPACE, workspace}});
        }

        strm.wait();

        check_zero_tail<data_t>(0, diff_src);
        check_pool_bwd<data_t>(p, diff_src, diff_dst, workspace);
    }
};

using pooling_bwd_test_float = pooling_bwd_test_t<float>;
using pool_bwd_test_params_float = pool_bwd_test_params_t;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D( \
        mb, ic, ih, iw, oh, ow, kh, kw, dh, dw, padt, padl, strh, strw) \
    4, { \
        mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, dh, dw, 0, padt, padl, 1, \
                strh, strw \
    }

TEST_P(pooling_bwd_test_float, TestsPoolingBackward) {}

INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardZeroDim, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 0, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                0, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 0, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1)}));

INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardEF, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, -4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                -2, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments},
                pool_bwd_test_params_float {algorithm::eltwise_square,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1),
                        true, dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw16c_padded, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nChw16c,
                                  memory::format_tag::nChw16c,
                                  EXPAND_SIZES_2D(4, 17, 6, 6, 7, 7, 2, 2, 0, 0,
                                          1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 23, 60, 60, 31, 31, 3, 4, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 2, 2, 1, 1, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 2, 2, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

INSTANTIATE_TEST_SUITE_P(TestPooling_nChw8c_padded, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 5, 6, 6, 7, 7, 2, 2, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 23, 60, 60, 31, 31, 3, 4, 0, 0, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 3, 2, 0, 0, 1, 1, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 17, 60, 60, 31, 31, 4, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                4, 14, 60, 60, 31, 31, 2, 3, 1, 1, 1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(4, 28, 60, 60, 31, 31, 4, 2, 1, 1, 1, 1,
                                2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxKernelSlipsToPadding,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nhwc, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10, 5, 5)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(1, 16, 10, 10, 6, 6, 5, 5, 0, 0, 10, 10,
                                5, 5)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw16c, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nCdhw16c,
                                  memory::format_tag::nCdhw16c,
                                  EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30,
                                          2, 3, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 23, 23, 23, 11, 11, 11, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                1, 1, 1, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 2, 2, 2, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 2, 2, 2, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ncdhw, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_ndhwc, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30, 2, 3, 4,
                                1, 1, 0, 0, 0, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPooling3D_nCdhw8c, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                                  memory::format_tag::nCdhw8c,
                                  memory::format_tag::nCdhw8c,
                                  EXPAND_SIZES_3D(2, 32, 60, 60, 60, 31, 30, 30,
                                          2, 3, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 30, 31, 4, 3, 2,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 60, 60, 60, 30, 31, 30, 4, 2, 3,
                                0, 0, 0, 1, 1, 1, 2, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw8c,
                        memory::format_tag::nCdhw8c,
                        EXPAND_SIZES_3D(2, 32, 30, 30, 30, 30, 30, 30, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax3DunetNCDHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                1, 1, 1, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax3DunetNDHWC,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 64, 64, 64, 64, 64, 64, 64, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 128, 28, 28, 28, 28, 28, 28, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ndhwc, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(1, 256, 12, 12, 12, 12, 12, 12, 2, 2, 2,
                                0, 0, 0, 0, 0, 0, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxAlexNetNCHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxCIFAR10NCHW,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 32, 16, 16, 8, 8, 3, 3, 0, 0, 0, 0, 2, 2)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 64, 8, 8, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMax, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nchw, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(
                                2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 1, 1, 1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked,
        pooling_bwd_test_float,
        ::testing::Values(

                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 2, 2, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1,
                                1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 3, 3, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 3, 2, 2, 2, 3, 3, 5, 5, 1, 1, 2, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                2, 8, 3, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked16,
        pooling_bwd_test_float,
        ::testing::Values(

                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                1, 16, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 2, 2, 3, 3, 2, 2, 0, 0, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(122, 32, 32, 2, 32, 2, 3, 3, 2, 2, 1, 1,
                                1, 1)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked16,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 13, 13, 11, 11, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 3, 3, 1, 1, 3, 3, 0, 0, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 0, 0, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 2, 2, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 32, 5, 5, 2, 2, 3, 3, 3, 3, 0, 0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 3, 2, 2, 2, 3, 3, 5, 5, 1, 1, 2, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(
                                2, 16, 3, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlockedPerf,
        pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                EXPAND_SIZES_2D(
                        16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlockedPerf,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 1,
                                1, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardMaxBlocked16Perf,
        pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_float {algorithm::pooling_max,
                memory::format_tag::nChw16c, memory::format_tag::nChw16c,
                EXPAND_SIZES_2D(
                        16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0, 0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAvgBlocked16Perf,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 0, 0, 0,
                                0, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw16c,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(16, 64, 32, 32, 16, 16, 3, 3, 1, 1, 0,
                                0, 2, 2)}));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingBackwardAsymmPadding,
        pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 0, 0, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 1, 0, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 14, 1, 8, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 100, 1, 51, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 3, 102, 1, 52, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 0, 0, 0, 1, 1, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 96, 9, 103, 7, 52, 3, 3, 1, 1, 0, 1, 1, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 0, 0,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 0, 0,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 1, 1,
                                1, 1, 2, 2)}

                ));

CPU_INSTANTIATE_TEST_SUITE_P(TestPoolingAsymmDilation, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 1, 0, 1, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(
                                1, 8, 3, 4, 1, 5, 3, 3, 0, 1, 1, 1, 1, 1)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 2, 4,
                                1, 1, 2, 2)}

                ,
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nChw8c, memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(1, 96, 300, 500, 151, 251, 3, 3, 4, 2,
                                1, 1, 2, 2)}));

GPU_INSTANTIATE_TEST_SUITE_P(TestPoolingSlipsToPadding, pooling_bwd_test_float,
        ::testing::Values(pool_bwd_test_params_t {algorithm::pooling_max,
                                  memory::format_tag::NChw16n16c,
                                  memory::format_tag::NChw16n16c,
                                  EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3,
                                          0, 0, 1, 1, 1, 1)},
                pool_bwd_test_params_t {algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, 0, 1,
                                1, 1, 1)},
                pool_bwd_test_params_t {algorithm::pooling_avg_include_padding,
                        memory::format_tag::NChw16n16c,
                        memory::format_tag::NChw16n16c,
                        EXPAND_SIZES_2D(64, 64, 56, 56, 56, 56, 3, 3, 0, 0, 1,
                                1, 1, 1)}));

GPU_INSTANTIATE_TEST_SUITE_P(TestPooling_ncdhw, pooling_bwd_test_float,
        ::testing::Values(
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::ncdhw, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(5, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                0, 0, 0, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::NCdhw16n16c,
                        memory::format_tag::NCdhw16n16c,
                        EXPAND_SIZES_3D(32, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {algorithm::pooling_max,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                2, 2, 2, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_exclude_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                3, 3, 3, 1, 1, 1, 1, 1, 1)},
                pool_bwd_test_params_float {
                        algorithm::pooling_avg_include_padding,
                        memory::format_tag::nCdhw16c,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(3, 32, 14, 14, 14, 14, 14, 14, 3, 3, 3,
                                5, 5, 5, 1, 1, 1, 1, 1, 1)}));

} // namespace dnnl
