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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

struct test_resampling_desc_t {
    memory::dim mb, c;
    memory::dim id, ih, iw;
    memory::dim od, oh, ow;
    float fd, fh, fw;
};

struct resampling_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag src_format;
    int ndims;
    test_resampling_desc_t test_pd;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

float linear_map(memory::dim y, memory::dim y_max, memory::dim x_max) {
    const float s = (y + 0.5f) * x_max / y_max;
    return s - 0.5f;
}
memory::dim left_edge(memory::dim y, memory::dim y_max, memory::dim x_max) {
    return std::max((int64_t)floor(linear_map(y, y_max, x_max)), (int64_t)0);
}
memory::dim right_edge(memory::dim y, memory::dim y_max, memory::dim x_max) {
    return std::min((int64_t)ceil(linear_map(y, y_max, x_max)), x_max - 1);
}
memory::dim nearest_edge(memory::dim y, memory::dim y_max, memory::dim x_max) {
    return std::round(linear_map(y, y_max, x_max));
}
float linear_weight(memory::dim y, memory::dim y_max, memory::dim x_max) {
    return fabs(linear_map(y, y_max, x_max) - left_edge(y, y_max, x_max));
}

template <typename data_t>
void compute_ref_resampling_fwd(const resampling_test_params_t &p,
        const memory &src_m, const memory &dst_m) {
    auto src_data = map_memory<data_t>(src_m);
    auto dst_data = map_memory<data_t>(dst_m);

    const memory::desc src_d = src_m.get_desc();
    const memory::desc dst_d = dst_m.get_desc();

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    auto pd = p.test_pd;
    auto padded_c = src_mdw.padded_dims()[1];

    auto src = [&](memory::dim n, memory::dim c, memory::dim d, memory::dim h,
                       memory::dim w) {
        memory::dim idx = n * padded_c * pd.id * pd.ih * pd.iw
                + c * pd.id * pd.ih * pd.iw + d * pd.ih * pd.iw + h * pd.iw + w;
        return src_data[src_mdw.off_l(idx, true)];
    };

    dnnl::impl::parallel_nd(pd.mb, pd.c, [&](memory::dim n, memory::dim c) {
        for_(memory::dim od = 0; od < pd.od; od++)
        for_(memory::dim oh = 0; oh < pd.oh; oh++)
        for (memory::dim ow = 0; ow < pd.ow; ow++) {
            memory::dim oidx = n * padded_c * pd.od * pd.oh * pd.ow
                    + c * pd.od * pd.oh * pd.ow + od * pd.oh * pd.ow
                    + oh * pd.ow + ow;

            if (p.aalgorithm == algorithm::resampling_nearest) {
                memory::dim id = nearest_edge(od, pd.od, pd.id),
                            ih = nearest_edge(oh, pd.oh, pd.ih),
                            iw = nearest_edge(ow, pd.ow, pd.iw);
                memory::dim iidx = n * padded_c * pd.id * pd.ih * pd.iw
                        + c * pd.id * pd.ih * pd.iw + id * pd.ih * pd.iw
                        + ih * pd.iw + iw;
                dst_data[dst_mdw.off_l(oidx, true)]
                        = src_data[src_mdw.off_l(iidx, true)];
            } else if (p.aalgorithm == algorithm::resampling_linear) {
                memory::dim id_left = left_edge(od, pd.od, pd.id),
                            id_right = right_edge(od, pd.od, pd.id),
                            ih_left = left_edge(oh, pd.oh, pd.ih),
                            ih_right = right_edge(oh, pd.oh, pd.ih),
                            iw_left = left_edge(ow, pd.ow, pd.iw),
                            iw_right = right_edge(ow, pd.ow, pd.iw);
                float w_d = linear_weight(od, pd.od, pd.id),
                      w_h = linear_weight(oh, pd.oh, pd.ih),
                      w_w = linear_weight(ow, pd.ow, pd.iw);
                float c00 = src(n, c, id_left, ih_left, iw_left) * (1 - w_d)
                        + src(n, c, id_right, ih_left, iw_left) * w_d;
                float c01 = src(n, c, id_left, ih_left, iw_right) * (1 - w_d)
                        + src(n, c, id_right, ih_left, iw_right) * w_d;
                float c10 = src(n, c, id_left, ih_right, iw_left) * (1 - w_d)
                        + src(n, c, id_right, ih_right, iw_left) * w_d;
                float c11 = src(n, c, id_left, ih_right, iw_right) * (1 - w_d)
                        + src(n, c, id_right, ih_right, iw_right) * w_d;
                float c0 = c00 * (1 - w_h) + c10 * w_h;
                float c1 = c01 * (1 - w_h) + c11 * w_h;
                dst_data[dst_mdw.off_l(oidx, true)] = c0 * (1 - w_w) + c1 * w_w;
            }
        }
    });
}

template <typename data_t>
void compute_ref_resampling_bwd(const resampling_test_params_t &p,
        const memory &diff_dst_m, const memory &diff_src_m) {
    auto diff_src_data = map_memory<data_t>(diff_src_m);
    auto diff_dst_data = map_memory<data_t>(diff_dst_m);

    const memory::desc diff_src_d = diff_src_m.get_desc();
    const memory::desc diff_dst_d = diff_dst_m.get_desc();

    const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);

    auto pd = p.test_pd;
    auto padded_c = diff_src_mdw.padded_dims()[1];

    auto off = [&](memory::dim n, memory::dim c, memory::dim d, memory::dim h,
                       memory::dim w) {
        return diff_src_mdw.off_l(n * padded_c * pd.id * pd.ih * pd.iw
                        + c * pd.id * pd.ih * pd.iw + d * pd.ih * pd.iw
                        + h * pd.iw + w,
                true);
    };
    dnnl::impl::parallel_nd(pd.mb, pd.c, [&](memory::dim n, memory::dim c) {
        for_(memory::dim id = 0; id < pd.id; id++)
        for_(memory::dim ih = 0; ih < pd.ih; ih++)
        for (memory::dim iw = 0; iw < pd.iw; iw++) {
            memory::dim iidx = n * padded_c * pd.id * pd.ih * pd.iw
                    + c * pd.id * pd.ih * pd.iw + id * pd.ih * pd.iw
                    + ih * pd.iw + iw;

            diff_src_data[diff_src_mdw.off_l(iidx, true)] = 0.f;
        }
        for_(memory::dim od = 0; od < pd.od; od++)
        for_(memory::dim oh = 0; oh < pd.oh; oh++)
        for (memory::dim ow = 0; ow < pd.ow; ow++) {
            memory::dim oidx = n * padded_c * pd.od * pd.oh * pd.ow
                    + c * pd.od * pd.oh * pd.ow + od * pd.oh * pd.ow
                    + oh * pd.ow + ow;

            if (p.aalgorithm == algorithm::resampling_nearest) {
                memory::dim id = nearest_edge(od, pd.od, pd.id),
                            ih = nearest_edge(oh, pd.oh, pd.ih),
                            iw = nearest_edge(ow, pd.ow, pd.iw);
                memory::dim iidx = n * padded_c * pd.id * pd.ih * pd.iw
                        + c * pd.id * pd.ih * pd.iw + id * pd.ih * pd.iw
                        + ih * pd.iw + iw;
                diff_src_data[diff_src_mdw.off_l(iidx, true)]
                        += diff_dst_data[diff_dst_mdw.off_l(oidx, true)];
            } else if (p.aalgorithm == algorithm::resampling_linear) {
                memory::dim id_left = left_edge(od, pd.od, pd.id),
                            id_right = right_edge(od, pd.od, pd.id),
                            ih_left = left_edge(oh, pd.oh, pd.ih),
                            ih_right = right_edge(oh, pd.oh, pd.ih),
                            iw_left = left_edge(ow, pd.ow, pd.iw),
                            iw_right = right_edge(ow, pd.ow, pd.iw);
                float w_d = linear_weight(od, pd.od, pd.id),
                      w_h = linear_weight(oh, pd.oh, pd.ih),
                      w_w = linear_weight(ow, pd.ow, pd.iw);
                float dd = diff_dst_data[diff_dst_mdw.off_l(oidx, true)];

                diff_src_data[off(n, c, id_left, ih_left, iw_left)]
                        += (1 - w_d) * (1 - w_h) * (1 - w_w) * dd;
                diff_src_data[off(n, c, id_right, ih_left, iw_left)]
                        += w_d * (1 - w_h) * (1 - w_w) * dd;
                diff_src_data[off(n, c, id_left, ih_right, iw_left)]
                        += (1 - w_d) * w_h * (1 - w_w) * dd;
                diff_src_data[off(n, c, id_left, ih_left, iw_right)]
                        += (1 - w_d) * (1 - w_h) * w_w * dd;
                diff_src_data[off(n, c, id_right, ih_right, iw_left)]
                        += w_d * w_h * (1 - w_w) * dd;
                diff_src_data[off(n, c, id_left, ih_right, iw_right)]
                        += (1 - w_d) * w_h * w_w * dd;
                diff_src_data[off(n, c, id_right, ih_left, iw_right)]
                        += w_d * (1 - w_h) * w_w * dd;
                diff_src_data[off(n, c, id_right, ih_right, iw_right)]
                        += w_d * w_h * w_w * dd;
            }
        }
    });
}

template <typename data_t>
class resampling_test_t
    : public ::testing::TestWithParam<resampling_test_params_t> {
private:
    std::shared_ptr<test_memory> src, dst, diff_src, diff_dst;
    std::shared_ptr<memory::desc> src_desc, dst_desc;
    std::vector<float> factors;
    resampling_forward::primitive_desc resampling_pd;

    resampling_test_params_t p;
    engine eng;
    stream strm;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);

        test_resampling_desc_t pd = p.test_pd;

        memory::dims src_dims = {pd.mb, pd.c}, dst_dims = {pd.mb, pd.c};
        // When `out_of_memory` testing is enabled, factors is expanded each
        // time `Test()` is executed for any test. Clear the vector to avoid
        // having its size > DNNL_MAX_NDIMS.
        factors.clear();
        if (p.ndims == 5) {
            factors.push_back(pd.fd);
            src_dims.push_back(pd.id);
            dst_dims.push_back(pd.od);
        }
        if (p.ndims >= 4) {
            factors.push_back(pd.fh);
            src_dims.push_back(pd.ih);
            dst_dims.push_back(pd.oh);
        }
        if (p.ndims >= 3) {
            factors.push_back(pd.fw);
            src_dims.push_back(pd.iw);
            dst_dims.push_back(pd.ow);
        }

        memory::data_type data_type = data_traits<data_t>::data_type;
        src_desc.reset(new memory::desc(src_dims, data_type, p.src_format));
        dst_desc.reset(new memory::desc(dst_dims, data_type, p.src_format));

        Forward();
        Backward();
    }

    void Forward() {
        auto resampling_desc = resampling_forward::desc(
                p.aprop_kind, p.aalgorithm, *src_desc, *dst_desc);

        resampling_pd
                = resampling_forward::primitive_desc(resampling_desc, eng);
        resampling_pd = resampling_forward::primitive_desc(
                resampling_pd.get()); // test construction from a C pd

        if (true) {
            auto resampling_desc_no_dst = resampling_forward::desc(p.aprop_kind,
                    p.aalgorithm, factors, resampling_pd.src_desc());
            auto resampling_pd_no_dst = resampling_forward::primitive_desc(
                    resampling_desc_no_dst, eng);
            ASSERT_EQ(
                    resampling_pd.dst_desc(), resampling_pd_no_dst.dst_desc());
        }

        auto src = test::make_memory(resampling_pd.src_desc(), eng);
        auto dst = test::make_memory(resampling_pd.dst_desc(), eng);
        auto dst_ref = test::make_memory(resampling_pd.dst_desc(), eng);

        fill_data<data_t>(src.get_desc().get_size() / sizeof(data_t), src);
        check_zero_tail<data_t>(1, src);

        resampling_forward(resampling_pd)
                .execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
        strm.wait();

        compute_ref_resampling_fwd<data_t>(p, src, dst_ref);
        check_zero_tail<data_t>(1, dst_ref);
        compare_data<data_t>(dst_ref, dst);

        check_zero_tail<data_t>(0, dst);
    }

    void Backward() {
        auto resampling_bwd_desc = resampling_backward::desc(
                p.aalgorithm, factors, *src_desc, *dst_desc);

        auto resampling_bwd_pd = resampling_backward::primitive_desc(
                resampling_bwd_desc, eng, resampling_pd);

        auto diff_src
                = test::make_memory(resampling_bwd_pd.diff_src_desc(), eng);
        auto diff_dst
                = test::make_memory(resampling_bwd_pd.diff_dst_desc(), eng);
        auto diff_src_ref
                = test::make_memory(resampling_bwd_pd.diff_src_desc(), eng);

        fill_data<data_t>(
                diff_dst.get_desc().get_size() / sizeof(data_t), diff_dst);
        check_zero_tail<data_t>(1, diff_dst);
        check_zero_tail<data_t>(1, diff_src);

        resampling_backward(resampling_bwd_pd)
                .execute(strm,
                        {{DNNL_ARG_DIFF_SRC, diff_src},
                                {DNNL_ARG_DIFF_DST, diff_dst}});
        strm.wait();

        compute_ref_resampling_bwd<data_t>(p, diff_dst, diff_src_ref);
        check_zero_tail<data_t>(1, diff_src_ref);
        compare_data<data_t>(diff_src_ref, diff_src);
        check_zero_tail<data_t>(0, diff_src);
    }
};

using resampling_test_float = resampling_test_t<float>;

#define EXPAND_SIZES_3D(...) \
    5, { __VA_ARGS__ }
#define EXPAND_SIZES_2D(mb, c, ih, iw, oh, ow, fh, fw) \
    4, { mb, c, 1, ih, iw, 1, oh, ow, 1.f, fh, fw }
#define EXPAND_SIZES_1D(mb, c, iw, ow, fw) \
    3, { mb, c, 1, 1, iw, 1, 1, ow, 1.f, 1.f, fw }

TEST_P(resampling_test_float, TestsResampleF32) {}

INSTANTIATE_TEST_SUITE_P(TestResampleEF, resampling_test_float,
        ::testing::Values(resampling_test_params_t {prop_kind::forward,
                algorithm::resampling_linear, memory::format_tag::any,
                EXPAND_SIZES_1D(1, 1, 5, 10, 2.f), true,
                dnnl_invalid_arguments}));

INSTANTIATE_TEST_SUITE_P(TestResampleForwardPlainLinear, resampling_test_float,
        ::testing::Values(
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(1, 1, 5, 10, 2.f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(1, 1, 525, 5, 0.01f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(13, 10, 7, 13, 1.99f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(10, 16, 7, 13, 1.9f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(32, 10, 14, 7, 29, 5, 2.1f, 0.72f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(2, 14, 5, 5, 2, 3, 0.5f, 0.6f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(
                                1, 16, 5, 10, 1, 10, 5, 1, 2.f, 0.5f, 1.f)}));

INSTANTIATE_TEST_SUITE_P(TestResampleForwardBlockedLinear,
        resampling_test_float,
        ::testing::Values(
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(32, 16, 14, 6, 28, 3, 2, 0.5f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(32, 10, 14, 7, 29, 5, 2.1f, 0.72f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_linear, memory::format_tag::ncdhw,
                        EXPAND_SIZES_3D(
                                1, 1, 5, 10, 15, 10, 5, 7, 2.f, 0.5f, 0.5f)}));

INSTANTIATE_TEST_SUITE_P(TestResampleForwardPlainNN, resampling_test_float,
        ::testing::Values(
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(10, 16, 5, 10, 2.f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(13, 10, 7, 13, 1.99f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest, memory::format_tag::ncw,
                        EXPAND_SIZES_1D(10, 16, 7, 13, 1.9f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest, memory::format_tag::nchw,
                        EXPAND_SIZES_2D(32, 10, 14, 7, 29, 5, 2.1f, 0.72f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest, memory::format_tag::nhwc,
                        EXPAND_SIZES_2D(64, 32, 5, 5, 2, 3, 0.5f, 0.6f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest,
                        memory::format_tag::ndhwc,
                        EXPAND_SIZES_3D(
                                5, 5, 5, 10, 15, 10, 5, 7, 2.f, 0.5f, 0.5f)}));

INSTANTIATE_TEST_SUITE_P(TestResampleForwardBlockedNN, resampling_test_float,
        ::testing::Values(
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest,
                        memory::format_tag::nChw8c,
                        EXPAND_SIZES_2D(32, 16, 14, 6, 28, 3, 2, 0.5f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest,
                        memory::format_tag::nChw16c,
                        EXPAND_SIZES_2D(32, 10, 14, 7, 29, 5, 2.1f, 0.72f)},
                resampling_test_params_t {prop_kind::forward,
                        algorithm::resampling_nearest,
                        memory::format_tag::nCdhw16c,
                        EXPAND_SIZES_3D(
                                5, 5, 5, 10, 15, 10, 5, 7, 2.f, 0.5f, 0.5f)}));
} // namespace dnnl
