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

#include <cmath>
#include <memory>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, lnorm_test_t, ::testing::Values(__VA_ARGS__));

namespace dnnl {

struct test_lnorm_params_t {
    memory::format_tag data_tag;
    memory::format_tag stat_tag;
    memory::format_tag diff_tag;
    memory::dims dims;
    float epsilon;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename T>
void fill(const memory &m) {
    auto numElements = m.get_desc().get_size() / sizeof(T);
    fill_data<T>(numElements, m);
}

class lnorm_test_t : public ::testing::TestWithParam<test_lnorm_params_t> {
private:
    std::shared_ptr<test_memory> src, dst, diff_src, diff_dst;
    memory weights, diff_weights, mean, variance;

    std::shared_ptr<memory::desc> data_d;
    std::shared_ptr<memory::desc> stat_d;
    std::shared_ptr<memory::desc> diff_d;

    layer_normalization_forward::primitive_desc lnorm_fwd_pd;
    layer_normalization_backward::primitive_desc lnorm_bwd_pd;

    test_lnorm_params_t p;
    engine eng;
    stream strm;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        eng = get_test_engine();
        strm = make_stream(eng);

        data_d = std::make_shared<memory::desc>(
                p.dims, memory::data_type::f32, p.data_tag);
        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        stat_d = std::make_shared<memory::desc>(
                stat_dims, memory::data_type::f32, p.stat_tag);
        diff_d = std::make_shared<memory::desc>(
                p.dims, memory::data_type::f32, p.diff_tag);

        src = std::make_shared<test_memory>(*data_d, eng);
        dst = std::make_shared<test_memory>(*data_d, eng);
        diff_src = std::make_shared<test_memory>(*diff_d, eng);
        diff_dst = std::make_shared<test_memory>(*diff_d, eng);

        auto training = prop_kind::forward_training;
        auto inference = prop_kind::forward_inference;

        using flags = normalization_flags;
        Forward(training);
        Forward(training, flags::use_global_stats);
        Forward(training, flags::use_scale_shift);
        Forward(training, flags::use_scale_shift | flags::use_global_stats);
        Forward(inference);
        Forward(inference, flags::use_global_stats);
        Forward(inference, flags::use_scale_shift);

        Backward(prop_kind::backward_data);
        Backward(prop_kind::backward_data, flags::use_global_stats);
        Backward(prop_kind::backward, flags::use_scale_shift);
        Backward(prop_kind::backward,
                flags::use_scale_shift | flags::use_global_stats);
    }

    void Forward(prop_kind pk,
            normalization_flags flags = normalization_flags::none) {
        fwd_iface_test_stat_any(pk, flags);

        bool useScaleShift
                = (bool)(flags & normalization_flags::use_scale_shift);
        bool useGlobalStats
                = (bool)(flags & normalization_flags::use_global_stats);
        bool isTraining = pk == prop_kind::forward_training;

        auto lnorm_fwd_d = layer_normalization_forward::desc(
                pk, *data_d, *stat_d, p.epsilon, flags);

        lnorm_fwd_pd
                = layer_normalization_forward::primitive_desc(lnorm_fwd_d, eng);
        lnorm_fwd_pd = layer_normalization_forward::primitive_desc(
                lnorm_fwd_pd.get()); // test construction from a C pd

        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lnorm_fwd_pd.src_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == lnorm_fwd_pd.dst_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN)
                == lnorm_fwd_pd.mean_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == lnorm_fwd_pd.variance_desc());
        ASSERT_TRUE(
                lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SCALE_SHIFT)
                == lnorm_fwd_pd.weights_desc());

        weights = test::make_memory(lnorm_fwd_pd.weights_desc(), eng);
        if (isTraining || useGlobalStats) {
            mean = test::make_memory(*stat_d, eng);
            variance = test::make_memory(*stat_d, eng);
        }

        fill<float>(src->get());
        fill<float>(dst->get());
        if (useScaleShift) fill<float>(weights);
        if (useGlobalStats) {
            fill<float>(mean);
            fill<float>(variance);
        }

        execlnormFwd(isTraining, useGlobalStats, useScaleShift);
        check_lnorm_fwd(
                p, src->get(), mean, variance, weights, dst->get(), flags, pk);
    }

    void Backward(prop_kind pk,
            normalization_flags flags = normalization_flags::none) {
        bwd_iface_test_stat_any(pk, flags);

        bool useScaleShift
                = (bool)(flags & normalization_flags::use_scale_shift);

        auto lnorm_fwd_d
                = layer_normalization_forward::desc(prop_kind::forward_training,
                        *data_d, *stat_d, p.epsilon, flags);
        lnorm_fwd_pd
                = layer_normalization_forward::primitive_desc(lnorm_fwd_d, eng);

        auto lnorm_bwd_d = layer_normalization_backward::desc(
                pk, *diff_d, *data_d, *stat_d, p.epsilon, flags);
        lnorm_bwd_pd = layer_normalization_backward::primitive_desc(
                lnorm_bwd_d, eng, lnorm_fwd_pd);
        lnorm_bwd_pd = layer_normalization_backward::primitive_desc(
                lnorm_bwd_pd.get()); // test construction from a C pd

        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lnorm_bwd_pd.src_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == lnorm_bwd_pd.diff_src_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == lnorm_bwd_pd.diff_dst_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN)
                == lnorm_bwd_pd.mean_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == lnorm_bwd_pd.variance_desc());
        ASSERT_TRUE(
                lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SCALE_SHIFT)
                == lnorm_bwd_pd.weights_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SCALE_SHIFT)
                == lnorm_bwd_pd.diff_weights_desc());

        if (useScaleShift)
            weights = test::make_memory(lnorm_bwd_pd.weights_desc(), eng);
        diff_weights = test::make_memory(lnorm_bwd_pd.diff_weights_desc(), eng);
        mean = test::make_memory(*stat_d, eng);
        variance = test::make_memory(*stat_d, eng);

        if (useScaleShift) fill<float>(weights);
        fill<float>(diff_src->get());
        fill<float>(diff_dst->get());
        fill<float>(mean);
        fill<float>(variance);

        execlnormBwd(useScaleShift, pk);
        check_lnorm_bwd(p, src->get(), diff_dst->get(), mean, variance, weights,
                diff_src->get(), diff_weights, flags, pk);
    }

    void execlnormFwd(
            bool isTraining, bool useGlobalStats, bool useScaleShift) {
        std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, src->get()},
                {DNNL_ARG_DST, dst->get()},
        };

        if (useScaleShift) args.insert({DNNL_ARG_SCALE_SHIFT, weights});

        if (isTraining || useGlobalStats) {
            args.insert({DNNL_ARG_MEAN, mean});
            args.insert({DNNL_ARG_VARIANCE, variance});
        }

        layer_normalization_forward(lnorm_fwd_pd).execute(strm, args);
        strm.wait();
    }

    void execlnormBwd(bool useScaleShift, prop_kind pk) {
        std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, src->get()},
                {DNNL_ARG_DIFF_DST, diff_dst->get()},
                {DNNL_ARG_MEAN, mean},
                {DNNL_ARG_VARIANCE, variance},
                {DNNL_ARG_DIFF_SRC, diff_src->get()},
        };

        if (useScaleShift) {
            args.insert({DNNL_ARG_SCALE_SHIFT, weights});
            if (pk == prop_kind::backward)
                args.insert({DNNL_ARG_DIFF_SCALE_SHIFT, diff_weights});
        }

        layer_normalization_backward(lnorm_bwd_pd).execute(strm, args);
        strm.wait();
    }

    void check_lnorm_fwd(const test_lnorm_params_t &p, const memory &src,
            const memory &mean, const memory &variance, const memory &weights,
            const memory &dst, normalization_flags flags, prop_kind pk) {
        const size_t nelems = std::accumulate(p.dims.begin(), p.dims.end(),
                size_t(1), std::multiplies<size_t>());
        if (!nelems) return;

        const bool use_weights
                = (bool)(flags & normalization_flags::use_scale_shift);
        const bool calculate_stats
                = !(bool)(flags & normalization_flags::use_global_stats);
        const bool is_training = pk == prop_kind::forward_training;

        auto src_data = map_memory<const float>(src);
        auto dst_data = map_memory<const float>(dst);
        auto weights_data
                = use_weights ? map_memory<const float>(weights) : nullptr;
        auto mean_data = (!calculate_stats || is_training)
                ? map_memory<const float>(mean)
                : nullptr;
        auto variance_data = (!calculate_stats || is_training)
                ? map_memory<const float>(variance)
                : nullptr;

        const memory::desc src_d = src.get_desc();
        const memory::desc dst_d = dst.get_desc();
        const memory::desc weights_d
                = use_weights ? weights.get_desc() : memory::desc();
        const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
        const dnnl::impl::memory_desc_wrapper stat_mdw((*stat_d).data);
        const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);
        const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);

        if (!calculate_stats || is_training) {
            const memory::desc stat_d = mean.get_desc();
            const dnnl::impl::memory_desc_wrapper stat_mdw(stat_d.data);
        }
        const auto ndims = src_mdw.ndims();
        const auto C = src_mdw.dims()[ndims - 1];

        float eps = static_cast<float>(1.e-4 * nelems / C);
        dnnl::impl::parallel_nd(nelems / C, [&](memory::dim n) {
            if (is_current_test_failed()) return;
            float ref_mean = float(0);
            float ref_variance = float(0);
            const auto stat_off = stat_mdw.off_l(n);

            if (calculate_stats) {
                for (memory::dim c = 0; c < C; c++)
                    ref_mean += src_data[src_mdw.off_l(n * C + c)];
                ref_mean /= C;

                if (is_training) {
                    float mean_norm_max = std::max(
                            std::abs(mean_data[stat_off]), std::abs(ref_mean));
                    if (mean_norm_max < eps) mean_norm_max = float(1);
                    ASSERT_NEAR(
                            (mean_data[stat_off] - ref_mean) / mean_norm_max,
                            0., eps);
                }

                for (memory::dim c = 0; c < C; c++) {
                    float tmp = src_data[src_mdw.off_l(n * C + c)] - ref_mean;
                    ref_variance += tmp * tmp;
                }
                ref_variance /= C;

                if (is_training) {
                    float variance_norm_max
                            = std::max(std::abs(variance_data[stat_off]),
                                    std::abs(ref_variance));
                    if (variance_norm_max < eps) variance_norm_max = float(1);
                    ASSERT_NEAR((variance_data[stat_off] - ref_variance)
                                    / variance_norm_max,
                            0., eps);
                }
            } else {
                ref_mean = mean_data[stat_off];
                ref_variance = variance_data[stat_off];
            }

            float ref_sqrt_variance
                    = static_cast<float>(sqrt(ref_variance + p.epsilon));
            float ref_rsqrt_variance = float(1) / (ref_sqrt_variance);

            for (memory::dim c = 0; c < C; c++) {
                float ref_dst = float(0);
                if (use_weights) {
                    ref_dst = weights_data[c]
                                    * ((float)src_data[src_mdw.off_l(n * C + c)]
                                            - ref_mean)
                                    * ref_rsqrt_variance
                            + weights_data[C + c];
                } else {
                    ref_dst = ((float)src_data[src_mdw.off_l(n * C + c)]
                                      - ref_mean)
                            * ref_rsqrt_variance;
                }

                float out = dst_data[dst_mdw.off_l(n * C + c)];
                float norm_max = std::max(std::abs(out), std::abs(ref_dst));
                if (norm_max < 1e-2) norm_max = 1.;
                ASSERT_NEAR((out - ref_dst) / norm_max, 0., eps);
            }
        });
    }

    void check_lnorm_bwd(const test_lnorm_params_t &p, const memory &src,
            const memory &diff_dst, const memory &mean, const memory &variance,
            const memory &weights, const memory &diff_src,
            const memory &diff_weights, normalization_flags flags,
            prop_kind pk) {
        const ptrdiff_t nelems = std::accumulate(p.dims.begin(), p.dims.end(),
                size_t(1), std::multiplies<size_t>());

        const bool use_weights
                = (bool)(flags & normalization_flags::use_scale_shift);
        const bool calculate_diff_stats
                = !(bool)(flags & normalization_flags::use_global_stats);

        auto src_data = map_memory<const float>(src);
        auto weights_data
                = use_weights ? map_memory<const float>(weights) : nullptr;
        auto diff_dst_data = map_memory<const float>(diff_dst);
        auto mean_data = map_memory<const float>(mean);
        auto variance_data = map_memory<const float>(variance);
        const auto diff_src_data = map_memory<float>(diff_src);
        const auto diff_weights_data = (pk == prop_kind::backward)
                ? map_memory<float>(diff_weights)
                : nullptr;

        const memory::desc src_d = src.get_desc();
        const memory::desc diff_dst_d = diff_dst.get_desc();
        const memory::desc weights_d = weights.get_desc();
        const memory::desc diff_src_d = diff_src.get_desc();
        const memory::desc diff_weights_d = diff_weights.get_desc();

        const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
        const dnnl::impl::memory_desc_wrapper stat_mdw((*stat_d).data);
        const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);
        const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);
        const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);
        const dnnl::impl::memory_desc_wrapper diff_weights_mdw(
                diff_weights_d.data);

        const auto ndims = src_mdw.ndims();
        const auto C = src_mdw.dims()[ndims - 1];

        if (nelems == 0) {
            if (pk == prop_kind::backward) {
                for (memory::dim c = 0; c < C; ++c) {
                    auto dg = diff_weights_data[diff_weights_mdw.off_l(
                            c, true)];
                    auto db = diff_weights_data[diff_weights_mdw.off_l(
                            C + c, true)];
                    ASSERT_NEAR(dg, 0., 1e-7);
                    ASSERT_NEAR(db, 0., 1e-7);
                }
            }
            return;
        }

        const float eps = static_cast<float>(1.e-4 * nelems / C);

        dnnl::impl::parallel_nd(C, [&](memory::dim c) {
            if (is_current_test_failed()) return;

            float ref_diff_gamma = float(0);
            float ref_diff_beta = float(0);
            for (memory::dim n = 0; n < nelems / C; n++) {
                size_t stat_off = stat_mdw.off_l(n);
                const float sqrt_variance
                        = 1.0f / sqrt(variance_data[stat_off] + p.epsilon);

                ref_diff_gamma += (src_data[src_mdw.off_l(n * C + c)]
                                          - mean_data[stat_off])
                        * diff_dst_data[diff_dst_mdw.off_l(n * C + c)]
                        * sqrt_variance;
                ref_diff_beta += diff_dst_data[diff_dst_mdw.off_l(n * C + c)];
            }

            if (pk == prop_kind::backward) {
                auto diff_gamma = diff_weights_data[diff_weights_mdw.off_l(c)];
                float norm_max = std::max(
                        std::abs(diff_gamma), std::abs(ref_diff_gamma));
                if (norm_max < 1e-2) norm_max = float(1);
                ASSERT_NEAR((diff_gamma - ref_diff_gamma) / norm_max, 0., eps);

                auto diff_beta
                        = diff_weights_data[diff_weights_mdw.off_l(C + c)];
                norm_max = std::max(
                        std::abs(diff_beta), std::abs(ref_diff_beta));
                if (norm_max < 1e-2) norm_max = float(1);
                ASSERT_NEAR((diff_beta - ref_diff_beta) / norm_max, 0., eps);
            }
        });

        dnnl::impl::parallel_nd(nelems / C, [&](memory::dim n) {
            if (is_current_test_failed()) return;

            size_t stat_off = stat_mdw.off_l(n);
            const float sqrt_variance
                    = 1.0f / sqrt(variance_data[stat_off] + p.epsilon);

            float ref_dd_gamma = float(0);
            float ref_dd_gamma_x = float(0);
            if (calculate_diff_stats) {
                for (memory::dim c = 0; c < C; c++) {
                    auto gamma = use_weights
                            ? weights_data[weights_mdw.off_l(c)]
                            : 1;
                    ref_dd_gamma += diff_dst_data[diff_dst_mdw.off_l(n * C + c)]
                            * gamma;
                    ref_dd_gamma_x
                            += diff_dst_data[diff_dst_mdw.off_l(n * C + c)]
                            * gamma
                            * (src_data[src_mdw.off_l(n * C + c)]
                                    - mean_data[stat_off]);
                }
                ref_dd_gamma_x *= sqrt_variance;
            }
            for (memory::dim c = 0; c < C; c++) {
                auto gamma
                        = use_weights ? weights_data[weights_mdw.off_l(c)] : 1;
                float ref_diff_src
                        = diff_dst_data[diff_dst_mdw.off_l(n * C + c)] * gamma;
                if (calculate_diff_stats) {
                    ref_diff_src -= ref_dd_gamma / C
                            + (src_data[src_mdw.off_l(n * C + c)]
                                      - mean_data[stat_off])
                                    * ref_dd_gamma_x * sqrt_variance / C;
                }
                ref_diff_src *= sqrt_variance;
                float out_diff_src
                        = diff_src_data[diff_src_mdw.off_l(n * C + c)];
                float norm_max = std::max(
                        std::abs(out_diff_src), std::abs(ref_diff_src));
                if (norm_max < eps) norm_max = float(1);
                ASSERT_NEAR((out_diff_src - ref_diff_src) / norm_max, 0., eps);
            }
        });
    }

    void fwd_iface_test_stat_any(prop_kind pk, normalization_flags flags) {
        // non stats if inference w/o use global stats
        if (pk == prop_kind::forward_inference
                && !(bool)(flags & normalization_flags::use_global_stats))
            return;

        using tag = memory::format_tag;

        tag expect_stat_tag = derive_stat_tag();
        if (expect_stat_tag == tag::undef) return; // optimism

        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        memory::desc expect_stat_md(
                stat_dims, memory::data_type::f32, expect_stat_tag);

        // no stat_md provided at all
        {
            layer_normalization_forward::primitive_desc fwd_pd(
                    {pk, *data_d, p.epsilon, flags}, eng);

            EXPECT_EQ(fwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(fwd_pd.variance_desc(), expect_stat_md);
        }

        // stat_md with format_tag::any
        {
            memory::desc any_stat_md(
                    stat_dims, memory::data_type::f32, tag::any);
            layer_normalization_forward::primitive_desc fwd_pd(
                    {pk, *data_d, any_stat_md, p.epsilon, flags}, eng);

            EXPECT_EQ(fwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(fwd_pd.variance_desc(), expect_stat_md);
        }
    }

    void bwd_iface_test_stat_any(prop_kind pk, normalization_flags flags) {
        using tag = memory::format_tag;

        tag expect_stat_tag = derive_stat_tag();
        if (expect_stat_tag == tag::undef) return; // optimism

        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        memory::desc expect_stat_md(
                stat_dims, memory::data_type::f32, expect_stat_tag);

        layer_normalization_forward::primitive_desc fwd_pd(
                {prop_kind::forward_training, *data_d, p.epsilon, flags}, eng);

        // no stat_md provided at all
        {
            layer_normalization_backward::primitive_desc bwd_pd(
                    {pk, *diff_d, *data_d, p.epsilon, flags}, eng, fwd_pd);

            EXPECT_EQ(bwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(bwd_pd.variance_desc(), expect_stat_md);
        }

        // stat_md with format_tag::any
        {
            memory::desc any_stat_md(
                    stat_dims, memory::data_type::f32, tag::any);
            layer_normalization_backward::primitive_desc bwd_pd(
                    {pk, *diff_d, *data_d, any_stat_md, p.epsilon, flags}, eng,
                    fwd_pd);

            EXPECT_EQ(bwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(bwd_pd.variance_desc(), expect_stat_md);
        }
    }

private:
    memory::format_tag derive_stat_tag() const {
        using tag = memory::format_tag;
        tag expect_stat_tag = tag::undef;

        // TODO: add more cases and test cases
        // XXX: currently test only simple cases like `abc`, `acb`. Extend,
        //      if possible, to blocked formats too.
        switch (p.data_tag) {
            case tag::abc: expect_stat_tag = tag::ab; break;
            case tag::bac: expect_stat_tag = tag::ba; break;
            default: break;
        }

        return expect_stat_tag;
    }
};

TEST_P(lnorm_test_t, TestsLnormF32) {}

#include "layer_normalization.h"
} // namespace dnnl
