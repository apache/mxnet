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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
namespace dnnl {
using fmt = memory::format_tag;
struct deconvolution_test_params_t {
    dnnl::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};
template <typename data_t>
void compute_bias_fwd(const test_convolution_sizes_t &c,
        const dnnl::memory &dst, const dnnl::memory &bias) {
    auto bias_data = map_memory<data_t>(bias);
    auto dst_data = map_memory<data_t>(dst);

    const memory::desc bias_d = bias.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    dnnl::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
            [&](memory::dim n, memory::dim g, memory::dim oc, memory::dim oh,
                    memory::dim ow) {
                data_t b
                        = bias_data[bias_mdw.off_l(g * c.oc / c.ng + oc, true)];
                memory::dim oidx = n * c.oc * c.oh * c.ow
                        + g * c.oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                        + oh * c.ow + ow;
                dst_data[dst_mdw.off_l(oidx, true)] += b;
            });
}

template <typename data_t>
void compute_bias_bwd(const test_convolution_sizes_t &c,
        const dnnl::memory &dst, const dnnl::memory &bias) {
    auto bias_data = map_memory<data_t>(bias);
    auto dst_data = map_memory<data_t>(dst);

    const memory::desc bias_d = bias.get_desc();
    const memory::desc dst_d = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper bias_mdw(bias_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    dnnl::impl::parallel_nd(
            c.ng, c.oc / c.ng, [&](memory::dim g, memory::dim oc) {
                memory::dim bidx = g * c.oc / c.ng + oc;
                bias_data[bias_mdw.off_l(bidx, true)] = 0.0;
                for_(memory::dim mb = 0; mb < c.mb; ++mb)
                for_(memory::dim oh = 0; oh < c.oh; ++oh)
                for (memory::dim ow = 0; ow < c.ow; ++ow) {
                    memory::dim oidx = mb * c.oc * c.oh * c.ow
                            + g * c.oc / c.ng * c.oh * c.ow + oc * c.oh * c.ow
                            + oh * c.ow + ow;
                    bias_data[bias_mdw.off_l(bidx, true)]
                            += dst_data[dst_mdw.off_l(oidx, true)];
                }
            });
}

template <typename data_t>
void transpose_wei(const test_convolution_sizes_t &c,
        const dnnl::memory &weights, const dnnl::memory &weights_tr) {

    auto weights_data = map_memory<data_t>(weights);
    const memory::desc weights_d = weights.get_desc();
    const dnnl::impl::memory_desc_wrapper weights_mdw(weights_d.data);
    auto weights_tr_data = map_memory<data_t>(weights_tr);
    const memory::desc weights_tr_d = weights_tr.get_desc();
    const dnnl::impl::memory_desc_wrapper weights_tr_mdw(weights_tr_d.data);

    dnnl::impl::parallel_nd(c.ng, c.oc / c.ng, c.ic / c.ng, c.kh, c.kw,
            [&](memory::dim g, memory::dim oc, memory::dim ic, memory::dim kh,
                    memory::dim kw) {
                memory::dim widx = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
                        + oc * c.ic / c.ng * c.kh * c.kw + ic * c.kh * c.kw
                        + kh * c.kw + kw;
                memory::dim widx_tr
                        = g * c.oc / c.ng * c.ic / c.ng * c.kh * c.kw
                        + ic * c.oc / c.ng * c.kh * c.kw + oc * c.kh * c.kw
                        + kh * c.kw + kw;
                weights_tr_data[weights_tr_mdw.off_l(widx_tr, true)]
                        = weights_data[weights_mdw.off_l(widx, true)];
            });
}

template <typename data_t>
class deconvolution_test_t
    : public ::testing::TestWithParam<deconvolution_test_params_t> {
private:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> weights;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> bias;

    std::shared_ptr<memory::desc> dec_src_desc;
    std::shared_ptr<memory::desc> dec_weights_desc;
    std::shared_ptr<memory::desc> dec_bias_desc;
    std::shared_ptr<memory::desc> dec_dst_desc;

    std::shared_ptr<memory::desc> con_src_desc;
    std::shared_ptr<memory::desc> con_bias_desc;
    std::shared_ptr<memory::desc> con_dst_desc;
    std::shared_ptr<memory::desc> con_weights_desc;

    engine eng;
    stream strm;
    bool with_bias;
    memory::dims padR;

protected:
    void SetUp() override {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);

        ASSERT_EQ(p.aalgorithm, algorithm::deconvolution_direct);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_convolution_sizes_t dd = p.sizes;
        with_bias = p.formats.bias_format != memory::format_tag::undef;

        memory::dims src_dims = {dd.mb, dd.ic, dd.ih, dd.iw};
        memory::dims dst_dims = {dd.mb, dd.oc, dd.oh, dd.ow};
        memory::dims weights_dims, c_weights_dims;
        if (dd.ng > 1) {
            weights_dims = {dd.ng, dd.oc / dd.ng, dd.ic / dd.ng, dd.kh, dd.kw};
            c_weights_dims
                    = {dd.ng, dd.ic / dd.ng, dd.oc / dd.ng, dd.kh, dd.kw};
        } else {
            weights_dims = {dd.oc, dd.ic, dd.kh, dd.kw};
            c_weights_dims = {dd.ic, dd.oc, dd.kh, dd.kw};
        }
        memory::dims bias_dims;
        if (with_bias)
            bias_dims = {dd.oc};
        else
            bias_dims = {};

        dec_src_desc.reset(
                new memory::desc(src_dims, data_type, p.formats.src_format));
        dec_dst_desc.reset(
                new memory::desc(dst_dims, data_type, p.formats.src_format));
        dec_weights_desc.reset(new memory::desc(
                weights_dims, data_type, p.formats.weights_format));
        dec_bias_desc.reset(
                new memory::desc(bias_dims, data_type, p.formats.bias_format));

        con_src_desc.reset(
                new memory::desc(dst_dims, data_type, p.formats.src_format));
        con_dst_desc.reset(
                new memory::desc(src_dims, data_type, p.formats.src_format));
        con_weights_desc.reset(new memory::desc(
                c_weights_dims, data_type, p.formats.weights_format));

        src.reset(new test_memory(*dec_src_desc, eng));
        weights.reset(new test_memory(*dec_weights_desc, eng));
        bias.reset(new test_memory(*dec_bias_desc, eng));
        dst.reset(new test_memory(*dec_dst_desc, eng));

        padR = {right_padding(dd.oh, dd.ih, dd.kh, dd.padh, dd.strh, dd.dilh),
                right_padding(dd.ow, dd.iw, dd.kw, dd.padw, dd.strw, dd.dilw)};
        Forward();
        BackwardData();
        BackwardWeights();
    }
    void Forward() {
        auto aprop_kind = prop_kind::forward;
        deconvolution_test_params_t p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();
        auto conv_src = test_memory(*con_src_desc, eng);
        auto conv_dst = src;
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(weights->get_size() / sizeof(data_t), weights->get());
        if (with_bias) {
            fill_data<data_t>(bias->get_size() / sizeof(data_t), bias->get());
        }

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);
        auto deconv_desc = with_bias
                ? deconvolution_forward::desc(aprop_kind,
                        algorithm::deconvolution_direct, *dec_src_desc,
                        *dec_weights_desc, *dec_bias_desc, *dec_dst_desc,
                        {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR)
                : deconvolution_forward::desc(aprop_kind,
                        algorithm::deconvolution_direct, *dec_src_desc,
                        *dec_weights_desc, *dec_dst_desc, {dd.strh, dd.strw},
                        {dd.padh, dd.padw}, padR);

        auto deconv_primitive_desc
                = deconvolution_forward::primitive_desc(deconv_desc, eng);
        deconv_primitive_desc = deconvolution_forward::primitive_desc(
                deconv_primitive_desc.get()); // test construction from a C pd

        ASSERT_TRUE(
                deconv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == deconv_primitive_desc.src_desc());
        ASSERT_TRUE(
                deconv_primitive_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == deconv_primitive_desc.dst_desc());
        ASSERT_TRUE(deconv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == deconv_primitive_desc.weights_desc());
        ASSERT_TRUE(deconv_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_BIAS)
                == deconv_primitive_desc.bias_desc());

        deconvolution_forward(deconv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src->get()},
                                {DNNL_ARG_WEIGHTS, weights->get()},
                                {DNNL_ARG_BIAS, bias->get()},
                                {DNNL_ARG_DST, dst->get()}});
        strm.wait();

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                algorithm::convolution_direct, *con_src_desc, *con_weights_desc,
                *con_dst_desc, {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto conv_primitive_desc
                = convolution_forward::primitive_desc(conv_desc, eng);

        auto conv_bwd_data_desc = convolution_backward_data::desc(
                algorithm::convolution_direct, *con_src_desc, *con_weights_desc,
                *con_dst_desc, {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto conv_bwd_data_primitive_desc
                = convolution_backward_data::primitive_desc(
                        conv_bwd_data_desc, eng, conv_primitive_desc);

        convolution_backward_data(conv_bwd_data_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, conv_dst->get()},
                                {DNNL_ARG_WEIGHTS, weights_tr},
                                {DNNL_ARG_DIFF_SRC, conv_src.get()}});
        strm.wait();

        if (with_bias)
            compute_bias_fwd<data_t>(dd, conv_src.get(), bias->get());
        compare_data<data_t>(conv_src.get(), dst->get());
    }

    void BackwardData() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();
        auto conv_src = dst;
        auto conv_dst = test_memory(*con_dst_desc, eng);
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(weights->get_size() / sizeof(data_t), weights->get());

        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);

        auto deconv_desc = deconvolution_forward::desc(
                prop_kind::forward_training, algorithm::deconvolution_direct,
                *dec_src_desc, *dec_weights_desc, *dec_dst_desc,
                {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto deconv_primitive_desc
                = deconvolution_forward::primitive_desc(deconv_desc, eng);

        auto deconv_bwd_data_desc = deconvolution_backward_data::desc(
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_dst_desc, {dd.strh, dd.strw},
                {dd.padh, dd.padw}, padR);
        auto deconv_bwd_data_primitive_desc
                = deconvolution_backward_data::primitive_desc(
                        deconv_bwd_data_desc, eng, deconv_primitive_desc);
        deconv_bwd_data_primitive_desc
                = deconvolution_backward_data::primitive_desc(
                        deconv_bwd_data_primitive_desc
                                .get()); // test construction from a C pd

        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == deconv_bwd_data_primitive_desc.diff_src_desc());
        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == deconv_bwd_data_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(deconv_bwd_data_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == deconv_bwd_data_primitive_desc.weights_desc());

        deconvolution_backward_data(deconv_bwd_data_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, dst->get()},
                                {DNNL_ARG_WEIGHTS, weights->get()},
                                {DNNL_ARG_DIFF_SRC, src->get()}});
        strm.wait();

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                algorithm::convolution_direct, *con_src_desc, *con_weights_desc,
                *con_dst_desc, {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto conv_primitive_desc
                = convolution_forward::primitive_desc(conv_desc, eng);

        convolution_forward(conv_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, conv_src->get()},
                                {DNNL_ARG_WEIGHTS, weights_tr},
                                {DNNL_ARG_DST, conv_dst.get()}});
        strm.wait();

        compare_data<data_t>(conv_dst.get(), src->get());
    }

    void BackwardWeights() {
        auto p = ::testing::TestWithParam<
                deconvolution_test_params_t>::GetParam();
        auto conv_src = dst;
        auto conv_dst = src;
        auto conv_weights = test::make_memory(*con_weights_desc, eng);
        test_convolution_sizes_t dd = p.sizes;

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());

        auto deconv_desc = deconvolution_forward::desc(
                prop_kind::forward_training, algorithm::deconvolution_direct,
                *dec_src_desc, *dec_weights_desc, *dec_bias_desc, *dec_dst_desc,
                {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto deconv_primitive_desc
                = deconvolution_forward::primitive_desc(deconv_desc, eng);

        auto deconv_bwd_weights_desc = deconvolution_backward_weights::desc(
                algorithm::deconvolution_direct, *dec_src_desc,
                *dec_weights_desc, *dec_bias_desc, *dec_dst_desc,
                {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);
        auto deconv_bwd_weights_primitive_desc
                = deconvolution_backward_weights::primitive_desc(
                        deconv_bwd_weights_desc, eng, deconv_primitive_desc);

        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_SRC)
                == deconv_bwd_weights_primitive_desc.src_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == deconv_bwd_weights_primitive_desc.diff_dst_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS)
                == deconv_bwd_weights_primitive_desc.diff_weights_desc());
        ASSERT_TRUE(deconv_bwd_weights_primitive_desc.query_md(
                            query::exec_arg_md, DNNL_ARG_DIFF_BIAS)
                == deconv_bwd_weights_primitive_desc.diff_bias_desc());

        deconvolution_backward_weights(deconv_bwd_weights_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, dst->get()},
                                {DNNL_ARG_SRC, src->get()},
                                {DNNL_ARG_DIFF_WEIGHTS, weights->get()},
                                {DNNL_ARG_DIFF_BIAS, bias->get()}});
        strm.wait();

        auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                algorithm::convolution_direct, *con_src_desc, *con_weights_desc,
                *con_dst_desc, {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);

        auto conv_primitive_desc
                = convolution_forward::primitive_desc(conv_desc, eng);

        auto conv_bwd_weights_desc = convolution_backward_weights::desc(
                algorithm::convolution_direct, *con_src_desc, *con_weights_desc,
                *con_dst_desc, {dd.strh, dd.strw}, {dd.padh, dd.padw}, padR);
        deconv_bwd_weights_primitive_desc
                = deconvolution_backward_weights::primitive_desc(
                        deconv_bwd_weights_primitive_desc
                                .get()); // test construction from a C pd

        auto conv_bwd_weights_primitive_desc
                = convolution_backward_weights::primitive_desc(
                        conv_bwd_weights_desc, eng, conv_primitive_desc);

        convolution_backward_weights(conv_bwd_weights_primitive_desc)
                .execute(strm,
                        {{DNNL_ARG_DIFF_DST, conv_dst->get()},
                                {DNNL_ARG_SRC, conv_src->get()},
                                {DNNL_ARG_DIFF_WEIGHTS, conv_weights}});
        strm.wait();

        auto weights_tr = test::make_memory(*con_weights_desc, eng);
        transpose_wei<data_t>(dd, weights->get(), weights_tr);

        compare_data<data_t>(weights_tr, conv_weights);

        if (with_bias) {
            auto ref_bias = test::make_memory(*dec_bias_desc, eng);
            compute_bias_bwd<data_t>(dd, dst->get(), ref_bias);
            compare_data<data_t>(ref_bias, bias->get());
        }
    }
};

using deconvolution_test_float = deconvolution_test_t<float>;

TEST_P(deconvolution_test_float, TestDeconvolution) {}

#define EXPAND_FORMATS(src, weights, bias, dst) \
    { \
        dnnl::memory::format_tag::src, dnnl::memory::format_tag::weights, \
                dnnl::memory::format_tag::bias, dnnl::memory::format_tag::dst \
    }

#define ALGORITHM dnnl::algorithm::deconvolution_direct

#define PARAMS(src, weights, bias, dst, ...) \
    deconvolution_test_params_t { \
        ALGORITHM, EXPAND_FORMATS(src, weights, bias, dst), {}, { \
            __VA_ARGS__ \
        } \
    }

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            str, deconvolution_test_float, ::testing::Values(__VA_ARGS__))
#define GPU_INST_TEST_CASE(str, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            str, deconvolution_test_float, ::testing::Values(__VA_ARGS__))

#define FMT_BIAS x
#define FMT_DATA_BLOCKED nChw8c
#define FMT_WEIGHTS_BLOCKED Ohwi8o
#define FMT_DATA_BLOCKED_GPU NChw16n16c
#define FMT_WEIGHTS_BLOCKED_GPU IOhw16i16o

CPU_INST_TEST_CASE(SimpleSmall_NCHW,
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, oihw, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1)

);

CPU_INST_TEST_CASE(SimpleSmall_Blocked,
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 12, 12, 32, 13, 13, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 4, 4, 32, 3, 3, 3, 3, 1, 1, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 2, 2, 32, 3, 3, 3, 3, 0, 0, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1, 1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 48, 13, 13, 32, 13, 13, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS,
                FMT_DATA_BLOCKED, 2, 1, 48, 11, 11, 32, 13, 13, 3, 3, 0, 0, 1,
                1));

GPU_INST_TEST_CASE(SimpleSmall_Blocked,
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 12, 12, 32, 10, 10, 3, 3, 0, 0,
                1, 1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 4, 4, 32, 3, 3, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 4, 4, 32, 4, 4, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 2, 2, 32, 3, 3, 3, 3, 0, 0, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 32, 2, 2, 32, 2, 2, 3, 3, 1, 1, 1,
                1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 48, 13, 13, 32, 13, 13, 3, 3, 1, 1,
                1, 1),
        PARAMS(FMT_DATA_BLOCKED_GPU, FMT_WEIGHTS_BLOCKED_GPU, FMT_BIAS,
                FMT_DATA_BLOCKED_GPU, 32, 1, 48, 11, 11, 32, 13, 13, 3, 3, 0, 0,
                1, 1));

GPU_INST_TEST_CASE(SimpleSmall_NCHW,
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nchw, oihw, x, nchw, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, oihw, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1),
        PARAMS(nhwc, hwio, x, nhwc, 2, 1, 6, 2, 2, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, goihw, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1),
        PARAMS(nhwc, hwigo, x, nhwc, 2, 2, 6, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1)

);

} // namespace dnnl
