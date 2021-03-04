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

#include "src/cpu/platform.hpp"

namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class runtime_attr_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    void SetUp() override {}

    static primitive_attr gen_attr_with_oscale(bool is_runtime) {
        primitive_attr attr;
        attr.set_output_scales(0, {is_runtime ? DNNL_RUNTIME_F32_VAL : 2.f});
        return attr;
    }

    static primitive_attr gen_attr_with_zp(
            bool is_runtime, int arg, int mask = 0) {
        primitive_attr attr;
        attr.set_zero_points(
                arg, mask, {is_runtime ? DNNL_RUNTIME_S32_VAL : 2});
        return attr;
    }

    template <typename F>
    static void check_status(const F &f, dnnl_status_t status) {
        catch_expected_failures(f, status != dnnl_success, status, false);
    }
};
#define CHECK_STATUs(status, ...) check_status([&]() { __VA_ARGS__; }, status)
#define CHECK_STATUS(status, ...) CHECK_STATUs(status, __VA_ARGS__)

#define CHECK_OK(...) CHECK_STATUS(dnnl_success, __VA_ARGS__)
#define CHECK_INVALID(...) CHECK_STATUS(dnnl_invalid_arguments, __VA_ARGS__)
#define CHECK_UNIMPL(...) CHECK_STATUS(dnnl_unimplemented, __VA_ARGS__)

// TODO: replace primitive descriptor creation with iterator fetching
//       to test all possible implementations

TEST_F(runtime_attr_test_t, TestBNorm) {
    for (auto dt : {data_type::f32, data_type::s8}) {
        // no s8 -> s8 batch norm on GPU yet
        if (get_test_engine_kind() == engine::kind::gpu && dt == data_type::s8)
            continue;

        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};
        normalization_flags flags = normalization_flags::use_global_stats;
        batch_normalization_forward::desc op_d(
                prop_kind::forward_inference, md, 0.1f, flags);
        CHECK_OK(batch_normalization_forward::primitive_desc(op_d, eng));
        CHECK_UNIMPL(batch_normalization_forward::primitive_desc(
                op_d, gen_attr_with_oscale(false), eng));
        CHECK_UNIMPL(batch_normalization_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS,
                     DNNL_ARG_MEAN, DNNL_ARG_VARIANCE, DNNL_ARG_DST}) {
            CHECK_UNIMPL(batch_normalization_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_UNIMPL(batch_normalization_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        }
    }
}

TEST_F(runtime_attr_test_t, TestBinary) {
    memory::desc md {{1, 16, 3, 3}, data_type::f32, tag::abcd};
    binary::desc op_d(algorithm::binary_add, md, md, md);
    CHECK_OK(binary::primitive_desc(op_d, eng));
    CHECK_UNIMPL(
            binary::primitive_desc(op_d, gen_attr_with_oscale(false), eng));
    CHECK_UNIMPL(binary::primitive_desc(op_d, gen_attr_with_oscale(true), eng));

    for (auto arg : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1, DNNL_ARG_DST}) {
        CHECK_UNIMPL(binary::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(
                binary::primitive_desc(op_d, gen_attr_with_zp(true, arg), eng));
    }
}

TEST_F(runtime_attr_test_t, TestConcat) {
    memory::desc md {{1, 16, 3, 3}, data_type::s8, tag::abcd};
    CHECK_OK(concat::primitive_desc(1, {md, md}, eng));
    CHECK_UNIMPL(concat::primitive_desc(
            1, {md, md}, eng, gen_attr_with_oscale(false)));
    CHECK_UNIMPL(concat::primitive_desc(
            1, {md, md}, eng, gen_attr_with_oscale(true)));

    for (auto arg :
            {DNNL_ARG_MULTIPLE_SRC, DNNL_ARG_MULTIPLE_SRC + 1, DNNL_ARG_DST}) {
        CHECK_UNIMPL(concat::primitive_desc(
                1, {md, md}, eng, gen_attr_with_zp(false, arg)));
        CHECK_UNIMPL(concat::primitive_desc(
                1, {md, md}, eng, gen_attr_with_zp(true, arg)));
    }
}

TEST_F(runtime_attr_test_t, TestConv) {
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::s32, tag::any};
    convolution_forward::desc op_d(prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1});
    CHECK_OK(convolution_forward::primitive_desc(op_d, eng));
    CHECK_OK(convolution_forward::primitive_desc(
            op_d, gen_attr_with_oscale(false), eng));
    if (get_test_engine_kind() == engine::kind::gpu) {
        CHECK_OK(convolution_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));
    } else {
        CHECK_UNIMPL(convolution_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));
    }

    for (auto arg :
            {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS, DNNL_ARG_DST}) {
        if ((src_md.data_type() == data_type::s8
                    || src_md.data_type() == data_type::u8)
                && (arg == DNNL_ARG_SRC || arg == DNNL_ARG_DST)) {
            CHECK_OK(convolution_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_OK(convolution_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        } else {
            CHECK_UNIMPL(convolution_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_UNIMPL(convolution_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        }
        CHECK_UNIMPL(convolution_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg, 1 << 1), eng));
    }
}

TEST_F(runtime_attr_test_t, TestDeconv) {
    memory::desc src_md {{1, 16, 7, 7}, data_type::f32, tag::any};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::f32, tag::any};
    memory::desc dst_md {{1, 32, 7, 7}, data_type::f32, tag::any};
    deconvolution_forward::desc op_d(prop_kind::forward,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1});
    // TODO: add u8s8 combination to replace CHECK_UNIMPL with CHECK_OK below
    CHECK_UNIMPL(deconvolution_forward::primitive_desc(
            op_d, gen_attr_with_oscale(false), eng));
    CHECK_UNIMPL(deconvolution_forward::primitive_desc(
            op_d, gen_attr_with_oscale(true), eng));

    for (auto arg :
            {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS, DNNL_ARG_DST}) {
        CHECK_UNIMPL(deconvolution_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(deconvolution_forward::primitive_desc(
                op_d, gen_attr_with_zp(true, arg), eng));
    }
}

TEST_F(runtime_attr_test_t, TestEltwise) {
    for (auto dt : {data_type::f32, data_type::s8}) {
        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};
        eltwise_forward::desc op_d(
                prop_kind::forward, algorithm::eltwise_relu, md, 0.f, 0.f);
        CHECK_OK(eltwise_forward::primitive_desc(op_d, eng));
        CHECK_UNIMPL(eltwise_forward::primitive_desc(
                op_d, gen_attr_with_oscale(false), eng));
        CHECK_UNIMPL(eltwise_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            CHECK_UNIMPL(eltwise_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_UNIMPL(eltwise_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        }
    }
}

TEST_F(runtime_attr_test_t, TestInnerProduct) {
    memory::desc src_md {{1, 16, 7, 7}, data_type::u8, tag::any};
    memory::desc wei_md {{32, 16, 7, 7}, data_type::s8, tag::any};
    memory::desc dst_md {{1, 32}, data_type::s32, tag::any};
    inner_product_forward::desc op_d(
            prop_kind::forward, src_md, wei_md, dst_md);
    CHECK_OK(inner_product_forward::primitive_desc(op_d, eng));
    CHECK_OK(inner_product_forward::primitive_desc(
            op_d, gen_attr_with_oscale(false), eng));
    CHECK_UNIMPL(inner_product_forward::primitive_desc(
            op_d, gen_attr_with_oscale(true), eng));

    for (auto arg :
            {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS, DNNL_ARG_DST}) {
        CHECK_UNIMPL(inner_product_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(inner_product_forward::primitive_desc(
                op_d, gen_attr_with_zp(true, arg), eng));
    }
}

TEST_F(runtime_attr_test_t, TestLNorm) {
    for (auto dt : {data_type::f32}) {
        memory::desc md {{1, 16, 16}, dt, tag::abc};
        memory::desc stat_md {{1, 16}, data_type::f32, tag::ab};
        normalization_flags flags = normalization_flags::use_global_stats;
        layer_normalization_forward::desc op_d(
                prop_kind::forward_inference, md, stat_md, 0.1f, flags);
        CHECK_OK(layer_normalization_forward::primitive_desc(op_d, eng));
        CHECK_UNIMPL(layer_normalization_forward::primitive_desc(
                op_d, gen_attr_with_oscale(false), eng));
        CHECK_UNIMPL(layer_normalization_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_MEAN, DNNL_ARG_VARIANCE,
                     DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS, DNNL_ARG_DST}) {
            CHECK_UNIMPL(layer_normalization_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_UNIMPL(layer_normalization_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        }
    }
}

TEST_F(runtime_attr_test_t, TestLRN) {
    for (auto dt : {data_type::f32}) {
        memory::desc md {{1, 16, 3, 3}, dt, tag::abcd};
        lrn_forward::desc op_d(prop_kind::forward_inference,
                algorithm::lrn_across_channels, md, 5, 1.f, 0.75f);
        CHECK_OK(lrn_forward::primitive_desc(op_d, eng));
        CHECK_UNIMPL(lrn_forward::primitive_desc(
                op_d, gen_attr_with_oscale(false), eng));
        CHECK_UNIMPL(lrn_forward::primitive_desc(
                op_d, gen_attr_with_oscale(true), eng));

        for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
            CHECK_UNIMPL(lrn_forward::primitive_desc(
                    op_d, gen_attr_with_zp(false, arg), eng));
            CHECK_UNIMPL(lrn_forward::primitive_desc(
                    op_d, gen_attr_with_zp(true, arg), eng));
        }
    }
}

CPU_TEST_F(runtime_attr_test_t, TestMatmul) {
    for (auto a_dt : {data_type::f32, data_type::u8}) {
        const data_type b_dt
                = a_dt == data_type::f32 ? data_type::f32 : data_type::s8;

        memory::desc a_md {{10, 3}, a_dt, tag::ab};
        memory::desc b_md {{3, 20}, b_dt, tag::ba};
        memory::desc c_md {{10, 20}, data_type::f32, tag::ab};

        matmul::desc op_d(a_md, b_md, c_md);
        CHECK_OK(matmul::primitive_desc(op_d, eng));
        CHECK_OK(
                matmul::primitive_desc(op_d, gen_attr_with_oscale(false), eng));
        CHECK_OK(matmul::primitive_desc(op_d, gen_attr_with_oscale(true), eng));

        for (auto arg :
                {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS, DNNL_ARG_DST}) {
            if ((a_dt != data_type::u8 && a_dt != data_type::s8)
                    || arg == DNNL_ARG_BIAS) {
                CHECK_UNIMPL(matmul::primitive_desc(
                        op_d, gen_attr_with_zp(false, arg), eng));
                CHECK_UNIMPL(matmul::primitive_desc(
                        op_d, gen_attr_with_zp(true, arg), eng));
            } else {
                CHECK_OK(matmul::primitive_desc(
                        op_d, gen_attr_with_zp(false, arg), eng));
                CHECK_OK(matmul::primitive_desc(
                        op_d, gen_attr_with_zp(true, arg), eng));
            }
        }
    }
}

TEST_F(runtime_attr_test_t, TestPool) {
    memory::desc src_md {{1, 16, 8, 8}, data_type::s8, tag::abcd};
    memory::desc dst_md {{1, 16, 4, 4}, data_type::s8, tag::abcd};
    pooling_forward::desc op_d(prop_kind::forward_inference,
            algorithm::pooling_max, src_md, dst_md, {2, 2}, {2, 2}, {0, 0},
            {0, 0});
    CHECK_OK(pooling_forward::primitive_desc(op_d, eng));
    CHECK_UNIMPL(pooling_forward::primitive_desc(
            op_d, gen_attr_with_oscale(false), eng));
    CHECK_UNIMPL(pooling_forward::primitive_desc(
            op_d, gen_attr_with_oscale(true), eng));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(pooling_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(pooling_forward::primitive_desc(
                op_d, gen_attr_with_zp(true, arg), eng));
    }
}

CPU_TEST_F(runtime_attr_test_t, TestReorder) {
    memory::desc src_md {{1, 16, 8, 8}, data_type::s8, tag::abcd};
    memory::desc dst_md {{1, 16, 8, 8}, data_type::s8, tag::acdb};
    CHECK_OK(reorder::primitive_desc(eng, src_md, eng, dst_md));
    CHECK_OK(reorder::primitive_desc(
            eng, src_md, eng, dst_md, gen_attr_with_oscale(false)));
    CHECK_OK(reorder::primitive_desc(
            eng, src_md, eng, dst_md, gen_attr_with_oscale(true)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_OK(reorder::primitive_desc(
                eng, src_md, eng, dst_md, gen_attr_with_zp(false, arg)));
        CHECK_OK(reorder::primitive_desc(
                eng, src_md, eng, dst_md, gen_attr_with_zp(true, arg)));
    }
}

TEST_F(runtime_attr_test_t, TestRNN) {
#if !DNNL_X64
    return;
#endif

    memory::dim n = 1, t = 1, l = 10, c = 8, g = 4, d = 1;
    memory::desc src_layer_md {{t, n, c}, data_type::u8, tag::tnc};
    memory::desc src_iter_md {{l, d, n, c}, data_type::u8, tag::ldnc};
    memory::desc src_iter_c_md {{l, d, n, c}, data_type::f32, tag::ldnc};
    memory::desc wei_layer_md {{l, d, c, g, c}, data_type::s8, tag::any};
    memory::desc wei_iter_md {{l, d, c, g, c}, data_type::s8, tag::any};
    memory::desc bia_md {{l, d, g, c}, data_type::f32, tag::ldgo};
    memory::desc dst_layer_md {{t, n, c}, data_type::u8, tag::tnc};
    memory::desc dst_iter_md {{l, d, n, c}, data_type::u8, tag::ldnc};
    memory::desc dst_iter_c_md {{l, d, n, c}, data_type::f32, tag::ldnc};
    lstm_forward::desc op_d(prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
            src_iter_c_md, wei_layer_md, wei_iter_md, bia_md, dst_layer_md,
            dst_iter_md, dst_iter_c_md);

    for_(auto is_runtime_data_scale : {true, false})
    for_(auto is_runtime_data_shift : {true, false})
    for_(auto is_runtime_weights_scale : {true, false})
    {
        primitive_attr attr;
        attr.set_rnn_data_qparams(
                is_runtime_data_scale ? DNNL_RUNTIME_F32_VAL : 2.f,
                is_runtime_data_shift ? DNNL_RUNTIME_F32_VAL : 2.f);
        attr.set_rnn_weights_qparams(
                0, {is_runtime_weights_scale ? DNNL_RUNTIME_F32_VAL : 2.f});
        bool rt = is_runtime_data_scale || is_runtime_data_shift
                || is_runtime_weights_scale;
        CHECK_STATUS(rt ? dnnl_unimplemented : dnnl_success,
                lstm_forward::primitive_desc(op_d, attr, eng));
    }

    for (auto arg : {DNNL_ARG_SRC_LAYER, DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C,
                 DNNL_ARG_WEIGHTS_LAYER, DNNL_ARG_WEIGHTS_ITER, DNNL_ARG_BIAS,
                 DNNL_ARG_DST_LAYER, DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C}) {
        CHECK_UNIMPL(lstm_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(lstm_forward::primitive_desc(
                op_d, gen_attr_with_zp(true, arg), eng));
    }
}

TEST_F(runtime_attr_test_t, TestShuffle) {
    memory::desc md {{1, 16, 3, 3}, data_type::f32, tag::abcd};
    shuffle_forward::desc op_d(prop_kind::forward, md, 1, 4);
    CHECK_OK(shuffle_forward::primitive_desc(op_d, eng));
    CHECK_UNIMPL(shuffle_forward::primitive_desc(
            op_d, eng, gen_attr_with_oscale(false)));
    CHECK_UNIMPL(shuffle_forward::primitive_desc(
            op_d, eng, gen_attr_with_oscale(true)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(shuffle_forward::primitive_desc(
                op_d, eng, gen_attr_with_zp(false, arg)));
        CHECK_UNIMPL(shuffle_forward::primitive_desc(
                op_d, eng, gen_attr_with_zp(true, arg)));
    }
}

TEST_F(runtime_attr_test_t, TestSoftmax) {
    memory::desc md {{2, 16}, data_type::f32, tag::ab};
    softmax_forward::desc op_d(prop_kind::forward, md, 1);
    CHECK_OK(softmax_forward::primitive_desc(op_d, eng));
    CHECK_UNIMPL(softmax_forward::primitive_desc(
            op_d, gen_attr_with_oscale(false), eng));
    CHECK_UNIMPL(softmax_forward::primitive_desc(
            op_d, gen_attr_with_oscale(true), eng));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(softmax_forward::primitive_desc(
                op_d, gen_attr_with_zp(false, arg), eng));
        CHECK_UNIMPL(softmax_forward::primitive_desc(
                op_d, gen_attr_with_zp(true, arg), eng));
    }
}

TEST_F(runtime_attr_test_t, TestSum) {
    memory::desc md {{1, 16, 3, 3}, data_type::s8, tag::abcd};
    CHECK_OK(sum::primitive_desc({1.f, 1.f}, {md, md}, eng));
    CHECK_UNIMPL(sum::primitive_desc(
            {1.f, 1.f}, {md, md}, eng, gen_attr_with_oscale(false)));
    CHECK_UNIMPL(sum::primitive_desc(
            {1.f, 1.f}, {md, md}, eng, gen_attr_with_oscale(true)));

    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        CHECK_UNIMPL(sum::primitive_desc(
                {1.f, 1.f}, {md, md}, eng, gen_attr_with_zp(false, arg)));
        CHECK_UNIMPL(sum::primitive_desc(
                {1.f, 1.f}, {md, md}, eng, gen_attr_with_zp(true, arg)));
    }
}

} // namespace dnnl
