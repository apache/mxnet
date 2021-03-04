/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <memory>
#include <numeric>
#include <utility>
#include <type_traits>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {

using dt = memory::data_type;
using tag = memory::format_tag;
using md = memory::desc;

using ip_fwd = inner_product_forward;
using ip_bwd_d = inner_product_backward_data;
using ip_bwd_w = inner_product_backward_weights;

class ip_formats_test : public ::testing::Test {
public:
    engine e;

protected:
    virtual void SetUp() { e = get_test_engine(); }
};

HANDLE_EXCEPTIONS_FOR_TEST_F(ip_formats_test, TestChecksAllFormats) {
    SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
            "GPU takes a lot of time to complete this test.");
    static auto isa = get_effective_cpu_isa();
    bool supports_bf16 = isa >= cpu_isa::avx512_core;

    bool is_cpu = get_test_engine_kind() == engine::kind::cpu;

    memory::dims SP1D = {2};
    memory::dims SP2D = {2, 2};
    memory::dims SP3D = {2, 2, 2};
    memory::dims SP4D = {2, 2, 2, 2};
    memory::dims SP5D = {2, 2, 2, 2, 2};
    memory::dims SP6D = {2, 2, 2, 2, 2, 2};
    memory::dims SP7D = {2, 2, 2, 2, 2, 2, 2};
    memory::dims SP8D = {2, 2, 2, 2, 2, 2, 2, 2};
    memory::dims SP9D = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    memory::dims SP10D = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    memory::dims SP11D = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    memory::dims SP12D = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<memory::dims> v_dims = {SP1D, SP2D, SP3D, SP4D, SP5D, SP6D,
            SP7D, SP8D, SP9D, SP10D, SP11D, SP12D};
    std::vector<memory::dims> unsup_dims
            = {SP1D, SP6D, SP7D, SP8D, SP9D, SP10D, SP11D, SP12D};

    unsigned start_tag = static_cast<unsigned>(tag::any);
    unsigned end_tag = static_cast<unsigned>(tag::format_tag_last);

    md src_md, wei_md, dst_md;
    tag src_tag {tag::any}, wei_tag {tag::any}, dst_tag {tag::any};
    std::vector<std::vector<dt>> cfg {
            {dt::f32, dt::f32, dt::f32},
            {dt::bf16, dt::bf16, dt::bf16},
            {dt::u8, dt::s8, dt::u8},
    };

    for (const auto &i_cfg : cfg) {
        if (i_cfg[0] == dt::f16 && is_cpu) continue;

        bool cfg_has_bf16 = i_cfg[0] == dt::bf16 || i_cfg[1] == dt::bf16
                || i_cfg[2] == dt::bf16;
        if (cfg_has_bf16 && !supports_bf16) continue;

        for (unsigned stag = start_tag; stag < end_tag; stag++) {
            src_tag = static_cast<tag>(stag);

            // ip does not support 1D and 6D-12D cases
            bool skip_tag = false;
            for (const auto &i_dims : unsup_dims) {
                src_md = md(i_dims, i_cfg[0], src_tag, true);
                if (src_md) {
                    skip_tag = true;
                    break;
                }
            }
            if (skip_tag) continue;

            memory::dims cur_dims {};
            for (const auto &i_dims : v_dims) {
                src_md = md(i_dims, i_cfg[0], src_tag, true);
                if (src_md) {
                    cur_dims = i_dims;
                    break;
                }
            }
            ASSERT_TRUE(src_md);

            for (unsigned wtag = start_tag; wtag < end_tag; wtag++) {
                wei_tag = static_cast<tag>(wtag);
                wei_md = md(cur_dims, i_cfg[1], wei_tag, true);
                if (!wei_md) continue;

                dst_md = md(SP2D, i_cfg[2], dst_tag);
                ASSERT_TRUE(dst_md);

                ip_fwd::desc ip_fwd_desc(
                        prop_kind::forward_training, src_md, wei_md, dst_md);
                ip_fwd::primitive_desc ip_fwd_pd(ip_fwd_desc, e, true);
                if (ip_fwd_pd) {
                    auto ip_fwd_prim = ip_fwd(ip_fwd_pd);
                    auto strm = make_stream(ip_fwd_pd.get_engine());
                    auto src = test::make_memory(ip_fwd_pd.src_desc(), e);
                    auto wei = test::make_memory(ip_fwd_pd.weights_desc(), e);
                    auto dst = test::make_memory(ip_fwd_pd.dst_desc(), e);
                    ip_fwd_prim.execute(strm,
                            {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                                    {DNNL_ARG_DST, dst}});
                    strm.wait();
                }

                // no sense to test backward if forward was not created
                if (!ip_fwd_pd) continue;
                // int8 is not supported on backward;
                if (i_cfg[1] == dt::s8) continue;

                ip_bwd_d::desc ip_bwd_d_desc(src_md, wei_md, dst_md);
                ip_bwd_d::primitive_desc ip_bwd_d_pd(
                        ip_bwd_d_desc, e, ip_fwd_pd, true);
                if (ip_bwd_d_pd) {
                    auto ip_bwd_d_prim = ip_bwd_d(ip_bwd_d_pd);
                    auto strm = make_stream(ip_bwd_d_pd.get_engine());
                    auto d_src
                            = test::make_memory(ip_bwd_d_pd.diff_src_desc(), e);
                    auto d_wei
                            = test::make_memory(ip_bwd_d_pd.weights_desc(), e);
                    auto d_dst
                            = test::make_memory(ip_bwd_d_pd.diff_dst_desc(), e);
                    ip_bwd_d_prim.execute(strm,
                            {{DNNL_ARG_DIFF_SRC, d_src},
                                    {DNNL_ARG_WEIGHTS, d_wei},
                                    {DNNL_ARG_DIFF_DST, d_dst}});
                    strm.wait();
                }

                ip_bwd_w::desc ip_bwd_w_desc(src_md, wei_md, dst_md);
                ip_bwd_w::primitive_desc ip_bwd_w_pd(
                        ip_bwd_w_desc, e, ip_fwd_pd, true);
                if (ip_bwd_w_pd) {
                    auto ip_bwd_w_prim = ip_bwd_w(ip_bwd_w_pd);
                    auto strm = make_stream(ip_bwd_w_pd.get_engine());
                    auto src = test::make_memory(ip_bwd_w_pd.src_desc(), e);
                    auto d_wei = test::make_memory(
                            ip_bwd_w_pd.diff_weights_desc(), e);
                    auto d_dst
                            = test::make_memory(ip_bwd_w_pd.diff_dst_desc(), e);
                    ip_bwd_w_prim.execute(strm,
                            {{DNNL_ARG_SRC, src},
                                    {DNNL_ARG_DIFF_WEIGHTS, d_wei},
                                    {DNNL_ARG_DIFF_DST, d_dst}});
                    strm.wait();
                }
            }
        }
    }
}

} // namespace dnnl
