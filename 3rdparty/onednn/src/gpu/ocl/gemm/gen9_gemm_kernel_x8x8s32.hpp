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

#ifndef GPU_OCL_GEMM_GEN9_GEMM_KERNEL_X8X8S32_HPP
#define GPU_OCL_GEMM_GEN9_GEMM_KERNEL_X8X8S32_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_int8_gemm_kernel_t {
    static status_t init_cl_options(compute::kernel_ctx_t &kernel_ctx,
            impl::data_type_t a_type, impl::data_type_t b_type,
            impl::data_type_t c_type) {
        using namespace data_type;

        if (c_type != s32) return status::unimplemented;

        kernel_ctx.define_int("DT_S32", c_type == s32);

        kernel_ctx.add_option("-cl-mad-enable");
        kernel_ctx.add_option("-cl-strict-aliasing");
#ifdef CL_VERSION_2_0
        kernel_ctx.add_option("-cl-std=CL2.0");
#else
        kernel_ctx.add_option("-Dget_enqueued_local_size=get_local_size");
#endif
        return status::success;
    }

    struct copy_params_t {
        static constexpr auto unroll_m = 32;
        static constexpr auto unroll_n = 16;
        static constexpr auto unroll_k = 4;
    };
};

struct gen9_gemm_x8x8s32_kernel_t : public gen9_int8_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            bool trans_a, bool trans_b, bool fixed_c, bool column_c, bool row_c,
            const attr_info_t &attr_info, impl::data_type_t a_type,
            impl::data_type_t b_type, impl::data_type_t c_type) {

        auto status = init_cl_options(kernel_ctx, a_type, b_type, c_type);
        if (status) return status;

        using namespace data_type;

        if ((a_type == s8) && (b_type == s8)) {
            kernel_ctx.add_option("-DS8S8");
        } else if ((a_type == u8) && (b_type == s8)) {
            kernel_ctx.add_option("-DU8S8");
        } else if ((a_type == s8) && (b_type == u8)) {
            kernel_ctx.add_option("-DS8U8");
        } else {
            kernel_ctx.add_option("-DU8U8");
        }

        if (!trans_a && !trans_b)
            kernel_ctx.add_option("-DNN");
        else if (trans_a && !trans_b)
            kernel_ctx.add_option("-DTN");
        else if (!trans_a && trans_b)
            kernel_ctx.add_option("-DNT");
        else
            kernel_ctx.add_option("-DTT");

        if (fixed_c)
            kernel_ctx.add_option("-DFF");
        else if (column_c)
            kernel_ctx.add_option("-DCC");
        else if (row_c)
            kernel_ctx.add_option("-DRR");
        else
            return status::unimplemented;

        kernel_ctx.define_int("UNROLL_M", copy_params_t::unroll_m);
        kernel_ctx.define_int("UNROLL_N", copy_params_t::unroll_n);
        kernel_ctx.define_int("UNROLL_K", copy_params_t::unroll_k);

        def_attr_info(kernel_ctx, attr_info);

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params_t::unroll_m;
        unroll_n = copy_params_t::unroll_n;
    }
};

struct gen9_gemm_scale_x8x8s32_kernel_t : public gen9_int8_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            const attr_info_t &attr_info, impl::data_type_t a_type,
            impl::data_type_t b_type, impl::data_type_t c_type) {

        auto status = init_cl_options(kernel_ctx, a_type, b_type, c_type);
        if (status) return status;

        kernel_ctx.define_int("UNROLL_M", copy_params_t::unroll_m);
        kernel_ctx.define_int("UNROLL_N", copy_params_t::unroll_n);
        kernel_ctx.define_int("UNROLL_K", copy_params_t::unroll_k);

        def_attr_info(kernel_ctx, attr_info);

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params_t::unroll_m;
        unroll_n = copy_params_t::unroll_n;
    }
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
