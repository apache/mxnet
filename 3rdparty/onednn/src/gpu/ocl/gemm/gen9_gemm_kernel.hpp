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

#ifndef GPU_OCL_GEMM_GEN9_GEMM_KERNEL_HPP
#define GPU_OCL_GEMM_GEN9_GEMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_gemm_kernel_t {
    static status_t init_cl_options(compute::kernel_ctx_t &kernel_ctx,
            impl::data_type_t type,
            impl::data_type_t src_type = impl::data_type::undef,
            impl::data_type_t dst_type = impl::data_type::undef) {
        using namespace data_type;
        if (src_type == impl::data_type::undef) src_type = type;
        if (dst_type == impl::data_type::undef) dst_type = type;
        if (type != f32 && type != f16) return status::unimplemented;

        def_data_type(kernel_ctx, src_type, "SRC");
        def_data_type(kernel_ctx, dst_type, "DST");

        kernel_ctx.define_int("DT_F32", type == f32);
        kernel_ctx.define_int("DT_F16", type == f16);

        kernel_ctx.add_option("-cl-mad-enable");
        kernel_ctx.add_option("-cl-strict-aliasing");
#if defined(CL_VERSION_2_0) && !defined(DNNL_SYCL_COMPUTECPP)
        kernel_ctx.add_option("-cl-std=CL2.0");
#else
        kernel_ctx.add_option("-Dget_enqueued_local_size=get_local_size");
#endif
#ifndef _WIN32
        kernel_ctx.add_option("-DALLOW_READ_OVERRUNS");
#endif
        return status::success;
    }

    struct copy_params_t {
        static constexpr auto unroll_m = 16;
        static constexpr auto unroll_n = 32;
    };
};

struct gen9_gemm_beta_kernel_t : public gen9_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            impl::data_type_t src_type,
            impl::data_type_t type = impl::data_type::undef) {
        if (type == impl::data_type::undef) type = src_type;
        auto status = init_cl_options(kernel_ctx, type, src_type, src_type);
        if (status) return status;

        kernel_ctx.print_options();
        return status::success;
    }
};

struct gen9_gemm_copy_kernel_t : public gen9_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            bool outer, bool trans, impl::data_type_t src_type,
            impl::data_type_t type = impl::data_type::undef) {
        if (type == impl::data_type::undef) type = src_type;
        auto status = init_cl_options(kernel_ctx, type, src_type);
        if (status) return status;

        kernel_ctx.define_int("COPY_UNROLL",
                !outer ? copy_params_t::unroll_m : copy_params_t::unroll_n);

        kernel_ctx.add_option(trans ? "-DUSE_TRANS" : "-DUSE_NOTRANS");

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params_t::unroll_m;
        unroll_n = copy_params_t::unroll_n;
    }
};

struct gen9_gemm_compute_kernel_t : public gen9_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            bool beta0, const attr_info_t &attr_info, impl::data_type_t type,
            impl::data_type_t dst_type = impl::data_type::undef) {
        if (dst_type == impl::data_type::undef) dst_type = type;
        auto status = init_cl_options(kernel_ctx, type, type, dst_type);
        if (status) return status;

        if (beta0) kernel_ctx.add_option("-DBETA_ZERO");

        kernel_ctx.define_int("UNROLL_M", copy_params_t::unroll_m);
        kernel_ctx.define_int("UNROLL_N", copy_params_t::unroll_n);

        def_attr_info(kernel_ctx, attr_info);

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params_t::unroll_m;
        unroll_n = copy_params_t::unroll_n;
    }
};

struct gen9_gemm_nocopy_kernel_t : public gen9_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            bool trans_a, bool trans_b, bool with_k_unroll, int unroll_k,
            const attr_info_t &attr_info, impl::data_type_t type) {

        auto status = init_cl_options(kernel_ctx, type);
        if (status) return status;

        if (trans_a) kernel_ctx.add_option("-DTRANS_A");
        if (trans_b) kernel_ctx.add_option("-DTRANS_B");
        if (with_k_unroll) {
            kernel_ctx.add_option("-DWITH_K_UNROLL");
            kernel_ctx.define_int("UNROLL_K", unroll_k);
        }
        def_attr_info(kernel_ctx, attr_info);
        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(bool trans_a, bool trans_b, int &unroll_m,
            int &unroll_n, int &unroll_k, impl::data_type_t type) {

        unroll_m = unroll_n = unroll_k = 0;

        if (type == data_type::f32) {
            static constexpr int unroll_m_table[2][2] = {{32, 32}, {16, 16}};
            static constexpr int unroll_n_table[2][2] = {{16, 16}, {32, 32}};

            unroll_m = unroll_m_table[trans_a][trans_b];
            unroll_n = unroll_n_table[trans_a][trans_b];
            unroll_k = 128;
        } else if (type == data_type::f16) {
            unroll_m = unroll_n = 32;
            unroll_k = 128;
        }
    }
};

struct gen9_gemm_nocopy_superkernel_t : public gen9_gemm_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            bool trans_a, bool trans_b, const attr_info_t &attr_info,
            impl::data_type_t type) {

        if (trans_a) return status::unimplemented;

        return gen9_gemm_nocopy_kernel_t::init_kernel_ctx(
                kernel_ctx, trans_a, trans_b, false, 32, attr_info, type);
    }

    static void get_unrolls(
            bool trans_a, bool trans_b, int (&unroll_m)[2], int &unroll_n) {

        unroll_m[0] = 32;
        unroll_m[1] = 16;
        unroll_n = 16;
    }
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
