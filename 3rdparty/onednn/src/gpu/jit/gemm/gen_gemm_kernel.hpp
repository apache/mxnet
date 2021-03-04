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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gen_gemm_kernel_generator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/jit_generator_base.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen_gemm_kernel_t : public jit_generator_base {

    status_t init_gemm(compute::gpu_arch_t arch) {
        hw_ = convert_dnnl_arch_to_hw(arch);
        if (hw_ == ngen::HW::Unknown) return status::unimplemented;

        auto status = complete_strategy();
        if (status != status::success) return status;

        return init_interface();
    }

    const char *kernel_name() const override { return "gemm_kernel"; }
    std::vector<unsigned char> get_binary(
            cl_context ctx, cl_device_id dev) override;

    CommonDriverInfo driver_info() const;

protected:
    static Type convert_dnnl_to_kernel_type(data_type_t type) {
        switch (type) {
            default: assert(!"Unknown type");
            case data_type::f32: return Type::f32;
            case data_type::f16: return Type::f16;
            case data_type::s32: return Type::s32;
            case data_type::u8: return Type::u8;
            case data_type::s8: return Type::s8;
        }
    }

    static ngen::HW convert_dnnl_arch_to_hw(compute::gpu_arch_t arch) {
        switch (arch) {
            case compute::gpu_arch_t::gen9: return ngen::HW::Gen9;
            case compute::gpu_arch_t::gen12lp: return ngen::HW::Gen12LP;
            default: return ngen::HW::Unknown;
        }
    }

    ngen::HW hw_ = ngen::HW::Unknown;
    GEMMProblem problem_;
    GEMMStrategy strategy_;
    ngen::NEOInterfaceHandler interface_ {ngen::HW::Unknown};

private:
    status_t complete_strategy();
    status_t read_strategy(const char *str);
    status_t init_interface();
};

struct gen_gemm_nocopy_kernel_t : public gen_gemm_kernel_t {
    status_t init(compute::gpu_arch_t arch, bool batch, bool trans_a,
            bool trans_b, bool c_offset, data_type_t a_type, data_type_t b_type,
            data_type_t c_type, int unroll_m, int unroll_n) {

        problem_.Ta = convert_dnnl_to_kernel_type(a_type);
        problem_.Tb = convert_dnnl_to_kernel_type(b_type);
        problem_.Tc = convert_dnnl_to_kernel_type(c_type);
        problem_.Ts = problem_.Tc;
        problem_.A.layout = trans_a ? MatrixLayout::T : MatrixLayout::N;
        problem_.B.layout = trans_b ? MatrixLayout::T : MatrixLayout::N;
        problem_.C.layout = MatrixLayout::N;
        problem_.A.crosspack = problem_.B.crosspack = problem_.C.crosspack = 1;
        problem_.A.packSize = problem_.B.packSize = problem_.C.packSize = 0;
        problem_.A.padded = problem_.B.padded = problem_.C.padded = false;
        problem_.A.alignment = uint8_t(types::data_type_size(a_type));
        problem_.B.alignment = uint8_t(types::data_type_size(b_type));
        problem_.C.alignment = uint8_t(types::data_type_size(c_type));
        problem_.A.base = ngen::AddressBase::createA64(true);
        problem_.B.base = ngen::AddressBase::createA64(true);
        problem_.C.base = ngen::AddressBase::createA64(true);
        problem_.batchedS = batch;
        if (c_type == data_type::s32) {
            problem_.abOffset = ABOffset::Calc;
            problem_.Ts = Type::f32;
        }
        if (c_offset) {
            problem_.CO.base = ngen::AddressBase::createBTS(0);
            problem_.cOffset = COffset::Post;
            problem_.CO.crosspack = 1;
            problem_.CO.padded = false;
            problem_.CO.alignment = problem_.C.alignment;
        }

        strategy_.unroll[LoopM] = unroll_m;
        strategy_.unroll[LoopN] = unroll_n;

        return init_gemm(arch);
    }

    static void choose_unrolls(compute::gpu_arch_t arch, int hw_threads,
            bool trans_a, bool trans_b, data_type_t a_type, data_type_t b_type,
            data_type_t c_type, dim_t m, dim_t n, dim_t k, dim_t batch,
            int &unroll_m, int &unroll_n);
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
