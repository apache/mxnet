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
#include <numeric>
#include "cpu/x64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector_utils {

static std::size_t get_vmm_size_bytes(const Xbyak::Xmm &vmm) {
    static constexpr int byte_size_bits = 8;
    return vmm.getBit() / byte_size_bits;
}

static std::size_t calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak::Xmm> &vmm_to_preserve) {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u), [](std::size_t accum, const Xbyak::Xmm &vmm) {
                return accum + get_vmm_size_bytes(vmm);
            });
}

register_preserve_guard_t::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
        std::initializer_list<Xbyak::Xmm> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve)
        host_->push(reg);

    if (!vmm_stack_.empty()) {
        host_->sub(host_->rsp, vmm_to_preserve_size_bytes_);

        auto stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            stack_offset -= get_vmm_size_bytes(vmm);
            const auto idx = vmm.getIdx();
            if (vmm.isXMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Xmm(idx));
            else if (vmm.isYMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Ymm(idx));
            else
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Zmm(idx));
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;

    while (!vmm_stack_.empty()) {
        const Xbyak::Xmm &vmm = vmm_stack_.top();
        const auto idx = vmm.getIdx();
        if (vmm.isXMM())
            host_->uni_vmovups(
                    Xbyak::Xmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else if (vmm.isYMM())
            host_->uni_vmovups(
                    Xbyak::Ymm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else
            host_->uni_vmovups(
                    Xbyak::Zmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);

        tmp_stack_offset += get_vmm_size_bytes(vmm);
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add(host_->rsp, vmm_to_preserve_size_bytes_);

    while (!reg64_stack_.empty()) {
        host_->pop(reg64_stack_.top());
        reg64_stack_.pop();
    }
}

output_dims_t make_output_dims(const memory_desc_wrapper &dst_d) {

    const dim_t n_dims = dst_d.ndims();
    const auto dims = dst_d.dims();
    const dim_t &mb = dims[0];
    const dim_t &oc = n_dims >= 2 ? dims[1] : 1;
    const dim_t &ow = n_dims >= 3 ? dims[n_dims - 1] : 1;
    const dim_t &oh = n_dims >= 4 ? dims[n_dims - 2] : 1;
    const dim_t &od = n_dims >= 5 ? dims[n_dims - 3] : 1;

    switch (n_dims) {
        case 1: return output_dims_t {{mb, 0, 0, 0, 0}};
        case 2: return output_dims_t {{mb, 0, 0, 0, 0}};
        case 3: return output_dims_t {{mb, oc, ow, 0, 0}};
        case 4: return output_dims_t {{mb, oc, oh, ow, 0}};
        case 5: return output_dims_t {{mb, oc, od, oh, ow}};
        default: assert(!"dimension count error"); break;
    }

    return output_dims_t();
}

} // namespace injector_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
