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

#ifndef CPU_X64_JIT_GENERATOR_HPP
#define CPU_X64_JIT_GENERATOR_HPP

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/x64/jit_utils/jit_utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#if defined(_WIN32)
#define OFFSET_SHADOWSPACE 0x28
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

static inline void tc_configure_tile(
        palette_config_t *tc, int t, int rows, int cols) {
    tc->rows[t] = rows;
    tc->cols[t] = cols;
}

// TODO: A GPR class that hides ABI details from the JIT kernels and allows
// numbering registers from 0 to 14 (x86_64) / 6 (x32) (gpr0, gpr1, ...) and
// stack register (sr).
//
// This will allow using syntax like this:
//
// param = gpr0;
// reg_input = gpr0;
// reg_output =  gpr1;
// ...
//
// #ifndef XBYAK64
// mov(param, ptr[sr])
// #endif
//
// (Roma)

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
        Xbyak::Operand::RBX,
        Xbyak::Operand::RBP,
        Xbyak::Operand::R12,
        Xbyak::Operand::R13,
        Xbyak::Operand::R14,
        Xbyak::Operand::R15,
#ifdef _WIN32
        Xbyak::Operand::RDI,
        Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN32
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
        abi_param2(Xbyak::Operand::RDX), abi_param3(Xbyak::Operand::R8),
        abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
        abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
        abi_param4(Xbyak::Operand::RCX), abi_param5(Xbyak::Operand::R8),
        abi_param6(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RCX);
#endif
#endif

} // namespace

class jit_generator : public Xbyak::CodeGenerator {
private:
    const size_t xmm_len = 16;
#ifdef _WIN32
    const size_t xmm_to_preserve_start = 6;
    const size_t xmm_to_preserve = 10;
#else
    const size_t xmm_to_preserve_start = 0;
    const size_t xmm_to_preserve = 0;
#endif

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t size_of_abi_save_regs
            = num_abi_save_gpr_regs * rax.getBit() / 8
            + xmm_to_preserve * xmm_len;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    Xbyak::Reg64 param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        if (xmm_to_preserve) {
            sub(rsp, xmm_to_preserve * xmm_len);
            for (size_t i = 0; i < xmm_to_preserve; ++i)
                movdqu(ptr[rsp + i * xmm_len],
                        Xbyak::Xmm(xmm_to_preserve_start + i));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            push(Xbyak::Reg64(abi_save_gpr_regs[i]));
        if (mayiuse(avx512_common)) {
            mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        }
    }

    // This function returns the address on the stack of the fist argument
    // that is not passed by register
    // By default it assumes to be called after the prologue
    // Note: that we cannot use RBP inside as we override it in preamble
    // for address computation in EVEX instructions
    inline const Xbyak::RegExp get_stack_params_address(
            bool after_prolog = true) {
        int saved_regs_size = after_prolog ? get_size_of_abi_save_regs() : 0;
#ifdef _WIN32
        // Using stack layout described in MS ABI
        // (https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=vs-2019)
        // here, the return address and the first 4 parameters are allocated
        // on the stack
        int first_params_and_return_addr_size = 40;
#else
        // In System V ABI, only the return address is stacked
        // before the arguments
        int first_params_and_return_addr_size = 8;
#endif
        return rsp + saved_regs_size + first_params_and_return_addr_size;
    }

    void mic_prefetcht0(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) prefetcht0(a);
    }

    void mic_prefetcht1(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) prefetcht1(a);
    }

    void mic_prefetcht2(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) prefetcht2(a);
    }

    void uni_vzeroupper() {
        if (mayiuse(avx) && !mayiuse(avx512_mic)) vzeroupper();
    }

    void postamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        if (xmm_to_preserve) {
            for (size_t i = 0; i < xmm_to_preserve; ++i)
                movdqu(Xbyak::Xmm(xmm_to_preserve_start + i),
                        ptr[rsp + i * xmm_len]);
            add(rsp, xmm_to_preserve * xmm_len);
        }
        uni_vzeroupper();
        ret();
    }

    template <typename T>
    Xbyak::Address EVEX_compress_addr(
            Xbyak::Reg64 base, T raw_offt, bool bcast = false) {
        using Xbyak::Address;
        using Xbyak::Reg64;
        using Xbyak::RegExp;
        using Xbyak::Zmm;

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt
                && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = RegExp() + base + offt;
        if (scale) re = re + reg_EVEX_max_8b_offt * scale;

        if (bcast)
            return zword_b[re];
        else
            return zword[re];
    }

    Xbyak::Address make_safe_addr(const Xbyak::Reg64 &reg_out, size_t offt,
            const Xbyak::Reg64 &tmp_reg, bool bcast = false) {
        if (offt > INT_MAX) {
            mov(tmp_reg, offt);
            return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
        } else {
            return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
        }
    }

    Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64 &base,
            size_t raw_offt, const Xbyak::Reg64 &reg_offt, bool bcast = false) {
        if (raw_offt > INT_MAX) {
            return make_safe_addr(base, raw_offt, reg_offt, bcast);
        } else {
            return EVEX_compress_addr(base, raw_offt, bcast);
        }
    }

    void safe_add(const Xbyak::Reg64 &base, size_t raw_offt,
            const Xbyak::Reg64 &reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            add(base, reg_offt);
        } else {
            add(base, raw_offt);
        }
    }

    void safe_sub(const Xbyak::Reg64 &base, size_t raw_offt,
            const Xbyak::Reg64 &reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            sub(base, reg_offt);
        } else {
            sub(base, raw_offt);
        }
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak::Label &label) { Xbyak::CodeGenerator::L(label); }

    void L_aligned(Xbyak::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    void uni_vpxor(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx512_core))
            vpxord(x1, x2, op);
        else if (mayiuse(avx))
            vpxor(x1, x2, op);
        else {
            assert(x1.isEqualIfNotInherited(x2));
            pxor(x2, op);
        }
    }
    void uni_vpxor(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx512_core))
            vpxord(x1, x2, op);
        else if (mayiuse(avx2))
            vpxor(x1, x2, op);
        else
            vxorps(x1, x2, op);
    }
    void uni_vpxor(const Xbyak::Zmm &x1, const Xbyak::Zmm &x2,
            const Xbyak::Operand &op) {
        vpxord(x1, x2, op);
    }

    void uni_vmovss(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        if (mayiuse(avx))
            vmovss(addr, x);
        else
            movss(addr, x);
    }
    void uni_vmovss(const Xbyak::Xmm &x, const Xbyak::Address &addr) {
        if (mayiuse(avx))
            vmovss(x, addr);
        else
            movss(x, addr);
    }
    void uni_vmovss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2) {
        if (mayiuse(avx))
            vmovss(x1, x1, x2);
        else
            movss(x1, x2);
    }
    void uni_vmovss(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovss(addr, Xbyak::Xmm(x.getIdx()));
    }
    void uni_vmovss(const Xbyak::Ymm &x, const Xbyak::Address &addr) {
        vmovss(Xbyak::Xmm(x.getIdx()), addr);
    }
    void uni_vmovss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2) {
        vmovss(Xbyak::Xmm(x1.getIdx()), Xbyak::Xmm(x2.getIdx()));
    }

    void uni_vmovsd(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movsd(addr, x);
    }
    void uni_vmovsd(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovsd(addr, x);
    }
    void uni_vmovsd(const Xbyak::Xmm &x, const Xbyak::Address &addr) {
        movsd(x, addr);
    }
    void uni_vmovsd(const Xbyak::Ymm &x, const Xbyak::Address &addr) {
        vmovsd(x, addr);
    }

    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovdqu(addr, x);
    }
    void uni_vmovdqu(const Xbyak::Address &addr, const Xbyak::Zmm &x) {
        vmovdqu32(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Xmm &x, const Xbyak::Address &addr) {
        movdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Ymm &x, const Xbyak::Address &addr) {
        vmovdqu(x, addr);
    }
    void uni_vmovdqu(const Xbyak::Zmm &x, const Xbyak::Address &addr) {
        vmovdqu32(x, addr);
    }

    void uni_vmovups(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movups(addr, x);
    }
    void uni_vmovups(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovups(addr, x);
    }

    void uni_vmovups(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movups(x, op);
    }
    void uni_vmovups(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vmovups(x, op);
    }

    void uni_vmovups_tail(const Xbyak::Address &addr, const Xbyak::Ymm &mask,
            const Xbyak::Ymm &x) {
        vmaskmovps(addr, mask, x);
    }
    void uni_vmovups_tail(const Xbyak::Ymm &x, const Xbyak::Ymm &mask,
            const Xbyak::Address &addr) {
        vmaskmovps(x, mask, addr);
    }

    void uni_vmovups_tail(const Xbyak::Address &addr, const Xbyak::Opmask &mask,
            const Xbyak::Zmm &x) {
        vmovups(addr | mask, x);
    }
    void uni_vmovups_tail(const Xbyak::Zmm &x, const Xbyak::Opmask &mask,
            const Xbyak::Address &addr) {
        vmovups(x | mask | T_z, addr);
    }

    void uni_vmovntps(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        movntps(addr, x);
    }
    void uni_vmovntps(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovntps(addr, x);
    }

    void uni_vbroadcastss(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movss(x, op);
        shufps(x, x, 0x0);
    }
    void uni_vbroadcastss(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        if (op.isMEM() || mayiuse(avx2)) {
            vbroadcastss(x, op);
        } else {
            Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) movss(t, op);
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        movss(x, op);
        pshufd(x, x, 0x0);
    }
    void uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        if (mayiuse(avx2)) {
            vpbroadcastd(x, op);
        } else {
            const Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                if (op.isMEM())
                    vmovss(t, op.getAddress());
                else
                    vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vshufps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, Xbyak::uint8 imm) {
        if (mayiuse(avx))
            vshufps(x1, x2, op, imm);
        else {
            movups(x1, x2);
            shufps(x1, op, imm);
        }
    }

    void uni_vrcpss(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        rcpss(x, op);
    }
    void uni_vrcpss(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2) {
        Xbyak::Xmm x1_(x1.getIdx());
        Xbyak::Xmm x2_(x2.getIdx());
        vrcpss(x1_, x1_, x2_);
    }
    void uni_vrcpss(const Xbyak::Ymm &x, const Xbyak::Address &op) {
        Xbyak::Xmm x_(x.getIdx());
        vrcpss(x_, x_, op);
    }

    void uni_vrcpps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        rcpps(x, op);
    }
    void uni_vrcpps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vrcpps(x, op);
    }
    void uni_vrcpps(const Xbyak::Zmm &x, const Xbyak::Operand &op) {
        vrcp14ps(x, op);
    }

    void uni_vdivps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        divps(x, op2);
    }
    void uni_vdivps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vdivps(x, op1, op2);
    }

    void uni_vdivps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2, const Xbyak::Xmm &buf) {
        movups(buf, op1);
        divps(buf, op2);
        if (x.getIdx() != buf.getIdx()) { movups(x, buf); }
    }

    void uni_vdivps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2, const Xbyak::Ymm &buf) {
        vdivps(x, op1, op2);
    }

    void uni_vaddps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        addps(x, op2);
    }
    void uni_vaddps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vaddps(x, op1, op2);
    }
    void uni_vaddss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        addss(x, op2);
    }
    void uni_vaddss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vaddss(x, op1, op2);
    }

    void uni_vpsignd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        assert(x1.getIdx() == x2.getIdx());
        psignd(x1, op);
    }
    void uni_vpsignd(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        vpsignd(x1, x2, op);
    }

    void uni_vpsubd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        psubd(x1, op);
    }
    void uni_vpsubd(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        vpsubd(x1, x2, op);
    }

    void uni_vpsubb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        psubb(x1, op);
    }
    void uni_vpsubb(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        vpsubb(x1, x2, op);
    }

    void uni_vsubss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        subps(x, op2);
    }
    void uni_vsubss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vsubss(x, Xbyak::Xmm(op1.getIdx()), Xbyak::Xmm(op2.getIdx()));
    }

    void uni_vsubps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        subps(x, op2);
    }
    void uni_vsubps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vsubps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2, const Xbyak::Xmm &buf) {
        movups(buf, op1);
        subps(buf, op2);
        if (x.getIdx() != buf.getIdx()) { movups(x, buf); }
    }

    void uni_vsubps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2, const Xbyak::Ymm &buf) {
        vsubps(x, op1, op2);
    }

    void uni_vpmulld(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        if (mayiuse(avx)) {
            vpmulld(x1, x2, op);
        } else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            pmulld(x1, op);
        }
    }
    void uni_vpmulld(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        vpmulld(x1, x2, op);
    }

    void uni_vmulps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        if (mayiuse(avx))
            vmulps(x, op1, op2);
        else {
            assert(x.isEqualIfNotInherited(op1));
            mulps(x, op2);
        }
    }
    void uni_vmulps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vmulps(x, op1, op2);
    }

    void uni_vmulss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        mulss(x, op2);
    }
    void uni_vmulss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Address &op2) {
        vmulss(x, Xbyak::Xmm(op1.getIdx()), op2);
    }
    void uni_vmulss(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Ymm &op2) {
        vmulss(x, Xbyak::Xmm(op1.getIdx()), Xbyak::Xmm(op2.getIdx()));
    }

    void uni_vfmadd132ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x1 gets overriden by x1*op
        // This is incorrect if x1 == x2
        assert(x1.getIdx() != x2.getIdx());
        mulps(x1, op);
        addps(x1, x2);
    }
    void uni_vfmadd132ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmadd132ps(x1, x2, op);
        else {
            // Note: x1 gets overriden by x1*op
            // This is incorrect if x1 == x2
            assert(x1.getIdx() != x2.getIdx());
            vmulps(x1, x1, op);
            vaddps(x1, x1, x2);
        }
    }

    void uni_vfmadd213ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x1 gets overriden by x1*x2
        // This is incorrect if x1 == op
        assert(!x1.isEqualIfNotInherited(op));
        mulps(x1, x2);
        addps(x1, op);
    }
    void uni_vfmadd213ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmadd213ps(x1, x2, op);
        else {
            // Note: x1 gets overriden by x1*x2
            // This is incorrect if x1 == op
            assert(!x1.isEqualIfNotInherited(op));
            vmulps(x1, x1, x2);
            vaddps(x1, x1, op);
        }
    }

    void uni_vfmadd213ss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x1 gets overriden by x1*x2
        // This is incorrect if x1 == op
        assert(!x1.isEqualIfNotInherited(op));
        mulss(x1, x2);
        addss(x1, op);
    }
    void uni_vfmadd213ss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmadd213ss(x1, x2, op);
        else {
            // Note: x1 gets overriden by x1*x2
            // This is incorrect if x1 == op
            assert(!x1.isEqualIfNotInherited(op));
            vmulss(x1, x1, x2);
            vaddss(x1, x1, op);
        }
    }

    void uni_vfmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x2 gets overriden by x2*op
        // This is incorrect if x1 == x2
        assert(x1.getIdx() != x2.getIdx());
        mulps(x2, op);
        addps(x1, x2);
    }
    void uni_vfmadd231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmadd231ps(x1, x2, op);
        else {
            // Note: x2 gets overriden by x2*op
            // This is incorrect if x1 == x2
            assert(x1.getIdx() != x2.getIdx());
            vmulps(x2, x2, op);
            vaddps(x1, x1, x2);
        }
    }
    void uni_vfmadd231ss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x2 gets overriden by x2*op
        // This is incorrect if x1 == x2
        assert(x1.getIdx() != x2.getIdx());
        mulss(x2, op);
        addss(x1, x2);
    }
    void uni_vfmadd231ss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmadd231ss(Xbyak::Xmm(x1.getIdx()), Xbyak::Xmm(x2.getIdx()), op);
        else {
            // Note: x2 gets overriden by x2*op
            // This is incorrect if x1 == x2
            assert(x1.getIdx() != x2.getIdx());
            vmulss(x2, x2, op);
            vaddss(x1, x1, x2);
        }
    }

    void uni_vfnmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x2 gets overriden by x2*op
        // This is incorrect if x1 == x2
        assert(x1.getIdx() != x2.getIdx());
        mulps(x2, op);
        subps(x1, x2);
    }

    void uni_vfnmadd231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfnmadd231ps(x1, x2, op);
        else {
            // Note: x2 gets overriden by x2*op
            // This is incorrect if x1 == x2
            assert(x1.getIdx() != x2.getIdx());
            vmulps(x2, x2, op);
            vsubps(x1, x1, x2);
        }
    }

    void uni_vfmsub213ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        // Note: x1 gets overriden by x1*x2
        // This is incorrect if x1 == op
        assert(!x1.isEqualIfNotInherited(op));
        mulps(x1, x2);
        subps(x1, op);
    }
    void uni_vfmsub213ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx2))
            vfmsub213ps(x1, x2, op);
        else {
            // Note: x1 gets overriden by x1*x2
            // This is incorrect if x1 == op
            assert(!x1.isEqualIfNotInherited(op));
            vmulps(x1, x1, x2);
            vsubps(x1, x1, op);
        }
    }

    void uni_vsqrtps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        sqrtps(x, op);
    }
    void uni_vsqrtps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vsqrtps(x, op);
    }

    void uni_vpaddd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx))
            vpaddd(x1, x2, op);
        else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            paddd(x1, op);
        }
    }
    void uni_vpaddd(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        vpaddd(x1, x2, op);
    }

    void uni_vpaddb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx))
            vpaddb(x1, x2, op);
        else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            paddb(x1, op);
        }
    }
    void uni_vpaddb(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        vpaddb(x1, x2, op);
    }

    void uni_vpmaddwd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx))
            vpmaddwd(x1, x2, op);
        else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            pmaddwd(x1, op);
        }
    }
    void uni_vpmaddwd(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        vpmaddwd(x1, x2, op);
    }

    void uni_vpmaddubsw(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx))
            vpmaddubsw(x1, x2, op);
        else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            pmaddubsw(x1, op);
        }
    }
    void uni_vpmaddubsw(const Xbyak::Ymm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        vpmaddubsw(x1, x2, op);
    }

    void uni_vandps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        andps(x1, op);
    }
    void uni_vandps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        if (!mayiuse(avx512_common) || x1.getBit() < 512)
            vandps(x1, x2, op);
        else
            vpandd(x1, x2, op);
    }

    void uni_vorps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        assert(x1.getIdx() == x2.getIdx());
        orps(x1, op);
    }
    void uni_vorps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        if (!mayiuse(avx512_common) || x1.getBit() < 512)
            vorps(x1, x2, op);
        else
            vpord(x1, x2, op);
    }

    void uni_vxorps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        if (x1.getIdx() != x2.getIdx()) { uni_vmovups(x1, x2); }
        xorps(x1, op);
    }
    void uni_vxorps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op = Xbyak::Operand()) {
        if (!mayiuse(avx512_common) || x1.getBit() < 512)
            vxorps(x1, x2, op);
        else
            vpxord(x1, x2, op);
    }

    void uni_vpslld(
            const Xbyak::Xmm &x, const Xbyak::Operand &op, const int imm) {
        assert(x.isEqualIfNotInherited(op));
        pslld(x, imm);
    }
    void uni_vpslld(
            const Xbyak::Ymm &x, const Xbyak::Operand &op, const int imm) {
        vpslld(x, op, imm);
    }

    void uni_vpsrld(
            const Xbyak::Xmm &x, const Xbyak::Operand &op, const int imm) {
        if (!x.isEqualIfNotInherited(op)) uni_vmovups(x, op);
        psrld(x, imm);
    }
    void uni_vpsrld(
            const Xbyak::Ymm &x, const Xbyak::Operand &op, const int imm) {
        vpsrld(x, op, imm);
    }

    void uni_vmaxps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        maxps(x, op2);
    }
    void uni_vmaxps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vmaxps(x, op1, op2);
    }

    void uni_vminps(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        assert(x.isEqualIfNotInherited(op1));
        minps(x, op2);
    }
    void uni_vminps(const Xbyak::Ymm &x, const Xbyak::Operand &op1,
            const Xbyak::Operand &op2 = Xbyak::Operand()) {
        vminps(x, op1, op2);
    }

    void uni_vpmovsxbd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        pmovsxbd(x, op);
    }

    void uni_vpmovsxbd(const Xbyak::Ymm &y, const Xbyak::Operand &op) {
        vpmovsxbd(y, op);
    }

    void uni_vpmovzxbd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        pmovzxbd(x, op);
    }

    void uni_vpmovzxbd(const Xbyak::Ymm &y, const Xbyak::Operand &op) {
        vpmovzxbd(y, op);
    }

    void uni_vcmpps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, int cmp_predicate) {
        if (x1.getIdx() != x2.getIdx()) uni_vmovups(x1, x2);
        cmpps(x1, op, cmp_predicate);
    }
    void uni_vcmpps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, int cmp_predicate) {
        vcmpps(x1, x2, op, cmp_predicate);
    }

    void uni_vtestps(const Xbyak::Xmm &x1, const Xbyak::Operand &op) {
        ptest(x1, op);
    }

    void uni_vtestps(const Xbyak::Ymm &x1, const Xbyak::Operand &op) {
        assert(!(x1.isZMM() || op.isZMM()));
        vtestps(x1, op);
    }

    void uni_vblendvps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const Xbyak::Xmm &msk) {
        assert(x1.getIdx() == x2.getIdx());
        assert(msk.getIdx() == 0);
        blendvps(x1, op);
    }
    void uni_vblendvps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, const Xbyak::Ymm &msk) {
        vblendvps(x1, x2, op, msk);
    }

    void uni_vroundps(
            const Xbyak::Xmm &x, const Xbyak::Operand &op, const int imm) {
        roundps(x, op, imm);
    }
    void uni_vroundps(
            const Xbyak::Ymm &x, const Xbyak::Operand &op, const int imm) {
        vroundps(x, op, imm);
    }
    void uni_vroundps(
            const Xbyak::Zmm &x, const Xbyak::Operand &op, const int imm) {
        vrndscaleps(x, op, imm & 0x3);
    }

    void uni_vcvtps2dq(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        cvtps2dq(x, op);
    }
    void uni_vcvtps2dq(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vcvtps2dq(x, op);
    }

    void uni_vcvtdq2ps(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
        cvtdq2ps(x, op);
    }
    void uni_vcvtdq2ps(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
        vcvtdq2ps(x, op);
    }

    void uni_vmovmskps(const Xbyak::Reg &x1, const Xbyak::Xmm &x2) {
        movmskps(x1.cvt64(), x2);
    }
    void uni_vmovmskps(const Xbyak::Reg &x1, const Xbyak::Ymm &x2) {
        vmovmskps(x1, x2);
    }

    void uni_vmovq(const Xbyak::Xmm &x, const Xbyak::Reg64 &r) {
        if (mayiuse(avx))
            vmovq(x, r);
        else
            movq(x, r);
    }
    void uni_vmovq(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        if (mayiuse(avx))
            vmovq(addr, x);
        else
            movq(addr, x);
    }

    void uni_vpackssdw(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        assert(x1.getIdx() == x1.getIdx());
        packssdw(x1, op);
    }
    void uni_vpackssdw(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        vpackssdw(x1, x2, op);
    }

    void uni_vpackuswb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        assert(x1.getIdx() == x1.getIdx());
        packuswb(x1, op);
    }
    void uni_vpackuswb(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        vpackuswb(x1, x2, op);
    }

    void uni_vpacksswb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        assert(x1.getIdx() == x1.getIdx());
        packsswb(x1, op);
    }
    void uni_vpacksswb(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        vpacksswb(x1, x2, op);
    }

    void uni_vpinsrb(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const int imm) {
        assert(x1.getIdx() == x2.getIdx());
        if (mayiuse(avx))
            vpinsrb(x1, x2, op, imm);
        else
            pinsrb(x1, op, imm);
    }

    void uni_vpinsrb(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, const int imm) {
        vpinsrb(x1, x2, op, imm);
    }

    void uni_vpinsrd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const int imm) {
        assert(x1.getIdx() == x2.getIdx());
        if (mayiuse(avx))
            vpinsrd(x1, x2, op, imm);
        else
            pinsrd(x1, op, imm);
    }
    void uni_vpinsrd(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, const int imm) {
        vpinsrd(x1, x2, op, imm);
    }

    void uni_vpinsrq(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const int imm) {
        assert(x1.getIdx() == x2.getIdx());
        if (mayiuse(avx))
            vpinsrq(x1, x2, op, imm);
        else
            pinsrq(x1, op, imm);
    }
    void uni_vpinsrq(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, const int imm) {
        vpinsrq(x1, x2, op, imm);
    }

    void uni_vpinsrw(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const int imm) {
        assert(x1.getIdx() == x2.getIdx());
        if (mayiuse(avx))
            vpinsrw(x1, x2, op, imm);
        else
            pinsrw(x1, op, imm);
    }
    void uni_vpinsrw(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op, const int imm) {
        vpinsrw(x1, x2, op, imm);
    }

    void uni_vpextrb(
            const Xbyak::Operand &op, const Xbyak::Xmm &x, const int imm) {
        if (mayiuse(avx))
            vpextrb(op, x, imm);
        else
            pextrb(op, x, imm);
    }

    void uni_vpextrb(
            const Xbyak::Operand &op, const Xbyak::Ymm &x, const int imm) {
        vpextrb(op, x, imm);
    }

    void uni_vpextrw(
            const Xbyak::Operand &op, const Xbyak::Xmm &x, const int imm) {
        if (mayiuse(avx))
            vpextrw(op, x, imm);
        else
            pextrw(op, x, imm);
    }
    void uni_vpextrw(
            const Xbyak::Operand &op, const Xbyak::Ymm &x, const int imm) {
        vpextrw(op, x, imm);
    }

    void uni_vpextrd(
            const Xbyak::Operand &op, const Xbyak::Xmm &x, const int imm) {
        if (mayiuse(avx))
            vpextrd(op, x, imm);
        else
            pextrd(op, x, imm);
    }
    void uni_vpextrd(
            const Xbyak::Operand &op, const Xbyak::Ymm &x, const int imm) {
        vpextrd(op, x, imm);
    }

    void uni_vpextrq(
            const Xbyak::Operand &op, const Xbyak::Xmm &x, const int imm) {
        if (mayiuse(avx))
            vpextrq(op, x, imm);
        else
            pextrq(op, x, imm);
    }
    void uni_vpextrq(
            const Xbyak::Operand &op, const Xbyak::Ymm &x, const int imm) {
        vpextrq(op, x, imm);
    }

    void uni_vpmaxsd(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (mayiuse(avx))
            vpmaxsd(x1, x2, op);
        else {
            if (x1.getIdx() != x2.getIdx()) movdqa(x1, x2);
            pmaxsd(x1, op);
        }
    }

    void uni_vpmaxsd(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        vpmaxsd(x1, x2, op);
    }

    void mul_by_const(
            const Xbyak::Reg &out, const Xbyak::Reg64 &tmp, int value) {
        // Generates a shift + add sequence for multiplicating contents of the
        // out register by a known JIT-time value. Clobbers the tmp register.
        //
        // Pros compared to mul/imul:
        // - does not require using known registers
        // - not microcoded on Intel(R) Xeon Phi(TM) processors
        // Still, there are probably a lot of cases when mul/imul is faster on
        // Intel(R) Core(TM) processors. Not intended for critical path.

        // TODO: detect when overflow is emminent (Roma)
        // TODO: detect when using mul/imul is a better option (Roma)

        int p = 0; // the current power of 2
        int old_p = 0; // the last seen power of 2 such that value[old_p] != 0

        xor_(tmp, tmp);
        while (value) {
            if (value & 1) {
                int shift = p - old_p;
                if (shift) {
                    shl(out, shift);
                    old_p = p;
                }
                add(tmp, out);
            }
            value >>= 1;
            p++;
        }
        mov(out, tmp);
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak::Reg64 reg_tmp,
            data_type_t idt, data_type_t odt) {
        using namespace data_type;
        if (!((idt == f32) && utils::one_of(odt, u8, s8, s32))) return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) uni_vpxor(vmm_lbound, vmm_lbound, vmm_lbound);

        Xbyak::Xmm tmp(vmm_ubound.getIdx());
        float saturation_ubound = types::max_value<float>(odt);
        mov(reg_tmp, float2int(saturation_ubound));
        uni_vmovq(tmp, reg_tmp);
        if (vmm_ubound.isYMM() || vmm_ubound.isZMM())
            uni_vbroadcastss(vmm_ubound, tmp);
        else
            uni_vshufps(vmm_ubound, tmp, tmp, 0);
    }

    // This function is used to saturate to odt in f32 before converting to s32
    // in order to avoid bad saturation due to cvtps2dq behavior (it returns
    // INT_MIN if the f32 is out of the s32 range)
    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, const Vmm &vmm_tmp, data_type_t odt) {
        using namespace data_type;
        if (!utils::one_of(odt, u8, s8, s32)) return;

        // no need to apply lower saturation bound when odt is signed, as
        // cvtps2dq will return MIN_INT if the value does not fit.
        // The comment below for a certain order applied for maxps instruction
        // as well. No changes here since NaN with positive sign was not met
        // yet.
        if (odt == u8) {
            if (mayiuse(avx))
                vmaxps(vmm, vmm, vmm_lbound);
            else
                maxps(vmm, vmm_lbound);
        }

        // Order matters for minps due to peculiar behavior of the instruction
        // with NaNs:
        //     if (SRC1 == NaN)
        //         return SRC2;
        //     else if (SRC2 == NaN)
        //         return SRC2;
        // that's why we keep user's data at SRC2 reg to pass NaNs further to
        // cvtps2dq which handles them properly.
        if (mayiuse(avx))
            vminps(vmm, vmm_ubound, vmm);
        else {
            movups(vmm_tmp, vmm_ubound);
            minps(vmm_tmp, vmm);
            movups(vmm, vmm_tmp);
        }
    }

    // AVX+ version of saturate_f32 which does not require an additional vector
    // register.
    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, data_type_t odt) {
        assert(mayiuse(avx));
        saturate_f32(vmm, vmm_lbound, vmm_ubound, Vmm(), odt);
    }

    /**
    * load_bytes is the utility function to facilitate loading of
    * load_size (0 <= load_size <= 32) many contiguous bytes into the Xmm/Ymm
    * register from the memory referenced by ptr[reg + offset] address.
    *
    * Functionally, invocation of load_bytes is equivalent to
    * the following loop:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    *
    * TODO: Add an option to zero-out unloaded bytes in the Xmm register.
    * TODO: Add an option for unsafe_load wherein one could read outside the
    * provided memory buffer so as to minimize the total number of read
    * memory instructions.
    */
    template <typename Vmm>
    void load_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset,
            int load_size, bool force_sse = false) {

        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        static_assert(
                is_xmm || is_ymm, "only Xmm or Ymm registers are allowed");
        const bool use_avx = mayiuse(avx) && !force_sse;

        MAYBE_UNUSED(is_xmm);
        MAYBE_UNUSED(is_ymm);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(load_size >= 0 && load_size <= 32);

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // At most 16 bytes can fit inside the Xmm register
        assert(IMPLICATION(load_size > 16, is_ymm));

        assert(mayiuse(sse41)
                && "routine is not supported for the current isa");

        auto xmm = Xbyak::Xmm(vmm.getIdx());
        auto ymm = Xbyak::Ymm(vmm.getIdx());

        // addr(i) denotes the memory pointed by ptr[reg + offset + (i bytes)]
        const auto addr = [&](int bytes_offset) {
            return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
        };

        // VEX-fying macro when AVX and SSE41 instructions have
        // same number of arguments
#define MAYBE_VEX2(instr, arg1, arg2) \
    do { \
        if (use_avx) \
            CONCAT2(v, instr)(arg1, arg2); \
        else \
            instr(arg1, arg2); \
    } while (0)

        // VEX-fying macro when AVX have one extra argument for
        // destination (namely, the first argument)
#define MAYBE_VEX(instr, arg1, arg2, arg3) \
    do { \
        if (use_avx) \
            CONCAT2(v, instr)(arg1, arg1, arg2, arg3); \
        else \
            instr(arg1, arg2, arg3); \
    } while (0)

        if (load_size == 32) {
            vmovups(ymm, addr(0));
            return;
        }

        int start_bytes = 0;
        int bytes_to_load = load_size;

        if (load_size > 16) {
            // Prepare to insert to upper bits of ymm
            start_bytes = 16;
            bytes_to_load -= 16;
        }

        if (bytes_to_load >= 8 && bytes_to_load < 16)
            MAYBE_VEX(pinsrq, xmm, addr(start_bytes), 0);
        else if (bytes_to_load == 16)
            MAYBE_VEX2(movdqu, xmm, addr(start_bytes));

        switch (bytes_to_load) {
            case 0: break;
            case 1: MAYBE_VEX(pinsrb, xmm, addr(start_bytes), 0); break;
            case 2: MAYBE_VEX(pinsrw, xmm, addr(start_bytes), 0); break;
            case 3:
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes), 0);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 2), 2);
                break;
            case 4: MAYBE_VEX(pinsrd, xmm, addr(start_bytes), 0); break;
            case 5:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes), 0);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 4), 4);
                break;
            case 6:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes), 0);
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 4), 2);
                break;
            case 7:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes), 0);
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 4), 2);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 6), 6);
                break;
            case 8: break;
            case 9: MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 8), 8); break;
            case 10: MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 8), 4); break;
            case 11:
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 8), 4);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 10), 10);
                break;
            case 12: MAYBE_VEX(pinsrd, xmm, addr(start_bytes + 8), 2); break;
            case 13:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes + 8), 2);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 12), 12);
                break;
            case 14:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes + 8), 2);
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 12), 6);
                break;
            case 15:
                MAYBE_VEX(pinsrd, xmm, addr(start_bytes + 8), 2);
                MAYBE_VEX(pinsrw, xmm, addr(start_bytes + 12), 6);
                MAYBE_VEX(pinsrb, xmm, addr(start_bytes + 14), 14);
                break;
            case 16: break;
            default: assert(!"improper load size");
        }

        if (load_size > 16) {
            vinsertf128(ymm, ymm, xmm, 1); // insert to upper bits of ymm
            vinsertf128(ymm, ymm, addr(0), 0); // insert to lower bits of ymm
        }
#undef MAYBE_VEX2
#undef MAYBE_VEX
    }

    /**
    * store_bytes is the utility function to facilitate storing of
    * store_size (0 <= store_size <= 32) many contiguous bytes from the Xmm/Ymm
    * register into the memory referenced by ptr[reg + offset] address.
    *
    * Additionally, when store_size > 16, the input Ymm register will not be
    * preserved due to the usage of vextracti128 instruction.
    *
    * Functionally, invocation of store_bytes is equivalent
    * to the following loop:
    *
    * for (int idx = 0; idx < store_size; ++idx)
    *     vpextrb(ptr[reg + offset + idx], xmm, idx);
    *
    * TODO: Add an option for unsafe_store wherein one could store extra dwords
    * past the provided memory buffer so as to minimize the total number of
    * write memory instructions.
    */
    template <typename Vmm>
    void store_bytes(const Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset,
            int store_size, bool force_sse = false) {

        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        static_assert(
                is_xmm || is_ymm, "only Xmm or Ymm registers are allowed");
        const bool use_avx = mayiuse(avx) && !force_sse;

        MAYBE_UNUSED(is_xmm);
        MAYBE_UNUSED(is_ymm);

        // Ensure data fits completely inside the Xmm/Ymm register
        assert(store_size >= 0 && store_size <= 32);

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        // At most 16 bytes can fit inside the Xmm register
        assert(IMPLICATION(store_size > 16, is_ymm));

        assert(mayiuse(sse41)
                && "routine is not supported for the current isa");

        auto xmm = Xbyak::Xmm(vmm.getIdx());
        auto ymm = Xbyak::Ymm(vmm.getIdx());

        const auto addr = [&](int bytes_offset) {
            return ptr[reg + offset + bytes_offset * sizeof(int8_t)];
        };

        // VEX-fying macro when AVX and SSE41 instructions have
        // same number of arguments
#define MAYBE_VEX2(instr, arg1, arg2) \
    do { \
        if (use_avx) \
            CONCAT2(v, instr)(arg1, arg2); \
        else \
            instr(arg1, arg2); \
    } while (0)

#define MAYBE_VEX3(instr, arg1, arg2, arg3) \
    do { \
        if (use_avx) \
            CONCAT2(v, instr)(arg1, arg2, arg3); \
        else \
            instr(arg1, arg2, arg3); \
    } while (0)

        if (store_size == 32) {
            vmovups(addr(0), ymm);
            return;
        }

        int start_bytes = 0;
        int bytes_to_store = store_size;

        if (store_size > 16) {
            vmovdqu(addr(0), xmm); // load lower bits from ymm
            start_bytes = 16;
            bytes_to_store -= 16;
            vextractf128(xmm, ymm, 1); // load upper bits from ymm into xmm
        }

        if (bytes_to_store >= 8 && bytes_to_store < 16)
            MAYBE_VEX3(pextrq, addr(start_bytes), xmm, 0);
        else if (bytes_to_store == 16)
            MAYBE_VEX2(movdqu, addr(start_bytes), xmm);

        switch (bytes_to_store) {
            case 0: break;
            case 1: MAYBE_VEX3(pextrb, addr(start_bytes), xmm, 0); break;
            case 2: MAYBE_VEX3(pextrw, addr(start_bytes), xmm, 0); break;
            case 3:
                MAYBE_VEX3(pextrw, addr(start_bytes), xmm, 0);
                MAYBE_VEX3(pextrb, addr(start_bytes + 2), xmm, 2);
                break;
            case 4: MAYBE_VEX3(pextrd, addr(start_bytes), xmm, 0); break;
            case 5:
                MAYBE_VEX3(pextrd, addr(start_bytes), xmm, 0);
                MAYBE_VEX3(pextrb, addr(start_bytes + 4), xmm, 4);
                break;
            case 6:
                MAYBE_VEX3(pextrd, addr(start_bytes), xmm, 0);
                MAYBE_VEX3(pextrw, addr(start_bytes + 4), xmm, 2);
                break;
            case 7:
                MAYBE_VEX3(pextrd, addr(start_bytes), xmm, 0);
                MAYBE_VEX3(pextrw, addr(start_bytes + 4), xmm, 2);
                MAYBE_VEX3(pextrb, addr(start_bytes + 6), xmm, 6);
                break;
            case 8: break;
            case 9: MAYBE_VEX3(pextrb, addr(start_bytes + 8), xmm, 8); break;
            case 10: MAYBE_VEX3(pextrw, addr(start_bytes + 8), xmm, 4); break;
            case 11:
                MAYBE_VEX3(pextrw, addr(start_bytes + 8), xmm, 4);
                MAYBE_VEX3(pextrb, addr(start_bytes + 10), xmm, 10);
                break;
            case 12: MAYBE_VEX3(pextrd, addr(start_bytes + 8), xmm, 2); break;
            case 13:
                MAYBE_VEX3(pextrd, addr(start_bytes + 8), xmm, 2);
                MAYBE_VEX3(pextrb, addr(start_bytes + 12), xmm, 12);
                break;
            case 14:
                MAYBE_VEX3(pextrd, addr(start_bytes + 8), xmm, 2);
                MAYBE_VEX3(pextrw, addr(start_bytes + 12), xmm, 6);
                break;
            case 15:
                MAYBE_VEX3(pextrd, addr(start_bytes + 8), xmm, 2);
                MAYBE_VEX3(pextrw, addr(start_bytes + 12), xmm, 6);
                MAYBE_VEX3(pextrb, addr(start_bytes + 14), xmm, 14);
                break;
            case 16: break;
            default: assert(!"improper store size");
        }
#undef MAYBE_VEX2
#undef MAYBE_VEX3
    }

    /**
    * load_bytes_to_dword_extension is the utility function to facilitate
    * loading of load_size (0 <= load_size <= 16) many contiguous bytes in
    * the Xmm register from the memory referenced by ptr[reg + offset]
    * address and then do signed/zero extension of those to double words.
    *
    * Functionally, invocation of load_bytes_to_dword_extension is equivalent
    * to the following:
    *
    * for (int idx = 0; idx < load_size; ++idx)
    *     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
    * if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
    *
    * Valid values for the load_size variable are:
    * [0..4] for XMM version of the function
    * [0..8] for YMM version of the function.
    * TODO: Implement this routine for every ISA.
    */

    template <typename Vmm>
    void load_bytes_to_dword_extension(const Vmm &vmm, const Xbyak::Reg64 &reg,
            int64_t offset, bool is_signed, int load_size) {

        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        static_assert(
                is_xmm || is_ymm, "only Xmm or Ymm registers are allowed");
        MAYBE_UNUSED(is_xmm);
        MAYBE_UNUSED(is_ymm);

        // Ensure extended double words fit inside Ymm (32 * load_size <= 256)
        assert(load_size >= 0 && load_size <= 8);
        // For Xmm register, load capacity is halved (32 * load_size <= 128)
        assert(IMPLICATION(is_xmm, load_size <= 4));

        // Ensure offset is at most 4 bytes to be encoded in the instruction
        assert(offset >= INT_MIN && offset <= INT_MAX);

        assert(mayiuse(sse41)
                && "routine is not supported for the current isa");

        // For load_size == 8/4, do load/extension in one go
        if (load_size == 8) {
            const auto ymm = Xbyak::Ymm(vmm.getIdx());
            if (is_signed)
                vpmovsxbd(ymm, ptr[reg + offset]);
            else
                vpmovzxbd(ymm, ptr[reg + offset]);
        } else if (load_size == 4) {
            const auto xmm = Xbyak::Xmm(vmm.getIdx());
            if (is_signed)
                uni_vpmovsxbd(xmm, ptr[reg + offset]);
            else
                uni_vpmovzxbd(xmm, ptr[reg + offset]);
        } else {
            load_bytes(vmm, reg, offset, load_size);
            if (is_signed)
                uni_vpmovsxbd(vmm, vmm);
            else
                uni_vpmovzxbd(vmm, vmm);
        }
    }

    /* A utility function to store data of type type_out from vmm register
     * into the memory. Moreover store_size many chunks are written to the
     * memory beginning with ptr[reg + offset] address.
     *
     * Note: Content of Vmm register is not guaranteed to be preserved after the
     * invocation of this routine.
     *
     * TODO: Support for every possible data type.
     */
    template <typename Vmm>
    void store_data(data_type_t type_out, const Vmm &vmm,
            const Xbyak::Reg64 &reg, int64_t offset, int store_size) {

        assert(mayiuse(sse41)
                && "routine is not supported for the current isa");
        constexpr bool is_xmm = std::is_same<Vmm, Xbyak::Xmm>::value;
        // Avoid using Ymm with avx isa
        assert(IMPLICATION(!mayiuse(avx2), is_xmm));
        MAYBE_UNUSED(is_xmm);

        auto ymm = Xbyak::Ymm(vmm.getIdx());

        switch (type_out) {
            case data_type::f32:
            case data_type::s32:
                store_bytes(vmm, reg, offset, sizeof(int32_t) * store_size);
                break;
            case data_type::u8:
            case data_type::s8:
                uni_vpackssdw(vmm, vmm, vmm);
                if (mayiuse(avx2)) vpermq(ymm, ymm, 0x08);
                if (type_out == data_type::s8)
                    uni_vpacksswb(vmm, vmm, vmm);
                else
                    uni_vpackuswb(vmm, vmm, vmm);
                store_bytes(vmm, reg, offset, store_size);
                break;
            default: assert(!"unsupported destination data type");
        }
    }

    /* A utility function to load data of type type_in to vmm register
     * from the memory. Moreover load_size many chunks are read from the memory
     * beginning with ptr[reg + offset] address.
     *
     * TODO: Support for every possible data type.
     */
    template <typename Vmm>
    void load_data(data_type_t type_in, const Vmm &vmm, const Xbyak::Reg64 &reg,
            int64_t offset, int load_size) {

        assert(mayiuse(sse41)
                && "routine is not supported for the current isa");

        switch (type_in) {
            case data_type::f32:
            case data_type::s32:
                load_bytes(vmm, reg, offset, sizeof(int32_t) * load_size);
                break;
            case data_type::s8:
            case data_type::u8:
                load_bytes_to_dword_extension(
                        vmm, reg, offset, type_in == data_type::s8, load_size);
                break;
            default: assert(!"unsupported source data type");
        }
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true)
        : Xbyak::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak::AutoGrow
                                                      : code_ptr) {}
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const Xbyak::uint8 *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const Xbyak::uint8 *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const Xbyak::uint8 *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const Xbyak::uint8 *code = CodeGenerator::getCode();
        register_jit_code(code, getSize());
        return code;
    }

    static inline bool is_initialized() {
        return Xbyak::GetError() == Xbyak::ERR_NONE;
    }

protected:
    virtual void generate() = 0;
    const Xbyak::uint8 *jit_ker_ = nullptr;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
