/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_transpose_src_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_trans_iw_ic_t : public jit_trans_src_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_iw_ic_t)

    jit_trans_iw_ic_t(const jit_conv_conf_t *conf) : jit_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float), transpose_size = 16, small_spatial = 14 };
    int src_stride = 0, tr_src_stride = 0;
    int tail = 0;
    bool enable_prefetch = false;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores);
    void generate() override;
};

void jit_trans_iw_ic_t::transpose(
        int nrows, int l_pad, int r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto pf_src_t0 = [=](int i) {
        if (enable_prefetch)
            prefetcht0(EVEX_compress_addr(
                    reg_src, (transpose_size + i) * src_stride));
    };

    auto pf_tr_src_t0 = [=](int i) {
        int offset = (transpose_size)*typesize + i * tr_src_stride;
        if (enable_prefetch) prefetcht0(EVEX_compress_addr(reg_tr_src, offset));
        if (enable_prefetch)
            prefetcht0(EVEX_compress_addr(reg_tr_src, offset + 64));
    };

    auto pf_src_t1 = [=](int i) {
        if (enable_prefetch)
            prefetcht1(EVEX_compress_addr(reg_src_prf, i * src_stride));
    };

    auto pf_tr_src_t1 = [=](int i) {
        if (enable_prefetch)
            prefetchwt1(EVEX_compress_addr(reg_tr_src_prf, i * tr_src_stride));
    };

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [=](int i) {
        vmovups(src_zmm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        auto kmovw = [=](Opmask k, unsigned w) {
            mov(regw_tmp, w);
            jit_generator::kmovw(k, regw_tmp);
        };

        auto padding = [=](Reg64 reg, int pad) {
            kmovw(kTail, (1 << pad) - 1);
            auto k = kTail;
            auto base = reg;
            base.setOpmaskIdx(k.getIdx(), true);

            auto zmm_zero = r;
            vpxord(zmm_zero, zmm_zero, zmm_zero);
            auto addr = EVEX_compress_addr(base, i * tr_src_stride);
            vmovups(addr, zmm_zero);
        };

        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0) add(reg_tr_src_tmp, l_pad * typesize);

        if (tail != transpose_size) kmovw(kTail, (1 << tail) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < 16;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        if (nontemporal_stores && !partial_store)
            vmovntps(addr, r);
        else
            vmovups(addr, r);

        if (r_pad > 0) {
            add(reg_tr_src_tmp, tail * typesize);
            padding(reg_tr_src_tmp, r_pad);
        }

        if (l_pad > 0) { padding(reg_tr_src, l_pad); }
    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);
            pf_src_t1(base_idx + i);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);
            pf_src_t0(base_idx + i);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            int select_half = (i < 2) ? 0 : 2;
            int src_idx0 = base_idx + i + select_half + 0;
            int src_idx2 = src_idx0 + 2;

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx2);
            auto src0 = src_zmm(src_idx0);
            auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            pf_src_t1(base_idx + 4 + i);
            valignd(tmp1, src2, src2, 0xe);
            pf_src_t0(base_idx + 4 + i);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i;
            int src_idx4 = src_idx0 + 4;

            auto tmp0 = tmp_zmm(src_idx0);
            auto src0 = src_zmm(src_idx0);
            auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            pf_tr_src_t1(base_idx / 2 + i);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
            pf_tr_src_t0(base_idx / 2 + i);
        }
    };

    auto fixup16x16 = [=]() {
        // swap 8
        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
            if (i % 2 == 0) {
                pf_tr_src_t1(8 + i / 2);
                pf_tr_src_t0(8 + i / 2);
            }
        }

        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
            if (i % 2 == 0) {
                pf_tr_src_t1(12 + i / 2);
                pf_tr_src_t0(12 + i / 2);
            }
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_trans_iw_ic_t::generate() {
    preamble();

    const int ic_block = conf_->ic_block;
    const int iw = conf_->iw;
    const int tr_iw = conf_->tr_iw;
    const int transposes = utils::div_up(iw, transpose_size);
    int loop_iters = nstl::max(0, transposes - 1);
    tail = iw - loop_iters * transpose_size;

    src_stride = ic_block * typesize;
    assert(src_stride == 64);
    tr_src_stride = tr_iw * typesize;

    bool nontemporal_stores = false;
    enable_prefetch = iw > small_spatial;

    assert(transpose_size == ic_block);
    const int src_step = ic_block * transpose_size * typesize;
    const int tr_src_step = ic_block * typesize;

    const int left_pad = conf_->l_pad;
    const int right_pad = tr_iw - iw - left_pad;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    if (left_pad > 0 && loop_iters > 0) {
        loop_iters--;
        transpose(transpose_size, left_pad, 0, nontemporal_stores);
        add(reg_src, src_step);
        add(reg_tr_src, tr_src_step + left_pad * typesize);
        add(reg_src_prf, src_step);
        add(reg_tr_src_prf, tr_src_step + left_pad * typesize);
    }

    if (loop_iters) {
        mov(reg_loop, loop_iters);
        Label loop;
        L(loop);
        {
            transpose(transpose_size, 0, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step);
            sub(reg_loop, 1);
            jnz(loop);
        }
    }
    if (transposes > 1)
        transpose(tail, 0, right_pad, nontemporal_stores);
    else
        transpose(tail, left_pad, right_pad, nontemporal_stores);

    postamble();
}

struct jit_trans_iw_ic_int16_t : public jit_trans_src_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_iw_ic_int16_t)
    jit_trans_iw_ic_int16_t(const jit_conv_conf_t *conf)
        : jit_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
        small_spatial = 14
    };
    size_t src_stride, tr_src_stride;
    int tail;
    bool enable_prefetch;

    opmask_t kFFFF = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t k55 = k5;
    opmask_t kCC = k6;
    opmask_t k33 = k7;
    opmask_t kTail = k1;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = rbx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores);
    void generate() override;
};

void jit_trans_iw_ic_int16_t::transpose(
        int nrows, int l_pad, int r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) { return Zmm(i); };

    auto src_ymm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto load_ymm = [=](int i) {
        vmovups(src_ymm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    auto store = [=](Zmm r, int i) {
        auto padding = [=](Reg64 base, int pad_rows, int pad_tail) {
            // note: pad can be bigger than 16 because of dilation
            const size_t row_offset = 2 * transpose_size * typesize;
            auto zmm_zero = zmm_tmp;
            vpxord(zmm_zero, zmm_zero, zmm_zero);
            for (int i_row = 0; i_row < pad_rows; i_row++) {
                auto addr = EVEX_compress_addr(
                        base, i * tr_src_stride + i_row * row_offset);
                vmovups(addr, zmm_zero);
            }
            if (pad_tail > 0) {
                kmovw(kTail, (1 << pad_tail) - 1);
                base.setOpmaskIdx(kTail.getIdx(), true);
                auto addr = EVEX_compress_addr(
                        base, i * tr_src_stride + pad_rows * row_offset);
                vmovups(addr, zmm_zero);
            }
        };

        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0) {
            int store_pad = 2 * transpose_size;
            int pad_rows = l_pad / store_pad;
            int tail = l_pad % store_pad;
            padding(reg_tr_src_tmp, pad_rows, div_up(tail, 2));
            add(reg_tr_src_tmp, (pad_rows * store_pad + tail) * typesize);
        }
        if (r_pad > 0) {
            int addr_shift = nrows - r_pad % 2;
            int store_pad = div_up(r_pad, 2);
            int pad_rows = store_pad / transpose_size;
            add(reg_tr_src_tmp, addr_shift * typesize);
            padding(reg_tr_src_tmp, pad_rows, store_pad % transpose_size);
            sub(reg_tr_src_tmp, addr_shift * typesize);
        }

        int store_tail = rnd_up(nrows, 2);
        kmovw(kTail, (1 << store_tail / 2) - 1);
        auto k = kTail;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    const bool is_layout_nxc = utils::one_of(conf_->src_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const int ic_block = conf_->ic_block;
    const bool is_tail_block = ic_block != 16;
    const int ic_tail = conf_->ic_tail;
    // Assertion below as we need vmovdqu16 for ic_tails.
    // If needed, can be extended by using load_bytes() helper.
    assert(IMPLICATION(ic_tail, mayiuse(avx512_core)));
    if (mayiuse(avx512_core)) {
        if (conf_->stride_w > 1 || nrows % 2 || is_layout_nxc)
            kmovd(kFFFF, (1 << ic_block) - 1);
        if (conf_->stride_w > 1 || is_layout_nxc) kmovd(k33, 0xffff0000);
        if (is_layout_nxc && conf_->ic_tail) {
            Label done;
            cmp(dword[param1 + GET_OFF(ch_work)], ic_block);
            je(done, T_NEAR);
            kmovd(kFFFF, (1 << conf_->ic_tail) - 1);
            kshiftld(k33, kFFFF, 16);
            L(done);
        }

        for (int i = 0; i < nrows / 2; i++) {
            auto zmm_src0 = src_zmm(2 * i);
            if (conf_->stride_w == 1 && !is_layout_nxc) {
                vmovdqu16(zmm_src0,
                        EVEX_compress_addr(reg_src, 2 * i * src_stride));
            } else {
                vmovdqu16(zmm_src0 | kFFFF | T_z,
                        EVEX_compress_addr(reg_src, 2 * i * src_stride));
                if (is_tail_block || ic_tail) {
                    auto zmm_tmp = src_zmm(2 * i + 1);
                    vmovdqu16(zmm_tmp | kFFFF | T_z,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride));
                    vinsertf64x4(zmm_src0, zmm_src0, src_ymm(2 * i + 1), 1);
                } else {
                    vmovdqu16(zmm_src0 | k33,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride - 32));
                }
            }
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = nrows / 2;
            auto zmm_src0 = src_zmm(2 * i);
            vmovdqu16(zmm_src0 | kFFFF | T_z,
                    EVEX_compress_addr(reg_src, 2 * i * src_stride));
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        if (conf_->stride_w > 1 || is_layout_nxc) kmovw(k33, 0x33);

        for (int i = rnd_up(nrows, 2); i < 16; i += 2) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }
    } else {
        kmovw(kFFFF, 0xffff);
        // all loads
        for (int i = 0; i < 16; i++) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }

        for (int i = 0; i < nrows / 2; i++) {
            auto src0 = src_ymm(2 * i);
            auto src1 = src_ymm(2 * i + 1);
            auto zmm_src0 = src_zmm(2 * i);
            load_ymm(2 * i);

            vpunpcklwd(src1, src0,
                    EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
            vpunpckhwd(src0, src0,
                    EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
            vinserti64x4(zmm_src0, zmm_src0, src1, 1);
            vpermps(zmm_src0 | kFFFF, vidx4, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = nrows - 1;
            auto src0 = src_ymm(i);
            auto src1 = src_ymm(i + 1); // zero

            auto zmm_src0 = src_zmm(i);
            vpxor(src1, src1, src1);

            load_ymm(i);
            vpunpckhwd(src0, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src0, 0);
            vpxor(src0, src0, src0);
            load_ymm(i);
            vpunpcklwd(src1, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src1, 1);
            vpxord(zmm_src0, zmm_src0, zmm_src0);
            vmovups(zmm_src0, zmm_tmp);
            vpermps(zmm_src0 | kFFFF, vidx4, zmm_src0);
        }
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [=](int ic_idx) {
        assert(ic_idx < 16 && ic_idx >= 0);
        switch (ic_idx) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
            case 4: return 9;
            case 5: return 8;
            case 6: return 11;
            case 7: return 10;
            case 8: return 5;
            case 9: return 4;
            case 10: return 7;
            case 11: return 6;
            case 12: return 13;
            case 13: return 12;
            case 14: return 15;
            default: return 14;
        }
    };

    for (int ic = 0; ic < ic_block; ic++)
        store(src_zmm(get_vec_idx(ic)), ic);
}

void jit_trans_iw_ic_int16_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

    const int ic_block = conf_->ic_block;
    const bool is_layout_nxc = utils::one_of(conf_->src_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const size_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->ic : ic_block;
    const int iw = conf_->iw;
    const int tr_iw = conf_->tr_iw;
    const int str_w = conf_->stride_w;
    assert(tr_iw % str_w == 0);
    const int tr_iw_s = tr_iw / str_w;
    assert(transpose_size >= ic_block);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff);
    kmovw(k5555, 0x5555);
    kmovw(kAAAA, 0xaaaa);
    kmovw(kAA, 0xaa);
    kmovw(k55, 0x55);
    kmovw(kCC, 0xcc);
    kmovw(k33, 0x33);

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [=](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, idx2);
    vmovdqa32(vidx3, idx3);
    vmovdqa32(vidx4, idx4);
    vmovdqa32(vidx5, (const int32_t *)idx5);

    // Data for every strided case is placed consecutively
    for (int s = 0; s < str_w; s++) {
        const int left_pad = div_up(conf_->l_pad - s, str_w);
        const int iw1 = iw + conf_->l_pad;
        const int iw_s = (s < (iw1 % str_w) ? div_up(iw1, str_w) : iw1 / str_w)
                - left_pad;
        const int right_pad = tr_iw_s - iw_s - left_pad;

        const int transposes = utils::div_up(iw_s, transpose_size);
        int loop_iters = nstl::max(0, transposes - 1);
        tail = iw_s - loop_iters * transpose_size;

        src_stride = src_mult * typesize * str_w;
        tr_src_stride = tr_iw * typesize;

        bool nontemporal_stores = false;
        enable_prefetch = iw > small_spatial ? true : false;

        const size_t src_step = src_mult * transpose_size * str_w * typesize;
        const size_t tr_src_step = transpose_size * typesize;

        mov(reg_src, ptr[param1 + GET_OFF(src)]);
        mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
        mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

        if (str_w > 1) {
            int tr_src_shift = s;
            int src_shift = (str_w - (conf_->l_pad % str_w) + s) % str_w;
            add(reg_src, src_shift * src_mult * typesize);
            add(reg_tr_src, tr_src_shift * tr_iw_s * typesize);
            add(reg_src_prf, src_shift * src_mult * typesize);
            add(reg_tr_src_prf, tr_src_shift * tr_iw_s * typesize);
        }

        if (left_pad > 0 && loop_iters > 0) {
            loop_iters--;
            transpose(transpose_size, left_pad, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step + left_pad * typesize);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step + left_pad * typesize);
        }

        if (loop_iters) {
            mov(reg_loop, loop_iters);
            Label loop;
            L(loop);
            {
                transpose(transpose_size, 0, 0, nontemporal_stores);
                add(reg_src, src_step);
                add(reg_tr_src, tr_src_step);
                add(reg_src_prf, src_step);
                add(reg_tr_src_prf, tr_src_step);
                sub(reg_loop, 1);
                jnz(loop);
            }
        }
        if (transposes > 1)
            transpose(tail, 0, right_pad, nontemporal_stores);
        else
            transpose(tail, left_pad, right_pad, nontemporal_stores);
    }
    postamble();
}

struct jit_trans_ow_oc_t : public jit_trans_dst_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_ow_oc_t)
    jit_trans_ow_oc_t(const jit_conv_conf_t *conf) : jit_trans_dst_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
        small_spatial = 14
    };
    size_t src_stride = 0, tr_src_stride = 0;
    int tail = 0;
    bool enable_prefetch = false;

    opmask_t kFF = k1;
    opmask_t mask_lo = k2;
    opmask_t k_oc_tail = k3;

    zmm vidx1 = zmm31;
    zmm vidx2 = zmm30;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = rbx;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores);
    void generate() override;
};

void jit_trans_ow_oc_t::transpose(
        int nrows, int l_pad, int r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) { return Zmm(i); };

    auto src_ymm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto load_ymm = [=](int i) {
        auto ymm_reg = src_ymm(i);
        auto addr = EVEX_compress_addr(reg_src, i * src_stride);
        if (conf_->oc_tail) {
            ymm_reg = ymm_reg | k_oc_tail | T_z;
            // Assertion below as we need vmovdqu16 for tails.
            // If needed, can be removed by using load_bytes() helper.
            assert(mayiuse(avx512_core));
            vmovdqu16(ymm_reg, addr);
        } else {
            vmovups(ymm_reg, addr);
        }
    };

    auto store = [=](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride);
        if (nontemporal_stores)
            vmovntps(addr, r);
        else
            vmovups(addr, r);
    };
    const bool is_layout_nxc = utils::one_of(conf_->dst_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);

    if (mayiuse(avx512_core) && !is_layout_nxc) {
        // TODO: adopt for nhwc?
        for (int i = 0; i < nrows / 2; i++) {
            auto zmm_src0 = src_zmm(i);
            vmovdqu16(
                    zmm_src0, EVEX_compress_addr(reg_src, 2 * i * src_stride));
            vpermw(zmm_src0, vidx2, zmm_src0);
            store(zmm_src0, 2 * i);
        }
        if (r_pad > 0) {
            auto zmm_src0 = src_zmm(29);
            vmovdqu16(zmm_src0 | mask_lo | T_z,
                    EVEX_compress_addr(reg_src, (nrows - 1) * src_stride));
            vpermw(zmm_src0, vidx2, zmm_src0);
            store(zmm_src0, nrows - 1);
        }
    } else {
        for (int i = 0; i < nrows / 2; i++) {
            auto src0 = src_ymm(2 * i);
            auto src1 = src_ymm(2 * i + 1);
            auto zmm_src0 = src_zmm(2 * i);
            load_ymm(2 * i);
            if (is_layout_nxc && conf_->oc_tail) {
                load_ymm(2 * i + 1);
                auto ymm_tmp = Ymm(30);
                vpunpcklwd(ymm_tmp, src0, src1);
                vpunpckhwd(src0, src0, src1);
                vinserti64x4(zmm_src0, zmm_src0, ymm_tmp, 1);
            } else {
                vpunpcklwd(src1, src0,
                        EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
                vpunpckhwd(src0, src0,
                        EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
                vinserti64x4(zmm_src0, zmm_src0, src1, 1);
            }
            vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            store(zmm_src0, 2 * i);
        }
        if (r_pad > 0) {
            auto src0 = src_ymm(nrows - 1);
            auto src1 = src_ymm(nrows);
            auto zmm_src0 = src_zmm(30);
            load_ymm(nrows - 1);

            vpxor(src1, src1, src1);
            vpunpckhwd(src1, src0, src1);
            vinserti64x4(zmm_src0, zmm_src0, src1, 0);
            vpxor(src1, src1, src1);
            vpunpcklwd(src0, src0, src1);
            vinserti64x4(zmm_src0, zmm_src0, src0, 1);
            vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            store(zmm_src0, nrows - 1);
        }
    }
}

void jit_trans_ow_oc_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {4, 5, 0, 1, 6, 7, 2, 3};
    alignas(64) static constexpr const int16_t idx2[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    const int oc_block = conf_->oc_block;
    const bool is_layout_nxc = utils::one_of(conf_->dst_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const size_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->oc : oc_block;
    const int ow = conf_->ow;
    const int transposes = utils::div_up(ow, transpose_size);
    int loop_iters = nstl::max(0, transposes - 1);
    tail = ow - loop_iters * transpose_size;

    src_stride = src_mult * typesize;
    tr_src_stride = oc_block * typesize;

    bool nontemporal_stores = conf_->use_nt_stores_ddst;
    enable_prefetch = ow > small_spatial;

    const size_t src_step = src_mult * transpose_size * typesize;
    const size_t tr_src_step = (size_t)oc_block * transpose_size * typesize;
    const int right_pad = ow % 2;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };
    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    kmovw(kFF, 0xFF);
    kmovd(mask_lo, 0x0000ffff);

    if (is_layout_nxc && conf_->oc_tail) {
        Label done;
        kxnorw(k_oc_tail, k_oc_tail, k_oc_tail);
        cmp(dword[param1 + GET_OFF(ch_work)], conf_->oc_block);
        je(done, T_NEAR);
        kmovw(k_oc_tail, (1 << conf_->oc_tail) - 1);
        L(done);
    }

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, (const int64_t *)idx2);
    if (loop_iters) {
        mov(reg_loop, loop_iters);
        Label loop;
        L(loop);
        {
            transpose(transpose_size, 0, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step);
            sub(reg_loop, 1);
            jnz(loop);
        }
    }
    transpose(tail, 0, right_pad, nontemporal_stores);

    postamble();
}

struct jit_trans_iw_x4_4x_t : public jit_trans_src_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_iw_x4_4x_t)

    jit_trans_iw_x4_4x_t(const jit_conv_conf_t *conf) : jit_trans_src_t(conf) {}

    void generate() override;

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    enum { typesize = (int)sizeof(float) };
};

/** @brief transposition of the form [:][iw/4][4] -> [:][4][iw/4]
 * required for 1st 4fma backward by weights convolution */
void jit_trans_iw_x4_4x_t::generate() {
    using namespace utils;

    /* TODO: put into code */
    static int mask[16] = {
            0,
            4,
            8,
            12,
            1,
            5,
            9,
            13,
            2,
            6,
            10,
            14,
            3,
            7,
            11,
            15,
    };

    const auto &c = *conf_;
    const int simd_w = cpu_isa_traits<avx512_common>::vlen / typesize;
    const int niters = c.tr_ld / simd_w;

    assert(niters <= 4); /* [bwd_w:tr_src:r1] */

    Reg64 reg_ptr_src = r8;
    Reg64 reg_ptr_tr_src = r9;

    Reg64 reg_ih = rax;
    Reg64 reg_ih_end = rbx;

    Reg64 reg_nthr_oc_b = rsi;
    Reg64 reg_ptr_tr_src_bctx = abi_not_param1;

    Reg64 reg_tmp = rdx;

    Zmm vmsk = Zmm(31);
    Opmask kmsk = k7;

    auto emit_tr_sync = [&]() {
        simple_barrier::generate(*this, reg_ptr_tr_src_bctx, reg_nthr_oc_b);
    };

    auto emit_tr_iw = [&]() {
        auto vreg = [](int iter, int i) {
            assert(4 * iter + i < 24);
            return Zmm(4 * iter + i);
        };
        auto vtmp = [](int i) { return Zmm(24 + i); };

        auto emit_load = [&](int iter) {
            for (int i = 0; i < 4; ++i) {
                auto v = vreg(iter, i);
                const int off = (iter * 4 + i) * simd_w;

                if (off + simd_w <= c.iw)
                    vmovups(v, ptr[reg_ptr_src + off * typesize]);
                else if (off < c.iw)
                    vmovups(v | kmsk | T_z, ptr[reg_ptr_src + off * typesize]);
                else
                    vpxord(v, v, v);
            }
        };

        auto emit_tr = [&](int iter) {
            for (int i = 0; i < 4; ++i)
                vpermps(vreg(iter, i), vmsk, vreg(iter, i));

            vshuff32x4(vtmp(0), vreg(iter, 0), vreg(iter, 1), 0x88);
            vshuff32x4(vtmp(1), vreg(iter, 0), vreg(iter, 1), 0xdd);
            vshuff32x4(vtmp(2), vreg(iter, 2), vreg(iter, 3), 0x88);
            vshuff32x4(vtmp(3), vreg(iter, 2), vreg(iter, 3), 0xdd);

            vshuff32x4(vreg(iter, 0), vtmp(0), vtmp(2), 0x88);
            vshuff32x4(vreg(iter, 2), vtmp(0), vtmp(2), 0xdd);
            vshuff32x4(vreg(iter, 1), vtmp(1), vtmp(3), 0x88);
            vshuff32x4(vreg(iter, 3), vtmp(1), vtmp(3), 0xdd);
        };

        auto emit_store = [&]() {
            for (int i = 0; i < 4; ++i) {
                for (int iter = 0; iter < niters; ++iter) {
                    const size_t off = i * c.tr_ld + iter * simd_w;
                    vmovups(ptr[reg_ptr_tr_src + off * typesize],
                            vreg(iter, i));
                }
            }
        };

        for (int iter = 0; iter < niters; ++iter)
            emit_load(iter);

        for (int iter = 0; iter < niters; ++iter)
            emit_tr(iter);

        emit_store();
    };

    preamble();

    mov(reg_ptr_src, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_ptr_tr_src, ptr[abi_param1 + GET_OFF(tr_src)]);

    mov(reg_nthr_oc_b.cvt32(), ptr[abi_param1 + GET_OFF(nthr_oc_b)]);
    mov(reg_ih.cvt32(), ptr[abi_param1 + GET_OFF(tr_src_ih_start)]);
    mov(reg_ih_end.cvt32(), ptr[abi_param1 + GET_OFF(tr_src_ih_end)]);
    mov(reg_ptr_tr_src_bctx, ptr[abi_param1 + GET_OFF(tr_src_bctx)]);

    emit_tr_sync();

    Label l_ih_loop, l_tr_done;
    cmp(reg_ih, reg_ih_end);
    je(l_tr_done, T_NEAR);

    mov(reg_tmp, (size_t)&mask[0]);
    vmovups(vmsk, ptr[reg_tmp]);

    if (c.iw % simd_w) {
        const char load_mask = (1 << (c.iw % simd_w)) - 1;
        mov(reg_tmp, load_mask);
        kmovw(kmsk, reg_tmp.cvt32());
    }

    /* src += ih_start * c.iw; */
    imul(reg_tmp, reg_ih, c.iw * typesize);
    add(reg_ptr_src, reg_tmp);
    /* tr_src += ih_start * c.stride_w * c.tr_ld; */
    imul(reg_tmp, reg_ih, c.stride_w * c.tr_ld * typesize);
    add(reg_ptr_tr_src, reg_tmp);

    L(l_ih_loop);
    {
        emit_tr_iw();

        add(reg_ptr_src, c.iw * typesize);
        add(reg_ptr_tr_src, c.stride_w * c.tr_ld * typesize);

        inc(reg_ih);
        cmp(reg_ih, reg_ih_end);
        jl(l_ih_loop, T_NEAR);
    }

    L(l_tr_done);

    emit_tr_sync();

    postamble();
}

/*
// -------------------------------------------------
// jit_transpose4x16_src
// -------------------------------------------------
*/

void jit_transpose4x16_src::transpose(int nrows) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 4, "Unsupported transpose size");
    if (!nrows) return;

    auto pf_src_t0 = [=](int i) {
        if (tparams->src_pf0_distance)
            prefetcht0(EVEX_compress_addr(
                    reg_src, (tparams->src_pf0_distance + i) * src_stride));
    };

    auto pf_tr_src_t0 = [=](int i) {
        if (tparams->tr_src_pf0_distance)
            prefetcht0(EVEX_compress_addr(reg_tr_src,
                    (tparams->tr_src_pf0_distance + i) * src_stride));
    };

    auto pf_src_t1 = [=](int i) {
        if (tparams->src_pf1)
            prefetcht1(EVEX_compress_addr(reg_src_prf, i * src_stride));
    };

    auto pf_tr_src_t1 = [=](int i) {
        if (tparams->tr_src_pf1)
            prefetchwt1(EVEX_compress_addr(reg_tr_src_prf, i * tr_src_stride));
    };

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(4 + i);
    };

    auto load = [=](int i) {
        vmovups(src_zmm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        vmovups(EVEX_compress_addr(reg_tr_src, i * tr_src_stride), r);
    };

    auto tmp0 = tmp_zmm(0);
    auto tmp1 = tmp_zmm(1);
    auto tmp2 = tmp_zmm(2);
    auto tmp3 = tmp_zmm(3);

    auto src0 = src_zmm(0);
    auto src1 = src_zmm(1);
    auto src2 = src_zmm(2);
    auto src3 = src_zmm(3);
    for (int i = 0; i < nrows; i++) {
        load(i);
    }

    for (size_t i = nrows; i < 4; i++) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src1);
    pf_src_t0(0);
    vpermpd(tmp0 | kF0, vidx01, src2);
    vpermpd(tmp1 | kF0, vidx01, src3);

    valignd(src0, src0, src0, 8);
    valignd(src1, src1, src1, 8);
    pf_src_t0(1);
    vmovupd(tmp2, src0);
    vmovupd(tmp3, src1);
    pf_src_t0(2);
    vpermpd(tmp2 | kF0, vidx10, src2);
    vpermpd(tmp3 | kF0, vidx10, src3);
    pf_src_t0(3);

    vmovupd(src0, tmp0);
    pf_src_t1(0);
    vmovupd(src1, tmp2);
    pf_src_t1(1);
    vmovupd(src2, tmp1);
    pf_src_t1(2);
    vmovupd(src3, tmp3);
    pf_src_t1(3);
    vpermpd(src0 | kCC, vidx1, tmp1);
    vpermpd(src1 | kCC, vidx1, tmp3);
    pf_tr_src_t0(0);
    vpermpd(src2 | k33, vidx1, tmp0);
    vpermpd(src3 | k33, vidx1, tmp2);
    pf_tr_src_t0(1);

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src2);
    pf_tr_src_t0(2);
    vmovupd(tmp2, src1);
    vmovupd(tmp3, src3);
    pf_tr_src_t0(3);
    vpermps(tmp0 | kFFFF, vidxP, src0);
    pf_tr_src_t1(0);
    vpermps(tmp1 | kFFFF, vidxP, src2);
    pf_tr_src_t1(1);
    vpermps(tmp2 | kFFFF, vidxP, src1);
    pf_tr_src_t1(3);
    vpermps(tmp3 | kFFFF, vidxP, src3);
    pf_tr_src_t1(4);

    store(tmp0, 0);
    store(tmp1, 1);
    store(tmp2, 2);
    store(tmp3, 3);
}

alignas(64) static constexpr const int64_t idx01[8] = {0, 0, 0, 0, 0, 1, 2, 3};
alignas(64) static constexpr const int64_t idx10[8] = {0, 0, 0, 0, 4, 5, 6, 7};
alignas(64) static constexpr const int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5};
alignas(64) static constexpr const int32_t idxP[16]
        = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

void jit_transpose4x16_src::generate() {
    preamble();

    const int ic_block = params->ic_block;
    const int is = params->is;
    int tail = is % transpose_size;

    src_stride = ic_block * typesize;
    assert(src_stride == 64);
    tr_src_stride = ic_block * typesize;

    const int src_step = ic_block * transpose_size * typesize;
    const int tr_src_step = ic_block * transpose_size * typesize;

#define GET_TR_OFF(x) offsetof(jit_src_transpose_s, x)
    mov(reg_loop, ptr[param1 + GET_TR_OFF(size)]);
    mov(reg_src, ptr[param1 + GET_TR_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_TR_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_TR_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_TR_OFF(tr_src_prf)]);
#undef GET_TR_OFF

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [=](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    kmovw(kF0, 0xf0); // 11110000
    kmovw(kCC, 0xcc); // 11001100
    kmovw(k33, 0x33); // 00110011
    kmovw(kFFFF, 0xffff); // 1111111111111111

    vmovdqa64(vidx01, idx01);
    vmovdqa64(vidx10, idx10);
    vmovdqa64(vidx1, idx1);
    vmovdqa32(vidxP, idxP);

    Label loop_label;
    Label tail_label;

    cmp(reg_loop, transpose_size);
    jl(tail_label, T_NEAR);

    L(loop_label);
    {
        transpose(transpose_size);
        add(reg_src, src_step);
        add(reg_tr_src, tr_src_step);
        add(reg_src_prf, src_step);
        add(reg_tr_src_prf, tr_src_step);
        sub(reg_loop, transpose_size);
        cmp(reg_loop, transpose_size);
        jge(loop_label, T_NEAR);
    }
    L(tail_label);
    transpose(tail);

    postamble();
}

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf) {
    if (conf->ver == ver_4fma && !conf->is_1stconv)
        return new jit_trans_iw_ic_t(conf);
    if (conf->ver == ver_vnni
            && IMPLICATION(conf->is_1stconv, conf->transpose_src))
        return new jit_trans_iw_ic_int16_t(conf);
    if (conf->ver == ver_4fma && conf->is_1stconv)
        return new jit_trans_iw_x4_4x_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}

jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf) {
    if (conf->ver == ver_vnni) return new jit_trans_ow_oc_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
