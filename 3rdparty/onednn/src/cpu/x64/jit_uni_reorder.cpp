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

#include <assert.h>
#include <numeric>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/cpu_reorder_pd.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) \
    do { \
        __VA_ARGS__ \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
constexpr static bool is_windows = true;
#else
constexpr static bool is_windows = false;
#endif

using namespace Xbyak;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    void operator()(const call_param_t *c) const override {
        jit_generator::operator()(c);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll;
        int len_last_dim_unroll;
        int len_unroll;
    };

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1;

        for (int d = 0; d < ndims; ++d) {
            auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max) return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true && p.ndims > 0
                && utils::one_of(p.itype, f32, bf16, s32, s8, u8)
                && utils::one_of(p.otype, f32, bf16, s32, s8, u8)
                && IMPLICATION(p.itype == bf16,
                        utils::one_of(p.otype, s8, u8, f32, bf16))
                && IMPLICATION(p.otype == bf16,
                        utils::one_of(p.itype, s8, u8, f32, bf16))
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr) && mayiuse(sse41)
                && IMPLICATION((p.itype == bf16 || p.otype == bf16),
                        mayiuse(avx512_core));
        if (!ok) return false;

        const ptrdiff_t max_stride = (1LL << 31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                    && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                    && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok) return false;
        }

        return true;
    }

    int n(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].n;
    }
    int is(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].is;
    }
    int os(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].os;
    }
    int ss(int d) {
        assert(d < prb_.ndims);
        return (int)prb_.nodes[d].ss;
    }

    Address i_addr(int i_off) {
        return ptr[reg_ptr_in + reg_off_in + i_off * itype_sz];
    }

    Address o_addr(int o_off) {
        return ptr[reg_ptr_out + reg_off_out + o_off * otype_sz];
    }

    Address s_addr(int s_off) {
        return ptr[reg_ptr_scale + reg_off_scale + s_off * stype_sz];
    }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int &i_off, int &o_off, int &s_off, int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
            i_off += is(d);
            o_off += os(d);
            s_off += ss(d);

            if (off % n(d)) break;

            i_off += -n(d) * is(d);
            o_off += -n(d) * os(d);
            s_off += -n(d) * ss(d);
            off /= n(d);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, i_off, o_off, dummy,
                step_size);
    }

    void tr8x8_avx2(int i_off, int o_off) {
        using namespace data_type;

        const auto cvt2ps
                = [=](const Ymm &dst, const Operand &src, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              if (src.isMEM() || src.getIdx() != dst.getIdx())
                                  vmovups(dst, src);
                              break;
                          case bf16:
                              vpmovzxwd(dst, src);
                              vpslld(dst, dst, 0x10);
                              break;
                          case s32: vcvtdq2ps(dst, src); break;
                          case s8:
                              vpmovsxbd(dst, src);
                              vcvtdq2ps(dst, dst);
                              break;
                          case u8:
                              vpmovzxbd(dst, src);
                              vcvtdq2ps(dst, dst);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        const auto cvt2odt = [=](const Ymm &ymm, data_type_t odt,
                                     data_type_t idt) {
            Xmm xmm = Xmm(ymm.getIdx());
            switch (odt) {
                case bf16:
                    if (utils::one_of(idt, f32, s8, u8)) {
                        if (idt != f32) cvt2ps(ymm, ymm, idt);
                        if (mayiuse(avx512_core_bf16)) {
                            vcvtneps2bf16(Xmm(ymm.getIdx()), ymm);
                        } else {
                            bf16_emu_->vcvtneps2bf16(
                                    Ymm(ymm.getIdx()), Zmm(ymm.getIdx()));
                        }
                    }
                    break;
                case s32:
                    if (idt == f32)
                        vcvtps2dq(ymm, ymm);
                    else if (idt == s8)
                        vpmovsxbd(ymm, ymm);
                    else if (idt == u8)
                        vpmovzxbd(ymm, ymm);
                    break;
                case s8:
                    if (idt == bf16) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmovsdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero);
                            vpermq(ymm, ymm, 0x58);
                            vpacksswb(ymm, ymm, ymm_zero);
                        }
                    }
                    if (idt == u8) vpminub(ymm, ymm, ymm_8x127b);
                    break;
                case u8:
                    if (idt == bf16) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmaxsd(ymm, ymm, ymm_zero);
                            vpmovusdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero);
                            vpermq(ymm, ymm, 0x58);
                            vpackuswb(ymm, ymm, ymm_zero);
                        }
                    }
                    if (idt == s8) vpmaxsb(ymm, ymm, ymm_zero);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load = [=](const Ymm &ymm, const Address &addr, int size) {
            Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(ymm, addr); break;
                case 16: vmovups(xmm, addr); break;
                case 8: vmovsd(xmm, addr); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [=](const Address &addr, const Ymm &ymm, int size) {
            Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(addr, ymm); break;
                case 16: vmovups(addr, xmm); break;
                case 8: vmovsd(addr, xmm); break;
                default: assert(!"unreachable");
            }
        };

        const int unroll = 8;

        const bool interim_f32 = (prb_.itype != f32)
                || utils::one_of(f32, prb_.itype, prb_.otype);

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        for (int i = 0; i < unroll; i++) {
            using namespace data_type;

            load(Ymm(i), i_addr(i_off + i * is(0)), unroll * itype_sz);

            if (interim_f32) cvt2ps(Ymm(i), Ymm(i), prb_.itype);
        }

        for (int i = 0; i < unroll / 2; i++) {
            vunpcklps(Ymm(unroll + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < unroll / 2; i++) {
            int j = i % 2 == 0 ? unroll + i : i - 1;
            vshufps(Ymm(unroll / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(unroll / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < unroll / 2; i++)
            vperm2f128(Ymm(i), Ymm(unroll / 2 + i), Ymm(unroll + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = unroll / 2; i < unroll; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(unroll / 2 + i), uquad);

        if (need_saturation) {
            init_saturate_f32(ymm_zero, ymm_saturation_ubound, reg_tmp,
                    interim_f32 ? f32 : prb_.itype, prb_.otype);
            for (int i = 0; i < unroll; i++)
                saturate_f32(
                        Ymm(i), ymm_zero, ymm_saturation_ubound, prb_.otype);
        }

        for (int i = 0; i < unroll; i++) {
            if (prb_.otype != f32)
                cvt2odt(Ymm(i), prb_.otype, interim_f32 ? f32 : prb_.itype);
            store(o_addr(o_off + i * os(1)), Ymm(i), unroll * otype_sz);
        }
    }

    bool can_do_tr8x8() {
        using namespace data_type;

        return mayiuse(avx2) && prb_.ndims >= 2
                && ((utils::one_of(prb_.itype, u8, s8, s32, f32, bf16)
                        && utils::one_of(prb_.otype, u8, s8, s32, f32, bf16)))
                && utils::everyone_is(8, n(0), n(1))
                && utils::everyone_is(1, os(0), is(1))
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
    }

    bool process_unroll_tr8x8(int len) {
        if (!can_do_tr8x8()) return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(int len) {
        using namespace data_type;

        using Vmm = typename cpu_isa_traits<isa>::Vmm;
        const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz;

        bool can_do = true && mayiuse(isa)
                && utils::everyone_is(1, os(0), is(0))
                && (false || prb_.itype == prb_.otype
                        || (prb_.itype == s32 && prb_.otype == f32)
                        || (prb_.itype == f32 && prb_.otype == s32))
                && len % simd_w == 0 && n(0) % len == 0
                && prb_.scale_type == scale_type_t::NONE && prb_.beta == 0.f;
        if (!can_do) return false;

        for (int off = 0; off < len;) {
            // TODO: we need extra reg for proper saturation if otype == s32
            const int unroll
                    = nstl::min(16 - (prb_.otype == s32), (len - off) / simd_w);

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(Vmm(ur), i_addr(off + ur * simd_w));

            if (prb_.itype != prb_.otype) {
                for (int ur = 0; ur < unroll; ++ur) {
                    if (prb_.itype == s32 && prb_.otype == f32)
                        uni_vcvtdq2ps(Vmm(ur), Vmm(ur));
                    else if (prb_.itype == f32 && prb_.otype == s32)
                        uni_vcvtps2dq(Vmm(ur), Vmm(ur));
                    else
                        assert(!"unreachable");
                }
            }

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(o_addr(off + ur * simd_w), Vmm(ur));

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off) {
        using namespace data_type;

        // TODO: Clean up the code by using "uni" instructions once
        // jit_generator properly supports avx versions of instructions
        // on Xmm registers.
        const auto cvt2ps = [=](const Xmm &dst, const Operand &src,
                                    data_type_t idt) {
            Xmm dst_pure = Xmm(dst.getIdx());
            if (mayiuse(avx)) {
                switch (idt) {
                    case f32:
                        if (src.isMEM() || src.getIdx() != dst.getIdx())
                            vmovups(dst, src);
                        break;
                    case bf16:
                        vpmovzxwd(dst, src);
                        vpslld(dst, dst, 0x10);
                        break;
                    case s32: vcvtdq2ps(dst, src); break;
                    case s8:
                        vpmovsxbd(dst, src);
                        vcvtdq2ps(dst_pure, dst);
                        break;
                    case u8:
                        vpmovzxbd(dst, src);
                        vcvtdq2ps(dst_pure, dst);
                        break;
                    default: assert(!"unreachable");
                }
            } else {
                switch (idt) {
                    case f32:
                        if (src.isMEM() || src.getIdx() != dst.getIdx())
                            movups(dst, src);
                        break;
                    case s32: cvtdq2ps(dst, src); break;
                    case s8:
                        pmovsxbd(dst, src);
                        cvtdq2ps(dst_pure, dst);
                        break;
                    case u8:
                        pmovzxbd(dst, src);
                        cvtdq2ps(dst_pure, dst);
                        break;
                    default: assert(!"unreachable");
                }
            }
        };

        const auto cvt2odt = [=](const Xmm &xmm, data_type_t odt,
                                     data_type_t idt) {
            if (mayiuse(avx)) {
                switch (odt) {
                    case bf16:
                        if (utils::one_of(idt, f32, s8, u8)) {
                            if (idt != f32) cvt2ps(xmm, xmm, idt);
                            if (mayiuse(avx512_core_bf16)) {
                                vcvtneps2bf16(xmm, xmm);
                            } else {
                                bf16_emu_->vcvtneps2bf16(
                                        Ymm(xmm.getIdx()), Zmm(xmm.getIdx()));
                            }
                        }
                        break;
                    case s32:
                        if (idt == f32)
                            vcvtps2dq(xmm, xmm);
                        else if (idt == s8)
                            vpmovsxbd(xmm, xmm);
                        else if (idt == u8)
                            vpmovzxbd(xmm, xmm);
                        break;
                    case s8:
                        if (idt == bf16) cvt2ps(xmm, xmm, idt);
                        if (utils::one_of(idt, f32, bf16)) vcvtps2dq(xmm, xmm);
                        if (utils::one_of(idt, bf16, f32, s32)) {
                            if (mayiuse(avx512_core)) {
                                vpmovsdb(xmm, xmm);
                            } else {
                                vpackssdw(xmm, xmm, xmm_zero);
                                vpacksswb(xmm, xmm, xmm_zero);
                            }
                        }
                        if (idt == u8) vpminub(xmm, xmm, xmm_4x127b);
                        break;
                    case u8:
                        if (idt == bf16) cvt2ps(xmm, xmm, idt);
                        if (utils::one_of(idt, f32, bf16)) vcvtps2dq(xmm, xmm);
                        if (utils::one_of(idt, bf16, f32, s32)) {
                            if (mayiuse(avx512_core)) {
                                vpmaxsd(xmm, xmm, xmm_zero);
                                vpmovusdb(xmm, xmm);
                            } else {
                                vpackssdw(xmm, xmm, xmm_zero);
                                vpackuswb(xmm, xmm, xmm_zero);
                            }
                        }
                        if (idt == s8) vpmaxsb(xmm, xmm, xmm_zero);
                        break;
                    default: assert(!"unreachable");
                }
            } else {
                switch (odt) {
                    case s32:
                        if (idt == f32)
                            cvtps2dq(xmm, xmm);
                        else if (idt == s8)
                            pmovsxbd(xmm, xmm);
                        else if (idt == u8)
                            pmovzxbd(xmm, xmm);
                        break;
                    case s8:
                        if (idt == f32) cvtps2dq(xmm, xmm);
                        if (utils::one_of(idt, f32, s32)) {
                            packssdw(xmm, xmm_zero);
                            packsswb(xmm, xmm_zero);
                        }
                        if (idt == u8) pminub(xmm, xmm_4x127b);
                        break;
                    case u8:
                        if (idt == f32) cvtps2dq(xmm, xmm);
                        if (utils::one_of(idt, f32, s32)) {
                            packssdw(xmm, xmm_zero);
                            packuswb(xmm, xmm_zero);
                        }
                        if (idt == s8) pmaxsb(xmm, xmm_zero);
                        break;
                    default: assert(!"unreachable");
                }
            }
        };

        auto load = [=](const Xmm &xmm, const Address &addr, int size) {
            switch (size) {
                case 16: movups(xmm, addr); break;
                case 8: movsd(xmm, addr); break;
                case 4: movss(xmm, addr); break;
                case 2: pinsrw(xmm, addr, 0x0); break;
                case 1: pinsrb(xmm, addr, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [=](const Address &addr, const Xmm &xmm, int size) {
            switch (size) {
                case 16: movups(addr, xmm); break;
                case 8: movsd(addr, xmm); break;
                case 4: movss(addr, xmm); break;
                case 2: pextrw(addr, xmm, 0x0); break;
                case 1: pextrb(addr, xmm, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        static constexpr int xmm_vlen = 4;
        bool can_load_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1) can_load_xmm = false;
        const int load_step = can_load_xmm ? xmm_vlen : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1) can_store_xmm = false;
        const int ur_step = can_store_xmm ? 4 : 1;

        const bool interim_f32 = false
                || utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.scale_type != scale_type_t::NONE || prb_.beta != 0.f;

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        if (!can_load_xmm && can_store_xmm) {
            assert(ur_step == xmm_vlen);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                for (int r = 0; r < ur_step; ++r) {
                    if (itype_sz == 4)
                        pinsrd(Xmm(ur), i_addr(i_off[ur + r]), r);
                    else if (itype_sz == 2)
                        pinsrw(Xmm(ur), i_addr(i_off[ur + r]), r);
                    else
                        pinsrb(Xmm(ur), i_addr(i_off[ur + r]), r);
                }
            }
        } else {
            for (int ur = 0; ur < reg_unroll; ur += load_step)
                load(Xmm(ur), i_addr(i_off[ur]), load_step * itype_sz);
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(Xmm(ur), Xmm(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            const bool fast_return = true // transposition on the fly
                    && prb_.scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f;
            if (fast_return) {
                if (prb_.scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        mulps(Xmm(ur), xmm_scale);
                if (prb_.otype != f32) {
                    init_saturate_f32(xmm_zero, xmm_saturation_ubound, reg_tmp,
                            interim_f32 ? f32 : prb_.itype, prb_.otype);
                    for (int ur = 0; ur < reg_unroll; ur += load_step) {
                        if (need_saturation)
                            saturate_f32(Xmm(ur), xmm_zero,
                                    xmm_saturation_ubound, xmm_saturate,
                                    prb_.otype);
                        cvt2odt(Xmm(ur), prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                    }
                }
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz == 4)
                            pextrd(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else if (otype_sz == 2)
                            pextrw(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else
                            pextrb(o_addr(o_off[ur + r]), Xmm(ur), r);
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        if (mayiuse(avx))
                            vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
                        else {
                            movups(Xmm(ur + r), Xmm(ur));
                            shufps(Xmm(ur + r), Xmm(ur), r);
                        }
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        if (mayiuse(avx))
                            vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur),
                                    itype_sz * r);
                        else {
                            movups(Xmm(ur + r), Xmm(ur));
                            palignr(Xmm(ur + r), Xmm(ur), itype_sz * r);
                        }
                    }
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulps(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type
                            = scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast) {
                        movss(xmm_scale, s_addr(s_off[ur]));
                        shufps(xmm_scale, xmm_scale, 0x0);
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load) {
                        movups(xmm_scale, s_addr(s_off[ur]));
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r)
                        pinsrd(xmm_scale, s_addr(s_off[r]), r - ur);
                    mulps(Xmm(ur), xmm_scale);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        /* non VEX instructions do not support unaligned
                         * memory for instructions other than movups. */
                        if (mayiuse(avx)) {
                            vaddps(Xmm(ur), o_addr(o_off[ur]));
                        } else {
                            /* register xmm(1) is unused */
                            movups(Xmm(1), o_addr(o_off[ur]));
                            addps(Xmm(ur), Xmm(1));
                        }
                    } else {
                        cvt2ps(Xmm(1), o_addr(o_off[ur]), prb_.otype);
                        if (mayiuse(avx))
                            vaddps(Xmm(ur), Xmm(1));
                        else
                            addps(Xmm(ur), Xmm(1));
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulss(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    mulss(Xmm(ur), s_addr(s_off[ur]));
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        addss(Xmm(ur), o_addr(o_off[ur]));
                    } else {
                        if (prb_.otype == s32) {
                            vmovss(xmm_tmp, o_addr(o_off[ur]));
                        } else if (utils::one_of(prb_.otype, s8, u8)) {
                            pinsrb(xmm_tmp, o_addr(o_off[ur]), 0x0);
                        } else if (prb_.otype == bf16) {
                            pinsrw(xmm_tmp, o_addr(o_off[ur]), 0x0);
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp, xmm_tmp, prb_.otype);
                        if (mayiuse(avx))
                            vaddps(Xmm(ur), xmm_tmp);
                        else
                            addps(Xmm(ur), xmm_tmp);
                    }
                }
            }
        }

        if (need_saturation) {
            init_saturate_f32(
                    xmm_zero, xmm_saturation_ubound, reg_tmp, f32, prb_.otype);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                saturate_f32(Xmm(ur), xmm_zero, xmm_saturation_ubound,
                        xmm_saturate, prb_.otype);
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.otype != f32)
                cvt2odt(Xmm(ur), prb_.otype, interim_f32 ? f32 : prb_.itype);
            store(o_addr(o_off[ur]), Xmm(ur), ur_step * otype_sz);
        }
    }

    void process_unroll_generic(int len) {
        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;

            /* compute offsets */
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr * blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        i_off[ur_c], o_off[ur_c], s_off[ur_c]);
            }

            process_unroll_generic_step(reg_unroll, i_off + curr * blk,
                    o_off + curr * blk, s_off + curr * blk);

            curr = 1 - curr;
        }
    }

    void loop_begin(Label &l, Reg64 reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void loop_end(Label &l, Reg64 reg_cnt, int len, int i_step, int o_step,
            int s_step) {
        add(reg_off_in, i_step * itype_sz);
        add(reg_off_out, o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            add(reg_off_scale, s_step * stype_sz);
        dec(reg_cnt);
        jnz(l);

        sub(reg_off_in, len * i_step * itype_sz);
        sub(reg_off_out, len * o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            sub(reg_off_scale, len * s_step * stype_sz);
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        xor_(reg_off_in, reg_off_in);
        xor_(reg_off_out, reg_off_out);
        if (prb_.scale_type == scale_type_t::MANY)
            xor_(reg_off_scale, reg_off_scale);

        Label l_loop[3];
        Reg64 reg_cnt[3] = {r15, r14, r13};

        if (n_jit_loops > 2) loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1) loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        bool optimized = false;
        optimized = optimized || process_direct_copy<avx>(d.len_unroll);
        optimized = optimized || process_direct_copy<sse41>(d.len_unroll);
        optimized = optimized || process_unroll_tr8x8(d.len_unroll);
        if (!optimized) process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu, is(nfu + 0) * ldu,
                    os(nfu + 0) * ldu, ss(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1], n(nfu + 1), is(nfu + 1),
                    os(nfu + 1), ss(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2], n(nfu + 2), is(nfu + 2),
                    os(nfu + 2), ss(nfu + 2));

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32_t(const desc_t &desc)
        : kernel_t(desc), bf16_emu_(nullptr) {
        itype_sz = data_type_size(prb_.itype);
        otype_sz = data_type_size(prb_.otype);
        stype_sz = sizeof(float);
        if (prb_.otype == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4);
            bf16_emu_->init_vcvtneps2bf16();
        }
    }

    void generate() override {
        preamble();
#define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.scale_type == scale_type_t::COMMON) {
            auto reg_ptr_scale_tmp = reg_ptr_in;
            mov(reg_ptr_scale_tmp, PARAM(scale));
            if (mayiuse(avx))
                vbroadcastss(xmm_scale, ptr[reg_ptr_scale_tmp]);
            else {
                movss(xmm_scale, ptr[reg_ptr_scale_tmp]);
                shufps(xmm_scale, xmm_scale, 0x0);
            }
        } else if (prb_.scale_type == scale_type_t::MANY) {
            mov(reg_ptr_scale, PARAM(scale));
        }
        mov(reg_ptr_in, PARAM(in));
        mov(reg_ptr_out, PARAM(out));
#undef PARAM

        if (can_do_tr8x8()) {
            vxorps(ymm_zero, ymm_zero, ymm_zero);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp, 0x7f7f7f7f7f7f7f7f);
                movq(Xmm(ymm_8x127b.getIdx()), reg_tmp);
            }
        } else {
            if (mayiuse(avx))
                vxorps(xmm_zero, xmm_zero, xmm_zero);
            else
                xorps(xmm_zero, xmm_zero);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp.cvt32(), 0x7f7f7f7f);
                movd(xmm_4x127b, reg_tmp.cvt32());
            }
        }

        impl();
        postamble();
    }
    ~jit_uni_reorder_kernel_f32_t() override { delete bf16_emu_; }

private:
    int itype_sz;
    int otype_sz;
    int stype_sz;

    Reg64 reg_ptr_in = rsi;
    Reg64 reg_ptr_out = rdx;
    Reg64 reg_ptr_scale = abi_not_param1;

    Reg64 reg_off_in = r8;
    Reg64 reg_off_out = r9;
    Reg64 reg_off_scale = r10;

    Reg64 reg_tmp = rax;

    Xmm xmm_scale = xmm15;
    Xmm xmm_zero = xmm14;
    Xmm xmm_4x127b = xmm13; // TODO: unite with ymm_zero
    Ymm ymm_zero = ymm14;
    Ymm ymm_8x127b = ymm13;
    Xmm xmm_tmp = xmm12;
    Xmm xmm_saturation_ubound = xmm12;
    Ymm ymm_saturation_ubound = ymm12;
    Xmm xmm_saturate = xmm11;

    /* bf16 support on SKX */
    bf16_emulation_t *bf16_emu_;
    Zmm bf16_emu_reserv_1 = Zmm(16);
    Zmm bf16_emu_reserv_2 = Zmm(17);
    Reg64 bf16_emu_scratch = reg_tmp;
    Zmm bf16_emu_reserv_3 = Zmm(18);
    Zmm bf16_emu_reserv_4 = Zmm(19);
};

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims >= 2 && mayiuse(avx2)
                && p.scale_type == scale_type_t::NONE
                && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
                && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f;
        if (!ok) return false;

        int64_t n0 = p.nodes[0].n;
        auto i0 = p.nodes[0].is;
        auto o0 = p.nodes[0].os;
        int64_t n1 = p.nodes[1].n;
        auto i1 = p.nodes[1].is;
        auto o1 = p.nodes[1].os;

        /*
         * for a transpose of plain to 8c case, nodes would be like:
         *     n    is   os
         *     m    1    8
         *     8    m    1
         * or
         *     8    m    1
         *     m    1    8
         */
        ok = (utils::one_of(n0, 8, 16) || utils::one_of(n1, 8, 16))
                && ((i0 == 1 && o1 == 1 && n0 == i1 && o0 == n1)
                        || (o0 == 1 && i1 == 1 && n0 == o1 && i0 == n1));
        if (!ok) return false;

        // Do not handle transpose of dimensions other than last 2
        for (int i = 2; i < p.ndims; ++i) {
            if (p.nodes[i].is != p.nodes[i].os) {
                ok = false;
                break;
            }
        }

        return ok;
    }

    jit_single_blk_kernel_t(const tr::prb_t &prb)
        : prb_(prb)
        , itype_sz(data_type_size(prb_.itype))
        , otype_sz(data_type_size(prb_.otype))
        , block_sz(prb.nodes[0].n) {}

    void generate() override {
        auto input_stride
                = prb_.nodes[0].is != 1 ? prb_.nodes[0].is : prb_.nodes[1].is;
        auto output_stride
                = prb_.nodes[0].os != 1 ? prb_.nodes[0].os : prb_.nodes[1].os;

        Label tail_processing;

        preamble();
        cmp(reg_ptr_tail, true);
        je(tail_processing, T_NEAR);

        if (block_sz == 8) {
            gen_ker8x8(0, 0, input_stride, output_stride, 8, 8);
            block_sz = 8;
        } else if (block_sz == 16) {
            gen_ker16x16_in_8x8(input_stride, output_stride);
            block_sz = 16;
        } else {
            assert(!"unimplemented");
        }

        postamble();

        L(tail_processing);

        if (block_sz == 8) {
            auto i_tail = input_stride % 8 != 0 ? input_stride % 8 : 8;
            auto o_tail = output_stride % 8 != 0 ? output_stride % 8 : 8;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 8 ? o_tail : i_tail;
                gen_setmask(t_mask);
                gen_ker8x8(0, 0, input_stride, output_stride, i_tail, o_tail);
            }
        } else if (block_sz == 16) {
            auto i_tail = input_stride % 16 != 0 ? input_stride % 16 : 16;
            auto o_tail = output_stride % 16 != 0 ? output_stride % 16 : 16;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 16 ? o_tail : i_tail;
                t_mask %= 8;
                if (t_mask != 0) gen_setmask(t_mask);
                gen_ker16x16_in_8x8(
                        input_stride, output_stride, i_tail, o_tail);
            }
        } else {
            assert(!"unimplemented");
        }

        postamble();
    }

    void gen_loadu(const Ymm &ymm, const Address &addr, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(ymm, addr); break;
            case 16: vmovups(xmm, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_storeu(const Address &addr, const Ymm &ymm, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(addr, ymm); break;
            case 16: vmovups(addr, xmm); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskloadu(
            const Ymm &ymm, const Address &addr, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(ymm, mask, addr); break;
            case 16: vmaskmovps(xmm, mask128, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskstoreu(
            const Address &addr, const Ymm &ymm, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(addr, mask, ymm); break;
            case 16: vmaskmovps(addr, mask128, xmm); break;
            default: assert(!"unreachable");
        }
    }

    // Register allocation xmm0~11
    void gen_transpose_8x8() {
        constexpr int lane = 8;
        for (int i = 0; i < lane / 2; i++) {
            vunpcklps(Ymm(lane + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < lane / 2; i++) {
            int j = i % 2 == 0 ? lane + i : i - 1;
            vshufps(Ymm(lane / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(lane / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < lane / 2; i++)
            vperm2f128(Ymm(i), Ymm(lane / 2 + i), Ymm(lane + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = lane / 2; i < lane; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(lane / 2 + i), uquad);
    }

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask) {
        // all 0, all 1
        vxorps(ymm_tmp, ymm_tmp, ymm_tmp);
        vpcmpeqd(ymm_mask, ymm_mask, ymm_mask);
        // blend in
        auto in_mask = -1 << mask;
        vpblendd(ymm_mask, ymm_mask, ymm_tmp, in_mask);
    }

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        constexpr int lane = 8;

        if (in_tail == 0 || out_tail == 0) return;

        for (int i = 0; i < out_tail; ++i) {
            if (in_tail != lane) {
                gen_maskloadu(Ymm(i),
                        ptr[reg_ptr_in + i_off + i * input_stride * itype_sz],
                        ymm_mask, lane * itype_sz);
            } else {
                gen_loadu(Ymm(i),
                        ptr[reg_ptr_in + i_off + i * input_stride * itype_sz],
                        lane * itype_sz);
            }
        }

        gen_transpose_8x8();

        for (int i = 0; i < in_tail; ++i) {
            if (out_tail == lane) {
                gen_storeu(
                        ptr[reg_ptr_out + o_off + i * output_stride * otype_sz],
                        Ymm(i), lane * otype_sz);
            } else {
                gen_maskstoreu(
                        ptr[reg_ptr_out + o_off + i * output_stride * otype_sz],
                        Ymm(i), ymm_mask, lane * otype_sz);
            }
        }
    }

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
    }

    void gen_ker16x16_in_8x8(int input_stride, int output_stride) {
        const auto lane = 16;
        const auto sub_lane = lane / 2;
        gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(input_stride * sub_lane * itype_sz, sub_lane * otype_sz,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(sub_lane * itype_sz, output_stride * sub_lane * otype_sz,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8((input_stride * sub_lane + sub_lane) * itype_sz,
                (output_stride * sub_lane + sub_lane) * otype_sz, input_stride,
                output_stride, sub_lane, sub_lane);
    }

    // tail can be 1 ~ 16, using avx2 for now
    void gen_ker16x16_in_8x8(
            int input_stride, int output_stride, int in_tail, int out_tail) {
        constexpr auto lane = 16;
        constexpr auto sub_lane = lane / 2;
        auto tail = in_tail != lane ? in_tail : out_tail;

        const auto l_tail = tail < sub_lane ? tail : sub_lane;
        const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

        if (tail == in_tail) {
            gen_tr8x8(0, 0, input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(input_stride * sub_lane * itype_sz, sub_lane * otype_sz,
                    input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(sub_lane * itype_sz, output_stride * sub_lane * otype_sz,
                    input_stride, output_stride, u_tail, sub_lane);
            gen_tr8x8(itype_sz * (input_stride * sub_lane + sub_lane),
                    otype_sz * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, u_tail, sub_lane);
        } else {
            gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, l_tail);
            gen_tr8x8(input_stride * sub_lane * itype_sz, sub_lane * otype_sz,
                    input_stride, output_stride, sub_lane, u_tail);
            gen_tr8x8(sub_lane * itype_sz, output_stride * sub_lane * itype_sz,
                    input_stride, output_stride, sub_lane, l_tail);
            gen_tr8x8(itype_sz * (input_stride * sub_lane + sub_lane),
                    otype_sz * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, sub_lane, u_tail);
        }
    }

private:
    // 6 ~ 12
    constexpr static int xmm_save_for_windows = is_windows ? 7 : 0;
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble() {
        if (is_windows) {
            sub(rsp, xmm_save_for_windows * xmm_width);
            for (int i = 0; i < xmm_save_for_windows; ++i) {
                movdqu(ptr[rsp + i * xmm_width],
                        Xbyak::Xmm(xmm_save_start_from + i));
            }
        }
    }

    void postamble() {
        if (is_windows) {
            for (int i = 0; i < xmm_save_for_windows; ++i)
                movdqu(Xbyak::Xmm(xmm_save_start_from + i),
                        ptr[rsp + i * xmm_width]);
            add(rsp, xmm_save_for_windows * xmm_width);
        }
        uni_vzeroupper();
        ret();
    }

    const prb_t &prb_;

    int itype_sz;
    int otype_sz;
    int block_sz;

    Reg64 reg_ptr_in = abi_param1;
    Reg64 reg_ptr_out = abi_param2;
    // Windows bool is 1-byte in register
    Reg8 reg_ptr_tail = is_windows ? r8b : dl;

    Ymm ymm_mask = ymm12;
    Ymm ymm_tmp = ymm0;
};

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32_t::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0: return new jit_uni_reorder_kernel_f32_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    const bool cache_blocking_needed = false
            || (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16)
            || (prb.ndims > 1 && prb.nodes[1].is % 64 == 0
                    && prb.nodes[1].n > 16);
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > 16) && (num_elems % 16 == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed) prb_node_split(prb, unit_input_stride_idx, 16);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto input_stride = prb.nodes[0].is;
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = true && (num_elems > 16)
                && (num_elems % 16 == 0) && (input_stride >= 256)
                && (input_stride % 64 == 0);
        if (split_needed) {
            prb_node_split(prb, 0, 16);
            prb_node_move(prb, 1, 2);
        }
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(
        tr::prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min
            = nstl::min<size_t>(16 * nthr, utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_borrow_ker_from_drv = true && kdims < prb.ndims
            && sz_ker_cur < tr::ker_prb_size_min && sz_drv_cur > sz_drv_min;
    if (want_borrow_ker_from_drv) {
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the sz_drv_cur is too small (less than sz_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * sz_drv_cur. */
    bool want_borrow_drv_from_ker = true && sz_ker_cur > tr::ker_prb_size_min
            && sz_drv_cur < sz_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t sz_want_borrow = utils::div_up(sz_drv_min, sz_drv_cur);
        for (; prb.nodes[kdims - 1].n % sz_want_borrow; ++sz_want_borrow)
            ;
        if (sz_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / sz_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            printf("split: ");
            prb_dump(prb);
            printf("ndims_ker_max = %d\n", ndims_ker_max);
        });
    }
}

struct jit_uni_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
            if (prb_init_status != status::success) return prb_init_status;

            DEBUG({
                printf("init : ");
                prb_dump(prb);
            });
            // Sort the prb array in increasing sizes of the output stride
            prb_normalize(prb);
            DEBUG({
                printf("norm : ");
                prb_dump(prb);
            });
            /* Combine the variables, which appear together on both
             * sides of the reorder */
            prb_simplify(prb);
            DEBUG({
                printf("smpl : ");
                prb_dump(prb);
            });

            prb_block_for_cache(prb);
            DEBUG({
                printf("cache: ");
                prb_dump(prb);
            });

            int ndims_ker_max;
            int nthr = dnnl_get_max_threads();
            prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

            tr::kernel_t::desc_t ker_desc;
            status_t ker_init_status
                    = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
            if (ker_init_status != status::success) return ker_init_status;

            const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
            if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
                return status::unimplemented;

            DEBUG({
                printf("ker  : ");
                prb_dump(ker_desc.prb);
            });

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->prb_ = prb;
            _pd->ker_desc_ = ker_desc;
            _pd->init_scratchpad_md();
            _pd->nthr_ = nthr;
            return safe_ptr_assign(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
        int nthr_;
    };

    jit_uni_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    void omp_driver_0d(
            int off, const char *in, char *out, const float *scale) const {
        tr::call_param_t c {in, out, scale};
        (*kernel_)(&c);
    }

    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
            c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss;
                    (*kernel_)(&c);
                });
    }

    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
                (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss
                            + d2 * ns[2].ss;
                    (*kernel_)(&c);
                });
    }

    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
                (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                    auto c = tr::call_param_t();
                    c.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                                      + d3 * ns[3].is)
                                    * data_type_size(pd()->prb_.itype);
                    c.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                                      + d3 * ns[3].os)
                                    * data_type_size(pd()->prb_.otype);
                    c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss
                            + d2 * ns[2].ss + d3 * ns[3].ss;
                    (*kernel_)(&c);
                });
    }

    void omp_driver(const char *in, char *out, const float *scale) const {
        in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
        out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

        DEBUG({
            printf("prb : ");
            tr::prb_dump(pd()->prb_);
        });
        DEBUG({
            printf("ker : ");
            tr::prb_dump(pd()->ker_desc_.prb);
        });

        int ndims = pd()->prb_.ndims;
        int ndims_ker = pd()->ker_desc_.prb.ndims;
        assert(ndims - ndims_ker <= ndims_driver_max);

        if (ndims - ndims_ker == 0) {
            omp_driver_0d(ndims_ker, in, out, scale);
        } else {
            parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
                switch (ndims - ndims_ker) {
                    case 1:
                        omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale);
                        break;
                    case 2:
                        omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale);
                        break;
                    case 3:
                        omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale);
                        break;
                    case 4:
                        omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale);
                        break;
                    default: assert(!"unimplemented");
                }
            });
        }
    }

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
        auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);
        DEFINE_SCALES_BUFFER(scales);

        omp_driver(in, out, scales);

        return status::success;
    }

    enum { ndims_driver_max = 4 };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::kernel_t> kernel_;
};

struct jit_blk_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("jit:blk", jit_blk_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
            if (prb_init_status != status::success) return prb_init_status;

            DEBUG({
                printf("init : ");
                prb_dump(prb);
            });
            // Sort the prb array in increasing sizes of the output stride
            prb_normalize(prb);
            DEBUG({
                printf("norm : ");
                prb_dump(prb);
            });
            /* Combine the variables, which appear together on both
             * sides of the reorder */
            prb_simplify(prb);
            DEBUG({
                printf("smpl : ");
                prb_dump(prb);
            });
            prb_tile_normalize(prb);
            DEBUG({
                printf("tile : ");
                prb_dump(prb);
            });

            if (!tr::jit_single_blk_kernel_t::applicable(prb)) {
                return status::unimplemented;
            }

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->prb_ = prb;
            _pd->init_scratchpad_md();
            return safe_ptr_assign(*reorder_pd, _pd);
        }

        tr::prb_t prb_;

    private:
        // Swap last two nodes, put block 4, 8, 16 nodes to first
        static void prb_tile_normalize(tr::prb_t &p) {
            if (!utils::one_of(p.nodes[0].n, 8ul, 16ul)
                    && utils::one_of(p.nodes[1].n, 8ul, 16ul)) {
                nstl::swap(p.nodes[0], p.nodes[1]);
            }
        }
    };

    jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    size_t n(int d) const {
        assert(d < pd()->prb_.ndims);
        return (int)pd()->prb_.nodes[d].n;
    }
    ptrdiff_t is(int d) const {
        assert(d < pd()->prb_.ndims);
        return pd()->prb_.nodes[d].is;
    }
    ptrdiff_t os(int d) const {
        assert(d < pd()->prb_.ndims);
        return pd()->prb_.nodes[d].os;
    }

    status_t init(engine_t *engine) override {
        kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
        auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

        // kernel handle 2-dimension tiles, a tail is possible
        auto &prb = this->pd()->prb_;
        ptrdiff_t BH = 1;
        for (int i = 2; i < prb.ndims; ++i) {
            BH *= prb.nodes[i].n;
        }

        auto block_sz = n(0);
        auto n1 = n(1);
        auto i1 = is(1);
        auto o1 = os(1);
        auto FL = (n1 + block_sz - 1) / block_sz;
        auto bh_stride = BH == 1 ? 0 : is(2);

        auto itype_sz = data_type_size(pd()->prb_.itype);
        auto otype_sz = data_type_size(pd()->prb_.otype);

        parallel_nd(BH, FL, [&](dim_t bh, dim_t fl) {
            auto fl_b = fl * block_sz;
            auto bh_b = bh_stride * bh;
            auto *i = in + (bh_b + fl_b * i1) * itype_sz;
            auto *o = out + (bh_b + fl_b * o1) * otype_sz;
            (*kernel_)(i, o, n1 - fl_b < block_sz);
        });

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::jit_single_blk_kernel_t> kernel_;
};

status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd, engine_t *engine,
        const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto ret = jit_blk_reorder_t::pd_t::create(
            reorder_pd, engine, attr, src_engine, src_md, dst_engine, dst_md);
    if (status::success != ret)
        ret = jit_uni_reorder_t::pd_t::create(reorder_pd, engine, attr,
                src_engine, src_md, dst_engine, dst_md);
    return ret;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
