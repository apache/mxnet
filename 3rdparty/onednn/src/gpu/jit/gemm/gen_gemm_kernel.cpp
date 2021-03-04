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

#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gemm_recipes.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

namespace {

char layout_char(MatrixLayout layout) {
    switch (layout) {
        default: assert(!"Unknown layout.");
        case MatrixLayout::PackedColumns: return 'A';
        case MatrixLayout::PackedRows: return 'B';
        case MatrixLayout::Nontranspose: return 'N';
        case MatrixLayout::Transpose: return 'T';
    }
}

char precision_char(Type T) {
    switch (T) {
        default: assert(!"Unknown type.");
        case Type::f16: return 'H';
        case Type::f32: return 'S';
        case Type::u8:
        case Type::s8: return 'O';
        case Type::u16:
        case Type::s16: return 'W';
        case Type::u32:
        case Type::s32: return 'I';
    }
}

AccessType get_access_type(char c) {
    switch (c) {
        default: assert(!"Unknown access type.");
        case 'b': return AccessType::Block;
        case 's': return AccessType::Scattered;
        case 'u': return AccessType::SurfaceScattered;
        case 'm': return AccessType::MediaBlock;
    }
}

ngen::AddressBase get_address_base(char c) {
    switch (c) {
        default: assert(!"Unknown address space.");
        case 'a': return ngen::AddressBase::createA64(true);
        case 'c': return ngen::AddressBase::createCC(0);
        case 'm': return ngen::AddressBase::createSC(0);
        case 's': return ngen::AddressBase::createBTS(0);
    }
}

} // anonymous namespace

status_t gen_gemm_kernel_t::complete_strategy() {
    using ngen::HW;

    problem_.nonuniformWGs = false;
    problem_.fused = (hw_ >= HW::Gen12LP);
    strategy_.emulate64 = (hw_ == HW::Gen11 || hw_ == HW::Gen12LP);
    strategy_.emulateDWxDW = (hw_ >= HW::Gen12LP);
    strategy_.checkAdd32 = strategy_.emulate64;
    strategy_.spf = !problem_.fused;

    for (int r = 0; r < gemm_recipe_count; r++) {
        auto &recipe = gemm_recipes[r];
        if (recipe.hw == hw_
                && recipe.precisions[0] == precision_char(problem_.Ta)
                && recipe.precisions[1] == precision_char(problem_.Tb)
                && recipe.precisions[2] == precision_char(problem_.Tc)
                && recipe.layouts[0] == layout_char(problem_.A.layout)
                && recipe.layouts[1] == layout_char(problem_.B.layout)
                && recipe.layouts[2] == layout_char(problem_.C.layout)
                && recipe.crosspacks.a == problem_.A.crosspack
                && recipe.crosspacks.b == problem_.B.crosspack
                && recipe.unrollM == strategy_.unroll[LoopM]
                && recipe.unrollN == strategy_.unroll[LoopN]) {

            return read_strategy(recipe.strategyString);
        }
    }

    return status::unimplemented;
}

status_t gen_gemm_kernel_t::read_strategy(const char *str) {
    using ngen::HW;
    std::stringstream s(str);

    bool override_fused_loop = false;
    bool override_register_scheme = false;
    bool override_c_remainder = false;

    bool dp4aIGEMM = hw_ >= HW::Gen12LP && problem_.Ta.size() == 1
            && problem_.Tb.size() == 1 && problem_.Tc.size() == 4;

    strategy_.ka_load_masked = strategy_.kb_load_masked = 0;
    strategy_.unroll[LoopK] = 1;
    strategy_.fmaSIMD = 64
            / std::max<int>({problem_.Ta.size(), problem_.Tb.size(),
                    problem_.Tc.size()});

    strategy_.kernelCrosspack = dp4aIGEMM ? 4 : 1;

    strategy_.remHandling[LoopM] = RemainderHandling::Split;
    strategy_.remHandling[LoopN] = RemainderHandling::Split;
    strategy_.remHandling[LoopK] = RemainderHandling::General;

    char asA, asB, asC, accessA, accessB, accessC, eat;
    s >> std::ws >> asA >> accessA >> strategy_.ka_load;
    if (s.peek() == '/') s >> eat >> strategy_.ka_load_masked;
    if (s.peek() == 'x') s >> eat >> strategy_.A_copies;
    s >> std::ws >> asB >> accessB >> strategy_.kb_load;
    if (s.peek() == '/') s >> eat >> strategy_.kb_load_masked;
    if (s.peek() == 'x') s >> eat >> strategy_.B_copies;
    s >> std::ws >> asC >> accessC;

    problem_.A.base = get_address_base(asA);
    problem_.B.base = get_address_base(asB);
    problem_.C.base = get_address_base(asC);
    strategy_.A.accessType = get_access_type(accessA);
    strategy_.B.accessType = get_access_type(accessB);
    strategy_.C.accessType = get_access_type(accessC);

    while (!s.eof()) {
        std::string mod;
        s >> mod;
        if (mod == "cs")
            strategy_.registerScheme = GEMMStrategy::CSeparate;
        else if (mod == "acb")
            strategy_.registerScheme = GEMMStrategy::ACB;
        else if (mod == "bca")
            strategy_.registerScheme = GEMMStrategy::BCA;
        else if (mod == "vnc")
            strategy_.registerScheme = GEMMStrategy::VNC;
        else if (mod == "int")
            strategy_.registerScheme = GEMMStrategy::ABInterleave;
        else if (mod == "ar") {
            override_c_remainder = true;
            strategy_.altCRemainder = true;
        } else if (mod == "sr") {
            override_c_remainder = true;
            strategy_.altCRemainder = false;
        } else if (mod == "ac")
            strategy_.cAccumulators = true;
        else if (mod == "da")
            strategy_.duplicateA = true;
        else if (mod == "db")
            strategy_.duplicateB = true;
        else if (mod == "el")
            strategy_.cLoadAhead = true;
        else if (mod == "di")
            strategy_.delayABInc = true;
        else if (mod == "ws")
            strategy_.wgInSS = true;
        else if (mod == "nmk") {
            strategy_.loopOrder[0] = LoopN;
            strategy_.loopOrder[1] = LoopM;
            strategy_.loopOrder[2] = LoopK;
        } else if (mod == "fm") {
            problem_.fusedLoop = LoopM;
            override_fused_loop = true;
        } else if (mod == "fn") {
            problem_.fusedLoop = LoopN;
            override_fused_loop = true;
        } else if (mod == "njs")
            strategy_.jointSplit = false;
        else if (mod == "kb") {
            strategy_.kBlocking = true;
            strategy_.C.atomic = true;
        } else if (mod == "wg") {
            char x;
            s >> strategy_.wg[LoopM];
            s >> std::ws >> x;
            s >> strategy_.wg[LoopN];
        } else if (mod.length() >= 2) {
            switch (mod[0]) {
                case 'b':
                    switch (mod[1]) {
                        case 'm':
                            strategy_.blocking[LoopM] = stoi(mod.substr(2));
                            break;
                        case 'n':
                            strategy_.blocking[LoopN] = stoi(mod.substr(2));
                            break;
                        case 'k':
                            strategy_.blocking[LoopK] = stoi(mod.substr(2));
                            break;
                        default: break;
                    }
                    break;
                case 'c': {
                    mod.erase(0, 1);
                    if (mod[0] == 'a') {
                        mod.erase(0, 1);
                        strategy_.slmA = true;
                    }
                    if (mod[0] == 'b') {
                        mod.erase(0, 1);
                        strategy_.slmB = true;
                    }
                    std::stringstream ms(mod);
                    ms >> strategy_.slmBuffers;
                    ms >> eat;
                    if (!ms.eof()) ms >> strategy_.slmCopies;
                    break;
                }
                case 'k': {
                    std::stringstream ms(mod);
                    ms >> eat >> strategy_.unroll[LoopK];
                    if (!ms.eof() && (ms.peek() == '/'))
                        ms >> eat >> strategy_.unrollK_masked;
                    break;
                }
                case 'l': strategy_.optAlignAB = stoi(mod.substr(1)); break;
                case 'r': {
                    bool is_a = (mod[1] == 'a');
                    (is_a ? strategy_.ka_repack : strategy_.kb_repack)
                            = stoi(mod.substr(2));
                    break;
                }
                default:
                    if (mod.substr(0, 2) == "ms")
                        strategy_.mSplitThresh = stoi(mod.substr(2));
                    else if (mod.substr(0, 2) == "ns")
                        strategy_.nSplitThresh = stoi(mod.substr(2));
                    else
                        return status::runtime_error;
                    break;
            }
        } else if (!mod.empty())
            return status::runtime_error;
    }

    if (!override_fused_loop) {
        problem_.fusedLoop = strategy_.loopOrder[0];
        if (problem_.fused) {
            if (strategy_.wg[LoopM] == 1)
                problem_.fusedLoop = LoopN;
            else if (strategy_.wg[LoopN] == 1)
                problem_.fusedLoop = LoopM;
        }
    }

    if (!override_c_remainder) {
        strategy_.altCRemainder = (strategy_.C.accessType == AccessType::Block)
                || strategy_.kBlocking;
    }

    if (!override_register_scheme && (hw_ >= HW::Gen12LP)) {
        strategy_.registerScheme
                = (strategy_.unroll[LoopM] * problem_.Ta.size()
                          == strategy_.unroll[LoopN] * problem_.Tb.size())
                ? GEMMStrategy::VNC
                : GEMMStrategy::ABInterleave;
    }

    if (strategy_.ka_load_masked == 0)
        strategy_.ka_load_masked = strategy_.ka_load;
    if (strategy_.kb_load_masked == 0)
        strategy_.kb_load_masked = strategy_.kb_load;

    strategy_.sanityCheck(hw_, problem_);

    return status::success;
}

status_t gen_gemm_kernel_t::init_interface() {
    using namespace ngen;

    interface_ = NEOInterfaceHandler {hw_};
    auto s_type_ngen = problem_.Ts.ngen();

    interface_.newArgument("A", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("B", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("C", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("offset_A", DataType::q);
    interface_.newArgument("offset_B", DataType::q);
    interface_.newArgument("offset_C", DataType::q);
    interface_.newArgument("lda", DataType::d);
    interface_.newArgument("ldb", DataType::d);
    interface_.newArgument("ldc", DataType::d);
    interface_.newArgument("m", DataType::d);
    interface_.newArgument("n", DataType::d);
    interface_.newArgument("k", DataType::d);
    interface_.newArgument("alpha_real", s_type_ngen);
    interface_.newArgument("beta_real", s_type_ngen);
    if (problem_.abOffset != ABOffset::None)
        interface_.newArgument("abo", DataType::ud);
    if (problem_.cOffset != COffset::None) {
        interface_.newArgument("CO", ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_CO", DataType::d);
    }
    interface_.newArgument("flags", DataType::ud);
    interface_.newArgument("eltwise_alpha", DataType::f);
    interface_.newArgument("eltwise_beta", DataType::f);
    interface_.newArgument("eltwise_scale", DataType::f);
    if (problem_.batchedS) {
        interface_.newArgument("stride_A", DataType::d);
        interface_.newArgument("stride_B", DataType::d);
        interface_.newArgument("stride_C", DataType::d);
    }

    interface_.externalName(kernel_name());

    return status::success;
}

std::vector<unsigned char> gen_gemm_kernel_t::get_binary(
        cl_context ctx, cl_device_id dev) {
    using ngen::HW;

    std::vector<unsigned char> program_binary;

    switch (hw_) {
        case HW::Gen9: {
            gemm_kernel_generator_t<HW::Gen9> generator;
            generator.gemm(problem_, strategy_, interface_);
            program_binary = generator.getBinary(ctx, dev);
            break;
        }
        case HW::Gen12LP: {
            gemm_kernel_generator_t<HW::Gen12LP> generator;
            generator.gemm(problem_, strategy_, interface_);
            program_binary = generator.getBinary(ctx, dev);
            break;
        }
        default: assert(!"Unsupported architecture"); break;
    }

    return program_binary;
}

CommonDriverInfo gen_gemm_kernel_t::driver_info() const {
    return strategy_.driverInfo(problem_);
}

namespace {

// clang-format off
struct kernel_table_t {
    int unrolls[2];
    int max_accept[2]; // Maximum values for m/n for which this kernel will
                       //   always be chosen. (-1 = last in list)
    int min_reject[2]; // Minimum values for m/n beyond which this kernel will
                       //   always be rejected (0 if none)
};

const kernel_table_t gen9_f32_nocopy_nn_table[] = {
    {{8,  4 }, { 0,  0}, {256, 0}},
    {{16, 8 }, { 0,  0}, {0,   0}},
    {{16, 16}, { 0,  0}, {0,   0}},
    {{32, 16}, {-1, -1}, {0,   0}},
};

const kernel_table_t gen9_f32_nocopy_nt_table[] = {
    {{8,  8 }, { 0,  0}, {512, 0}},
    {{16, 16}, { 0,  0}, {0,   0}},
    {{32, 16}, {-1, -1}, {0,   0}}
};

const kernel_table_t gen9_f32_nocopy_tn_table[] = {
    {{8,  4 }, {16, 32}, {0, 0}},
    {{8,  8 }, { 0,  0}, {0, 0}},
    {{16, 8 }, { 0,  0}, {0, 0}},
    {{16, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_f32_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen9_f32_nocopy_tables[2][2] = {
    {gen9_f32_nocopy_nn_table, gen9_f32_nocopy_nt_table},
    {gen9_f32_nocopy_tn_table, gen9_f32_nocopy_tt_table}
};

const kernel_table_t gen9_f16_nocopy_nn_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_f16_nocopy_nt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_f16_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_f16_nocopy_tt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen9_f16_nocopy_tables[2][2] = {
    {gen9_f16_nocopy_nn_table, gen9_f16_nocopy_nt_table},
    {gen9_f16_nocopy_tn_table, gen9_f16_nocopy_tt_table}
};

const kernel_table_t gen9_x8_nocopy_nn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_x8_nocopy_nt_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_x8_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen9_x8_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen9_x8_nocopy_tables[2][2] = {
    {gen9_x8_nocopy_nn_table, gen9_x8_nocopy_nt_table},
    {gen9_x8_nocopy_tn_table, gen9_x8_nocopy_tt_table}
};

const kernel_table_t gen12lp_f32_nocopy_nn_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}},
    {{8,  8 }, { 0,  0}, {0, 0}},
    {{16, 8 }, { 0,  0}, {0, 0}},
    {{32, 8 }, { 0,  0}, {0, 0}},
    {{32, 12}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f32_nocopy_nt_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}},
    {{8,  8 }, { 0,  0}, {0, 0}},
    {{16, 16}, { 0,  0}, {0, 0}},
    {{32, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f32_nocopy_tn_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}},
    {{16, 8 }, { 0,  0}, {0, 0}},
    {{16, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f32_nocopy_tt_table[] = {
    {{12, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen12lp_f32_nocopy_tables[2][2] = {
    {gen12lp_f32_nocopy_nn_table, gen12lp_f32_nocopy_nt_table},
    {gen12lp_f32_nocopy_tn_table, gen12lp_f32_nocopy_tt_table}
};

const kernel_table_t gen12lp_f16_nocopy_nn_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f16_nocopy_nt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f16_nocopy_tn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_f16_nocopy_tt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen12lp_f16_nocopy_tables[2][2] = {
    {gen12lp_f16_nocopy_nn_table, gen12lp_f16_nocopy_nt_table},
    {gen12lp_f16_nocopy_tn_table, gen12lp_f16_nocopy_tt_table}
};

const kernel_table_t gen12lp_x8_nocopy_nn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_x8_nocopy_nt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_x8_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}}
};

const kernel_table_t gen12lp_x8_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}}
};

const kernel_table_t *gen12lp_x8_nocopy_tables[2][2] = {
    {gen12lp_x8_nocopy_nn_table, gen12lp_x8_nocopy_nt_table},
    {gen12lp_x8_nocopy_tn_table, gen12lp_x8_nocopy_tt_table}
};
// clang-format on

} // anonymous namespace

void gen_gemm_nocopy_kernel_t::choose_unrolls(compute::gpu_arch_t arch,
        int hw_threads, bool trans_a, bool trans_b, data_type_t a_type,
        data_type_t b_type, data_type_t c_type, dim_t m, dim_t n, dim_t k,
        dim_t batch, int &unroll_m, int &unroll_n) {

    unroll_m = unroll_n = 1;

    using tables_t = decltype(gen9_f32_nocopy_tables);
    const tables_t *all_tables[3][2]
            = {{&gen9_f32_nocopy_tables, &gen12lp_f32_nocopy_tables},
                    {&gen9_f16_nocopy_tables, &gen12lp_f16_nocopy_tables},
                    {&gen9_x8_nocopy_tables, &gen12lp_x8_nocopy_tables}};

    int arch_idx = (arch == compute::gpu_arch_t::gen12lp) ? 1 : 0;
    int type_idx = (c_type == data_type::f16)
            ? 1
            : (c_type == data_type::s32) ? 2 : 0;

    const kernel_table_t *table
            = (*all_tables[type_idx][arch_idx])[trans_a][trans_b];
    if (!table) {
        assert(!"Unsupported type for hardware.");
        return;
    }

    // Loop through kernel set, from smallest to largest unrolls.
    for (; table->max_accept[0] != -1; table++) {
        // If m/n under "always use" threshold, use this kernel.
        // If m/n over "reject" threshold, don't use this kernel.
        if (m <= table->max_accept[0] || n <= table->max_accept[1]) break;
        if (table->min_reject[0] > 0 && m > table->min_reject[0]) continue;
        if (table->min_reject[1] > 0 && n > table->min_reject[1]) continue;

        // Otherwise, check if more HW threads would be spawned than are
        // available on the GPU. If so, enlarge unrolls.
        auto trial_unroll_m = table->unrolls[0];
        auto trial_unroll_n = table->unrolls[1];
        auto mnb_threads = utils::div_up(m, trial_unroll_m)
                * utils::div_up(n, trial_unroll_n) * batch;
        if (mnb_threads <= hw_threads) break;
    }

    unroll_m = table->unrolls[0];
    unroll_n = table->unrolls[1];
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
