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

/*
 * When compiling nGEN in C++11 or C++14 mode; \
PREFIX constexpr ngen::GRF this header file should be
 *  #include'd exactly once in your source code.
 */

#if (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS)

#define NGEN_REGISTER_DECL_MAIN(CG, PREFIX) \
PREFIX constexpr ngen::IndirectRegisterFrame CG::indirect; \
\
PREFIX constexpr ngen::GRF CG::r0; \
PREFIX constexpr ngen::GRF CG::r1; \
PREFIX constexpr ngen::GRF CG::r2; \
PREFIX constexpr ngen::GRF CG::r3; \
PREFIX constexpr ngen::GRF CG::r4; \
PREFIX constexpr ngen::GRF CG::r5; \
PREFIX constexpr ngen::GRF CG::r6; \
PREFIX constexpr ngen::GRF CG::r7; \
PREFIX constexpr ngen::GRF CG::r8; \
PREFIX constexpr ngen::GRF CG::r9; \
PREFIX constexpr ngen::GRF CG::r10; \
PREFIX constexpr ngen::GRF CG::r11; \
PREFIX constexpr ngen::GRF CG::r12; \
PREFIX constexpr ngen::GRF CG::r13; \
PREFIX constexpr ngen::GRF CG::r14; \
PREFIX constexpr ngen::GRF CG::r15; \
PREFIX constexpr ngen::GRF CG::r16; \
PREFIX constexpr ngen::GRF CG::r17; \
PREFIX constexpr ngen::GRF CG::r18; \
PREFIX constexpr ngen::GRF CG::r19; \
PREFIX constexpr ngen::GRF CG::r20; \
PREFIX constexpr ngen::GRF CG::r21; \
PREFIX constexpr ngen::GRF CG::r22; \
PREFIX constexpr ngen::GRF CG::r23; \
PREFIX constexpr ngen::GRF CG::r24; \
PREFIX constexpr ngen::GRF CG::r25; \
PREFIX constexpr ngen::GRF CG::r26; \
PREFIX constexpr ngen::GRF CG::r27; \
PREFIX constexpr ngen::GRF CG::r28; \
PREFIX constexpr ngen::GRF CG::r29; \
PREFIX constexpr ngen::GRF CG::r30; \
PREFIX constexpr ngen::GRF CG::r31; \
PREFIX constexpr ngen::GRF CG::r32; \
PREFIX constexpr ngen::GRF CG::r33; \
PREFIX constexpr ngen::GRF CG::r34; \
PREFIX constexpr ngen::GRF CG::r35; \
PREFIX constexpr ngen::GRF CG::r36; \
PREFIX constexpr ngen::GRF CG::r37; \
PREFIX constexpr ngen::GRF CG::r38; \
PREFIX constexpr ngen::GRF CG::r39; \
PREFIX constexpr ngen::GRF CG::r40; \
PREFIX constexpr ngen::GRF CG::r41; \
PREFIX constexpr ngen::GRF CG::r42; \
PREFIX constexpr ngen::GRF CG::r43; \
PREFIX constexpr ngen::GRF CG::r44; \
PREFIX constexpr ngen::GRF CG::r45; \
PREFIX constexpr ngen::GRF CG::r46; \
PREFIX constexpr ngen::GRF CG::r47; \
PREFIX constexpr ngen::GRF CG::r48; \
PREFIX constexpr ngen::GRF CG::r49; \
PREFIX constexpr ngen::GRF CG::r50; \
PREFIX constexpr ngen::GRF CG::r51; \
PREFIX constexpr ngen::GRF CG::r52; \
PREFIX constexpr ngen::GRF CG::r53; \
PREFIX constexpr ngen::GRF CG::r54; \
PREFIX constexpr ngen::GRF CG::r55; \
PREFIX constexpr ngen::GRF CG::r56; \
PREFIX constexpr ngen::GRF CG::r57; \
PREFIX constexpr ngen::GRF CG::r58; \
PREFIX constexpr ngen::GRF CG::r59; \
PREFIX constexpr ngen::GRF CG::r60; \
PREFIX constexpr ngen::GRF CG::r61; \
PREFIX constexpr ngen::GRF CG::r62; \
PREFIX constexpr ngen::GRF CG::r63; \
PREFIX constexpr ngen::GRF CG::r64; \
PREFIX constexpr ngen::GRF CG::r65; \
PREFIX constexpr ngen::GRF CG::r66; \
PREFIX constexpr ngen::GRF CG::r67; \
PREFIX constexpr ngen::GRF CG::r68; \
PREFIX constexpr ngen::GRF CG::r69; \
PREFIX constexpr ngen::GRF CG::r70; \
PREFIX constexpr ngen::GRF CG::r71; \
PREFIX constexpr ngen::GRF CG::r72; \
PREFIX constexpr ngen::GRF CG::r73; \
PREFIX constexpr ngen::GRF CG::r74; \
PREFIX constexpr ngen::GRF CG::r75; \
PREFIX constexpr ngen::GRF CG::r76; \
PREFIX constexpr ngen::GRF CG::r77; \
PREFIX constexpr ngen::GRF CG::r78; \
PREFIX constexpr ngen::GRF CG::r79; \
PREFIX constexpr ngen::GRF CG::r80; \
PREFIX constexpr ngen::GRF CG::r81; \
PREFIX constexpr ngen::GRF CG::r82; \
PREFIX constexpr ngen::GRF CG::r83; \
PREFIX constexpr ngen::GRF CG::r84; \
PREFIX constexpr ngen::GRF CG::r85; \
PREFIX constexpr ngen::GRF CG::r86; \
PREFIX constexpr ngen::GRF CG::r87; \
PREFIX constexpr ngen::GRF CG::r88; \
PREFIX constexpr ngen::GRF CG::r89; \
PREFIX constexpr ngen::GRF CG::r90; \
PREFIX constexpr ngen::GRF CG::r91; \
PREFIX constexpr ngen::GRF CG::r92; \
PREFIX constexpr ngen::GRF CG::r93; \
PREFIX constexpr ngen::GRF CG::r94; \
PREFIX constexpr ngen::GRF CG::r95; \
PREFIX constexpr ngen::GRF CG::r96; \
PREFIX constexpr ngen::GRF CG::r97; \
PREFIX constexpr ngen::GRF CG::r98; \
PREFIX constexpr ngen::GRF CG::r99; \
PREFIX constexpr ngen::GRF CG::r100; \
PREFIX constexpr ngen::GRF CG::r101; \
PREFIX constexpr ngen::GRF CG::r102; \
PREFIX constexpr ngen::GRF CG::r103; \
PREFIX constexpr ngen::GRF CG::r104; \
PREFIX constexpr ngen::GRF CG::r105; \
PREFIX constexpr ngen::GRF CG::r106; \
PREFIX constexpr ngen::GRF CG::r107; \
PREFIX constexpr ngen::GRF CG::r108; \
PREFIX constexpr ngen::GRF CG::r109; \
PREFIX constexpr ngen::GRF CG::r110; \
PREFIX constexpr ngen::GRF CG::r111; \
PREFIX constexpr ngen::GRF CG::r112; \
PREFIX constexpr ngen::GRF CG::r113; \
PREFIX constexpr ngen::GRF CG::r114; \
PREFIX constexpr ngen::GRF CG::r115; \
PREFIX constexpr ngen::GRF CG::r116; \
PREFIX constexpr ngen::GRF CG::r117; \
PREFIX constexpr ngen::GRF CG::r118; \
PREFIX constexpr ngen::GRF CG::r119; \
PREFIX constexpr ngen::GRF CG::r120; \
PREFIX constexpr ngen::GRF CG::r121; \
PREFIX constexpr ngen::GRF CG::r122; \
PREFIX constexpr ngen::GRF CG::r123; \
PREFIX constexpr ngen::GRF CG::r124; \
PREFIX constexpr ngen::GRF CG::r125; \
PREFIX constexpr ngen::GRF CG::r126; \
PREFIX constexpr ngen::GRF CG::r127; \
\
PREFIX constexpr ngen::NullRegister CG::null; \
PREFIX constexpr ngen::AddressRegister CG::a0; \
PREFIX constexpr ngen::AccumulatorRegister CG::acc0; \
PREFIX constexpr ngen::AccumulatorRegister CG::acc1; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc2; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc3; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc4; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc5; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc6; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc7; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc8; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc9; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme0; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme1; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme2; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme3; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme4; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme5; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme6; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme7; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::nomme; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::noacc; \
PREFIX constexpr ngen::FlagRegister CG::f0; \
PREFIX constexpr ngen::FlagRegister CG::f1; \
PREFIX constexpr ngen::FlagRegister CG::f0_0; \
PREFIX constexpr ngen::FlagRegister CG::f0_1; \
PREFIX constexpr ngen::FlagRegister CG::f1_0; \
PREFIX constexpr ngen::FlagRegister CG::f1_1; \
PREFIX constexpr ngen::ChannelEnableRegister CG::ce0; \
PREFIX constexpr ngen::StackPointerRegister CG::sp; \
PREFIX constexpr ngen::StateRegister CG::sr0; \
PREFIX constexpr ngen::StateRegister CG::sr1; \
PREFIX constexpr ngen::ControlRegister CG::cr0; \
PREFIX constexpr ngen::NotificationRegister CG::n0; \
PREFIX constexpr ngen::InstructionPointerRegister CG::ip; \
PREFIX constexpr ngen::ThreadDependencyRegister CG::tdr0; \
PREFIX constexpr ngen::PerformanceRegister CG::tm0; \
PREFIX constexpr ngen::PerformanceRegister CG::pm0; \
PREFIX constexpr ngen::PerformanceRegister CG::tp0; \
PREFIX constexpr ngen::DebugRegister CG::dbg0; \
\
PREFIX constexpr ngen::InstructionModifier CG::NoDDClr; \
PREFIX constexpr ngen::InstructionModifier CG::NoDDChk; \
PREFIX constexpr ngen::InstructionModifier CG::AccWrEn; \
PREFIX constexpr ngen::InstructionModifier CG::NoSrcDepSet; \
PREFIX constexpr ngen::InstructionModifier CG::Breakpoint; \
PREFIX constexpr ngen::InstructionModifier CG::sat; \
PREFIX constexpr ngen::InstructionModifier CG::NoMask; \
PREFIX constexpr ngen::InstructionModifier CG::AutoSWSB; \
PREFIX constexpr ngen::InstructionModifier CG::Serialize; \
PREFIX constexpr ngen::InstructionModifier CG::EOT; \
PREFIX constexpr ngen::InstructionModifier CG::Align1; \
PREFIX constexpr ngen::InstructionModifier CG::Align16; \
PREFIX constexpr ngen::InstructionModifier CG::Atomic; \
PREFIX constexpr ngen::InstructionModifier CG::Switch; \
PREFIX constexpr ngen::InstructionModifier CG::NoPreempt; \
\
PREFIX constexpr ngen::PredCtrl CG::anyv; \
PREFIX constexpr ngen::PredCtrl CG::allv; \
PREFIX constexpr ngen::PredCtrl CG::any2h; \
PREFIX constexpr ngen::PredCtrl CG::all2h; \
PREFIX constexpr ngen::PredCtrl CG::any4h; \
PREFIX constexpr ngen::PredCtrl CG::all4h; \
PREFIX constexpr ngen::PredCtrl CG::any8h; \
PREFIX constexpr ngen::PredCtrl CG::all8h; \
PREFIX constexpr ngen::PredCtrl CG::any16h; \
PREFIX constexpr ngen::PredCtrl CG::all16h; \
PREFIX constexpr ngen::PredCtrl CG::any32h; \
PREFIX constexpr ngen::PredCtrl CG::all32h; \
\
PREFIX constexpr ngen::InstructionModifier CG::x_repl; \
PREFIX constexpr ngen::InstructionModifier CG::y_repl; \
PREFIX constexpr ngen::InstructionModifier CG::z_repl; \
PREFIX constexpr ngen::InstructionModifier CG::w_repl; \
\
PREFIX constexpr ngen::InstructionModifier CG::ze; \
PREFIX constexpr ngen::InstructionModifier CG::eq; \
PREFIX constexpr ngen::InstructionModifier CG::nz; \
PREFIX constexpr ngen::InstructionModifier CG::ne; \
PREFIX constexpr ngen::InstructionModifier CG::gt; \
PREFIX constexpr ngen::InstructionModifier CG::ge; \
PREFIX constexpr ngen::InstructionModifier CG::lt; \
PREFIX constexpr ngen::InstructionModifier CG::le; \
PREFIX constexpr ngen::InstructionModifier CG::ov; \
PREFIX constexpr ngen::InstructionModifier CG::un; \
PREFIX constexpr ngen::InstructionModifier CG::eo; \
\
PREFIX constexpr ngen::InstructionModifier CG::M0; \
PREFIX constexpr ngen::InstructionModifier CG::M4; \
PREFIX constexpr ngen::InstructionModifier CG::M8; \
PREFIX constexpr ngen::InstructionModifier CG::M12; \
PREFIX constexpr ngen::InstructionModifier CG::M16; \
PREFIX constexpr ngen::InstructionModifier CG::M20; \
PREFIX constexpr ngen::InstructionModifier CG::M24; \
PREFIX constexpr ngen::InstructionModifier CG::M28; \
\
PREFIX constexpr ngen::SBID CG::sb0; \
PREFIX constexpr ngen::SBID CG::sb1; \
PREFIX constexpr ngen::SBID CG::sb2; \
PREFIX constexpr ngen::SBID CG::sb3; \
PREFIX constexpr ngen::SBID CG::sb4; \
PREFIX constexpr ngen::SBID CG::sb5; \
PREFIX constexpr ngen::SBID CG::sb6; \
PREFIX constexpr ngen::SBID CG::sb7; \
PREFIX constexpr ngen::SBID CG::sb8; \
PREFIX constexpr ngen::SBID CG::sb9; \
PREFIX constexpr ngen::SBID CG::sb10; \
PREFIX constexpr ngen::SBID CG::sb11; \
PREFIX constexpr ngen::SBID CG::sb12; \
PREFIX constexpr ngen::SBID CG::sb13; \
PREFIX constexpr ngen::SBID CG::sb14; \
PREFIX constexpr ngen::SBID CG::sb15; \
\
PREFIX constexpr ngen::AddressBase CG::A32; \
PREFIX constexpr ngen::AddressBase CG::A32NC; \
PREFIX constexpr ngen::AddressBase CG::A64; \
PREFIX constexpr ngen::AddressBase CG::A64NC; \
PREFIX constexpr ngen::AddressBase CG::SLM; \

#define NGEN_REGISTER_DECL_EXTRA(CG,PREFIX)

#ifdef NGEN_SHORT_NAMES
#define NGEN_REGISTER_DECL(CG,PREFIX) \
NGEN_REGISTER_DECL_MAIN(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA(CG,PREFIX) \
PREFIX constexpr const ngen::IndirectRegisterFrame &CG::r; \
PREFIX constexpr const ngen::InstructionModifier &CG::W;
#else
#define NGEN_REGISTER_DECL(CG,PREFIX) \
NGEN_REGISTER_DECL_MAIN(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA(CG,PREFIX)
#endif

#include "ngen.hpp"
NGEN_REGISTER_DECL(ngen::BinaryCodeGenerator<hw>, template <ngen::HW hw>)

#ifdef NGEN_ASM
#include "ngen_asm.hpp"
NGEN_REGISTER_DECL(ngen::AsmCodeGenerator, /* nothing */)
#endif

template class ngen::BinaryCodeGenerator<ngen::HW::Gen9>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen10>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen11>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen12LP>;

#endif /* (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS) */
