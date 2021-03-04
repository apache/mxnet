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

// nGEN: a C++ library for runtime Gen assembly generation.
//
// Macros that control nGEN's interface:
//    NGEN_SAFE             if defined, enables run-time safety checks. Exceptions will be thrown if checks fail.
//    NGEN_SHORT_NAMES      if defined, enables some short names (r[...] for indirect addressing, W for NoMask)
//    NGEN_GLOBAL_REGS      if defined, register names and instruction modifiers (r7, cr0, Switch, etc.) are
//                           global variables in the ngen namespace. Otherwise, they are members of the code
//                           generator classes
//    NGEN_CPP11            if defined, ngen is C++11-compatible (C++17 not required)

#ifndef NGEN_HPP
#define NGEN_HPP

#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ngen_core.hpp"
#include "ngen_auto_swsb.hpp"

namespace ngen {

// Forward declarations.
template <HW hw> class BinaryCodeGenerator;

// MSVC v140 workaround for enum comparison in template arguments.
static constexpr bool hwLT(HW hw1, HW hw2) { return hw1 < hw2; }
static constexpr bool hwLE(HW hw1, HW hw2) { return hw1 <= hw2; }
static constexpr bool hwGE(HW hw1, HW hw2) { return hw1 >= hw2; }
static constexpr bool hwGT(HW hw1, HW hw2) { return hw1 > hw2; }

// -----------------------------------------------------------------------

enum RegFiles : unsigned {
    RegFileARF = 0,
    RegFileGRF = 1,
    RegFileIMM = 3,
};

inline unsigned getRegFile(const RegData &rd)          { return rd.isARF() ? RegFileARF : RegFileGRF; }
inline unsigned getRegFile(const Align16Operand &o)    { return getRegFile(o.getReg()); }
inline unsigned getRegFile(const ExtendedReg &reg)     { return getRegFile(reg.getBase()); }
inline unsigned getRegFile(const Immediate &imm)       { return RegFileIMM; }

// -----------------------------------------------------------------------
// Binary formats, split between pre-Gen12 and post-Gen12.

#include "ngen_gen8.hpp"
#include "ngen_gen12.hpp"

// -----------------------------------------------------------------------


class LabelFixup {
public:
    uint32_t labelID;
    int32_t anchor;
    int32_t offset;

    LabelFixup(uint32_t labelID_, int32_t offset_) : labelID(labelID_), anchor(0), offset(offset_) {}

    static constexpr auto JIPOffset = 12;
    static constexpr auto JIPOffsetJMPI = -4;
    static constexpr auto UIPOffset = 8;
};

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#define NGEN_GLOBAL_REGS_DEFINED
#include "ngen_registers.hpp"
#endif

template <HW hw>
class BinaryCodeGenerator
{
protected:
    class InstructionStream {
        friend class BinaryCodeGenerator;

        std::vector<LabelFixup> fixups;
        std::vector<uint32_t> labels;
        std::vector<uint64_t> code;
        bool appended = false;

        int length() const { return int(code.size() * sizeof(uint64_t)); }

        void db(const Instruction8 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void db(const Instruction12 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void addFixup(LabelFixup fixup) {
            fixup.anchor = length();
            fixups.push_back(fixup);
        }

        void mark(Label &label, LabelManager &man) {
            uint32_t id = label.getID(man);

            man.setTarget(id, length());
            labels.push_back(id);
        }

        void fixLabels(LabelManager &man) {
            for (const auto &fixup : fixups) {
                int32_t target = man.getTarget(fixup.labelID);
                uint8_t *field = ((uint8_t *) code.data()) + fixup.anchor + fixup.offset;
                *((int32_t *) field) = target - fixup.anchor;
            }
        }

        void append(InstructionStream &other, LabelManager &man) {
            auto offset = length();
            auto sz = code.size();

            code.resize(sz + other.code.size());
            std::copy(other.code.begin(), other.code.end(), code.begin() + sz);

            sz = labels.size();
            labels.resize(sz + other.labels.size());
            std::copy(other.labels.begin(), other.labels.end(), labels.begin() + sz);

            for (LabelFixup fixup : other.fixups) {
                fixup.anchor += offset;
                fixups.push_back(fixup);
            }

#ifdef NGEN_SAFE
            if (other.appended && !other.labels.empty())
                throw multiple_label_exception();
#endif

            for (uint32_t id : other.labels)
                man.offsetTarget(id, offset);

            other.appended = true;
        }

        InstructionStream() {}
    };

    class Program {
        friend class BinaryCodeGenerator;

        std::vector<uint64_t> &code;

        Program(InstructionStream &stream) : code(stream.code) {};

    public:
        size_t size() const                                 { return code.size() >> 1; }
        Instruction12 &operator[](size_t index)             { return *reinterpret_cast<Instruction12 *>(&code[index * 2]); }
        const Instruction12 &operator[](size_t index) const { return *reinterpret_cast<Instruction12 *>(&code[index * 2]); }
    };

    static constexpr HW hardware = hw;
    static constexpr bool isGen12 = (hw >= HW::Gen12LP);

private:
    InstructionModifier defaultModifier;

    LabelManager labelManager;
    InstructionStream rootStream;
    std::vector<InstructionStream*> streamStack;

    void db(const Instruction8 &i)  { streamStack.back()->db(i); }
    void db(const Instruction12 &i) { streamStack.back()->db(i); }
    void addFixup(LabelFixup fixup) { streamStack.back()->addFixup(fixup); }

    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0);

    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1);
    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1);

    template <HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2);

    template <typename DS0>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0);
    template <typename DS0, typename S1>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1);

    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, D desc);
    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc);

    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0);

    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip);
    template <bool forceWE = false>
    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip);
    void opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip);
    void opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip);

    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const RegData &src0);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0);

    void opNop(Opcode op);

    inline void unsupported();

#include "ngen_compiler_fix.hpp"

public:
    BinaryCodeGenerator() : defaultModifier{}, labelManager{}, sync{this} {
        _workaround_();
        pushStream(rootStream);
    }

    ~BinaryCodeGenerator() {
        for (size_t sn = 1; sn < streamStack.size(); sn++)
            delete streamStack[sn];
    }

    std::vector<uint8_t> getCode();

protected:
    // Configuration.
    void setDefaultAutoSWSB(bool def = true)        { defaultModifier.setAutoSWSB(def); }
    void setDefaultNoMask(bool def = true)          { defaultModifier.setWrEn(def); }

    // Stream handling.
    void pushStream()                               { pushStream(new InstructionStream()); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); }
    void pushStream(InstructionStream &s)           { pushStream(&s); }

    InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s, labelManager); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    template <typename String>
    void comment(String)                            {}

    // Registers.
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif

    // Labels.
    inline void mark(Label &label)          { streamStack.back()->mark(label, labelManager); }

    // Instructions.
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), jip, uip);
    }
    void brc(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), src0);
    }
    void brd(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), jip);
    }
    void brd(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), src0);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::break_, mod, null, jip, uip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip) {
        opCall(Opcode::call, mod, dst, jip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, RegData jip) {
        if (isGen12)
            opBranch<true, true>(Opcode::call, mod, dst, jip);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::call, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip);
        }
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip) {
        if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip));
    }
    void calla(const InstructionModifier &mod, const RegData &dst, RegData jip) {
        if (isGen12)
            opBranch<true, true>(Opcode::calla, mod, dst, jip);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::calla, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip);
        }
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::cont, mod, null, jip, uip);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::else_, mod, null, jip, uip);
    }
    void else_(InstructionModifier mod, Label &jip) {
        else_(mod, jip, jip);
    }
    void endif(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::endif, mod, null, jip);
    }
    void endif(const InstructionModifier &mod) {
        opBranch(Opcode::endif, mod, null, sizeof(Instruction8));
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0);
    }
    void goto_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::goto_, mod, null, jip, uip);
    }
    void goto_(const InstructionModifier &mod, Label &jip) {
        goto_(mod, jip, jip);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::halt, mod, null, jip, uip);
    }
    void halt(const InstructionModifier &mod, Label &jip) {
        halt(mod, jip, jip);
    }
    void if_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::if_, mod, null, jip, uip);
    }
    void if_(const InstructionModifier &mod, Label &jip) {
        if_(mod, jip, jip);
    }
    void illegal() {
        opX(Opcode::illegal, DataType::invalid, InstructionModifier(), null, null, null);
    }
    void join(InstructionModifier mod, Label &jip) {
        opBranch(Opcode::join, mod, null, jip);
    }
    void join(InstructionModifier mod) {
        opBranch(Opcode::join, mod, null, sizeof(Instruction8));
    }
    void jmpi(const InstructionModifier &mod, Label &jip) {
        auto dst = isGen12 ? ARF(null) : ARF(ip);
        opJmpi(Opcode::jmpi, mod, dst, dst, jip);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip) {
#ifdef NGEN_SAFE
        if (!isGen12 && jip.getType() != DataType::d && jip.getType() != DataType::invalid)
            throw invalid_type_exception();
#endif
        if (isGen12)
            opBranch<true, false>(Opcode::jmpi, mod, null, jip);
        else
            opX(Opcode::jmpi, DataType::d, mod, ip, ip, jip);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mach, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::mach, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2) {
        opX(Opcode::madm, getDataType<DT>(), mod, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), extToAlign16(src2));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGT(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, const ExtendedReg &src2) {
        src0.getBase().setRegion(4,4,1);
        src1.getBase().setRegion(4,4,1);
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 1) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 2) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1.forceInt32());
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11)
            src0.getBase().setRegion(2,2,1);
        else
            src0.getBase().setRegion(1,1,0);
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11) {
            src0.getBase().setRegion(2,2,1);
            src1.getBase().setRegion(2,2,1);
        } else {
            src0.getBase().setRegion(1,1,0);
            src1.getBase().setRegion(1,1,0);
        }
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    void nop() {
        opNop(isGen12 ? Opcode::nop_gen12 : Opcode::nop);
    }
    void nop(const InstructionModifier &mod) {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop, DataType::invalid, mod, null, null, null);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        not_<DT>(mod, dst, src0);
    }
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        not_<DT>(mod, dst, src0);
    }
#endif
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1);
    }
    void ret(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2,2,1);
        if (isGen12)
            opBranch<true, true>(Opcode::ret, mod, null, src0);
        else
            opX<true>(Opcode::ret, DataType::ud, mod, null, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }

    /* Gen12-style sends */
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    /* Pre-Gen12-style sends; also supported on Gen12. */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc);
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }

    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    void wait(const InstructionModifier &mod, const RegData &nreg) {
#ifdef NGEN_SAFE
        if (!nreg.isARF() || nreg.getARFType() != ARFType::n) throw invalid_arf_exception();
#endif
        opX(Opcode::wait, DataType::invalid, mod, nreg, nreg);
    }
    void while_(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::while_, mod, null, jip);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
#endif

private:
    struct Sync {
        BinaryCodeGenerator<hw> &parent;

        Sync(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier()) {
            parent.opSync(Opcode::sync, fc, mod);
        }
        void operator()(SyncFunction fc, const RegData &src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, const RegData &src0) {
            parent.opSync(Opcode::sync, fc, mod, src0);
        }
        void operator()(SyncFunction fc, int src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, uint32_t src0) {
            parent.opSync(Opcode::sync, fc, mod, Immediate::ud(src0));
        }
        void allrd(const RegData &src0) {
            allrd(InstructionModifier(), src0);
        }
        void allrd(const InstructionModifier &mod, const RegData &src0) {
            this->operator()(SyncFunction::allrd, mod, src0);
        }
        void allrd(uint32_t src0) {
            allrd(InstructionModifier(), src0);
        }
        void allrd(const InstructionModifier &mod, uint32_t src0) {
            this->operator()(SyncFunction::allrd, mod, src0);
        }
        void allwr(const RegData &src0) {
            allwr(InstructionModifier(), src0);
        }
        void allwr(const InstructionModifier &mod, const RegData &src0) {
            this->operator()(SyncFunction::allwr, mod, src0);
        }
        void allwr(uint32_t src0) {
            allwr(InstructionModifier(), src0);
        }
        void allwr(const InstructionModifier &mod, uint32_t src0) {
            this->operator()(SyncFunction::allwr, mod, src0);
        }
        void bar(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::bar, mod);
        }
        void host(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::host, mod);
        }
        void nop(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::nop, mod);
        }
    };
public:
    Sync sync;

#include "ngen_pseudo.hpp"
};

#define NGEN_FORWARD(hw) \
using InstructionStream = typename ngen::BinaryCodeGenerator<hw>::InstructionStream; \
using ngen::BinaryCodeGenerator<hw>::isGen12; \
template <typename DT = void, typename... Targs> void add(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template add<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void addc(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template addc<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void and_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template and_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void asr(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template asr<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void avg(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template avg<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfe(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template bfe<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi1(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template bfi1<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi2(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template bfi2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfrev(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template bfrev<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cbit(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template cbit<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cmp(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template cmp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cmpn(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template cmpn<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void csel(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template csel<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp2(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template dp2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp3(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template dp3<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp4(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template dp4<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dph(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template dph<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fbh(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template fbh<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fbl(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template fbl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void frc(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template frc<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void line(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template line<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void lrp(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template lrp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void lzd(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template lzd<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mac(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template mac<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void macl(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template macl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mach(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template mach<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mad(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template mad<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void madm(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template madm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void math(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template math<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mov(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template mov<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void movi(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template movi<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mul(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template mul<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void not_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template not_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void or_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template or_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void pln(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template pln<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndd(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rndd<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rnde(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rnde<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndu(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rndu<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndz(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rndz<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rol(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rol<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void ror(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template ror<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sad2(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sad2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sada2(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sada2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sel(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sel<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void shl(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template shl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void shr(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template shr<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void smov(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template smov<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void subb(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template subb<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void xor_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template xor_<DT>(std::forward<Targs>(args)...); } \
template <typename... Targs> void brc(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::brc(std::forward<Targs>(args)...); } \
template <typename... Targs> void brd(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::brd(std::forward<Targs>(args)...); } \
template <typename... Targs> void break_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::break_(std::forward<Targs>(args)...); } \
template <typename... Targs> void call(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::call(std::forward<Targs>(args)...); } \
template <typename... Targs> void calla(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::calla(std::forward<Targs>(args)...); } \
template <typename... Targs> void cont(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::cont(std::forward<Targs>(args)...); } \
template <typename... Targs> void else_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::else_(std::forward<Targs>(args)...); } \
template <typename... Targs> void endif(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::endif(std::forward<Targs>(args)...); } \
template <typename... Targs> void goto_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::goto_(std::forward<Targs>(args)...); } \
template <typename... Targs> void halt(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::halt(std::forward<Targs>(args)...); } \
template <typename... Targs> void if_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::if_(std::forward<Targs>(args)...); } \
template <typename... Targs> void illegal(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::illegal(std::forward<Targs>(args)...); } \
template <typename... Targs> void join(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::join(std::forward<Targs>(args)...); } \
template <typename... Targs> void jmpi(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::jmpi(std::forward<Targs>(args)...); } \
template <typename... Targs> void nop(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::nop(std::forward<Targs>(args)...); } \
template <typename... Targs> void ret(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::ret(std::forward<Targs>(args)...); } \
template <typename... Targs> void send(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::send(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendc(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::sendc(std::forward<Targs>(args)...); } \
template <typename... Targs> void sends(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::sends(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendsc(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::sendsc(std::forward<Targs>(args)...); } \
using ngen::BinaryCodeGenerator<hw>::sync; \
template <typename... Targs> void wait(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::wait(std::forward<Targs>(args)...); } \
template <typename... Targs> void while_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::while_(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void min_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template min_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void max_(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template max_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template bfi<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cos(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template cos<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void exp(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template exp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fdiv(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template fdiv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void idiv(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template idiv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void inv(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template inv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void invm(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template invm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void iqot(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template iqot<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void irem(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template irem<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void log(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template log<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void pow(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template pow<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rsqt(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rsqt<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rsqtm(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template rsqtm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sin(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sin<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sqt(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sqt<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fdiv_ieee(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template fdiv_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void inv_ieee(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template inv_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sqt_ieee(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template sqt_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename... Targs> void threadend(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::threadend(std::forward<Targs>(args)...); } \
template <typename... Targs> void barriermsg(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::barriermsg(std::forward<Targs>(args)...); } \
template <typename... Targs> void barriersignal(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::barriersignal(std::forward<Targs>(args)...); } \
template <typename... Targs> void barrierwait(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::barrierwait(std::forward<Targs>(args)...); } \
template <typename... Targs> void barrier(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::barrier(std::forward<Targs>(args)...); } \
template <typename... Targs> void load(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::load(std::forward<Targs>(args)...); } \
template <typename... Targs> void store(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::store(std::forward<Targs>(args)...); } \
template <typename... Targs> void atomic(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::atomic(std::forward<Targs>(args)...); } \
template <typename... Targs> void memfence(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::memfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void slmfence(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::slmfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void pushStream(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::pushStream(std::forward<Targs>(args)...); } \
template <typename... Targs> InstructionStream *popStream(Targs&&... args) { return ngen::BinaryCodeGenerator<hw>::popStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendStream(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::appendStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendCurrentStream(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::appendCurrentStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void discardStream(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::discardStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void mark(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::mark(std::forward<Targs>(args)...); } \
template <typename... Targs> void comment(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::comment(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultNoMask(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::setDefaultNoMask(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultAutoSWSB(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::setDefaultAutoSWSB(std::forward<Targs>(args)...); } \
NGEN_FORWARD_EXTRA \
NGEN_FORWARD_OP_NAMES \
NGEN_FORWARD_MIN_MAX \
NGEN_FORWARD_REGISTERS

#define NGEN_FORWARD_EXTRA \
template <typename DT = void, typename... Targs> void dp4a(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template dp4a<DT>(std::forward<Targs>(args)...); }

#ifdef NGEN_NO_OP_NAMES
#define NGEN_FORWARD_OP_NAMES
#else
#define NGEN_FORWARD_OP_NAMES \
template <typename DT = void, typename... Targs> void and(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template and_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void not(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template not_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void or(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template or_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void xor(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template xor_<DT>(std::forward<Targs>(args)...); }
#endif

#ifdef NGEN_WINDOWS_COMPAT
#define NGEN_FORWARD_MIN_MAX
#else
#define NGEN_FORWARD_MIN_MAX \
template <typename DT = void, typename... Targs> void min(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template min<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void max(Targs&&... args) { ngen::BinaryCodeGenerator<hw>::template max<DT>(std::forward<Targs>(args)...); }
#endif

#ifdef NGEN_GLOBAL_REGS
#define NGEN_FORWARD_REGISTERS
#else
#define NGEN_FORWARD_REGISTERS_BASE \
using ngen::BinaryCodeGenerator<hw>::indirect; \
using ngen::BinaryCodeGenerator<hw>::r0; using ngen::BinaryCodeGenerator<hw>::r1; using ngen::BinaryCodeGenerator<hw>::r2; using ngen::BinaryCodeGenerator<hw>::r3; \
using ngen::BinaryCodeGenerator<hw>::r4; using ngen::BinaryCodeGenerator<hw>::r5; using ngen::BinaryCodeGenerator<hw>::r6; using ngen::BinaryCodeGenerator<hw>::r7; \
using ngen::BinaryCodeGenerator<hw>::r8; using ngen::BinaryCodeGenerator<hw>::r9; using ngen::BinaryCodeGenerator<hw>::r10; using ngen::BinaryCodeGenerator<hw>::r11; \
using ngen::BinaryCodeGenerator<hw>::r12; using ngen::BinaryCodeGenerator<hw>::r13; using ngen::BinaryCodeGenerator<hw>::r14; using ngen::BinaryCodeGenerator<hw>::r15; \
using ngen::BinaryCodeGenerator<hw>::r16; using ngen::BinaryCodeGenerator<hw>::r17; using ngen::BinaryCodeGenerator<hw>::r18; using ngen::BinaryCodeGenerator<hw>::r19; \
using ngen::BinaryCodeGenerator<hw>::r20; using ngen::BinaryCodeGenerator<hw>::r21; using ngen::BinaryCodeGenerator<hw>::r22; using ngen::BinaryCodeGenerator<hw>::r23; \
using ngen::BinaryCodeGenerator<hw>::r24; using ngen::BinaryCodeGenerator<hw>::r25; using ngen::BinaryCodeGenerator<hw>::r26; using ngen::BinaryCodeGenerator<hw>::r27; \
using ngen::BinaryCodeGenerator<hw>::r28; using ngen::BinaryCodeGenerator<hw>::r29; using ngen::BinaryCodeGenerator<hw>::r30; using ngen::BinaryCodeGenerator<hw>::r31; \
using ngen::BinaryCodeGenerator<hw>::r32; using ngen::BinaryCodeGenerator<hw>::r33; using ngen::BinaryCodeGenerator<hw>::r34; using ngen::BinaryCodeGenerator<hw>::r35; \
using ngen::BinaryCodeGenerator<hw>::r36; using ngen::BinaryCodeGenerator<hw>::r37; using ngen::BinaryCodeGenerator<hw>::r38; using ngen::BinaryCodeGenerator<hw>::r39; \
using ngen::BinaryCodeGenerator<hw>::r40; using ngen::BinaryCodeGenerator<hw>::r41; using ngen::BinaryCodeGenerator<hw>::r42; using ngen::BinaryCodeGenerator<hw>::r43; \
using ngen::BinaryCodeGenerator<hw>::r44; using ngen::BinaryCodeGenerator<hw>::r45; using ngen::BinaryCodeGenerator<hw>::r46; using ngen::BinaryCodeGenerator<hw>::r47; \
using ngen::BinaryCodeGenerator<hw>::r48; using ngen::BinaryCodeGenerator<hw>::r49; using ngen::BinaryCodeGenerator<hw>::r50; using ngen::BinaryCodeGenerator<hw>::r51; \
using ngen::BinaryCodeGenerator<hw>::r52; using ngen::BinaryCodeGenerator<hw>::r53; using ngen::BinaryCodeGenerator<hw>::r54; using ngen::BinaryCodeGenerator<hw>::r55; \
using ngen::BinaryCodeGenerator<hw>::r56; using ngen::BinaryCodeGenerator<hw>::r57; using ngen::BinaryCodeGenerator<hw>::r58; using ngen::BinaryCodeGenerator<hw>::r59; \
using ngen::BinaryCodeGenerator<hw>::r60; using ngen::BinaryCodeGenerator<hw>::r61; using ngen::BinaryCodeGenerator<hw>::r62; using ngen::BinaryCodeGenerator<hw>::r63; \
using ngen::BinaryCodeGenerator<hw>::r64; using ngen::BinaryCodeGenerator<hw>::r65; using ngen::BinaryCodeGenerator<hw>::r66; using ngen::BinaryCodeGenerator<hw>::r67; \
using ngen::BinaryCodeGenerator<hw>::r68; using ngen::BinaryCodeGenerator<hw>::r69; using ngen::BinaryCodeGenerator<hw>::r70; using ngen::BinaryCodeGenerator<hw>::r71; \
using ngen::BinaryCodeGenerator<hw>::r72; using ngen::BinaryCodeGenerator<hw>::r73; using ngen::BinaryCodeGenerator<hw>::r74; using ngen::BinaryCodeGenerator<hw>::r75; \
using ngen::BinaryCodeGenerator<hw>::r76; using ngen::BinaryCodeGenerator<hw>::r77; using ngen::BinaryCodeGenerator<hw>::r78; using ngen::BinaryCodeGenerator<hw>::r79; \
using ngen::BinaryCodeGenerator<hw>::r80; using ngen::BinaryCodeGenerator<hw>::r81; using ngen::BinaryCodeGenerator<hw>::r82; using ngen::BinaryCodeGenerator<hw>::r83; \
using ngen::BinaryCodeGenerator<hw>::r84; using ngen::BinaryCodeGenerator<hw>::r85; using ngen::BinaryCodeGenerator<hw>::r86; using ngen::BinaryCodeGenerator<hw>::r87; \
using ngen::BinaryCodeGenerator<hw>::r88; using ngen::BinaryCodeGenerator<hw>::r89; using ngen::BinaryCodeGenerator<hw>::r90; using ngen::BinaryCodeGenerator<hw>::r91; \
using ngen::BinaryCodeGenerator<hw>::r92; using ngen::BinaryCodeGenerator<hw>::r93; using ngen::BinaryCodeGenerator<hw>::r94; using ngen::BinaryCodeGenerator<hw>::r95; \
using ngen::BinaryCodeGenerator<hw>::r96; using ngen::BinaryCodeGenerator<hw>::r97; using ngen::BinaryCodeGenerator<hw>::r98; using ngen::BinaryCodeGenerator<hw>::r99; \
using ngen::BinaryCodeGenerator<hw>::r100; using ngen::BinaryCodeGenerator<hw>::r101; using ngen::BinaryCodeGenerator<hw>::r102; using ngen::BinaryCodeGenerator<hw>::r103; \
using ngen::BinaryCodeGenerator<hw>::r104; using ngen::BinaryCodeGenerator<hw>::r105; using ngen::BinaryCodeGenerator<hw>::r106; using ngen::BinaryCodeGenerator<hw>::r107; \
using ngen::BinaryCodeGenerator<hw>::r108; using ngen::BinaryCodeGenerator<hw>::r109; using ngen::BinaryCodeGenerator<hw>::r110; using ngen::BinaryCodeGenerator<hw>::r111; \
using ngen::BinaryCodeGenerator<hw>::r112; using ngen::BinaryCodeGenerator<hw>::r113; using ngen::BinaryCodeGenerator<hw>::r114; using ngen::BinaryCodeGenerator<hw>::r115; \
using ngen::BinaryCodeGenerator<hw>::r116; using ngen::BinaryCodeGenerator<hw>::r117; using ngen::BinaryCodeGenerator<hw>::r118; using ngen::BinaryCodeGenerator<hw>::r119; \
using ngen::BinaryCodeGenerator<hw>::r120; using ngen::BinaryCodeGenerator<hw>::r121; using ngen::BinaryCodeGenerator<hw>::r122; using ngen::BinaryCodeGenerator<hw>::r123; \
using ngen::BinaryCodeGenerator<hw>::r124; using ngen::BinaryCodeGenerator<hw>::r125; using ngen::BinaryCodeGenerator<hw>::r126; using ngen::BinaryCodeGenerator<hw>::r127; \
using ngen::BinaryCodeGenerator<hw>::null; \
using ngen::BinaryCodeGenerator<hw>::a0; \
using ngen::BinaryCodeGenerator<hw>::acc0; using ngen::BinaryCodeGenerator<hw>::acc1; using ngen::BinaryCodeGenerator<hw>::acc2; using ngen::BinaryCodeGenerator<hw>::acc3; \
using ngen::BinaryCodeGenerator<hw>::acc4; using ngen::BinaryCodeGenerator<hw>::acc5; using ngen::BinaryCodeGenerator<hw>::acc6; using ngen::BinaryCodeGenerator<hw>::acc7; \
using ngen::BinaryCodeGenerator<hw>::acc8; using ngen::BinaryCodeGenerator<hw>::acc9; \
using ngen::BinaryCodeGenerator<hw>::mme0; using ngen::BinaryCodeGenerator<hw>::mme1; using ngen::BinaryCodeGenerator<hw>::mme2; using ngen::BinaryCodeGenerator<hw>::mme3; \
using ngen::BinaryCodeGenerator<hw>::mme4; using ngen::BinaryCodeGenerator<hw>::mme5; using ngen::BinaryCodeGenerator<hw>::mme6; using ngen::BinaryCodeGenerator<hw>::mme7; \
using ngen::BinaryCodeGenerator<hw>::noacc; using ngen::BinaryCodeGenerator<hw>::nomme; \
using ngen::BinaryCodeGenerator<hw>::f0; using ngen::BinaryCodeGenerator<hw>::f1; \
using ngen::BinaryCodeGenerator<hw>::ce0; using ngen::BinaryCodeGenerator<hw>::sp; using ngen::BinaryCodeGenerator<hw>::sr0; using ngen::BinaryCodeGenerator<hw>::sr1; \
using ngen::BinaryCodeGenerator<hw>::cr0; using ngen::BinaryCodeGenerator<hw>::n0; using ngen::BinaryCodeGenerator<hw>::ip; using ngen::BinaryCodeGenerator<hw>::tdr0; \
using ngen::BinaryCodeGenerator<hw>::tm0; using ngen::BinaryCodeGenerator<hw>::pm0; using ngen::BinaryCodeGenerator<hw>::tp0; using ngen::BinaryCodeGenerator<hw>::dbg0; \
using ngen::BinaryCodeGenerator<hw>::NoDDClr; using ngen::BinaryCodeGenerator<hw>::NoDDChk; \
using ngen::BinaryCodeGenerator<hw>::AccWrEn; using ngen::BinaryCodeGenerator<hw>::NoSrcDepSet; using ngen::BinaryCodeGenerator<hw>::Breakpoint; using ngen::BinaryCodeGenerator<hw>::sat; \
using ngen::BinaryCodeGenerator<hw>::NoMask; \
using ngen::BinaryCodeGenerator<hw>::Serialize; using ngen::BinaryCodeGenerator<hw>::EOT; \
using ngen::BinaryCodeGenerator<hw>::Atomic; using ngen::BinaryCodeGenerator<hw>::Switch; using ngen::BinaryCodeGenerator<hw>::NoPreempt; \
using ngen::BinaryCodeGenerator<hw>::anyv; using ngen::BinaryCodeGenerator<hw>::allv; using ngen::BinaryCodeGenerator<hw>::any2h; using ngen::BinaryCodeGenerator<hw>::all2h; \
using ngen::BinaryCodeGenerator<hw>::any4h; using ngen::BinaryCodeGenerator<hw>::all4h; using ngen::BinaryCodeGenerator<hw>::any8h; using ngen::BinaryCodeGenerator<hw>::all8h; \
using ngen::BinaryCodeGenerator<hw>::any16h; using ngen::BinaryCodeGenerator<hw>::all16h; using ngen::BinaryCodeGenerator<hw>::any32h; using ngen::BinaryCodeGenerator<hw>::all32h; \
using ngen::BinaryCodeGenerator<hw>::x_repl; using ngen::BinaryCodeGenerator<hw>::y_repl; using ngen::BinaryCodeGenerator<hw>::z_repl; using ngen::BinaryCodeGenerator<hw>::w_repl; \
using ngen::BinaryCodeGenerator<hw>::ze; using ngen::BinaryCodeGenerator<hw>::eq; using ngen::BinaryCodeGenerator<hw>::nz; using ngen::BinaryCodeGenerator<hw>::ne; \
using ngen::BinaryCodeGenerator<hw>::gt; using ngen::BinaryCodeGenerator<hw>::ge; using ngen::BinaryCodeGenerator<hw>::lt; using ngen::BinaryCodeGenerator<hw>::le; \
using ngen::BinaryCodeGenerator<hw>::ov; using ngen::BinaryCodeGenerator<hw>::un; using ngen::BinaryCodeGenerator<hw>::eo; \
using ngen::BinaryCodeGenerator<hw>::M0; using ngen::BinaryCodeGenerator<hw>::M4; using ngen::BinaryCodeGenerator<hw>::M8; using ngen::BinaryCodeGenerator<hw>::M12; \
using ngen::BinaryCodeGenerator<hw>::M16; using ngen::BinaryCodeGenerator<hw>::M20; using ngen::BinaryCodeGenerator<hw>::M24; using ngen::BinaryCodeGenerator<hw>::M28; \
using ngen::BinaryCodeGenerator<hw>::sb0; using ngen::BinaryCodeGenerator<hw>::sb1; using ngen::BinaryCodeGenerator<hw>::sb2; using ngen::BinaryCodeGenerator<hw>::sb3; \
using ngen::BinaryCodeGenerator<hw>::sb4; using ngen::BinaryCodeGenerator<hw>::sb5; using ngen::BinaryCodeGenerator<hw>::sb6; using ngen::BinaryCodeGenerator<hw>::sb7; \
using ngen::BinaryCodeGenerator<hw>::sb8; using ngen::BinaryCodeGenerator<hw>::sb9; using ngen::BinaryCodeGenerator<hw>::sb10; using ngen::BinaryCodeGenerator<hw>::sb11; \
using ngen::BinaryCodeGenerator<hw>::sb12; using ngen::BinaryCodeGenerator<hw>::sb13; using ngen::BinaryCodeGenerator<hw>::sb14; using ngen::BinaryCodeGenerator<hw>::sb15; \
using ngen::BinaryCodeGenerator<hw>::A32; using ngen::BinaryCodeGenerator<hw>::A32NC; using ngen::BinaryCodeGenerator<hw>::A64; using ngen::BinaryCodeGenerator<hw>::A64NC; \
using ngen::BinaryCodeGenerator<hw>::SLM; \
template <typename... Targs> ngen::InstructionModifier ExecutionOffset(Targs&&... args) { return ngen::BinaryCodeGenerator<hw>::ExecutionOffset(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::AddressBase Surface(Targs&&... args) { return ngen::BinaryCodeGenerator<hw>::Surface(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::AddressBase CC(Targs&&... args) { return ngen::BinaryCodeGenerator<hw>::CC(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::AddressBase SC(Targs&&... args) { return ngen::BinaryCodeGenerator<hw>::SC(std::forward<Targs>(args)...); }
#define NGEN_FORWARD_REGISTERS_EXTRA
#define NGEN_FORWARD_REGISTERS NGEN_FORWARD_REGISTERS_BASE NGEN_FORWARD_REGISTERS_EXTRA
#endif

template <HW hw>
inline void BinaryCodeGenerator<hw>::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

template <HW hw>
typename BinaryCodeGenerator<hw>::InstructionStream *BinaryCodeGenerator<hw>::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    return result;
}

static inline Instruction12 encodeSyncInsertion(autoswsb::SyncInsertion &si)
{
    Instruction12 i;

    i.common.opcode = static_cast<int>(Opcode::sync);
    i.common.swsb = si.swsb;
    i.common.maskCtrl = true;
    i.binary.cmod = static_cast<int>(si.fc);

    if (si.mask) {
        i.binary.src0Type = getTypecode12(DataType::ud);
        i.binary.src0Imm = true;
        i.imm32.value = si.mask;
    }
    i.binary.dst = 1;

    return i;
}

template <HW hw>
std::vector<uint8_t> BinaryCodeGenerator<hw>::getCode()
{
#ifdef NGEN_SAFE
    if (streamStack.size() > 1) throw unfinished_stream_exception();
#endif
    rootStream.fixLabels(labelManager);

    Program program(rootStream);
    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hw, program);
    std::vector<uint8_t> result;

    if (analysis.empty()) {
        result.resize(rootStream.length());
        std::memmove(result.data(), rootStream.code.data(), rootStream.length());
    } else {
        std::multimap<int32_t, autoswsb::SyncInsertion*> syncs;

        for (auto &bb : analysis)
            for (auto &sync : bb.syncs)
                syncs.insert(std::make_pair(sync.inum, &sync));

        result.resize(rootStream.length() + syncs.size() * sizeof(Instruction12));

        auto *psrc = reinterpret_cast<const Instruction12 *>(rootStream.code.data());
        auto *pdst = reinterpret_cast<Instruction12 *>(result.data());
        auto nextSync = syncs.begin();

        for (uint32_t isrc = 0; isrc < program.size(); isrc++) {
            while ((nextSync != syncs.end()) && (nextSync->second->inum == isrc))
                *pdst++ = encodeSyncInsertion(*(nextSync++)->second);
            *pdst++ = *psrc++;
        }
    }

    return result;
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 1);
    src0.fixup(emod.getExecSize(), defaultType, false, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 1);
    src0.fixup(emod.getExecSize(), defaultType, false, 1);

    encodeCommon12(i, op, emod);

    i.binary.dst  = encodeBinaryOperand12<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand12<false>(src0).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 1);
    src0.fixup(emod.getExecSize(), defaultType, false, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getImmediateTypecode<hw>(src0.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;

    if (getBytes(src0.getType()) == 8)
        i.imm64.value = static_cast<uint64_t>(src0);
    else
        i.imm32.value = static_cast<uint64_t>(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 1);
    src0.fixup(emod.getExecSize(), defaultType, false, 1);

    encodeCommon12(i, op, emod);

    i.binary.dst = encodeBinaryOperand12<true>(dst).bits;

    i.binary.dstAddrMode = dst.isIndirect();

    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Imm = true;

    i.binary.cmod = static_cast<int>(mod.getCMod());

    auto val = static_cast<uint64_t>(src0);
    i.imm32.value = val;
    if (getBytes(src0.getType()) == 8) {
#ifdef NGEN_SAFE
        if (mod.getCMod() != ConditionModifier::none) throw invalid_modifiers_exception();
#endif
        i.imm64.high = val >> 32;
    }

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1)
{
    Instruction8 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 2);
    src0.fixup(emod.getExecSize(), defaultType, false, 2);
    src1.fixup(emod.getExecSize(), defaultType, false, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(src1).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;
    if (src1.isIndirect()) i.binary.src1AddrImm9 = src1.getOffset() >> 9;

    i.binary.dstType  = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getTypecode<hw>(src1.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileGRF;

#ifdef NGEN_SAFE
    if (src1.isARF()) throw grf_expected_exception();
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 2);
    src0.fixup(emod.getExecSize(), defaultType, false, 2);
    src1.fixup(emod.getExecSize(), defaultType, false, 2);

    encodeCommon12(i, op, emod);

    i.binary.dst  = encodeBinaryOperand12<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand12<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand12<false>(src1).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();
    i.binary.src1Mods = src1.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 2);
    src0.fixup(emod.getExecSize(), defaultType, false, 2);
    src1.fixup(emod.getExecSize(), defaultType, false, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getImmediateTypecode<hw>(src1.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

    i.imm32.value = static_cast<uint64_t>(src1);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    dst.fixup(emod.getExecSize(), defaultType, true, 2);
    src0.fixup(emod.getExecSize(), defaultType, false, 2);
    src1.fixup(emod.getExecSize(), defaultType, false, 2);

    encodeCommon12(i, op, emod);

    i.binary.dst  = encodeBinaryOperand12<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand12<false>(src0).bits;
    i.binary.src1 = static_cast<uint64_t>(src1);

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    i.binary.src1Imm = true;
    i.imm32.value = static_cast<uint64_t>(src1);

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2)
{
    opX(op, defaultType, mod, emulateAlign16Dst(dst),  emulateAlign16Src(src0),
                              emulateAlign16Src(src1), emulateAlign16Src(src2));
}


template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2)
{
#ifdef NGEN_SAFE
    if (dst.getReg().isARF())  throw grf_expected_exception();
    if (src0.getReg().isARF()) throw grf_expected_exception();
    if (src1.getReg().isARF()) throw grf_expected_exception();
    if (src2.getReg().isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | Align16;

    dst.getReg().fixup(emod.getExecSize(), defaultType, true, 3);
    src0.getReg().fixup(emod.getExecSize(), defaultType, false, 3);
    src1.getReg().fixup(emod.getExecSize(), defaultType, false, 3);
    src2.getReg().fixup(emod.getExecSize(), defaultType, false, 3);

    encodeCommon8(i, op, emod);

    i.ternary16.dstChanEn = dst.getChanEn();
    i.ternary16.dstRegNum = dst.getReg().getBase();
    i.ternary16.dstSubregNum2_4 = dst.getReg().getByteOffset() >> 2;
    i.ternary16.dstType = getTernary16Typecode8(dst.getReg().getType());

    i.ternary16.srcType = getTernary16Typecode8(src0.getReg().getType());

    bool isFOrHF = (src0.getReg().getType() == DataType::f
                 || src0.getReg().getType() == DataType::hf);

    i.ternary16.src1Type = isFOrHF && (src1.getReg().getType() == DataType::hf);
    i.ternary16.src2Type = isFOrHF && (src1.getReg().getType() == DataType::hf);

    encodeTernaryCommon8(i, src0, src1, src2);

    db(i);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2)
{
    if (hw < HW::Gen10)
        unsupported();

#ifdef NGEN_SAFE
    if (src0.isARF()) throw grf_expected_exception();
    if (src2.isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    dst.fixup(emod.getExecSize(), defaultType, true, 3);
    src0.fixup(emod.getExecSize(), defaultType, false, 3);
    src1.fixup(emod.getExecSize(), defaultType, false, 3);
    src2.fixup(emod.getExecSize(), defaultType, false, 3);

    encodeCommon8(i, op, emod);

    i.ternary1.src0RegFile = std::is_base_of<Immediate, S0>::value;
    i.ternary1.src1RegFile = src1.isARF();
    i.ternary1.src2RegFile = std::is_base_of<Immediate, S2>::value;

    encodeTernaryCommon8(i, src0, src1, src2);
    encodeTernary1Dst10(i, dst);

    db(i);
}

template <HW hw>
template <typename D, typename S0,typename S1, typename S2, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;

    dst.fixup(emod.getExecSize(), defaultType, true, 3);
    src0.fixup(emod.getExecSize(), defaultType, false, 3);
    src1.fixup(emod.getExecSize(), defaultType, false, 3);
    src2.fixup(emod.getExecSize(), defaultType, false, 3);

    encodeCommon12(i, op, emod);

    i.ternary.dst  = encodeTernaryOperand12<true>(dst).bits;
    encodeTernarySrc0(i, src0);
    encodeTernarySrc1(i, src1);
    encodeTernarySrc2(i, src2);
    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <typename DS0>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0);
}

template <HW hw>
template <typename DS0, typename S1>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc)
{
    exdesc |= uint32_t(static_cast<uint8_t>(sfid));
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, D desc)
{
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc)
{
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod);

    i.send.fusionCtrl = emod.isSerialized();

    i.send.dstReg = dst.getBase();
    i.send.src0Reg = src0.getBase();
    i.send.src1Reg = src1.getBase();

    i.send.dstRegFile = getRegFile(dst);
    i.send.src0RegFile = getRegFile(src0);
    i.send.src1RegFile = getRegFile(src1);

    i.send.sfid = static_cast<int>(sfid);

    encodeSendDesc(i, desc);
    encodeSendExDesc(i, exdesc);

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileIMM;

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;
    i.sendsGen9.desc = desc;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(desc).bits;

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(desc);
    i.binary.src1Type = getTypecode<hw>(desc.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc)
{
    opSends(op, mod, dst, src0, null, exdesc, desc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.binary.src0RegFile = 0;                   // ?
    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.sendsGen9.src1RegFile = getRegFile(src1);
    i.sendsGen9.src1RegNum = src1.getBase();

    if (dst.isIndirect())  i.sendsGen9.dstAddrImm9  =  dst.getOffset() >> 9;
    if (src0.isIndirect()) i.sendsGen9.src0AddrImm9 = src0.getOffset() >> 9;

    encodeSendsDesc(i, desc);
    encodeSendsExDesc(i, exdesc);

    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc)
{
#ifdef NGEN_SAFE
    throw sfid_needed_exception();
#endif
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc)
{
    Opcode mop = static_cast<Opcode>(static_cast<int>(op) & ~2);
    opSend(mop, mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0RegFile = getRegFile(Immediate());
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip)
{
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod);

    i.binary.dst = encodeBinaryOperand12<true, false>(dst).bits;

    i.binary.src0Imm = true;
    i.binary.src1Imm = true;

    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip)
{
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod);

    i.binary.dst = encodeBinaryOperand12<true, false>(dst).bits;
    i.binary.src0Imm = true;
    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(DataType::d);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0)
{
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod);

    i.binary.dst = encodeBinaryOperand12<true, false>(dst).bits;
    i.binary.src0 = encodeBinaryOperand12<false, false>(src0).bits;
    if (small12)
        i.binary.src0 &= 0xFFFF;

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    addFixup(LabelFixup(uip.getID(labelManager), LabelFixup::UIPOffset));
    opBranch(op, mod, dst, 0, 0);
}

template <HW hw>
template <bool forceWE>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opBranch<forceWE>(op, mod, dst, 0);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    if (isGen12)
        opBranch<true>(op, mod, dst, 0);
    else
        opX<true>(op, DataType::d, mod, dst, null.ud(0)(0, 1, 0), Immediate::d(0));
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | NoMask;

    encodeCommon8(i, op, emod);

    src0.fixup(emod.getExecSize(), DataType::d, false, 2);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);

    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip)
{
    opBranch<true>(op, mod, dst, jip);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip)
{
    if (hw >= HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opJmpi(op, mod, dst, src0, 0);
    if (hw < HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetJMPI));
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod)
{
    if (hw < HW::Gen12LP)
        unsupported();

    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod);

    i.binary.dst = 0x1;
    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const RegData &src0)
{
    if (hw < HW::Gen12LP)
        unsupported();

    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod);

    i.binary.dst = 0x1;
    i.binary.src0 = encodeBinaryOperand12<false>(src0).bits;
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0)
{
    if (hw < HW::Gen12LP)
        unsupported();

    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod);

    i.binary.dst = 0x1;
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src0Imm = true;
    i.binary.cmod = static_cast<int>(fc);

    i.imm32.value = static_cast<uint64_t>(src0);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opNop(Opcode op)
{
    Instruction8 i{};

    i.qword[0] = static_cast<int>(op);
    i.qword[1] = 0;

    db(i);
}

} /* namespace ngen */

#endif /* header guard */
