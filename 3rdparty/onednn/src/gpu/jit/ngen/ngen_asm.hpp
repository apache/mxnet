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

#ifndef NGEN_ASM_HPP
#define NGEN_ASM_HPP

#include <array>
#include <cstdint>
#include <sstream>
#include <string>

#define NGEN_ASM
#include "ngen.hpp"


namespace ngen {


inline void RegData::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif
    auto vs = getVS();
    if (detail == PrintDetail::vs_hs)
        if (vs > 8 && (getHS() != 0))
            vs = 8;

    if (getNeg()) str << '-';
    if (getAbs()) str << "(abs)";

    if (isARF()) {
        str << getARFType();
        switch (getARFType()) {
            case ARFType::null:
            case ARFType::sp:
            case ARFType::ip:
                break;
            default:
                str << getARFBase();
        }
    } else if (isIndirect()) {
        str << "r[a" << getIndirectBase() << '.' << getIndirectOff();
        if (getOffset())
            str << ',' << getOffset();
        str << ']';
    } else
        str << 'r' << base;

    if (detail <= PrintDetail::base) return;

    if (!isIndirect() && !isNull())
        str << '.' << getOffset();

    if (detail <= PrintDetail::sub_no_type) return;

    if (detail >= PrintDetail::hs && !isNull()) {
        str << '<';
        if (detail >= PrintDetail::vs_hs && !isVxIndirect())
            str << vs << ';';
        if (detail == PrintDetail::full)
            str << getWidth() << ',';
        str << getHS();
        str << '>';
    }

    str << ':' << getType();
}

static inline std::ostream& operator<<(std::ostream &str, const RegData &r)
{
    LabelManager man;
    r.outputText(str, PrintDetail::full, man);
    return str;
}

inline void Immediate::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
    uint64_t nbytes = getBytes(getType());
    uint64_t val;

    if (nbytes == 8)
        val = payload;
    else
        val = payload & ((uint64_t(1) << (nbytes * 8)) - 1);

    str << "0x" << std::hex << val << std::dec;
    if (!hiddenType && detail >= PrintDetail::sub)
        str << ':' << type;
}

inline void ExtendedReg::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

    if (base.getNeg()) str << '-';
    if (base.getAbs()) str << "(abs)";

    str << 'r' << base.getBase() << '.';
    if (mmeNum == 8)
        str << "nomme";
    else
        str << "mme" << int(mmeNum);

    if (detail >= PrintDetail::sub)
        str << ':' << base.getType();
}

inline void Align16Operand::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
    throw iga_align16_exception();
#else
    str << "<unsupported Align16 operand>";
#endif
}

inline void Label::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) {
    str << 'L' << getID(man);
}

struct NoOperand {
    static const bool emptyOp = true;
    void fixup(int esize, DataType defaultType, bool isDest, int arity) const {}
    constexpr bool isScalar() const { return false; }

    void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const {}
};

struct AsmOperand {
    union {
        RegData reg;
        ExtendedReg ereg;
        Immediate imm;
        Label label;
    };
    enum class Type : uint8_t {
        none = 0,
        reg = 1,
        ereg = 2,
        imm = 3,
        label = 4
    } type;

    AsmOperand()                  : type{Type::none} {}
    AsmOperand(NoOperand)         : AsmOperand() {}
    AsmOperand(RegData reg_)      : reg{reg_}, type{Type::reg} {}
    AsmOperand(ExtendedReg ereg_) : ereg{ereg_}, type{Type::ereg} {}
    AsmOperand(Immediate imm_)    : imm{imm_}, type{Type::imm} {}
    AsmOperand(Label label_)      : label{label_}, type{Type::label} {}
    AsmOperand(uint32_t imm_)     : imm{imm_}, type{Type::imm} {}

    void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const {
        switch (type) {
            case Type::none:    break;
            case Type::ereg:    ereg.outputText(str, detail, man); break;
            case Type::reg:     reg.outputText(str, detail, man); break;
            case Type::imm:     imm.outputText(str, detail, man); break;
            case Type::label: {
                auto clone = label;
                clone.outputText(str, detail, man);
                break;
            }
        }
    }
};

struct AsmInstruction {
    Opcode op;
    uint16_t ext;
    uint32_t inum;
    InstructionModifier mod;
    AsmOperand dst, src[4];
    LabelManager *labelManager;
    std::string comment;

    AsmInstruction(Opcode op_, uint16_t ext_, uint32_t inum_, InstructionModifier mod_, AsmOperand dst_,
        AsmOperand src0, AsmOperand src1, AsmOperand src2, AsmOperand src3, LabelManager *man)
            : op(op_), ext(ext_), inum(inum_), mod(mod_), dst(dst_), src{src0, src1, src2, src3}, labelManager{man}, comment{} {}
    explicit AsmInstruction(uint32_t inum_, const std::string &comment_)
            : op(Opcode::illegal), ext(0), inum(inum_), mod{}, dst{}, src{}, labelManager{nullptr}, comment{comment_} {}
    inline AsmInstruction(const autoswsb::SyncInsertion &si);

    bool isLabel() const   { return (op == Opcode::illegal) && (dst.type == AsmOperand::Type::label); }
    bool isComment() const { return (op == Opcode::illegal) && !comment.empty(); }

    // Auto-SWSB interface.
    bool autoSWSB() const       { return mod.isAutoSWSB(); }
    SWSBInfo swsb() const       { return mod.getSWSB(); }
    void setSWSB(SWSBInfo swsb) { mod.setSWSB(swsb); }
    void clearAutoSWSB()        { mod.setAutoSWSB(false); }
    Opcode opcode() const       { return op; }
    SyncFunction syncFC() const { return static_cast<SyncFunction>(ext); }
    SharedFunction sfid() const { return static_cast<SharedFunction>(ext); }
    bool eot() const            { return mod.isEOT(); }
    bool predicated() const     { return !mod.isWrEn() || (mod.getPredCtrl() != PredCtrl::None); }

    unsigned dstTypecode() const;
    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;

    void shiftJIP(int32_t shift) const {}
    void shiftUIP(int32_t shift) const {}

    bool getImm32(uint32_t &imm, int opNum = 0) const {
        if (src[opNum].type == AsmOperand::Type::imm) {
            imm = uint32_t(static_cast<uint64_t>(src[opNum].imm));
            return true;
        } else
            return false;
    }
    bool getARFType(ARFType &arfType, int opNum) const {
        auto &opd = src[opNum];
        if (opd.type == AsmOperand::Type::reg && opd.reg.isARF()) {
            arfType = opd.reg.getARFType();
            return true;
        } else
            return false;
    }
    bool getSendDesc(MessageDescriptor &desc) const { return getImm32(desc.all, 3); }
};

AsmInstruction::AsmInstruction(const autoswsb::SyncInsertion &si)
{
    op = Opcode::sync;
    ext = static_cast<uint8_t>(si.fc);
    mod = InstructionModifier::createMaskCtrl(true);
    mod.setSWSB(si.swsb);
    dst = NoOperand();
    for (auto n = 0; n < 4; n++)
        src[n] = NoOperand();
    if (si.mask)
        src[0] = Immediate::ud(si.mask);
    else
        src[0] = NullRegister();
}

unsigned AsmInstruction::dstTypecode() const
{
    DataType dt = DataType::invalid;

    switch (dst.type) {
        case AsmOperand::Type::reg:  dt = dst.reg.getType(); break;
        case AsmOperand::Type::ereg: dt = dst.ereg.getType(); break;
        default: break;
    }

    return getTypecode12(dt);
}

autoswsb::DestinationMask AsmInstruction::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(op))
        return eot() ? DestNone : DestNextIP;

    if (src[0].type == AsmOperand::Type::reg)
        return DestUnknown;

    DestinationMask mask = DestNextIP;
    if (src[0].type == AsmOperand::Type::label) {
        auto label = src[0].label;
        mask |= DestJIP;
        jip = labelManager->getTarget(label.getID(*labelManager)) - inum;
    }

    if (src[1].type == AsmOperand::Type::label) {
        auto label = src[1].label;
        mask |= DestUIP;
        uip = labelManager->getTarget(label.getID(*labelManager)) - inum;
    }

    if (op == Opcode::jmpi && mod.getPredCtrl() == PredCtrl::None)
        mask &= ~DestNextIP;

    return mask;
}

bool AsmInstruction::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using namespace autoswsb;
    const AsmOperand &operand = (opNum < 0) ? dst : src[opNum];
    RegData rd;

    switch (operand.type) {
        case AsmOperand::Type::reg:  rd = operand.reg; break;
        case AsmOperand::Type::ereg: rd = operand.ereg.getBase(); break;
        default: return false;
    }

    if (rd.isARF())
        return false;

    if (rd.isIndirect())
        region = DependencyRegion();
    else if (op == Opcode::send || op == Opcode::sendc) {
        int len = 0;
        if (opNum <= 0) {
            if (src[3].type == AsmOperand::Type::imm) {
                MessageDescriptor desc;
                desc.all = static_cast<uint64_t>(src[3].imm);
                len = (opNum < 0) ? desc.parts.responseLen : desc.parts.messageLen;
            } else
                len = -1;
        } else if (opNum == 1) {
            if (src[2].type == AsmOperand::Type::imm) {
                ExtendedMessageDescriptor exdesc;
                exdesc.all = static_cast<uint64_t>(src[2].imm);
                len = exdesc.parts.extMessageLen;
            } else
                len = -1;
        }
        if (len == 0)
            return false;
        else if (len == -1)
            region = DependencyRegion();
        else
            region = DependencyRegion(GRFRange(rd.getBase(), len));
    } else
        region = DependencyRegion(mod.getExecSize(), rd);

    return true;
}

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#include "ngen_registers.hpp"
#endif

class AsmCodeGenerator {
private:
#include "ngen_compiler_fix.hpp"
public:
    AsmCodeGenerator(HW hardware_) : hardware(hardware_), isGen12(hardware_ >= HW::Gen12LP),
            defaultOutput{nullptr}, sync{this} {
        _workaround_();
        streamStack.push_back(new InstructionStream());
    }
    AsmCodeGenerator(HW hardware_, std::ostream &defaultOutput_) : AsmCodeGenerator(hardware_) {
        defaultOutput = &defaultOutput_;
    }
    ~AsmCodeGenerator() noexcept(false) {
        if (defaultOutput != nullptr)
            getCode(*defaultOutput);
        for (auto &s : streamStack)
            delete s;
    }
    inline void getCode(std::ostream &out);

protected:
    struct InstructionStream {
        std::vector<AsmInstruction> buffer;
        std::vector<uint32_t> labels;

        template <typename... Remaining>
        AsmInstruction &append(Opcode op, uint16_t ext, Remaining&&... args) {
            buffer.emplace_back(op, ext, 0, std::forward<Remaining>(args)...);
            return buffer.back();
        }

        void appendComment(const std::string &str) { buffer.emplace_back(0, str); }

        void mark(Label &label, LabelManager &man) {
            uint32_t id = label.getID(man);

            man.setTarget(id, buffer.size());
            labels.push_back(id);
            buffer.emplace_back(Opcode::illegal, 0, 0, InstructionModifier(), label, NoOperand(), NoOperand(), NoOperand(), NoOperand(), &man);
        }

        void append(InstructionStream &other, LabelManager &man) {
            for (uint32_t id : other.labels)
                man.offsetTarget(id, buffer.size());

            buffer.insert(buffer.end(), other.buffer.begin(), other.buffer.end());
            labels.insert(labels.end(), other.labels.begin(), other.labels.end());
        }
    };

    HW hardware;
    bool isGen12;
    std::ostream *defaultOutput;

private:
    InstructionModifier defaultModifier;
    LabelManager labelManager;
    std::vector<InstructionStream*> streamStack;

    inline void unsupported();

    // Output functions.
    template <typename D, typename S0, typename S1, typename S2>
    inline void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext);

    template <typename D, typename S0, typename S1, typename S2> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2) {
        opX(op, defaultType, mod, dst, src0, src1, src2, 0);
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, defaultType, mod, dst, src0, src1, NoOperand());
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, DataType::invalid, mod, dst, src0, src1);
    }
    template <typename D, typename S0> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, defaultType, mod, dst, src0, NoOperand());
    }
    template <typename D, typename S0> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, DataType::invalid, mod, dst, src0);
    }
    template <typename D> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst) {
        opX(op, defaultType, mod, dst, NoOperand());
    }
    template <typename D> void opX(Opcode op, const InstructionModifier &mod, D dst) {
        opX(op, DataType::invalid, mod, dst);
    }
    void opX(Opcode op) {
        opX(op, InstructionModifier(), NoOperand());
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip);
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand());
    }

    template <typename S1, typename ED, typename D>
    void opSend(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, RegData src0, S1 src1, ED exdesc, D desc) {
        auto &i = streamStack.back()->append(op, static_cast<uint8_t>(sf), mod | defaultModifier, dst, src0, src1, exdesc, desc, &labelManager);
        if (i.src[2].type == AsmOperand::Type::imm) {
            if (isGen12)
                i.src[2].imm = uint32_t(static_cast<uint64_t>(i.src[2].imm) & ~0x2F);
            else
                i.src[2].imm = uint32_t(static_cast<uint64_t>(i.src[2].imm) | static_cast<uint8_t>(sf));
        }
    }
    template <typename D, typename S0> void opCall(Opcode op, const InstructionModifier &mod, D dst, S0 src0) {
        (void) streamStack.back()->append(op, 0, mod | defaultModifier | NoMask, dst, src0, NoOperand(), NoOperand(), NoOperand(), &labelManager);
    }
    template <typename S1> void opJmpi(Opcode op, const InstructionModifier &mod, S1 src1) {
        (void) streamStack.back()->append(op, 0, mod | defaultModifier | NoMask, NoOperand(), src1, NoOperand(), NoOperand(), NoOperand(), &labelManager);
    }
    template <typename S0> void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, S0 src0) {
        (void) streamStack.back()->append(op, static_cast<uint8_t>(fc), mod | defaultModifier, NoOperand(), src0, NoOperand(), NoOperand(), NoOperand(), &labelManager);
    }

    inline void finalize();

    enum class ModPlacementType {Pre, Mid, Post};
    inline void outX(std::ostream &out, const AsmInstruction &i);
    inline void outExt(std::ostream &out, const AsmInstruction &i);
    inline void outMods(std::ostream &out, const InstructionModifier &mod, Opcode op, ModPlacementType location);
    inline void outSync(std::ostream &out, const autoswsb::SyncInsertion &si);

protected:
    // Configuration.
    void setDefaultNoMask(bool def = true)          { defaultModifier.setWrEn(def); }
    void setDefaultAutoSWSB(bool def = true)        { defaultModifier.setAutoSWSB(def); }

    // Stream handling.
    void pushStream()                               { pushStream(new InstructionStream()); }
    void pushStream(InstructionStream &s)           { pushStream(&s); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); }

    inline InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s, labelManager); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    void comment(const std::string &str)            { streamStack.back()->appendComment(str); }

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
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::brc, mod, jip, uip);
    }
    void brc(const InstructionModifier &mod, const RegData &src0) {
        opCall(Opcode::brc, mod, NoOperand(), src0);
    }
    void brd(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::brd, mod, jip);
    }
    void brd(const InstructionModifier &mod, const RegData &src0) {
        opCall(Opcode::brd, mod, NoOperand(), src0);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::break_, mod, jip, uip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip) {
        (void) jip.getID(labelManager);
        opCall(Opcode::call, mod, dst, jip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, const RegData &jip) {
        opCall(Opcode::call, mod, dst, jip);
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip) {
        opCall(Opcode::calla, mod, dst, Immediate::ud(jip));
    }
    void calla(const InstructionModifier &mod, const RegData &dst, const RegData &jip) {
        opCall(Opcode::calla, mod, dst, jip);
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
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::cont, mod, jip, uip);
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
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
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
    void else_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::else_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
    }
    void else_(InstructionModifier mod, Label &jip) {
        else_(mod, jip, jip);
    }
    void endif(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::endif, mod, NoOperand(), jip);
    }
    void endif(const InstructionModifier &mod) {
        Label next;
        endif(mod, next);
        mark(next);
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
    void goto_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::goto_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
    }
    void goto_(const InstructionModifier &mod, Label &jip) {
        goto_(mod, jip, jip);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::halt, mod, jip, uip);
    }
    void halt(const InstructionModifier &mod, Label &jip) {
        halt(mod, jip, jip);
    }
    void if_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::if_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
    }
    void if_(const InstructionModifier &mod, Label &jip) {
        if_(mod, jip, jip);
    }
    void illegal() {
        opX(Opcode::illegal);
    }
    void join(const InstructionModifier &mod, Label &jip) {
        opX(Opcode::join, mod, jip);
    }
    void join(const InstructionModifier &mod) {
        Label next;
        join(mod, next);
        mark(next);
    }
    void jmpi(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opJmpi(Opcode::jmpi, mod, jip);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip) {
        opJmpi(Opcode::jmpi, mod, jip);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
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
        if (hardware < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const Align16Operand &dst, const Align16Operand &src0, const Align16Operand &src1, const Align16Operand &src2) {
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
    template <typename DT = void>
    void madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2) {
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 1) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::rsqtm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme);
        else
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (mathArgCount(fc) != 2) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::invm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme, src1 | nomme);
        else
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1.forceInt32(), NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        mod.setCMod(ConditionModifier::eo);
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        mod.setCMod(ConditionModifier::eo);
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
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
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
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
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop);
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
        opX(isGen12 ? Opcode::or_gen12 : Opcode:: or_ , getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode:: or_ , getDataType<DT>(), mod, dst, src0, src1);
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
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1);
    }
    void ret(const InstructionModifier &mod, const RegData &src0) {
        opJmpi(Opcode::ret, mod, src0);
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
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
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
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, Immediate::ud(exdesc), desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, Immediate::ud(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    template <typename T1, typename T2> void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, NoOperand src1, T1 exdesc, T2 desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    template <typename T1, typename T2> void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, NoOperand src1, T1 exdesc, T2 desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    /* Pre-Gen12 style sends */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
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
        opX(Opcode::wait, mod, NoOperand(), nreg);
    }
    void while_(const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(Opcode::while_, mod, jip);
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
        AsmCodeGenerator &parent;

        Sync(AsmCodeGenerator *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier()) {
            parent.opSync(Opcode::sync, fc, mod, null);
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
        void operator()(SyncFunction fc, const InstructionModifier &mod, int src0) {
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

    inline void mark(Label &label)          { streamStack.back()->mark(label, labelManager); }

#include "ngen_pseudo.hpp"
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif
};


void AsmCodeGenerator::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

AsmCodeGenerator::InstructionStream *AsmCodeGenerator::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    return result;
}

void AsmCodeGenerator::finalize()
{
#ifdef NGEN_SAFE
    if (streamStack.size() > 1) throw unfinished_stream_exception();
#endif
    auto &buffer = streamStack.back()->buffer;
    int inum = 0;
    for (auto &i : buffer)
        i.inum = inum++;
}

void AsmCodeGenerator::getCode(std::ostream &out)
{
    finalize();

    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hardware, streamStack.back()->buffer);
    std::multimap<int32_t, autoswsb::SyncInsertion*> syncs;      // Syncs inserted by auto-SWSB.

    for (auto &bb : analysis)
        for (auto &sync : bb.syncs)
            syncs.insert(std::make_pair(sync.inum, &sync));

    auto nextSync = syncs.begin();

    for (auto &i : streamStack.back()->buffer) {
        if (i.isLabel()) {
            i.dst.label.outputText(out, PrintDetail::full, labelManager);
            out << ':' << std::endl;
        } else if (i.isComment()) {
            out << "// " << i.comment << std::endl;
        } else {
            while ((nextSync != syncs.end()) && (nextSync->second->inum == i.inum))
                outX(out, *(nextSync++)->second);
            outX(out, i);
        }
    }
}

template <typename D, typename S0, typename S1, typename S2>
void AsmCodeGenerator::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext)
{
    bool is2Src = !S1::emptyOp;
    bool is3Src = !S2::emptyOp;
    int arity = 1 + is2Src + is3Src;

    InstructionModifier emod = mod | defaultModifier;
    auto esize = emod.getExecSize();

    if (is3Src && hardware < HW::Gen10)
        esize = std::min<int>(esize, 8);        // WA for IGA Align16 emulation issue

#ifdef NGEN_SAFE
    if (esize > 1 && dst.isScalar())
        throw invalid_execution_size_exception();
#endif

    dst.fixup(esize, defaultType, true, arity);
    src0.fixup(esize, defaultType, false, arity);
    src1.fixup(esize, defaultType, false, arity);
    src2.fixup(esize, defaultType, false, arity);

    streamStack.back()->append(op, ext, emod, dst, src0, src1, src2, NoOperand{}, &labelManager);
}

void AsmCodeGenerator::outX(std::ostream &out, const AsmInstruction &i)
{
    bool ternary = (i.src[2].type != AsmOperand::Type::none);
    PrintDetail ddst = PrintDetail::hs;
    PrintDetail dsrc01 = ternary ? PrintDetail::vs_hs : PrintDetail::full;
    PrintDetail dsrc[4] = {dsrc01, dsrc01, PrintDetail::hs, PrintDetail::base};

    switch (i.op) {
        case Opcode::send:
        case Opcode::sends:
        case Opcode::sendc:
        case Opcode::sendsc:
            ddst = dsrc[0] = dsrc[1] = PrintDetail::base;
            dsrc[2] = dsrc[3] = PrintDetail::sub_no_type;
            break;
        case Opcode::brc:
        case Opcode::brd:
        case Opcode::call:
        case Opcode::calla:
            ddst = PrintDetail::sub;
            dsrc[0] = PrintDetail::sub_no_type;
            break;
        case Opcode::jmpi:
        case Opcode::ret:
            dsrc[0] = PrintDetail::sub_no_type;
            break;
        case Opcode::sync:
            if (isGen12) dsrc[0] = PrintDetail::sub_no_type;
        default: break;
    }

    outMods(out, i.mod, i.op, ModPlacementType::Pre);

    out << getMnemonic(i.op, hardware);
    outExt(out, i);
    out << '\t';

    outMods(out, i.mod, i.op, ModPlacementType::Mid);

    i.dst.outputText(out, ddst, labelManager); out << '\t';
    for (int n = 0; n < 4; n++) {
        i.src[n].outputText(out, dsrc[n], labelManager); out << '\t';
    }

    outMods(out, i.mod, i.op, ModPlacementType::Post);
    out << std::endl;
}

void AsmCodeGenerator::outExt(std::ostream &out, const AsmInstruction &i)
{
    switch (i.opcode()) {
        case Opcode::else_:
        case Opcode::goto_:
        case Opcode::if_:       if (i.ext) out << ".b";                                         break;
        case Opcode::math:      out << '.' << static_cast<MathFunction>(i.ext);                 break;
        default: break;
    }

    if (isGen12) switch (i.opcode()) {
        case Opcode::send:
        case Opcode::sends:     out << '.' << static_cast<SharedFunction>(i.ext);               break;
        case Opcode::sync:      out << '.' << static_cast<SyncFunction>(i.ext);                 break;
        default: break;
    }
}

void AsmCodeGenerator::outMods(std::ostream &out,const InstructionModifier &mod, Opcode op, AsmCodeGenerator::ModPlacementType location)
{
    ConditionModifier cmod = mod.getCMod();
    PredCtrl ctrl = mod.getPredCtrl();
    bool wrEn = mod.isWrEn();
    bool havePred = (ctrl != PredCtrl::None) && (cmod != ConditionModifier::eo);

    switch (location) {
        case ModPlacementType::Pre:
            if (wrEn || havePred) {
                out << '(';
                if (wrEn) {
                    out << 'W';
                    if (havePred) out << '&';
                }
                if (havePred) {
                    if (mod.isPredInv()) out << '~';
                    mod.getFlagReg().outputText(out, PrintDetail::sub_no_type, labelManager);
                    if (ctrl != PredCtrl::Normal)
                        out << '.' << toText(ctrl, mod.isAlign16());
                }
                out << ')';
            }
            out << '\t';
            break;
        case ModPlacementType::Mid:
            if (mod.getExecSize() > 0)
                out << '(' << mod.getExecSize() << "|M" << mod.getChannelOffset() << ')' << '\t';

            if (cmod != ConditionModifier::none) {
                out << '(' << cmod << ')';
                mod.getFlagReg().outputText(out, PrintDetail::sub_no_type, labelManager);
                out << '\t';
            }

            if (mod.isSaturate()) out << "(sat)";
            break;
        case ModPlacementType::Post:
        {
            bool havePostMod = false;
            auto startPostMod = [&]() {
                out << (havePostMod ? ',' : '{');
                havePostMod = true;
            };
            auto printPostMod = [&](const char *name) {
                startPostMod(); out << name;
            };

            SWSBInfo swsb = mod.getSWSB();
            if (swsb.hasSB()) {
                SBInfo sb = swsb.sb();
                startPostMod(); out << '$' << sb.getID();
                if (sb.isSrc()) out << ".src";
                if (sb.isDst() || (!isVariableLatency(op) && (swsb.dist() > 0)))
                    out << ".dst";
            }
            if (swsb.dist() > 0) {
                startPostMod();
                if (hardware > HW::Gen12LP || !swsb.hasSB())
                    out << swsb.pipe();
                out << '@' << swsb.dist();
            }

            if (mod.isAlign16())                                          printPostMod("Align16");
            if (mod.isNoDDClr())                                          printPostMod("NoDDClr");
            if (mod.isNoDDChk())                                          printPostMod("NoDDChk");
            if (mod.getThreadCtrl() == ThreadCtrl::Atomic)                printPostMod("Atomic");
            if (!isGen12 && mod.getThreadCtrl() == ThreadCtrl::Switch)    printPostMod("Switch");
            if (!isGen12 && mod.getThreadCtrl() == ThreadCtrl::NoPreempt) printPostMod("NoPreempt");
            if (mod.isAccWrEn())                                          printPostMod("AccWrEn");
            if (mod.isCompact())                                          printPostMod("Compact");
            if (mod.isBreakpoint())                                       printPostMod("Breakpoint");
            if (mod.isSerialized())                                       printPostMod("Serialize");
            if (mod.isEOT())                                              printPostMod("EOT");

            if (havePostMod) out << '}';
        }
        break;
    }
}

} /* namespace ngen */

#endif
