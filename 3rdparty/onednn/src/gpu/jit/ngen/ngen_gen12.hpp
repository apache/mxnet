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
 * Do not #include this file directly; ngen uses it internally.
 */

// Gen12 binary encoding.

// 24 bits of data common between src0 and src1 (lower 16 bits common with dst)
union BinaryOperand12 {
    uint32_t bits;
    struct {
        unsigned hs : 2;
        unsigned regFile : 1;
        unsigned subRegNum : 5;
        unsigned regNum : 8;
        unsigned addrMode : 1;          // = 0 (direct)
        unsigned width : 3;
        unsigned vs : 4;
    } direct;
    struct {
        unsigned hs : 2;
        unsigned addrOff : 10;
        unsigned addrReg : 4;
        unsigned addrMode : 1;          // = 1 (indirect)
        unsigned width : 3;
        unsigned vs : 4;
    } indirect;
};

// 16 bits of data common between dst, src0/1/2 for 3-source instructions
union TernaryOperand12 {
    uint16_t bits;
    struct {
        unsigned hs : 2;
        unsigned regFile : 1;
        unsigned subRegNum : 5;         // mme# for math
        unsigned regNum : 8;
    } direct;
};

union Instruction12 {
    struct {                            // Lower 35 bits are essentially common.
        unsigned opcode : 8;            // High bit reserved, used for auto-SWSB flag.
        unsigned swsb : 8;
        unsigned execSize : 3;
        unsigned execOffset : 3;
        unsigned flagReg : 2;
        unsigned predCtrl : 4;
        unsigned predInv : 1;
        unsigned cmptCtrl : 1;
        unsigned debugCtrl : 1;
        unsigned maskCtrl : 1;
        //
        unsigned atomicCtrl : 1;
        unsigned accWrCtrl : 1;
        unsigned saturate : 1;
        unsigned : 29;
        //
        unsigned : 32;
        unsigned : 32;
    } common;
    struct {
        unsigned : 32;
        //
        unsigned : 3;
        unsigned dstAddrMode : 1;
        unsigned dstType : 4;
        unsigned src0Type : 4;
        unsigned src0Mods : 2;
        unsigned src0Imm : 1;
        unsigned src1Imm : 1;
        unsigned dst : 16;              // first 16 bits of BinaryOperand12
        //
        unsigned src0 : 24;             // BinaryOperand12
        unsigned src1Type : 4;
        unsigned cmod : 4;
        //
        unsigned src1 : 24;             // BinaryOperand12
        unsigned src1Mods : 2;
        unsigned _ : 6;
    } binary;
    struct {
        uint64_t _;
        uint32_t __;
        uint32_t value;
    } imm32;
    struct {
        uint64_t _;
        uint32_t high;
        uint32_t low;
    } imm64;
    struct {
        unsigned : 32;                  // common
        unsigned : 3;
        unsigned src0VS0 : 1;
        unsigned dstType : 3;
        unsigned execType : 1;
        unsigned src0Type : 3;
        unsigned src0VS1 : 1;
        unsigned src0Mods : 2;
        unsigned src0Imm : 1;
        unsigned src2Imm : 1;
        unsigned dst : 16;              // TernaryOperand12 or immediate
        //
        unsigned src0 : 16;
        unsigned src2Type : 3;
        unsigned src1VS0 : 1;
        unsigned src2Mods : 2;
        unsigned src1Mods : 2;
        unsigned src1Type : 3;
        unsigned src1VS1 : 1;
        unsigned cmod : 4;              // same location as binary
        //
        unsigned src1 : 16;             // TernaryOperand12
        unsigned src2 : 16;             // TernaryOperand12 or immediate
    } ternary;
    struct {
        unsigned : 32;
        //
        unsigned : 1;
        unsigned fusionCtrl : 1;
        unsigned eot : 1;
        unsigned exDesc11_23 : 13;
        unsigned descIsReg : 1;
        unsigned exDescIsReg : 1;
        unsigned dstRegFile : 1;
        unsigned desc20_24 : 5;
        unsigned dstReg : 8;
        //
        unsigned exDesc24_25 : 2;
        unsigned src0RegFile : 1;
        unsigned desc25_29 : 5;
        unsigned src0Reg : 8;
        unsigned : 1;
        unsigned desc0_10 : 11;
        unsigned sfid : 4;
        //
        unsigned exDesc26_27 : 2;
        unsigned src1RegFile : 1;
        unsigned exDesc6_10 : 5;
        unsigned src1Reg : 8;
        unsigned : 1;
        unsigned desc11_19 : 9;
        unsigned desc30_31 : 2;
        unsigned exDesc28_31 : 4;
    } send;
    struct {
        unsigned : 32;
        unsigned : 8;
        unsigned exDescReg : 3;
        unsigned : 21;
        unsigned : 32;
        unsigned : 32;
    } sendIndirect;
    struct {
        unsigned : 32;                  // common
        unsigned : 1;
        unsigned branchCtrl : 1;
        unsigned : 30;
        int32_t uip;
        int32_t jip;
    } branches;
    uint64_t qword[2];

    constexpr Instruction12() : qword{0,0} {};

    // Decoding routines for auto-SWSB.
    bool autoSWSB() const        { return (common.opcode & 0x80); }
    SWSBInfo swsb() const        { return SWSBInfo::createFromRaw(common.swsb); }
    void setSWSB(SWSBInfo swsb)  { common.swsb = swsb.raw(); }
    void clearAutoSWSB()         { common.opcode &= 0x7F; }
    Opcode opcode() const        { return static_cast<Opcode>(common.opcode & 0x7F); }
    SyncFunction syncFC() const  { return static_cast<SyncFunction>(binary.cmod); }
    SharedFunction sfid() const  { return static_cast<SharedFunction>(send.sfid); }
    bool eot() const             { return (opcode() == Opcode::send || opcode() == Opcode::sendc) && send.eot; }
    bool predicated() const      { return !common.maskCtrl || (static_cast<PredCtrl>(common.predCtrl) != PredCtrl::None); }
    unsigned dstTypecode() const { return binary.dstType; }
    void shiftJIP(int32_t shift) { branches.jip += shift * sizeof(Instruction12); }
    void shiftUIP(int32_t shift) { branches.uip += shift * sizeof(Instruction12); }

    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;
    inline bool getImm32(uint32_t &imm) const;
    inline bool getSendDesc(MessageDescriptor &desc) const;
    inline bool getARFType(ARFType &arfType, int opNum) const;

    bool isMathMacro() const {
        if (opcode() != Opcode::math) return false;
        auto fc = static_cast<MathFunction>(binary.cmod);
        return (fc == MathFunction::invm || fc == MathFunction::rsqtm);
    }
};

static_assert(sizeof(Instruction12) == 16, "Internal error: Instruction12 has been padded by the compiler.");

// Encoding routines.

static inline unsigned getTypecode12(DataType type)
{
    static const uint8_t conversionTable[16] = {2,6,1,5,0,4,11,10,3,7,9,0,2,0,4,8};
    return conversionTable[static_cast<unsigned>(type) & 0xF];
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const RegData &rd)
{
    BinaryOperand12 op{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        op.indirect.addrOff = rd.getOffset();
        op.indirect.addrReg = rd.getIndirectOff();
        op.indirect.addrMode = 1;
        if (!dest) {
            op.indirect.vs = (rd.isVxIndirect()) ? 0xFFFF :
                               (rd.getVS() == 0) ? 0 :
                                                   (1 + utils::log2(rd.getVS()));
        }
    } else {
        op.direct.regFile = getRegFile(rd);
        op.direct.subRegNum = rd.getByteOffset();
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (!dest)
            op.direct.vs = (rd.getVS() == 0) ? 0 : (1 + utils::log2(rd.getVS()));
    }

    if (encodeHS)
        op.direct.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));

    if (!dest) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <bool dest>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const ExtendedReg &reg)
{
    auto op = encodeBinaryOperand12<dest>(reg.getBase());
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const RegData &rd)
{
#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand12 op{0};

    if (encodeHS) {
        if (dest)
            op.direct.hs = utils::log2(rd.getHS());
        else
            op.direct.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));
    }

    op.direct.regFile = getRegFile(rd);
    op.direct.subRegNum = rd.getByteOffset();
    op.direct.regNum = rd.getBase();

    return op;
}

template <bool dest>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const ExtendedReg &reg)
{
    auto op = encodeTernaryOperand12<dest>(reg.getBase());
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod)
{
    i.common.opcode = static_cast<unsigned>(opcode) | (mod.parts.autoSWSB << 7);
    i.common.swsb = mod.parts.swsb;
    i.common.execSize = mod.parts.eSizeField;
    i.common.execOffset = mod.parts.chanOff;
    i.common.flagReg = (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i.common.predCtrl = mod.parts.predCtrl;
    i.common.predInv = mod.parts.predInv;
    i.common.cmptCtrl = mod.parts.cmptCtrl;
    i.common.debugCtrl = mod.parts.debugCtrl;
    i.common.maskCtrl = mod.parts.maskCtrl;
    i.common.atomicCtrl = mod.parts.threadCtrl;
    i.common.accWrCtrl = mod.parts.accWrCtrl;
    i.common.saturate = mod.parts.saturate;
}

static inline unsigned encodeTernaryVS01(const RegData &rd)
{
    switch (rd.getVS()) {
        case 0: return 0;
        case 1: return 1;
        case 4: return 2;
        case 8: return 3;
        default:
#ifdef NGEN_SAFE
            if (rd.getHS() == 0)
                throw invalid_region_exception();
#endif
            return 3;
    }
}

static inline unsigned encodeTernaryVS01(const ExtendedReg &reg)
{
    return encodeTernaryVS01(reg.getBase());
}

template <typename D, typename S0, typename S1, typename S2>
static inline void encodeTernaryTypes(Instruction12 &i, D dst, S0 src0, S1 src1, S2 src2)
{
    auto dtype = getTypecode12(dst.getType());
    auto s0type = getTypecode12(src0.getType());
    auto s1type = getTypecode12(src1.getType());
    auto s2type = getTypecode12(src2.getType());

    i.ternary.execType = (dtype >> 3);
    i.ternary.dstType  = dtype;
    i.ternary.src0Type = s0type;
    i.ternary.src1Type = s1type;
    i.ternary.src2Type = s2type;

#ifdef NGEN_SAFE
    if (((dtype & s0type & s1type & s2type) ^ (dtype | s0type | s1type | s2type)) & 8)
        throw ngen::invalid_type_exception();
#endif
}

template <typename S0>
static inline void encodeTernarySrc0(Instruction12 &i, S0 src0)
{
    i.ternary.src0 = encodeTernaryOperand12<false>(src0).bits;
    i.ternary.src0Mods = src0.getMods();

    auto vs0 = encodeTernaryVS01(src0);

    i.ternary.src0VS0 = vs0;
    i.ternary.src0VS1 = vs0 >> 1;
}

static inline void encodeTernarySrc0(Instruction12 &i, const Immediate &src0)
{
    i.ternary.src0Imm = true;
    i.ternary.src0 = static_cast<uint64_t>(src0);
}

template <typename S1>
static inline void encodeTernarySrc1(Instruction12 &i, S1 src1)
{
    i.ternary.src1 = encodeTernaryOperand12<false>(src1).bits;
    i.ternary.src1Mods = src1.getMods();

    auto vs1 = encodeTernaryVS01(src1);

    i.ternary.src1VS0 = vs1;
    i.ternary.src1VS1 = vs1 >> 1;
}

template <typename S2>
static inline void encodeTernarySrc2(Instruction12 &i, S2 src2)
{
    i.ternary.src2 = encodeTernaryOperand12<false>(src2).bits;
    i.ternary.src2Mods = src2.getMods();
}

static inline void encodeTernarySrc2(Instruction12 &i, const Immediate &src2)
{
    i.ternary.src2Imm = true;
    i.ternary.src2 = static_cast<uint64_t>(src2);
}

static inline void encodeSendExDesc(Instruction12 &i, uint32_t exdesc)
{
    i.send.eot = (exdesc >> 5);
    i.send.exDesc6_10 = (exdesc >> 6);
    i.send.exDesc11_23 = (exdesc >> 11);
    i.send.exDesc24_25 = (exdesc >> 24);
    i.send.exDesc26_27 = (exdesc >> 26);
    i.send.exDesc28_31 = (exdesc >> 28);
}

static inline void encodeSendExDesc(Instruction12 &i, RegData exdesc)
{
#ifdef NGEN_SAFE
    // Only a0.x:ud is allowed for extended descriptor.
    if (!exdesc.isARF() || exdesc.getARFType() != ARFType::a || exdesc.getARFBase() != 0 || exdesc.getType() != DataType::ud)
        throw invalid_arf_exception();
#endif
    i.sendIndirect.exDescReg = exdesc.getOffset();
    i.send.exDescIsReg = true;
}

static inline void encodeSendDesc(Instruction12 &i, uint32_t desc)
{
    i.send.desc0_10 = (desc >> 0);
    i.send.desc11_19 = (desc >> 11);
    i.send.desc20_24 = (desc >> 20);
    i.send.desc25_29 = (desc >> 25);
    i.send.desc30_31 = (desc >> 30);
}

static inline void encodeSendDesc(Instruction12 &i, RegData desc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    i.send.descIsReg = true;
}

/*********************/
/* Decoding Routines */
/*********************/

static inline DataType decodeRegTypecode12(unsigned dt)
{
    static const DataType conversionTable[16] = {
        DataType::ub,      DataType::uw,      DataType::ud,      DataType::uq,
        DataType::b,       DataType::w,       DataType::d,       DataType::q,
        DataType::invalid, DataType::hf,      DataType::f,       DataType::df,
        DataType::invalid, DataType::invalid, DataType::invalid, DataType::invalid
    };
    return conversionTable[dt & 0xF];
}

bool Instruction12::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using namespace autoswsb;

    auto op = opcode();
    RegData rd;

    switch (op) {
        case Opcode::nop_gen12:
        case Opcode::illegal:
            return false;
        case Opcode::send:
        case Opcode::sendc: {
            int base = 0, len = 0;
            switch (opNum) {
                case -1:
                    if (send.dstRegFile == RegFileARF) return false;
                    base = send.dstReg;
                    len = send.descIsReg ? -1 : send.desc20_24;
                    break;
                case 0:
                    if (send.src0RegFile == RegFileARF) return false;
                    base = send.src0Reg;
                    len = send.descIsReg ? -1 : (send.desc25_29 & 0xF);
                    break;
                case 1:
                    if (send.src1RegFile == RegFileARF) return false;
                    base = send.src1Reg;
                    len = send.exDescIsReg ? -1 : send.exDesc6_10;
                    break;
                default: return false;
            }

            if (len == 0)
                return false;
            else if (len == -1)
                region = DependencyRegion();
            else
                region = DependencyRegion(GRFRange(base, len));
            return true;
        }
        case Opcode::dp4a:
        case Opcode::bfe_gen12:
        case Opcode::bfi2_gen12:
        case Opcode::csel_gen12:
        case Opcode::mad:
        case Opcode::madm: {  // ternary
            TernaryOperand12 o;
            unsigned dt = 0, vs = 0;
            switch (opNum) {
                case -1:
                    o.bits = ternary.dst;
                    dt = ternary.dstType;
                    break;
                case 0:
                    if (ternary.src0Imm) return false;
                    o.bits = ternary.src0;
                    dt = ternary.src0Type;
                    vs = ternary.src0VS0 + (ternary.src0VS1 * 3);
                    break;
                case 1:
                    o.bits = ternary.src1;
                    dt = ternary.src1Type;
                    vs = ternary.src1VS0 + (ternary.src1VS1 * 3);
                    break;
                case 2:
                    if (ternary.src2Imm) return false;
                    o.bits = ternary.src2;
                    dt = ternary.src2Type;
                    break;
                default: return false;
            }
            dt |= (ternary.execType << 3);
            if (o.direct.regFile == RegFileARF) return false;
            if (op == Opcode::madm) o.direct.subRegNum = 0;
            auto base = GRF(o.direct.regNum).retype(decodeRegTypecode12(dt));
            auto sub = base[o.direct.subRegNum / getBytes(base.getType())];
            auto hs = (1 << o.direct.hs);
            if (opNum >= 0) hs >>= 1;
            if ((opNum < 0) || (opNum == 2))
                rd = sub(hs);
            else
                rd = sub((1 << vs) >> 1, hs);
            break;
        }
        default: {    // unary/binary
            BinaryOperand12 o;
            unsigned dt;
            switch (opNum) {
                case -1:
                    o.bits = binary.dst;
                    dt = binary.dstType;
                    break;
                case 0:
                    if (binary.src0Imm) return false;
                    o.bits = binary.src0;
                    dt = binary.src0Type;
                    break;
                case 1:
                    if (binary.src0Imm || binary.src1Imm) return false;
                    o.bits = binary.src1;
                    dt = binary.src1Type;
                    break;
                default: return false;
            }
            if (o.direct.addrMode) { region = DependencyRegion(); return true; } // indirect
            if (o.direct.regFile == RegFileARF) return false;
            if (isMathMacro())
                o.direct.subRegNum = 0;
            auto base = GRF(o.direct.regNum).retype(decodeRegTypecode12(dt));
            auto sub = base[o.direct.subRegNum / getBytes(base.getType())];
            auto hs = (1 << o.direct.hs) >> 1;
            if (opNum < 0)
                rd = sub(hs);
            else
                rd = sub((1 << o.direct.vs) >> 1, 1 << o.direct.width, hs);
            break;
        }
    }

    auto esize = 1 << common.execSize;
    rd.fixup(esize, DataType::invalid, opNum < 0, 2);
    region = DependencyRegion(esize, rd);
    return true;
}

bool Instruction12::getImm32(uint32_t &imm) const
{
    // Only need to support sync.allrd/wr.
    if (binary.src0Imm)
        imm = imm32.value;
    return binary.src0Imm;
}

bool Instruction12::getSendDesc(MessageDescriptor &desc) const
{
    if (!send.descIsReg)
        desc.all = send.desc0_10 | (send.desc11_19 << 11) | (send.desc20_24 << 20)
                                 | (send.desc25_29 << 25) | (send.desc30_31 << 30);
    return !send.descIsReg;
}

bool Instruction12::getARFType(ARFType &arfType, int opNum) const
{
    if (opNum > 1) return false;

    // Only need to support unary/binary, for detecting ce/cr/sr usage.
    switch (opcode()) {
        case Opcode::nop:
        case Opcode::illegal:
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::bfe:
        case Opcode::bfi2:
        case Opcode::csel:
        case Opcode::mad:
        case Opcode::madm:
        case Opcode::dp4a:
            return false;
        default: {
            BinaryOperand12 o;
            switch (opNum) {
                case -1:
                    o.bits = binary.dst;
                    break;
                case 0:
                    if (binary.src0Imm) return false;
                    o.bits = binary.src0;
                    break;
                case 1:
                    if (binary.src0Imm || binary.src1Imm) return false;
                    o.bits = binary.src1;
                    break;
                default: return false;
            }
            if (o.direct.addrMode) return false;
            if (o.direct.regFile != RegFileARF) return false;
            arfType = static_cast<ARFType>(o.direct.regNum >> 4);
            return true;
        }
    }
}

autoswsb::DestinationMask Instruction12::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(opcode())) {
        if (opcode() == Opcode::send || opcode() == Opcode::sendc)
            if (send.eot)
                return DestNone;
        return DestNextIP;
    }

    DestinationMask mask = DestNextIP;
    switch (opcode()) {
        case Opcode::ret:
        case Opcode::endif:
        case Opcode::while_:
        case Opcode::call:
        case Opcode::calla:
        case Opcode::join:
        case Opcode::jmpi:
        case Opcode::brd:
            mask = binary.src0Imm ? (DestNextIP | DestJIP) : DestUnknown; break;
        case Opcode::goto_:
        case Opcode::if_:
        case Opcode::else_:
        case Opcode::break_:
        case Opcode::cont:
        case Opcode::halt:
        case Opcode::brc:
            mask = binary.src0Imm ? (DestNextIP | DestJIP | DestUIP) : DestUnknown; break;
        default: break;
    }

    if ((opcode() == Opcode::jmpi) && !predicated())
        mask &= ~DestNextIP;

    if (mask & DestJIP) jip = branches.jip / sizeof(Instruction12);
    if (mask & DestUIP) uip = branches.uip / sizeof(Instruction12);

    return mask;
}
