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

// Gen8-11 binary encoding implementation.

// 25 bits of data common between src0 and src1.
union BinaryOperand8 {
    uint32_t bits;
    struct {
        unsigned chanSel03 : 4;     // chanEn for dst
        unsigned subRegNum4 : 1;
        unsigned regNum : 8;
        unsigned srcMod : 2;
        unsigned addrMode : 1;
        unsigned chanSel47 : 4;
        unsigned _ : 1;
        unsigned vs : 4;
        unsigned : 7;
    } direct16;
    struct {
        unsigned subRegNum : 5;
        unsigned regNum : 8;
        unsigned srcMod : 2;
        unsigned addrMode : 1;
        unsigned hs : 2;            // hs for dst
        unsigned width : 3;
        unsigned vs : 4;
        unsigned : 7;
    } direct1;
    struct {
        unsigned chanSel03 : 4;     // chanEn for dst
        unsigned addrImm48 : 5;
        unsigned addrSubreg : 4;
        unsigned srcMod : 2;
        unsigned addrMode : 1;
        unsigned chanSel47 : 4;
        unsigned _ : 1;
        unsigned vs : 4;
        unsigned : 7;
    } indirect16;
    struct {
        unsigned addrImm08 : 9;
        unsigned addrSubreg : 4;
        unsigned srcMod : 2;
        unsigned addrMode : 1;      // hs for dst
        unsigned hs : 2;
        unsigned width : 3;
        unsigned vs : 4;
        unsigned : 7;
    } indirect1;
};

// Ternary operands: 21 bits each.
union TernaryOperand8 {
    uint32_t bits;
    struct {
        unsigned type : 3;
        unsigned vs : 2;
        unsigned hs : 2;
        unsigned subRegNum : 5;
        unsigned regNum : 8;
        unsigned : 1;
        //
        unsigned : 11;
    } direct1;
    struct {
        unsigned repCtrl : 1;
        unsigned chanSel : 8;
        unsigned subReg2_4 : 3;
        unsigned regNum : 8;
        unsigned subReg1 : 1;
        unsigned : 11;
    } direct16;
    struct {
        unsigned type : 3;
        unsigned value : 16;
        unsigned : 2;
        unsigned : 11;
    } immediate1;
};

union Instruction8 {
    struct {                            // Lower 35 bits are essentially common.
        unsigned opcode : 8;
        unsigned accessMode : 1;
        unsigned depCtrl : 2;
        unsigned execOffset : 3;
        unsigned threadCtrl : 2;
        unsigned predCtrl : 4;
        unsigned predInv : 1;
        unsigned execSize : 3;
        unsigned cmod : 4;              // FC for math, SFID for send, zero for branches
        unsigned accWrCtrl : 1;         // aka branchCtrl, noSrcDepSet
        unsigned cmptCtrl : 1;
        unsigned debugCtrl : 1;
        unsigned saturate : 1;
        //
        unsigned flagSubRegNum : 1;
        unsigned flagRegNum : 1;
        unsigned maskCtrl : 1;
        unsigned : 29;
        //
        unsigned : 32;
        unsigned : 32;
    } common;
    struct {
        unsigned : 32;
        //
        unsigned : 3;
        unsigned dstRegFile : 2;
        unsigned dstType : 4;
        unsigned src0RegFile : 2;
        unsigned src0Type : 4;
        unsigned dstAddrImm9 : 1;       // indirect only
        unsigned dst : 16;              // first 16 bits of BinaryOperand8
        //
        unsigned src0 : 25;
        unsigned src1RegFile : 2;
        unsigned src1Type : 4;
        unsigned src0AddrImm9 : 1;
        //
        unsigned src1 : 25;
        unsigned src1AddrImm9 : 1;      // indirect only
        unsigned _ : 6;
    } binary;
    struct {
        uint64_t _;
        uint32_t __;
        uint32_t value;
    } imm32;
    struct {
        uint64_t _;
        uint64_t value;
    } imm64;
    struct {
        unsigned : 32;                  // common
        unsigned : 3;
        unsigned execDataType : 1;
        unsigned dstRegFile : 1;
        unsigned src0Mod : 2;
        unsigned src1Mod : 2;
        unsigned src2Mod : 2;
        unsigned src0RegFile : 1;
        unsigned src1RegFile : 1;
        unsigned src2RegFile : 1;
        unsigned dstType : 3;
        unsigned dstHS : 1;
        unsigned : 2;
        unsigned dstSubRegNum : 4;
        unsigned dstRegNum : 8;
        //
        unsigned src0 : 21;
        unsigned src1L : 11;
        //
        unsigned src1H : 10;
        unsigned src2 : 21;
        unsigned _ : 1;
    } ternary1;
    struct {
        unsigned : 32;                  // common
        unsigned : 3;
        unsigned src2Type : 1;
        unsigned src1Type : 1;
        unsigned src0Mod : 2;
        unsigned src1Mod : 2;
        unsigned src2Mod : 2;
        unsigned srcType : 3;
        unsigned dstType : 3;
        unsigned dstChanEn : 4;
        unsigned dstSubregNum2_4 : 3;
        unsigned dstRegNum : 8;
        //
        unsigned src0 : 21;
        unsigned src1L : 11;
        //
        unsigned src1H : 10;
        unsigned src2 : 21;
        unsigned _ : 1;
    } ternary16;
    struct {
        unsigned : 24;                  // common
        unsigned sfid : 4;
        unsigned noSrcDepSet : 1;
        unsigned : 3;                   // common
        //
        unsigned : 3;
        unsigned dstRegFile : 1;
        unsigned src1RegFile : 1;
        unsigned : 7;                   // common
        unsigned src1RegNum : 8;
        unsigned : 9;                   // common
        unsigned selReg32ExDesc : 1;
        unsigned dstAddrImm9 : 1;
        unsigned : 1;
        //
        unsigned exDesc6_9 : 4;
        unsigned : 9;                   // common
        unsigned selReg32Desc : 1;
        unsigned src0AddrImm9 : 1;
        unsigned : 1;
        unsigned exDesc16_31 : 16;      // reg: address subregister
        //
        unsigned desc : 31;             // reg?
        unsigned eot : 1;
    } sendsGen9;
    struct {                            // Differences between send and sends
        uint64_t _;
        unsigned exDesc16_19 : 4;
        unsigned : 12;
        unsigned exDesc20_23 : 4;
        unsigned zero : 1;
        unsigned exDesc24_27 : 4;
        unsigned : 2;
        unsigned exDesc28_31 : 4;
        unsigned : 1;
        //
        unsigned : 32;
    } sendGen8;
    struct {
        unsigned : 28;                  // common
        unsigned branchCtrl : 1;
        unsigned : 3;                   // common
        //
        unsigned : 32;
        unsigned uip : 32;
        unsigned jip : 32;
    } branches;
    uint64_t qword[2];

    constexpr Instruction8() : qword{0,0} {};
};

static_assert(sizeof(Instruction8) == 16, "Internal error: Instruction8 has been padded by the compiler.");

// Encoding routines.

static inline unsigned getImmediateTypecode8(DataType type)
{
    static const uint8_t conversionTable[16] = {0,1,2,3,2,3,10,7,8,9,11,0,0,4,6,5};
    return conversionTable[static_cast<unsigned>(type) & 0xF];
}

static inline unsigned getTernary16Typecode8(DataType type)
{
    // 0-4: :f, :d, :ud, :df, :hf
    static const uint8_t conversionTable[16] = {2,1,2,1,2,1,3,0,2,1,4,2,2,2,2,2};
    return conversionTable[static_cast<unsigned>(type) & 0xF];
}

static inline unsigned getTypecode11(DataType type)
{
    static const uint8_t conversionTable[16] = {0,1,2,3,4,5,10,9,6,7,8,0,0,4,5,11};
    return conversionTable[static_cast<unsigned>(type) & 0xF];
}

template <HW hw>
static inline unsigned getTypecode(DataType type)
{
    return static_cast<int>(type) & 0xF;
}

template <>
inline unsigned getTypecode<HW::Gen11>(DataType type)
{
    return getTypecode11(type);
}

template <HW hw>
static inline unsigned getImmediateTypecode(DataType type)
{
    return getImmediateTypecode8(type);
}

template <>
inline unsigned getImmediateTypecode<HW::Gen11>(DataType type)
{
    return getTypecode11(type);
}

template <bool dest>
static inline constexpr14 BinaryOperand8 encodeBinaryOperand8(const RegData &rd)
{
    BinaryOperand8 result{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        result.indirect1.addrImm08 = rd.getOffset() & 0x1FF;
        result.indirect1.addrMode = 1;
        result.indirect1.addrSubreg = rd.getIndirectOff();
        if (!dest) {
            result.indirect1.vs = (rd.isVxIndirect()) ? 0xFFFF :
                                    (rd.getVS() == 0) ? 0 :
                                                        (1 + utils::log2(rd.getVS()));
        }
    } else {
        result.direct1.subRegNum = rd.getByteOffset();
        result.direct1.regNum = rd.getBase();
        result.direct1.addrMode = 0;            // direct
        if (!dest)
            result.direct1.vs = (rd.getVS() == 0) ? 0 : (1 + utils::log2(rd.getVS()));
    }

    int hsEncoded = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));

    if (dest)
        result.direct1.srcMod = hsEncoded;
    else {
        result.direct1.hs = hsEncoded;
        result.direct1.width = utils::log2(rd.getWidth());
        result.direct1.srcMod = rd.getMods();
    }

    return result;
}

template <bool dest>
static inline constexpr14 BinaryOperand8 encodeBinaryOperand8(const ExtendedReg &reg)
{
    BinaryOperand8 result{0};

#ifdef NGEN_SAFE
    if (reg.isInvalid()) throw invalid_object_exception();
    if (reg.isIndirect()) throw invalid_operand_exception();
#endif

    RegData rd = reg.getBase();

    result.direct1.subRegNum = reg.getMMENum();
    result.direct1.regNum = rd.getBase();
    result.direct1.addrMode = 0;

    int hsEncoded = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));

    if (dest)
        result.direct1.srcMod = hsEncoded;
    else {
        result.direct1.hs = hsEncoded;
        result.direct1.width = utils::log2(rd.getWidth());
        result.direct1.srcMod = rd.getMods();
        result.direct1.vs = (rd.getVS() == 0) ? 0 : (1 + utils::log2(rd.getVS()));
    }

    return result;
}

template <bool dest>
static inline constexpr14 BinaryOperand8 encodeBinaryOperand8(const Align16Operand &op)
{
    BinaryOperand8 result{0};
    auto &rd = op.getReg();

#ifdef NGEN_SAFE
    if (op.isInvalid()) throw invalid_object_exception();
#endif

    if (op.getReg().isIndirect()) {
        result.indirect16.addrImm48 = rd.getOffset() >> 4;
        result.indirect16.addrSubreg = rd.getIndirectOff();
        result.indirect16.addrMode = 1;     // indirect
    } else {
        result.direct16.subRegNum4 = rd.getByteOffset() >> 4;
        result.direct16.regNum = rd.getBase();
        result.direct16.addrMode = 0;       // direct
    }

    if (dest)
        result.direct16.chanSel03 = op.getChanEn();
    else {
        result.direct16.srcMod = rd.getMods();
        result.direct16.chanSel03 = op.getChanSel() & 0xF;
        result.direct16.chanSel47 = op.getChanSel() >> 4;
        result.direct16.vs = (rd.getVS() == 0) ? 0 :
                          (rd.getBytes() == 8) ? 2 : 3;
    }

    return result;
}

template <bool src2>
static inline constexpr14 TernaryOperand8 encodeTernarySrcOperand8(const RegData &rd)
{
#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand8 result{0};

    result.direct1.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));
    if (!src2)
        result.direct1.vs = (rd.getVS() == 0) ? 0 : utils::log2(rd.getVS());
    result.direct1.regNum = rd.getBase();
    result.direct1.subRegNum = rd.getByteOffset();
    result.direct1.type = getTypecode11(rd.getType());

    return result;
}

template <bool src2>
static inline constexpr14 TernaryOperand8 encodeTernarySrcOperand8(const ExtendedReg &reg)
{
#ifdef NGEN_SAFE
    if (reg.isInvalid()) throw invalid_object_exception();
    if (reg.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand8 result{0};

    RegData rd = reg.getBase();

    result.direct1.hs = (rd.getHS() == 0) ? 0 : (1 + utils::log2(rd.getHS()));
    if (!src2)
        result.direct1.vs = (rd.getVS() == 0) ? 0 : utils::log2(rd.getVS());
    result.direct1.regNum = rd.getBase();
    result.direct1.subRegNum = reg.getMMENum() << 1;
    result.direct1.type = getTypecode11(rd.getType());

    return result;
}

template <bool src2>
static inline constexpr14 TernaryOperand8 encodeTernarySrcOperand8(const Align16Operand &rd)
{
#ifdef NGEN_SAFE
    if (rd.getReg().isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand8 result{0};

    result.direct16.chanSel = rd.getChanSel();
    result.direct16.regNum = rd.getReg().getBase();
    result.direct16.repCtrl = rd.isRep();

    int sr = rd.getReg().getByteOffset();
    result.direct16.subReg2_4 = sr >> 2;
    result.direct16.subReg1 = (sr >> 1) & 1;

    return result;
}

template <bool src2>
static inline constexpr14 TernaryOperand8 encodeTernarySrcOperand8(const Immediate &imm)
{
#ifdef NGEN_SAFE
    if (getBytes(imm.getType()) != 2)
        throw invalid_operand_exception();
#endif

    TernaryOperand8 result{0};

    result.immediate1.type = getTypecode11(imm.getType());
    result.immediate1.value = static_cast<uint64_t>(imm);

    return result;
}

template <typename S0, typename S1, typename S2>
static inline void encodeTernaryCommon8(Instruction8 &i, S0 src0, S1 src1, S2 src2)
{
    i.ternary16.src0Mod = src0.getMods();
    i.ternary16.src1Mod = src1.getMods();
    i.ternary16.src2Mod = src2.getMods();

    uint64_t src0bits = encodeTernarySrcOperand8<false>(src0).bits;
    uint64_t src1bits = encodeTernarySrcOperand8<false>(src1).bits;
    uint64_t src2bits = encodeTernarySrcOperand8<true>(src2).bits;

    // Manually encode upper qword because src1 crosses 32-bit boundary.
    i.qword[1] = (src2bits << 42) | (src1bits << 21) | src0bits;
}

static inline void encodeTernary1Dst10(Instruction8 &i, const RegData &dst)
{
    int dtype = getTypecode11(dst.getType());
    i.ternary1.execDataType = dtype >> 3;
    i.ternary1.dstType = dtype;
    i.ternary1.dstRegFile = dst.isARF();
    i.ternary1.dstRegNum = dst.getBase();
    i.ternary1.dstSubRegNum = dst.getByteOffset() >> 1;
}

static inline void encodeTernary1Dst10(Instruction8 &i, const ExtendedReg &dst)
{
    int dtype = getTypecode11(dst.getType());
    i.ternary1.execDataType = dtype >> 3;
    i.ternary1.dstType = dtype;
    i.ternary1.dstRegFile = dst.isARF();
    i.ternary1.dstRegNum = dst.getBase().getBase();
    i.ternary1.dstSubRegNum = dst.getMMENum();
}

static inline void encodeCommon8(Instruction8 &i, Opcode opcode, const InstructionModifier &mod)
{
    i.qword[0] = (mod.getAll() & ~0xFF) | static_cast<unsigned>(opcode);
}

static inline void encodeSendsExDesc(Instruction8 &i, uint32_t exdesc)
{
    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendsGen9.exDesc6_9 = (exdesc >> 6) & 0xF;
    i.sendsGen9.exDesc16_31 = (exdesc >> 16) & 0xFFFF;

    i.sendsGen9.selReg32ExDesc = false;
    i.sendsGen9.eot = (exdesc >> 5) & 1;
}

static inline void encodeSendsExDesc(Instruction8 &i, RegData exdesc)
{
#ifdef NGEN_SAFE
    // Only a0.x:ud is allowed for extended descriptor.
    if (!exdesc.isARF() || exdesc.getARFType() != ARFType::a || exdesc.getARFBase() != 0 || exdesc.getType() != DataType::ud)
        throw invalid_arf_exception();
#endif
    i.sendsGen9.selReg32ExDesc = true;
    i.sendsGen9.eot = false;                // No support for EOT with register exdesc currently.
    i.sendsGen9.exDesc16_31 = exdesc.getOffset();
}

static inline void encodeSendsDesc(Instruction8 &i, uint32_t desc)
{
    i.sendsGen9.desc = desc;
    i.sendsGen9.selReg32Desc = false;
}

static inline void encodeSendsDesc(Instruction8 &i, RegData desc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    i.sendsGen9.desc = desc.getOffset();
    i.sendsGen9.selReg32Desc = true;
}

static inline constexpr14 Align16Operand emulateAlign16Dst(const RegData &rd)
{
#ifdef NGEN_SAFE
    if (rd.getHS() != 1 || (rd.getVS() != rd.getWidth()))
        throw invalid_region_exception();
#endif
    return Align16Operand(rd, 0xF);
}

static inline constexpr14 Align16Operand emulateAlign16Src(const RegData &rd)
{
    // Try to emulate Align1 regioning with Align16. Fun stuff!
    auto hs = rd.getHS(), vs = rd.getVS(), width = rd.getWidth();

    if (hs == 0 && vs == 0) {
        // Broadcast, using RepCtrl. DF doesn't support repCtrl though;
        //  use swizzles to "emulate", like IGA does.
        if (rd.getType() == DataType::df) {
            auto shift = (rd.getOffset() & 1) << 1;
            RegData rdmod = rd;
            rdmod.setOffset(rdmod.getOffset() & ~1);

            return Align16Operand(rdmod, shift, shift + 1, shift, shift + 1);
        } else
            return Align16Operand::createBroadcast(rd);
    } else if (hs == 1 && vs == width) {
        // Unit stride. Trivial swizzle.
        return Align16Operand(rd, 0, 1, 2, 3);
    } else {
#ifdef NGEN_SAFE
        throw invalid_region_exception();
#else
        return Align16Operand(rd, 0, 1, 2, 3);
#endif
    }
}

static inline constexpr14 Align16Operand extToAlign16(const ExtendedReg &reg)
{
    return Align16Operand::createWithMME(reg.getBase(), reg.getMMENum());
}
