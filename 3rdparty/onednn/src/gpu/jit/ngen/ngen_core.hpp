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

#ifndef NGEN_CORE_HPP
#define NGEN_CORE_HPP


#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>

#include "ngen_utils.hpp"

#ifndef NGEN_NO_OP_NAMES
#if not +0
#error use -fno-operator-names [Linux/OS X] or without /Za [Windows] if you want to use and(), or(), xor(), not(), or define NGEN_NO_OP_NAMES and use and_(), or_(), xor_(), not().
#endif
#endif

#ifdef NGEN_ASM
#include <ostream>
#endif

#ifdef NGEN_SAFE
#include <stdexcept>
#endif

/*
  Syntax
  ------

  Register Syntax Overview
    r17                 Plain register
    r17.f(4)            -> r17.4:f
                        In fact, r17.4<0;1,0>:f, as subregisters default to
                          being scalar
    r17.sub<float>(4)   Same as above, allowing for C++ templating.
    r17.f()             -> r17.0:f (defaults to offset 0)
    r17.sub<float>()    Same as above
    r17.df(3)(8,8,1)    Register regioning (vertical stride, width, horizontal stride)
    r17.df(3)(8,1)      (Width, horiz. stride): vertical stride is inferred
    r17.df(3)(1)        Horizontal stride only: width, vertical stride inferred from execution size.
    r[a0.w(8)].f(4,4,1) Indirect addressing: VxH (if NGEN_SHORT_NAMES defined otherwise use indirect[a0...])
    r[a0.w(8)].f(4,1)   Indirect addressing: Vx1
    -r17.q(1)           Source modifier: negation
    abs(r17)            Source modifier: absolute value. Note that abs is defined in namespace ngen.
    -abs(r3)
    ~r17                Alternative syntax to -r17 for logical operations.
    r17 + 3             ...is r20. Operators ++ and += are defined similarly.

  Command Syntax Overview
    add(8, r3.f(0)(8,8,1), r9.f(0)(8,8,1), r12.f(0)(0,1,0))         ->   add (8) r3.0<8;8,1>:f r9.0<8;8,1>:f r12.f<0;1,0>
    add(8, r3.f(), r9.f(), r12.f())                                 Same as above. Register regions default to unit stride.
    add<float>(8, r3, r9, r12)                                      A default operand data type can be provided.
    add<uint32_t>(8, r3, r9, r12.uw(8)(0,1,0))                      Default operand types can be overridden.
    add<float>(8, r3, r9, 3.14159f)                                 The data type of scalar immediate values is inferred.
    add<int32_t>(8, r3, r9, int16_t(12))                            Here an int16_t immediate is mapped to the :w data type.
    mul<float>(8, r3, r9, Immediate::vf(-1.0,1.0,-1.0,1.25))        Vector immediates require helper functions.
    mov(8, r2.d(), Immediate::uv(7,6,5,4,3,2,1,0))
    mov(8, r2.d(), Immediate::v(7,-6,5,-4,3,-2,1,0))

  All modifiers for an instruction go in the first parameter, OR'ed together.
    add(8 | M0, ...)
    add(8 | W | ~f0.w(0) | sat, ...)            Use NoMask instead of W if NGEN_SHORT_NAMES not defined.
    add(8 | lt | f1_0, ...)
    add(8 | ~any2h | f1, ...)
 */

namespace ngen {

#ifdef NGEN_SAFE
static constexpr bool _safe_ = 1;
#else
static constexpr bool _safe_ = 0;
#endif

// Forward declarations.
class RegData;
class Register;
class Subregister;
class RegisterRegion;
class NullRegister;
class InstructionModifier;
union Instruction12;
enum class Opcode;

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod);

// Exceptions, used when NGEN_SAFE is defined.

#ifdef NGEN_SAFE
class invalid_type_exception : public std::runtime_error {
public:
    invalid_type_exception() : std::runtime_error("Instruction does not support this type or combination of types") {}
};
class invalid_object_exception : public std::runtime_error {
public:
    invalid_object_exception() : std::runtime_error("Object is invalid") {}
};
class invalid_immediate_exception : public std::runtime_error {
public:
    invalid_immediate_exception() : std::runtime_error("Invalid immediate value") {}
};
class invalid_modifiers_exception : public std::runtime_error {
public:
    invalid_modifiers_exception() : std::runtime_error("Invalid or conflicting modifiers") {}
};
class invalid_operand_exception : public std::runtime_error {
public:
    invalid_operand_exception() : std::runtime_error("Invalid operand to instruction") {}
};
class invalid_operand_count_exception : public std::runtime_error {
public:
    invalid_operand_count_exception() : std::runtime_error("Invalid operand count") {}
};
class invalid_arf_exception : public std::runtime_error {
public:
    invalid_arf_exception() : std::runtime_error("Invalid ARF specified") {}
};
class grf_expected_exception : public std::runtime_error {
public:
    grf_expected_exception() : std::runtime_error("GRF expected, but found an ARF") {}
};
class invalid_model_exception : public std::runtime_error {
public:
    invalid_model_exception() : std::runtime_error("Invalid addressing model specified") {}
};
class invalid_load_store_exception : public std::runtime_error {
public:
    invalid_load_store_exception() : std::runtime_error("Invalid operands for load/store/atomic") {}
};
class invalid_range_exception : public std::runtime_error {
public:
    invalid_range_exception() : std::runtime_error("Invalid register range") {}
};
class invalid_region_exception : public std::runtime_error {
public:
    invalid_region_exception() : std::runtime_error("Unsupported register region") {}
};
class missing_type_exception : public std::runtime_error {
public:
    missing_type_exception() : std::runtime_error("Operand is missing its type") {}
};
class read_only_exception : public std::runtime_error {
public:
    read_only_exception() : std::runtime_error("Memory model is read-only") {}
};
class stream_stack_underflow : public std::runtime_error {
public:
    stream_stack_underflow() : std::runtime_error("Stream stack underflow occurred") {}
};
class unfinished_stream_exception : public std::runtime_error {
public:
    unfinished_stream_exception() : std::runtime_error("An unfinished instruction stream is still active") {}
};
class dangling_label_exception : public std::runtime_error {
public:
    dangling_label_exception() : std::runtime_error("A label was referenced, but its location was not defined") {}
};
class multiple_label_exception : public std::runtime_error {
public:
    multiple_label_exception() : std::runtime_error("Label already has a location") {}
};
class unsupported_instruction : public std::runtime_error {
public:
    unsupported_instruction() : std::runtime_error("Instruction is not supported by the chosen hardware") {}
};
class iga_align16_exception : public std::runtime_error {
public:
    iga_align16_exception() : std::runtime_error("Align16 not supported by the IGA assembler; use binary output") {}
};
class sfid_needed_exception : public std::runtime_error {
public:
    sfid_needed_exception() : std::runtime_error("SFID must be specified on Gen12+") {}
};
class invalid_execution_size_exception : public std::runtime_error {
public:
    invalid_execution_size_exception() : std::runtime_error("Invalid execution size") {}
};
#endif

// Gen hardware generations.
enum class HW {
    Unknown,
    Gen9,
    Gen10,
    Gen11,
    Gen12LP,
};

// Data types. Bits[0:3] are the ID, bits[4:7] hold the width, in bytes.
enum class DataType : uint8_t {
    ud = 0x40,
    d  = 0x41,
    uw = 0x22,
    w  = 0x23,
    ub = 0x14,
    b  = 0x15,
    df = 0x86,
    f  = 0x47,
    uq = 0x88,
    q  = 0x89,
    hf = 0x2A,
    uv = 0x4D,
    v  = 0x4E,
    vf = 0x4F,
    invalid = 0x00
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, DataType type)
{
    static const char *names[16] = {"ud", "d", "uw", "w", "ub", "b", "df", "f", "uq", "q", "hf", "", "", "uv", "v", "vf"};
    str << names[static_cast<uint8_t>(type) & 0xF];
    return str;
}
#endif

static inline constexpr   int getBytes(DataType type)                  { return static_cast<int>(type) >> 4; }
static inline constexpr14 int getDwords(DataType type)                 { return std::max<int>(getBytes(type) >> 2, 1); }

static inline constexpr bool isSigned(DataType type)
{
    return !(type == DataType::ub || type == DataType::uw || type == DataType::ud || type == DataType::uq);
}

template <typename T> static inline DataType getDataType() { return DataType::invalid; }

template <> inline DataType getDataType<uint64_t>() { return DataType::uq; }
template <> inline DataType getDataType<int64_t>()  { return DataType::q;  }
template <> inline DataType getDataType<uint32_t>() { return DataType::ud; }
template <> inline DataType getDataType<int32_t>()  { return DataType::d;  }
template <> inline DataType getDataType<uint16_t>() { return DataType::uw; }
template <> inline DataType getDataType<int16_t>()  { return DataType::w;  }
template <> inline DataType getDataType<uint8_t>()  { return DataType::ub; }
template <> inline DataType getDataType<int8_t>()   { return DataType::b;  }
template <> inline DataType getDataType<double>()   { return DataType::df; }
template <> inline DataType getDataType<float>()    { return DataType::f;  }
#ifdef NGEN_HALF_TYPE
template <> inline DataType getDataType<half>()     { return DataType::hf; }
#endif

// Math function codes.
enum class MathFunction : uint8_t {
    inv   = 1,
    log   = 2,
    exp   = 3,
    sqt   = 4,
    rsqt  = 5,
    sin   = 6,
    cos   = 7,
    fdiv  = 9,
    pow   = 10,
    idiv  = 11,
    iqot  = 12,
    irem  = 13,
    invm  = 14,
    rsqtm = 15
};

static inline int mathArgCount(MathFunction func)
{
    static const char argCounts[16] = {0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 1};
    return argCounts[static_cast<uint8_t>(func) & 0xF];
}

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, MathFunction func)
{
    static const char *names[16] = {"", "inv", "log", "exp", "sqt", "rsqt", "sin", "cos", "", "fdiv", "pow", "idiv", "iqot", "irem", "invm", "rsqtm"};
    str << names[static_cast<uint8_t>(func) & 0xF];
    return str;
}
#endif

// Sync function codes.
enum class SyncFunction : uint8_t {
    nop   = 0,
    allrd = 2,
    allwr = 3,
    bar   = 14,
    host  = 15
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, SyncFunction func)
{
    static const char *names[16] = {"nop", "", "allrd", "allwr", "", "", "", "", "", "", "", "", "", "", "bar", "host"};
    str << names[static_cast<uint8_t>(func) & 0xF];
    return str;
}
#endif

// Shared function IDs (SFIDs).
enum class SharedFunction : uint8_t {
    null = 0,
    smpl = 2,
    sampler = 2,
    gtwy = 3,
    gateway = 3,
    dc2 = 4,
    rc = 5,
    urb = 6,
    ts = 7,
    spawner = 7,
    vme = 8,
    dcro = 9,
    dc0 = 10,
    pixi = 11,
    dc1 = 12,
    cre = 13
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, SharedFunction sfid)
{
    static const char *names[16] = {"null", "", "smpl", "gtwy", "dc2", "rc", "urb", "ts", "vme", "dcro", "dc0", "pixi", "dc1", "cre", "", ""};
    str << names[static_cast<uint8_t>(sfid) & 0xF];
    return str;
}
#endif

// ARFs: high nybble of register # specifies type
enum class ARFType : uint8_t {
    null = 0,
    a    = 1,
    acc  = 2,
    f    = 3,
    ce   = 4,
    msg  = 5,
    sp   = 6,
    sr   = 7,
    cr   = 8,
    n    = 9,
    ip   = 10,
    tdr  = 11,
    tm   = 12,
    fc   = 13,
    dbg  = 15,
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, ARFType type)
{
    static const char *names[16] = {"null", "a", "acc", "f", "ce", "msg", "sp", "sr", "cr", "n", "ip", "tdr", "tm", "fc", "", "dbg"};
    str << names[static_cast<uint8_t>(type) & 0xF];
    return str;
}

enum class PrintDetail {base = 0, sub_no_type = 1, sub = 2, hs = 3, vs_hs = 4, full = 5};
#endif

// Invalid singleton class. Can be assigned to nGEN objects to invalidate them.
static constexpr class Invalid {} invalid{};

class LabelManager {
protected:
    uint32_t nextID;
    std::vector<uint32_t> targets;

    enum TargetConstants : uint32_t {
        noTarget = uint32_t(-1),
    };

public:
    LabelManager() : nextID(0) {}

    uint32_t getNewID() {
        targets.push_back(TargetConstants::noTarget);
        return nextID++;
    }

    void setTarget(uint32_t id, uint32_t target) {
#ifdef NGEN_SAFE
        if (targets[id] != TargetConstants::noTarget) throw multiple_label_exception();
#endif
        targets[id] = target;
    }

    void offsetTarget(uint32_t id, uint32_t offset) {
#ifdef NGEN_SAFE
        if (targets[id] == TargetConstants::noTarget) throw dangling_label_exception();
#endif
        targets[id] += offset;
    }

    uint32_t getTarget(uint32_t id) const {
#ifdef NGEN_SAFE
        if (targets[id] == TargetConstants::noTarget) throw dangling_label_exception();
#endif
        return targets[id];
    }
};

// An object representing a label.
class Label {
protected:
    unsigned id : 31;
    unsigned uninit : 1;

public:
    Label() : id(0), uninit(true) {}

    uint32_t getID(LabelManager &man) {
        if (uninit) {
            id = man.getNewID();
            uninit = false;
        }
        return id;
    }

    /* for compatibility with RegData */
    void fixup(int execSize, DataType defaultType, bool isDest, int arity) {}
    constexpr14 bool isScalar() const { return false; }

#ifdef NGEN_ASM
    static const bool emptyOp = false;
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man);
#endif
};

static inline bool operator==(const RegData &r1, const RegData &r2);
static inline bool operator!=(const RegData &r1, const RegData &r2);

// Superclass for registers, subregisters, and register regions, possibly
// with source modifiers.
class RegData {
protected:
    unsigned base : 8;
    unsigned arf : 1;
      signed off : 10;
    unsigned mods : 2;
    unsigned type : 8;
    unsigned indirect : 1;
    unsigned _pad1 : 2;
    unsigned vs : 7;
    unsigned width : 5;
    unsigned hs : 6;
    unsigned _pad2 : 13;
    unsigned invalid : 1;

    constexpr RegData(int base_, bool arf_, int off_, bool indirect_, DataType type_, int vs_, int width_, int hs_)
        : base(base_), arf(arf_), off(off_), mods(0), type(static_cast<int>(type_)), indirect(indirect_), _pad1(0), vs(vs_), width(width_), hs(hs_), _pad2(0), invalid(0) {}

public:
#ifdef NGEN_ASM
    static const bool emptyOp = false;
#endif

    constexpr RegData()
        : base(0), arf(0), off(0), mods(0), type(0), indirect(0), _pad1(0), vs(0), width(0), hs(0), _pad2(0), invalid(1) {}

    constexpr int getBase()         const { return base; }
    constexpr bool isARF()          const { return arf; }
    constexpr int getARFBase()      const { return base & 0xF; }
    constexpr ARFType getARFType()  const { return static_cast<ARFType>(base >> 4); }
    constexpr bool isIndirect()     const { return indirect; }
    constexpr bool isVxIndirect()   const { return indirect && (vs == 0x7F); }
    constexpr int getIndirectBase() const { return base >> 4; }
    constexpr int getIndirectOff()  const { return base & 0xF; }
    constexpr bool isNull()         const { return isARF() && (getARFType() == ARFType::null); }
    constexpr bool isInvalid()      const { return invalid; }
    constexpr bool isValid()        const { return !invalid; }
    constexpr int getOffset()       const { return off; }
    constexpr int getByteOffset()   const { return off * getBytes(); }
    constexpr DataType getType()    const { return static_cast<DataType>(type); }
    constexpr int getVS()           const { return vs; }
    constexpr int getWidth()        const { return width; }
    constexpr int getHS()           const { return hs; }
    constexpr bool getNeg()         const { return mods & 2; }
    constexpr bool getAbs()         const { return mods & 1; }
    constexpr int getMods()         const { return mods; }
    constexpr int getBytes()        const { return ngen::getBytes(getType()); }
    constexpr14 int getDwords()     const { return ngen::getDwords(getType()); }
    constexpr bool isScalar()       const { return hs == 0 && vs == 0 && width == 1; }

    constexpr14 RegData &setBase(int base_)                      { base = base_; return *this; }
    constexpr14 RegData &setOffset(int off_)                     { off = off_; return *this; }
    constexpr14 RegData &setType(DataType newType)               { type = static_cast<unsigned>(newType); return *this; }
    constexpr14 RegData &setMods(int mods_)                      { mods = mods_; return *this; }
    constexpr14 RegData &setRegion(int vs_, int width_, int hs_) { vs = vs_; width = width_; hs = hs_; return *this; }

    void invalidate()                     { invalid = true; }
    RegData &operator=(const Invalid &i)  { this->invalidate(); return *this; }

    inline void fixup(int execSize, DataType defaultType, bool isDest, int arity);                    // Adjust automatically-computed strides given ESize.

    constexpr RegData operator+() const { return *this; }
    constexpr14 RegData operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 RegData operator~() const { return -*this; }
    constexpr14 void negate()             { mods = mods ^ 2; }

    friend inline bool operator==(const RegData &r1, const RegData &r2);
    friend inline bool operator!=(const RegData &r1, const RegData &r2);

    friend inline RegData abs(const RegData &r);

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

static_assert(sizeof(RegData) == 8, "RegData structure is not laid out correctly in memory.");

static inline bool operator==(const RegData &r1, const RegData &r2) {
    return *((uint64_t *) &r1) == *((uint64_t *) &r2);
}

static inline bool operator!=(const RegData &r1, const RegData &r2) {
    return !(r1 == r2);
}

inline RegData abs(const RegData &r)
{
    RegData result = r;
    return result.setMods(1);
}

inline void RegData::fixup(int execSize, DataType defaultType, bool isDest, int arity)
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

    if (getType() == DataType::invalid) {
#ifdef NGEN_SAFE
        if (defaultType == DataType::invalid)
            throw missing_type_exception();
#endif
        setType(defaultType);
    }
    if (execSize == 1) {
        vs = hs = 0;
        width = 1;
    } else if (width == 0) {
        int maxWidth = 32 / getBytes();
        width = (hs == 0) ? 1 : std::min<int>({int(maxWidth / hs), execSize, 16});
        vs = width * hs;
    }
    if (isDest && hs == 0)
        hs = 1;
}

// Operands for Align16 instructions
class Align16Operand {
protected:
    RegData rd;
    unsigned chanSel : 8;
    unsigned chanEn : 4;
    bool rep : 1;

public:
    constexpr Align16Operand(RegData rd_, int chanEn_) : rd(rd_), chanSel(0b11100100), chanEn(chanEn_), rep(false) {}
    constexpr Align16Operand(RegData rd_, int s0, int s1, int s2, int s3) : rd(rd_),
        chanSel((s0 & 3) | ((s1 & 3) << 2) | ((s2 & 3) << 4) | ((s3 & 3) << 6)), chanEn(0xF), rep(false) {}

    static constexpr14 Align16Operand createBroadcast(RegData rd_) {
        Align16Operand op{rd_, 0xF};
        op.rep = true;
        return op;
    }

    static constexpr14 Align16Operand createWithMME(RegData rd_, int mme) {
        Align16Operand op{rd_, mme};
        op.chanSel = mme;
        return op;
    }

    RegData &getReg()                           { return rd; }
    constexpr const RegData &getReg()     const { return rd; }
    constexpr uint8_t getChanSel()        const { return chanSel; }
    constexpr uint8_t getChanEn()         const { return chanEn; }
    constexpr bool isRep()                const { return rep; }

    constexpr bool isIndirect()           const { return rd.isIndirect(); }
    constexpr DataType getType()          const { return rd.getType(); }
    constexpr int getOffset()             const { return rd.getOffset(); }
    constexpr int getMods()               const { return rd.getMods(); }
    constexpr bool isARF()                const { return rd.isARF(); }

    void invalidate() { rd.invalidate(); }
    Align16Operand &operator=(const Invalid &i) { this->invalidate(); return *this; }
    bool isInvalid()                      const { return rd.isInvalid(); }
    bool isValid()                        const { return !rd.isInvalid(); }
    constexpr bool isScalar()             const { return rd.isScalar(); }

    void fixup(int execSize, DataType defaultType, bool isDest, int arity) {
        rd.fixup(execSize, defaultType, isDest, arity);
    }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
    static const bool emptyOp = false;
#endif
};

// Register regions.
class RegisterRegion : public RegData
{
public:
    constexpr RegisterRegion() : RegData() {}
    constexpr14 RegisterRegion(RegData rdata_, int vs_, int width_, int hs_) {
        *static_cast<RegData *>(this) = rdata_;
        vs = vs_;
        width = width_;
        hs = hs_;
    }

    RegisterRegion &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr RegisterRegion operator+() const { return *this; }
    constexpr14 RegisterRegion operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 RegisterRegion operator~() const { return -*this; }
};

// Subregister; always associated with a specific data type.
class Subregister : public RegData
{
protected:
    void checkGRF() const {
#ifdef NGEN_SAFE
        if (isARF()) throw grf_expected_exception();
#endif
    }

public:
    constexpr Subregister() : RegData() {}
    constexpr14 Subregister(RegData reg_, int offset_, DataType type_) {
        *static_cast<RegData *>(this) = reg_;
        off = offset_;
        type = static_cast<int>(type_);
        hs = vs = 0;
        width = 1;
    }
    constexpr14 Subregister(RegData reg_, DataType type_) {
        *static_cast<RegData *>(this) = reg_;
        off = 0;
        type = static_cast<int>(type_);
    }

    inline RegisterRegion operator()(int vs, int width, int hs) const;
    inline RegisterRegion operator()(int vs, int hs) const;
    inline RegisterRegion operator()(int hs) const;

    Subregister &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr Subregister operator+() const { return *this; }
    constexpr14 Subregister operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 Subregister operator~() const { return -*this; }

    Align16Operand swizzle(int s0, int s1, int s2, int s3)    const { checkGRF(); return Align16Operand(*this, s0, s1, s2, s3); }
    Align16Operand broadcast()                                const { checkGRF(); return Align16Operand::createBroadcast(*this); }
    Align16Operand enable(bool c0, bool c1, bool c2, bool c3) const { checkGRF(); return Align16Operand(*this, (int(c3) << 3) | (int(c2) << 2) | (int(c1) << 1) | int(c0)); }
    Align16Operand noSwizzle()                                const { return swizzle(0, 1, 2, 3); }
    Align16Operand enableAll()                                const { return enable(true, true, true, true); }

    inline Subregister reinterpret(int offset, DataType type_) const;
    template <typename T> Subregister reinterpret(int offset = 0) const { return reinterpret(offset, getDataType<T>()); }

    inline Subregister offset(int off) const { return reinterpret(off, getType()); }

    Subregister uq(int offset = 0) const { return reinterpret(offset, DataType::uq); }
    Subregister  q(int offset = 0) const { return reinterpret(offset, DataType::q);  }
    Subregister ud(int offset = 0) const { return reinterpret(offset, DataType::ud); }
    Subregister  d(int offset = 0) const { return reinterpret(offset, DataType::d);  }
    Subregister uw(int offset = 0) const { return reinterpret(offset, DataType::uw); }
    Subregister  w(int offset = 0) const { return reinterpret(offset, DataType::w);  }
    Subregister ub(int offset = 0) const { return reinterpret(offset, DataType::ub); }
    Subregister  b(int offset = 0) const { return reinterpret(offset, DataType::b);  }
    Subregister df(int offset = 0) const { return reinterpret(offset, DataType::df); }
    Subregister  f(int offset = 0) const { return reinterpret(offset, DataType::f);  }
    Subregister hf(int offset = 0) const { return reinterpret(offset, DataType::hf); }
};

// Single register.
class Register : public RegData
{
public:
    constexpr Register() : RegData() {}
    constexpr Register(int reg_, bool arf_, DataType defaultType = DataType::invalid, int off_ = 0)
        : RegData(reg_, arf_, off_, false, defaultType, 0, 0, 1) {}

    constexpr Register operator+() const { return *this; }
    constexpr14 Register operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 Register operator~() const { return -*this; }

    constexpr14 Subregister sub(int offset, DataType type_)        const { return Subregister(*this, offset, type_); }
    template <typename T> constexpr14 Subregister sub(int offset)  const { return sub(offset, getDataType<T>()); }

    constexpr14 Register retype(DataType type_)         const { auto clone = *this; clone.setType(type_); return clone; }
    template <typename T> constexpr14 Register retype() const { return retype(getDataType<T>()); }

    constexpr14 Subregister uq(int offset) const { return sub(offset, DataType::uq); }
    constexpr14 Subregister  q(int offset) const { return sub(offset, DataType::q);  }
    constexpr14 Subregister ud(int offset) const { return sub(offset, DataType::ud); }
    constexpr14 Subregister  d(int offset) const { return sub(offset, DataType::d);  }
    constexpr14 Subregister uw(int offset) const { return sub(offset, DataType::uw); }
    constexpr14 Subregister  w(int offset) const { return sub(offset, DataType::w);  }
    constexpr14 Subregister ub(int offset) const { return sub(offset, DataType::ub); }
    constexpr14 Subregister  b(int offset) const { return sub(offset, DataType::b);  }
    constexpr14 Subregister df(int offset) const { return sub(offset, DataType::df); }
    constexpr14 Subregister  f(int offset) const { return sub(offset, DataType::f);  }
    constexpr14 Subregister hf(int offset) const { return sub(offset, DataType::hf); }

    constexpr14 Register uq() const { return retype(DataType::uq); }
    constexpr14 Register  q() const { return retype(DataType::q);  }
    constexpr14 Register ud() const { return retype(DataType::ud); }
    constexpr14 Register  d() const { return retype(DataType::d);  }
    constexpr14 Register uw() const { return retype(DataType::uw); }
    constexpr14 Register  w() const { return retype(DataType::w);  }
    constexpr14 Register ub() const { return retype(DataType::ub); }
    constexpr14 Register  b() const { return retype(DataType::b);  }
    constexpr14 Register df() const { return retype(DataType::df); }
    constexpr14 Register  f() const { return retype(DataType::f);  }
    constexpr14 Register hf() const { return retype(DataType::hf); }

    constexpr14 Subregister operator[](int offset) const { return sub(offset, getType()); }

    Register &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class GRF : public Register
{
public:
    GRF() : Register() {}
    explicit constexpr GRF(int reg_) : Register(reg_, false) {}

    constexpr GRF operator+() const { return *this; }
    constexpr14 GRF operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 GRF operator~() const { return -*this; }

    constexpr14 GRF retype(DataType type_)              const { auto clone = *this; clone.setType(type_); return clone; }
    template <typename T> constexpr14 Register retype() const { return retype(getDataType<T>()); }

    constexpr14 Subregister uq(int offset) const { return sub(offset, DataType::uq); }
    constexpr14 Subregister  q(int offset) const { return sub(offset, DataType::q);  }
    constexpr14 Subregister ud(int offset) const { return sub(offset, DataType::ud); }
    constexpr14 Subregister  d(int offset) const { return sub(offset, DataType::d);  }
    constexpr14 Subregister uw(int offset) const { return sub(offset, DataType::uw); }
    constexpr14 Subregister  w(int offset) const { return sub(offset, DataType::w);  }
    constexpr14 Subregister ub(int offset) const { return sub(offset, DataType::ub); }
    constexpr14 Subregister  b(int offset) const { return sub(offset, DataType::b);  }
    constexpr14 Subregister df(int offset) const { return sub(offset, DataType::df); }
    constexpr14 Subregister  f(int offset) const { return sub(offset, DataType::f);  }
    constexpr14 Subregister hf(int offset) const { return sub(offset, DataType::hf); }

    constexpr14 GRF uq() const { return retype(DataType::uq); }
    constexpr14 GRF  q() const { return retype(DataType::q);  }
    constexpr14 GRF ud() const { return retype(DataType::ud); }
    constexpr14 GRF  d() const { return retype(DataType::d);  }
    constexpr14 GRF uw() const { return retype(DataType::uw); }
    constexpr14 GRF  w() const { return retype(DataType::w);  }
    constexpr14 GRF ub() const { return retype(DataType::ub); }
    constexpr14 GRF  b() const { return retype(DataType::b);  }
    constexpr14 GRF df() const { return retype(DataType::df); }
    constexpr14 GRF  f() const { return retype(DataType::f);  }
    constexpr14 GRF hf() const { return retype(DataType::hf); }

    Align16Operand swizzle(int s0, int s1, int s2, int s3)    const { return Align16Operand(*this, s0, s1, s2, s3); }
    Align16Operand enable(bool c0, bool c1, bool c2, bool c3) const { return Align16Operand(*this, (int(c3) << 3) | (int(c2) << 2) | (int(c1) << 1) | int(c0)); }
    Align16Operand noSwizzle()                                const { return swizzle(0, 1, 2, 3); }
    Align16Operand enableAll()                                const { return enable(true, true, true, true); }

    GRF &operator=(const Invalid &i) { this->invalidate(); return *this; }

    GRF &operator+=(const int &inc) {
        base += inc;
        return *this;
    }

    GRF operator++(int i) {
        GRF old = *this;
        ++*this;
        return old;
    }

    GRF &operator++() {
        *this += 1;
        return *this;
    }

    GRF operator+(const int &a) const {
        GRF sum = *this;
        sum += a;
        return sum;
    }
};

class ARF : public Register
{
public:
    constexpr ARF() : Register() {}
    constexpr ARF(ARFType type_, int reg_, DataType defaultType = DataType::invalid, int off_ = 0)
        : Register((static_cast<int>(type_) << 4) | (reg_ & 0xF), true, defaultType, off_) {}

    ARF &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class NullRegister : public ARF
{
public:
    constexpr NullRegister() : ARF(ARFType::null, 0, DataType::ud) {}
};

class AddressRegister : public ARF
{
public:
    constexpr AddressRegister() : ARF() {}
    explicit constexpr AddressRegister(int reg_) : ARF(ARFType::a, reg_, DataType::uw) {}

    AddressRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class AccumulatorRegister : public ARF
{
public:
    constexpr AccumulatorRegister() : ARF() {}
    explicit constexpr AccumulatorRegister(int reg_) : ARF(ARFType::acc, reg_) {}

    AccumulatorRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class SpecialAccumulatorRegister : public AccumulatorRegister
{
    uint8_t mmeNum;

public:
    constexpr SpecialAccumulatorRegister() : AccumulatorRegister(), mmeNum(0) {}
    constexpr SpecialAccumulatorRegister(int reg_, int mmeNum_) : AccumulatorRegister(reg_), mmeNum(mmeNum_) {}

    static constexpr SpecialAccumulatorRegister createNoMME() { return SpecialAccumulatorRegister(0, 8); }

    constexpr uint8_t getMME() const { return mmeNum; }

    SpecialAccumulatorRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

// An "extended register" is a combination of a regular GRF and some extra accumulator bits, used for math macro operations.
class ExtendedReg {
    RegData base;
    uint8_t mmeNum;

public:
    constexpr ExtendedReg(RegData base_, uint8_t mmeNum_) : base(base_), mmeNum(mmeNum_) {}
    constexpr ExtendedReg(RegData base_, SpecialAccumulatorRegister acc) : base(base_), mmeNum(acc.getMME()) {}

    void fixup(int execSize, DataType defaultType, bool isDest, int arity) {
        base.fixup(execSize, defaultType, isDest, arity);
    }

    constexpr int getMods()         const { return base.getMods(); }
    constexpr DataType getType()    const { return base.getType(); }
    constexpr int getOffset()       const { return base.getOffset(); }
    constexpr bool isIndirect()     const { return base.isIndirect(); }
    constexpr bool isInvalid()      const { return base.isInvalid(); }
    constexpr bool isValid()        const { return !base.isInvalid(); }
    constexpr bool isScalar()       const { return base.isScalar(); }
    constexpr bool isARF()          const { return base.isARF(); }

    constexpr14 RegData &getBase()        { return base; }
    constexpr RegData getBase()     const { return base; }
    constexpr uint8_t getMMENum()   const { return mmeNum; }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
    static const bool emptyOp = false;
#endif
};

static inline ExtendedReg operator|(const RegData &base, const SpecialAccumulatorRegister &acc)
{
    return ExtendedReg(base, acc);
}

class FlagRegister : public ARF
{
public:
    constexpr FlagRegister() : ARF() {}
    explicit constexpr FlagRegister(int reg_, int off_ = 0) : ARF(ARFType::f, reg_, DataType::uw, off_) {}

    static FlagRegister createFromIndex(int index) {
        return FlagRegister(index >> 1, index & 1);
    }

    FlagRegister operator~() const {
        FlagRegister result = *this;
        result.mods = result.mods ^ 2;
        return result;
    }

    FlagRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr FlagRegister operator[](int offset) const { return FlagRegister(getARFBase(), getOffset() + offset); }

    int index() const { return (getARFBase() << 1) + getOffset(); }
};

class ChannelEnableRegister : public ARF
{
public:
    explicit constexpr ChannelEnableRegister(int reg_ = 0) : ARF(ARFType::ce, reg_, DataType::ud) {}
};

class StackPointerRegister : public ARF
{
public:
    explicit constexpr StackPointerRegister(int reg_ = 0) : ARF(ARFType::sp, reg_, DataType::uq) {}
};

class StateRegister : public ARF
{
public:
    explicit constexpr StateRegister(int reg_ = 0) : ARF(ARFType::sr, reg_, DataType::ud) {}
};

class ControlRegister : public ARF
{
public:
    explicit constexpr ControlRegister(int reg_ = 0) : ARF(ARFType::cr, reg_, DataType::ud) {}
};

class NotificationRegister : public ARF
{
public:
    explicit constexpr NotificationRegister(int reg_ = 0) : ARF(ARFType::n, reg_, DataType::ud) {}
};

class InstructionPointerRegister : public ARF
{
public:
    constexpr InstructionPointerRegister() : ARF(ARFType::ip, 0, DataType::ud) {}
};

class ThreadDependencyRegister : public ARF
{
public:
    explicit constexpr ThreadDependencyRegister(int reg_ = 0) : ARF(ARFType::tdr, reg_, DataType::uw) {}
};

class PerformanceRegister : public ARF
{
public:
    explicit constexpr PerformanceRegister(int reg_ = 0, int off_ = 0) : ARF(ARFType::tm, reg_, DataType::ud, off_) {}
};

class DebugRegister : public ARF
{
public:
    explicit constexpr DebugRegister(int reg_ = 0) : ARF(ARFType::dbg, reg_, DataType::ud) {}
};

inline RegisterRegion Subregister::operator()(int vs, int width, int hs) const
{
    RegisterRegion rr(*this, vs, width, hs);
    return rr;
}

inline RegisterRegion Subregister::operator()(int vs_or_width, int hs) const
{
    int vs, width;

    if (isIndirect()) {
        vs = -1;
        width = vs_or_width;
    } else {
        vs = vs_or_width;
        width = (hs == 0) ? ((vs == 0) ? 1 : vs) : vs / hs;
    }

    return operator()(vs, width, hs);
}

inline RegisterRegion Subregister::operator()(int hs) const
{
    return operator()(0, 0, hs);
}

inline Subregister Subregister::reinterpret(int offset, DataType type_) const
{
    Subregister r = *this;
    r.setType(type_);

    int o = getOffset();
    int oldbytes = getBytes(), newbytes = r.getBytes();
    int bitdiff = (oldbytes == 0) ? 0
                                  : (utils::log2(newbytes) - utils::log2(oldbytes));

    if (newbytes < oldbytes)
        r.setOffset((o << -bitdiff) + offset);
    else
        r.setOffset((o >>  bitdiff) + offset);

    return r;
}

// Indirect register and frames for making them.
class IndirectRegister : public Register {
protected:
    explicit constexpr14 IndirectRegister(const RegData &reg) : Register((reg.getARFBase() << 4) | reg.getOffset(), false) {
        indirect = true;
    }
    friend class IndirectRegisterFrame;

    IndirectRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class IndirectRegisterFrame {
public:
    IndirectRegister operator[](const RegData &reg) const {
#ifdef NGEN_SAFE
        if (!reg.isARF() || reg.getARFType() != ARFType::a)
            throw invalid_arf_exception();
#endif
        return IndirectRegister(reg);
    }
};

// GRFRange represents a contiguous range of GRF registers.
class GRFRange {
protected:
    uint8_t base;
    uint8_t len;

    static constexpr uint8_t invalidLen = 0xFF;

public:
    GRFRange() : GRFRange(0, invalidLen) {}
    GRFRange(int base_, int len_) : base(base_), len(len_) {}
    GRFRange(GRF base_, int len_) : GRFRange(base_.getBase(), len_) {}

    int getBase()    const { return base; }
    int getLen()     const { return len; }
    bool isEmpty()   const { return len == 0; }
    bool isNull()    const { return false; }

    void invalidate()      { len = invalidLen; }
    bool isInvalid() const { return len == invalidLen; }
    bool isValid()   const { return !isInvalid(); }

    GRFRange &operator=(const Invalid &i) { this->invalidate(); return *this; }

    GRF operator[](int i) const {
#ifdef NGEN_SAFE
        if (isInvalid()) throw invalid_object_exception();
#endif
        return GRF(base + i);
    }

    operator GRF() const { return (*this)[0]; }
};

static inline GRFRange operator-(const GRF &reg1, const GRF &reg2)
{
    uint8_t b1 = reg1.getBase(), b2 = reg2.getBase();
    int len = int(b2) + 1 - int(b1);

#ifdef NGEN_SAFE
    if (len < 0) throw invalid_range_exception();
#endif

    return GRFRange(reg1, len);
}

static inline bool operator==(const GRFRange &r1, const GRFRange &r2)
{
    return (r1.getBase() == r2.getBase()) && (r1.getLen() == r2.getLen());
}

static inline bool operator!=(const GRFRange &r1, const GRFRange &r2)
{
    return !(r1 == r2);
}

enum class ConditionModifier {
    none = 0,
    ze = 1,
    eq = 1,
    nz = 2,
    ne = 2,
    gt = 3,
    ge = 4,
    lt = 5,
    le = 6,
    ov = 8,
    un = 9,
    eo = 0xF
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, ConditionModifier cmod)
{
    static const char *names[16] = {"", "eq", "ne", "gt", "ge", "lt", "le", "", "ov", "un", "", "", "", "", "", "eo"};
    str << names[static_cast<uint8_t>(cmod) & 0xF];
    return str;
}
#endif

enum class ChannelMask {
    rgba = 0,
    gba = 1,
    rba = 2,
    ba = 3,
    rga = 4,
    bga = 5,
    ga = 6,
    a = 7,
    rgb = 8,
    gb = 9,
    rb = 10,
    b = 11,
    rg = 12,
    g = 13,
    r = 14
};

enum class PredCtrl {
    None = 0,
    Normal = 1,
    anyv = 2,
    allv = 3,
    any2h = 4,
    all2h = 5,
    any4h = 6,
    all4h = 7,
    any8h = 8,
    all8h = 9,
    any16h = 10,
    all16h = 11,
    any32h = 12,
    all32h = 13,
    x = 2,
    y = 3,
    z = 4,
    w = 5
};

#ifdef NGEN_ASM
static const char *toText(PredCtrl ctrl, bool align16) {
    const char *names[2][16] = {{"", "", "anyv", "allv", "any2h", "all2h", "any4h", "all4h", "any8h", "all8h", "any16h", "all16h", "any32h", "all32h", "", ""},
                                {"", "", "x",    "y",    "z",     "w",     "",      "",      "",      "",      "",       "",       "",       "",       "", ""}};
    return names[align16][static_cast<int>(ctrl) & 0xF];
}
#endif

enum class ThreadCtrl {
    Normal = 0,
    Atomic = 1,
    Switch = 2,
    NoPreempt = 3
};

enum class Opcode {
    illegal = 0x00,
    sync = 0x01,
    mov = 0x01,
    sel = 0x02,
    movi = 0x03,
    not_ = 0x04,
    and_ = 0x05,
    or_ = 0x06,
    xor_ = 0x07,
    shr = 0x08,
    shl = 0x09,
    smov = 0x0A,
    asr = 0x0C,
    ror = 0x0E,
    rol = 0x0F,
    cmp = 0x10,
    cmpn = 0x11,
    csel = 0x12,
    bfrev = 0x17,
    bfe = 0x18,
    bfi1 = 0x19,
    bfi2 = 0x1A,
    jmpi = 0x20,
    brd = 0x21,
    if_ = 0x22,
    brc = 0x23,
    else_ = 0x24,
    endif = 0x25,
    while_ = 0x27,
    break_ = 0x28,
    cont = 0x29,
    halt = 0x2A,
    calla = 0x2B,
    call = 0x2C,
    ret = 0x2D,
    goto_ = 0x2E,
    join = 0x2F,
    wait = 0x30,
    send = 0x31,
    sendc = 0x32,
    sends = 0x33,
    sendsc = 0x34,
    math = 0x38,
    add = 0x40,
    mul = 0x41,
    avg = 0x42,
    frc = 0x43,
    rndu = 0x44,
    rndd = 0x45,
    rnde = 0x46,
    rndz = 0x47,
    mac = 0x48,
    mach = 0x49,
    lzd = 0x4A,
    fbh = 0x4B,
    fbl = 0x4C,
    cbit = 0x4D,
    addc = 0x4E,
    subb = 0x4F,
    sad2 = 0x50,
    sada2 = 0x51,
    dp4 = 0x54,
    dph = 0x55,
    dp3 = 0x56,
    dp2 = 0x57,
    dp4a = 0x58,
    line = 0x59,
    pln = 0x5A,
    mad = 0x5B,
    lrp = 0x5C,
    madm = 0x5D,
    nop_gen12 = 0x60,
    mov_gen12 = 0x61,
    sel_gen12 = 0x62,
    movi_gen12 = 0x63,
    not_gen12 = 0x64,
    and_gen12 = 0x65,
    or_gen12 = 0x66,
    xor_gen12 = 0x67,
    shr_gen12 = 0x68,
    shl_gen12 = 0x69,
    smov_gen12 = 0x6A,
    asr_gen12 = 0x6C,
    ror_gen12 = 0x6E,
    rol_gen12 = 0x6F,
    cmp_gen12 = 0x70,
    cmpn_gen12 = 0x71,
    csel_gen12 = 0x72,
    bfrev_gen12 = 0x77,
    bfe_gen12 = 0x78,
    bfi1_gen12 = 0x79,
    bfi2_gen12 = 0x7A,
    nop = 0x7E,
};

static inline bool isVariableLatency(Opcode op)
{
    switch (op) {
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::math:
            return true;
        default:
            return false;
    }
}

static inline bool isBranch(Opcode op)
{
    return (static_cast<int>(op) >> 4) == 2;
}

#ifdef NGEN_ASM
static const char *getMnemonic(Opcode op, HW hw)
{
    const char *names[0x80] = {
        "illegal", "sync", "sel", "movi", "not", "and", "or", "xor",
        "shr", "shl", "smov", "", "asr", "", "ror", "rol",
        "cmp", "cmpn", "csel", "", "", "", "", "bfrev",
        "bfe", "bfi1", "bfi2", "", "", "", "", "",
        "jmpi", "brd", "if", "brc", "else", "endif", "", "while",
        "break", "cont", "halt", "calla", "call", "ret", "goto", "join",
        "wait", "send", "sendc", "sends", "sendsc", "", "", "",
        "math", "", "", "", "", "", "", "",
        "add", "mul", "avg", "frc", "rndu", "rndd", "rnde", "rndz",
        "mac", "mach", "lzd", "fbh", "fbl", "cbit", "addc", "subb",
        "sad2", "sada2", "", "", "dp4", "dph", "dp3", "dp2",
        "dp4a", "", "", "mad", "lrp", "madm", "", "",
        "nop", "mov", "sel", "movi", "not", "and", "or", "xor",
        "shr", "shl", "smov", "", "asr", "", "ror", "rol",
        "cmp", "cmpn", "csel", "", "", "", "", "bfrev",
        "bfe", "bfi1", "bfi2", "", "", "", "nop", ""
    };

    const char *mnemonic = names[static_cast<int>(op) & 0x7F];

    if (hw < HW::Gen12LP) switch (op) {
        case Opcode::mov:   mnemonic = "mov";   break;
        case Opcode::line:  mnemonic = "line";  break;
        case Opcode::pln:   mnemonic = "pln";   break;
        default: break;
    }

    return mnemonic;
}
#endif

class SBInfo
{
protected:
    unsigned id : 4;
    unsigned src : 1;
    unsigned dst : 1;

public:
    constexpr SBInfo(int id_) : id(id_), src(false), dst(false) {}
    constexpr SBInfo(int id_, bool src_, bool dst_) : id(id_), src(src_), dst(dst_) {}
    constexpr unsigned getID() const { return id; }
    constexpr bool isSrc() const { return src; }
    constexpr bool isDst() const { return dst; }

    constexpr unsigned getMode() const {
        return isDst() ? 0b010 : isSrc() ? 0b011 : 0b100;
    }
};

class SBID
{
public:
    SBInfo set;
    SBInfo src;
    SBInfo dst;

    constexpr SBID(int id) : set(id), src(id, true, false), dst(id, false, true) {}
    constexpr operator SBInfo() const { return set; }
};

class AllPipes {};
enum class Pipe {
    Default = 0,
    A = 1,
};

template <typename T> static constexpr Pipe getPipe() { return Pipe::Default; }
template <> constexpr Pipe getPipe<AllPipes>()        { return Pipe::A; }

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, Pipe pipe)
{
    if (pipe == Pipe::A) str << 'A';
    return str;
}
#endif

class SWSBInfo
{
    friend class InstructionModifier;
protected:
    union {
        struct {
            unsigned dist : 3;
            unsigned pipe : 4;
            unsigned combined : 1;
        } pipeline;
        struct {
            unsigned sbid : 4;
            unsigned mode : 3;
            unsigned combined : 1;
        } scoreboard;
        struct {
            unsigned sbid : 4;
            unsigned dist : 3;
            unsigned combined : 1;
        } combined;
        uint8_t all;
    };

    constexpr SWSBInfo(uint8_t all_, bool dummy) : all{all_} {}

    constexpr bool isPipeline() const {
        return !combined.combined && ((scoreboard.mode < 2) || (scoreboard.mode > 4));
    }

public:
    constexpr SWSBInfo() : all{0} {}
    constexpr SWSBInfo(SBInfo info) : scoreboard{info.getID(), info.getMode(), false} {}
    constexpr SWSBInfo(Pipe pipe, unsigned dist) : pipeline{dist,static_cast<unsigned>(pipe),false} {}
    constexpr SWSBInfo(SBInfo info, unsigned dist) : combined{info.getID(), dist, true} {}

    constexpr bool empty() const { return all == 0; }
    constexpr14 int dist() const {
        if (combined.combined)
            return combined.dist;
        else if (isPipeline())
            return pipeline.dist;
        else
            return 0;
    }
    constexpr bool hasSB() const { return !isPipeline(); }
    constexpr14 SBInfo sb() const {
        if (combined.combined)
            return SBInfo(combined.sbid);
        else if (isPipeline())
            return SBInfo(0);
        else
            return SBInfo(scoreboard.sbid, scoreboard.mode == 3, scoreboard.mode == 2);
    }
    constexpr14 Pipe pipe() const {
        if (combined.combined)
            return Pipe::A;
        else if (isPipeline())
            return static_cast<Pipe>(pipeline.pipe);
        else
            return Pipe::Default;
    }

    constexpr uint8_t raw() const { return all; }
    static constexpr14 SWSBInfo createFromRaw(uint8_t all_) { return SWSBInfo(all_, false); }
};

constexpr SWSBInfo SWSB(SBInfo info)                             { return SWSBInfo(info); }
template <typename T = void> constexpr SWSBInfo SWSB(int dist)   { return SWSBInfo(getPipe<T>(), dist); }
constexpr SWSBInfo SWSB(SBInfo info, int dist)                   { return SWSBInfo(info, dist); }

class InstructionModifier {
protected:
    union {
        struct {
            unsigned execSize : 8;          // Execution size as integer (for internal use).
            unsigned accessMode : 1;        // From here on matches the low 64-bits of the binary format for Gen8-11
            unsigned noDDClr : 1;
            unsigned noDDChk : 1;
            unsigned chanOff : 3;
            unsigned threadCtrl : 2;
            unsigned predCtrl : 4;
            unsigned predInv : 1;
            unsigned eSizeField : 3;
            unsigned cmod : 4;              // Also stores channel mask temporarily for surface r/w
            unsigned accWrCtrl : 1;         // = noSrcDepSet for send, = branchCtrl for branch instructions
            unsigned cmptCtrl : 1;
            unsigned debugCtrl : 1;
            unsigned saturate : 1;
            unsigned flagSubRegNum : 1;
            unsigned flagRegNum : 1;
            unsigned maskCtrl : 1;
            unsigned _zeros_: 18;
            unsigned autoSWSB : 1;
            unsigned fusionCtrl : 1;        // Gen12
            unsigned eot : 1;
            unsigned swsb : 8;
        } parts;
        uint64_t all;
    };

    constexpr InstructionModifier(uint64_t all_) : all(all_) {}

    friend inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod);

public:
    constexpr int getExecSize()            const { return parts.execSize; }
    constexpr bool isAlign16()             const { return parts.accessMode; }
    constexpr bool isNoDDClr()             const { return parts.noDDClr; }
    constexpr bool isNoDDChk()             const { return parts.noDDChk; }
    constexpr int getChannelOffset()       const { return parts.chanOff << 2; }
    constexpr ThreadCtrl getThreadCtrl()   const { return static_cast<ThreadCtrl>(parts.threadCtrl); }
    constexpr PredCtrl getPredCtrl()       const { return static_cast<PredCtrl>(parts.predCtrl); }
    constexpr bool isPredInv()             const { return parts.predInv; }
    constexpr ConditionModifier getCMod()  const { return static_cast<ConditionModifier>(parts.cmod); }
    constexpr bool isAccWrEn()             const { return parts.accWrCtrl; }
    constexpr bool isCompact()             const { return parts.cmptCtrl; }
    constexpr bool isBreakpoint()          const { return parts.debugCtrl; }
    constexpr bool isSaturate()            const { return parts.saturate; }
    constexpr14 FlagRegister getFlagReg()  const { return FlagRegister(parts.flagRegNum, parts.flagSubRegNum); }
    constexpr bool isWrEn()                const { return parts.maskCtrl; }
    constexpr bool isAutoSWSB()            const { return parts.autoSWSB; }
    constexpr bool isSerialized()          const { return parts.fusionCtrl; }
    constexpr bool isEOT()                 const { return parts.eot; }
    constexpr SWSBInfo getSWSB()           const { return SWSBInfo(parts.swsb, false); }
    constexpr uint64_t getAll()            const { return all; }

    constexpr14 void setExecSize(int execSize_)              { parts.execSize = execSize_; parts.eSizeField = utils::log2(execSize_); }
    constexpr14 void setPredInv(bool predInv_)               { parts.predInv = predInv_; }
    constexpr14 void setCMod(const ConditionModifier &cmod_) { parts.cmod = static_cast<unsigned>(cmod_); }
    constexpr14 void setBranchCtrl(bool branchCtrl)          { parts.accWrCtrl = branchCtrl; }
    constexpr14 void setWrEn(bool maskCtrl_)                 { parts.maskCtrl = maskCtrl_; }
    constexpr14 void setAutoSWSB(bool autoSWSB_)             { parts.autoSWSB = autoSWSB_; }
    constexpr14 void setSWSB(SWSBInfo swsb_)                 { parts.swsb = swsb_.raw(); }
    constexpr14 void setSWSB(uint8_t swsb_)                  { parts.swsb = swsb_; }

    constexpr InstructionModifier() : all(0) {}

    // Hardcoded shift counts are a workaround for MSVC v140 bug.
    constexpr /* implicit */ InstructionModifier(const PredCtrl &predCtrl_)
        : all{static_cast<uint64_t>(predCtrl_) << 16} {}

    constexpr /* implicit */ InstructionModifier(const ThreadCtrl &threadCtrl_)
        : all{static_cast<uint64_t>(threadCtrl_) << 14} {}

    constexpr /* implicit */ InstructionModifier(const ConditionModifier &cmod_)
        : all{static_cast<uint64_t>(cmod_) << 24} {}

    constexpr14 /* implicit */ InstructionModifier(const int &execSize_) : InstructionModifier() {
        setExecSize(execSize_);
    }
    constexpr14 /* implicit */ InstructionModifier(const SWSBInfo &swsb) : InstructionModifier() {
        parts.swsb = swsb.all;
    }
    constexpr14 /* implicit */ InstructionModifier(const SBInfo &sb) : InstructionModifier(SWSB(sb)) {}
    constexpr14 /* implicit */ InstructionModifier(const SBID &sb)   : InstructionModifier(SWSB(sb)) {}

protected:
    constexpr InstructionModifier(bool accessMode_, bool noDDClr_, bool noDDChk_, unsigned chanOff_, bool accWrCtrl_,
                                  bool debugCtrl_, bool saturate_, bool maskCtrl_, bool autoSWSB_, bool fusionCtrl_, bool eot_)
        : all{(uint64_t(accessMode_) << 8) | (uint64_t(noDDClr_) << 9) | (uint64_t(noDDChk_) << 10) | (uint64_t(chanOff_ >> 2) << 11)
            | (uint64_t(accWrCtrl_) << 28) | (uint64_t(debugCtrl_) << 30) | (uint64_t(saturate_) << 31)
            | (uint64_t(maskCtrl_) << 34) | (uint64_t(autoSWSB_) << 53) | (uint64_t(fusionCtrl_) << 54) | (uint64_t(eot_) << 55)} {}

public:
    static constexpr InstructionModifier createAccessMode(int accessMode_) {
        return InstructionModifier(accessMode_, false, false, 0, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createNoDDClr() {
        return InstructionModifier(false, true, false, 0, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createNoDDChk() {
        return InstructionModifier(false, false, true, 0, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createChanOff(int offset) {
        return InstructionModifier(false, false, false, offset, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createAccWrCtrl() {
        return InstructionModifier(false, false, false, 0, true, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createDebugCtrl() {
        return InstructionModifier(false, false, false, 0, false, true, false, false, false, false, false);
    }
    static constexpr InstructionModifier createSaturate() {
        return InstructionModifier(false, false, false, 0, false, false, true, false, false, false, false);
    }
    static constexpr InstructionModifier createMaskCtrl(bool maskCtrl_) {
        return InstructionModifier(false, false, false, 0, false, false, false, maskCtrl_, false, false, false);
    }
    static constexpr InstructionModifier createAutoSWSB() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, true, false, false);
    }
    static constexpr InstructionModifier createSerialized() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, false, true, false);
    }
    static constexpr InstructionModifier createEOT() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, false, false, true);
    }

    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const InstructionModifier &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const FlagRegister &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const PredCtrl &mod2);

    friend constexpr14 InstructionModifier operator^(const InstructionModifier &mod1, const InstructionModifier &mod2);

    constexpr14 InstructionModifier operator~() {
        InstructionModifier mod = *this;
        mod.parts.predInv = ~mod.parts.predInv;
        return mod;
    }

    template <typename T>
    InstructionModifier &operator|=(const T &mod) {
        *this = *this | mod;
        return *this;
    }

    InstructionModifier &operator^=(const InstructionModifier &mod) {
        *this = *this ^ mod;
        return *this;
    }
};

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const InstructionModifier &mod2)
{
    return InstructionModifier(mod1.all | mod2.all);
}


inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const FlagRegister &flag)
{
    InstructionModifier mod = mod1;

    mod.parts.flagRegNum = flag.getBase();
    mod.parts.flagSubRegNum = flag.getOffset();

    if (mod.getCMod() == ConditionModifier::none) {
        mod.parts.predInv = flag.getNeg();
        mod.parts.predCtrl = static_cast<int>(PredCtrl::Normal);
    }

    return mod;
}

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const PredCtrl &mod2)
{
    InstructionModifier mod = mod1;
    mod.parts.predCtrl = static_cast<int>(mod2);
    return mod;
}

inline constexpr14 InstructionModifier operator^(const InstructionModifier &mod1, const InstructionModifier &mod2)
{
    return InstructionModifier(mod1.all ^ mod2.all);
}

class Immediate {
protected:
    uint64_t payload;
    DataType type;
    bool hiddenType = false;

    Immediate(uint64_t payload_, DataType type_) : payload(payload_), type(type_) {}

    template <typename T> typename std::enable_if<sizeof(T) == 2>::type setPayload(T imm) {
        uint32_t ximm = utils::bitcast<T, uint16_t>(imm);
        payload = ximm | (ximm << 16);
    }
    template <typename T> typename std::enable_if<sizeof(T) == 4>::type setPayload(T imm) {
        payload = utils::bitcast<T, uint32_t>(imm);
    }
    template <typename T> typename std::enable_if<sizeof(T) == 8>::type setPayload(T imm) {
        payload = utils::bitcast<T, uint64_t>(imm);
    }

    template <typename T> void set(T imm) {
        setPayload<T>(imm);
        type = getDataType<T>();
    }

    template <typename T> void shrinkSigned(T imm) {
        if (imm == T(int16_t(imm)))       set<int16_t>(imm);
        else if (imm == T(uint16_t(imm))) set<uint16_t>(imm);
        else if (imm == T(int32_t(imm)))  set<int32_t>(imm);
        else if (imm == T(uint32_t(imm))) set<uint32_t>(imm);
        else                              set(imm);
    }

    template <typename T> void shrinkUnsigned(T imm) {
        if (imm == T(uint16_t(imm)))      set<uint16_t>(imm);
        else if (imm == T(uint32_t(imm))) set<uint32_t>(imm);
        else                              set(imm);
    }

public:
    Immediate() : payload(0), type(DataType::invalid) {}

#ifdef NGEN_ASM
    static const bool emptyOp = false;
#endif

    constexpr14 DataType getType()           const { return type; }
    explicit constexpr14 operator uint64_t() const { return payload; }
    constexpr14 int getMods()                const { return 0; }
    constexpr14 bool isARF()                 const { return false; }

    Immediate &setType(DataType type_)             { type = type_; return *this; }

    Immediate(uint16_t imm) { set(imm); }
    Immediate(int16_t  imm) { set(imm); }
    Immediate(uint32_t imm) { shrinkUnsigned(imm); }
    Immediate(int32_t imm)  { shrinkSigned(imm); }
    Immediate(uint64_t imm) { shrinkUnsigned(imm); }
    Immediate(int64_t imm)  { shrinkSigned(imm); }

    Immediate(float    imm) { set(imm); }
    Immediate(double   imm) { set(imm); }
#ifdef NGEN_HALF_TYPE
    Immediate(half     imm) { set(imm); }
#endif

    Immediate hideType() const {
        Immediate result = *this;
        result.hiddenType = true;
        return result;
    }

    static inline Immediate uw(uint16_t imm) { return Immediate(imm); }
    static inline Immediate  w(int16_t  imm) { return Immediate(imm); }
    static inline Immediate ud(uint32_t imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  d(int32_t  imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate uq(uint64_t imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  q(int64_t  imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  f(float    imm) { return Immediate(imm); }
    static inline Immediate df(double   imm) { return Immediate(imm); }

    static inline Immediate hf(uint16_t f) {
        uint32_t fimm = f;
        fimm |= (fimm << 16);
        return Immediate(fimm, DataType::hf);
    }

protected:
    static inline uint32_t toUV(int8_t i) {
#ifdef NGEN_SAFE
        if (i & 0xF0) throw invalid_immediate_exception();
#endif
        return i;
    }

public:
    static inline Immediate uv(uint32_t i) {
        return Immediate(i, DataType::uv);
    }

    static inline Immediate uv(uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3, uint8_t i4, uint8_t i5, uint8_t i6, uint8_t i7) {
        uint32_t payload = (toUV(i0) << 0)
                         | (toUV(i1) << 4)
                         | (toUV(i2) << 8)
                         | (toUV(i3) << 12)
                         | (toUV(i4) << 16)
                         | (toUV(i5) << 20)
                         | (toUV(i6) << 24)
                         | (toUV(i7) << 28);
        return uv(payload);
    }

protected:
    static inline uint32_t toV(int8_t i) {
#ifdef NGEN_SAFE
        if (i & 0x78) throw invalid_immediate_exception();
#endif
        return (i & 0x7) | ((i >> 4) & 0x8);
    }

public:
    static inline Immediate v(uint32_t i) {
        return Immediate(i, DataType::v);
    }

    static inline Immediate v(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7) {
        uint32_t payload = (toV(i0) << 0)
                         | (toV(i1) << 4)
                         | (toV(i2) << 8)
                         | (toV(i3) << 12)
                         | (toV(i4) << 16)
                         | (toV(i5) << 20)
                         | (toV(i6) << 24)
                         | (toV(i7) << 28);
        return v(payload);
    }

    static inline uint32_t toVF(float f) {
        uint32_t fi = utils::bitcast<float, uint32_t>(f);
        int exp = (fi >> 23) & 0xFF;
        int new_exp = exp - 127 + 3;

        if (f == 0.) new_exp = 0;

#ifdef NGEN_SAFE
        if ((new_exp & ~7) || (fi & 0x0007FFFF))
            throw invalid_immediate_exception();
#endif

        return ((fi >> 24) & 0x80)
             | ((new_exp & 0x7) << 4)
             | ((fi >> 19) & 0xF);
    }

    static inline Immediate vf(float f0, float f1, float f2, float f3) {
        uint32_t payload = (toVF(f0) << 0)
                         | (toVF(f1) << 8)
                         | (toVF(f2) << 16)
                         | (toVF(f3) << 24);

        return Immediate(payload, DataType::vf);
    }

    void fixup(int execSize, DataType defaultType, bool isDest, int arity) const {
#ifdef NGEN_SAFE
        if (getBytes(type) > (16 >> arity))
            throw invalid_immediate_exception();
#endif
    }

    constexpr14 bool isScalar() const {
        switch (type) {
            case DataType::uv:
            case DataType::v:
            case DataType::vf:
                return false;
            default:
                return true;
        }
    }

    Immediate forceInt32() const {
        auto result = *this;
        if (result.type == DataType::uw)
            result.set<uint32_t>(uint16_t(payload));
        else if (result.type == DataType::w)
            result.set<int32_t>(int16_t(payload));
        return result;
    }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

/********************************************************************/
/* HDC sends                                                        */
/********************************************************************/
union MessageDescriptor {
    uint32_t all;
    struct {
        unsigned funcCtrl : 19;     /* SF-dependent */
        unsigned header : 1;        /* is a header present? */
        unsigned responseLen : 5;   /* # of GRFs returned: valid range 0-16 */
        unsigned messageLen : 4;    /* # of GRFs sent in src0: valid range 1-15 */
        unsigned : 3;
    } parts;
    struct {
        unsigned index : 8;
        unsigned rest : 24;
    } bti;
    struct {
        unsigned index : 8;
        unsigned elements : 3;
        unsigned subtype : 2;
        unsigned subtype2 : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } block;
    struct {
        unsigned index : 8;
        unsigned simd16 : 1;
        unsigned legacySIMD : 1;
        unsigned elements : 2;
        unsigned : 1;
        unsigned : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } scattered;
    struct {
        unsigned index : 8;
        unsigned subtype : 2;
        unsigned elements : 2;
        unsigned simd16 : 1;
        unsigned : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } a64_scattered;
    struct {
        unsigned index : 8;
        unsigned atomicOp : 4;
        unsigned simd8 : 1;         // or data width.
        unsigned returnData : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } atomic;
    struct {
        unsigned index : 8;
        unsigned cmask : 4;
        unsigned simdMode : 2;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } surface;

    MessageDescriptor() : all(0) {}
};

union ExtendedMessageDescriptor {
    uint32_t all;
    struct {
        unsigned sfid : 4;
        unsigned : 1;
        unsigned eot : 1;
        unsigned extMessageLen : 5;    /* # of GRFs sent in src1: valid range 0-15 (pre-Gen12) */
        unsigned : 1;
        unsigned : 4;                  /* Part of exFuncCtrl for non-immediate sends */
        unsigned exFuncCtrl : 16;
    } parts;

    ExtendedMessageDescriptor() : all(0) {}
    /* implicit */ ExtendedMessageDescriptor(SharedFunction sfid_) : all(0) { parts.sfid = static_cast<int>(sfid_); }
};

enum class AtomicOp : uint8_t {
    cmpwr_2w = 0x00,
    and_ = 0x1,
    or_ = 0x2,
    xor_ = 0x3,
    mov_ = 0x4,
    inc = 0x5,
    dec = 0x6,
    add = 0x7,
    sub = 0x8,
    revsub = 0x9,
    imax = 0xA,
    imin = 0xB,
    umax = 0xC,
    umin = 0xD,
    cmpwr = 0xE,
    predec = 0xF,
    fmax = 0x11,
    fmin = 0x12,
    fcmpwr = 0x13,
};

static inline int operandCount(AtomicOp op) {
    switch (op) {
    case AtomicOp::inc:
    case AtomicOp::dec:
    case AtomicOp::predec:
        return 1;
    case AtomicOp::cmpwr_2w:
    case AtomicOp::cmpwr:
    case AtomicOp::fcmpwr:
        return 3;
    default:
        return 2;
    }
}

static inline constexpr bool isFloatAtomicOp(AtomicOp op) {
    return static_cast<int>(op) >> 4;
}

// Access types.
enum class Access {Read, Write, AtomicInteger, AtomicFloat};

// Address models.
enum AddressModel : uint8_t {
    ModelInvalid = 0,
    ModelBTS = 1,
    ModelA32 = 2,
    ModelA64 = 4,
    ModelSLM = 8,
    ModelCC = 16,
    ModelSC = 32,
    ModelScratch = 64
};

class AddressBase {
protected:
    uint8_t index;
    AddressModel model;

    constexpr AddressBase(uint8_t index_, AddressModel model_) : index(index_), model(model_) {}

    static const uint8_t invalidIndex = 0xF0;

public:
    constexpr AddressBase() : AddressBase(invalidIndex, ModelInvalid) {}

    constexpr uint8_t getIndex()      const { return index; }
    constexpr AddressModel getModel() const { return model; }

    void setIndex(uint8_t newIndex)         { index = newIndex; }

    static constexpr AddressBase createBTS(uint8_t index) {
        return AddressBase(index, ModelBTS);
    }
    static constexpr AddressBase createA32(bool coherent) {
        return AddressBase(coherent ? 0xFF : 0xFD, ModelA32);
    }
    static constexpr AddressBase createA64(bool coherent) {
        return AddressBase(coherent ? 0xFF : 0xFD, ModelA64);
    }
    static constexpr AddressBase createSLM() {
        return AddressBase(0xFE, ModelSLM);
    }
    static constexpr AddressBase createCC(uint8_t index) {
        return AddressBase(index, ModelCC);
    }
    static constexpr AddressBase createSC(uint8_t index) {
        return AddressBase(index, ModelSC);
    }

    inline constexpr bool isRO() const {
        return (getModel() == ModelSC || getModel() == ModelCC);
    }
    inline constexpr bool isStateless() const {
        return (index >= 0xF0);
    }
    void apply(ExtendedMessageDescriptor &exdesc, MessageDescriptor &desc) const {
        desc.bti.index = getIndex();
    }

    void checkModel(uint8_t allowed) { checkModel(static_cast<AddressModel>(allowed)); }
    void checkModel(AddressModel allowed) {
#ifdef NGEN_SAFE
        if (!(model & allowed))
            throw invalid_model_exception();
#endif
    }


};


class block_hword {
protected:
    uint8_t count;

public:
    block_hword(int count_ = 1) : count(count_) {};

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        base.checkModel(ModelA64 | ModelBTS | ModelA32 | ModelSLM);
        desc.all = 0;
        desc.block.elements = 1 + utils::log2(count);
        desc.block.header = true;
        desc.block.messageLen = 1;
        desc.block.responseLen = count;

        if (base.getModel() == ModelA64) {
            exdesc = SharedFunction::dc1;
            desc.block.subtype = 0x3;
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
        } else {
            exdesc = SharedFunction::dc0;
            desc.block.messageType = 0x1;
            desc.block.subtype2 = 1;
        }
    }
};

class block_oword {
protected:
    uint8_t count;
    uint8_t highHalf;

    constexpr block_oword(uint8_t count_, bool highHalf_) : count(count_), highHalf(highHalf_) {}

public:
    block_oword(int count_ = 1) : count(count_), highHalf(false) {}
    static block_oword high() { return block_oword(1, true); }

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        int dataGRFCount = (count + 1) >> 1;

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelCC | ModelSLM);
        exdesc = (base.getModel() == ModelCC)  ? SharedFunction::dcro :
                 (base.getModel() == ModelA64) ? SharedFunction::dc1  :
                                                 SharedFunction::dc0;

        desc.all = 0;
        desc.parts.header = true;
        desc.parts.messageLen = 1;
        desc.parts.responseLen = dataGRFCount;
        desc.block.elements = (count == 1) ? highHalf : (1 + utils::log2(count));

        if (base.getModel() == ModelA64)
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
        else
            desc.block.messageType = (access == Access::Write) << 3;
    }
};

class aligned_block_oword {
protected:
    uint8_t count;
    uint8_t highHalf;

    constexpr aligned_block_oword(uint8_t count_, bool highHalf_) : count(count_), highHalf(highHalf_) {}

public:
    aligned_block_oword(int count_ = 1) : count(count_), highHalf(false) {}
    static aligned_block_oword high() { return aligned_block_oword(1, true); }

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        int dataGRFCount = (count + 1) >> 1;

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelCC | ModelSLM | ModelSC);
        exdesc = (base.getModel() == ModelCC || base.getModel() == ModelSC) ? SharedFunction::dcro :
                                              (base.getModel() == ModelA64) ? SharedFunction::dc1 :
                                                                              SharedFunction::dc0;

        desc.all = 0;
        desc.parts.header = true;
        desc.parts.messageLen = 1;
        desc.parts.responseLen = dataGRFCount;
        desc.block.elements = (count == 1) ? highHalf : (1 + utils::log2(count));

        if (base.getModel() == ModelA64) {
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
            desc.block.subtype = 1;
        } else if (base.getModel() == ModelSC)
            desc.block.messageType = 4;
        else
            desc.block.messageType = ((access == Access::Write) << 3) + 1;
    }
};

class scattered_byte {
protected:
    uint8_t count;

public:
    scattered_byte(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int dataGRFCount = 1 + simd16;

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        desc.all = 0;
        desc.parts.header = false;
        desc.parts.messageLen = dataGRFCount << int(a64);
        desc.parts.responseLen = dataGRFCount;

        if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0;
        } else {
            exdesc = SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.simd16 = simd16;
        }

        if (access == Access::Write)
            desc.scattered.messageType = a64 ? 0x1A : 0xC;
        else
            desc.scattered.messageType = a64 ? 0x10 : 0x4;
    }
};

class scattered_word {
public:
    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = 1 + simd16;

#ifdef NGEN_SAFE
        if (!(access == Access::AtomicInteger || access == Access::AtomicFloat))
            throw invalid_load_store_exception();
#endif
        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        exdesc = SharedFunction::dc1;
        desc.all = 0;
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicFloat)
            desc.atomic.messageType = a64 ? 0x1E : 0x1C;
        else
            desc.atomic.messageType = a64 ? 0x13 : 0x03;

        desc.atomic.simd8 = a64 ? 0 : !simd16;
    }
};

class scattered_dword {
protected:
    uint8_t count;

public:
    scattered_dword(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = count * (1 + simd16);

        desc.all = 0;
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicInteger || access == Access::AtomicFloat) {
            base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
            exdesc = SharedFunction::dc1;
            if (access == Access::AtomicFloat)
                desc.atomic.messageType = a64 ? 0x1D : 0x1B;
            else
                desc.atomic.messageType = a64 ? 0x12 : 0x02;
            desc.atomic.simd8 = a64 ? 0 : !simd16;
        } else if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0x1;
            desc.a64_scattered.messageType = (access == Access::Write) ? 0x1A : 0x10;
        } else {
            base.checkModel(ModelA32 | ModelBTS | ModelCC);
            exdesc = (base.getModel() == ModelCC) ? SharedFunction::dcro : SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.legacySIMD = 1;
            desc.scattered.simd16 = simd16;
            desc.scattered.messageType = (access == Access::Write) ? 0xB : 0x3;
        }
    }
};

class scattered_qword {
protected:
    uint8_t count;

public:
    scattered_qword(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = count * 2 * (1 + simd16);

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        desc.all = 0;
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicInteger || access == Access::AtomicFloat) {
            // Note: atomics have same encoding as scattered dword. The atomic operation type
            //   determines the length. The one exception is A64 atomic float.
            exdesc = SharedFunction::dc1;
            if (access == Access::AtomicFloat) {
                desc.atomic.messageType = a64 ? 0x1D : 0x1B;
                desc.atomic.simd8 = a64 ? 0 : !simd16;
            } else {
                desc.atomic.messageType = a64 ? 0x12 : 0x02;
                desc.atomic.simd8 = a64 ? 1 : !simd16;
            }
        } else if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0x2;
            desc.a64_scattered.messageType = (access == Access::Write) ? 0x1A : 0x10;
        } else {
            exdesc = SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.legacySIMD = 1;
            desc.scattered.simd16 = simd16;
            desc.scattered.messageType = (access == Access::Write) ? 0xD : 0x5;
        }
    }
};

class surface_dword {
protected:
    ChannelMask cmask;
    bool structured;

public:
    surface_dword(ChannelMask cmask_, bool structured_ = false) : cmask(cmask_), structured(structured_) {}

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        int simd16 = mod.getExecSize() >> 4;
        int nChannels = utils::popcnt(0xF ^ static_cast<int8_t>(cmask));
        bool isA64 = base.getModel() == ModelA64;
        int addrGRFCount = (1 + simd16) << int(isA64) << int(structured);
        int dataGRFCount = nChannels * (1 + simd16);

        base.checkModel(ModelBTS | ModelA32 | ModelA64 | ModelSLM);

        exdesc = SharedFunction::dc1;

        desc.all = 0;
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;
        desc.surface.messageType = (isA64 << 4) | ((access == Access::Write) << 3) | 0x01;
        desc.surface.cmask = static_cast<int>(cmask);
        desc.surface.simdMode = 2 - simd16;
    }
};

class media_block {
protected:
    bool vls_override;
    uint8_t vls_offset;
    uint8_t width;
    uint8_t height;

public:
    media_block(int width_, int height_) : vls_override(false), vls_offset(0),
        width(width_), height(height_) {}
    media_block(int width_, int height_, int vls_offset_) : vls_override(true),
        vls_offset(vls_offset_), width(width_), height(height_) {}
    media_block() : media_block(0, 0) {}

    template <Access access> void getDescriptors(const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc) const
    {
        exdesc = SharedFunction::dc1;
        desc.all = 0;
        desc.block.messageType = (base.getModel() == ModelSC) ? 0x05 :
                                    (access == Access::Write) ? 0x0A :
                                                                0x04;
        desc.block.elements = (vls_override << 2) | (vls_offset & 1);
        desc.block.header = true;

        int dataGRFCount = 0;
        if (width > 0) {
            int lg2_rows_per_2grf = std::min<int>(4, 6 - utils::bsr(width));
            dataGRFCount = utils::roundup_pow2((height + (1 << lg2_rows_per_2grf) - 1) >> lg2_rows_per_2grf);
        }

        desc.parts.responseLen = dataGRFCount;
        desc.parts.messageLen = 1;
    }
};

// Generate descriptors for a load operation.
template <typename DataSpec>
void encodeLoadDescriptors(MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    const InstructionModifier &mod, const DataSpec &spec, AddressBase base)
{
    spec.template getDescriptors<Access::Read>(mod, base, desc, exdesc);
    base.apply(exdesc, desc);
}

// Generate descriptors for a store operation. Requires split send for pre-Gen12.
template <typename DataSpec>
void encodeStoreDescriptors(MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    const InstructionModifier &mod, const DataSpec &spec, AddressBase base)
{
#ifdef NGEN_SAFE
    if (base.isRO()) throw read_only_exception();
#endif

    spec.template getDescriptors<Access::Write>(mod, base, desc, exdesc);
    exdesc.parts.extMessageLen = desc.parts.responseLen;
    desc.parts.responseLen = 0;

    base.apply(exdesc, desc);
}

// Generate descriptors for an atomic operation. Requires split send for binary and ternary atomics pre-Gen12.
template <typename DataSpec>
void encodeAtomicDescriptors(MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base)
{
    if (isFloatAtomicOp(op))
        spec.template getDescriptors<Access::AtomicFloat>(mod, base, desc, exdesc);
    else
        spec.template getDescriptors<Access::AtomicInteger>(mod, base, desc, exdesc);

    desc.atomic.returnData = !dst.isNull();
    desc.atomic.atomicOp = static_cast<int>(op);

    exdesc.parts.extMessageLen = desc.parts.responseLen * (operandCount(op) - 1);
    if (dst.isNull())
        desc.parts.responseLen = 0;

    base.apply(exdesc, desc);
}





} /* namespace ngen */


#endif /* header guard */
