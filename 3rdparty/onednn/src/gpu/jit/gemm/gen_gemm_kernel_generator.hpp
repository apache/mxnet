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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP

/* Embargo support */
#define STANDALONE 0

#include "common/float16.hpp"
#include "common/utils.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel_common.hpp"
#include "gpu/jit/gemm/utils.hpp"

namespace ngen {
using half = dnnl::impl::float16_t;
}

#define NGEN_HALF_TYPE

#include "../ngen/ngen_interface.hpp"
#include "../ngen/ngen_opencl.hpp"
#include "../ngen/ngen_register_allocator.hpp"

#include <array>
#include <complex>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class Type {
public:
    enum _Type : uint32_t {
        f16 = 0x000201,
        f32 = 0x010402,
        u8 = 0x840100,
        s8 = 0x850100,
        u16 = 0x860201,
        s16 = 0x870201,
        u32 = 0x880402,
        s32 = 0x890402,
        u64 = 0x8A0803,
        s64 = 0x8B0803,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(f32) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr Type real() const { return *this; }
    constexpr bool isComplex() const { return false; }
    constexpr bool isInteger() const { return uint32_t(val) & 0x800000; }
    constexpr bool isFP() const { return !isInteger(); }
    constexpr bool isSigned() const {
        return (uint32_t(val) & 0x810000) != 0x800000;
    }
    constexpr int log2Size() const { return uint32_t(val) & 0xFF; }
    constexpr int size() const { return (uint32_t(val) >> 8) & 0xFF; }
    constexpr int components() const { return isComplex() ? 2 : 1; }

    template <typename U>
    constexpr friend int operator*(U a, Type t) {
        return a << t.log2Size();
    }
    template <typename U>
    constexpr friend int operator/(U a, Type t) {
        return a >> t.log2Size();
    }

    ngen::DataType ngen() const {
        using namespace ngen;
        static const DataType table[16] = {DataType::hf, DataType::f,
                DataType::df, DataType::invalid, DataType::ub, DataType::b,
                DataType::uw, DataType::w, DataType::ud, DataType::d,
                DataType::uq, DataType::q, DataType::invalid, DataType::invalid,
                DataType::invalid, DataType::invalid};
        return table[(uint32_t(val) >> 16) & 0xF];
    }
};

enum class MatrixLayout : uint8_t {
    N = 0,
    Nontranspose = 0,
    T = 1,
    Transpose = 1,
    Pc = 2,
    PackedColumns = 2,
    Pr = 3,
    PackedRows = 3
};

enum class AccessType : uint8_t {
    Scattered, // Use scattered accesses
    SurfaceScattered, // Use untyped surface reads
    Block, // Use block messages
    PseudoBlock, // Use scattered accesses to emulate block accesses
    MediaBlock // Use media block messages (not implemented yet)
};

enum LoopType : uint8_t {
    LoopM = 0,
    LoopN = 1,
    LoopK = 2,
    LoopAny = 0xFF,
    LoopNone = 0xFF
};

enum class RemainderHandling : uint8_t {
    Ignore, // Assume no remainder, or handled by hardware bounds checking.
    General, // Handle all remainder cases.
    Split, // Generate copies of the kernel with and without remainder handling.
    KnownRemainder, // Assume remainder case; don't create special code for non-remainder case.
};

enum class KernelScheduling : uint8_t {
    Static,
    EUStatic,
    Dynamic,
};

struct GRFMultirange {
    std::vector<ngen::GRFRange> ranges;

    GRFMultirange() {}
    GRFMultirange(ngen::GRFRange range) : ranges {1, range} {}

    ngen::GRF operator[](int idx) const {
        for (auto &r : ranges) {
            if (idx < r.getLen()) return r[idx];
            idx -= r.getLen();
        }
        throw std::runtime_error("Index out of bounds");
    }

    bool contiguous(int start, int count) const {
        for (auto &r : ranges) {
            if (start < r.getLen()) return (start + count) <= r.getLen();
            start -= r.getLen();
        }
        return false;
    }

    uint8_t getLen() const {
        uint8_t len = 0;
        for (auto &r : ranges)
            len += r.getLen();
        return len;
    }

    bool empty() const { return ranges.empty(); }
};

template <typename T>
class Scalar {
protected:
    bool fixed_value;
    union {
        ngen::Subregister regs[2];
        T value;
    };

public:
    Scalar() : Scalar(ngen::Subregister()) {}
    explicit Scalar(T value_) : fixed_value(true), value(value_) {}
    Scalar(ngen::Subregister reg0, ngen::Subregister reg1)
        : fixed_value(false), regs {reg0, reg1} {}
    explicit Scalar(ngen::Subregister reg) : Scalar(reg, reg) {}

    Scalar &operator=(T value_) {
        fixed_value = true;
        value = value_;
        return *this;
    }
    Scalar &operator=(ngen::Subregister reg) {
        fixed_value = false;
        regs[0] = regs[1] = reg;
        return *this;
    }

    template <typename U>
    friend inline bool operator==(const Scalar<T> &scalar, const U &val) {
        return scalar.fixed_value && (val == scalar.value);
    }
    template <typename U>
    friend inline bool operator==(const U &val, const Scalar<T> &scalar) {
        return scalar == val;
    }

    template <typename U>
    friend inline bool operator!=(const Scalar<T> &scalar, const U &val) {
        return !(scalar == val);
    }
    template <typename U>
    friend inline bool operator!=(const U &val, const Scalar<T> &scalar) {
        return !(scalar == val);
    }

    operator T() const {
        if (!fixed_value) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }

    bool fixed() const { return fixed_value; }

    ngen::Subregister getReg(int idx) const;
    ngen::Subregister getRegAvoiding(
            ngen::HW hw, const ngen::RegData &rd) const;
};

class MultishiftSubregister {
protected:
    static constexpr int maxShift = 5;
    ngen::Subregister regs[maxShift + 1] = {ngen::Subregister()};
    bool neg = false;

public:
    MultishiftSubregister operator-() const {
        auto copy = *this;
        copy.neg = !copy.neg;
        return copy;
    }

    ngen::Subregister operator>>(int shift) const {
        ngen::RegData sub = ngen::Subregister {};
        if (shift >= 0 && shift <= maxShift) sub = regs[shift];
        if (neg) sub = -sub;
        return *reinterpret_cast<ngen::Subregister *>(&sub);
    }

    void set(int shift, ngen::Subregister reg) { regs[shift] = reg; }
};

struct MatrixAddressing {
    MatrixLayout layout; // Layout type (N/T/Pr/Pc)
    ngen::AddressBase base; // Base for addressing (A64/BTS/...)
    bool padded; // Allow read/write overruns?
    uint8_t packSize; // # of elements in a packed row/column for packed layouts.
    uint8_t crosspack; // Crosspack for packed layouts.
    uint8_t alignment; // Alignment for all addresses, offsets, and leading dimensions.

    void setAlignment(int align) {
        alignment = std::min(128, largest_pow2_divisor(align));
    }
};

struct MatrixAddressingStrategy {
    AccessType accessType; // Block/scattered/etc. access
    bool atomic = false; // Atomic access? (only relevant for C)
};

struct MaskInfo {
    union {
        struct {
            bool isFixed; // = false (variable mask)
            uint8_t maskRep; // # of repetitions of mask pattern.
            uint8_t rsize; // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t bitRep : 4; // # of times each mask bit is repeated.
            uint8_t rdivide : 4; // Amount by which to divide index before forming mask. Fractions are rounded up.
                    // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
        } variable;
        struct {
            bool isFixed; // = true (fixed mask)
            uint8_t _; // not used
            uint16_t value; // mask value
        } fixed;
        uint32_t raw;
    };

    MaskInfo() : fixed {true, 0, 0xFFFF} {}

    bool operator!() const { return fixed.isFixed && fixed.value == 0xFFFF; }
    operator bool() const { return !!*this; }

    static MaskInfo None() { return MaskInfo(); }

    friend bool operator==(const MaskInfo &i1, const MaskInfo &i2) {
        return i1.raw == i2.raw;
    }
    friend bool operator!=(const MaskInfo &i1, const MaskInfo &i2) {
        return !(i1 == i2);
    }
};

struct MaskAssignment {
    MaskInfo mask; // Associated mask
    LoopType var; // Variable to base mask off of
    uint8_t offset; // Amount to subtract from variable.
    uint8_t flag; // Index of virtual flag register to use.

    bool compatible(const MaskAssignment &other) const {
        return mask == other.mask && var == other.var && offset == other.offset;
    }
};

struct RegisterBlock {
    /* Register layout information. */
    uint8_t nr, nc; // Size of this block.
    uint8_t ld; // Leading dimension, in elements.
    uint8_t offsetR, offsetC; // Row and column offset within matrix block
    uint8_t colMajor : 1; // Is this block column-major? (columns stored consecutively inside each register)
    uint8_t crosspack : 7; // Crosspack for this block (1 if none).
    uint16_t bytes; // # of bytes in this block
    uint16_t offsetBytes; // Byte offset within register block

    /* Load/store information. */
    uint8_t remainderR : 1; // Row remaindering enabled?
    uint8_t remainderC : 1; // Column remaindering enabled?
    uint8_t noRowsOK : 1; // Can handle no rows (in mask/descriptor)?
    uint8_t noColsOK : 1; // Can handle no columns (in mask/descriptor)?
    uint8_t descRemR : 1; // Row remainders can be handled by changing the descriptor?
    uint8_t descRemC : 1; // Column remainders can be handled by changing the descriptor?
    uint8_t descAssigned : 1; // True if address registers have been assigned for this block's descriptors.
    uint8_t writable : 1; // True if block is set up for writing.

    uint8_t ebytes; // Size of element in bytes, e.g. 4 for scattered_dword, 16 for block_hword
    uint8_t count; // Element count.
    uint8_t extra; // Extra info. For block accesses, 1 means aligned OWord, 0 unaligned. For scattered accesses, # of consecutive elements.
    uint8_t simdSize; // SIMD size for load/stores (0 indicating no separate load/store needs to be done.)
    uint8_t flag; // Assigned flag register index, or noFlag if none.
    uint8_t sfid; // SFID for this block.
    uint8_t rowFragment; // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment; //     Zero if no fragmenting needed.
    uint8_t addrShift; // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.
    MaskInfo rowMask; // Row mask for this block
    MaskInfo colMask; // Column mask for this block

    static constexpr uint8_t noFlag = 0xFF;

    void calcBytes(Type T); // Auto-calculate # of registers.

    bool hasMask() const { return flag != noFlag; }
    void eraseMask() {
        flag = noFlag;
        rowMask = MaskInfo();
        colMask = MaskInfo();
    }

    bool isLoadBlock() const { return simdSize > 0; }

    int nregs() const;
    int offsetReg() const;
};

struct VirtualFlagAllocator {
    VirtualFlagAllocator(ngen::HW hw) : nflag(4) {}

    int allocVirtual();
    ngen::FlagRegister alloc();

    void claim(int idx) { free &= ~(1 << idx); }
    void claim(const ngen::FlagRegister &reg) { claim(reg.index()); }
    void release(int idx) { free |= (1 << idx); }
    void release(const ngen::FlagRegister &reg) {
        release(reg.index());
        unlock(reg);
    }
    void safeRelease(ngen::FlagRegister &reg) {
        if (!reg.isInvalid()) release(reg);
        reg.invalidate();
    }

    bool isVirtual(int idx) { return (idx >= nflag); }

    bool lock(int idx) {
        bool wasLocked = isLocked(idx);
        locked |= (1 << idx);
        return wasLocked;
    }
    bool lock(const ngen::FlagRegister &reg) { return lock(reg.index()); }
    void unlock(int idx) { locked &= ~(1 << idx); }
    void unlock(const ngen::FlagRegister &reg) { unlock(reg.index()); }
    bool isLocked(int idx) const { return (locked & (1 << idx)); }

    ngen::FlagRegister assignPhysical(int idx);

protected:
    uint32_t free = 0xFFFFFFFF;
    uint8_t locked = 0;
    uint8_t nextPhys = 0;
    uint8_t nflag;
};

// State parameters shared between different kernel types.
struct CommonState {
    ngen::RegisterAllocator ra;
    ngen::GRF signChange, selectImag;
    ngen::GRF vflagStorage;
    std::array<uint8_t, 8> activeVFlags;
    VirtualFlagAllocator raVFlag;
    ngen::Subregister readFailures;
    ngen::Subregister fusedID;
    ngen::GRF emulate64Temp[2];
    ngen::Subregister lsDescConstant[2];
    ngen::FlagRegister flagSwizzle;
    ngen::GRFRange eatomicAddRegs[2];
    int vflagEAtomicAdd;
    ngen::Subregister all1s;

    CommonState(ngen::HW hw) : ra(hw), raVFlag(hw) {}

    void wipeActiveVFlags() {
        for (int i = 0; i < int(activeVFlags.size()); i++)
            if (!raVFlag.isLocked(i)) activeVFlags[i] = 0xFF;
    }

    void usePhysicalFlag(ngen::FlagRegister flag) {
        activeVFlags[flag.index()] = flag.index();
    }

    void allocEmulate64Temp(bool emulate64, bool emulateDWxDW) {
        if (emulateDWxDW || emulate64) emulate64Temp[0] = ra.alloc();
        if (emulate64) emulate64Temp[1] = ra.alloc();
    }
};

// Strategy parameters shared between different kernel types.
struct CommonStrategy {
    int subgroupSize = 8; // Subgroup size provided to OpenCL runtime.
    bool dualGRF = true; // Enable two-GRF instructions
    bool ieeeDenormals = true; // Enable IEEE-compliant denormals
    bool spf = false; // Enable Single Program Flow (SPF) mode in EUs.
    bool accR0 = true; // Stuff r0 header in an accumulator register.
    bool emulate64 = false; // Emulate 64-bit arithmetic (required for GenXLP)
    bool emulateDWxDW
            = false; // Emulate DW x DW -> DW multiplication (required for Gen12)
    bool emulate64_add32
            = false; // Use 32-bit adds for 64-bit arithmetic, assuming no 2^32 boundaries crossed.
    bool wgInSS
            = false; // Pretend to use barriers so that each WG belongs to 1 SS/DSS.
    int GRFs = 128; // # of GRFs to use.
};

// Problem parameters shared between kernel types.
struct CommonProblem {
    bool wgSupport
            = true; // Compile kernel with support for nontrivial workgroups?
    bool nonuniformWGs = true; // Support nonuniform workgroups?
    bool gtpinSupport = false; // Support GT-Pin?
    bool fused = false; // Fused kernels?
};

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize = 0;
    bool fusedEUs = false;
    LoopType loopOrder[3] = {LoopNone, LoopNone, LoopNone};
    int blocking[3] = {0};
    int unroll[3] = {0};
    int wg[3] = {1, 1, 1};
    bool fixedWG = false;
    bool kRemainderHandling = false;
    bool kBlocking = false;
};

// Types of updates for GEMM kernels.
enum class UpdateType {
    Full,
    UpperTriangle,
    UpperTriangleHermitian,
    LowerTriangle,
    LowerTriangleHermitian
};

// k loop bounds types for GEMM kernels.
enum class KRange {
    Full,
    ALowerTriangle,
    AUpperTriangle,
    BLowerTriangle,
    BUpperTriangle
};

// Preferences for using scattered accesses.
enum class ScatterSIMD { Default, Wide, Narrow };

// A/B offset mode.
enum class ABOffset {
    None, // No A/B offsets.
    Calc, // Calculate A/B row/column sums in kernel.
    Load, // Use precalculated row/column sums.
};

// C offset mode.
enum class COffset {
    None, // No C offsets.
    Post, // C offset after all other updates.
    Pre, // C offset before all other updates (bias).
};

// GEMM kernel problem description.
struct GEMMProblem : public CommonProblem {
    Type Ta, Tb, Tc, Ts; // Types for A/B/C/scalars

    Scalar<double> alpha_real, alpha_imag; // Alpha value, if fixed.
    Scalar<double> beta_real, beta_imag; // Beta value, if fixed.
    MatrixAddressing A, B, C, CO; // Addressing information for matrices.
    bool kPositive = false; // Can we assume k > 0?
    bool backward = false; // If true, k loop is backwards.
    bool checkBeta0 = true; // If true, check for beta = 0 and handle specially.
    LoopType fusedLoop = LoopM; // Direction of fusing if threads fused.
    bool batchedS = false; // Strided batch kernel
    bool batchedN = false; // Non-strided batch kernel
    ABOffset abOffset = ABOffset::None; // A/B offset mode.
    COffset cOffset = COffset::None; // C offset mode.

    bool beta0() const {
        return (beta_real == 0) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool beta1() const {
        return (beta_real == 1) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool alpha1() const {
        return (alpha_real == 1) && (!Tc.isComplex() || (alpha_imag == 0));
    }
    bool alphaM1() const {
        return (alpha_real == -1) && (!Tc.isComplex() || (alpha_imag == 0));
    }
    bool fusedM() const { return fused && (fusedLoop == LoopM); }
    bool fusedN() const { return fused && (fusedLoop == LoopN); }
};

// Strategy parameters for GEMM kernels.
struct GEMMStrategy : public CommonStrategy {
    int blocking[3] = {
            0}; // Recommended block size in each dimension (m/n/k) -- for driver.
    int unroll[3]; // Unrolls in each dimension (m/n/k)
    int unrollK_masked = 0; // k unroll to use when masking.
    LoopType loopOrder[3] = {LoopM, LoopN,
            LoopK}; // Expected order of loops in driver code (in order from innermost to outermost).
    int fmaSIMD; // Vector length for FMA.
    int wg[3] = {0}; // m/n/k workgroup sizes, 0 if unconstrained.
    int kernelCrosspack = 1; // Crosspack to use when accumulating C.
    MatrixAddressingStrategy A, B, C; // Strategies for accessing A/B/C.
    int ka_load, kb_load; // How much of A/B is loaded at once, in k dimension
    int ka_load_masked = 0,
        kb_load_masked = 0; // Same as above, when masking m/n.
    int ka_repack = 0,
        kb_repack = 0; // How often to repack loaded A/B (when crosspacked)
    bool slmA = false, slmB = false; // Whether to copy A/B to SLM.
    int slmBuffers = 0; // # of A/B SLM buffers, 0 for none.
    int unrollKSLM
            = 0; // k unroll for SLM copies (0 = auto = unroll[LoopK]/slmCopies)
    int A_copies = 1,
        B_copies = 1; // # of copies of A/B matrices, for latency absorption
    int slmCopies = 1; // # of copies of loaded A/B matrices for SLM copies.
    bool duplicateA = false,
         duplicateB
            = false; // Copy A/B to registers in another bank to avoid conflicts?
    int optAlignAB
            = 0; // Optional alignment for A/B. If > 0, create two versions of k loop, one for A/B aligned to this value, one not.
    enum {
        CSeparate, // C stored in its own bundle, A/B in the other bundle.
        ACB, // A, then C, then B
        BCA, // B, then C, then A
        VNC, // A/B (broadcast matrix second), then C
        ABInterleave, // A/B interleaved, then C
    } registerScheme
            = CSeparate; // Register layout scheme.
    bool kBlocking = false; // Are we doing k blocking?
    bool doubleWA
            = false; // Use explicit double broadcast instructions? (Gen9 only)
    int barrierFreq = 0; // If > 0, set a barrier every barrierFreq loops
    bool altCRemainder = false; // Use alternative double-loop C remainder code?
    bool cAccumulators
            = false; // Use accumulator registers for part of C (to save a few registers)?
    bool cLoadAhead = false; // Load C before doing FMAs?
    bool forceWideSIMDC = false; // Force wider SIMD for C?
    bool noJumpTables = false; // Disallow jump tables?
    RemainderHandling remHandling[3]; // m, n, k remainder handling.
    bool jointSplit
            = true; // Use remainder kernel for both m and n dimensions if both are split.
    int mSplitThresh = 0,
        nSplitThresh
            = 0; // m/n minimum thresholds for using split remainder handling. 0 means always use split.
    bool atomicFMA = false; // Use {Atomic} FMA chains.
    bool checkAdd32
            = true; // Check inside kernel if inner loop additions can be done in 32-bit.
    bool delayABInc
            = false; // Delay A/B increment a few outer products in the k loop.

    bool insideSK = false; // Inside a superkernel?

    CommonDriverInfo driverInfo(const GEMMProblem &problem) const;

    void sanityCheck(ngen::HW hw, const GEMMProblem &problem);
    bool minimize(ngen::HW hw, const GEMMProblem &problem);

    int slmABufBlockSize(Type Ta) const {
        return int(slmA) * Ta * unroll[LoopM] * unrollKSLM;
    }
    int slmBBufBlockSize(Type Tb) const {
        return int(slmB) * Tb * unroll[LoopN] * unrollKSLM;
    }
    int slmABufSize(Type Ta) const {
        return slmABufBlockSize(Ta) * wg[LoopM] * slmBuffers;
    }
    int slmBBufSize(Type Tb) const {
        return slmBBufBlockSize(Tb) * wg[LoopN] * slmBuffers;
    }

    int ka_inc() const { return slmA ? unrollKSLM : ka_load; }
    int kb_inc() const { return slmB ? unrollKSLM : kb_load; }

    bool needsBarrier() const { return (barrierFreq > 0) || (slmBuffers > 0); }

    int wgM() const { return wg[loopOrder[0]]; }
    int wgN() const { return wg[loopOrder[1]]; }
};

// State parameters for GEMM kernels.
struct GEMMState : public CommonState {
    struct {
        ngen::Subregister A, B, C[2], CO, base; // q
        ngen::Subregister ao, bo; // w
        ngen::Subregister offsetA, offsetB, offsetC[2]; // q
        ngen::Subregister offsetCO; // d
        ngen::Subregister lda, ldb, ldc[2]; // d
        ngen::Subregister m, n, k, k0; // d
        ngen::Subregister alpha_real, alpha_imag; // T_real
        ngen::Subregister beta_real, beta_imag; // T_real
        ngen::Subregister groupIDM, groupIDN, groupIDK; // ud
        ngen::GRF localIDM, localIDN, localIDK; // uw
        ngen::Subregister localSizeM, localSizeN, localSizeK; // ud
        ngen::Subregister mapping; // q
        ngen::Subregister flags; // ud
        ngen::Subregister diagA, diagB, diagC; // q
        uint8_t surfaceA, surfaceB; // BTS indices
        uint8_t surfaceC[2], surfaceCO; // BTS indices
        ngen::Subregister strideA, strideB,
                strideC; // ud, used for strided batch.
        ngen::Subregister offsetBatch; // ud, used for non-strided batch.
    } inputs;
    Type Tacc; // Current type in accumulator registers.
    ngen::Subregister effA, effB, effC[2],
            effCO; // Offsets to base of A/B/C/CO chunks for loading/storing.
    ngen::Subregister effAi, effBi;
    ngen::Subregister effAo, effBo;
    std::vector<ngen::GRFRange> A_addrs, B_addrs, C_addrs[2];
    std::vector<ngen::GRFRange> Ai_addrs, Bi_addrs;
    std::vector<ngen::GRFRange> Ao_addrs, Bo_addrs;
    std::vector<GRFMultirange> A_regs, B_regs, C_regs;
    GRFMultirange A1_regs, B1_regs; // A, B duplicate registers.
    GRFMultirange Ar_regs, Br_regs; // Repacked A/B registers.
    std::vector<GRFMultirange> Ai_regs,
            Bi_regs; // Incoming data to copy to SLM.
    GRFMultirange Ao_regs, Bo_regs; // Outgoing data to copy to SLM.
    GRFMultirange As_regs, Bs_regs; // A row sums/B column sums.
    ngen::GRFRange broadcast_regs;
    std::vector<ngen::GRFRange> tempMul_regs;
    ngen::Subregister i0, j0, h0; // d
    ngen::Subregister remainders[3]; // d
    ngen::Subregister remaindersFused[2]; // d
    ngen::Subregister remaindersWG[2]; // d
    ngen::Subregister remFusedStorage; // d
    ngen::FlagRegister mMask, nMask, kMask;
    ngen::Subregister lda_ka, ldb_kb; // d
    ngen::Subregister K; // d
    ngen::FlagRegister flagAP;
    ngen::Subregister beta1; // d
    ngen::Subregister add64; // uw
    ngen::Subregister lidM, lidN, lidStorage; // uw, uw, ud
    ngen::Subregister ha0_slm, hb0_slm, hab0Storage; // uw, uw, ud
    ngen::Subregister ia0_slm, jb0_slm; // uw
    int ma_slm, ka_slm, kb_slm, nb_slm;
    bool A_slmScatter = false, B_slmScatter = false;
    std::vector<RegisterBlock> A_layout, B_layout, C_layout;
    std::vector<RegisterBlock> Ar_layout, Br_layout;
    std::vector<RegisterBlock> Ai_layout, Bi_layout;
    std::vector<RegisterBlock> Ao_layout, Bo_layout;
    std::vector<RegisterBlock> As_layout, Bs_layout;
    bool aioShare, bioShare;
    MatrixAddressing Ai, Bi, Ao, Bo;
    MatrixAddressingStrategy Ai_strategy, Bi_strategy;
    MatrixAddressingStrategy Ao_strategy, Bo_strategy;

    bool isNested;
    int C_accCount;
    bool cSwapActive = false;
    int C_count = 1;
    bool allocedAo = false, allocedBo = false;

    struct {
        ngen::Subregister copyIDs; // ud
        uint8_t surfaceCopyPlan;
        bool copyA, copyB;
    } fused;

    GEMMState(ngen::HW hw) : CommonState(hw) {}
};

// GEMM superkernel strategy parameters.
struct GEMMSuperkernelStrategy {
    std::vector<GEMMStrategy> substrategies;
    KernelScheduling schedule;
    bool multiM, multiN;

    void sanityCheck(ngen::HW hw, const GEMMProblem &problem);
    int subgroupSize() const { return substrategies[0].subgroupSize; }
};

// GEMM superkernel state.
struct GEMMSuperkernelState : public GEMMState {
    struct {
        uint8_t surfacePlan;
        ngen::Subregister planCount;
        ngen::GRF localID;
        ngen::Subregister localSize;
    } inputsSK;
    ngen::Subregister last_i0, last_j0, last_h0;

    GEMMSuperkernelState(ngen::HW hw) : GEMMState(hw) {}
};

// Copy kernel problem description: D <- alpha*S
struct CopyProblem : public CommonProblem {
    Type Ts, Td, Tsum;
    Scalar<double> alpha_real, alpha_imag;
    MatrixAddressing S, D;
    bool conjugate;
    bool lower;
    bool unit;
    bool trsm;
    bool sum = false;
    bool reflecting() const { return false; }
};

// Strategy parameters for copy kernels.
struct CopyStrategy : public CommonStrategy {
    MatrixAddressingStrategy S, D;
    RemainderHandling remHandlingX,
            remHandlingY; // Remainder handling for X dimension (packed dimension) and Y dimension (length of panel)
    int s_load, d_load; // # of rows/columns to load from S/store to D at once
    int s_load_masked,
            d_load_masked; // Same as s_load/d_load, for use when masking.

    int unrollX, unrollY; // Unrolls for each dimension.
    bool duplicateAlpha; // True to make two copies of alpha, one for each register bank
    bool xLoop; // True to loop over x, false to loop over y within a kernel

    bool zBlocking = false; // Kernel parallelized in z dimension?

    int barrierFreq; // If > 0, set a barrier every barrierFreq loops
    int optionalAlignS; // If > 0, generate code to check if S is aligned to this #elements and branch to specific code for that case.

    CommonDriverInfo driverInfo(const CopyProblem &problem) const;

    void sanityCheck(ngen::HW hw, const CopyProblem &problem);

    int unrollW() const { return xLoop ? unrollY : unrollX; }
    int unrollZ() const { return xLoop ? unrollX : unrollY; }
};

// State parameters for copy kernels.
struct CopyState : public CommonState {
    struct {
        ngen::Subregister S, D; // q
        ngen::Subregister offsetS, offsetD; // q
        ngen::Subregister lds, ldd; // d
        ngen::Subregister m, n; // d
        ngen::Subregister alpha_real; // T_real
        ngen::Subregister alpha_imag; // T_real
        ngen::Subregister groupIDW, groupIDZ; // ud
        ngen::GRF localIDW, localIDZ; // uw
        ngen::Subregister localSizeW, localSizeZ; // ud
        ngen::Subregister diag; // d
        ngen::Subregister blockZ; // ud
        uint8_t surfaceS, surfaceD; // DTS indices
    } inputs;
    ngen::Subregister w0, z0; // ud
    ngen::Subregister effS,
            effD; // Offsets to base of S/D chunks for loading/storing.
    ngen::Subregister offsetS1,
            effS1; // Reflected variants of offsetS/effS for symmetric/Hermitian.
    std::vector<ngen::GRFRange> S_addrs, D_addrs;
    std::vector<ngen::GRFRange> S_addrSrcs[2];
    ngen::GRFRange S_regs, D_regs;
    std::vector<ngen::GRFRange> Ds_regs;
    ngen::Subregister lds_sl; // d
    ngen::Subregister ldd_dl; // d
    ngen::Subregister Z; // d
    ngen::FlagRegister flagAP, flagTri, flagDiag;
    ngen::FlagRegister flagReflect[2];
    std::vector<RegisterBlock> S_layout, D_layout;
    std::vector<RegisterBlock> Ds_layout;
    ngen::Subregister remainderX, remainderY; // ud
    ngen::GRF indexVec; // w
    ngen::GRF zero, one, complexOne; // T_real
    struct {
        ngen::GRFRange src1Storage;
        ngen::GRF src1, srcR1, srcI1, r, d;
        ngen::GRFRange mathTemp;
        ngen::GRF temp;
        ngen::Label label;
        int simd;
        ngen::Subregister callStorageSub, callStorage;
        bool use;
    } invertSub;

    bool isNested;

    CopyState(ngen::HW hw) : CommonState(hw) {}

    void dump();
};

template <ngen::HW hw>
class gemm_kernel_generator_t : public ngen::OpenCLCodeGenerator<hw> {
public:
    using super = ngen::OpenCLCodeGenerator<hw>;
    gemm_kernel_generator_t() {}

    NGEN_FORWARD_OPENCL(hw);

    void gemm(GEMMProblem problem, GEMMStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_);
    void gemmSuperkernel(GEMMProblem problem, GEMMSuperkernelStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_, bool loopless);
    void copy(CopyProblem problem, CopyStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_);

protected:
    ngen::NEOInterfaceHandler
            &interface = ngen::OpenCLCodeGenerator<hw>::interface_;

    std::exception_ptr lastException;

    std::ostream &getOutStream() const { return std::cerr; }

    std::ostream &noteStream() const { return getOutStream(); }

    class status_stream {
    protected:
        char cc;
        std::stringstream line;
        bool lineStart = true;

        gemm_kernel_generator_t<hw> &parent;

        friend class gemm_kernel_generator_t<hw>;

    public:
        status_stream(gemm_kernel_generator_t<hw> &parent_, int color = 1)
            : cc(color + '0'), parent(parent_) {}

        static constexpr struct Endl {
        } endl {};

        template <typename T>
        status_stream &operator<<(const T &obj) {
            return *this;
        }

        status_stream &operator<<(const Endl &e) { return *this; }
    } status {*this};

#ifdef SHOW_DISCARDS
    void discardStream() {
        InstructionStream *s = popStream();
        auto oldCC = status.cc;
        status.cc = '4';
        status << "------- \x1B[32mBEGIN\x1B[34m discarded stream -------"
               << status_stream::endl;
        auto &sbuffer = *reinterpret_cast<std::ostringstream *>(s->getBuffer());
        auto str = sbuffer.str();
        bool lastNL = false;
        for (int l = 0; l < str.length(); l++) {
            char c = str[l];

            if (c == '\n') {
                if (lastNL) status << "//";
                status << status_stream::endl;
                lastNL = true;
            } else {
                status << c;
                lastNL = false;
            }
        }
        status << "-------  \x1B[32mEND\x1B[34m discarded stream  -------"
               << status_stream::endl;
        status.cc = status.cc;
        delete s;
    }
#endif

    enum class HintType {
        Bank0,
        Bank1,
        TempComp0,
        TempComp1,
        LongTerm,
        A0,
        A0Broadcast,
        A1,
        A1Broadcast,
        B0,
        B0Broadcast,
        B1,
        B1Broadcast,
        C,
        CLoad,
        S,
        D,
        SAddr,
        DAddr
    };
    enum class StdCRemType { Ignore, Mask, Descriptor };
    enum class COperation { Load, Update, UpdateStore };

    friend std::ostream &operator<<(std::ostream &s, StdCRemType rt) {
        const char *names[3] = {"ignore", "mask", "custom descriptor"};
        return (s << names[static_cast<int>(rt)]);
    }

    ngen::FlagRegister getPhysicalFlag(int vflag, CommonState &state);
    void allocVFlagStorage(const CommonStrategy &strategy, CommonState &state);

    ngen::Bundle getHint(HintType type);
    ngen::Bundle getHint(HintType type, const CommonStrategy &strategy);
    ngen::Bundle getHint(HintType type, const GEMMStrategy &strategy);
    ngen::Bundle getHint(HintType type, const CopyStrategy &strategy);

    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip) {
        goto12(mod, jip, jip);
    }
    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip,
            ngen::Label &uip, bool branchCtrl = false);

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, const CommonStrategy &strategy);
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0, const CommonStrategy &strategy);

    void eaddSignExtend1(const ngen::InstructionModifier &mod, bool &doSub,
            const ngen::Immediate &src1, ngen::Immediate &s1LoPos,
            const ngen::Immediate &s1Lo, const ngen::Immediate &s1Hi, bool &s1Q,
            const ngen::GRF (&temp)[2]);
    void eaddSignExtend1(const ngen::InstructionModifier &mod, bool &doSub,
            const ngen::RegData &src1, ngen::RegData &s1LoPos,
            ngen::RegData &s1Lo, ngen::RegData &s1Hi, bool &s1Q,
            const ngen::GRF (&temp)[2]);
    void eaddHandleS1Neg(
            bool &doSub, ngen::RegData &s1LoPos, const ngen::RegData &s1Lo);
    void eaddHandleS1Neg(bool &doSub, const ngen::Immediate &s1LoPos,
            const ngen::Immediate &s1Lo);

    template <typename DT = void, typename S1>
    void eaddInternal(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, S1 src1, const CommonStrategy &strategy,
            const CommonState &state);
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, const CommonState &state);
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state);
    template <typename DT = void, typename S1>
    void emulInternal(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, S1 src1, const CommonStrategy &strategy,
            const CommonState &state);
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, const CommonState &state);
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state);
    template <typename S1>
    void emul32High(const ngen::InstructionModifier &mod,
            const ngen::RegData &dstHi, const ngen::RegData &src0,
            const S1 &src1);

    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state);
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state);

    template <typename DT = void>
    void mulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1);
    template <typename DT = void>
    void emulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1,
            const CommonStrategy &strategy, const CommonState &state);

    template <typename DT = void, typename S0, typename S2>
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2);
    void cmp0(const ngen::InstructionModifier &mod, ngen::RegData src0);

    template <typename DT = void>
    void alignDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void alignUp(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);

    void simtDoWhileLoop(
            const ngen::InstructionModifier &mod, ngen::Label &dest);

    void slmBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);

    template <typename T>
    void duplicateScalar(Scalar<T> &val, CommonState &state);
    MultishiftSubregister multishift(const ngen::Subregister &reg,
            unsigned shifts, const CommonStrategy &strategy, CommonState &state,
            ngen::Bundle hint = ngen::Bundle());

    void getFusedID(int scale, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void moveR0(const CommonStrategy &strategy, CommonState &state);
    void removeSG(const CommonProblem &problem, const CommonStrategy &strategy,
            const CommonState &state);
    void reorderFusedEUs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    ngen::Subregister copySubregister(const ngen::Subregister &reg,
            CommonState &state,
            ngen::Bundle hint = ngen::Bundle(ngen::Bundle::any, 0));
    void zeroMatrix(const GRFMultirange &r, const CommonStrategy &strategy);
    void releaseFusedRemainders(GEMMState &state);
    void saveLocalIDs(const GEMMStrategy &strategy, GEMMState &state);
    void releaseSavedLocalIDs(GEMMState &state);

    bool getBlockInfo(Type T, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy, int r, int c,
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            ScatterSIMD smode, int maxRBlock, int maxCBlock, int &rblock,
            int &cblock, RegisterBlock &layout);
    bool getSubblock(Type T, RegisterBlock &blockDst,
            const RegisterBlock &blockSrc, bool column, int x1, int x2,
            bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            const std::vector<RegisterBlock> &layout, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<int> &indices, const std::vector<RegisterBlock> &layout,
            bool column, int x1, int x2, bool overrunOK,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    void adjustSubblockAddrs(const std::vector<RegisterBlock> &sublayout,
            const std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs,
            const MatrixAddressing &atype, const CommonStrategy &strategy,
            const CommonState &state);
    bool relevantAddrBlocks(std::vector<int> &relevant,
            const std::vector<RegisterBlock> &layout,
            const std::vector<RegisterBlock> &sublayout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    bool addToRegLayout(Type T, std::vector<RegisterBlock> &layout, int r,
            int c, int roff, int coff, bool remainderR, bool remainderC,
            bool writable, bool avoidFragment, ScatterSIMD smode, int maxRBlock,
            int maxCBlock, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool add1DBlockToRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getRegLayout(Type T, std::vector<RegisterBlock> &layout, int r, int c,
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            ScatterSIMD smode, int maxRBlock, int maxCBlock,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void makeUnbackedRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool colMajor, int crosspack);

    void setupTeardownLoadStoreDesc(
            bool setup, const CommonStrategy &strategy, CommonState &state);
    void loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block,
            const ngen::Subregister &count, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);

    void loadMatrixBlock(const ngen::GRF &dest, const RegisterBlock &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, CommonState &state);
    void loadMatrix(const GRFMultirange &dest,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs, CommonState &state);
    void storeMatrixBlock(const ngen::GRF &src, const RegisterBlock &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, CommonState &state);
    void storeMatrix(const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs, CommonState &state);
    void atomicAddMatrixBlock(Type T, const ngen::GRF &src,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void atomicAddMatrix(Type T, const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);

    bool assignMasks(std::vector<RegisterBlock> &layout, LoopType rloop,
            LoopType cloop, std::vector<MaskAssignment> &assignments,
            CommonState &state);
    void loadMask(MaskAssignment assignment, ngen::Subregister index,
            CommonState &state);
    void loadMasks(const std::vector<MaskAssignment> &assignments,
            ngen::Subregister (&indices)[3], CommonState &state, int start = 0);

    ngen::Subregister startShift(
            const MultishiftSubregister &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value,
            BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value, BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(const BO &ptrShifted, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(const BO &ptrShifted, int shift, CommonState &state);

    template <typename BO>
    void setupAddrUnshifted(const ngen::GRFRange &addr, const BO &ptr,
            const RegisterBlock &layout, const ngen::Subregister &ld,
            size_t sizeofT, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename BO>
    void setupAddr(const ngen::GRFRange &addr, const BO &ptr,
            const RegisterBlock &layout, const ngen::Subregister &ld,
            size_t sizeofT, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename BO>
    void setupAddr(Type T, const std::vector<ngen::GRFRange> &addr,
            const BO &ptr, const std::vector<RegisterBlock> &layout,
            const ngen::Subregister &ld, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddrUnshifted(const ngen::GRFRange &addrDst,
            const ngen::GRFRange &addrSrc, I inc,
            const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, const CommonState &state);
    template <typename I>
    void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc,
            I inc, const RegisterBlock &layoutDst,
            const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddr(const std::vector<ngen::GRFRange> &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename A, typename I>
    void incDecAddr(const A &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state, bool decrement);

    void setupCAddr0(ngen::GRFRange (&C_addr0)[2],
            const std::vector<RegisterBlock> &C_layout, int C_count,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    std::tuple<int, int> targetKernelCrosspack(const GEMMProblem &problem,
            const GEMMStrategy &strategy, const GEMMState &state);
    void outerProductGen9IGEMM(int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void outerProduct(int h, int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void updateC(const GRFMultirange &C_acc, const GRFMultirange &C_accSwap,
            const GRFMultirange &C_load, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void updateCLayout(const std::vector<RegisterBlock> &layout,
            const ngen::GRFRange (&C_addr0)[2], COperation op,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool doStdCRemainder(std::vector<RegisterBlock> &layout, bool inside,
            bool columns[2], StdCRemType remTypes[2], bool fragments[2],
            bool fragPositives[2], int fragSizes[2],
            const ngen::GRFRange (&C_addr0)[2], COperation op,
            std::vector<MaskAssignment> &masks, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState state);
    void doAlternateCRemainder(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);

    void accumulateSum(bool column, Type Tsrc, const GRFMultirange &srcRegs,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            const GRFMultirange &dstRegs,
            const std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state);
    void makeSumLayout(bool column, Type Tsrc,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state);
    void horizontalAdd(bool column, Type T, const GRFMultirange &regs,
            std::vector<RegisterBlock> &layout);
    bool gemmFinalizeSums(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    void convert(const GRFMultirange &range, Type Told, Type Tnew,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmConvertC(Type Tnew, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmBetaScale(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmFixedOffsetC(const ngen::Subregister &offset,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmVariableOffsetC(bool column, const GRFMultirange &offsets,
            const ngen::Subregister &scale, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmLoadABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmApplyABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmApplyCOffset(bool row, bool column, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmApplyCOffsetDispatch(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocRegs(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocAoBoRegs(
            bool forceAlloc, const GEMMStrategy &strategy, GEMMState &state);
    void doAIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, int ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doAIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const MultishiftSubregister &ka_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    template <typename I>
    void doAIncrement(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doALoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void doALoadInc(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, int kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const MultishiftSubregister &kb_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    template <typename I>
    void doBIncrement(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBLoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void doBLoadInc(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmCalcIncrements(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int ka_load = 0,
            int kb_load = 0);

    bool gemmKLoop(int ka_repack, int kb_repack, bool lateKLoopCheck,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateC(
            GEMMProblem problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccessC(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    bool gemmUpdateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);
    bool gemmBodyInternal(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool mnRemainderHandling(LoopType loop, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    GEMMProblem, GEMMStrategy, GEMMState));
    bool mnJointSplitRemainderHandling(GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    GEMMProblem, GEMMStrategy, GEMMState));
    bool gemmMEdge(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);

    void gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    void gemmOffsetABC(bool initial, ngen::Subregister i0, ngen::Subregister j0,
            ngen::Subregister h0, GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool doA = true, bool doB = true,
            bool doC = true);
    void gemmSetupABC(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool doA = true, bool doB = true,
            bool doC = true);
    void gemmSubkernel(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state);
    void gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool inSK = false);
    void gemmTypeCheck(Type Ta, Type Tb, Type Tc);

    void gemmSuperkernelInitState(GEMMProblem &problem,
            GEMMSuperkernelStrategy &strategy, GEMMSuperkernelState &state);

    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, bool conjugate, CommonStrategy &strategy,
            CommonState &state);
    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, const Scalar<double> &alpha_real,
            const Scalar<double> &alpha_imag, bool conjugate,
            CommonStrategy &strategy, CommonState &state);

    bool copyBody(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyRemCheck(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyInternal(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copySlice(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void copyCalcIncrements(const CopyProblem &problem,
            const CopyStrategy &strategy, CopyState &state, int s_load = 0,
            int d_load = 0);

    void copyInitState(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void prologue(const CommonStrategy &strategy);
    void epilogue(const CommonStrategy &strategy);
    void padding();
    void initState(const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
