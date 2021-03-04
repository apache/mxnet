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

#ifndef _GEN_REGISTER_ALLOCATOR_HPP__
#define _GEN_REGISTER_ALLOCATOR_HPP__

#include "ngen.hpp"
#include <cstdint>
#include <stdexcept>

namespace ngen {

// Gen registers are organized in banks of bundles.
// Each bundle is modeled as groups of contiguous registers separated by a stride.
struct Bundle {
    static const int8_t any = -1;

    int8_t bundle_id;
    int8_t bank_id;

    Bundle() : bundle_id(any), bank_id(any) {}
    Bundle(int8_t bank_id_, int8_t bundle_id_) : bundle_id(bundle_id_), bank_id(bank_id_) {}

    // Number of bundles in each bank (per thread).
    static constexpr int bundle_count(HW hw)    { return (hw == HW::Gen12LP) ? 8 : 2; }
    // Number of banks.
    static constexpr int bank_count(HW hw)      { return 2; }

    static Bundle locate(HW hw, RegData reg);

    int first_reg(HW hw) const;                  // The first register in the bundle.
    int group_size(HW hw) const;                 // Number of registers in each contiguous group of the bundle.
    int stride(HW hw) const;                     // Stride between register groups of the bundle.

    int64_t reg_mask(HW hw, int offset) const;   // Get register mask for this bundle, for registers [64*offset, 64*(offset+1)).

    friend constexpr bool operator==(const Bundle &b1, const Bundle &b2) {
        return b1.bundle_id == b2.bundle_id && b1.bank_id == b2.bank_id;
    }

    static bool conflicts(HW hw, RegData r1, RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1) == locate(hw, r2));
    }

    static bool same_bank(HW hw, RegData r1, RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1).bank_id == locate(hw, r2).bank_id);
    }
};

// A group of register bundles.
struct BundleGroup {
    explicit BundleGroup(HW hw_) : hw(hw_) {}

    static BundleGroup AllBundles() {
        BundleGroup bg{HW::Gen9};
        for (int rchunk = 0; rchunk < nmasks; rchunk++)
            bg.reg_masks[rchunk] = ~uint64_t(0);
        return bg;
    }

    friend BundleGroup operator|(BundleGroup lhs, Bundle rhs) { lhs |= rhs; return lhs; }
    BundleGroup &operator|=(Bundle rhs) {
        for (int rchunk = 0; rchunk < nmasks; rchunk++)
            reg_masks[rchunk] |= rhs.reg_mask(hw, rchunk);
        return *this;
    }

    uint64_t reg_mask(int rchunk) const {
        return (rchunk < nmasks) ? reg_masks[rchunk % nmasks] : 0;
    }

private:
    HW hw;

    static constexpr int max_regs = 128;
    static constexpr int nmasks = max_regs / 64;

    uint64_t reg_masks[nmasks] = {0};
};

// Gen register allocator.
class RegisterAllocator {
public:
    explicit RegisterAllocator(HW hw_) : hw(hw_) { init(); }

    // Allocation functions: sub-GRFs, full GRFs, and GRF ranges.
    GRFRange alloc_range(int nregs, Bundle base_bundle = Bundle(),
                         BundleGroup bundle_mask = BundleGroup::AllBundles());
    GRF alloc(Bundle bundle = Bundle()) { return alloc_range(1, bundle)[0]; }

    Subregister alloc_sub(DataType type, Bundle bundle = Bundle());
    template <typename T>
    Subregister alloc_sub(Bundle bundle = Bundle()) { return alloc_sub(getDataType<T>(), bundle); }

    FlagRegister alloc_flag();

    // Attempted allocation. Return value is invalid if allocation failed.
    GRFRange try_alloc_range(int nregs, Bundle base_bundle = Bundle(),
                             BundleGroup bundle_mask = BundleGroup::AllBundles());
    GRF try_alloc(Bundle bundle = Bundle()) { return alloc_range(1, bundle)[0]; }

    Subregister try_alloc_sub(DataType type, Bundle bundle = Bundle());
    template <typename T>
    Subregister try_alloc_sub(Bundle bundle = Bundle()) { return try_alloc_sub(getDataType<T>(), bundle); }

    FlagRegister try_alloc_flag();

    // Release a previous allocation or claim.
    void release(GRF reg);
    void release(GRFRange range);
    void release(Subregister subreg);
    void release(FlagRegister flag);

    template <typename RD>
    void safeRelease(RD &reg) { if (!reg.isInvalid()) release(reg); reg.invalidate(); }

    // Claim specific registers.
    void claim(GRF reg);
    void claim(GRFRange range);
    void claim(Subregister subreg);
    void claim(FlagRegister flag);

    void dump(std::ostream &str);

protected:
    static constexpr int max_regs = 128;

    HW hw;                              // HW generation.
    uint8_t free_whole[max_regs / 8];   // Bitmap of free whole GRFs.
    uint8_t free_sub[max_regs];         // Bitmap of free partial GRFs, at dword granularity.
    uint16_t reg_count;                 // # of registers.
    uint8_t free_flag;                  // Bitmap of free flag registers.

    void init();
    void claim_sub(int r, int o, int dw);
};


// Exceptions.
class out_of_registers_exception : public std::runtime_error {
public:
    out_of_registers_exception() : std::runtime_error("Insufficient registers in requested bundle") {}
};

} /* namespace ngen */

#endif /* include guard */
