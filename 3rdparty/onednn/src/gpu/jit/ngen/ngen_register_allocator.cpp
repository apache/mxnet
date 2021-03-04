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

#include "ngen_register_allocator.hpp"
#include "ngen_utils.hpp"
#include <iomanip>
#include <iostream>

namespace ngen {

int Bundle::first_reg(HW hw) const
{
    int bundle0 = (bundle_id == any) ? 0 : bundle_id;
    int bank0 = (bank_id == any) ? 0 : bank_id;

    switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        return (bundle0 << 8) | bank0;
    case HW::Gen11:
        return (bundle0 << 8) | (bank0 << 1);
    case HW::Gen12LP:
        return (bundle0 << 1) | bank0;
    default:
        return 0;
    }
}

int Bundle::group_size(HW hw) const
{
    if (bundle_id == any && bank_id == any)
        return 128;
    else switch (hw) {
    case HW::Gen11:
        return 2;
    default:
        return 1;
    }
}

int Bundle::stride(HW hw) const
{
    if (bundle_id == any && bank_id == any)
        return 128;
    else switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        return 2;
    case HW::Gen11:
        return 4;
    case HW::Gen12LP:
        return 16;
    default:
        return 128;
    }
}

int64_t Bundle::reg_mask(HW hw, int offset) const
{
    int64_t bundle_mask = -1, bank_mask = -1, base_mask = -1;
    int bundle0 = (bundle_id == any) ? 0 : bundle_id;
    int bank0   = (bank_id == any)   ? 0 : bank_id;

    switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        if (bundle_id != any && bundle_id != offset)    bundle_mask = 0;
        if (bank_id != any)                             bank_mask = 0x5555555555555555 << bank_id;
        return bundle_mask & bank_mask;
    case HW::Gen11:
        if (bundle_id != any && bundle_id != offset)    bundle_mask = 0;
        if (bank_id != any)                             bank_mask = 0x3333333333333333 << (bank_id << 1);
        return bundle_mask & bank_mask;
    case HW::Gen12LP:
        if (bundle_id != any)                           base_mask  = 0x0003000300030003;
        if (bank_id != any)                             base_mask &= 0x5555555555555555;
        return base_mask << (bank0 + (bundle0 << 1));
    default:
        return -1;
    }
}

Bundle Bundle::locate(HW hw, RegData reg)
{
    int base = reg.getBase();

    switch (hw) {
        case HW::Gen9:
        case HW::Gen10:
            return Bundle(base & 1, base >> 6);
        case HW::Gen11:
            return Bundle((base >> 1) & 1, base >> 6);
        case HW::Gen12LP:
            return Bundle(base & 1, (base >> 1) & 7);
        default:
            return Bundle();
    }
}

// -----------------------------------------
//  Low-level register allocator functions.
// -----------------------------------------

void RegisterAllocator::init()
{
    for (int r = 0; r < max_regs; r++)
        free_sub[r] = 0xFF;
    for (int r_whole = 0; r_whole < (max_regs >> 3); r_whole++)
        free_whole[r_whole] = 0xFF;

    free_flag = 0xF;
    reg_count = max_regs;

}

void RegisterAllocator::claim(GRF reg)
{
    int r = reg.getBase();

    free_sub[r] = 0x00;
    free_whole[r >> 3] &= ~(1 << (r & 7));
}

void RegisterAllocator::claim(GRFRange range)
{
    for (int i = 0; i < range.getLen(); i++)
        claim(range[i]);
}

void RegisterAllocator::claim(Subregister subreg)
{
    int r = subreg.getBase();
    int dw = subreg.getDwords();
    int o = (subreg.getByteOffset()) >> 2;

    claim_sub(r, o, dw);
}

void RegisterAllocator::claim_sub(int r, int o, int dw)
{
    free_sub[r]        &= ~((1 << (o + dw)) - (1 << o));
    free_whole[r >> 3] &= ~(1 << (r & 7));
}

void RegisterAllocator::claim(FlagRegister flag)
{
    free_flag &= ~(1 << flag.index());
}

void RegisterAllocator::release(GRF reg)
{
    int r = reg.getBase();

    free_sub[r] = 0xFF;
    free_whole[r >> 3] |= (1 << (r & 7));
}

void RegisterAllocator::release(GRFRange range)
{
    for (int i = 0; i < range.getLen(); i++)
        release(range[i]);
}

void RegisterAllocator::release(Subregister subreg)
{
    int r = subreg.getBase();
    int dw = subreg.getDwords();
    int o = (subreg.getByteOffset()) >> 2;

    free_sub[r] |= (1 << (o + dw)) - (1 << o);
    if (free_sub[r] == 0xFF)
        free_whole[r >> 3] |= (1 << (r & 7));
}

void RegisterAllocator::release(FlagRegister flag)
{
    free_flag |= (1 << flag.index());
}

// -------------------------------------------
//  High-level register allocation functions.
// -------------------------------------------

GRFRange RegisterAllocator::alloc_range(int nregs, Bundle base_bundle, BundleGroup bundle_mask)
{
    auto result = try_alloc_range(nregs, base_bundle, bundle_mask);
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

Subregister RegisterAllocator::alloc_sub(DataType type, Bundle bundle)
{
    auto result = try_alloc_sub(type, bundle);
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

FlagRegister RegisterAllocator::alloc_flag()
{
    auto result = try_alloc_flag();
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

GRFRange RegisterAllocator::try_alloc_range(int nregs, Bundle base_bundle, BundleGroup bundle_mask)
{
    int64_t *free_whole64 = (int64_t *) free_whole;
    bool ok = false;
    int r_base = -1;

    for (int rchunk = 0; rchunk < (max_regs >> 6); rchunk++) {
        int64_t free = free_whole64[rchunk] & bundle_mask.reg_mask(rchunk);
        int64_t free_base = free & base_bundle.reg_mask(hw, rchunk);

        while (free_base) {
            // Find the first free base register.
            int first_bit = utils::bsf(free_base);
            r_base = first_bit + (rchunk << 6);

            // Check if required # of registers are available.
            int last_bit = first_bit + nregs;
            if (last_bit <= 64) {
                // Range to check doesn't cross 64-GRF boundary. Fast check using bitmasks.
                uint64_t mask = ((uint64_t(1) << (last_bit - 1)) << 1) - (uint64_t(1) << first_bit);
                ok = !(mask & ~free);
            } else {
                // Range to check crosses 64-GRF boundary. Check first part using bitmasks,
                // Check the rest using a loop (ho hum).
                uint64_t mask = ~uint64_t(0) << first_bit;
                ok = !(mask & ~free);
                if (ok) for (int rr = 64 - first_bit; rr < nregs; rr++) {
                    if (free_sub[r_base + rr] != 0xFF) {
                        ok = false;
                        break;
                    }
                }
            }

            if (ok) {
                // Create and claim GRF range.
                GRFRange result(r_base, nregs);
                claim(result);

                return result;
            }

            // Not enough consecutive registers. Save time when looking for next base
            //  register by clearing the entire range of registers we just considered.
            int64_t clear_mask = free + (uint64_t(1) << first_bit);
            free &= clear_mask;
            free_base &= clear_mask;
        }
    }

    return GRFRange();
}

Subregister RegisterAllocator::try_alloc_sub(DataType type, Bundle bundle)
{
    int dwords = getDwords(type);
    int r_alloc, o_alloc;

    auto find_alloc_sub = [&,bundle,dwords](bool search_full_grf) -> bool {
        static const uint8_t alloc_patterns[4] = {0b11111111, 0b01010101, 0, 0b00010001};
        uint8_t alloc_pattern = alloc_patterns[dwords - 1];
        int64_t *free_whole64 = (int64_t *) free_whole;

        for (int rchunk = 0; rchunk < (max_regs >> 6); rchunk++) {
            int64_t free = search_full_grf ? free_whole64[rchunk] : -1;
            free &= bundle.reg_mask(hw, rchunk);

            while (free) {
                int rr = utils::bsf(free);
                int r = rr + (rchunk << 6);
                free &= ~(int64_t(1) << rr);

                if (search_full_grf || free_sub[r] != 0xFF) {
                    int subfree = free_sub[r];
                    for (int dw = 1; dw < dwords; dw++)
                        subfree &= (subfree >> dw);
                    subfree &= alloc_pattern;

                    if (subfree) {
                        r_alloc = r;
                        o_alloc = utils::bsf(subfree);
                        return true;
                    }
                }
            }
        }

        return false;
    };

    // First try to find room in a partially allocated register; fall back to
    //  completely empty registers if unsuccessful.
    bool success = find_alloc_sub(false)
                || find_alloc_sub(true);

    if (!success)
        return Subregister();

    claim_sub(r_alloc, o_alloc, dwords);

    return Subregister(GRF(r_alloc), (o_alloc << 2) / getBytes(type), type);
}

FlagRegister RegisterAllocator::try_alloc_flag()
{
    if (!free_flag) return FlagRegister();

    int idx = utils::bsf(free_flag);
    free_flag &= (free_flag - 1);               // clear lowest bit.

    return FlagRegister::createFromIndex(idx);
}

void RegisterAllocator::dump(std::ostream &str)
{
    str << "\n// Flag registers: ";
    for (int r = 0; r < 4; r++)
        str << char((free_flag & (1 << r)) ? '.' : 'x');

    for (int r = 0; r < reg_count; r++) {
        if (!(r & 0x1F)) {
            str << "\n//\n// " << std::left;
            str << 'r' << std::setw(3) << r;
            str << " - r" << std::setw(3) << r+0x1F;
            str << "  ";
        }
        if (!(r & 0xF))  str << ' ';
        if (!(r & 0x3))  str << ' ';

        switch (free_sub[r]) {
            case 0x00: str << 'x'; break;
            case 0xFF: str << '.'; break;
            default:   str << '/'; break;
        }
    }

    str << "\n//\n";

    for (int r = 0; r < max_regs; r++) {
        int rr = r >> 3, rb = 1 << (r & 7);
        if ((free_sub[r] == 0xFF) != bool(free_whole[rr] & rb))
            str << "// Inconsistent bitmaps at r" << r << std::endl;
        if (free_sub[r] != 0x00 && free_sub[r] != 0xFF) {
            str << "//  r" << std::setw(3) << r << "   ";
            for (int s = 0; s < 8; s++)
                str << char((free_sub[r] & (1 << s)) ? '.' : 'x');
            str << std::endl;
        }
    }

    str << std::endl;
}

constexpr int BundleGroup::max_regs;
constexpr int BundleGroup::nmasks;
constexpr int RegisterAllocator::max_regs;

} /* namespace ngen */
