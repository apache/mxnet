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

#ifndef NGEN_UTILS_HPP
#define NGEN_UTILS_HPP

#include <cstdint>

#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef NGEN_CPP11
#define constexpr14
#else
#define constexpr14 constexpr
#endif

namespace ngen {
namespace utils {

template <typename T, typename U>
struct bitcast {
    union {
        T from;
        U to;
    };

    explicit constexpr bitcast(T t) : from{t} {}
    constexpr operator U() const { return to; }
};

template <typename T> static inline constexpr14 int bsf(T x)
{
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    if (sizeof(T) > 4)
        (void) _BitScanForward64(&index, x);
    else
        (void) _BitScanForward(&index, x);
    return index;
#else
    if (sizeof(T) > 4)
        return __builtin_ctzll(x);
    else
        return __builtin_ctz(x);
#endif
}

template <typename T> static inline constexpr14 int bsr(T x)
{
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    if (sizeof(T) > 4)
        (void) _BitScanReverse64(&index, x);
    else
        (void) _BitScanReverse(&index, x);
    return index;
#else
    if (sizeof(T) > 4)
        return (sizeof(unsigned long long) * 8 - 1) - __builtin_clzll(x);
    else
        return (sizeof(int) * 8 - 1) - __builtin_clz(x);
#endif
}

template <typename T> static inline constexpr14 int popcnt(T x)
{
#if defined(_MSC_VER) && !defined(__clang__)
    if (sizeof(T) > 4)
        return __popcnt64(x);
    else
        return __popcnt(x);
#else
    if (sizeof(T) > 4)
        return __builtin_popcountll(x);
    else
        return __builtin_popcount(x);
#endif
}

template <typename T> static inline constexpr14 T roundup_pow2(T x)
{
    if (x <= 1)
        return 1;
    else
        return 1 << (1 + bsr(x - 1));
}

template <typename T> static inline constexpr14 T rounddown_pow2(T x)
{
    return (x <= 1) ? x : (1 << bsr(x));
}

template <typename T> static inline constexpr bool is_zero_or_pow2(T x)
{
    return !(x & (x - 1));
}

template <typename T> static inline constexpr14 int log2(T x)
{
    return bsr(x);
}

template <typename T> static inline constexpr T alignup_pow2(T x, int align)
{
    return (x + align - 1) & -align;
}

} /* namespace utils */
} /* namespace ngen */

#endif
