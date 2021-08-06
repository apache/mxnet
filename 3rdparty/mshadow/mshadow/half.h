/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file half.h
 * \brief definition of half (float16) type.
 *
 * \author Junyuan Xie
 */
#ifndef MSHADOW_HALF_H_
#define MSHADOW_HALF_H_
#include "./base.h"

#if MSHADOW_USE_F16C
  #include <x86intrin.h>
#endif  // MSHADOW_USE_F16C

// This flag dictates rounding for the float2half() routine only (used generally on Windows),
// not the f16c lib or cuda v7.5 (or later) behavior which is fixed at round-to-nearest-even.
#ifndef MSHADOW_HALF_ROUND_TO_NEAREST
#define MSHADOW_HALF_ROUND_TO_NEAREST 1
#endif

#if (MSHADOW_USE_CUDA && CUDA_VERSION >= 7050)
  #define MSHADOW_CUDA_HALF 1
  #include <cuda_fp16.h>
  #if defined(__CUDA_ARCH__)
    /*! \brief __half2float_warp */
    __host__ __device__ float __half2float_warp(const volatile __half& h) { /* NOLINT(*) */
      __half val;
#if CUDA_VERSION >= 9000
      val = const_cast<__half&>(h);
#else
      val.x = h.x;
#endif
      return __half2float(val);
    }
  #endif
#else
  #define MSHADOW_CUDA_HALF 0
#endif

/*! \brief namespace for mshadow */
namespace mshadow {
/* \brief name space for host/device portable half-precision floats */
namespace half {
#define MSHADOW_HALF_OPERATOR(RTYPE, OP)                                  \
  MSHADOW_XINLINE RTYPE operator OP (half_t a, half_t b) {                \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (half_t a, T b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (T a, half_t b) {                     \
    return RTYPE(float(a) OP float(b));  /* NOLINT(*) */                  \
  }

#define MSHADOW_HALF_ASSIGNOP(AOP, OP)                                    \
  template<typename T>                                                    \
  MSHADOW_XINLINE half_t operator AOP (const T& a) {                      \
    return *this = half_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE half_t operator AOP (const volatile T& a) volatile {    \
    return *this = half_t(float(*this) OP float(a));  /* NOLINT(*)*/      \
  }

#if (MSHADOW_CUDA_HALF && defined(__CUDA_ARCH__))
#define MSHADOW_HALF_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(__half2float(cuhalf_));  /* NOLINT(*)*/                      \
  }                                                                       \
  MSHADOW_XINLINE operator T() const volatile {                           \
    return T(__half2float_warp(cuhalf_));  /* NOLINT(*)*/                 \
  }
#elif(MSHADOW_USE_F16C)
#define MSHADOW_HALF_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(_cvtsh_ss(half_));   /* NOLINT(*)*/                          \
  }                                                                       \
  MSHADOW_XINLINE operator T() const volatile {                           \
    return T(_cvtsh_ss(half_));   /* NOLINT(*)*/                          \
  }
#else
#define MSHADOW_HALF_CONVERSIONOP(T)                                      \
  MSHADOW_XINLINE operator T() const {                                    \
    return T(half2float(half_));  /* NOLINT(*)*/                          \
  }                                                                       \
  MSHADOW_XINLINE operator T() const volatile {                           \
    return T(half2float(half_));  /* NOLINT(*)*/                          \
  }
#endif  // (MSHADOW_CUDA_HALF && defined(__CUDA_ARCH__))

class MSHADOW_ALIGNED(2) half_t {
 public:
  union {
    uint16_t half_;
#if MSHADOW_CUDA_HALF
    __half cuhalf_;
#endif  // MSHADOW_CUDA_HALF
  };

  static MSHADOW_XINLINE half_t Binary(uint16_t value) {
    half_t res;
    res.half_ = value;
    return res;
  }

  MSHADOW_XINLINE half_t() {}

#if MSHADOW_CUDA_HALF
  MSHADOW_XINLINE explicit half_t(const __half& value) {
    cuhalf_ = value;
  }
#endif  // MSHADOW_CUDA_HALF

  MSHADOW_XINLINE half_t(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const int8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const int32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const uint32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const int64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit half_t(const uint64_t& value) { constructor(value); }

  MSHADOW_HALF_CONVERSIONOP(float)

  MSHADOW_HALF_ASSIGNOP(+=, +)
  MSHADOW_HALF_ASSIGNOP(-=, -)
  MSHADOW_HALF_ASSIGNOP(*=, *)
  MSHADOW_HALF_ASSIGNOP(/=, /)

  MSHADOW_XINLINE half_t operator+() {
    return *this;
  }

  MSHADOW_XINLINE half_t operator-() {
    return half_t(-float(*this));  // NOLINT(*)
  }

  MSHADOW_XINLINE half_t operator=(const half_t& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  MSHADOW_XINLINE half_t operator=(const T& a) {
    return *this = half_t(a);  /* NOLINT(*)*/
  }

  MSHADOW_XINLINE half_t operator=(const half_t& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  MSHADOW_XINLINE half_t operator=(const T& a) volatile {
    return *this = half_t(a);  /* NOLINT(*)*/
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);  // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;         // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;       // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;    // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;  // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;  // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;  // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;  // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  MSHADOW_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      // The only time it's *not* OK to add 0x1000 (i.e. half the flt16 fraction lsb) is
      // when the lsb of the flt16 fraction == 0 (so not rounding up to even) and the additional
      // bits to the right of the lsb are 1000... (including flt32 significand bits
      // that may be lost during the above vshift).  The first term below will always
      // be true for vshift >=12 (since even the 'hidden bit' has been shifted to the
      // right of the '1' bit in 0x1000). And when vshift <= 11, both terms combine to make
      // the proper test of the flt32 significand bits, including those lost during the vshift.
#if MSHADOW_HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent to 1, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
#endif
    } else if (v.si <= maxN) {
      // Handle norms
#if MSHADOW_HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent, possibly creating an inf, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
#endif
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  MSHADOW_XINLINE uint16_t float2half(const volatile float& value) const volatile {  // NOLINT (*)
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
#if MSHADOW_HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent to 1, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
#endif
    } else if (v.si <= maxN) {
      // Handle norms
#if MSHADOW_HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent, possibly creating an inf, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
#endif
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  MSHADOW_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  MSHADOW_XINLINE float half2float(const volatile uint16_t& value) const volatile {  // NOLINT(*)
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  MSHADOW_XINLINE void constructor(const T& value) {
#if (MSHADOW_CUDA_HALF && defined(__CUDA_ARCH__))
    cuhalf_ = __float2half(float(value));  // NOLINT(*)
#elif(MSHADOW_USE_F16C)
    half_ = _cvtss_sh(static_cast<float>(value), 0);
#else /* !MSHADOW_CUDA_HALF && !MSHADOW_USE_F16C */
    half_ = float2half(float(value));  // NOLINT(*)
#endif /* !MSHADOW_CUDA_HALF && !MSHADOW_USE_F16C */
  }
};

/*! \brief overloaded + operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, +)
/*! \brief overloaded - operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, -)
/*! \brief overloaded * operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, *)
/*! \brief overloaded / operator for half_t */
MSHADOW_HALF_OPERATOR(half_t, /)
/*! \brief overloaded > operator for half_t */
MSHADOW_HALF_OPERATOR(bool, >)
/*! \brief overloaded < operator for half_t */
MSHADOW_HALF_OPERATOR(bool, <)
/*! \brief overloaded >= operator for half_t */
MSHADOW_HALF_OPERATOR(bool, >=)
/*! \brief overloaded <= operator for half_t */
MSHADOW_HALF_OPERATOR(bool, <=)

#define MSHADOW_HALF_MIN mshadow::half::half_t::Binary(0xFBFF);
#define MSHADOW_HALF_MAX mshadow::half::half_t::Binary(0x7BFF);
#define MSHADOW_HALF_SIGN_BIT 0x8000
#define MSHADOW_HALF_EXPONENT_BITS 0x7c00
}  // namespace half
}  // namespace mshadow
#endif  // MSHADOW_HALF_H_
