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
 * \file half2.h
 * \brief definition of vector float16, half2 type.
 *
 * \author Antti-Pekka Hynninen
 */
#ifndef MSHADOW_HALF2_H_
#define MSHADOW_HALF2_H_

#if (defined(__CUDACC__) && __CUDA_ARCH__ >= 530 && MSHADOW_USE_CUDA && CUDA_VERSION >= 7050)
  #define MSHADOW_CUDA_HALF2 1
  #include <cuda_fp16.h>
#else
  #define MSHADOW_CUDA_HALF2 0
#endif

#include<math.h>

/*! \brief namespace for mshadow */
namespace mshadow {
/* \brief name space for host/device portable half-precision floats */
namespace half {

#define MSHADOW_HALF2_ASSIGNOP(AOP, OP)                                   \
  template<typename T>                                                    \
  MSHADOW_XINLINE half2_t operator AOP (const T& a) {                     \
    return *this = half2_t(*this OP a);  /* NOLINT(*)*/                   \
  }                                                                       \

class MSHADOW_ALIGNED(4) half2_t {
 public:
#if MSHADOW_CUDA_HALF2
  half2 half2_;
#else
  half_t half_t2[2];
#endif

  MSHADOW_XINLINE half2_t() {}

#if MSHADOW_CUDA_HALF2
  MSHADOW_XINLINE explicit half2_t(half2 a) : half2_(a) {}
#else
  MSHADOW_XINLINE explicit half2_t(half_t a, half_t b) {
    half_t2[0] = a;
    half_t2[1] = b;
  }
#endif

  MSHADOW_XINLINE explicit half2_t(int a) {
#if MSHADOW_CUDA_HALF2
    half2_ = __half2half2(__int2half_rz(a));
#else
    half_t2[0] = (half_t)a;
    half_t2[1] = (half_t)a;
#endif
  }

  MSHADOW_XINLINE half2_t operator+() {
    return *this;
  }

  MSHADOW_XINLINE half2_t operator-() {
#if MSHADOW_CUDA_HALF2
    return half2_t(__hneg2(half2_));
#else
    return half2_t(-half_t2[0], -half_t2[1]);
#endif
  }

  MSHADOW_XINLINE half2_t operator=(const half2_t& a) {
#if MSHADOW_CUDA_HALF2
    half2_ = a.half2_;
#else
    half_t2[0] = a.half_t2[0];
    half_t2[1] = a.half_t2[1];
#endif
    return a;
  }

  MSHADOW_HALF2_ASSIGNOP(+=, +)
  MSHADOW_HALF2_ASSIGNOP(-=, -)
  MSHADOW_HALF2_ASSIGNOP(*=, *)
  MSHADOW_HALF2_ASSIGNOP(/=, /)
};

/*! \brief overloaded + operator for half2_t */
MSHADOW_XINLINE half2_t operator+(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return half2_t(__floats2half2_rn(__low2float(a.half2_) + __low2float(b.half2_),
                                   __high2float(a.half2_) + __high2float(b.half2_)));
#else
  return half2_t(a.half_t2[0] + b.half_t2[0], a.half_t2[1] + b.half_t2[1]);
#endif
}
/*! \brief overloaded - operator for half2_t */
MSHADOW_XINLINE half2_t operator-(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return half2_t(__floats2half2_rn(__low2float(a.half2_) - __low2float(b.half2_),
                                   __high2float(a.half2_) - __high2float(b.half2_)));
#else
  return half2_t(a.half_t2[0] - b.half_t2[0], a.half_t2[1] - b.half_t2[1]);
#endif
}
/*! \brief overloaded * operator for half2_t */
MSHADOW_XINLINE half2_t operator*(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return half2_t(__floats2half2_rn(__low2float(a.half2_) * __low2float(b.half2_),
                                   __high2float(a.half2_) * __high2float(b.half2_)));
#else
  return half2_t(a.half_t2[0] * b.half_t2[0], a.half_t2[1] * b.half_t2[1]);
#endif
}
/*! \brief overloaded / operator for half2_t */
MSHADOW_XINLINE half2_t operator/(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return half2_t(__floats2half2_rn(__low2float(a.half2_) / __low2float(b.half2_),
                                   __high2float(a.half2_) / __high2float(b.half2_)));
#else
  return half2_t(a.half_t2[0] / b.half_t2[0], a.half_t2[1] / b.half_t2[1]);
#endif
}
/*! \brief overloaded % operator for half2_t */
MSHADOW_XINLINE half2_t operator%(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return half2_t(__floats2half2_rn(::fmod(__low2float(a.half2_), __low2float(b.half2_)),
                                   ::fmod(__high2float(a.half2_), __high2float(b.half2_))));
#else
  return half2_t(::fmod(a.half_t2[0], b.half_t2[0]), ::fmod(a.half_t2[1], b.half_t2[1]));
#endif
}
/*! \brief overloaded == operator for half2_t */
MSHADOW_XINLINE bool operator==(half2_t a, half2_t b) {
#if MSHADOW_CUDA_HALF2
  return __hbeq2(a.half2_, b.half2_);
#else
  return (a.half_t2[0] == b.half_t2[0] && a.half_t2[1] == b.half_t2[1]);
#endif
}

}  // namespace half
}  // namespace mshadow
#endif  // MSHADOW_HALF2_H_
