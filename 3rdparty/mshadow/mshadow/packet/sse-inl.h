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
 * \file sse-inl.h
 * \brief support of sse2 packet optimization of some operations
 * \author Tianqi Chen
 */
#ifndef MSHADOW_PACKET_SSE_INL_H_
#define MSHADOW_PACKET_SSE_INL_H_

#include <emmintrin.h>
#include "../base.h"
#include "../packet-inl.h"

namespace mshadow {
namespace packet {
template<>
struct Packet<float, kSSE2> {
 public:
  /*! \brief number of float in vector */
  static constexpr index_t size = 4;
  /*! \brief The internal data */
  __m128 data_;
  // enable default copy constructor
  Packet(void) {}
  // constructor from the intrinsic type
  explicit Packet(__m128 data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<float, kSSE2> Fill(float s) {
    return Packet<float, kSSE2>(_mm_set1_ps(s));
  }
  // load from address
  MSHADOW_CINLINE static Packet<float, kSSE2> Load(const float* src) {
    return Packet<float, kSSE2>(_mm_load_ps(src));
  }
  // load from address
  MSHADOW_CINLINE static Packet<float, kSSE2> LoadUnAligned(const float* src) {
    return Packet<float, kSSE2>(_mm_loadu_ps(src));
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<float, kSSE2>& operator=(float s) {
    data_ = _mm_set1_ps(s);
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(float* dst) const {
    _mm_store_ps(dst, data_);
  }
  // get the sum of all contents
  MSHADOW_CINLINE float Sum() const {
    __m128 ans  = _mm_add_ps(data_, _mm_movehl_ps(data_, data_));
    __m128 rst  = _mm_add_ss(ans, _mm_shuffle_ps(ans, ans, 1));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return rst.m128_f32[0];
#else
    float rr = _mm_cvtss_f32(rst);
    return rr;
#endif
  }
};


/*! \brief vector real type for float */
template<>
struct Packet<double, kSSE2> {
  /*! \brief number of float in vector */
  static constexpr index_t size = 2;
  // internal data
  __m128d data_;
  // constructor
  Packet(void) {}
  explicit Packet(__m128d data) : data_(data) {}
  // create a fill with the target value s
  MSHADOW_CINLINE static Packet<double, kSSE2> Fill(double s) {
    return Packet<double, kSSE2>(_mm_set1_pd(s));
  }
  // load from address
  MSHADOW_CINLINE static Packet<double, kSSE2> Load(const double* src) {
    return Packet<double, kSSE2>(_mm_load_pd(src));
  }
  MSHADOW_CINLINE static Packet<double, kSSE2> LoadUnAligned(const double* src) {
    return Packet<double, kSSE2>(_mm_loadu_pd(src));
  }
  // fill it with value s
  MSHADOW_CINLINE Packet<double, kSSE2>& operator=(double s) {
    data_ = _mm_set1_pd(s);
    return *this;
  }
  // store data into dst
  MSHADOW_CINLINE void Store(double* dst) const {
    _mm_store_pd(dst, data_);
  }
  // get sum of all content
  inline double Sum(void) const {
    __m128d tmp =  _mm_add_sd(data_, _mm_unpackhi_pd(data_, data_));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return tmp.m128d_f64[0];
#else
    double ans = _mm_cvtsd_f64(tmp);
    return ans;
#endif
  }
};

MSHADOW_CINLINE Packet<float, kSSE2> operator+(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_add_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator+(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_add_pd(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<float, kSSE2> operator-(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_sub_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator-(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_sub_pd(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<float, kSSE2> operator*(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_mul_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator*(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_mul_pd(lhs.data_, rhs.data_));
}


MSHADOW_CINLINE Packet<float, kSSE2> operator/(const Packet<float, kSSE2>& lhs,
                                                    const Packet<float, kSSE2>& rhs) {
  return Packet<float, kSSE2>(_mm_div_ps(lhs.data_, rhs.data_));
}

MSHADOW_CINLINE Packet<double, kSSE2> operator/(const Packet<double, kSSE2>& lhs,
                                                     const Packet<double, kSSE2>& rhs) {
  return Packet<double, kSSE2>(_mm_div_pd(lhs.data_, rhs.data_));
}

}  // namespace packet
}  // namespace mshadow
#endif  // MSHADOW_PACKET_SSE_INL_H_
