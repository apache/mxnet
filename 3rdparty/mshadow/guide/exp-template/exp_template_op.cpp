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

// Example code, expression template
// with binary operator definition and extension
// for simplicity, we use struct and make all members public
#include <cstdio>

// this is expression, all expressions must inheritate it,
// and put their type in subtype
template<typename SubType>
struct Exp{
  // returns const reference of the actual type of this expression
  inline const SubType& self(void) const {
    return *static_cast<const SubType*>(this);
  }
};

// binary operators
struct mul{
  inline static float Map(float a, float b) {
    return a * b;
  }
};

// binary add expression
// note how it is inheritates from Exp
// and put its own type into the template argument
template<typename OP, typename TLhs, typename TRhs>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, TLhs, TRhs> >{
  const TLhs& lhs;
  const TRhs& rhs;
  BinaryMapExp(const TLhs& lhs, const TRhs& rhs)
      :lhs(lhs), rhs(rhs) {}
  // evaluation function, evaluate this expression at position i
  inline float Eval(int i) const {
    return OP::Map(lhs.Eval(i), rhs.Eval(i));
  }
};
// no constructor and destructor to allocate and de-allocate memory
// allocation done by user
struct Vec: public Exp<Vec>{
  int len;
  float* dptr;
  Vec(void) {}
  Vec(float *dptr, int len)
      : len(len), dptr(dptr) {}
  // here is where evaluation happens
  template<typename EType>
  inline Vec& operator=(const Exp<EType>& src_) {
    const EType &src = src_.self();
    for (int i = 0; i < len; ++i) {
      dptr[i] = src.Eval(i);
    }
    return *this;
  }
  // evaluation function, evaluate this expression at position i
  inline float Eval(int i) const {
    return dptr[i];
  }
};
// template binary operation, works for any expressions
template<typename OP, typename TLhs, typename TRhs>
inline BinaryMapExp<OP, TLhs, TRhs>
F(const Exp<TLhs>& lhs, const Exp<TRhs>& rhs) {
  return BinaryMapExp<OP, TLhs, TRhs>(lhs.self(), rhs.self());
}

template<typename TLhs, typename TRhs>
inline BinaryMapExp<mul, TLhs, TRhs>
operator*(const Exp<TLhs>& lhs, const Exp<TRhs>& rhs) {
  return F<mul>(lhs, rhs);
}

// user defined operation
struct maximum{
  inline static float Map(float a, float b) {
    return a > b ? a : b;
  }
};

const int n = 3;
int main(void) {
  float sa[n] = {1, 2, 3};
  float sb[n] = {2, 3, 4};
  float sc[n] = {3, 4, 5};
  Vec A(sa, n), B(sb, n), C(sc, n);
  // run expression, this expression is longer:)
  A = B * F<maximum>(C, B);
  for (int i = 0; i < n; ++i) {
    printf("%d:%f == %f * max(%f, %f)\n",
           i, A.dptr[i], B.dptr[i], C.dptr[i], B.dptr[i]);
  }
  return 0;
}
