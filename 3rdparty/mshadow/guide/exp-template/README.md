<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Expression Template Tutorial
====
This page explains how mshadow works. The main trick behind mshadow is called [Expression Template](http://en.wikipedia.org/wiki/Expression_templates).
We will explain how it will affect the performance of compiled code. Expression template is the major trick behind the C++ matrix libraries such as Eigen, GSL, boost.uBLAS.

How to write efficient machine learning code
====
Before we start, let us think of the question above. Assume we want to write down the update rule (for illustration purpose here, while typical update would be `weight += - eta * (grad + lambda * weight)` )
```c++
weight =  - eta * (grad + lambda * weight);
```
Where weight and grad are vectors of length ```n```. When you choose C++ as your programming language,
I guess the major concern is efficiency. There is one principle that is important and used in most C/C++ programs:
* Pre-allocate necessary memory, **no temporal memory allocation** during running.

An example code is like
```c++
void UpdateWeight (const float *grad, float eta, float lambda,
                   int n, float *weight) {
  for (int i = 0; i < n; ++i) {
    weight[i] =  - eta * (grad[i] + lambda * weight[i]);
  }
}
``` 
The function takes the pre-allocated space grad, and weight, and run the calculation. Writing these functions are simple,
however, it can be annoying when we write them repeatedly. So the question is, can we write as follows, and get same performance as previous code?
```c++
void UpdateWeight (const Vec& grad, float eta, float lambda, Vec& weight) {
  weight = -eta * (grad + lambda * weight);
} 
```
The answer is yes, but not by the most obvious solution.

A Naive Bad Solution
====
Let us first take a look at a most straight forward solution: operator overloading.
```c++
// Naive solution for vector operation overloading 
struct Vec {
  int len;
  float* dptr;
  Vec(int len) : len(len) { 
    dptr = new float[len];
  }
  Vec(const Vec& src) : len(src.len) {
    dptr = new float[len];
    memcpy(dptr, src.dptr, sizeof(float)*len ); 
  }
  ~Vec(void) {
    delete [] dptr;
  }
};

inline Vec operator+(const Vec &lhs, const Vec &rhs) {
  Vec res(lhs.len);
  for (int i = 0; i < lhs.len; ++i) {
    res.dptr[i] = lhs.dptr[i] + rhs.dptr[i];
  } 
  return res;
} 
```
If we add more operators overloading in the same style, we can get what we want, and write equations instead of loop.
However, this kind of approach is inefficient, because temporal memory is allocated and de-allocated during each operation, while we could have done better.

An alternative, more effective way is only overload operator+=, operator-=, which can be implemented without temporal memory allocation. But this limits the equations we can write.

We will discuss why we still need expression template although C++11 provides move assignment operator and rvalue reference at the end of this tutorial. 

Lazy Evaluation
====
Let us think why we need temporal memory allocation when doing operator+. This is because we *do not know* the target that will be assigned to in operator+,
otherwise we could have directly storing into target memory instead of temporal memory. 

What if we can know the target? The following code ([exp_lazy.cpp](exp_lazy.cpp)) achieves this. 
```c++
// Example Lazy evaluation code
// for simplicity, we use struct and make all members public
#include <cstdio>
struct Vec;
// expression structure holds the expression
struct BinaryAddExp {
  const Vec &lhs;
  const Vec &rhs;
  BinaryAddExp(const Vec &lhs, const Vec &rhs)
  : lhs(lhs), rhs(rhs) {}
};
// no constructor and destructor to allocate and de-allocate memory,
//  allocation done by user
struct Vec {
  int len;
  float* dptr;
  Vec(void) {}
  Vec(float *dptr, int len)
      : len(len), dptr(dptr) {}
  // here is where evaluation happens
  inline Vec &operator=(const BinaryAddExp &src) {
    for (int i = 0; i < len; ++i) {
      dptr[i] = src.lhs.dptr[i] + src.rhs.dptr[i];
    }
    return *this;
  }
};
// no evaluation happens here
inline BinaryAddExp operator+(const Vec &lhs, const Vec &rhs) {
  return BinaryAddExp(lhs, rhs);
}

const int n = 3;
int main(void) {
  float sa[n] = {1, 2, 3};
  float sb[n] = {2, 3, 4};
  float sc[n] = {3, 4, 5};
  Vec A(sa, n), B(sb, n), C(sc, n);
  // run expression
  A = B + C;
  for (int i = 0; i < n; ++i) {
    printf("%d:%f==%f+%f\n", i, A.dptr[i], B.dptr[i], C.dptr[i]);
  }
  return 0;
}
```
The idea is that we do not actually do computation in operator+, but only return a expression structure (like abstract syntax tree),
and when we overload operator=, we see the target, as well as all the operands, and we can run computation without introducing extra memory!
Similarly, we can define a DotExp and lazily evaluate at operator=, and redirect matrix(vector) multiplications to BLAS.


More Lengthy Expressions and Expression Template
====
By using lazy evaluation, we are cool by avoiding temporal memory allocations. But the ability of the code is limited:
* We can only write ```A=B+C```, but not more lengthy expressions.
* When we add more expression, we need to write more operator= to evaluate each equations.

Here is where the magic of template programming comes to rescue. The following code ([exp_template.cpp](exp_template.cpp)),
which is a bit more lengthy, also allows you to write lengthy equations.
```c++
// Example code, expression template, and more length equations
// for simplicity, we use struct and make all members public
#include <cstdio>

// this is expression, all expressions must inheritate it,
//  and put their type in subtype
template<typename SubType>
struct Exp {
  // returns const reference of the actual type of this expression
  inline const SubType& self(void) const {
    return *static_cast<const SubType*>(this);
  }
};

// binary add expression
// note how it is inheritates from Exp
// and put its own type into the template argument
template<typename TLhs, typename TRhs>
struct BinaryAddExp: public Exp<BinaryAddExp<TLhs, TRhs> > {
  const TLhs &lhs;
  const TRhs &rhs;
  BinaryAddExp(const TLhs& lhs, const TRhs& rhs)
      : lhs(lhs), rhs(rhs) {}
  // evaluation function, evaluate this expression at position i
  inline float Eval(int i) const {
    return lhs.Eval(i) + rhs.Eval(i);
  }
};
// no constructor and destructor to allocate
// and de-allocate memory, allocation done by user
struct Vec: public Exp<Vec> {
  int len;
  float* dptr;
  Vec(void) {}
  Vec(float *dptr, int len)
      :len(len), dptr(dptr) {}
  // here is where evaluation happens
  template<typename EType>
  inline Vec& operator= (const Exp<EType>& src_) {
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
// template add, works for any expressions
template<typename TLhs, typename TRhs>
inline BinaryAddExp<TLhs, TRhs>
operator+(const Exp<TLhs> &lhs, const Exp<TRhs> &rhs) {
  return BinaryAddExp<TLhs, TRhs>(lhs.self(), rhs.self());
}

const int n = 3;
int main(void) {
  float sa[n] = {1, 2, 3};
  float sb[n] = {2, 3, 4};
  float sc[n] = {3, 4, 5};
  Vec A(sa, n), B(sb, n), C(sc, n);
  // run expression, this expression is longer:)
  A = B + C + C;
  for (int i = 0; i < n; ++i) {
    printf("%d:%f == %f + %f + %f\n", i,
           A.dptr[i], B.dptr[i],
           C.dptr[i], C.dptr[i]);
  }
  return 0;
}
```
The key idea of the code is the template ```Exp<SubType>``` takes type of its derived class as template argument, so it can convert itself to
the SubType via ```self()```.  BinaryAddExp now is a template class that can composite expressions together, like a template version of Composite pattern.
The evaluation is done through function Eval, which is done in a recursive way in BinaryAddExp.
* Due to inlining, the function calls of ```src.Eval(i)``` in ```operator=``` will be compiled into ```B.dptr[i] + C.dptr[i] + C.dptr[i]``` in compile time.
* We can write equations for element-wise operations with same efficiency as if we write a loop  

Make it more flexible
====
As we can find in the previous example, template programming is a powerful to make things flexible in compile time, our final example,
which is closer to mshadow, allows user customized binary operators ([exp_template_op.cpp](exp_template_op.cpp)). 
```c++
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
```

Summary
=====
Up to this point, you should have understand basic ideas how it works:
* Lazy evaluation, to allow us see all the operands and target
* Template composition and recursive evaluation, to allows us evaluate arbitrary composite expressions for element-wise operations.
* Due to template and inlining, writing expressions are as efficient as if we directly write a for loop to implement the update rule:)

So write expressions when you write machine learning codes, and focus your energy on the algorithm part that matters.

The Expression Template in MShadow
=====
Expression template in mshadow use the same key points as we introduced in the tutorial, with some minor differences:
* We separate evaluation code from expression construction and composition code.  
    - Instead of putting Eval in Exp class. A Plan class is created from expression, and used to evaluate the result. 
    - This allows us to put less variables in Plan, for example, we do not need array length when we evaluate a data.
    - One important reason is CUDA kernel cannot take class with const references 
    - This design choice is debatable, but we find it is useful so far.
* Lazy support for complex expressions such as matrix dot product
    - Besides element-wise expressions, we also want to support sugars such as ```A = dot(B.T(), C)```,  again, lazy evaluation is used and no extra memory is allocated.
* Type checking and array length checking.

Notes
====
* Expression Template and C++11: in C++11, move constructor can be used to save repetitive allocation memory, which removes some need to expression template. However, the space still needs to be allocated at least once. 
   - This only removes the need of expression template then expression generate space, say dst = A+B+C, dst does not contain space allocated before assignment.
   - If we want to keep the syntax that everything is pre-allocated, and expression executes without memory allocation (which is what we did in mshadow), we still need expression template.

