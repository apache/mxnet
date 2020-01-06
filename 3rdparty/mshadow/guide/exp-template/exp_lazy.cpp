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
