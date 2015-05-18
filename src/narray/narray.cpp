#include <iostream>
using namespace std;
#include "narray/narray.h"

/*! \brief a demonstration of how to define a new NArray operator */
class AdditionOperator : public NArrayOperator {
public:
  AdditionOperator(FloatT alpha_, FloatT beta_, FloatT gamma_)
    : alpha(alpha_), beta(beta_), gamma(gamma_){}

  void executeCPU(const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs) const {
    // c = a + b
    cout << "doing c=a+b on cpu" << endl;
  }

  void executeGPU(const std::vector<Blob> & inputs,
    const std::vector<Blob> & outputs) const {
    // use cuBLAS
    cout << "doing c=a+b on gpu" << endl;
  }
private:
  FloatT alpha;
  FloatT beta;
  FloatT gamma;
};

// c = alpha*a + beta*b + gamma*c
// c is probably a pre-allocated narray
void Add(FloatT alpha, const NArray &a,
  FloatT beta, const NArray & b,
  FloatT gamma, NArray & c) {
  assert(a.scale == b.scale);
  assert(a.scale == c.scale);
  ScheduleOP(AdditionOperator(alpha, beta, gamma), 
    std::vector<NArray>({ a, b }), std::vector<NArray*>({&c}));
}

// c = alpha*a + beta*b
NArray Add(FloatT alpha, const NArray & a, FloatT beta, const NArray & b) {
  assert(a.scale == b.scale);
  NArray c(a.scale);
  ScheduleOP(AdditionOperator(alpha, beta, 0),
    std::vector<NArray>({ a, b }), std::vector<NArray*>({ &c }));
  return c;
}

// plain c = a + b
NArray operator+ (const NArray & a, const NArray & b) {
  assert(a.scale == b.scale);
  NArray c(a.scale);
  ScheduleOP(AdditionOperator(1, 1, 0),
    std::vector<NArray>({ a, b }), std::vector<NArray*>({ &c }));
  return c;
}

