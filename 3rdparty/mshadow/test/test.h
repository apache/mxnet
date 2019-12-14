#ifndef TEST_H
#define TEST_H

#include "mshadow/tensor.h"
#include "assert.h"

#define EPS 0.0001
using namespace mshadow;
using namespace mshadow::expr;


template<typename xpu>
void Print2DTensor(Tensor<xpu, 2, float> const &ts);

template<typename xpu>
void Print1DTensor(Tensor<xpu, 1, float> const &ts);

template<>
void Print1DTensor(Tensor<cpu, 1, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    printf("%.2f ", ts[i]);
  }
  printf("\n");
}


template<>
void Print2DTensor(Tensor<cpu, 2, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    Print1DTensor(ts[i]);
  }
}

template<>
void Print2DTensor(Tensor<gpu, 2, float> const &tg) {
  Tensor<cpu, 2, float> tc = NewTensor<cpu, float>(tg.shape_, 0.0f);
  Copy(tc, tg);
  Print2DTensor(tc);
  FreeSpace(&tc);
}



bool Check2DTensor(Tensor<gpu, 2, float> const &tg, Tensor<cpu, 2, float> const &tc) {
  Tensor<cpu, 2, float> tcc = NewTensor<cpu, float>(tg.shape_, 0.0f);
  Copy(tcc, tg);
  for (index_t i = 0; i < tc.size(0); ++i) {
    for (index_t j = 0; j < tc.size(1); ++j) {
      assert(abs(tcc[i][j] - tc[i][j]) < EPS);
    }
  }
  FreeSpace(&tcc);
  return true;
}

bool Check1DTensor(Tensor<gpu, 1, float> const &tg, Tensor<cpu, 1, float> const &tc) {
  Tensor<cpu, 1, float> tcc = NewTensor<cpu, float>(tc.shape_, 0.0f);
  Copy(tcc, tg);
  printf("gpu result:\n");
  Print1DTensor(tcc);
  for (index_t i = 0; i < tc.size(0); ++i) {
    assert(abs(tcc[i] - tc[i]) < EPS);
  }
  FreeSpace(&tcc);
  return true;
}
#endif
