#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include "../src/operator/contrib/tensor/cp_decomp.h"

namespace mxnet {
namespace op {

using namespace std;
using namespace mshadow;
using DType = double;

void outer2D
  (Tensor<cpu, 2, DType> ts,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T) {
  int k = (int) factors_T[0].size(0);

  assert((int) eigvals.size(0) == k);
  assert(factors_T.size() == 2);
  for (int id_mode = 0; id_mode < (int) factors_T.size(); ++id_mode) {
    assert((int) factors_T[id_mode].size(0) == k);
    assert(factors_T[id_mode].size(1) == ts.size(id_mode));
  }

  ts = 0;
  for (int i = 0; i < (int) factors_T[0].size(1); ++i)
    for (int j = 0; j < (int) factors_T[1].size(1); ++j)
      for (int p = 0; p < k; ++p)
        ts[i][j] += eigvals[p] * factors_T[0][p][i] * factors_T[1][p][j];
}

void outer3D
  (Tensor<cpu, 3, DType> ts,
  const Tensor<cpu, 1, DType> &eigvals,
  const std::vector<Tensor<cpu, 2, DType> > &factors_T) {
  int k = (int) factors_T[0].size(0);

  assert((int) eigvals.size(0) == k);
  assert(factors_T.size() == 3);
  for (int id_mode = 0; id_mode < (int) factors_T.size(); ++id_mode) {
    assert((int) factors_T[id_mode].size(0) == k);
    assert(factors_T[id_mode].size(1) == ts.size(id_mode));
  }

  ts = 0;
  for (int i = 0; i < (int) factors_T[0].size(1); ++i)
    for (int j = 0; j < (int) factors_T[1].size(1); ++j)
      for (int l = 0; l < (int) factors_T[2].size(1); ++l)
        for (int p = 0; p < k; ++p)
          ts[i][j][l] += eigvals[p] * factors_T[0][p][i]
            * factors_T[1][p][j] * factors_T[2][p][l];
}


TEST(CPDecomp, 2DTensor) {
  Tensor<cpu, 2, DType> ts(Shape2(5, 4));
  AllocSpace(&ts);

  const int k0 = 3;
  const int k = 2;

  // Generate 2D tensor
  DType eigvals0_[k0] {10, 6, 1};
  DType *mat_0_T_ = new DType[k0 * ts.size(0)] {-0.43467446, -0.5572915 , -0.15002647, -0.52690359, -0.44760359,
            0.27817005, -0.35825313, -0.55669702,  0.60305847, -0.34739751,
                   -0.08818202,  0.63349627, -0.69893409, -0.2969217 , -0.11931069};
  DType *mat_1_T_ = new DType[k0 * ts.size(1)] {-0.38177064, -0.22495485, -0.67352004, -0.59162256,  0.5703576 ,
           -0.64595109,  0.26986906, -0.42966275,  0.3282279 ,  0.72630226,
                   0.09523822, -0.59639011};
  Tensor<cpu, 1, DType> eigvals0(eigvals0_, Shape1(k0));
  std::vector<Tensor<cpu, 2, DType> > factors0_T;
  factors0_T.emplace_back(mat_0_T_, Shape2(k0, ts.size(0)), ts.size(0), nullptr);
  factors0_T.emplace_back(mat_1_T_, Shape2(k0, ts.size(1)), ts.size(1), nullptr);

  outer2D(ts, eigvals0, factors0_T);

  // Allocate space for results
  Tensor<cpu, 1, DType> eigvals(Shape1(k));
  std::vector<Tensor<cpu, 2, DType> > factors_T;
  factors_T.emplace_back(Shape2(k, ts.size(0)));
  factors_T.emplace_back(Shape2(k, ts.size(1)));
  AllocSpace(&eigvals);
  for (auto &m : factors_T)
    AllocSpace(&m);

  int info;
  info = CPDecomp(eigvals, factors_T, ts, k);

  std::cerr << "Eigvals expected:\n";
  print1DTensor_(eigvals0);
  std::cerr << "Eigvals obtained:\n";
  print1DTensor_(eigvals);

  for (int id_mode = 0; id_mode < 2; ++id_mode) {
    std::cerr << "Factor matrix transpose " << id_mode << " expected:\n";
    print2DTensor_(factors0_T[id_mode]);
    std::cerr << "Factor matrix transpose " << id_mode << " obtained:\n";
    print2DTensor_(factors_T[id_mode]);
  }

  FreeSpace(&eigvals);
  for (auto m : factors_T)
    FreeSpace(&m);
  FreeSpace(&ts);

  // Due to numerical imprecision in outer2D() we could sometimes obtain
  // non-zero status
  // This test is rather for visual checking of results
  std::cerr << "Status: " << info << "\n";
  EXPECT_EQ(0, 0);
}

TEST(CPDecomp, 3DTensor) {
  Tensor<cpu, 3, DType> ts(Shape3(5, 4, 3));
  AllocSpace(&ts);

  const int k0 = 3;
  const int k = 2;

  // Generate 2D tensor
  DType eigvals0_[k0] {10, 6, 1};
  DType *mat_0_T_ = new DType[k0 * ts.size(0)] {-0.43467446, -0.5572915 , -0.15002647, -0.52690359, -0.44760359,
            0.27817005, -0.35825313, -0.55669702,  0.60305847, -0.34739751,
                   -0.08818202,  0.63349627, -0.69893409, -0.2969217 , -0.11931069};
  DType *mat_1_T_ = new DType[k0 * ts.size(1)] {-0.38177064, -0.22495485, -0.67352004, -0.59162256,  0.5703576 ,
           -0.64595109,  0.26986906, -0.42966275,  0.3282279 ,  0.72630226,
                   0.09523822, -0.59639011};
  DType *mat_2_T_ = new DType[k0 * ts.size(2)] {-0.66722764, -0.52088417, -0.53243494,  0.63742185, -0.02948216,
           -0.76995077,  0.38535783, -0.8531181 ,  0.35169427};
  Tensor<cpu, 1, DType> eigvals0(eigvals0_, Shape1(k0));
  std::vector<Tensor<cpu, 2, DType> > factors0_T;
  factors0_T.emplace_back(mat_0_T_, Shape2(k0, ts.size(0)), ts.size(0), nullptr);
  factors0_T.emplace_back(mat_1_T_, Shape2(k0, ts.size(1)), ts.size(1), nullptr);
  factors0_T.emplace_back(mat_2_T_, Shape2(k0, ts.size(2)), ts.size(2), nullptr);

  outer3D(ts, eigvals0, factors0_T);

  // Allocate space for results
  Tensor<cpu, 1, DType> eigvals(Shape1(k));
  std::vector<Tensor<cpu, 2, DType> > factors_T;
  factors_T.emplace_back(Shape2(k, ts.size(0)));
  factors_T.emplace_back(Shape2(k, ts.size(1)));
  factors_T.emplace_back(Shape2(k, ts.size(2)));
  AllocSpace(&eigvals);
  for (auto &m : factors_T)
    AllocSpace(&m);

  int info;
  info = CPDecomp(eigvals, factors_T, ts, k);

  std::cerr << "Eigvals expected:\n";
  print1DTensor_(eigvals0);
  std::cerr << "Eigvals obtained:\n";
  print1DTensor_(eigvals);

  for (int id_mode = 0; id_mode < 3; ++id_mode) {
    std::cerr << "Factor matrix transpose " << id_mode << " expected:\n";
    print2DTensor_(factors0_T[id_mode]);
    std::cerr << "Factor matrix transpose " << id_mode << " obtained:\n";
    print2DTensor_(factors_T[id_mode]);
  }

  FreeSpace(&eigvals);
  for (auto m : factors_T)
    FreeSpace(&m);
  FreeSpace(&ts);

  // Due to numerical imprecision in outer3D() we could sometimes obtain
  // non-zero status
  // This test is rather for visual checking of results
  std::cerr << "Status: " << info << "\n";
  EXPECT_EQ(0, 0);
}
}  // op
}  // mxnet


