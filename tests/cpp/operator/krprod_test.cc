/*!
 *  Copyright (c) 2017 by Contributors
 *  \file krprod_test.cc
 *  \brief Test Khatri-Rao product
 *  \author Jencir Lee
 */
#include <vector>
#include <gtest/gtest.h>
#include "operator/contrib/krprod.h"

namespace mxnet {
namespace op {

using DType = double;

void check_equal_matrix
  (const Tensor<cpu, 2, DType> & expected,
  const Tensor<cpu, 2, DType> & actual) {
  for (int i = 0; i < static_cast<int>(actual.size(0)); ++i)
    for (int j = 0; j < static_cast<int>(actual.size(1)); ++j)
      CHECK_DOUBLE_EQ(expected[i][j], actual[i][j];
}

TEST(krprod, OneInputMatrix) {
  // Input matrices of shape (2, 4) which is also the expected result
  DType mat[8] {1, 2, 3, 4, 5, 6, 7, 8};

  // Make input tensors
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  ts_arr.emplace_back(mat, Shape2(2, 4), 4, nullptr);

  // Compute Khatri-Rao product
  Tensor<cpu, 2, DType> result(Shape2(2, 4));
  AllocSpace(&result);
  krprod(result, ts_arr);

  // Check against expected result
  check_equal_matrix(mat, result);

  FreeSpace(&Space);
}

TEST(krprod, TwoInputMatrices) {
  // Input matrices of shape (2, 3) and (2, 4)
  DType mat1[6] {1, 2, 3, 4, 5, 6};
  DType mat2[8] {1, 2, 3, 4, 5, 6, 7, 8};

  // Expect result of shape (2, 12)
  DType expected[24] {1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12,
                      20, 24, 28, 32, 25, 30, 35, 40, 30, 36, 42, 48};

  // Make input tensors
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  ts_arr.emplace_back(mat1, Shape2(2, 3), 3, nullptr);
  ts_arr.emplace_back(mat2, Shape2(2, 4), 4, nullptr);

  // Compute Khatri-Rao product
  Tensor<cpu, 2, DType> result(Shape2(2, 12));
  AllocSpace(&result);
  krprod(result, ts_arr);

  // Check against expected result
  Tensor<cpu, 2, DType> ts_expected(expected, Shape2(2, 12), 12, nullptr);
  check_equal_matrix(ts_expected, result);

  FreeSpace(&Space);
}

/*
TEST(krprod, FourInputMatrices) {


}
*/
}  // namespace op
}  // namespace mxnet


