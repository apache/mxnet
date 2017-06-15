/*!
 *  Copyright (c) 2017 by Contributors
 *  \file krprod_test.cc
 *  \brief Test Khatri-Rao product
 *  \author Jencir Lee
 */
#include <vector>
#include <random>
#include "gtest/gtest.h"
#include "operator/contrib/krprod.h"

namespace mxnet {
namespace op {

using DType = double;

#define EXPECT_DOUBLE_EQ_MATRIX(expected, actual) \
{                                                \
  for (int i = 0; i < static_cast<int>(actual.size(0)); ++i) \
    for (int j = 0; j < static_cast<int>(actual.size(1)); ++j) \
      EXPECT_DOUBLE_EQ(expected[i][j], actual[i][j]); \
} \

TEST(row_wise_kronecker, ZeroInputMatrix) {
  Tensor<cpu, 2, DType> result(Shape2(4, 1)), expected(Shape2(4, 1));
  AllocSpace(&expected);
  AllocSpace(&result);

  expected = 1;
  row_wise_kronecker(result, std::vector<Tensor<cpu, 2, DType> > {});
  EXPECT_DOUBLE_EQ_MATRIX(expected, result);

  FreeSpace(&expected);
  FreeSpace(&result);
}

TEST(row_wise_kronecker, OneInputMatrix) {
  // Input matrices of shape (2, 4) which is also the expected result
  DType mat[8] {1, 2, 3, 4, 5, 6, 7, 8};

  // Make input tensors
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  ts_arr.emplace_back(mat, Shape2(2, 4), 4, nullptr);

  // Compute Khatri-Rao product
  Tensor<cpu, 2, DType> result(Shape2(2, 4));
  AllocSpace(&result);
  row_wise_kronecker(result, ts_arr);

  // Check against expected result
  EXPECT_DOUBLE_EQ_MATRIX(ts_arr[0], result);

  FreeSpace(&result);
}

TEST(row_wise_kronecker, TwoInputMatrices) {
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
  row_wise_kronecker(result, ts_arr);

  // Check against expected result
  Tensor<cpu, 2, DType> ts_expected(expected, Shape2(2, 12), 12, nullptr);
  EXPECT_DOUBLE_EQ_MATRIX(ts_expected, result);

  FreeSpace(&result);
}

TEST(row_wise_kronecker, TwoInputMatrices2) {
  // Input matrices of shape (2, 3) and (2, 1)
  DType mat1[6] {1, 2, 3, 4, 5, 6};
  DType mat2[2] {1, 2};

  // Expect result of shape (2, 3)
  DType expected[6] {1, 2, 3, 8, 10, 12};

  // Make input tensors
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  ts_arr.emplace_back(mat1, Shape2(2, 3), 3, nullptr);
  ts_arr.emplace_back(mat2, Shape2(2, 1), 1, nullptr);

  // Compute Khatri-Rao product
  Tensor<cpu, 2, DType> result(Shape2(2, 3));
  AllocSpace(&result);
  row_wise_kronecker(result, ts_arr);

  // Check against expected result
  Tensor<cpu, 2, DType> ts_expected(expected, Shape2(2, 3), 3, nullptr);
  EXPECT_DOUBLE_EQ_MATRIX(ts_expected, result);

  FreeSpace(&result);
}

TEST(row_wise_kronecker, ThreeInputMatrices) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 6);

  Tensor<cpu, 2, DType> in1(Shape2(3, 4)), in2(Shape2(3, 2)),
    in3(Shape2(3, 3)), kr12(Shape2(3, 8)), kr13(Shape2(3, 24)),
    result(Shape2(3, 24));
  AllocSpace(&in1);
  AllocSpace(&in2);
  AllocSpace(&in3);
  AllocSpace(&kr12);
  AllocSpace(&kr13);
  AllocSpace(&result);

  std::vector<Tensor<cpu, 2, DType> > ts_arr {in1, in2, in3};
  for (auto & in : ts_arr) {
    for (int i = 0; i < static_cast<int>(in.size(0)); ++i)
      for (int j = 0; j < static_cast<int>(in.size(1)); ++j)
        in[i][j] = distribution(generator);
  }

  row_wise_kronecker(kr12, in1, in2);
  row_wise_kronecker(kr13, kr12, in3);
  row_wise_kronecker(result, ts_arr);
  EXPECT_DOUBLE_EQ_MATRIX(kr13, result);

  for (auto & in : ts_arr)
    FreeSpace(&in);
  FreeSpace(&kr12);
  FreeSpace(&kr13);
  FreeSpace(&result);
}

TEST(row_wise_kronecker, ThreeInputMatrices2) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 6);

  Tensor<cpu, 2, DType> in1(Shape2(3, 4)), in2(Shape2(3, 1)),
    in3(Shape2(3, 3)), kr12(Shape2(3, 4)), kr13(Shape2(3, 12)),
    result(Shape2(3, 12));
  AllocSpace(&in1);
  AllocSpace(&in2);
  AllocSpace(&in3);
  AllocSpace(&kr12);
  AllocSpace(&kr13);
  AllocSpace(&result);

  std::vector<Tensor<cpu, 2, DType> > ts_arr {in1, in2, in3};
  for (auto & in : ts_arr) {
    for (int i = 0; i < static_cast<int>(in.size(0)); ++i)
      for (int j = 0; j < static_cast<int>(in.size(1)); ++j)
        in[i][j] = distribution(generator);
  }

  row_wise_kronecker(kr12, in1, in2);
  row_wise_kronecker(kr13, kr12, in3);
  row_wise_kronecker(result, ts_arr);
  EXPECT_DOUBLE_EQ_MATRIX(kr13, result);

  for (auto & in : ts_arr)
    FreeSpace(&in);
  FreeSpace(&kr12);
  FreeSpace(&kr13);
  FreeSpace(&result);
}

TEST(row_wise_kronecker, ThreeInputMatrices3) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 6);

  Tensor<cpu, 2, DType> in1(Shape2(3, 1)), in2(Shape2(3, 4)),
    in3(Shape2(3, 3)), kr12(Shape2(3, 4)), kr13(Shape2(3, 12)),
    result(Shape2(3, 12));
  AllocSpace(&in1);
  AllocSpace(&in2);
  AllocSpace(&in3);
  AllocSpace(&kr12);
  AllocSpace(&kr13);
  AllocSpace(&result);

  std::vector<Tensor<cpu, 2, DType> > ts_arr {in1, in2, in3};
  for (auto & in : ts_arr) {
    for (int i = 0; i < static_cast<int>(in.size(0)); ++i)
      for (int j = 0; j < static_cast<int>(in.size(1)); ++j)
        in[i][j] = distribution(generator);
  }

  row_wise_kronecker(kr12, in1, in2);
  row_wise_kronecker(kr13, kr12, in3);
  row_wise_kronecker(result, ts_arr);
  EXPECT_DOUBLE_EQ_MATRIX(kr13, result);

  for (auto & in : ts_arr)
    FreeSpace(&in);
  FreeSpace(&kr12);
  FreeSpace(&kr13);
  FreeSpace(&result);
}

TEST(row_wise_kronecker, FourInputMatrices) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 6);

  Tensor<cpu, 2, DType> in1(Shape2(3, 47)), in2(Shape2(3, 1)),
    in3(Shape2(3, 5)), in4(Shape2(3, 2173)), kr12(Shape2(3, 47)),
    kr13(Shape2(3, 47 * 5)), kr14(Shape2(3, 47 * 5 * 2173)),
    result(Shape2(3, 47 * 5 * 2173));
  AllocSpace(&in1);
  AllocSpace(&in2);
  AllocSpace(&in3);
  AllocSpace(&in4);
  AllocSpace(&kr12);
  AllocSpace(&kr13);
  AllocSpace(&kr14);
  AllocSpace(&result);

  std::vector<Tensor<cpu, 2, DType> > ts_arr {in1, in2, in3, in4};
  for (auto & in : ts_arr) {
    for (int i = 0; i < static_cast<int>(in.size(0)); ++i)
      for (int j = 0; j < static_cast<int>(in.size(1)); ++j)
        in[i][j] = distribution(generator);
  }

  row_wise_kronecker(kr12, in1, in2);
  row_wise_kronecker(kr13, kr12, in3);
  row_wise_kronecker(kr14, kr13, in4);
  row_wise_kronecker(result, ts_arr);
  EXPECT_DOUBLE_EQ_MATRIX(kr14, result);

  for (auto & in : ts_arr)
    FreeSpace(&in);
  FreeSpace(&kr12);
  FreeSpace(&kr13);
  FreeSpace(&kr14);
  FreeSpace(&result);
}

TEST(krprod, ZeroInputMatrix) {
  Tensor<cpu, 2, DType> result(Shape2(1, 4)), expected(Shape2(1, 4));
  AllocSpace(&expected);
  AllocSpace(&result);

  expected = 1;
  krprod(result, std::vector<Tensor<cpu, 2, DType> > {});
  EXPECT_DOUBLE_EQ_MATRIX(expected, result);

  FreeSpace(&expected);
  FreeSpace(&result);
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
  EXPECT_DOUBLE_EQ_MATRIX(ts_arr[0], result);

  FreeSpace(&result);
}

TEST(krprod, TwoInputMatrices) {
  // Input matrices of shape (3, 2) and (4, 2)
  DType mat1[6] {1, 4, 2, 5, 3, 6};
  DType mat2[8] {1, 5, 2, 6, 3, 7, 4, 8};

  // Expect result of shape (12, 2)
  DType expected[24] {1, 20, 2, 24, 3, 28, 4, 32, 2, 25, 4, 30,
                      6, 35, 8, 40, 3, 30, 6, 36, 9, 42, 12, 48};

  // Make input tensors
  std::vector<Tensor<cpu, 2, DType> > ts_arr;
  ts_arr.emplace_back(mat1, Shape2(3, 2), 2, nullptr);
  ts_arr.emplace_back(mat2, Shape2(4, 2), 2, nullptr);

  // Compute Khatri-Rao product
  Tensor<cpu, 2, DType> result(Shape2(12, 2));
  AllocSpace(&result);
  krprod(result, ts_arr);

  // Check against expected result
  Tensor<cpu, 2, DType> ts_expected(expected, Shape2(12, 2), 2, nullptr);
  EXPECT_DOUBLE_EQ_MATRIX(ts_expected, result);

  FreeSpace(&result);
}

TEST(krprod, ThreeInputMatrices) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 6);

  Tensor<cpu, 2, DType> in1(Shape2(4, 3)), in2(Shape2(2, 3)),
    in3(Shape2(3, 3)), kr12(Shape2(8, 3)), kr13(Shape2(24, 3)),
    result(Shape2(24, 3));
  AllocSpace(&in1);
  AllocSpace(&in2);
  AllocSpace(&in3);
  AllocSpace(&kr12);
  AllocSpace(&kr13);
  AllocSpace(&result);

  std::vector<Tensor<cpu, 2, DType> > ts_arr {in1, in2, in3};
  for (auto & in : ts_arr) {
    for (int i = 0; i < static_cast<int>(in.size(0)); ++i)
      for (int j = 0; j < static_cast<int>(in.size(1)); ++j)
        in[i][j] = distribution(generator);
  }

  krprod(kr12, in1, in2);
  krprod(kr13, kr12, in3);
  krprod(result, ts_arr);
  EXPECT_DOUBLE_EQ_MATRIX(kr13, result);

  for (auto & in : ts_arr)
    FreeSpace(&in);
  FreeSpace(&kr12);
  FreeSpace(&kr13);
  FreeSpace(&result);
}

}  // namespace op
}  // namespace mxnet
