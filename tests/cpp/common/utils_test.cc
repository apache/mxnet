/*!
 * Copyright (c) 2017 by Contributors
 * \file utils_test.cc
 * \brief cpu/gpu storage tests
*/
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <cstdio>
#include "test_util.h"
#include "../../src/common/utils.h"

TEST(Common, Lower_Bound) {
  using namespace mxnet::common;
  int64_t values[12] = {-1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 9, 11};
  EXPECT_EQ(lower_bound(values, values + 12, (int64_t) -2), values + 0);
  EXPECT_EQ(lower_bound(values, values + 12, (int64_t) 0), values + 1);
  EXPECT_EQ(lower_bound(values, values + 12, (int64_t) 1), values + 1);
  EXPECT_EQ(lower_bound(values, values + 12, (int64_t) 2), values + 2);
  EXPECT_EQ(lower_bound(values, values + 12, (int64_t) 12), values + 12);
}

TEST(Common, Upper_Bound) {
  using namespace mxnet::common;
  int64_t values[12] = {-1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 9, 11};
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) -2), values + 0);
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) -1), values + 1);
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) 0), values + 1);
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) 2), values + 4);
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) 11), values + 12);
  EXPECT_EQ(upper_bound(values, values + 12, (int64_t) 12), values + 12);
}
