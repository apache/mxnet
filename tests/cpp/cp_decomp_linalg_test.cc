#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include "../src/operator/contrib/tensor/cp_decomp_linalg.h"

namespace mxnet {
namespace op {
namespace cp_decomp {

using namespace mshadow;
using DType = double;

// Solve A X = B, where A is 3-by-3 matrix, B is 3-by-4 matrix
// the expected solution is stored in x
// All matrices are in the row-major layout
TEST(posv, SimpleCase) {
  // 3-by-3 matrix
  DType a[9] {1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1};

  // 3-by-4 matrix
  DType b[12] {17.8, 19.3, 20.8, 22.3, 24.2, 25.9, 27.6, 29.3, 25.6,
            27.2, 28.8, 30.4};

  // 3-by-4 matrix, expected solution
  DType x[12] {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

  int status = posv<cpu, DType>(3, 4, a, 3, b, 4);
  EXPECT_EQ(status, 0);

  for (int i = 0; i < 12; ++i)
    EXPECT_DOUBLE_EQ(x[i], b[i]);
}

// SVD on a 4-by-3 matrix
// the expected left singular vectors are stored in u0
// All matrices are in the row-major layout
TEST(gesdd, SimpleCase) {
  // 4-by-3 matrix
  DType a[12] {7.34352145,   8.21375407,   7.39442948,   9.38655556,
               10.97458645,  10.48197152,  12.73521335,  14.08464237,
               13.03607277,  15.59895938,  16.77779228,  17.4052726};

  // 4-by-3 matrix, expected left singular vectors
  DType u0[12] {-0.30817735,  0.45867683,  0.14174305, -0.4145949 ,
                0.08721548, -0.90453545, -0.53511294,  0.54153923,
                0.3272902 , -0.66842496, -0.69910249,  0.23367854};

  DType u[12], s[3], vt[9];
  int status = gesdd<cpu, DType>('S', 4, 3, a, 3, s, u, 3, vt, 3);
  EXPECT_EQ(status, 0);

  // difference between u and expected solution u0
  DType diff1[12], diff2[12];
  for (int i = 0; i < 12; ++i) {
    diff1[i] = u[i] - u0[i];
    diff2[i] = u[i] + u0[i];
  }

  DType diff_by_col[3];
  for (int j = 0; j < 3; ++j)
    diff_by_col[j] = std::min(nrm2<cpu, DType>(4, &diff1[j], 3),
        nrm2<cpu, DType>(4, &diff2[j], 3));

  DType norm_diff = nrm2<cpu, DType>(3, diff_by_col, 1);
  EXPECT_LE(norm_diff, 1e-6);
}
}  // namespace cp_decomp
}  // namespace op
}  // namespace mxnet
