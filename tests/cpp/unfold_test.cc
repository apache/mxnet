
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <vector>
#include "../src/operator/contrib/tensor/unfold.h"

namespace mxnet {
namespace op {

using namespace std;
using namespace mshadow;
using DType = double;

TEST(Unfold, ravel_multi_index_2D) {
  Tensor<cpu, 2, DType> ts(Shape2(20, 30));
  AllocSpace(&ts);

  Shape<2> strides = ts.shape_;
  strides[1] = ts.stride_;

  int c = 0;
  Shape<2> coord;
  for (int i = 0; i < (int) ts.size(0); ++i)
    for (int j = 0; j < (int) ts.size(1); ++j) {
      ts[i][j] = ++c;

      coord[0] = i;
      coord[1] = j;
      EXPECT_DOUBLE_EQ(c, ts.dptr_[ravel_multi_index(coord, strides)]);
    }

  FreeSpace(&ts);
}

TEST(Unfold, ravel_multi_index_3D) {
  Tensor<cpu, 3, DType> ts(Shape3(20, 30, 17));
  AllocSpace(&ts);

  Shape<3> strides = ts.shape_;
  strides[2] = ts.stride_;

  int c = 0;
  Shape<3> coord;
  for (int i = 0; i < (int) ts.size(0); ++i)
    for (int j = 0; j < (int) ts.size(1); ++j)
      for (int k = 0; k < (int) ts.size(2); ++k) {
        ts[i][j] = ++c;

        coord[0] = i;
        coord[1] = j;
        coord[2] = k;
        EXPECT_DOUBLE_EQ(c, ts.dptr_[ravel_multi_index(coord, strides)]);
      }

  FreeSpace(&ts);
}

}  // op
}  // mxnet

