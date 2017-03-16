#include <mxnet/tensor_blob.h>
#include <gtest/gtest.h>
#include <vector>
#include "../src/operator/tensor/unfold.h"

namespace mxnet {
namespace op {

using namespace std;
using namespace mshadow;
using DType = double;

__global__ 
void AccessElements(int n, const DType *dptr, DType *result, const int *indices) {
  for (int i = 0; i < n; ++i)
    result[i] = dptr[indices[i]]; 
}


TEST(Unfold, ravel_multi_index_2D_gpu) {
  Tensor<gpu, 2, DType> ts(Shape2(19, 29));
  Tensor<cpu, 2, DType> ts_cpu(ts.shape_);
  AllocSpace(&ts);
  AllocSpace(&ts_cpu);

  int *indices;
  DType *result;
  cudaMallocManaged((void **)&indices, ts.shape_.Size() * sizeof(int));
  cudaMallocManaged((void **)&result, ts.shape_.Size() * sizeof(DType));

  Shape<2> strides = ts.shape_;
  strides[1] = ts.stride_;

  int c = 0;
  Shape<2> coord;
  for (int i = 0; i < (int) ts.size(0); ++i)
    for (int j = 0; j < (int) ts.size(1); ++j) {
      coord[0] = i;
      coord[1] = j;
      indices[c] = ravel_multi_index(coord, strides);

      ts_cpu[i][j] = ++c;
    }
  cudaDeviceSynchronize();
  
  cudaMemcpy2D(ts.dptr_, ts.stride_ * sizeof(DType), 
      ts_cpu.dptr_, ts_cpu.stride_ * sizeof(DType),
      ts.size(1) * sizeof(DType), ts.size(0),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  AccessElements<<<1, 1>>>(ts.shape_.Size(), ts.dptr_, result, indices);
  cudaDeviceSynchronize();

  for (int i = 0; i < (int) ts.shape_.Size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], i + 1);
  }

  FreeSpace(&ts);
  FreeSpace(&ts_cpu);
  cudaFree(result);
  cudaFree(indices);
}

TEST(Unfold, ravel_multi_index_3D_gpu) {
  Tensor<gpu, 3, DType> ts(Shape3(19, 29, 53));
  Tensor<cpu, 3, DType> ts_cpu(ts.shape_);
  AllocSpace(&ts);
  AllocSpace(&ts_cpu);

  int *indices;
  DType *result;
  cudaMallocManaged((void **)&indices, ts.shape_.Size() * sizeof(int));
  cudaMallocManaged((void **)&result, ts.shape_.Size() * sizeof(DType));

  Shape<3> strides = ts.shape_;
  strides[2] = ts.stride_;

  int c = 0;
  Shape<3> coord;
  for (int i = 0; i < (int) ts.size(0); ++i)
    for (int j = 0; j < (int) ts.size(1); ++j) 
      for (int k = 0; k < (int) ts.size(2); ++k) {
        coord[0] = i;
        coord[1] = j;
        coord[2] = k;
        indices[c] = ravel_multi_index(coord, strides);

        ts_cpu[i][j][k] = ++c;
      }
  cudaDeviceSynchronize();
  
  cudaMemcpy2D(ts.dptr_, ts.stride_ * sizeof(DType), 
      ts_cpu.dptr_, ts_cpu.stride_ * sizeof(DType),
      ts.size(2) * sizeof(DType), ts.size(0) * ts.size(1),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  AccessElements<<<1, 1>>>(ts.shape_.Size(), ts.dptr_, result, indices);
  cudaDeviceSynchronize();

  for (int i = 0; i < (int) ts.shape_.Size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], i + 1);
  }

  FreeSpace(&ts);
  FreeSpace(&ts_cpu);
  cudaFree(result);
  cudaFree(indices);
}

}  // op
}  // mxnet

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  mshadow::InitTensorEngine<mshadow::gpu>();
  return RUN_ALL_TESTS();
}

