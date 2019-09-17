#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// user defined unary operator addone
struct addone {
  // map can be template function
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return  a + static_cast<DType>(1);
  }
};
// user defined binary operator max of two
struct maxoftwo {
  // map can also be normal functions,
  // however, this can only be applied to float tensor
  MSHADOW_XINLINE static float Map(float a, float b) {
    if(a > b) return a;
    else return b;
  }
};

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<cpu>();
  // take first subscript of the tensor
  Stream<cpu> *stream_ = NewStream<cpu>(0);
  Tensor<cpu,2, float> mat = NewTensor<cpu>(Shape2(2,3), 0.0f, stream_);
  Tensor<cpu,2, float> mat2= NewTensor<cpu>(Shape2(2,3), 0.0f, stream_);

  mat[0][0] = -2.0f;
  mat = F<maxoftwo>(F<addone>(mat) + 0.5f, mat2);

  for (index_t i = 0; i < mat.size(0); ++i) {
    for (index_t j = 0; j < mat.size(1); ++j) {
      printf("%.2f ", mat[i][j]);
    }
    printf("\n");
  }
  FreeSpace(&mat); FreeSpace(&mat2);
  DeleteStream(stream_);
  // shutdown tensor enigne after usage
  ShutdownTensorEngine<cpu>();
  return 0;
}
