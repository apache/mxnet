// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<gpu>();
  // create a 2 x 5 tensor, from existing space
  Stream<gpu> *sm1 = NewStream<gpu>(0);
  Stream<gpu> *sm2 = NewStream<gpu>(0);
  Tensor<gpu, 2, float> ts1 =
      NewTensor<gpu, float>(Shape2(2, 5), 0.0f, false, sm1);
  Tensor<gpu, 2, float> ts2 =
      NewTensor<gpu, float>(Shape2(2, 5), 0.0f, false, sm2);
  ts1 = 1; // Should use stream 1.
  ts2 = 2; // Should use stream 2. Can run in parallel with stream 1.
  Stream<gpu> *sm3 = NewStream<gpu>(0);
  Tensor<gpu, 2> res = NewTensor<gpu, float>(Shape2(2, 2), 0.0f, false, sm3);
  res = dot(ts1, ts2.T()); // Should use stream 3.

  Tensor<cpu, 2> cpu_res = NewTensor<cpu, float>(Shape2(2, 2), 0.0f);
  Copy(cpu_res, res, sm3);
  for (index_t i = 0; i < cpu_res.size(0); ++i) {
    for (index_t j = 0; j < cpu_res.size(1); ++j) {
      printf("%.2f ", cpu_res[i][j]);
    }
    printf("\n");
  }
  // shutdown tensor enigne after usage
  DeleteStream(sm1);
  DeleteStream(sm2);
  DeleteStream(sm3);
  ShutdownTensorEngine<gpu>();
  return 0;
}
