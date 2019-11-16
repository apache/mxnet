#include "./np_constraint_check.h"

namespace mxnet {
namespace op {

template<>
void GetReduceOutput<gpu>(mshadow::Stream<gpu> *s, const TBlob &output_blob, bool *red_output) {
  bool tmp = true;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  CUDA_CALL(cudaMemcpyAsync(&tmp, output_blob.dptr<bool>(),
                            sizeof(bool), cudaMemcpyDeviceToHost,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  *red_output = static_cast<bool>(tmp);
}

NNVM_REGISTER_OP(_npx_constraint_check)
.set_attr<FCompute>("FCompute<gpu>", ConstraintCheckForward<gpu>);

}  // namespace op
}  // namespace mxnet