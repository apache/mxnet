/*!
 * Copyright (c) 2017 by Contributors
 * \file sample_multinomial_op.h
 * \brief Operator for sampling from multinomial distributions
 */
#include "./sample_multinomial_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sample_multinomial)
.set_attr<FCompute>("FCompute<gpu>", SampleMultinomialForward<gpu>);


struct SampleMultinomialBackwardGPUKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t K, index_t M,
                                  DType* ograd, DType* dist, IType* out,
                                  DType* igrad) {
    for (index_t j = 0; j < M; ++j) {
      atomicAdd(&igrad[i*K + out[i*M + j]], ograd[i*M + j] / dist[i*K + out[i*M + j]]);
    }
  }
};


NNVM_REGISTER_OP(_backward_sample_multinomial)
.set_attr<FCompute>("FCompute<gpu>",
  SampleMultinomialBackward<SampleMultinomialBackwardGPUKernel, gpu>);


}  // namespace op
}  // namespace mxnet
