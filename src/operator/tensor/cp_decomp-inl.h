#ifndef MXNET_OPERATOR_TENSOR_CP_DECOMP_INL_H
#define MXNET_OPERATOR_TENSOR_CP_DECOMP_INL_H
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <vector>

namespace mxnet {
namespace op {
template <int order, typename DType>
inline int CPDecompForward
  (mshadow::Tensor<cpu, 1, DType> &eigvals,
  std::vector<mshadow::Tensor<cpu, 2, DType> > &factors,
  const mshadow::Tensor<cpu, order, DType> &in,
  int k,
  DType eps = 1e-6,
  int max_iter = 100,
  mshadow::Stream<cpu> *stream = NULL);

template <int order, typename DType>
inline void Unfold
  (mshadow::Tensor<cpu, 2, DType> &dst,
  const mshadow::Tensor<cpu, order, DType> &src,
  int mode,
  mshadow::Stream<cpu> *stream = NULL);
}
}

#endif

