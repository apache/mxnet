/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_op_common.h
 * \brief common function used for sequence layers
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
#define MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include "./operator_common.h"

namespace mxnet {
namespace op {

template <typename DType>
void IndexTensorToVector(mshadow::Tensor<gpu, 1, DType> data,
                         std::vector<index_t> *index_vec) {
  int max_seq_len = data.shape_.Size();
#if MXNET_USE_CUDA
  DType *temp_index =
      reinterpret_cast<DType *>(malloc(sizeof(DType) * max_seq_len));
  cudaError_t cuda_status =
      cudaMemcpyAsync(temp_index, data.dptr_, max_seq_len * sizeof(DType),
                      cudaMemcpyDeviceToHost, data.stream_->stream_);
  CHECK_EQ(cuda_status, cudaSuccess) << "cuda memcpy label error";
  for (int i = 0; i < max_seq_len; ++i) {
    (*index_vec)[i] = static_cast<index_t>(temp_index[i]);
  }
  free(temp_index);
#endif
}
template <typename DType>
void IndexTensorToVector(mshadow::Tensor<cpu, 1, DType> data,
                         std::vector<index_t> *index_vec) {
  int max_seq_len = data.shape_.Size();
  DType *index_array = static_cast<DType *>(data.dptr_);
  for (int i = 0; i < max_seq_len; ++i)
    (*index_vec)[i] = static_cast<index_t>(index_array[i]);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_OP_COMMON_H_
