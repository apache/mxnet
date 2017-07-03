/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage-inl.cuh
 * \brief implementation of cast_storage op on GPU
 */
#ifndef MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>

namespace mxnet {
namespace op {

inline void CastStorageDnsRspImpl(mshadow::Stream<gpu>* s, const TBlob& dns, NDArray* rsp) {
  LOG(FATAL) << "CastStorageDnsRspImpl gpu version is not implemented.";
}

inline void CastStorageDnsCsrImpl(mshadow::Stream<gpu>* s, const TBlob& dns, NDArray* csr) {
  LOG(FATAL) << "CastStorageDnsCsrImpl gpu version is not implemented.";
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
