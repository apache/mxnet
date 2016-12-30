/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_convolution.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./cudnn_convolution-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include <sstream>
#include <unordered_map>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
CuDNNAlgoReg* CuDNNAlgoReg::Get() {
  static CuDNNAlgoReg* ptr = new CuDNNAlgoReg();
  return ptr;
}
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet
