/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_algoreg.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./cudnn_algoreg-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include <sstream>
#include <unordered_map>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
template<>
CuDNNAlgoReg<ConvolutionParam> *CuDNNAlgoReg<ConvolutionParam>::Get() {
  static CuDNNAlgoReg<ConvolutionParam> inst;
  return &inst;
}

template<>
CuDNNAlgoReg<DeconvolutionParam> *CuDNNAlgoReg<DeconvolutionParam>::Get() {
  static CuDNNAlgoReg<DeconvolutionParam> inst;
  return &inst;
}
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet
