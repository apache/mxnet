/*!
 * Copyright (c) 2015 by Contributors
 * \file channel_op_common.h
 * \brief common function used for concat and split channel
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#define MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include "./operator_common.h"

namespace mxnet {
namespace op {

template<typename xpu, int dim>
inline void Concatenate(const std::vector<mshadow::Tensor<xpu, dim> > &input,
                        mshadow::Tensor<xpu, dim> *output) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  mshadow::Tensor<xpu, dim> out = *output;
  size_t size = input.size();
  index_t begin = 0;
  for (index_t i = 0; i < size; ++i) {
    index_t end = begin + input[i].size(1);
    slice<1>(out, begin, end) = input[i];
    begin = end;
  }
}

template<typename xpu, int dim>
void Split(const mshadow::Tensor<xpu, dim> &input,
           std::vector<mshadow::Tensor<xpu, dim> > *output) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  std::vector<mshadow::Tensor<xpu, dim> > out = *output;
  size_t size = out.size();
  index_t begin = 0;
  for (index_t i = 0; i < size; ++i) {
    index_t end = begin + out[i].size(1);
    out[i] = slice<1>(input, begin, end);
    begin = end;
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
