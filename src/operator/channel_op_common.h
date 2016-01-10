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
                        mshadow::Tensor<xpu, dim> *output, const int dimension) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  mshadow::Tensor<xpu, dim> out = *output;
  size_t size = input.size();

  index_t begin = 0;
  switch (dimension) {
    case 0: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(0);
          slice<0>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 1: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(1);
          slice<1>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 2: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(2);
          slice<2>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
    case 3: {
        for (index_t i = 0; i < size; ++i) {
          index_t end = begin + input[i].size(3);
          slice<3>(out, begin, end) = input[i];
          begin = end;
        }
        break;
    }
  }
}



template<typename xpu, int dim>
void Split(const mshadow::Tensor<xpu, dim> &input,
           std::vector<mshadow::Tensor<xpu, dim> > *output,
           const int dimension,
           std::vector<bool> mask = std::vector<bool>(31, true)) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;
  std::vector<mshadow::Tensor<xpu, dim> > out = *output;
  size_t size = out.size();
  index_t begin = 0;
  switch (dimension) {
    case 0: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(0);
        if (mask[i]) out[i] = slice<0>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 1: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(1);
        if (mask[i]) out[i] = slice<1>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 2: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(2);
        if (mask[i]) out[i] = slice<2>(input, begin, end);
        begin = end;
      }
      break;
    }
    case 3: {
      for (index_t i = 0; i < size; ++i) {
        index_t end = begin + out[i].size(3);
        if (mask[i]) out[i] = slice<3>(input, begin, end);
        begin = end;
      }
      break;
    }
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
