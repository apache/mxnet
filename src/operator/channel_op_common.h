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

using mshadow::expr::concat;


template<typename xpu, int dim>
inline void Concatenate(const std::vector<mshadow::Tensor<xpu, dim> > &input,
                        mshadow::Tensor<xpu, dim> *output) {
  mshadow::Tensor<xpu, dim> out = *output;
  size_t size = input.size();
  switch (size) {
    case 2:
      out = concat<1>(input[0], input[1]);
      break;
    case 3:
      out = concat<1>(input[0],
                      concat<1>(input[1], input[2]));
      break;
    case 4:
      out = concat<1>(input[0],
                      concat<1>(input[1],
                                concat<1>(input[2], input[3])));
      break;
    case 5:
      out = concat<1>(input[0],
                      concat<1>(input[1],
                                concat<1>(input[2],
                                          concat<1>(input[3], input[4]))));
      break;
    default:
      LOG(FATAL) << "Incorrect concat size: " << size;
  }
}

template<typename xpu, int dim>
void Split(const mshadow::Tensor<xpu, dim> &input,
           std::vector<mshadow::Tensor<xpu, dim> > *output) {
  std::vector<mshadow::Tensor<xpu, dim> > out = *output;
  size_t size = out.size();
  switch (size) {
    case 2: {
      concat<1>(out[0], out[1]) = input;
      break;
    }
    case 3: {
      concat<1>(out[0],
                concat<1>(out[1], out[2])) = input;
      break;
    }
    case 4: {
      concat<1>(out[0],
                concat<1>(out[1],
                          concat<1>(out[2], out[3]))) = input;
      break;
    }
    case 5: {
      concat<1>(out[0],
                concat<1>(out[1],
                          concat<1>(out[2],
                                    concat<1>(out[3], out[4])))) = input;
      break;
    }
    default:
      LOG(FATAL) << "Incorrect concat size: " << size;
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
