/*!
 * \file random.cc
 * \brief
 * \author Sebastian Nowozin
*/

#include "./random-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(RandomParam param) {
  return new RandomOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet



