/*!
 * \file random.cc
 * \brief
 * \author Sebastian Nowozin
*/

#include "./random-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(RandomParam param) {
  return new RandomOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *RandomProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(RandomParam);

MXNET_REGISTER_OP_PROPERTY(Random, RandomProp)
.describe("Generate random numbers")
.add_arguments(RandomParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


