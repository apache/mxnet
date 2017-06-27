/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DropoutOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe(R"(Applies dropout operation to input array.

- During training, each element of the input is set to zero with probability p.
  The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  sum of the input unchanged.

- During testing, this operator does not change the input.

Example::

  random.seed(998)
  input_array = array([[3., 0.5,  -0.5,  2., 7.],
                      [2., -0.4,   7.,  3., 0.2]])
  a = symbol.Variable('a')
  dropout = symbol.Dropout(a, p = 0.2)
  executor = dropout.simple_bind(a = input_array.shape)

  ## If training
  executor.forward(is_train = True, a = input_array)
  executor.outputs
  [[ 3.75   0.625 -0.     2.5    8.75 ]
   [ 2.5   -0.5    8.75   3.75   0.   ]]

  ## If testing
  executor.forward(is_train = False, a = input_array)
  executor.outputs
  [[ 3.     0.5   -0.5    2.     7.   ]
   [ 2.    -0.4    7.     3.     0.2  ]]
)" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to which dropout will be applied.")
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
