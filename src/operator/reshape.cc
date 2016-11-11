/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./reshape-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ReshapeParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ReshapeOp<cpu, DType>(param);
  });
  return op;
}

Operator* ReshapeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.describe("Reshape input according to a target shape spec.\n"
"The target shape is a tuple and can be a simple list of dimensions "
"such as (12,3) or it can incorporate special codes that correspond "
"to contextual operations that refer to the input dimensions.\n"
"The special codes are all expressed as integers less than 1. These "
"codes effectively refer to a machine that pops input dims off the "
"beginning of the input dims list and pushes resulting output dims "
"onto the end of the output dims list, which starts empty. The codes "
"are:\n"
"  0  Copy     Pop one input dim and push it onto the output dims\n"
" -1  Infer    Push a dim that is inferred later from all other output dims\n"
" -2  CopyAll  Pop all remaining input dims and push them onto output dims\n"
" -3  Merge2   Pop two input dims, multiply them, and push result\n"
" -4  Split2   Pop one input dim, and read two next target shape specs,\n"
"              push them both onto output dims (either can be -1 and will\n"
"              be inferred from the other\n"
" The exact mathematical behavior of these codes is given in the "
"description of the 'shape' parameter. All non-codes (positive "
"integers) just pop a dim off the input dims (if any), throw it away, "
"and then push the specified integer onto the output dims.\n"
"Examples:\n"
"Type     Input      Target            Output\n"
"Copy     (2,3,4)    (4,0,2)           (4,3,2)\n"
"Copy     (2,3,4)    (2,0,0)           (2,3,4)\n"
"Infer    (2,3,4)    (6,1,-1)          (6,1,4)\n"
"Infer    (2,3,4)    (3,-1,8)          (3,1,8)\n"
"CopyAll  (9,8,7)    (-2)              (9,8,7)\n"
"CopyAll  (9,8,7)    (9,-2)            (9,8,7)\n"
"CopyAll  (9,8,7)    (-2,1,1)          (9,8,7,1,1)\n"
"Merge2   (3,4)      (-3)              (12)\n"
"Merge2   (3,4,5)    (-3,0)            (12,5)\n"
"Merge2   (3,4,5)    (0,-3)            (3,20)\n"
"Merge2   (3,4,5,6)  (-3,0,0)          (12,5,6)\n"
"Merge2   (3,4,5,6)  (-3,-2)           (12,5,6)\n"
"Split2   (12)       (-4,6,2)          (6,2)\n"
"Split2   (12)       (-4,2,6)          (2,6)\n"
"Split2   (12)       (-4,-1,6)         (2,6)\n"
"Split2   (12,9)     (-4,2,6,0)        (2,6,9)\n"
"Split2   (12,9,9,9) (-4,2,6,-2)       (2,6,9,9,9)\n"
"Split2   (12,12)    (-4,2,-1,-4,-1,2) (2,6,6,2)\n"
)
.add_argument("data", "Symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.describe(R"(Flatten input into 2D by collapsing all the higher dimensions.
A (d1, d2, ..., dK) tensor is flatten to (d1, d2* ... *dK) matrix.)")
.add_argument("data", "Symbol", "Input data to flatten.");
}  // namespace op
}  // namespace mxnet
