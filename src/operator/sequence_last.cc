/*!
 * Copyright (c) 2015 by Contributors
 * \file sequence_last.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_last-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(SequenceLastParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceLastOp<cpu, DType>(param); })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceLastProp::CreateOperatorEx(Context ctx,
                                             std::vector<TShape> *in_shape,
                                             std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceLastParam);

MXNET_REGISTER_OP_PROPERTY(SequenceLast, SequenceLastProp)
    .describe(R"code(Returns an (n-1)-dimensional array of the form [batch size, other dims]

This function takes an n-dimensional input array of the form [max sequence length, batch size, other dims]
and returns a (n-1)-dimensional array of the form [batch size, other dims].

Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be an input array of
positive ints of dimension [batch size]. To use this parameter, set `use_sequence_length` to `True`,
otherwise each example in the batch is assumed to have the max sequence length.

Example::

   x = [[[  1.,   2.,   3.],
         [  4.,   5.,   6.],
         [  7.,   8.,   9.]],

        [[ 10.,   11.,   12.],
         [ 13.,   14.,   15.],
         [ 16.,   17.,   18.]],

        [[  19.,   20.,   21.],
         [  22.,   23.,   24.],
         [  25.,   26.,   27.]]]

   // returns last sequence when sequence_length vector is not used
   SequenceLast(x) = [[  19.,   20.,   21.],
                      [  22.,   23.,   24.],
                      [  25.,   26.,   27.]]

   y = [1,1,1]

   // variable-length sequence y is used
   SequenceLast(x, y, use_sequence_length=True) =  [[  1.,   2.,   3.],
                                                    [  4.,   5.,   6.],
                                                    [  7.,   8.,   9.]]

   y = [1,2,3]

   // variable-length sequence y is used
   SequenceLast(x, y, use_sequence_length=True) = [[  1.,    2.,   3.],
                                                   [  13.,  14.,  15.],
                                                   [  25.,  26.,  27.]]

)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol",
                  "n-dimensional input array of the form [max sequence "
                  "length, batch size, other dims] where n>2")
    .add_argument("sequence_length", "NDArray-or-Symbol",
                  "vector of sequence lengths of the form [batch size]")
    .add_arguments(SequenceLastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
