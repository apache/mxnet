/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
  MSHADOW_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceLastOp<cpu, DType>(param); })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceLastProp::CreateOperatorEx(Context ctx,
                                             std::vector<TShape> *in_shape,
                                             std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceLastParam);

MXNET_REGISTER_OP_PROPERTY(SequenceLast, SequenceLastProp)
    .describe(R"code(Takes the last element of a sequence.

This function takes an n-dimensional input array of the form
[max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array
of the form [batch_size, other_feature_dims].

Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be
an input array of positive ints of dimension [batch_size]. To use this parameter,
set `use_sequence_length` to `True`, otherwise each example in the batch is assumed
to have the max sequence length.

.. note:: Alternatively, you can also use `take` operator.

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

   // returns last sequence when sequence_length parameter is not used
   SequenceLast(x) = [[  19.,   20.,   21.],
                      [  22.,   23.,   24.],
                      [  25.,   26.,   27.]]

   // sequence_length is used
   SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
            [[  1.,   2.,   3.],
             [  4.,   5.,   6.],
             [  7.,   8.,   9.]]

   // sequence_length is used
   SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
            [[  1.,    2.,   3.],
             [  13.,  14.,  15.],
             [  25.,  26.,  27.]]

)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol",
                  "n-dimensional input array of the form [max_sequence_length,"
                  " batch_size, other_feature_dims] where n>2")
    .add_argument("sequence_length", "NDArray-or-Symbol",
                  "vector of sequence lengths of the form [batch_size]")
    .add_arguments(SequenceLastParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
