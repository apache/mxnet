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
 * \file sequence_mask.cc
 * \brief
 * \author Sebastian Bodenstein
*/
#include "./sequence_mask-inl.h"

namespace mxnet {
namespace op {

// (seqlen, batch, rest) case
template <int req>
struct SequenceMask0CPUKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int batch, DType *in, const IType *idx,
                                  index_t max_s_len, index_t batch_size,
                                  index_t restsize, DType value) {
    const index_t seqpos = static_cast<int>(idx[batch]);
#pragma unroll
    for (index_t s = seqpos; s < max_s_len; ++s) {
      index_t incr = (s * batch_size * restsize) + (batch * restsize);
#pragma unroll
      for (index_t r = 0; r < restsize; ++r)
        KERNEL_ASSIGN(in[incr + r], req, value);
    }
  }
};

// (batch, seqlen, rest) case
template <int req>
struct SequenceMask1CPUKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int batch, DType *in, const IType *idx,
                                  index_t max_s_len, index_t batch_size,
                                  index_t restsize, DType value) {
    const index_t seqpos = static_cast<int>(idx[batch]);
#pragma unroll
    for (index_t s = seqpos; s < max_s_len; ++s) {
      index_t incr = (batch * max_s_len * restsize) + (s * restsize);
#pragma unroll
      for (index_t r = 0; r < restsize; ++r)
        KERNEL_ASSIGN(in[incr + r], req, value);
    }
  }
};

template<typename DType, typename IType>
void SequenceMaskExec(
       const mshadow::Tensor<cpu, 3, DType> &data,
       const mshadow::Tensor<cpu, 1, IType> &indices,
       const OpReqType req, mshadow::Stream<cpu> *const s,
       int axis, DType val) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  index_t batch = indices.size(0);
  index_t max_seq_len = data.size(axis);
  index_t restsize = data.size(2);

  MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
    if (axis == 1) {
      Kernel<SequenceMask1CPUKernel<req_type>, cpu>::Launch(
        s, batch, data.dptr_, indices.dptr_, max_seq_len, batch, restsize,
        val);
    } else {
      Kernel<SequenceMask0CPUKernel<req_type>, cpu>::Launch(
        s, batch, data.dptr_, indices.dptr_, max_seq_len, batch, restsize,
        val);
    }
  });
}

template <>
Operator *CreateOp<cpu>(SequenceMaskParam param, int dtype, int itype) {
  Operator *op = nullptr;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
      MSHADOW_TYPE_SWITCH(itype, IType, {
          op = new SequenceMaskOp<cpu, DType, IType>(param);
        });
    });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SequenceMaskProp::CreateOperatorEx(Context ctx,
                                             mxnet::ShapeVector *in_shape,
                                             std::vector<int> *in_type) const {
  if (in_type->size() >= 2 && (*in_type)[1] != -1) {
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], (*in_type)[1]);
  }

  // sequence_length not passed in, so fall back to using input array dtype for second argument
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceMaskParam);

MXNET_REGISTER_OP_PROPERTY(SequenceMask, SequenceMaskProp)
    .describe(R"code(Sets all elements outside the sequence to a constant value.

This function takes an n-dimensional input array of the form
[max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.

Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`
should be an input array of positive ints of dimension [batch_size].
To use this parameter, set `use_sequence_length` to `True`,
otherwise each example in the batch is assumed to have the max sequence length and
this operator works as the `identity` operator.

Example::

   x = [[[  1.,   2.,   3.],
         [  4.,   5.,   6.]],

        [[  7.,   8.,   9.],
         [ 10.,  11.,  12.]],

        [[ 13.,  14.,   15.],
         [ 16.,  17.,   18.]]]

   // Batch 1
   B1 = [[  1.,   2.,   3.],
         [  7.,   8.,   9.],
         [ 13.,  14.,  15.]]

   // Batch 2
   B2 = [[  4.,   5.,   6.],
         [ 10.,  11.,  12.],
         [ 16.,  17.,  18.]]

   // works as identity operator when sequence_length parameter is not used
   SequenceMask(x) = [[[  1.,   2.,   3.],
                       [  4.,   5.,   6.]],

                      [[  7.,   8.,   9.],
                       [ 10.,  11.,  12.]],

                      [[ 13.,  14.,   15.],
                       [ 16.,  17.,   18.]]]

   // sequence_length [1,1] means 1 of each batch will be kept
   // and other rows are masked with default mask value = 0
   SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
                [[[  1.,   2.,   3.],
                  [  4.,   5.,   6.]],

                 [[  0.,   0.,   0.],
                  [  0.,   0.,   0.]],

                 [[  0.,   0.,   0.],
                  [  0.,   0.,   0.]]]

   // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
   // and other rows are masked with value = 1
   SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
                [[[  1.,   2.,   3.],
                  [  4.,   5.,   6.]],

                 [[  7.,   8.,   9.],
                  [  10.,  11.,  12.]],

                 [[   1.,   1.,   1.],
                  [  16.,  17.,  18.]]]

)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol",
                  "n-dimensional input array of the form [max_sequence_length,"
                  " batch_size, other_feature_dims] where n>2")
    .add_argument("sequence_length", "NDArray-or-Symbol",
                  "vector of sequence lengths of the form [batch_size]")
    .add_arguments(SequenceMaskParam::__FIELDS__());

NNVM_REGISTER_OP(SequenceMask)
.add_alias("_npx_SequenceMask");

}  // namespace op
}  // namespace mxnet
