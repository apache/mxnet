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
 * \file sequence_mask.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./sequence_mask-inl.h"

namespace mxnet {
namespace op {

// (seqlen, batch, rest) case
template <int req>
struct SequenceMask0GPUKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType *in, const IType *idx,
                                  index_t max_s_len, index_t batch_size,
                                  index_t restsize, DType value) {
    index_t batch = i / restsize % batch_size;
    const index_t seqpos = static_cast<int>(idx[batch]);
    index_t seq = i / restsize / batch_size;
    if (seq >= seqpos) {
      KERNEL_ASSIGN(in[i], req, value);
    }
  }
};

// (batch, seqlen, rest) case
template <int req>
struct SequenceMask1GPUKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType *in, const IType *idx,
                                  index_t max_s_len, index_t batch_size,
                                  index_t restsize, DType value) {
    index_t batch = i / restsize / max_s_len;
    const index_t seqpos = static_cast<int>(idx[batch]);
    index_t seq = i / restsize % max_s_len;
    if (seq >= seqpos) {
      KERNEL_ASSIGN(in[i], req, value);
    }
  }
};

template<typename DType, typename IType>
void SequenceMaskExec(
       const mshadow::Tensor<gpu, 3, DType> &data,
       const mshadow::Tensor<gpu, 1, IType> &indices,
       const OpReqType req, mshadow::Stream<gpu> *const s,
       int axis, DType val) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  index_t batch = indices.size(0);
  index_t max_seq_len = data.size(axis);
  index_t restsize = data.size(2);

  MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
    if (axis == 1) {
      Kernel<SequenceMask1GPUKernel<req_type>, gpu>::Launch(
        s, data.shape_.Size(), data.dptr_, indices.dptr_, max_seq_len, batch, restsize,
        val);
    } else {
      Kernel<SequenceMask0GPUKernel<req_type>, gpu>::Launch(
        s, data.shape_.Size(), data.dptr_, indices.dptr_, max_seq_len, batch, restsize,
        val);
    }
  });
}

template <> Operator *CreateOp<gpu>(SequenceMaskParam param, int dtype, int itype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
      MSHADOW_TYPE_SWITCH(itype, IType, {
          op = new SequenceMaskOp<gpu, DType, IType>(param);
        });
    });
  return op;
}

}  // namespace op
}  // namespace mxnet
