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
 * \file index_copy-inl.h
 * \brief implementation of index_copy tensor operation
 */

#ifndef MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
#define MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include <algorithm>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template<int req>
struct index_copy_forward {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  int dim,
                                  IType* index,
                                  DType* new_tensor,
                                  DType* out_tensor) {
    DType* out_ptr = out_tensor + static_cast<int>(index[i]) * dim;
    DType* new_ptr = new_tensor + i * dim;
    for (int idx = 0; idx < dim; ++idx) {
      KERNEL_ASSIGN(out_ptr[idx], req, new_ptr[idx]);
    }
  }
};

template<typename xpu>
void IndexCopyForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out = outputs[0];
  const TBlob& original_tensor = inputs[0];
  const TBlob& idx_vector = inputs[1];
  const TBlob& copied_tensor = inputs[2];
  int dim = inputs[2].Size() / inputs[1].Size();
  // copy original tensor to output
  mxnet_op::copy(s, out, original_tensor);
  // index copy
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(idx_vector.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<index_copy_forward<req_type>, xpu>::Launch(s,
                              idx_vector.Size(), dim,
                              idx_vector.dptr<IType>(),
                              copied_tensor.dptr<DType>(),
                              out.dptr<DType>());
      });
    });
  });
}

template<int req>
struct index_copy_backward {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  int dim,
                                  int index_size,
                                  DType* out_grad,
                                  IType* index,
                                  DType* in_grad_1,
                                  DType* in_grad_2) {
    // Copy to in_grad_2
    for (int p = 0; p < index_size; ++p) {
      int idx = static_cast<int>(index[p]);
      if (i >= idx*dim && i < (idx+1)*dim) {
        int offset = i - idx*dim;
        KERNEL_ASSIGN(in_grad_2[p*dim+offset], req, out_grad[i]);
        return;
      }
    }
    // Copy to in_grad_1
    KERNEL_ASSIGN(in_grad_1[i], req, out_grad[i]);
  }
};

template<typename xpu>
void IndexCopyBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& index = inputs[2];
  const TBlob& in_grad_1 = outputs[0];
  const TBlob& in_grad_2 = outputs[2];
  int dim = inputs[3].Size() / inputs[2].Size();
  int index_size = inputs[2].Size();
  // index_copy_backward
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(index.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<index_copy_backward<req_type>, xpu>::Launch(s,
                                      out_grad.Size(),
                                      dim, index_size,
                                      out_grad.dptr<DType>(),
                                      index.dptr<IType>(),
                                      in_grad_1.dptr<DType>(),
                                      in_grad_2.dptr<DType>());
      });
    });
  });
}

inline bool IndexCopyShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  // inputs[0]: original tensor
  // inputs[1]: index vector
  // inputs[2]: copied tensor
  CHECK_EQ(in_attrs->size(), 3U);
  // outputs[0]: a new tensor
  CHECK_EQ(out_attrs->size(), 1U);
  // inputs[1] must be a vector
  CHECK_EQ(in_attrs->at(1).ndim(), 1);
  // Shape matching
  CHECK_EQ(in_attrs->at(0).ndim(), in_attrs->at(2).ndim());
  for (size_t i = 0; i < in_attrs->at(0).ndim(); ++i) {
    if (i == 0) {
      CHECK_GE(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    } else {
      CHECK_EQ(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    }
  }
  // The the length of the fitrst dim of copied tensor
  // must equal to the size of index vector
  CHECK_EQ(in_attrs->at(1)[0], in_attrs->at(2)[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U &&
         out_attrs->at(0).Size() != 0U;
}

inline bool IndexCopyType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
