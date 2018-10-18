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

struct index_copy {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  int dim,
                                  IType* index,
                                  DType* new_tensor,
                                  DType* out_tensor) {
    DType* out_ptr = out_tensor + static_cast<int>(index[i]) * dim;
    DType* new_ptr = new_tensor + i * dim;
    for (int idx = 0; idx < dim; ++idx) {
      *(out_ptr + idx) = *(new_ptr + idx);
    }
  }
};

template<typename xpu>
void IndexCopyForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out = outputs[0];
  const TBlob& idx = inputs[1];
  int dim = inputs[2].Size() / inputs[1].Size();
  // copy all
  mxnet_op::copy(s, outputs[0], inputs[0]);
  // index copy
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(idx.type_flag_, IType, {
      Kernel<index_copy, xpu>::Launch(s, inputs[1].Size(),
                            dim,
                            inputs[1].dptr<IType>(),
                            inputs[2].dptr<DType>(),
                            outputs[0].dptr<DType>());
    })
  })
}

struct index_copy_assign {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  int dim,
                                  int index_size,
                                  DType* i_grad,
                                  IType* index,
                                  DType* o_grad_0,
                                  DType* o_grad_1) {
    // Copy to o_grad_1
    for (int p = 0; p < index_size; ++p) {
      int idx = static_cast<int>(index[p]);
      if (i >= idx*dim && i < (idx+1)*dim) {
        int offset = i - idx*dim;
        *(o_grad_1 + (p*dim) + offset) = *(i_grad + i);
        return;
      }
    }
    // Copy to o_grad_0
    *(o_grad_0 + i) = *(i_grad + i);
  }
};

template<typename xpu>
void IndexCopyBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int dim = inputs[3].Size() / inputs[2].Size();
  int index_size = inputs[2].Size();
  // index_copy_backward
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(inputs[2].type_flag_, IType, {
      Kernel<index_copy_assign, xpu>::Launch(s, inputs[0].Size(),
                                      dim,
                                      index_size,
                                      inputs[0].dptr<DType>(),
                                      inputs[2].dptr<IType>(),
                                      outputs[0].dptr<DType>(),
                                      outputs[2].dptr<DType>());
    })
  })
}

inline bool IndexCopyShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1);
  CHECK_EQ(in_attrs->at(0).ndim(), in_attrs->at(2).ndim());
  for (size_t i = 0; i < in_attrs->at(0).ndim(); ++i) {
    if (i == 0) {
      CHECK_GE(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    } else {
      CHECK_EQ(in_attrs->at(0)[i], in_attrs->at(2)[i]);
    }
  }
  CHECK_EQ(in_attrs->at(1)[0], in_attrs->at(2)[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool IndexCopyType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
