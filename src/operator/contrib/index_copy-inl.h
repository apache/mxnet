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
   * \brief implementation of index_copy operation
   */

#ifndef MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
#define MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
  
// Perform index_copy in mshadow  
struct index_copy {
  MSHADOW_XINLINE static void Map(int i, 
                                  int* index, 
                                  float* new_tensor, 
                                  float* old_tensor,
                                  int dim) {
    float* old_ptr = old_tensor + index[i] * dim;
    float* new_ptr = new_tensor + i * dim;
    // Copy new tensor to old tensor
    for (int idx = 0; idx < dim; ++idx) {
      *(old_ptr + idx) = *(new_ptr + idx);
    }
  }
};

template<typename xpu>
void IndexCopyCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  Kernel<index_copy, xpu>::Launch(s, inputs[1].Size(),
                            inputs[1].dptr<int>(),    // index_tensor
                            inputs[2].dptr<float>(),  // new_tensor
                            inputs[0].dptr<float>(),  // old_tensor
                            inputs[2].Size() / inputs[1].Size());  // dim   
}

inline bool IndexCopyShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  // 0. original tensor
  // 1. index tensot
  // 2. new tensor
  CHECK_EQ(in_attrs->size(), 3U);
  return true;
}

inline bool IndexCopyType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  // Check input tensor
  CHECK_EQ((*in_attrs)[0], mshadow::kFloat32);
  CHECK_EQ((*in_attrs)[1], mshadow::kInt32);
  CHECK_EQ((*in_attrs)[2], mshadow::kFloat32);
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_COPY_INL_H_
