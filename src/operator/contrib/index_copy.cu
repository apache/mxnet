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
 * \file index_copy.cu
 * \brief
 */
#include "./index_copy-inl.h"

namespace mxnet {
namespace op {

struct index_copy_fwd_gpu {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  const DType* new_tensor,
                                  const IType* idx,
                                  DType* out_tensor,
                                  int dim_size) {
    int index = static_cast<int>(idx[i / dim_size]);
    out_tensor[index * dim_size + i % dim_size] = new_tensor[i];
  }
};

template<>
void IndexCopyForward<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK(req[0] != kAddTo);
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  const TBlob& out = outputs[0];
  const TBlob& original_tensor = inputs[0];
  const TBlob& idx_vector = inputs[1];
  const TBlob& copied_tensor = inputs[2];
  int dim_size = inputs[2].Size() / inputs[1].Size();
  // copy original tensor to output
  copy(s, out, original_tensor);
  // index copy
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(idx_vector.type_flag_, IType, {
      Kernel<index_copy_fwd_gpu, gpu>::Launch(
        s, copied_tensor.Size(), copied_tensor.dptr<DType>(),
        idx_vector.dptr<IType>(), out.dptr<DType>(), dim_size);
    });
  });
}

struct index_copy_bwd_gpu {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i,
                                  const DType* out_grad,
                                  DType* orig_grad,
                                  DType* new_grad,
                                  const IType* idx,
                                  int dim_size,
                                  int idx_size,
                                  OpReqType orig_req,
                                  OpReqType new_req) {
    int index = idx[i / dim_size];
    KERNEL_ASSIGN(new_grad[i], new_req, out_grad[index * dim_size + i % dim_size]);
    if (orig_req == kAddTo) {
      orig_grad[index * dim_size + i % dim_size] -= new_grad[i];
    } else if (orig_req == kNullOp) {
      return;
    } else {
      orig_grad[index * dim_size + i % dim_size] = 0;
    }
  }
};

template<>
void IndexCopyBackward<gpu>(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& index = inputs[2];
  const TBlob& in_grad_1 = outputs[0];
  const TBlob& in_grad_2 = outputs[2];
  int dim_size = inputs[3].Size() / inputs[2].Size();
  int index_size = inputs[2].Size();
  OpReqType orig_req = req[0];
  OpReqType new_req = req[2];
  // index_copy_backward
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(index.type_flag_, IType, {
      switch (orig_req) {
        case kNullOp:
          break;
        case kWriteTo:
        case kWriteInplace:
          copy(s, in_grad_1, out_grad);
          break;
        case kAddTo:
          Kernel<op_with_req<op::mshadow_op::plus, kWriteInplace>, gpu>::Launch(
            s, out_grad.Size(), in_grad_1.dptr<DType>(),
            out_grad.dptr<DType>(), in_grad_1.dptr<DType>());
      }
      Kernel<index_copy_bwd_gpu, gpu>::Launch(
        s, in_grad_2.Size(), out_grad.dptr<DType>(),
        in_grad_1.dptr<DType>(), in_grad_2.dptr<DType>(),
        index.dptr<IType>(), dim_size, index_size, orig_req, new_req);
    });
  });
}

NNVM_REGISTER_OP(_contrib_index_copy)
.set_attr<FCompute>("FCompute<gpu>", IndexCopyForward<gpu>);

NNVM_REGISTER_OP(_contrib_backward_index_copy)
.set_attr<FCompute>("FCompute<gpu>", IndexCopyBackward<gpu>);

}  // namespace op
}  // namespace mxnet
