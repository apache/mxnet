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
 * \file square_sum.cu
 * \brief GPU Implementation of square_sum op.
 */
#include "./square_sum-inl.h"

namespace mxnet {
namespace op {

template<>
void CheckSameIdx<gpu>(mshadow::Stream<gpu>* s,
                       const TBlob ograd_row_idx,
                       const TBlob in_row_idx) {
  MSHADOW_IDX_TYPE_SWITCH(ograd_row_idx.type_flag_, IType, {
    const IType* ograd_idx = ograd_row_idx.dptr<IType>();
    const IType* in_idx = in_row_idx.dptr<IType>();
    const nnvm::dim_t idx_size = ograd_row_idx.Size();
    int32_t is_same = 0;
    int32_t* is_same_ptr = NULL;
    CUDA_CALL(cudaMalloc(&is_same_ptr, sizeof(int32_t)));
    mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, 1, is_same_ptr);
    mxnet_op::Kernel<CheckSameIdxKernel, gpu>::Launch(s, idx_size, ograd_idx, in_idx, &is_same);
    CUDA_CALL(cudaMemcpy(&is_same, is_same_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_EQ(is_same, 0) << "SquareSumRspGradImpl only supports"
                            " equal ograd_row_idx and input_row_idx"
                            " when ograd and input are both"
                            " row-sparse and input data is not a full"
                            " row-sparse matrix";
    CUDA_CALL(cudaFree(is_same_ptr));
  })
}


NNVM_REGISTER_OP(_square_sum)
.set_attr<FComputeEx>("FComputeEx<gpu>", SquareSumOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_square_sum)
.set_attr<FComputeEx>("FComputeEx<gpu>", SquareSumOpBackwardEx<gpu>);

}  // namespace op
}  // namespace mxnet
