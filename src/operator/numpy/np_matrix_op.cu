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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op.cu
 * \brief GPU Implementation of numpy matrix operations
 */

#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_transpose)
.set_attr<FCompute>("FCompute<gpu>", NumpyTranspose<gpu>);

NNVM_REGISTER_OP(_np_reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_np_squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_npi_concatenate)
.set_attr<FCompute>("FCompute<gpu>", NumpyConcatenateForward<gpu>);

NNVM_REGISTER_OP(_backward_np_concat)
.set_attr<FCompute>("FCompute<gpu>", NumpyConcatenateBackward<gpu>);

NNVM_REGISTER_OP(_npi_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_npi_vstack)
.set_attr<FCompute>("FCompute<gpu>", NumpyVstackForward<gpu>);

NNVM_REGISTER_OP(_backward_np_vstack)
.set_attr<FCompute>("FCompute<gpu>", NumpyVstackBackward<gpu>);

NNVM_REGISTER_OP(_npi_hstack)
.set_attr<FCompute>("FCompute<gpu>", HStackCompute<gpu>);

NNVM_REGISTER_OP(_backward_np_hstack)
.set_attr<FCompute>("FCompute<gpu>", HStackGradCompute<gpu>);

NNVM_REGISTER_OP(_npi_dstack)
.set_attr<FCompute>("FCompute<gpu>", DStackCompute<gpu>);

NNVM_REGISTER_OP(_backward_np_dstack)
.set_attr<FCompute>("FCompute<gpu>", DStackGradCompute<gpu>);

NNVM_REGISTER_OP(_npi_column_stack)
.set_attr<FCompute>("FCompute<gpu>", NumpyColumnStackForward<gpu>);

NNVM_REGISTER_OP(_backward_np_column_stack)
.set_attr<FCompute>("FCompute<gpu>", NumpyColumnStackBackward<gpu>);

NNVM_REGISTER_OP(_np_roll)
.set_attr<FCompute>("FCompute<gpu>", NumpyRollCompute<gpu>);

template<>
void NumpyFlipForwardImpl<gpu>(const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<TBlob>& outputs,
                               const std::vector<index_t>& stride_,
                               const std::vector<index_t>& trailing_,
                               const index_t& flip_index) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  mshadow::Tensor<gpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, uint8_t>(
      mshadow::Shape1(flip_index * sizeof(index_t) * 2), s);

  auto stride_workspace = workspace.dptr_;
  auto trailing_workspace = workspace.dptr_ + flip_index * sizeof(index_t);

  cudaMemcpyAsync(stride_workspace, thrust::raw_pointer_cast(stride_.data()),
                  stride_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));
  cudaMemcpyAsync(trailing_workspace, thrust::raw_pointer_cast(trailing_.data()),
                  trailing_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mxnet_op::Kernel<reverse, gpu>::Launch(s, inputs[0].Size(), flip_index,
      inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
      reinterpret_cast<index_t*>(stride_workspace), reinterpret_cast<index_t*>(trailing_workspace));
  });
}

NNVM_REGISTER_OP(_npi_flip)
.set_attr<FCompute>("FCompute<gpu>", NumpyFlipForward<gpu>);

NNVM_REGISTER_OP(_backward_npi_flip)
.set_attr<FCompute>("FCompute<gpu>", NumpyFlipForward<gpu>);

NNVM_REGISTER_OP(_np_moveaxis)
.set_attr<FCompute>("FCompute<gpu>", NumpyMoveaxisCompute<gpu>);

NNVM_REGISTER_OP(_npi_rot90)
.set_attr<FCompute>("FCompute<gpu>", NumpyRot90Compute<gpu>);

NNVM_REGISTER_OP(_npi_hsplit)
.set_attr<FCompute>("FCompute<gpu>", HSplitOpForward<gpu>);

NNVM_REGISTER_OP(_npi_hsplit_backward)
.set_attr<FCompute>("FCompute<gpu>", HSplitOpBackward<gpu>);

NNVM_REGISTER_OP(_npi_dsplit)
.set_attr<FCompute>("FCompute<gpu>", SplitOpForward<gpu>);

NNVM_REGISTER_OP(_npx_reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_np_diag)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_diag)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagOpBackward<gpu>);

NNVM_REGISTER_OP(_np_diagonal)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagonalOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_diagonal)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagonalOpBackward<gpu>);

NNVM_REGISTER_OP(_np_diagflat)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagflatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_np_diagflat)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagflatOpBackward<gpu>);

NNVM_REGISTER_OP(_npi_diag_indices_from)
.set_attr<FCompute>("FCompute<gpu>", NumpyDiagIndicesFromForward<gpu>);

}  // namespace op
}  // namespace mxnet
