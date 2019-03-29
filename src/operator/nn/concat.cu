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
 * \file concat.cu
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
void ConcatForwardImpl(const OpContext &ctx,
                       std::vector<mshadow::Tensor<gpu, 3, DType>>& data,
                       mshadow::Tensor<gpu, 3, DType>& out) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  std::vector<size_t> sizes(data.size(), 0);
  std::vector<size_t> indices(data.size() + 1, 0);
  std::vector<DType*> data_ptrs(data.size(), nullptr);
  size_t leading = out.shape_[0];
  size_t out_mid_size = out.shape_[1];
  size_t trailing = out.shape_[2];
  size_t num_inputs = data.size();
  size_t restsize = trailing * leading;
  for (size_t i = 0; i < num_inputs; ++i) {
    sizes[i] = data[i].shape_.Size() / restsize;
    data_ptrs[i] = data[i].dptr_;
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    indices[i + 1] = indices[i] + sizes[i];
  }

  size_t workspace_size = num_inputs * sizeof(DType*) + (num_inputs + 1) * sizeof(size_t);
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  DType **data_ptrs_gpu_ptr = reinterpret_cast<DType**>(workspace.dptr_);
  size_t *indices_gpu_ptr = reinterpret_cast<size_t*>(workspace.dptr_ + num_inputs * sizeof(DType*));
  Tensor<cpu, 1, DType*> data_ptrs_cpu(data_ptrs.data(), Shape1(num_inputs));
  Tensor<gpu, 1, DType*> data_ptrs_gpu(data_ptrs_gpu_ptr, Shape1(num_inputs), s);
  Tensor<cpu, 1, size_t> indices_cpu(indices.data(), Shape1(num_inputs + 1));
  Tensor<gpu, 1, size_t> indices_gpu(indices_gpu_ptr, Shape1(num_inputs + 1), s);

  // copy necessary data to GPU for kernel launch
  Copy(data_ptrs_gpu, data_ptrs_cpu, s);
  Copy(indices_gpu, indices_cpu, s);

  Kernel<ConcatenateKernel, gpu>::Launch(
    s, out.shape_.Size(), data_ptrs_gpu.dptr_, out.dptr_, indices_gpu.dptr_,
    data.size(), out_mid_size, trailing);
}

template<typename DType>
void ConcatBackwardImpl(const OpContext &ctx,
                        std::vector<mshadow::Tensor<gpu, 3, DType>>& grad_in,
                        mshadow::Tensor<gpu, 3, DType>& grad) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  std::vector<size_t> sizes(grad_in.size(), 0);
  std::vector<size_t> indices(grad_in.size() + 1, 0);
  std::vector<DType*> grad_in_ptrs(grad_in.size(), nullptr);
  size_t leading = grad.shape_[0];
  size_t grad_mid_size = grad.shape_[1];
  size_t trailing = grad.shape_[2];
  size_t num_inputs = grad_in.size();
  size_t restsize = trailing * leading;
  for (size_t i = 0; i < num_inputs; ++i) {
    sizes[i] = grad_in[i].shape_.Size() / restsize;
    grad_in_ptrs[i] = grad_in[i].dptr_;
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    indices[i + 1] = indices[i] + sizes[i];
  }

  size_t workspace_size = num_inputs * sizeof(DType*) + (num_inputs + 1) * sizeof(size_t);
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  DType **grad_in_ptrs_gpu_ptr = reinterpret_cast<DType**>(workspace.dptr_);
  size_t *indices_gpu_ptr = reinterpret_cast<size_t*>(workspace.dptr_ + num_inputs * sizeof(DType*));
  Tensor<cpu, 1, DType*> grad_in_ptrs_cpu(grad_in_ptrs.data(), Shape1(num_inputs));
  Tensor<gpu, 1, DType*> grad_in_ptrs_gpu(grad_in_ptrs_gpu_ptr, Shape1(num_inputs), s);
  Tensor<cpu, 1, size_t> indices_cpu(indices.data(), Shape1(num_inputs + 1));
  Tensor<gpu, 1, size_t> indices_gpu(indices_gpu_ptr, Shape1(num_inputs + 1), s);

  // copy necessary data to GPU for kernel launch
  Copy(grad_in_ptrs_gpu, grad_in_ptrs_cpu, s);
  Copy(indices_gpu, indices_cpu, s);

  Kernel<SplitKernel, gpu>::Launch(
    s, grad.shape_.Size(), grad.dptr_, grad_in_ptrs_gpu.dptr_, indices_gpu.dptr_,
    grad_in.size(), grad_mid_size, trailing);
}

static void ConcatComputeExGPU(const nnvm::NodeAttrs& attrs,
                               const OpContext& op_ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  if (common::ContainsOnlyStorage(inputs, kCSRStorage) &&
      outputs[0].storage_type() == kCSRStorage) {
    ConcatCSRImpl<gpu>(attrs, op_ctx, inputs, req, outputs);
  } else {
    LogUnimplementedOp(attrs, op_ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(Concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ConcatComputeExGPU);

NNVM_REGISTER_OP(_rnn_param_concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatCompute<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ConcatComputeExGPU);

NNVM_REGISTER_OP(_backward_Concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

