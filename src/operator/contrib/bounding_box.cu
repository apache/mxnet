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
  *  Copyright (c) 2017 by Contributors
  * \file bounding_box.cu
  * \brief Bounding box util functions and operators
  * \author Joshua Zhang
  */

#include "./bounding_box-inl.cuh"
#include "./bounding_box-inl.h"
#include "../elemwise_op_common.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

namespace {

using mshadow::Tensor;
using mshadow::Stream;

template <typename DType>
struct TempWorkspace {
  index_t scores_temp_space;
  DType* scores;
  index_t batch_temp_space;
  index_t* batches;
  index_t scratch_space;
  uint8_t* scratch;
  index_t nms_scratch_space;
  uint32_t* nms_scratch;
  index_t indices_temp_spaces;
  index_t* indices;
};

inline index_t ceil_div(index_t x, index_t y) {
  return (x + y - 1) / y;
}

inline index_t align(index_t x, index_t alignment) {
  return ceil_div(x, alignment)  * alignment;
}

template <typename DType>
__global__ void FilterAndPrepareAuxData_kernel(const DType* data, DType* out, DType* scores,
                                               index_t* batches, index_t num_elements_per_batch,
                                               const index_t element_width, const float threshold,
                                               const int id_index, const int score_index,
                                               const int background_id) {
  index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool first_in_element = (tid % element_width == 0);
  index_t my_batch = tid / (num_elements_per_batch * element_width);
  index_t start_of_my_element = tid - (tid % element_width);

  DType my_score = data[start_of_my_element + score_index];
  bool filtered_out = my_score <= threshold;
  if (id_index != -1 && background_id != -1) {
    DType my_id = data[start_of_my_element + id_index];
    filtered_out = filtered_out || (my_id == background_id);
  }
  if (!filtered_out) {
    out[tid] = data[tid];
  } else {
    out[tid] = -1;
    my_score = -1;
  }

  if (first_in_element) {
    index_t offset = tid / element_width;
    scores[offset] = my_score;
    batches[offset] = my_batch;
  }
}

template <typename DType>
void FilterAndPrepareAuxData(const Tensor<gpu, 3, DType>& data,
                             Tensor<gpu, 3, DType>* out,
                             const TempWorkspace<DType>& workspace,
                             const BoxNMSParam& param,
                             Stream<gpu>* s) {
  const int n_threads = 512;
  index_t N = data.shape_.Size();
  const auto blocks = ceil_div(N, n_threads);
  FilterAndPrepareAuxData_kernel<<<blocks,
                                   n_threads,
                                   0,
                                   Stream<gpu>::GetStream(s)>>>(
    data.dptr_, out->dptr_, workspace.scores,
    workspace.batches, data.shape_[1], data.shape_[2],
    param.valid_thresh, param.id_index,
    param.score_index, param.background_id);
}

template <bool write_whole_output, typename DType>
__global__ void CompactData_kernel(const index_t* indices, const DType* source,
                                   DType* destination, const index_t topk,
                                   const index_t element_width,
                                   const index_t num_elements_per_batch,
                                   const index_t N) {
  const index_t tid_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (index_t tid = tid_start; tid < N; tid += blockDim.x * gridDim.x) {
    const index_t my_element = tid / element_width;
    const index_t my_element_in_batch = my_element % num_elements_per_batch;
    if (write_whole_output && my_element_in_batch >= topk) {
      destination[tid] = -1;
    } else {
      const index_t source_element = indices[my_element];
      destination[tid] = source[source_element * element_width + tid % element_width];
    }
  }
}

template <typename DType>
void CompactData(const Tensor<gpu, 1, index_t>& indices,
                 const Tensor<gpu, 3, DType>& source,
                 Tensor<gpu, 3, DType>* destination,
                 const index_t topk,
                 Stream<gpu>* s) {
  const int n_threads = 512;
  const int max_blocks = 320;
  index_t N = source.shape_.Size();
  const auto blocks = std::min(ceil_div(N, n_threads), max_blocks);
  CompactData_kernel<true><<<blocks, n_threads, 0,
                             Stream<gpu>::GetStream(s)>>>(
    indices.dptr_, source.dptr_,
    destination->dptr_, topk,
    source.shape_[2], source.shape_[1], N);
}

template <typename DType>
void WorkspaceForSort(const int num_batch,
                      const int num_elem,
                      const int width_elem,
                      const int alignment,
                      TempWorkspace<DType>* workspace) {
  const index_t sort_scores_temp_space = mxnet::op::SortByKeyWorkspaceSize<DType, index_t, gpu>(num_batch * num_elem);
  const index_t sort_batch_temp_space = mxnet::op::SortByKeyWorkspaceSize<index_t, index_t, gpu>(num_batch * num_elem);
  workspace->scratch_space = align(std::max(sort_scores_temp_space,
                                            sort_batch_temp_space),
                                   alignment);
}

template <typename DType>
__global__ void CalculateGreedyNMSResults_kernel(const DType* data, uint32_t* result,
                                                 const index_t current_start);

template <typename DType>
struct NMS {
  static const int THRESHOLD = 1024;

  void operator()(Tensor<gpu, 3, DType>* data,
                  Tensor<gpu, 2, uint32_t>* scratch,
                  const index_t topk,
                  const BoxNMSParam& param,
                  Stream<gpu>* s) {
    const int n_threads = 512;
    const index_t n_batch = data->shape_[0];
    for (index_t current_start = 0; current_start < topk; current_start += THRESHOLD) {
      const index_t n_elems = topk - current_start;
      const int n_blocks = ceil_div(THRESHOLD / (sizeof(uint32_t) * 8) * n_elems * n_batch, n_threads);
      CalculateGreedyNMSResults_kernel<<<n_blocks, n_threads, 0, Stream<gpu>::GetStream(s)>>>(
          data->dptr_, scratch->dptr_, current_start);
    }
  }
};

template <typename DType>
__global__ void CalculateGreedyNMSResults_kernel(const DType* data, uint32_t* result,
                                                 const index_t current_start) {
}

template <typename DType>
TempWorkspace<DType> GetWorkspace(const int num_batch,
                                  const int num_elem,
                                  const int width_elem,
                                  const index_t topk,
                                  const OpContext& ctx) {
  TempWorkspace<DType> workspace;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  const int alignment = 128;

  // Get the workspace size
  workspace.scores_temp_space = align(num_batch * num_elem * sizeof(DType), alignment);
  workspace.batch_temp_space = align(num_batch * num_elem * sizeof(index_t), alignment);
  workspace.indices_temp_spaces = align(num_batch * num_elem * sizeof(index_t), alignment);
  WorkspaceForSort(num_batch, num_elem, width_elem, alignment, &workspace);
  // Place for a buffer
  workspace.scratch_space = std::max(workspace.scratch_space,
                                     align(num_batch * num_elem * width_elem * sizeof(DType),
                                           alignment));
  workspace.nms_scratch_space = align(NMS<DType>::THRESHOLD / (sizeof(uint32_t) * 8) *
                                      num_batch * topk * sizeof(uint32_t), alignment);

  const index_t workspace_size = workspace.scores_temp_space +
                                 workspace.batch_temp_space +
                                 workspace.scratch_space +
                                 workspace.nms_scratch_space +
                                 workspace.indices_temp_spaces;

  // Obtain the memory for workspace
  Tensor<gpu, 1, DType> scratch_memory = ctx.requested[box_nms_enum::kTempSpace]
    .get_space_typed<gpu, 1, DType>(mshadow::Shape1(workspace_size), s);

  // Populate workspace pointers
  workspace.scores = scratch_memory.dptr_;
  workspace.batches = reinterpret_cast<index_t*>(reinterpret_cast<uint8_t*>(workspace.scores) +
                                                 workspace.scores_temp_space);
  workspace.scratch = reinterpret_cast<uint8_t*>(workspace.batches) +
                                                 workspace.batch_temp_space;
  workspace.nms_scratch = reinterpret_cast<uint32_t*>(workspace.scratch +
                                                      workspace.scratch_space);
  workspace.indices = reinterpret_cast<index_t*>(
      reinterpret_cast<uint8_t*>(workspace.nms_scratch) + workspace.nms_scratch_space);
  return workspace;
}

}  // namespace

void BoxNMSForwardGPU_notemp(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using mshadow::Shape1;
  using mshadow::Shape2;
  using mshadow::Shape3;
  CHECK_NE(req[0], kAddTo) << "BoxNMS does not support kAddTo";
  CHECK_NE(req[0], kWriteInplace) << "BoxNMS does not support in place computation";
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U) << "BoxNMS output: [output, temp]";
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  mxnet::TShape in_shape = inputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<gpu, 3, DType> data = inputs[box_nms_enum::kData]
     .get_with_shape<gpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<gpu, 3, DType> out = outputs[box_nms_enum::kOut]
     .get_with_shape<gpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    // Special case for topk == 0
    if (param.topk == 0) {
      if (req[0] != kNullOp &&
          req[0] != kWriteInplace) {
        out = mshadow::expr::F<mshadow_op::identity>(data);
      }
      return;
    }

    index_t topk = param.topk > 0 ? std::min(param.topk, num_elem) : num_elem;
    const auto& workspace = GetWorkspace<DType>(num_batch, num_elem,
                                                width_elem, topk, ctx);

    FilterAndPrepareAuxData(data, &out, workspace, param, s);
    Tensor<gpu, 1, DType> scores(workspace.scores, Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, index_t> batches(workspace.batches, Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, index_t> indices(workspace.indices, Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, char> scratch(reinterpret_cast<char*>(workspace.scratch),
                                        Shape1(workspace.scratch_space), s);
    Tensor<gpu, 2, uint32_t> nms_scratch(workspace.nms_scratch,
                                        Shape2(NMS<DType>::THRESHOLD / (sizeof(uint32_t) * 8), topk * num_batch), s);
    indices = mshadow::expr::range<index_t>(0, num_batch * num_elem);
    mxnet::op::SortByKey(scores, indices, false, &scratch);
    batches = indices / mshadow::expr::ScalarExp<index_t>(num_elem);
    mxnet::op::SortByKey(batches, indices, true, &scratch);
    Tensor<gpu, 3, DType> buffer(reinterpret_cast<DType*>(workspace.scratch),
                                 Shape3(num_batch, num_elem, width_elem), s);
    CompactData(indices, out, &buffer, topk, s);
    NMS<DType> nms;
    nms(&buffer, &nms_scratch,  topk, param, s);
    mshadow::Copy(out, buffer, s);
  });
}

void BoxNMSForwardGPU(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U) << "BoxNMS output: [output, temp]";
  std::cout << "Reqs" << std::endl;
  for (const auto& r : req) {
    std::cout << r << std::endl;
  }
  std::cout << "END: Reqs" << std::endl;
  if (req[1] == kNullOp) {
    BoxNMSForwardGPU_notemp(attrs, ctx, inputs, req, outputs);
    return;
  }
  BoxNMSForward<gpu>(attrs, ctx, inputs, req, outputs);
}


NNVM_REGISTER_OP(_contrib_box_nms2)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSForwardGPU);

NNVM_REGISTER_OP(_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSBackward<gpu>);

NNVM_REGISTER_OP(_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapBackward<gpu>);

NNVM_REGISTER_OP(_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingBackward<gpu>);

NNVM_REGISTER_OP(_contrib_box_encode)
.set_attr<FCompute>("FCompute<gpu>", BoxEncodeForward<gpu>);

NNVM_REGISTER_OP(_contrib_box_decode)
.set_attr<FCompute>("FCompute<gpu>", BoxDecodeForward<gpu>);

}  // namespace op
}  // namespace mxnet
