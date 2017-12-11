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
 * \file proposal.cu
 * \brief Proposal Operator
 * \author Shaoqing Ren, Jian Guo
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./proposal-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {

// scores are (b, anchor, h, w)
// workspace_proposals are (h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename Dtype>
__global__ void ProposalGridKernel(const int count,
                                   const int num_anchors,
                                   const int height,
                                   const int width,
                                   const int feature_stride,
                                   const Dtype* scores,
                                   Dtype* workspace_proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    workspace_proposals[index * 5 + 0] = workspace_proposals[a * 5 + 0] + w * feature_stride;
    workspace_proposals[index * 5 + 1] = workspace_proposals[a * 5 + 1] + h * feature_stride;
    workspace_proposals[index * 5 + 2] = workspace_proposals[a * 5 + 2] + w * feature_stride;
    workspace_proposals[index * 5 + 3] = workspace_proposals[a * 5 + 3] + h * feature_stride;
    workspace_proposals[index * 5 + 4] = scores[(a * height + h) * width + w];
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void BBoxPredKernel(const int count,
                               const int num_anchors,
                               const int feat_height,
                               const int feat_width,
                               const int real_height,
                               const int real_width,
                               const float im_height,
                               const float im_width,
                               const Dtype* boxes,
                               const Dtype* deltas,
                               Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float width = boxes[index * 5 + 2] - boxes[index * 5 + 0] + 1.0f;
    float height = boxes[index * 5 + 3] - boxes[index * 5 + 1] + 1.0f;
    float ctr_x = boxes[index * 5 + 0] + 0.5f * (width - 1.0f);
    float ctr_y = boxes[index * 5 + 1] + 0.5f * (height - 1.0f);

    float dx = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dw = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dh = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);

    pred_x1 = max(min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = max(min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = max(min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = max(min(pred_y2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void IoUPredKernel(const int count,
                              const int num_anchors,
                              const int feat_height,
                              const int feat_width,
                              const int real_height,
                              const int real_width,
                              const float im_height,
                              const float im_width,
                              const Dtype* boxes,
                              const Dtype* deltas,
                              Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 5 + 0];
    float y1 = boxes[index * 5 + 1];
    float x2 = boxes[index * 5 + 2];
    float y2 = boxes[index * 5 + 3];

    float dx1 = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_x1 = max(min(x1 + dx1, im_width - 1.0f), 0.0f);
    float pred_y1 = max(min(y1 + dy1, im_height - 1.0f), 0.0f);
    float pred_x2 = max(min(x2 + dx2, im_width - 1.0f), 0.0f);
    float pred_y2 = max(min(y2 + dy2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// filter box with stride less than rpn_min_size
// filter: set score to zero
// dets (n, 5)
template<typename Dtype>
__global__ void FilterBoxKernel(const int count,
                                const float min_size,
                                Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    float iw = dets[index * 5 + 2] - dets[index * 5 + 0] + 1.0f;
    float ih = dets[index * 5 + 3] - dets[index * 5 + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[index * 5 + 0] -= min_size / 2;
      dets[index * 5 + 1] -= min_size / 2;
      dets[index * 5 + 2] += min_size / 2;
      dets[index * 5 + 3] += min_size / 2;
      dets[index * 5 + 4] = -1.0f;
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename Dtype>
__global__ void CopyScoreKernel(const int count,
                                const Dtype* dets,
                                Dtype* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ReorderProposalsKernel(const int count,
                                       const Dtype* prev_dets,
                                       const int* order,
                                       Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; j ++) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(const mshadow::Tensor<gpu, 2>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  float* boxes_dev = boxes.dptr_;
  uint64_t* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  FRCNN_CUDA_CHECK(cudaMalloc(&mask_dev,
                              boxes_num * col_blocks * sizeof(uint64_t)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  std::vector<uint64_t> mask_host(boxes_num * col_blocks);
  FRCNN_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  FRCNN_CUDA_CHECK(cudaFree(mask_dev));
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename Dtype>
__global__ void PrepareOutput(const int count,
                              const Dtype* dets,
                              const int* keep,
                              const int out_size,
                              Dtype* out,
                              Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 5] = 0;
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    } else {
      int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    }
  }
}

}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu>
class ProposalGPUOp : public Operator{
 public:
  explicit ProposalGPUOp(ProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);
    CHECK_EQ(in_data[proposal::kClsProb].shape_[0], 1)
      << "Sorry, multiple images each device is not implemented.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Shape<4> fg_scores_shape = Shape4(in_data[proposal::kClsProb].shape_[0],
                                      in_data[proposal::kClsProb].shape_[1] / 2,
                                      in_data[proposal::kClsProb].shape_[2],
                                      in_data[proposal::kClsProb].shape_[3]);
    real_t* foreground_score_ptr = in_data[proposal::kClsProb].dptr<real_t>()
                                    + fg_scores_shape.Size();
    Tensor<xpu, 4> scores = Tensor<xpu, 4>(foreground_score_ptr, fg_scores_shape);
    Tensor<xpu, 4> bbox_deltas = in_data[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> im_info = in_data[proposal::kImInfo].get<xpu, 2, real_t>(s);

    Tensor<xpu, 2> out = out_data[proposal::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out_score = out_data[proposal::kScore].get<xpu, 2, real_t>(s);

    int num_anchors = in_data[proposal::kClsProb].shape_[1] / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;  // count of total anchors
    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    // Generate first anchors based on base anchor
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.ndim() * param_.scales.ndim());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios,
                           param_.scales,
                           &anchors);

    // Copy generated anchors to GPU
    float* workspace_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_proposals_ptr, sizeof(float) * count * 5));
    Tensor<xpu, 2> workspace_proposals(workspace_proposals_ptr, Shape2(count, 5));
    FRCNN_CUDA_CHECK(cudaMemcpy(workspace_proposals.dptr_,
                                &anchors[0], sizeof(float) * anchors.size(),
      cudaMemcpyHostToDevice));

    // Copy proposals to a mesh grid
    dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "ProposalGrid");
    ProposalGridKernel<<<dimGrid, dimBlock>>>(
      count, num_anchors, height, width, param_.feature_stride,
      scores.dptr_, workspace_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // im_info is small, we want to copy them to cpu
    std::vector<float> cpu_im_info(3);
    FRCNN_CUDA_CHECK(cudaMemcpy(&cpu_im_info[0], im_info.dptr_,
                                sizeof(float) * cpu_im_info.size(),
                                cudaMemcpyDeviceToHost));

    // prevent padded predictions
    int real_height = static_cast<int>(cpu_im_info[0] / param_.feature_stride);
    int real_width = static_cast<int>(cpu_im_info[1] / param_.feature_stride);
    CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
    CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

    // Transform anchors and bbox_deltas into bboxes
    CheckLaunchParam(dimGrid, dimBlock, "BBoxPred");
    if (param_.iou_loss) {
      IoUPredKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, real_height, real_width,
        cpu_im_info[0], cpu_im_info[1],
        workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
    } else {
      BBoxPredKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, real_height, real_width,
        cpu_im_info[0], cpu_im_info[1],
        workspace_proposals.dptr_, bbox_deltas.dptr_, workspace_proposals.dptr_);
    }
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // filter boxes with less than rpn_min_size
    CheckLaunchParam(dimGrid, dimBlock, "FilterBox");
    FilterBoxKernel<<<dimGrid, dimBlock>>>(
      count, param_.rpn_min_size * cpu_im_info[2], workspace_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Copy score to a continuous memory
    float* score_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&score_ptr, sizeof(float) * count));
    Tensor<xpu, 1> score(score_ptr, Shape1(count));
    int* order_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&order_ptr, sizeof(int) * count));
    Tensor<xpu, 1, int> order(order_ptr, Shape1(count));

    CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
    CopyScoreKernel<<<dimGrid, dimBlock>>>(
      count, workspace_proposals.dptr_, score.dptr_, order.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // argsort score, save order
    thrust::stable_sort_by_key(thrust::device,
                               score.dptr_,
                               score.dptr_ + score.size(0),
                               order.dptr_,
                               thrust::greater<real_t>());
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // Reorder proposals according to order
    float* workspace_ordered_proposals_ptr = NULL;
    FRCNN_CUDA_CHECK(cudaMalloc(&workspace_ordered_proposals_ptr,
                                sizeof(float) * rpn_pre_nms_top_n * 5));
    Tensor<xpu, 2> workspace_ordered_proposals(workspace_ordered_proposals_ptr,
                                               Shape2(rpn_pre_nms_top_n, 5));

    dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "ReorderProposals");
    ReorderProposalsKernel<<<dimGrid, dimBlock>>>(
      rpn_pre_nms_top_n, workspace_proposals.dptr_, order.dptr_, workspace_ordered_proposals.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    FRCNN_CUDA_CHECK(cudaFree(workspace_proposals_ptr));
    FRCNN_CUDA_CHECK(cudaFree(score_ptr));
    FRCNN_CUDA_CHECK(cudaFree(order_ptr));

    // perform nms
    std::vector<int> _keep(workspace_ordered_proposals.size(0));
    int out_size = 0;
    _nms(workspace_ordered_proposals,
         param_.threshold,
         &_keep[0],
         &out_size);

    // copy nms result to gpu
    int* keep;
    FRCNN_CUDA_CHECK(cudaMalloc(&keep, sizeof(int) * _keep.size()));
    FRCNN_CUDA_CHECK(cudaMemcpy(keep, &_keep[0], sizeof(int) * _keep.size(),
                                cudaMemcpyHostToDevice));

    // copy results after nms
    dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
    PrepareOutput<<<dimGrid, dimBlock>>>(
      rpn_post_nms_top_n, workspace_ordered_proposals.dptr_, keep, out_size,
      out.dptr_, out_score.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    // free temporary memory
    FRCNN_CUDA_CHECK(cudaFree(keep));
    FRCNN_CUDA_CHECK(cudaFree(workspace_ordered_proposals_ptr));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  ProposalParam param_;
};  // class ProposalGPUOp

template<>
Operator* CreateOp<gpu>(ProposalParam param) {
  return new ProposalGPUOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
