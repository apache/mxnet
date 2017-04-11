/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_target.cu
 * \brief MultiBoxTarget op
 * \author Joshua Zhang
*/
#include "./multibox_target-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define MULTIBOX_TARGET_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void InitGroundTruthFlags(DType *gt_flags, const DType *labels,
                                     const int num_batches,
                                     const int num_labels,
                                     const int label_width) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_labels) return;
  int b = index / num_labels;
  int l = index % num_labels;
  if (*(labels + b * num_labels * label_width + l * label_width) == -1.f) {
    *(gt_flags + b * num_labels + l) = 0;
  } else {
    *(gt_flags + b * num_labels + l) = 1;
  }
}

template<typename DType>
__global__ void FindBestMatches(DType *best_matches, DType *gt_flags,
                                DType *anchor_flags, const DType *overlaps,
                                const int num_anchors, const int num_labels) {
  int nbatch = blockIdx.x;
  gt_flags += nbatch * num_labels;
  overlaps += nbatch * num_anchors * num_labels;
  best_matches += nbatch * num_anchors;
  anchor_flags += nbatch * num_anchors;
  const int num_threads = kMaxThreadsPerBlock;
  __shared__ int max_indices_y[kMaxThreadsPerBlock];
  __shared__ int max_indices_x[kMaxThreadsPerBlock];
  __shared__ float max_values[kMaxThreadsPerBlock];

  while (1) {
    // check if all done.
    bool finished = true;
    for (int i = 0; i < num_labels; ++i) {
      if (gt_flags[i] > .5) {
        finished = false;
        break;
      }
    }
    if (finished) break;  // all done.

    // finding max indices in different threads
    int max_x = -1;
    int max_y = -1;
    DType max_value = 1e-6;  // start with very small overlap
    for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
      if (anchor_flags[i] > .5) continue;
      for (int j = 0; j < num_labels; ++j) {
        if (gt_flags[j] > .5) {
          DType temp = overlaps[i * num_labels + j];
          if (temp > max_value) {
            max_x = j;
            max_y = i;
            max_value = temp;
          }
        }
      }
    }
    max_indices_x[threadIdx.x] = max_x;
    max_indices_y[threadIdx.x] = max_y;
    max_values[threadIdx.x] = max_value;
    __syncthreads();

    if (threadIdx.x == 0) {
      // merge results and assign best match
      int max_x = -1;
      int max_y = -1;
      DType max_value = -1;
      for (int k = 0; k < num_threads; ++k) {
        if (max_indices_y[k] < 0 || max_indices_x[k] < 0) continue;
        float temp = max_values[k];
        if (temp > max_value) {
          max_x = max_indices_x[k];
          max_y = max_indices_y[k];
          max_value = temp;
        }
      }
      if (max_x >= 0 && max_y >= 0) {
        best_matches[max_y] = max_x;
        // mark flags as visited
        gt_flags[max_x] = 0.f;
        anchor_flags[max_y] = 1.f;
      } else {
        // no more good matches
        for (int i = 0; i < num_labels; ++i) {
          gt_flags[i] = 0.f;
        }
      }
    }
    __syncthreads();
  }
}

template<typename DType>
__global__ void FindGoodMatches(DType *best_matches, DType *anchor_flags,
                                const DType *overlaps, const int num_anchors,
                                const int num_labels,
                                const float overlap_threshold) {
  int nbatch = blockIdx.x;
  overlaps += nbatch * num_anchors * num_labels;
  best_matches += nbatch * num_anchors;
  anchor_flags += nbatch * num_anchors;
  const int num_threads = kMaxThreadsPerBlock;

  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    if (anchor_flags[i] < 0) {
      int idx = -1;
      float max_value = -1.f;
      for (int j = 0; j < num_labels; ++j) {
        DType temp = overlaps[i * num_labels + j];
        if (temp > max_value) {
          max_value = temp;
          idx = j;
        }
      }
      if (max_value > overlap_threshold && (idx >= 0)) {
        best_matches[i] = idx;
        anchor_flags[i] = 0.9f;
      }
    }
  }
}

template<typename DType>
__global__ void UseAllNegatives(DType *anchor_flags, const int num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num) return;
  if (anchor_flags[idx] < 0.5) {
    anchor_flags[idx] = 0;  // regard all non-positive as negatives
  }
}

template<typename DType>
__global__ void NegativeMining(const DType *overlaps, const DType *cls_preds,
                               DType *anchor_flags, DType *buffer,
                               const float negative_mining_ratio,
                               const float negative_mining_thresh,
                               const int minimum_negative_samples,
                               const int num_anchors,
                               const int num_labels, const int num_classes) {
  int nbatch = blockIdx.x;
  overlaps += nbatch * num_anchors * num_labels;
  cls_preds += nbatch * num_classes * num_anchors;
  anchor_flags += nbatch * num_anchors;
  buffer += nbatch * num_anchors * 3;
  const int num_threads = kMaxThreadsPerBlock;
  int num_positive;
  __shared__ int num_negative;

  if (threadIdx.x == 0) {
    num_positive = 0;
    for (int i = 0; i < num_anchors; ++i) {
      if (anchor_flags[i] > .5) {
        ++num_positive;
      }
    }
    num_negative = num_positive * negative_mining_ratio;
    if (num_negative < minimum_negative_samples) {
      num_negative = minimum_negative_samples;
    }
    if (num_negative > (num_anchors - num_positive)) {
      num_negative = num_anchors - num_positive;
    }
  }
  __syncthreads();

  if (num_negative < 1) return;

  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    buffer[i] = -1.f;
    if (anchor_flags[i] < 0) {
      // compute max class prediction score
      DType max_val = cls_preds[i];
      for (int j = 1; j < num_classes; ++j) {
        DType temp = cls_preds[i + num_anchors * j];
        if (temp > max_val) max_val = temp;
      }
      DType sum = 0.f;
      for (int j = 0; j < num_classes; ++j) {
        DType temp = cls_preds[i + num_anchors * j];
        sum += exp(temp - max_val);
      }
      DType prob = exp(cls_preds[i] - max_val) / sum;
      DType max_iou = -1.f;
      for (int j = 0; j < num_labels; ++j) {
        DType temp = overlaps[i * num_labels + j];
        if (temp > max_iou) max_iou = temp;
      }
      if (max_iou < negative_mining_thresh) {
        // only do it for anchors with iou < thresh
        buffer[i] = -prob;  // -log(x) actually, but value does not matter
      }
    }
  }
  __syncthreads();

  // descend merge sorting for negative mining
  DType *index_src = buffer + num_anchors;
  DType *index_dst = buffer + num_anchors * 2;
  DType *src = index_src;
  DType *dst = index_dst;
  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    index_src[i] = i;
  }
  __syncthreads();

  for (int width = 2; width < (num_anchors << 1); width <<= 1) {
    int slices = (num_anchors - 1) / (num_threads * width) + 1;
    int start = width * threadIdx.x * slices;
    for (int slice = 0; slice < slices; ++slice) {
      if (start >= num_anchors) break;
      int middle = start + (width >> 1);
      if (num_anchors < middle) middle = num_anchors;
      int end = start + width;
      if (num_anchors < end) end = num_anchors;
      int i = start;
      int j = middle;
      for (int k = start; k < end; ++k) {
        int idx_i = static_cast<int>(src[i]);
        int idx_j = static_cast<int>(src[j]);
        if (i < middle && (j >= end || buffer[idx_i] > buffer[idx_j])) {
          dst[k] = src[i];
          ++i;
        } else {
          dst[k] = src[j];
          ++j;
        }
      }
      start += width;
    }
    __syncthreads();
    // swap src/dst
    src = src == index_src? index_dst : index_src;
    dst = dst == index_src? index_dst : index_src;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_negative; i += num_threads) {
    int idx = static_cast<int>(src[i]);
    if (anchor_flags[idx] < 0) {
      anchor_flags[idx] = 0;
    }
  }
}

template<typename DType>
__global__ void AssignTrainigTargets(DType *loc_target, DType *loc_mask,
                                     DType *cls_target, DType *anchor_flags,
                                     DType *best_matches, DType *labels,
                                     DType *anchors, const int num_anchors,
                                     const int num_labels, const int label_width,
                                     const float vx, const float vy,
                                     const float vw, const float vh) {
  const int nbatch = blockIdx.x;
  loc_target += nbatch * num_anchors * 4;
  loc_mask += nbatch * num_anchors * 4;
  cls_target += nbatch * num_anchors;
  anchor_flags += nbatch * num_anchors;
  best_matches += nbatch * num_anchors;
  labels += nbatch * num_labels * label_width;
  const int num_threads = kMaxThreadsPerBlock;

  for (int i = threadIdx.x; i < num_anchors; i += num_threads) {
    if (anchor_flags[i] > 0.5) {
      // positive sample
      int offset_l = static_cast<int>(best_matches[i]) * label_width;
      cls_target[i] = labels[offset_l] + 1;  // 0 reserved for background
      int offset = i * 4;
      loc_mask[offset] = 1;
      loc_mask[offset + 1] = 1;
      loc_mask[offset + 2] = 1;
      loc_mask[offset + 3] = 1;
      // regression targets
      float al = anchors[offset];
      float at = anchors[offset + 1];
      float ar = anchors[offset + 2];
      float ab = anchors[offset + 3];
      float aw = ar - al;
      float ah = ab - at;
      float ax = (al + ar) * 0.5;
      float ay = (at + ab) * 0.5;
      float gl = labels[offset_l + 1];
      float gt = labels[offset_l + 2];
      float gr = labels[offset_l + 3];
      float gb = labels[offset_l + 4];
      float gw = gr - gl;
      float gh = gb - gt;
      float gx = (gl + gr) * 0.5;
      float gy = (gt + gb) * 0.5;
      loc_target[offset] = DType((gx - ax) / aw / vx);  // xmin
      loc_target[offset + 1] = DType((gy - ay) / ah / vy);  // ymin
      loc_target[offset + 2] = DType(log(gw / aw) / vw);  // xmax
      loc_target[offset + 3] = DType(log(gh / ah) / vh);  // ymax
    } else if (anchor_flags[i] < 0.5 && anchor_flags[i] > -0.5) {
      // background
      cls_target[i] = 0;
    }
  }
}
}  // namespace cuda

template<typename DType>
inline void MultiBoxTargetForward(const Tensor<gpu, 2, DType> &loc_target,
                           const Tensor<gpu, 2, DType> &loc_mask,
                           const Tensor<gpu, 2, DType> &cls_target,
                           const Tensor<gpu, 2, DType> &anchors,
                           const Tensor<gpu, 3, DType> &labels,
                           const Tensor<gpu, 3, DType> &cls_preds,
                           const Tensor<gpu, 4, DType> &temp_space,
                           const float overlap_threshold,
                           const float background_label,
                           const float negative_mining_ratio,
                           const float negative_mining_thresh,
                           const int minimum_negative_samples,
                           const nnvm::Tuple<float> &variances) {
  const int num_batches = labels.size(0);
  const int num_labels = labels.size(1);
  const int label_width = labels.size(2);
  const int num_anchors = anchors.size(0);
  const int num_classes = cls_preds.size(1);
  CHECK_GE(num_batches, 1);
  CHECK_GT(num_labels, 2);
  CHECK_GE(num_anchors, 1);
  CHECK_EQ(variances.ndim(), 4);

  // init ground-truth flags, by checking valid labels
  temp_space[1] = 0.f;
  DType *gt_flags = temp_space[1].dptr_;
  const int num_threads = cuda::kMaxThreadsPerBlock;
  dim3 init_thread_dim(num_threads);
  dim3 init_block_dim((num_batches * num_labels - 1) / num_threads + 1);
  cuda::CheckLaunchParam(init_block_dim, init_thread_dim, "MultiBoxTarget Init");
  cuda::InitGroundTruthFlags<DType><<<init_block_dim, init_thread_dim>>>(
    gt_flags, labels.dptr_, num_batches, num_labels, label_width);
  MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());

  // compute best matches
  temp_space[2] = -1.f;
  temp_space[3] = -1.f;
  DType *anchor_flags = temp_space[2].dptr_;
  DType *best_matches = temp_space[3].dptr_;
  const DType *overlaps = temp_space[0].dptr_;
  cuda::CheckLaunchParam(num_batches, num_threads, "MultiBoxTarget Matching");
  cuda::FindBestMatches<DType><<<num_batches, num_threads>>>(best_matches,
    gt_flags, anchor_flags, overlaps, num_anchors, num_labels);
  MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());

  // find good matches with overlap > threshold
  if (overlap_threshold > 0) {
    cuda::FindGoodMatches<DType><<<num_batches, num_threads>>>(best_matches,
      anchor_flags, overlaps, num_anchors, num_labels,
      overlap_threshold);
    MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  }

  // do negative mining or not
  if (negative_mining_ratio > 0) {
    CHECK_GT(negative_mining_thresh, 0);
    temp_space[4] = 0;
    DType *buffer = temp_space[4].dptr_;
    cuda::NegativeMining<DType><<<num_batches, num_threads>>>(overlaps,
      cls_preds.dptr_, anchor_flags, buffer, negative_mining_ratio,
      negative_mining_thresh, minimum_negative_samples,
      num_anchors, num_labels, num_classes);
    MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  } else {
    int num_blocks = (num_batches * num_anchors - 1) / num_threads + 1;
    cuda::CheckLaunchParam(num_blocks, num_threads, "MultiBoxTarget Negative");
    cuda::UseAllNegatives<DType><<<num_blocks, num_threads>>>(anchor_flags,
      num_batches * num_anchors);
    MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  }

  cuda::AssignTrainigTargets<DType><<<num_batches, num_threads>>>(
    loc_target.dptr_, loc_mask.dptr_, cls_target.dptr_, anchor_flags,
    best_matches, labels.dptr_, anchors.dptr_, num_anchors, num_labels,
    label_width, variances[0], variances[1], variances[2], variances[3]);
  MULTIBOX_TARGET_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MultiBoxTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxTargetOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
