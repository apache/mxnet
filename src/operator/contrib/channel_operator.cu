/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file channel_operator.cu
 * \brief 
 * \author Haozhi Qi, Yi Li, Guodong Zhang, Jifeng Dai
*/
#include "./channel_operator-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define ChannelOperator_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
  namespace cuda {
    template <typename DType>
    __global__ void GroupMaxForwardKernel(
      const int count,
      const DType* bottom_data,
      const int channels,
      const int group,
      const int channels_in_group,
      const int spatial_dim,
      DType* top_data,
      DType* max_idx_data) {
      CUDA_KERNEL_LOOP(index, count) {

        int s = index % spatial_dim;
        int g = (index / spatial_dim) % group;
        int n = index / spatial_dim / group;

        DType max_val = -FLT_MAX;
        int max_idx = -1;
        for (int i = 0; i < channels_in_group; ++i) {
          int c = g*channels_in_group + i;
          int bottom_index = (n*channels + c)*spatial_dim + s;
          if (bottom_data[bottom_index]>max_val) {
            max_val = bottom_data[bottom_index];
            max_idx = c;
          }
        }
        top_data[index] = max_val;
        max_idx_data[index] = max_idx;
      }
    }

    template <typename DType>
    __global__ void GroupPickForwardKernel(
      const int count,
      const DType* bottom_data,
      const int channels,
      const int group,
      const int channels_in_group,
      const int spatial_dim,
      DType* top_data,
      const DType* pick_idx_data) {
      CUDA_KERNEL_LOOP(index, count) {

        int s = index % spatial_dim;
        int c = (index / spatial_dim) % channels_in_group;
        int n = index / spatial_dim / channels_in_group;
        int g = pick_idx_data[n];
        int bottom_index = (n*channels + g*channels_in_group + c)*spatial_dim + s;

        top_data[index] = (g < group && g >= 0) ? bottom_data[bottom_index] : DType(0);
      }
    }

    template<typename DType>
    inline void GroupMaxForward(const Tensor<gpu, 4, DType> &out,
      const Tensor<gpu, 4, DType> &data,
      const Tensor<gpu, 4, DType> &max_idx,
      const int group) {
      // LOG(INFO) << "GroupMaxForward";
      const DType *bottom_data = data.dptr_;
      DType *top_data = out.dptr_;
      DType *max_idx_data = max_idx.dptr_;
      const int count = out.shape_.Size();
      const int channels = data.size(1);
      const int height = data.size(2);
      const int width = data.size(3);
      const int spatial_dim = height * width;
      const int channels_in_group = channels / group;
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      GroupMaxForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, bottom_data, channels, group,
          channels_in_group, spatial_dim, top_data, max_idx_data);
      ChannelOperator_CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename DType>
    inline void GroupPickForward(const Tensor<gpu, 4, DType> &out,
      const Tensor<gpu, 4, DType> &data,
      const Tensor<gpu, 4, DType> &pick_idx,
      const int group) {
      // LOG(INFO) << "GroupPickForward";
      const DType *bottom_data = data.dptr_;
      DType *top_data = out.dptr_;
      const DType *pick_idx_data = pick_idx.dptr_;
      const int count = out.shape_.Size();
      const int channels = data.size(1);
      const int height = data.size(2);
      const int width = data.size(3);
      const int spatial_dim = height * width;
      const int channels_in_group = channels / group;
      cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
      GroupPickForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, bottom_data, channels, group,
          channels_in_group, spatial_dim, top_data, pick_idx_data);
      ChannelOperator_CUDA_CHECK(cudaPeekAtLastError());
    }


    template <typename DType>
    __global__ void GroupMaxBackwardAccKernel(
      const int count,
      const DType* top_diff,
      const DType* max_idx_data,
      const int channels,
      const int group,
      const int spatial_dim,
      DType* bottom_diff) {
      CUDA_KERNEL_LOOP(index, count) {
        int s = index % spatial_dim;
        int n = index / spatial_dim / group;

        int c = max_idx_data[index];
        int bottom_index = (n*channels + c)*spatial_dim + s;
        bottom_diff[bottom_index] = top_diff[index];
      }
    }

    template <typename DType>
    __global__ void GroupPickBackwardAccKernel(
      const int count,
      const DType* top_diff,
      const DType* pick_idx_data,
      const int channels,
      const int group,
      const int channels_in_group,
      const int spatial_dim,
      DType* bottom_diff) {
      CUDA_KERNEL_LOOP(index, count) {
        int s = index % spatial_dim;
        int c = (index / spatial_dim) % channels_in_group;
        int n = index / spatial_dim / channels_in_group;
        int g = pick_idx_data[n];

        int bottom_index = (n*channels + g*channels_in_group + c)*spatial_dim + s;
        bottom_diff[bottom_index] = (g < group && g >= 0) ? top_diff[index] : DType(0);
      }
    }


    template<typename DType>
    inline void GroupMaxBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
      const Tensor<gpu, 4, DType> &out_grad,
      const Tensor<gpu, 4, DType> &max_idx,
      const int group) {
      // LOG(INFO) << "GroupMaxBackward";
      const DType *top_diff = out_grad.dptr_;
      DType *bottom_diff = in_grad.dptr_;
      const DType *max_idx_data = max_idx.dptr_;
      const int count = out_grad.shape_.Size();
      const int channels = in_grad.size(1);
      const int height = in_grad.size(2);
      const int width = in_grad.size(3);
      const int spatial_dim = height * width;
      cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
      GroupMaxBackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, top_diff, max_idx_data, channels, group, spatial_dim, bottom_diff);
      ChannelOperator_CUDA_CHECK(cudaPeekAtLastError());
    }

    template<typename DType>
    inline void GroupPickBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
      const Tensor<gpu, 4, DType> &out_grad,
      const Tensor<gpu, 4, DType> &pick_idx,
      const int group) {
      // LOG(INFO) << "GroupPickBackward";
      const DType *top_diff = out_grad.dptr_;
      DType *bottom_diff = in_grad.dptr_;
      const DType *pick_idx_data = pick_idx.dptr_;
      const int count = out_grad.shape_.Size();
      const int channels = in_grad.size(1);
      const int height = in_grad.size(2);
      const int width = in_grad.size(3);
      const int spatial_dim = height * width;
      const int channels_in_group = channels / group;
      cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
      GroupPickBackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, top_diff, pick_idx_data, channels, group,
          channels_in_group, spatial_dim, bottom_diff);
      ChannelOperator_CUDA_CHECK(cudaPeekAtLastError());
    }
    // GetMaxIdx
    template <typename DType>
    __global__ void GetMaxIdxKernel(
      const int count,
      const DType* pick_score_data,
      DType* argmax_data,
      const int group) {
      CUDA_KERNEL_LOOP(index, count) {
        const DType* offset_pick_score_data = pick_score_data + index*group;
        int max_idx = -1;
        DType max_val = -FLT_MAX;
        for (int i = 1; i < group; ++i) {
          max_idx = offset_pick_score_data[i] > max_val ? i : max_idx;
          max_val = offset_pick_score_data[i] > max_val ? offset_pick_score_data[i] : max_val;
        }
        argmax_data[index] = static_cast<DType>(max_idx);
      }
    }

    template<typename DType>
    inline void GetMaxIdx(const Tensor<gpu, 4, DType> &pick_score,
      const Tensor<gpu, 4, DType> &argmax,
      const int group) {
      // LOG(INFO) << "GroupPickBackward";
      const DType *pick_score_data = pick_score.dptr_;
      DType *argmax_data = argmax.dptr_;
      const int count = argmax.shape_.Size();

      cudaStream_t stream = Stream<gpu>::GetStream(argmax.stream_);
      GetMaxIdxKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
        kBaseThreadNum, 0, stream >> >(
          count, pick_score_data, argmax_data, group);
      ChannelOperator_CUDA_CHECK(cudaPeekAtLastError());
    }
  }  // namespace cuda

  template<typename DType>
  inline void GroupMaxForward(const Tensor<gpu, 4, DType> &out,
    const Tensor<gpu, 4, DType> &data,
    const Tensor<gpu, 4, DType> &max_idx,
    const int group) {
    cuda::GroupMaxForward(out, data, max_idx, group);
  }
  template<typename DType>
  inline void GroupPickForward(const Tensor<gpu, 4, DType> &out,
    const Tensor<gpu, 4, DType> &data,
    const Tensor<gpu, 4, DType> &pick_idx,
    const int group) {
    cuda::GroupPickForward(out, data, pick_idx, group);
  }

  template<typename DType>
  inline void GroupMaxBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
    const Tensor<gpu, 4, DType> &out_grad,
    const Tensor<gpu, 4, DType> &max_idx,
    const int group) {
    cuda::GroupMaxBackwardAcc(in_grad, out_grad, max_idx, group);
  }

  template<typename DType>
  inline void GroupPickBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
    const Tensor<gpu, 4, DType> &out_grad,
    const Tensor<gpu, 4, DType> &pick_idx,
    const int group) {
    cuda::GroupPickBackwardAcc(in_grad, out_grad, pick_idx, group);
  }

  template<typename DType>
  inline void GetMaxIdx(const Tensor<gpu, 4, DType> &pick_score,
    const Tensor<gpu, 4, DType> &argmax,
    const int group) {
    cuda::GetMaxIdx(pick_score, argmax, group);
  }

}  // namespace mshadow


namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(ChannelOperatorParam param, int dtype) {
      Operator* op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new ChannelOperatorOp<gpu, DType>(param);
      });
      return op;
    }

  }  // namespace op
}  // namespace mxnet
