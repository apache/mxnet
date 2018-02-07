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
 * \file depthwise_convolution_tf.cuh
 * \brief some depthwise convolution CUDA kernel code. The main logic comes
 *        from tensorflow, but the filter's layerout and many argument names
 *        are different with origin version.
 * \author shuqian.qu@hobot.cc
*/
#ifndef MXNET_OPERATOR_DEPTHWISE_CONVOLUTION_TF_CUH_
#define MXNET_OPERATOR_DEPTHWISE_CONVOLUTION_TF_CUH_
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

namespace tf {
namespace depthwise_conv {

#define FULL_WARP_MASK 0xFFFFFFFF
#if CUDA_VERSION < 9000
template<typename DType>
__forceinline__ __device__ DType  __shfl_xor_sync(unsigned, DType val, int delta) {
  return __shfl_xor(val, delta);
}

template<typename DType>
__forceinline__ __device__ DType  __shfl_down_sync(unsigned, DType val, int delta) {
  return __shfl_down(val, delta);
}

// shuffle masks not used before CUDA 9.
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = 0u;
#else
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_height;
  int in_width;
  int in_channel;
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;

  // Output layer dimensions
  int out_height;
  int out_width;
  int out_channel;
};

namespace cuda {
template<typename DType, int kFilterHeight, int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
DepthwiseConv2dForwardKernel(const DType* input,
                             const DType* filter,
                             const DepthwiseArgs args,
                             int num_outputs,
                             DType* output) {
  const int in_channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_channel = args.out_channel;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  CUDA_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_channel * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp = (out_b * in_channel + in_c) * (in_height * in_width);
    const int filter_offset_temp = in_c * filter_height * filter_width;

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_h_start = out_h * stride_height - pad_height;
    const int input_w_start = out_w * stride_width - pad_width;
    const int input_h_end = input_h_start + filter_height;
    const int input_w_end = input_w_start + filter_width;

    DType sum = 0;
    if (input_h_start >= 0 && input_w_start >= 0 &&
        input_h_end < in_height && input_w_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          const int input_offset = (input_offset_temp) + (in_h * in_width) + in_w;
          const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          // TODO(vrv): the in_h check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            const int in_w = input_w_start + f_w;
            const int input_offset = input_offset_temp + (in_h * in_width) + in_w;
            const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }
    output[thread_id] = sum;
  }
}

// The DepthwiseConv2dKernelSmall perform either forward or backward input
// convolution depending on a template argument of this enum.
enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Only use this kernel if
// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backward input direction is the same as forward direction with the filter
// rotated by 180°.
template <typename DType, DepthwiseConv2dDirection kDirection,
          int kBlockSlices, bool kEvenHeight, int kFilterHeight, int kFilterWidth>
__global__ __launch_bounds__(1024, 2) void DepthwiseConv2dKernelSmall(
    const DepthwiseArgs args, const DType* input, const DType* filter, DType* output) {
  extern __shared__ __align__(sizeof(DType)) unsigned char shared_memory[];
  DType* const shared_data = reinterpret_cast<DType*>(shared_memory);

  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int in_channel = args.in_channel;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;

  // Fixed blockDim.z, tailored for maximum grid size for images of size 16x16.
  const int block_height = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockSlices;
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int even_height = kEvenHeight || (1 & ~in_height);
  const int tile_height = in_height + filter_height - even_height;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_slices = in_channel * args.batch;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;

  const int thread_width = threadIdx.x;
  const int thread_height = threadIdx.y;
  const int thread_channel = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_height * in_width + thread_width;
  const int thread_idx = thread_channel * block_pixels + thread_pix;

  // Initialize tile, in particular the padding.
  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = DType(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_channel * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_height * tile_width + thread_width;
  const int data_idx = thread_channel * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_idx = data_idx + pad_offset;

  const int filter_pix = thread_pix;
  const int filter_channel = thread_channel;
  const int filter_idx = filter_pixels * filter_channel + filter_pix;

  const int max_slice = in_slices - thread_channel;
  const int filter_write_offset = filter_pix < filter_pixels ? tile_size + filter_idx : 0;
  const int filter_read_offset = tile_size +
    (kDirection == DIRECTION_FORWARD ?
     filter_pixels * filter_channel : filter_pixels * (filter_channel + 1));
  const bool skip_second = !kEvenHeight && thread_height + (in_height & 1) == block_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const DType* const in_ptr = inout_offset + input;
      DType* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(block_pixels + in_ptr);
      }
    }

    if (filter_write_offset != 0) {
      const int filter_offset = ((slice + filter_channel) % in_channel)* filter_pixels + filter_pix;
      shared_data[filter_write_offset] = ldg(filter_offset + filter);
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    if (slice_in_range) {
      DType sum1 = 0;
      DType sum2 = 0;
      int shared_offset = data_idx;
      const DType* filter_ptr = filter_read_offset + shared_data;
      CUDA_UNROLL for (int r = 0; r < filter_height; ++r) {
        CUDA_UNROLL for (int c = 0; c < filter_width; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr--;
          }
          const DType filter_value = *filter_ptr;
          const DType* const tile_ptr = shared_offset + shared_data;
          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          ++shared_offset;
          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr++;
          }
        }
        shared_offset += in_increment;
      }
      DType* const out_ptr = inout_offset + output;
      if (kDirection == DIRECTION_FORWARD) {
        out_ptr[0] = sum1;
        if (!skip_second) {
          out_ptr[block_pixels] = sum2;
        }
      } else {
        out_ptr[0] += sum1;
        if (!skip_second) {
          out_ptr[block_pixels] += sum2;
        }
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
  }
}

template<typename DType>
__global__ void __launch_bounds__(640, 2)
DepthwiseConv2dBackwardDataKernel(const DepthwiseArgs args,
                                  const DType* out_grad,
                                  const DType* filter, DType* in_grad,
                                  int num_in_grad) {
  const int channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = args.filter_height;
  const int filter_width = args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  const int in_pixels = in_height * in_width;
  const int out_pixels = out_height * out_width;

  CUDA_KERNEL_LOOP(thread_id, num_in_grad) {
    // Compute the indexes of this thread in the input.
    const int in_w = thread_id % in_width;
    const int in_h = (thread_id / in_width) % in_height;
    const int channel_idx = (thread_id / in_width / in_height) % channel;
    const int batch_idx = thread_id / channel / in_width / in_height;
    DType sum = 0.0f;

    const int out_h_start = mxnet::common::cuda::CudaMax<int>(
        0, (in_h - filter_height + pad_height + stride_height) / stride_height);
    const int out_h_end = mxnet::common::cuda::CudaMin(
        out_height - 1, (in_h + pad_height) / stride_height);
    const int out_w_start = mxnet::common::cuda::CudaMax<int>(
            0, (in_w - filter_width + pad_width + stride_width) / stride_width);
    const int out_w_end = mxnet::common::cuda::CudaMin(
        out_width - 1, (in_w + pad_width) / stride_width);

    const int filter_offset_temp = channel_idx * filter_height * filter_width;
    const int out_grad_offset_temp = (batch_idx * channel * out_pixels) +
        (channel_idx * out_pixels);

    for (int out_h = out_h_start; out_h <= out_h_end; ++out_h) {
      const int f_h = in_h + pad_height - out_h * stride_height;
      const int filter_offset_h = filter_offset_temp + f_h * filter_width;
      const int out_grad_offset_h = out_grad_offset_temp + out_h * out_width;
      for (int out_w = out_w_start; out_w <= out_w_end; ++out_w) {
        const int f_w = in_w + pad_width - out_w * stride_width;
        const int filter_offset = filter_offset_h + f_w;
        const int out_grad_offset = out_grad_offset_h + out_w;
        sum += ldg(out_grad + out_grad_offset) * ldg(filter + filter_offset);
      }
    }
    const int in_grad_offset = (batch_idx * channel * in_pixels) +
        (channel_idx * in_pixels) + (in_h * in_width) + (in_w);
    in_grad[in_grad_offset] += sum;
  }
}

// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
// NCHW format, tailored for small images up to 32x32. Only use this kernel if
// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input tensor are loaded into shared memory before performing the
// convolution. Per iteration and filter element, each thread first performs
// a partial convolution for two elements, one each in the lower and upper half
// of a tile. The intermediate result of all pixels of a warp are then
// accumulated and written to shared memory. Finally, the values in shared
// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
// up in global memory using atomics.
// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
// kAccumPixels * 64 >= args.in_height * args.in_width * kBlockSlices.
template <typename DType, int kBlockSlices, int kAccumPixels, int kFilterHeight, int kFilterWidth>
__global__
__launch_bounds__(1024, 2) void DepthwiseConv2dBackwardFilterKernelSmall(
    const DepthwiseArgs args, const DType* output, const DType* input, DType* filter) {
  extern __shared__ __align__(sizeof(DType)) unsigned char shared_memory[];
  DType* const shared_data = reinterpret_cast<DType*>(shared_memory);

  const int in_height = args.in_height;
  const int in_width = blockDim.x;  // slower (see b/62280718): args.in_width;
  const int in_channel = args.in_channel;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;

  const int block_height = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockSlices;
  assert((block_size & 31) == 0);
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int tile_height = 2 * block_height + filter_height - 1;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockSlices;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_slices = in_channel * args.batch;
  const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;
  // The accumulator has a fixed number of pixels that can be reduced by one
  // warp. Pixels beyond ceil(in_pixels * kBlockSlices / 64) are never written.
  assert(kAccumPixels * 64 >= in_height * in_width * kBlockSlices);
  const int accum_increment = kAccumPixels * kBlockSlices;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_width = threadIdx.x;
  const int thread_height = threadIdx.y;
  const int thread_channel = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_height * in_width + thread_width;
  const int thread_idx = thread_channel * block_pixels + thread_pix;

  // Initialize tile, in particular the padding and accumulator.
  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = DType(0);
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_channel * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_height * tile_width + thread_width;
  const int data_idx = thread_channel * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_idx = data_idx + pad_offset;

  // Position in accumulator (kBlockSlices per warp, depth major).
  const int accum_pix = thread_pix / (32 / kBlockSlices);
  const int accum_idx = thread_channel * kAccumPixels + accum_pix;

  const int max_slice = in_slices - thread_channel;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_height + thread_height >= in_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int slice = b * kBlockSlices;

    const int inout_offset = slice * in_pixels + tensor_idx;
    const bool slice_in_range = slice < max_slice;

    if (slice_in_range) {
      const DType* const in_ptr = inout_offset + input;
      DType* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = ldg(in_ptr);
      if (!skip_second) {
        tile_ptr[tile_offset] = ldg(block_pixels + in_ptr);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    // Not all threads of a warp may reach the __shfl_down_sync instruction
    // so we cannot use the FULL_WARP_MASK there
    CREATE_SHFL_MASK(active_threads, slice_in_range);

    if (slice_in_range) {
      const DType* const out_ptr = inout_offset + output;
      const DType out1 = ldg(out_ptr);
      const DType out2 = skip_second ? DType(0) : ldg(block_pixels + out_ptr);
      int shared_offset = data_idx;
      DType* accum_ptr = accum_offset + shared_data;
      CUDA_UNROLL for (int r = 0; r < filter_height; ++r) {
        CUDA_UNROLL for (int c = 0; c < filter_width; ++c) {
          const DType* const tile_ptr = shared_offset + shared_data;
          DType val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
          // Warp-accumulate pixels of the same depth and write to accumulator.
          for (int delta = 16 / kBlockSlices; delta > 0; delta /= 2) {
            val += __shfl_down_sync(active_threads, val, delta);
          }
          if (!(thread_idx & 32 / kBlockSlices - 1)) {
            *accum_ptr = val;
          }
          ++shared_offset;
          accum_ptr += accum_increment;
        }
        shared_offset += in_increment;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    const DType* const accum_data = tile_size + shared_data;
    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockSlices;
      const int filter_channel = (slice + filter_idx % kBlockSlices) % in_channel;
      // convert to CHW
      const int filter_offset = filter_channel * filter_pixels +
          (filter_pix/filter_width) * filter_height + filter_pix % filter_width;

      if (filter_channel < in_channel) {
        DType val = accum_data[i];
        // Warp-accumulate pixels of the same depth from the accumulator.
        int lane_id;
        asm volatile ("mov.u32 %0, %laneid;" : "=r"(lane_id));
        int sub_warp = lane_id / kAccumPixels;
        int zeros = sub_warp * kAccumPixels;
        unsigned mask = (kAccumPixels == 32) ? FULL_WARP_MASK : (((1U << kAccumPixels) - 1) << zeros);
        for (int delta = kAccumPixels / 2; delta > 0; delta /= 2) {
          val += __shfl_xor_sync(mask, val, delta);
        }
        if (!(thread_idx & kAccumPixels - 1)) {
          atomicAdd(filter_offset + filter, val);
        }
      }
    }
  }
}


}  // namespace cuda

// Returns whether depthwise convolution forward or backward input pass can be
// performed using the faster ('Small') variant of the kernel.
bool CanLaunchDepthwiseConv2dGPUSmall(const DepthwiseArgs& args) {
  return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
      args.in_width <= 32 && args.in_height == args.out_height &&
      args.in_width == args.out_width && args.pad_height >= 0 &&
      args.pad_height < args.filter_height && args.pad_width >= 0 &&
      args.pad_width < args.filter_width &&
      args.filter_height * args.filter_width <= (args.in_height + 1) / 2 * args.in_width;
}

// Returns whether depthwise convolution backward filter pass can be performed
// using the faster ('Small') variant of the kernel.
bool CanLaunchDepthwiseConv2dBackwardFilterGPUSmall(const DepthwiseArgs args,
                                                    const int block_height) {
  return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
      args.in_width <= 32 && args.in_height == args.out_height &&
      args.in_width == args.out_width && args.pad_height >= 0 &&
      args.pad_height < args.filter_height && args.pad_width >= 0 &&
      args.pad_width < args.filter_width && block_height <= args.in_height &&
      args.filter_height * args.filter_width <= block_height * args.in_width;
}

template <typename DType, cuda::DepthwiseConv2dDirection kDirection,
          int kBlockSlices, bool kEvenHeight>
void LaunchDepthwiseConv2dGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                   const DepthwiseArgs args,
                                   const DType* input, const DType* filter, DType* output) {
  const int block_height = (args.in_height + 1) / 2;
  dim3 block_dim = dim3(args.in_width, block_height, kBlockSlices);

  const int tile_width = args.in_width + args.filter_width - 1;
  const int tile_height = block_height * 2 + args.filter_height - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = args.filter_height * args.filter_width;
  const int shared_memory_size =
      kBlockSlices * (tile_pixels + filter_pixels) * sizeof(DType);
  const int num_outputs =
      args.batch * args.out_height * args.out_width * args.out_channel;
  int block_count = std::min(num_outputs/(block_dim.x * block_dim.y * block_dim.z) + 1,
                             (unsigned)mshadow::cuda::kMaxGridNum);
  auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
  if (args.filter_height == 3 && args.filter_width == 3) {
    cuda::DepthwiseConv2dKernelSmall<DType, kDirection, kBlockSlices, kEvenHeight, 3, 3>
        <<<block_count, block_dim, shared_memory_size, s>>>(args, input, filter, output);
  } else {
    cuda::DepthwiseConv2dKernelSmall<DType, kDirection, kBlockSlices, kEvenHeight, -1, -1>
        <<<block_count, block_dim, shared_memory_size, s>>>(args, input, filter, output);
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dKernelSmall);
}

template <typename DType, cuda::DepthwiseConv2dDirection kDirection, int kBlockSlices>
void LaunchDepthwiseConv2dGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                   const DepthwiseArgs args,
                                   const DType* input, const DType* filter, DType* output) {
  if (args.in_height & 1) {
    LaunchDepthwiseConv2dGPUSmall<DType, kDirection, kBlockSlices, false>(
        stream, args, input, filter, output);
  } else {
    LaunchDepthwiseConv2dGPUSmall<DType, kDirection, kBlockSlices, true>(
        stream, args, input, filter, output);
  }
}

template <typename DType, cuda::DepthwiseConv2dDirection kDirection>
void LaunchDepthwiseConv2dGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                   const DepthwiseArgs args,
                                   const DType* input, const DType* filter, DType* output) {
  // Maximize (power of two) kBlockSlices while keeping a block within 1024
  // threads (2 pixels per thread).
  const int block_pixels = (args.in_height + 1) / 2 * args.in_width;
  if (block_pixels > 256) {
    LaunchDepthwiseConv2dGPUSmall<DType, kDirection, 2>(stream, args, input, filter, output);
  } else if (block_pixels > 128) {
    LaunchDepthwiseConv2dGPUSmall<DType, kDirection, 4>(stream, args, input, filter, output);
  } else {
    LaunchDepthwiseConv2dGPUSmall<DType, kDirection, 8>(stream, args, input, filter, output);
  }
}

template <typename DType, int kBlockSlices, int kAccumPixels>
bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                                    const DepthwiseArgs args,
                                                    const int block_height,
                                                    const DType* out_grad,
                                                    const DType* input,
                                                    DType* filter_grad) {
  const int tile_width = args.in_width + args.filter_width - 1;
  const int tile_height = block_height * 2 + args.filter_height - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = args.filter_height * args.filter_width;
  const int shared_memory_size =
      kBlockSlices * (tile_pixels + filter_pixels * kAccumPixels) * sizeof(DType);
  if (shared_memory_size > 46 * 1024) {
    return false;
  }

  dim3 block_dim = dim3(args.in_width, block_height, kBlockSlices);
  const int num_out_grad =
      args.batch * args.out_height * args.out_width * args.out_channel;
  int block_count = num_out_grad/(block_dim.x * block_dim.y * block_dim.z) + 1;
  auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
  if (args.filter_height == 3 && args.filter_width == 3) {
    cuda::DepthwiseConv2dBackwardFilterKernelSmall<DType, kBlockSlices, kAccumPixels, 3, 3>
        <<<block_count, block_dim, shared_memory_size, s>>>(
            args, out_grad, input, filter_grad);
  } else {
    cuda::DepthwiseConv2dBackwardFilterKernelSmall<DType, kBlockSlices, kAccumPixels, -1, -1>
        <<<block_count, block_dim, shared_memory_size, s>>>(
            args, out_grad, input, filter_grad);
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernelSmall);
  return true;
}

template <typename DType, int kBlockSlices>
bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                                    const DepthwiseArgs args,
                                                    const int block_height,
                                                    const DType* out_grad,
                                                    const DType* input,
                                                    DType* filter_grad) {
  // Minimize (power of two) kAccumPixels, while satisfying
  // kAccumPixels * 32 >= block_height * in_width * kBlockSlices.
  const int block_pixels = block_height * args.in_width * kBlockSlices;
  if (block_pixels > 512) {
    return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, kBlockSlices, 32>(
        stream, args, block_height, out_grad, input, filter_grad);
  } else if (block_pixels > 256) {
    return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, kBlockSlices, 16>(
        stream, args, block_height, out_grad, input, filter_grad);
  } else {
    return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, kBlockSlices, 8>(
        stream, args, block_height, out_grad, input, filter_grad);
  }
}

template <typename DType>
bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(mshadow::Stream<mxnet::gpu> *stream,
                                                    const DepthwiseArgs args,
                                                    const DType* out_grad,
                                                    const DType* input,
                                                    DType* filter_grad) {
  // Maximize (power of two) kBlockSlices while keeping a block within 1024
  // threads (2 pixels per thread).
  int block_slices = 8;
  int block_height = (args.in_height + 1) / 2;
  int round_mask = 1;
  for (; block_slices > 1; block_slices /= 2) {
    // args.in_width * block_height * kBlockSlices must be multiple of 32.
    for (; block_height * args.in_width * block_slices & 31;
         round_mask = round_mask * 2 + 1) {
      block_height = block_height + round_mask & ~round_mask;
    }
    int block_size = block_height * args.in_width * block_slices;
    if (block_size <= 1024) {
      break;
    }
  }

  if (!CanLaunchDepthwiseConv2dBackwardFilterGPUSmall(args, block_height)) {
    return false;
  }

  switch (block_slices) {
    case 8:
      return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, 8>(
          stream, args, block_height, out_grad, input, filter_grad);
    case 4:
      return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, 4>(
          stream, args, block_height, out_grad, input, filter_grad);
    case 2:
      return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<DType, 2>(
          stream, args, block_height, out_grad, input, filter_grad);
    default:
      return false;
  }
}

}  // namespace depthwise_conv
}  // namespace tf

#endif  // MXNET_OPERATOR_DEPTHWISE_CONVOLUTION_TF_CUH_
