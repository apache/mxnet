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
 * Copyright (c) 2017 by Contributors
 * \file softmax.cu
 * \brief GPU Implementation of softmax
 */
#include <string>
#include "./softmax-inl.h"
#include "../../common/cuda/utils.h"
#include "../../common/utils.h"
#include "../../common/cuda/rtc.h"
#include "../../common/cuda/rtc/vectorization-inl.h"

namespace mxnet {
namespace op {

namespace {

struct softmax_params {
  const void* inputs[3];
  void* outputs[1];
  index_t stride;
  index_t num_elements;
  double temperature;
  int rows_per_block;
  index_t total_rows;
};

const char softmax_common_functions[] = R"code(
struct softmax_params {
  const void* inputs[3];
  void* outputs[1];
  index_t stride;
  index_t num_elements;
  double temperature;
  int rows_per_block;
  index_t total_rows;
};

template <typename DType, typename DType2>
__device__ inline type_util::mixed_type<DType, DType2>
softmax_fwd(const DType a, const DType2 b) {
  return op::exp(a) / b;
}

template <typename DType, typename DType2>
__device__ inline type_util::mixed_type<DType, DType2>
log_softmax_fwd(const DType a, const DType2 b) {
  return a - op::log(b);
}

template <typename DType, typename DType2, typename DType3>
__device__ inline type_util::mixed_type<DType, DType2, DType3>
softmax_bwd(DType ograd, DType2 out, DType3 sum) {
    return out * (ograd - sum);
}

template <typename DType, typename DType2, typename DType3>
__device__ inline type_util::mixed_type<DType, DType2, DType3>
log_softmax_bwd(DType ograd, DType2 out, DType3 sum) {
    return ograd - op::exp(out) * sum;
}

)code";

const char simple_softmax_kernel_fwd[] = R"code(
__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void simple_softmax_kernel(const softmax_params param,
                                      const index_t lead_dim) {
  using LengthType = AccType<InputType1>;
  const InputType0* input = reinterpret_cast<const InputType0*>(param.inputs[0]);
  const InputType1* length = reinterpret_cast<const InputType1*>(param.inputs[1]);
  const index_t len = length == nullptr
                      ? lead_dim
                      : static_cast<index_t>(LengthType::from(length[blockIdx.x]));
  const int my_row = threadIdx.x % param.rows_per_block;
  const int my_id = threadIdx.x / param.rows_per_block;
  const int threads_per_row = blockDim.x / param.rows_per_block;
  const index_t base_x = (blockIdx.x * param.rows_per_block + my_row) % param.stride;
  const index_t base_n = (blockIdx.x * param.rows_per_block + my_row) / param.stride;
  const index_t base = base_x + param.stride * lead_dim * base_n;
  if (base >= param.num_elements * param.total_rows) return;
  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;
  using AType = type_util::mixed_type<typename IType::type,
                                      typename OType::type>;
  __shared__ AType smem[kRTCMaxThreadsPerBlock];
  AType max;
  red::maximum::SetInitValue(max);
  for (index_t i = my_id; i < len; i += threads_per_row) {
    auto val = IType::from(input[base + i * param.stride]);
    max = op::max(max, negate ? -val : val);
  }
  smem[threadIdx.x] = max;
  __syncthreads();
  for (int size = blockDim.x / 2; size >= warp_size; size /= 2) {
    if (threadIdx.x < size) {
      smem[threadIdx.x] = op::max(smem[threadIdx.x], smem[threadIdx.x + size]);
    }
    __syncthreads();
  }
  if (threadIdx.x < warp_size) {
    AType my_value = util::strided_grouped_warp_reduce(smem[threadIdx.x],
                                                       [](AType x, AType y)
                                                         { return op::max(x, y); },
                                                       param.rows_per_block);
    smem[threadIdx.x] = my_value;
  }
  __syncthreads();
  AType smax = smem[my_row];
  __syncthreads();

  AType sum;
  red::sum::SetInitValue(sum);
  for (index_t i = my_id; i < len; i += threads_per_row) {
    auto val = IType::from(input[base + i * param.stride]);
    val = negate ? -val :val;
    sum += op::exp((val - smax) / static_cast<AType>(param.temperature));
  }
  smem[threadIdx.x] = sum;
  __syncthreads();
  for (int size = blockDim.x / 2; size >= warp_size; size /= 2) {
    if (threadIdx.x < size) {
      smem[threadIdx.x] = op::add(smem[threadIdx.x], smem[threadIdx.x + size]);
    }
    __syncthreads();
  }
  if (threadIdx.x < warp_size) {
    AType my_value = util::strided_grouped_warp_reduce(smem[threadIdx.x],
                                                       [](AType x, AType y)
                                                         { return op::add(x, y); },
                                                       param.rows_per_block);
    smem[threadIdx.x] = my_value;
  }
  __syncthreads();
  sum = smem[my_row];
  __syncthreads();

  OutputType0* output = reinterpret_cast<OutputType0*>(param.outputs[0]);
  for (index_t i = my_id; i < lead_dim; i += threads_per_row) {
    auto val = IType::from(input[base + i * param.stride]);
    val = negate ? -val : val;
    val = (i < len) ? OP((val - smax)/static_cast<AType>(param.temperature), sum) : 0;
    if (req == OpReqType::kAddTo) {
      if (i < len) {
        output[base + i * param.stride] = OType::to(val +
                                                    OType::from(output[base + i * param.stride]));
      }
    } else {
      output[base + i * param.stride] = OType::to(val);
    }
  }
}
)code";

const char softmax_stride1_kernel_fwd[] = R"code(
__launch_bounds__(vector::vectorized_kernel_thread_num)
__global__ void softmax_stride1_compute_kernel(const softmax_params param,
                                               const index_t total_length,
                                               const index_t other_dim,
                                               const index_t N,
                                               const index_t num_aligned_elements) {
  using namespace vector;
  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;
  using LengthType = AccType<InputType1>;
  const InputType1* length = reinterpret_cast<const InputType1*>(param.inputs[1]);
  using AType = type_util::mixed_type<typename IType::type,
                                      typename OType::type>;
  __shared__ AType scratch[vectorized_kernel_thread_num];
  __shared__ AType persistent_storage[20 * 1024 / sizeof(AType)];
  const int threads_per_row = vectorized_kernel_thread_num / param.rows_per_block;
  const int my_local_row = threadIdx.x / threads_per_row;
  const int base_row = blockIdx.x * param.rows_per_block;
  const int my_row = base_row + my_local_row;
  const index_t len = (length == nullptr ||
                       my_row >= param.total_rows) ? param.num_elements
                                                   : LengthType::from(length[my_row]);
  const int my_id = threadIdx.x % threads_per_row;

  AType* row;
  if (only_full_blocks || blockIdx.x < gridDim.x - 1) {
    // full rows_per_block rows to compute
    VectorizedLoader<InputType0, nvec, aligned> loader(
      reinterpret_cast<const InputType0*>(param.inputs[0]) + base_row * param.num_elements,
      total_length);
    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      loader.load(i, total_length);
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        persistent_storage[i*nvec + j] = IType::from(loader.separate()[j]);
      }
    }
    row = persistent_storage + my_local_row * param.num_elements + loader.alignment();
  } else {
    // less than rows_per_block rows to compute
    const index_t real_length = min(total_length,
                                    (param.total_rows - base_row) * param.num_elements);
    VectorizedLoader<InputType0, nvec, false> loader(
      reinterpret_cast<const InputType0*>(param.inputs[0]) + base_row * param.num_elements,
      real_length);
    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      loader.load(i, real_length);
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        persistent_storage[i*nvec + j] = IType::from(loader.separate()[j]);
      }
    }
    row = persistent_storage + my_local_row * param.num_elements + loader.alignment();
  }
  __syncthreads();

  AType my_max_value;
  red::maximum::SetInitValue(my_max_value);

  for (index_t i = my_id; i < len; i += threads_per_row) {
    my_max_value = ::max(my_max_value, negate ? -row[i] : row[i]);
  }
  AType smax;
  if (!reduction_inside_warp) {
    scratch[threadIdx.x] = my_max_value;
    __syncthreads();
    for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
      if (my_id < size) {
        scratch[threadIdx.x] = ::max(scratch[threadIdx.x], scratch[threadIdx.x + size]);
      }
      __syncthreads();
    }
    if (my_id < warp_size) {
      AType my_value = util::grouped_warp_allreduce(scratch[threadIdx.x],
                                                    [](AType x, AType y) { return op::max(x, y); },
                                                    min(threads_per_row, warp_size));
      scratch[threadIdx.x] = my_value;
    }
    __syncthreads();
    smax = scratch[threadIdx.x - my_id];
    __syncthreads();
  } else {
    smax = util::grouped_warp_allreduce(my_max_value,
                                        [](AType x, AType y) { return op::max(x, y); },
                                        threads_per_row);
  }

  AType my_sum;
  red::sum::SetInitValue(my_sum);

  for (index_t i = my_id; i < len; i += threads_per_row) {
    const AType val = negate ? -row[i] : row[i];
    my_sum += op::exp((val - smax) / static_cast<AType>(param.temperature));
  }
  AType ssum;
  if (!reduction_inside_warp) {
    scratch[threadIdx.x] = my_sum;
    __syncthreads();
    for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
      if (my_id < size) {
        scratch[threadIdx.x] += scratch[threadIdx.x + size];
      }
      __syncthreads();
    }
    if (my_id < warp_size) {
      AType my_value = util::grouped_warp_allreduce(scratch[threadIdx.x],
                                                    [](AType x, AType y) { return x + y;},
                                                    min(threads_per_row, warp_size));
      scratch[threadIdx.x] = my_value;
    }
    __syncthreads();

    ssum = scratch[threadIdx.x - my_id];
    __syncthreads();
  } else {
      ssum = util::grouped_warp_allreduce(my_sum,
                                          [](AType x, AType y) { return x + y;},
                                          threads_per_row);
  }

  for (index_t i = my_id; i < param.num_elements; i += threads_per_row) {
    const AType val = negate ? -row[i] : row[i];
    row[i] = (i < len) ? OP((val - smax)/static_cast<AType>(param.temperature), ssum) :
                         0;
  }
  __syncthreads();

  if (only_full_blocks || blockIdx.x < gridDim.x - 1) {
    VectorizedStorer<OutputType0, nvec, aligned> storer(
      reinterpret_cast<OutputType0*>(param.outputs[0]) + base_row * param.num_elements,
      total_length);

    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      if (req == OpReqType::kAddTo) {
        storer.load(i, total_length);
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(op::add(persistent_storage[i*nvec + j],
                                                   OType::from(storer.separate()[j])));
        }
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(persistent_storage[i*nvec + j]);
        }
      }
      storer.store(i, total_length);
    }
  } else {
    const index_t real_length = min(total_length,
                                    (param.total_rows - base_row) * param.num_elements);
    VectorizedStorer<OutputType0, nvec, false> storer(
      reinterpret_cast<OutputType0*>(param.outputs[0]) + base_row * param.num_elements,
      real_length);

    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      if (req == OpReqType::kAddTo) {
        storer.load(i, real_length);
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(op::add(persistent_storage[i*nvec + j],
                                                   OType::from(storer.separate()[j])));
        }
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(persistent_storage[i*nvec + j]);
        }
      }
      storer.store(i, real_length);
    }
  }
}
)code";

int get_rows_per_block(const index_t row_size, const int nvec,
                       const index_t max_storage, const int num_threads_per_block,
                       const index_t total_rows, const int dev_id) {
  CHECK(common::IsPower2(num_threads_per_block))
    << "Number of threads in a block must be power of 2 to use get_rows_per_block function";
  // How many read instructions should 1 thread at least do
  const int read_instructions = 16;
  const size_t row_size_in_vec = (row_size + nvec - 1) / nvec;
  int desired_num_threads_per_row = (row_size_in_vec + read_instructions - 1) / read_instructions;
  desired_num_threads_per_row = common::RoundToPower2(desired_num_threads_per_row);
  desired_num_threads_per_row = std::min(desired_num_threads_per_row, num_threads_per_block);
  const int desired_rows_per_block = num_threads_per_block / desired_num_threads_per_row;
  int actual_rows_per_block = desired_rows_per_block;
  int num_sms = MultiprocessorCount(dev_id);
  while (actual_rows_per_block > 1 &&
         ((max_storage != -1 && max_storage < row_size * actual_rows_per_block) ||
          (total_rows + actual_rows_per_block - 1) / actual_rows_per_block < num_sms)) {
    actual_rows_per_block /= 2;
  }
  return actual_rows_per_block;
}

}  // namespace

void SoftmaxRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using common::mshadow_type_info;
  using namespace common::cuda::rtc;
  using common::div_round;
  if (req[0] == kNullOp || inputs[0].Size() == 0U) return;
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature = param.temperature.has_value() ?
                             param.temperature.value() : 1.0;
  mxnet::TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);

  void* length_ptr = nullptr;
  std::string length_typename = "int";
  if (param.use_length.value()) {
    CHECK(inputs.size() > 1)
      << "Mask needs to be provided when using softmax with use_length=True.";
    length_ptr = inputs[1].dptr_;
    length_typename = mshadow_type_info(inputs[1].type_flag_).name;
  }
  CHECK_EQ(outputs.size(), 1);
  index_t M = shape[axis];
  if (M == 0 || shape.Size() == 0) return;
  index_t stride = 1;
  if (axis == shape.ndim() - 2) {
    stride = shape[shape.ndim() - 1];
  }
  const index_t N = shape.Size() / M;
  softmax_params params = {{inputs[0].dptr_, length_ptr, nullptr},
                           {outputs[0].dptr_},
                           stride, M,
                           temperature, 1, N};
  std::string code = "#define OP " + OP + "\n"
                     "const OpReqType req = " + util::to_string(req[0]) + ";\n"
                     "const bool negate = " + std::to_string(negate) + ";\n"
                     "using InputType1 = " + length_typename + ";\n";
  Stream<gpu>* s = ctx.get_stream<gpu>();

  constexpr int nvec = 2;
  // Using 20 kB of shared memory for persistent storage in the optimized case
  const size_t acc_type_size = std::max(mshadow_type_info(inputs[0].type_flag_).acc_size,
                                        mshadow_type_info(outputs[0].type_flag_).acc_size);
  const size_t max_opt_M = 20 * 1024 / acc_type_size;
  int rows_per_block = get_rows_per_block(M, nvec, max_opt_M,
                                          vectorized_kernel_thread_num,
                                          N, ctx.run_ctx.ctx.dev_id);
  constexpr int warp_size = common::cuda::warp_size;
  if (stride == 1 &&
      static_cast<size_t>(M * rows_per_block) <= max_opt_M) {
    code += "const bool only_full_blocks = " + std::to_string(N % rows_per_block == 0) + ";\n"
            "const bool reduction_inside_warp = " +
            std::to_string(vectorized_kernel_thread_num / rows_per_block <= warp_size) + ";\n";
    params.rows_per_block = rows_per_block;
    int nblocks = (N + rows_per_block - 1) / rows_per_block;
    VectorizedKernelRTCLauncher(code + softmax_common_functions, "softmax_stride1_compute_kernel",
                                softmax_stride1_kernel_fwd, nvec,
                                M * rows_per_block, N / rows_per_block, s, params,
                                inputs, outputs,
                                ctx.run_ctx.ctx.dev_id, 0, nblocks);
    MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_stride1_compute_kernel);
  } else {
    code += "using InputType0 = " + mshadow_type_info(inputs[0].type_flag_).name + ";\n"
            "using OutputType0 = " + mshadow_type_info(outputs[0].type_flag_).name + ";\n";
    std::vector<const void*> args;
    args.emplace_back(&params);
    args.emplace_back(&M);
    int num_threads = std::min(static_cast<size_t>(128),
                               common::RoundToPower2(div_round(M, warp_size) * warp_size));
    if (stride != 1) {
      const int num_sms = MultiprocessorCount(ctx.run_ctx.ctx.dev_id);
      const index_t rows_per_sm = div_round(N, (512 / num_threads) * num_sms);
      params.rows_per_block = std::min(static_cast<size_t>(warp_size),
                                       common::RoundToPower2(rows_per_sm));
    }
    const auto& kernel_func = get_function(code + softmax_common_functions,
                                           "simple_softmax_kernel",
                                           simple_softmax_kernel_fwd,
                                           ctx.run_ctx.ctx.dev_id);
    launch(kernel_func, div_round(N, params.rows_per_block), num_threads, 0, s, &args);
    MSHADOW_CUDA_POST_KERNEL_CHECK(simple_softmax_kernel);
  }
}

const char simple_softmax_kernel_bwd[] = R"code(
__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void simple_softmax_grad_kernel(const softmax_params param,
                                           const index_t lead_dim) {
  using LengthType = AccType<InputType2>;
  const InputType0* out = reinterpret_cast<const InputType0*>(param.inputs[0]);
  const InputType1* ograd = reinterpret_cast<const InputType1*>(param.inputs[1]);
  const InputType2* length = reinterpret_cast<const InputType2*>(param.inputs[2]);
  const index_t len = length == nullptr
                      ? lead_dim
                      : static_cast<index_t>(LengthType::from(length[blockIdx.x]));
  const int my_row = threadIdx.x % param.rows_per_block;
  const int my_id = threadIdx.x / param.rows_per_block;
  const int threads_per_row = blockDim.x / param.rows_per_block;
  const index_t base_x = (blockIdx.x * param.rows_per_block + my_row) % param.stride;
  const index_t base_n = (blockIdx.x * param.rows_per_block + my_row) / param.stride;
  const index_t base = base_x + param.stride * lead_dim * base_n;
  if (base >= param.num_elements * param.total_rows) return;
  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using OType = AccType<OutputType0>;
  using AType = type_util::mixed_type<typename IType0::type,
                                      typename IType1::type,
                                      typename OType::type>;
  __shared__ AType smem[kRTCMaxThreadsPerBlock];
  AType sum;
  red::sum::SetInitValue(sum);
  for (index_t i = my_id; i < len; i += threads_per_row) {
    auto out_val = IType0::from(out[base + i * param.stride]);
    auto ograd_val = IType1::from(ograd[base + i * param.stride]);
    sum += OP1(ograd_val, out_val);
  }
  smem[threadIdx.x] = sum;
  __syncthreads();
  for (int size = blockDim.x / 2; size >= warp_size; size /= 2) {
    if (threadIdx.x < size) {
      smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + size];
    }
    __syncthreads();
  }
  if (threadIdx.x < warp_size) {
    AType my_value = util::strided_grouped_warp_reduce(smem[threadIdx.x],
                                                       [](AType x, AType y) { return x + y; },
                                                       param.rows_per_block);
    smem[threadIdx.x] = my_value;
  }
  __syncthreads();
  sum = smem[my_row];
  __syncthreads();

  OutputType0* igrad = reinterpret_cast<OutputType0*>(param.outputs[0]);
  for (index_t i = my_id; i < lead_dim; i += threads_per_row) {
    auto out_val = IType0::from(out[base + i * param.stride]);
    auto ograd_val = IType1::from(ograd[base + i * param.stride]);
    auto val = (i < len) ? OP2(ograd_val, out_val, sum) / static_cast<AType>(param.temperature) : 0;
    val = negate ? -val : val;
    if (req == OpReqType::kAddTo) {
      if (i < len) {
        igrad[base + i * param.stride] = OType::to(val +
                                                   OType::from(igrad[base + i * param.stride]));
      }
    } else {
        igrad[base + i * param.stride] = OType::to(val);
    }
  }
}
)code";

const char softmax_stride1_kernel_bwd[] = R"code(
__launch_bounds__(vector::vectorized_kernel_thread_num)
__global__ void softmax_stride1_compute_grad_kernel(const softmax_params param,
                                                    const index_t total_length,
                                                    const index_t other_dim,
                                                    const index_t N,
                                                    const index_t num_aligned_elements) {
  using namespace vector;
  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using OType = AccType<OutputType0>;
  using LengthType = AccType<InputType2>;
  const InputType2* length = reinterpret_cast<const InputType2*>(param.inputs[2]);
  using AType = type_util::mixed_type<typename IType0::type,
                                      typename IType1::type,
                                      typename OType::type>;
  __shared__ AType scratch[vectorized_kernel_thread_num];
  __shared__ AType output_persistent_storage[10 * 1024 / sizeof(AType)];
  __shared__ AType ograd_persistent_storage[10 * 1024 / sizeof(AType)];
  const int warp_size = 32;
  const int threads_per_row = vectorized_kernel_thread_num / param.rows_per_block;
  const int my_local_row = threadIdx.x / threads_per_row;
  const int base_row = blockIdx.x * param.rows_per_block;
  const int my_row = base_row + my_local_row;
  const index_t len = (length == nullptr ||
                       my_row >= param.total_rows) ? param.num_elements
                                                   : LengthType::from(length[my_row]);
  const int my_id = threadIdx.x % threads_per_row;

  AType* output_row;
  AType* ograd_row;
  if (only_full_blocks || blockIdx.x < gridDim.x - 1) {
    // full rows_per_block rows to compute
    VectorizedLoader<InputType0, nvec, aligned> output_loader(
      reinterpret_cast<const InputType0*>(param.inputs[0]) + base_row * param.num_elements,
      total_length);
    VectorizedLoader<InputType1, nvec, aligned> ograd_loader(
      reinterpret_cast<const InputType1*>(param.inputs[1]) + base_row * param.num_elements,
      total_length);
    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      output_loader.load(i, total_length);
      ograd_loader.load(i, total_length);
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        output_persistent_storage[i*nvec + j] = IType0::from(output_loader.separate()[j]);
        ograd_persistent_storage[i*nvec + j] = IType1::from(ograd_loader.separate()[j]);
      }
    }
    output_row = output_persistent_storage +
                 my_local_row * param.num_elements +
                 output_loader.alignment();
    ograd_row = ograd_persistent_storage +
                my_local_row * param.num_elements +
                ograd_loader.alignment();
  } else {
    // less than rows_per_block rows to compute
    const index_t real_length = min(total_length,
                                    (param.total_rows - base_row) * param.num_elements);
    VectorizedLoader<InputType0, nvec, false> output_loader(
      reinterpret_cast<const InputType0*>(param.inputs[0]) + base_row * param.num_elements,
      real_length);
    VectorizedLoader<InputType1, nvec, false> ograd_loader(
      reinterpret_cast<const InputType1*>(param.inputs[1]) + base_row * param.num_elements,
      real_length);
    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      output_loader.load(i, real_length);
      ograd_loader.load(i, real_length);
#pragma unroll
      for (int j = 0; j < nvec; ++j) {
        output_persistent_storage[i*nvec + j] = IType0::from(output_loader.separate()[j]);
        ograd_persistent_storage[i*nvec + j] = IType1::from(ograd_loader.separate()[j]);
      }
    }
    output_row = output_persistent_storage +
                 my_local_row * param.num_elements +
                 output_loader.alignment();
    ograd_row = ograd_persistent_storage +
                my_local_row * param.num_elements +
                ograd_loader.alignment();
  }
  __syncthreads();

  AType my_sum;
  red::sum::SetInitValue(my_sum);

  for (index_t i = my_id; i < len; i += threads_per_row) {
    const AType val = OP1(ograd_row[i], output_row[i]);
    my_sum += val;
  }
  AType ssum;
  if (!reduction_inside_warp) {
    scratch[threadIdx.x] = my_sum;
    __syncthreads();
    for (int size = threads_per_row / 2; size >= warp_size; size /= 2) {
      if (my_id < size) {
        scratch[threadIdx.x] += scratch[threadIdx.x + size];
      }
      __syncthreads();
    }
    if (my_id < warp_size) {
      AType my_value = util::grouped_warp_allreduce(scratch[threadIdx.x],
                                                    [](AType x, AType y) { return x + y;},
                                                    min(threads_per_row, warp_size));
      scratch[threadIdx.x] = my_value;
    }
    __syncthreads();

    ssum = scratch[threadIdx.x - my_id];
    __syncthreads();
  } else {
      ssum = util::grouped_warp_allreduce(my_sum,
                                          [](AType x, AType y) { return x + y;},
                                          threads_per_row);
  }

  for (index_t i = my_id; i < param.num_elements; i += threads_per_row) {
    AType val = (i < len)
                ? OP2(ograd_row[i], output_row[i], ssum) / static_cast<AType>(param.temperature)
                : 0;
    output_row[i] = negate ? -val : val;
  }
  __syncthreads();

  if (only_full_blocks || blockIdx.x < gridDim.x - 1) {
    VectorizedStorer<OutputType0, nvec, aligned> storer(
      reinterpret_cast<OutputType0*>(param.outputs[0]) + base_row * param.num_elements,
      total_length);

    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      if (req == OpReqType::kAddTo) {
        storer.load(i, total_length);
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(op::add(output_persistent_storage[i*nvec + j],
                                                   OType::from(storer.separate()[j])));
        }
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(output_persistent_storage[i*nvec + j]);
        }
      }
      storer.store(i, total_length);
    }
  } else {
    const index_t real_length = min(total_length,
                                    (param.total_rows - base_row) * param.num_elements);
    VectorizedStorer<OutputType0, nvec, false> storer(
      reinterpret_cast<OutputType0*>(param.outputs[0]) + base_row * param.num_elements,
      real_length);

    for (index_t i = threadIdx.x; i < num_aligned_elements; i += blockDim.x) {
      if (req == OpReqType::kAddTo) {
        storer.load(i, real_length);
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(op::add(output_persistent_storage[i*nvec + j],
                                                   OType::from(storer.separate()[j])));
        }
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          storer.separate()[j] = OType::to(output_persistent_storage[i*nvec + j]);
        }
      }
      storer.store(i, real_length);
    }
  }
}
)code";

void SoftmaxRTCGradCompute::operator()(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using common::mshadow_type_info;
  using namespace common::cuda::rtc;
  using common::div_round;
  Stream<gpu>* s = ctx.get_stream<gpu>();
  if (softmax_use_length(attrs)) {
    if (req[1] != kNullOp) {
      cudaMemsetAsync(outputs[1].dptr_, 0,
                      outputs[1].Size() * mshadow_type_info(outputs[1].type_flag_).size,
                      Stream<gpu>::GetStream(s));
    }
  }
  if (req[0] == kNullOp || inputs[0].Size() == 0U) return;
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());
  const double temperature = param.temperature.has_value() ?
                             param.temperature.value() : 1.0;
  mxnet::TShape shape = AxisShapeCompact(inputs[0].shape_, &axis, true);

  int out_idx = softmax_has_dtype_override(attrs) ? 2 : 1;
  out_idx = softmax_use_length(attrs) ? 3 : out_idx;

  void* length_ptr = nullptr;
  std::string length_typename = "int";
  if (softmax_use_length(attrs)) {
    length_ptr = inputs[2].dptr_;
    length_typename = mshadow_type_info(inputs[2].type_flag_).name;
  }
  index_t M = shape[axis];
  if (M == 0 || shape.Size() == 0) return;
  index_t stride = 1;
  if (axis == shape.ndim() - 2) {
    stride = shape[shape.ndim() - 1];
  }
  const index_t N = shape.Size() / M;
  softmax_params params = {{inputs[out_idx].dptr_, inputs[0].dptr_, length_ptr},
                           {outputs[0].dptr_},
                           stride, M,
                           temperature, 1, N};
  std::string code = "#define OP1 " + OP1 + "\n"
                     "#define OP2 " + OP2 + "\n"
                     "const OpReqType req = " + util::to_string(req[0]) + ";\n"
                     "const bool negate = " + std::to_string(negate) + ";\n"
                     "using InputType2 = " + length_typename + ";\n";

  constexpr int nvec = 2;
  // Using 20 kB of shared memory for persistent storage in the optimized case
  const size_t acc_type_size = std::max(mshadow_type_info(inputs[0].type_flag_).acc_size,
                                        mshadow_type_info(outputs[0].type_flag_).acc_size);
  const size_t max_opt_M = 10 * 1024 / acc_type_size;
  int rows_per_block = get_rows_per_block(M, nvec, max_opt_M,
                                          vectorized_kernel_thread_num,
                                          N, ctx.run_ctx.ctx.dev_id);
  params.rows_per_block = rows_per_block;
  bool debug_softmax = dmlc::GetEnv("DEBUG_SOFTMAX_GRAD", false);
  if (!debug_softmax && stride == 1 &&
      static_cast<size_t>(M * rows_per_block) <= max_opt_M) {
    const int warp_size = 32;
    code += "const bool only_full_blocks = " + std::to_string(N % rows_per_block == 0) + ";\n"
            "const bool reduction_inside_warp = " +
            std::to_string(vectorized_kernel_thread_num / rows_per_block <= warp_size) + ";\n";
    int nblocks = div_round(N, rows_per_block);
    std::vector<TBlob> new_inputs = {inputs[out_idx], inputs[0]};
    if (softmax_use_length(attrs)) {
      new_inputs.emplace_back(inputs[2]);
    }
    std::vector<TBlob> new_outputs = {outputs[0]};
    VectorizedKernelRTCLauncher(code + softmax_common_functions,
                                "softmax_stride1_compute_grad_kernel",
                                softmax_stride1_kernel_bwd, nvec,
                                M * rows_per_block, N / rows_per_block, s, params,
                                new_inputs, new_outputs,
                                ctx.run_ctx.ctx.dev_id, 0, nblocks);
    MSHADOW_CUDA_POST_KERNEL_CHECK(softmax_stride1_compute_grad_kernel);
  } else {
    code += "using InputType0 = " + mshadow_type_info(inputs[out_idx].type_flag_).name + ";\n"
            "using InputType1 = " + mshadow_type_info(inputs[0].type_flag_).name + ";\n"
            "using OutputType0 = " + mshadow_type_info(outputs[0].type_flag_).name + ";\n";
    std::vector<const void*> args;
    args.emplace_back(&params);
    args.emplace_back(&M);
    const int warp_size = 32;
    int num_threads = std::min(static_cast<size_t>(128),
                               common::RoundToPower2(div_round(M, warp_size) * warp_size));
    if (stride != 1) {
      const int num_sms = MultiprocessorCount(ctx.run_ctx.ctx.dev_id);
      const index_t rows_per_sm = div_round(N, (512 / num_threads) * num_sms);
      params.rows_per_block = std::min(static_cast<size_t>(warp_size),
                                       common::RoundToPower2(rows_per_sm));
    }
    const auto& kernel_func = get_function(code + softmax_common_functions,
                                           "simple_softmax_grad_kernel",
                                           simple_softmax_kernel_bwd,
                                           ctx.run_ctx.ctx.dev_id);
    launch(kernel_func, div_round(N, params.rows_per_block), num_threads, 0, s, &args);
    MSHADOW_CUDA_POST_KERNEL_CHECK(simple_softmax_grad_kernel);
  }
}

NNVM_REGISTER_OP(softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxRTCCompute{"softmax_fwd"});

NNVM_REGISTER_OP(_backward_softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxRTCGradCompute{"op::mul", "softmax_bwd"});

NNVM_REGISTER_OP(masked_softmax)
.set_attr<FCompute>("FCompute<gpu>", MaskedSoftmaxCompute<gpu, mxnet_op::softmax_fwd,
                                                          false>);

NNVM_REGISTER_OP(_backward_masked_softmax)
.set_attr<FCompute>("FCompute<gpu>", MaskedSoftmaxGradCompute<gpu, op::mshadow_op::mul,
                                                              mxnet_op::softmax_bwd>);
}  // namespace op
}  // namespace mxnet
