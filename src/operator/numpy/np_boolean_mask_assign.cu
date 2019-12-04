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
 * \file np_boolean_mask_assign.cu
 * \brief GPU implementation of Boolean Mask Assign
 */

#include <cub/cub.cuh>
#include "../contrib/boolean_mask-inl.h"

namespace mxnet {
namespace op {

template<bool scalar>
struct BooleanAssignGPUKernel {
 private:
  static size_t __device__ bin_search(const size_t* idx,
                                      const size_t idx_size,
                                      const size_t i) {
    size_t left = 0, right = idx_size, mid = (left + right) / 2;
    while (left != right) {
      if (idx[mid] == i + 1) {
        if (idx[mid - 1] == i) {
          mid -= 1;
          break;
        } else if (idx[mid - 1] == i + 1) {
          right = mid;
          mid = (left + right) / 2;
        }
      } else if (idx[mid] == i) {
        if (idx[mid + 1] == i + 1) {
          break;
        } else {
          left = mid;
          mid = (left + right + 1) / 2;
        }
      } else if (idx[mid] < i + 1) {
        left = mid;
        mid = (left + right + 1) / 2;
      } else if (idx[mid] > i + 1) {
        right = mid;
        mid = (left + right) / 2;
      }
    }
    return mid;
  }

 public:
  template<typename DType>
  static void __device__ Map(int i,
                             DType* data,
                             const size_t* idx,
                             const size_t idx_size,
                             const size_t leading,
                             const size_t middle,
                             const size_t trailing,
                             const DType val) {
    // binary search for the turning point
    size_t m = i / trailing % middle;
    size_t mid = bin_search(idx, idx_size, m);
    // final answer is in mid
    data[i + (mid - m) * trailing] = val;
  }

  template<typename DType>
  static void __device__ Map(int i,
                             DType* data,
                             const size_t* idx,
                             const size_t idx_size,
                             const size_t leading,
                             const size_t middle,
                             const size_t trailing,
                             DType* tensor) {
    // binary search for the turning point
    size_t m = i / trailing % middle;
    size_t mid = bin_search(idx, idx_size, m);
    // final answer is in mid
    data[i + (mid - m) * trailing] = (scalar) ? tensor[0] : tensor[m];
  }
};

struct NonZeroWithCast {
  template<typename OType, typename IType>
  static void __device__ Map(int i, OType* out, const IType* in) {
    out[i] = (in[i]) ? OType(1) : OType(0);
  }
};

// completing the prefix_sum vector and return the pointer to it
template<typename DType>
size_t* GetValidNumGPU(const OpContext &ctx, const DType *idx, const size_t idx_size) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow_op;
  size_t* prefix_sum = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  Stream<gpu>* s = ctx.get_stream<gpu>();

  // Calculate total temporary memory size
  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size + 1,
                                Stream<gpu>::GetStream(s));
  size_t buffer_size = (idx_size + 1) * sizeof(size_t);
  temp_storage_bytes += buffer_size;
  // Allocate memory on GPU and allocate pointer
  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  prefix_sum = reinterpret_cast<size_t*>(workspace.dptr_);
  d_temp_storage = workspace.dptr_ + buffer_size;

  // Robustly set the bool values in mask
  // TODO(haojin2): Get a more efficient way to preset the buffer
  Kernel<set_zero, gpu>::Launch(s, idx_size + 1, prefix_sum);
  if (!std::is_same<DType, bool>::value) {
    Kernel<NonZeroWithCast, gpu>::Launch(s, idx_size, prefix_sum, idx);
  } else {
    Kernel<identity_with_cast, gpu>::Launch(s, idx_size, prefix_sum, idx);
  }

  // Calculate prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                prefix_sum,
                                prefix_sum,
                                idx_size + 1,
                                Stream<gpu>::GetStream(s));

  return prefix_sum;
}

void NumpyBooleanAssignForwardGPU(const nnvm::NodeAttrs& attrs,
                                  const OpContext &ctx,
                                  const std::vector<TBlob> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK(inputs.size() == 2U || inputs.size() == 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteInplace)
    << "Only WriteInplace is supported for npi_boolean_assign";

  Stream<gpu>* s = ctx.get_stream<gpu>();

  const TBlob& data = inputs[0];
  const TBlob& mask = inputs[1];
  // Get valid_num
  size_t mask_size = mask.shape_.Size();
  size_t valid_num = 0;
  size_t* prefix_sum = nullptr;
  if (mask_size != 0) {
    MSHADOW_TYPE_SWITCH(mask.type_flag_, MType, {
      prefix_sum = GetValidNumGPU<MType>(ctx, mask.dptr<MType>(), mask_size);
    });
    CUDA_CALL(cudaMemcpy(&valid_num, &prefix_sum[mask_size], sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  }
  // If there's no True in mask, return directly
  if (valid_num == 0) return;

  if (inputs.size() == 3U) {
    if (inputs[2].shape_.Size() != 1) {
      // tensor case, check tensor size with the valid_num
      CHECK_EQ(static_cast<size_t>(valid_num), inputs[2].shape_.Size())
        << "boolean array indexing assignment cannot assign " << inputs[2].shape_.Size()
        << " input values to the " << valid_num << " output values where the mask is true"
        << std::endl;
    }
  }

  size_t leading = 1U;
  size_t middle = mask_size;
  size_t trailing = 1U;

  if (inputs.size() == 3U) {
    if (inputs[2].shape_.Size() == 1) {
      MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
        Kernel<BooleanAssignGPUKernel<true>, gpu>::Launch(
          s, leading * valid_num * trailing, data.dptr<DType>(), prefix_sum, mask_size + 1,
          leading, middle, trailing, inputs[2].dptr<DType>());
      });
    } else {
      MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
        Kernel<BooleanAssignGPUKernel<false>, gpu>::Launch(
          s, leading * valid_num * trailing, data.dptr<DType>(), prefix_sum, mask_size + 1,
          leading, middle, trailing, inputs[2].dptr<DType>());
      });
    }
  } else {
    CHECK(attrs.dict.find("value") != attrs.dict.end())
      << "value is not provided";
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      Kernel<BooleanAssignGPUKernel<true>, gpu>::Launch(
        s, leading * valid_num * trailing, data.dptr<DType>(), prefix_sum, mask_size + 1,
        leading, middle, trailing, static_cast<DType>(std::stod(attrs.dict.at("value"))));
    });
  }
}

NNVM_REGISTER_OP(_npi_boolean_mask_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", NumpyBooleanAssignForwardGPU);

NNVM_REGISTER_OP(_npi_boolean_mask_assign_tensor)
.set_attr<FCompute>("FCompute<gpu>", NumpyBooleanAssignForwardGPU);

}  // namespace op
}  // namespace mxnet
