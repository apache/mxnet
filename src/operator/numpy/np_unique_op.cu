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
 * Copyright (c) 2019 by Contributors
 * \file np_unique_op.cu
 */

#include "./np_unique_op.h"

namespace mxnet {
namespace op {

struct UniqueComputeAuxGPUKernel {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // i is the index of out_data
  template<typename DType>
  MSHADOW_XINLINE static void Map(dim_t i, DType* out_data, const DType* in_data,
                                  const dim_t* idx, const dim_t M) {
    dim_t j = idx[i/M];
    out_data[i] = in_data[j * M + i % M];
  }
};

struct UniqueComputeMaskGPUKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(dim_t i,
                                  dim_t* out_data,
                                  const DType* in_data,
                                  const dim_t numel) {
    if (i == 0) {
      out_data[i] = 1;
    } else {
      out_data[i] = 0;
      for (dim_t j = 0; j < numel; ++j) {
        if (in_data[(i - 1) * numel + j] != in_data[i * numel + j]) {
          out_data[i] = 1;
          break;
        }
      }
    }
  }
};

template<typename DType>
struct argsort_functor1d : public thrust::unary_function<dim_t, dim_t> {
  explicit argsort_functor1d(DType* data_) : data(data_) {}
  __device__
  bool operator()(dim_t a, dim_t b) {
    return data[a] < data[b];
  }
  DType* data;
};

template<typename DType>
struct argsort_functor2d : public thrust::unary_function<dim_t, dim_t> {
  argsort_functor2d(DType* data_, dim_t numel_) : data(data_), numel(numel_) {}
  __device__
  bool operator()(dim_t a, dim_t b) {
    for (dim_t i = 0; i < numel; ++i) {
      DType lhs = data[i + a * numel];
      DType rhs = data[i + b * numel];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
  DType* data;
  dim_t numel;
};

void NumpyUniqueGPUNoneAxisImpl(const NumpyUniqueParam& param,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  MXNET_NO_FLOAT16_TYPE_SWITCH(outputs[0].dtype(), DType, {
    mshadow::Stream<gpu> *stream = ctx.get_stream<gpu>();
    auto policy = thrust::cuda::par.on(stream->stream_);

    DType* input_data = inputs[0].data().dptr<DType>();
    dim_t input_size = inputs[0].shape().Size();
    // argsort, result in perm
    thrust::device_vector<dim_t> perm(input_size);
    thrust::sequence(policy, perm.begin(), perm.end());
    thrust::stable_sort(policy, perm.begin(), perm.end(),
                        argsort_functor1d<DType>(input_data));
    // sorted data in aux
    thrust::device_vector<DType> aux(input_size);
    mxnet_op::Kernel<UniqueComputeAuxGPUKernel, gpu>::Launch(
      stream, input_size, thrust::raw_pointer_cast(aux.data()), input_data,
      thrust::raw_pointer_cast(perm.data()), 1);
    // calculate unique mask
    thrust::device_vector<dim_t> mask(input_size);
    mxnet_op::Kernel<UniqueComputeMaskGPUKernel, gpu>::Launch(
      stream, input_size, thrust::raw_pointer_cast(mask.data()),
      thrust::raw_pointer_cast(aux.data()), 1);
    // Calculate prefix sum
    thrust::device_vector<int32_t> prefix_sum(input_size, 0);
    thrust::inclusive_scan(policy, mask.begin(), mask.end(), prefix_sum.begin());
    int32_t valid_num = 0;
    CUDA_CALL(cudaMemcpy(&valid_num, thrust::raw_pointer_cast(&prefix_sum[input_size - 1]),
                          sizeof(int32_t), cudaMemcpyDeviceToHost));
    // set the output shape forcefully
    mxnet::TShape s(1, valid_num);
    const_cast<NDArray &>(outputs[0]).Init(s);
    // launch kernal to obtain unique array, reuse boolean_mask kernel
    mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
      stream, input_size, outputs[0].data().dptr<DType>(),
      thrust::raw_pointer_cast(aux.data()), thrust::raw_pointer_cast(prefix_sum.data()), 1);
    // handle other optional outputs
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(s);
      dim_t* unique_indices = outputs[output_flag].data().dptr<dim_t>();
      // reuse boolean_mask kernel
      mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
        stream, input_size, unique_indices,
        thrust::raw_pointer_cast(perm.data()), thrust::raw_pointer_cast(prefix_sum.data()), 1);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, input_size));
      dim_t* unique_inverse = outputs[output_flag].data().dptr<dim_t>();
      mxnet_op::Kernel<UniqueReturnInverseKernel, gpu>::Launch(
        stream, input_size, unique_inverse,
        thrust::raw_pointer_cast(prefix_sum.data()), thrust::raw_pointer_cast(perm.data()));
    }
    if (param.return_counts) {
      output_flag += 1;
      thrust::device_vector<dim_t> idx(valid_num + 1);
      auto iter = idx.begin();
      for (dim_t i = 0; i < input_size; ++i) {
        if (mask[i]) {
          *iter = i;
          ++iter;
        }
      }
      *iter = input_size;
      const_cast<NDArray &>(outputs[output_flag]).Init(s);
      dim_t* unique_counts = outputs[output_flag].data().dptr<dim_t>();
      mxnet_op::Kernel<UniqueReturnCountsKernel, gpu>::Launch(
        stream, valid_num, unique_counts,
        thrust::raw_pointer_cast(idx.data()));
    }
  });
}

void NumpyUniqueGPUImpl(const NumpyUniqueParam& param,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
  CHECK(param.axis.value() >= -1 * inputs[0].shape().ndim() &&
      param.axis.value() < inputs[0].shape().ndim())
      << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
  MXNET_NO_FLOAT16_TYPE_SWITCH(outputs[0].dtype(), DType, {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<gpu> *stream = ctx.get_stream<gpu>();
    auto policy = thrust::cuda::par.on(stream->stream_);
    const index_t actual_axis =
        param.axis.value() + ((param.axis.value() < 0) ? inputs[0].shape().ndim() : 0);
    // reshape tensor to [origin_shape[axis], -1]
    const mxnet::TShape origin_shape = inputs[0].shape();
    Tensor<gpu, 3, DType> input_tensor_3d =
        inputs[0].data().FlatTo3D<gpu, DType>(actual_axis, stream);
    Tensor<gpu, 1, DType> workspace = ctx.requested[0].get_space_typed<gpu, 1, DType>(
        Shape1(input_tensor_3d.shape_.Size() * 2), stream);
    Tensor<gpu, 3, DType> input_tensor(workspace.dptr_, Shape3(
        input_tensor_3d.shape_[1], input_tensor_3d.shape_[0], input_tensor_3d.shape_[2]), stream);
    input_tensor = swapaxis<1, 0>(input_tensor_3d);
    const Shape<3> temp_shape = input_tensor.shape_;
    DType* input_data = input_tensor.dptr_;
    dim_t numel = temp_shape[1] * temp_shape[2];
    // argsort, result in perm
    thrust::device_vector<dim_t> perm(temp_shape[0]);
    thrust::sequence(policy, perm.begin(), perm.end());
    thrust::stable_sort(policy, perm.begin(), perm.end(),
                        argsort_functor2d<DType>(input_data, numel));
    // sorted data in aux
    Tensor<gpu, 2, DType> aux(workspace.dptr_ + input_tensor_3d.shape_.Size(),
        Shape2(temp_shape[0], temp_shape[1] * temp_shape[2]), stream);
    mxnet_op::Kernel<UniqueComputeAuxGPUKernel, gpu>::Launch(
      stream, temp_shape.Size(), aux.dptr_, input_data,
      thrust::raw_pointer_cast(perm.data()), numel);
    // calculate unique mask
    thrust::device_vector<dim_t> mask(temp_shape[0]);
    mxnet_op::Kernel<UniqueComputeMaskGPUKernel, gpu>::Launch(
      stream, temp_shape[0], thrust::raw_pointer_cast(mask.data()), aux.dptr_, numel);
    // calculate prefix sum
    thrust::device_vector<int32_t> prefix_sum(temp_shape[0], 0);
    thrust::inclusive_scan(policy, mask.begin(), mask.end(), prefix_sum.begin());
    int32_t valid_num = 0;
    CUDA_CALL(cudaMemcpy(&valid_num, thrust::raw_pointer_cast(&prefix_sum[temp_shape[0] - 1]),
                          sizeof(int32_t), cudaMemcpyDeviceToHost));
    // store the temp output data, reuse the space of 'input_tensor'
    Tensor<gpu, 3, DType> temp_tensor(workspace.dptr_,
        Shape3(valid_num, temp_shape[1], temp_shape[2]), stream);
    // launch kernal to obtain unique array, reuse boolean_mask kernel
    mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
      stream, temp_shape.Size(), temp_tensor.dptr_, aux.dptr_,
      thrust::raw_pointer_cast(prefix_sum.data()), numel);
    // set the output shape forcefully and swap axis back
    mxnet::TShape out_shape(origin_shape);
    out_shape[actual_axis] = valid_num;
    const_cast<NDArray &>(outputs[0]).Init(out_shape);
    Tensor<gpu, 3, DType> output_tensor(outputs[0].data().dptr<DType>(),
        Shape3(temp_shape[1], valid_num, temp_shape[2]), stream);
    output_tensor = swapaxis<1, 0>(temp_tensor);
    // handle other optional outputs
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, valid_num));
      dim_t* unique_indices = outputs[output_flag].data().dptr<dim_t>();
      // reuse boolean_mask kernel
      mxnet_op::Kernel<BooleanMaskForwardKernel, gpu>::Launch(
        stream, temp_shape[0], unique_indices, thrust::raw_pointer_cast(perm.data()),
        thrust::raw_pointer_cast(prefix_sum.data()), 1);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, temp_shape[0]));
      dim_t* unique_inverse = outputs[output_flag].data().dptr<dim_t>();
      mxnet_op::Kernel<UniqueReturnInverseKernel, gpu>::Launch(
        stream, temp_shape[0], unique_inverse,
        thrust::raw_pointer_cast(prefix_sum.data()), thrust::raw_pointer_cast(perm.data()));
    }
    if (param.return_counts) {
      output_flag += 1;
      thrust::device_vector<dim_t> idx(valid_num + 1);
      auto iter = idx.begin();
      for (dim_t i = 0; i < temp_shape[0]; ++i) {
        if (mask[i]) {
          *iter = i;
          ++iter;
        }
      }
      *iter = temp_shape[0];
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, valid_num));
      dim_t* unique_counts = outputs[output_flag].data().dptr<dim_t>();
      mxnet_op::Kernel<UniqueReturnCountsKernel, gpu>::Launch(
        stream, valid_num, unique_counts,
        thrust::raw_pointer_cast(idx.data()));
    }
  });
}

void NumpyUniqueGPUForward(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
  using namespace mshadow;
  const NumpyUniqueParam& param = nnvm::get<NumpyUniqueParam>(attrs.parsed);
  if (inputs[0].shape().ndim() == 0) {
    CHECK(!param.axis.has_value() || param.axis.value() == -1 || param.axis.value() == 0)
      << "Axis can only be -1 or 0 for scalor tensor";
    Stream<gpu> *s = ctx.get_stream<gpu>();
    mxnet::TShape shape_1(1, 1);
    const_cast<NDArray &>(outputs[0]).Init(shape_1);
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      CUDA_CALL(cudaMemcpy(outputs[0].data().dptr<DType>(), inputs[0].data().dptr<DType>(),
                           sizeof(DType), cudaMemcpyDeviceToDevice));
    });
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<gpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<gpu, 1, dim_t>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 0);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<gpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<gpu, 1, dim_t>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 0);
    }
    if (param.return_counts) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<gpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<gpu, 1, dim_t>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 1);
    }
  } else if (inputs[0].shape().Size() == 0) {
    // If the input tensor is zero size, only a check on axis is needed
    if (param.axis.has_value()) {
      int axis = param.axis.value();
      if (axis < 0) axis += inputs[0].shape().ndim();
      CHECK(axis >= 0 && axis < inputs[0].shape().ndim())
        << "Axis must be within the range of input tensor's dimension";
    }
    // set the shapes of outputs
    mxnet::TShape shape_0(1, 0);
    const_cast<NDArray &>(outputs[0]).Init(shape_0);
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_0);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_0);
    }
    if (param.return_counts) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_0);
    }
  } else {
    if (!param.axis.has_value()) {
      NumpyUniqueGPUNoneAxisImpl(param, ctx, inputs, req, outputs);
    } else {
      NumpyUniqueGPUImpl(param, ctx, inputs, req, outputs);
    }
  }
}

NNVM_REGISTER_OP(_npi_unique)
.set_attr<FComputeEx>("FComputeEx<gpu>", NumpyUniqueGPUForward);

}  // namespace op
}  // namespace mxnet
