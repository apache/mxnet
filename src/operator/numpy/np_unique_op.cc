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
 * \file np_unique_op.cc
 */

#include "./np_unique_op.h"

namespace mxnet {
namespace op {

inline bool NumpyUniqueType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  for (size_t i = 1; i < out_attrs->size(); ++i) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, mshadow::kInt64);
  }
  return out_attrs->at(0) != -1;
}

inline bool NumpyUniqueStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  // CHECK_EQ(out_attrs->size(), 1U);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

struct UniqueComputeAuxCPUKernel {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // i is the index of out_data
  template<typename DType>
  MSHADOW_XINLINE static void Map(dim_t i, DType* out_data, const DType* in_data,
                                  const dim_t* idx, const dim_t M) {
    dim_t j = idx[i];
    std::memcpy(out_data + i * M, in_data + j * M, M * sizeof(DType));
  }
};

struct UniqueComputeMaskCPUKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(dim_t i,
                                  dim_t* out_data,
                                  const DType* in_data,
                                  const dim_t numel) {
    if (i == 0) {
      out_data[i] = 1;
    } else {
      out_data[i] = (std::memcmp(in_data + i * numel,
                     in_data + (i - 1) * numel, numel * sizeof(DType)) == 0) ? 0 : 1;
    }
  }
};

void NumpyUniqueCPUNoneAxisImpl(const NumpyUniqueParam& param,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();

    DType* input_data = inputs[0].data().dptr<DType>();
    dim_t input_size = inputs[0].shape().Size();
    if (param.return_index || param.return_inverse || param.return_counts) {
      // argsort, result in perm
      std::vector<dim_t> perm(input_size);
      std::iota(perm.begin(), perm.end(), 0);
      std::stable_sort(perm.begin(), perm.end(),
        [&input_data] (dim_t i1, dim_t i2) {return input_data[i1] < input_data[i2];});
      // sorted data in aux
      std::vector<DType> aux(input_size);
      mxnet_op::Kernel<UniqueComputeAuxCPUKernel, cpu>::Launch(
        stream, input_size, aux.data(), input_data, perm.data(), 1);
      // calculate unique mask
      std::vector<dim_t> mask(input_size);
      mxnet_op::Kernel<UniqueComputeMaskCPUKernel, cpu>::Launch(
        stream, input_size, mask.data(), aux.data(), 1);
      // Calculate prefix sum
      std::vector<int32_t> prefix_sum(input_size, 0);
      int32_t valid_num = 0;
      for (dim_t i = 0; i < input_size; i++) {
        prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
        prefix_sum[i] += (mask[i]) ? 1 : 0;
      }
      valid_num = prefix_sum[input_size - 1];
      // set the output shape forcefully
      mxnet::TShape s(1, valid_num);
      const_cast<NDArray &>(outputs[0]).Init(s);
      // launch kernal to obtain unique array, reuse boolean_mask kernel
      mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
        stream, input_size, outputs[0].data().dptr<DType>(), aux.data(),
        prefix_sum.data(), 1);
      // handle other optional outputs
      int output_flag = 0;
      if (param.return_index) {
        output_flag += 1;
        const_cast<NDArray &>(outputs[output_flag]).Init(s);
        dim_t* unique_indices = outputs[output_flag].data().dptr<dim_t>();
        // reuse boolean_mask kernel
        mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
          stream, input_size, unique_indices, perm.data(),
          prefix_sum.data(), 1);
      }
      if (param.return_inverse) {
        output_flag += 1;
        const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, input_size));
        dim_t* unique_inverse = outputs[output_flag].data().dptr<dim_t>();
        mxnet_op::Kernel<UniqueReturnInverseKernel, cpu>::Launch(
          stream, input_size, unique_inverse, prefix_sum.data(), perm.data());
      }
      if (param.return_counts) {
        output_flag += 1;
        std::vector<dim_t> idx(valid_num + 1);
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
        mxnet_op::Kernel<UniqueReturnCountsKernel, cpu>::Launch(
          stream, valid_num, unique_counts, idx.data());
      }
    } else {
      std::set<DType> set(input_data, input_data + input_size);
      mxnet::TShape s(1, set.size());
      const_cast<NDArray &>(outputs[0]).Init(s);
      std::copy(set.begin(), set.end(), outputs[0].data().dptr<DType>());
    }
  });
}

void NumpyUniqueCPUImpl(const NumpyUniqueParam& param,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
  CHECK(param.axis.value() >= -1 * inputs[0].shape().ndim() &&
      param.axis.value() < inputs[0].shape().ndim())
      << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<cpu> *stream = ctx.get_stream<cpu>();
    const index_t actual_axis =
        param.axis.value() + ((param.axis.value() < 0) ? inputs[0].shape().ndim() : 0);
    // reshape tensor to [origin_shape[axis], -1]
    const mxnet::TShape origin_shape = inputs[0].shape();
    Tensor<cpu, 3, DType> input_tensor_3d =
        inputs[0].data().FlatTo3D<cpu, DType>(actual_axis, stream);
    Tensor<cpu, 1, DType> workspace = ctx.requested[0].get_space_typed<cpu, 1, DType>(
        Shape1(input_tensor_3d.shape_.Size() * 2), stream);
    Tensor<cpu, 3, DType> input_tensor(workspace.dptr_, Shape3(
        input_tensor_3d.shape_[1], input_tensor_3d.shape_[0], input_tensor_3d.shape_[2]), stream);
    input_tensor = swapaxis<1, 0>(input_tensor_3d);
    const Shape<3> temp_shape = input_tensor.shape_;
    DType* input_data = input_tensor.dptr_;
    dim_t numel = temp_shape[1] * temp_shape[2];
    // argsort, result in perm
    std::vector<dim_t> perm(temp_shape[0]);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
      [&](dim_t a, dim_t b) -> bool {
        for (dim_t i = 0; i < numel; ++i) {
          DType lhs = input_data[i + a * numel];
          DType rhs = input_data[i + b * numel];
          if (lhs < rhs) {
            return true;
          } else if (lhs > rhs) {
            return false;
          }
        }
        return false;
      });
    // sorted data in aux
    Tensor<cpu, 2, DType> aux(workspace.dptr_ + input_tensor_3d.shape_.Size(),
        Shape2(temp_shape[0], temp_shape[1] * temp_shape[2]), stream);
    mxnet_op::Kernel<UniqueComputeAuxCPUKernel, cpu>::Launch(
      stream, temp_shape[0], aux.dptr_, input_data, perm.data(), numel);
    // calculate unique mask
    std::vector<dim_t> mask(temp_shape[0]);
    mxnet_op::Kernel<UniqueComputeMaskCPUKernel, cpu>::Launch(
      stream, temp_shape[0], mask.data(), aux.dptr_, numel);
    // calculate prefix sum
    std::vector<int32_t> prefix_sum(temp_shape[0], 0);
    int32_t valid_num = 0;
    for (dim_t i = 0; i < temp_shape[0]; i++) {
      prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
      prefix_sum[i] += (mask[i]) ? 1 : 0;
    }
    valid_num = prefix_sum[temp_shape[0] - 1];
    // store the temp output data, reuse the space of 'input_tensor'
    Tensor<cpu, 3, DType> temp_tensor(workspace.dptr_,
        Shape3(valid_num, temp_shape[1], temp_shape[2]), stream);
    // launch kernal to obtain unique array, reuse boolean_mask kernel
    mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
      stream, temp_shape[0], temp_tensor.dptr_, aux.dptr_,
      prefix_sum.data(), numel);
    // set the output shape forcefully and swap axis back
    mxnet::TShape out_shape(origin_shape);
    out_shape[actual_axis] = valid_num;
    const_cast<NDArray &>(outputs[0]).Init(out_shape);
    Tensor<cpu, 3, DType> output_tensor(outputs[0].data().dptr<DType>(),
        Shape3(temp_shape[1], valid_num, temp_shape[2]), stream);
    output_tensor = swapaxis<1, 0>(temp_tensor);
    // handle other optional outputs
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, valid_num));
      dim_t* unique_indices = outputs[output_flag].data().dptr<dim_t>();
      // reuse boolean_mask kernel
      mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
        stream, temp_shape[0], unique_indices, perm.data(),
        prefix_sum.data(), 1);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, temp_shape[0]));
      dim_t* unique_inverse = outputs[output_flag].data().dptr<dim_t>();
      mxnet_op::Kernel<UniqueReturnInverseKernel, cpu>::Launch(
        stream, temp_shape[0], unique_inverse, prefix_sum.data(), perm.data());
    }
    if (param.return_counts) {
      output_flag += 1;
      std::vector<dim_t> idx(valid_num + 1);
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
      mxnet_op::Kernel<UniqueReturnCountsKernel, cpu>::Launch(
          stream, valid_num, unique_counts, idx.data());
    }
  });
}

void NumpyUniqueCPUForward(const nnvm::NodeAttrs& attrs,
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
    Stream<cpu> *s = ctx.get_stream<cpu>();
    mxnet::TShape shape_1(1, 1);
    const_cast<NDArray &>(outputs[0]).Init(shape_1);
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      Tensor<cpu, 1, DType> unique_out =
          outputs[0].data().get_with_shape<cpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(unique_out, OpReqType::kWriteTo, inputs[0].data().dptr<DType>()[0]);
    });
    int output_flag = 0;
    if (param.return_index) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<cpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<cpu, 1, dim_t>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 0);
    }
    if (param.return_inverse) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<cpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<cpu, 1, dim_t>(Shape1(1), s);
      ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, 0);
    }
    if (param.return_counts) {
      output_flag += 1;
      const_cast<NDArray &>(outputs[output_flag]).Init(shape_1);
      Tensor<cpu, 1, dim_t> outdata =
          outputs[output_flag].data().get_with_shape<cpu, 1, dim_t>(Shape1(1), s);
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
      NumpyUniqueCPUNoneAxisImpl(param, ctx, inputs, req, outputs);
    } else {
      NumpyUniqueCPUImpl(param, ctx, inputs, req, outputs);
    }
  }
}

DMLC_REGISTER_PARAMETER(NumpyUniqueParam);

NNVM_REGISTER_OP(_npi_unique)
.set_attr_parser(ParamParser<NumpyUniqueParam>)
.set_num_inputs(1)
.set_num_outputs([](const NodeAttrs& attrs) {
    const NumpyUniqueParam& param = nnvm::get<NumpyUniqueParam>(attrs.parsed);
    int output_num = 1;
    if (param.return_index) output_num += 1;
    if (param.return_inverse) output_num += 1;
    if (param.return_counts) output_num += 1;
    return output_num;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferType>("FInferType", NumpyUniqueType)
.set_attr<FComputeEx>("FComputeEx<cpu>", NumpyUniqueCPUForward)
.set_attr<FInferStorageType>("FInferStorageType", NumpyUniqueStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input array")
.add_arguments(NumpyUniqueParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
