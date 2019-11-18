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
 * Copyright (c) 2018 by Contributors
 * \file boolean_mask.cc
*/

#include "./boolean_mask-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BooleanMaskParam);

bool BooleanMaskType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return in_attrs->at(0) != -1 && in_attrs->at(1) != -1 && out_attrs->at(0) != -1;
}

bool BooleanMaskStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

bool BooleanMaskBackStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 2);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  for (size_t i = 0; i < out_attrs->size(); i++)
    out_attrs->at(i) = kDefaultStorage;
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

struct BooleanMaskBackwardCPUWriteKernel {
  template<typename DType>
  static void Map(int i,
                  DType* igrad,
                  const OpReqType /*req*/,
                  const DType* ograd,
                  const int32_t* idx,
                  const size_t col_size) {
    // i is row id already
    int32_t prev = (i == 0) ? 0 : idx[i - 1];
    int32_t curr = idx[i];
    if (prev != curr) {
      std::memcpy(igrad + i * col_size, ograd + prev * col_size, col_size * sizeof(DType));
    } else {
      std::memset(igrad + i * col_size, 0, col_size * sizeof(DType));
    }
  }
};

template<>
inline void BooleanMaskForward<cpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
  const BooleanMaskParam& param = nnvm::get<BooleanMaskParam>(attrs.parsed);
  const int axis = param.axis;
  const NDArray &data = inputs[0];
  const NDArray &idx = inputs[1];
  const NDArray &out = outputs[0];
  CHECK_EQ(axis, 0) << "Not supported yet";
  CHECK_EQ(data.shape()[axis], idx.shape()[0]);
  CHECK_EQ(idx.shape().ndim(), 1U);  // idx is required to be 1-d.
  // count the number of 1s in `idx`, so that we could know the output dimension
  size_t idx_size = idx.shape()[0];
  std::vector<int32_t> prefix_sum(idx_size, 0);
  size_t valid_num = 0;
  // Calculate prefix sum
  MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), DType, {
    DType* idx_dptr = idx.data().dptr<DType>();
    for (size_t i = 0; i < idx_size; i++) {
      prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
      prefix_sum[i] += (idx_dptr[i]) ? 1 : 0;
    }
    valid_num = prefix_sum[idx_size - 1];
  });
  // set the output shape forcefully
  mxnet::TShape s = data.shape();
  s[axis] = valid_num;

  const_cast<NDArray &>(out).Init(s);
  // do the copy
  MSHADOW_TYPE_SWITCH_WITH_BOOL(data.dtype(), DType, {
    size_t input_size = data.shape().Size();
    size_t col_size = input_size / idx_size;
    mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
    mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
      stream, idx_size, out.data().dptr<DType>(), data.data().dptr<DType>(),
      prefix_sum.data(), col_size);
  });
}

template<>
inline void BooleanMaskBackward<cpu>(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  if (req[0] == kNullOp) return;
  // inputs: {ograd, data, idx}
  // outputs: {igrad_data, igrad_idx}
  const NDArray& ograd = inputs[0];
  const NDArray& idx = inputs[2];
  const NDArray& igrad_data = outputs[0];
  MSHADOW_TYPE_SWITCH(igrad_data.dtype(), DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(idx.dtype(), IType, {
      size_t input_size = igrad_data.shape().Size();
      size_t idx_size = idx.shape()[0];
      size_t col_size = input_size / idx_size;
      std::vector<int32_t> prefix_sum(idx_size, 0);
      IType* idx_dptr = idx.data().dptr<IType>();
      for (size_t i = 0; i < idx_size; i++) {
        prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
        prefix_sum[i] += (idx_dptr[i]) ? 1 : 0;
      }
      mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
      if (req[0] == kAddTo) {
        mxnet_op::Kernel<BooleanMaskBackwardKernel, cpu>::Launch(
          stream, idx_size, igrad_data.data().dptr<DType>(), req[0],
          ograd.data().dptr<DType>(), prefix_sum.data(), col_size);
      } else {
        mxnet_op::Kernel<BooleanMaskBackwardCPUWriteKernel, cpu>::Launch(
          stream, idx_size, igrad_data.data().dptr<DType>(), req[0],
          ograd.data().dptr<DType>(), prefix_sum.data(), col_size);
      }
    });
  });
}

NNVM_REGISTER_OP(_contrib_boolean_mask)
.add_alias("_npi_boolean_mask")
.describe(R"code(
Given an n-d NDArray data, and a 1-d NDArray index,
the operator produces an un-predeterminable shaped n-d NDArray out,
which stands for the rows in x where the corresonding element in index is non-zero.

>>> data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
>>> index = mx.nd.array([0, 1, 0])
>>> out = mx.nd.contrib.boolean_mask(data, index)
>>> out

[[4. 5. 6.]]
<NDArray 1x3 @cpu(0)>

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BooleanMaskParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "index"};
  })
.set_attr<nnvm::FInferType>("FInferType", BooleanMaskType)
.set_attr<FComputeEx>("FComputeEx<cpu>", BooleanMaskForward<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", BooleanMaskStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_contrib_boolean_mask"})
.add_argument("data", "NDArray-or-Symbol", "Data")
.add_argument("index", "NDArray-or-Symbol", "Mask")
.add_arguments(BooleanMaskParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_contrib_boolean_mask)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BooleanMaskBackStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", BooleanMaskBackward<cpu>)
.add_arguments(BooleanMaskParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
