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
 * \file np_boolean_assign.cc
 * \brief CPU implementation of Boolean Mask Assign
 */

#include "../../common/utils.h"
#include "../contrib/boolean_mask-inl.h"

namespace mxnet {
namespace op {

template<bool scalar = false>
struct BooleanAssignCPUKernel {
 private:
  static size_t bin_search(const size_t* idx,
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
  static void Map(int i,
                  DType* data,
                  const size_t* idx,
                  const size_t idx_size,
                  const size_t leading,
                  const size_t middle,
                  const size_t trailing,
                  const DType val) {
    // binary search for the turning point
    size_t mid = bin_search(idx, idx_size, i);
    // final answer is in mid
    for (size_t l = 0; l < leading; ++l) {
      for (size_t t = 0; t < trailing; ++t) {
        data[(l * middle + mid) * trailing + t] = val;
      }
    }
  }

  template<typename DType>
  static void Map(int i,
                  DType* data,
                  const size_t* idx,
                  const size_t idx_size,
                  const size_t leading,
                  const size_t middle,
                  const size_t valid_num,
                  const size_t trailing,
                  DType* tensor,
                  const bool broadcast = false) {
    // binary search for the turning point
    size_t mid = bin_search(idx, idx_size, i);
    // final answer is in mid
    for (size_t l = 0; l < leading; ++l) {
      for (size_t t = 0; t < trailing; ++t) {
        if (scalar) {
          data[(l * middle + mid) * trailing + t] = tensor[0];
        } else {
          data[(l * middle + mid) * trailing + t] =
            (broadcast) ?
            tensor[l * trailing + t] :
            tensor[(l * valid_num + i) * trailing + t];
        }
      }
    }
  }
};

bool BooleanAssignShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  CHECK(in_attrs->size() == 2U || in_attrs->size() == 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK(shape_is_known(in_attrs->at(0)) && shape_is_known(in_attrs->at(1)))
    << "shape of both input and mask should be known";
  const TShape& dshape = in_attrs->at(0);
  const TShape& mshape = in_attrs->at(1);
  const int start_axis = std::stoi(common::attr_value_string(attrs, "start_axis", "0"));

  for (int i = 0; i < mshape.ndim(); ++i) {
    CHECK_EQ(dshape[i + start_axis], mshape[i])
      << "boolean index did not match indexed array along dimension " << i + start_axis
      << "; dimension is " << dshape[i + start_axis] << " but corresponding boolean dimension is "
      << mshape[i];
  }

  // check if output shape is the same as the input data
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);

  // for tensor version, the tensor should have less than 1 dimension
  if (in_attrs->size() == 3U) {
    if (mshape.ndim() == dshape.ndim()) {
      CHECK_LE(in_attrs->at(2).ndim(), 1U)
        << "boolean array indexing assignment requires a 0 or 1-dimensional input, input has "
        << in_attrs->at(2).ndim() <<" dimensions";
    } else {
      const TShape& vshape = in_attrs->at(2);
      if (vshape.Size() > 1) {
        for (int i = 0; i < start_axis; ++i) {
          CHECK_EQ(dshape[i], vshape[i])
            << "shape mismatch of value with input at dimension " << i
            << "; dimension is " << dshape[i] << " but corresponding value dimension is "
            << vshape[i];
        }
        for (int i = 1; i <= (dshape.ndim() - start_axis - mshape.ndim()); ++i) {
          CHECK_EQ(dshape[dshape.ndim() - i], vshape[vshape.ndim() - i])
            << "shape mismatch of value with input at dimension " << (dshape.ndim() - i)
            << "; dimension is " << dshape[dshape.ndim() - i]
            << " but corresponding value dimension is " << vshape[vshape.ndim() - 1];
        }
      }
    }
  }

  return shape_is_known(out_attrs->at(0));
}

bool BooleanAssignType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK(in_attrs->size() == 2U || in_attrs->size() == 3U);
  CHECK_EQ(out_attrs->size(), 1U);

  // input and output should always have the same type
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  if (in_attrs->size() == 3U) {
    // if tensor version, the tensor should also have the same type as input
    TYPE_ASSIGN_CHECK(*in_attrs, 2, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(2));
    CHECK_NE(in_attrs->at(2), -1);
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1 && in_attrs->at(1) != -1;
}

// calculate the number of valid (masked) values, also completing the prefix_sum vector
template<typename DType>
size_t GetValidNumCPU(const DType* idx, size_t* prefix_sum, const size_t idx_size) {
  prefix_sum[0] = 0;
  for (size_t i = 0; i < idx_size; i++) {
    prefix_sum[i + 1] = prefix_sum[i] + ((idx[i]) ? 1 : 0);
  }
  return prefix_sum[idx_size];
}

void NumpyBooleanAssignForwardCPU(const nnvm::NodeAttrs& attrs,
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

  Stream<cpu>* s = ctx.get_stream<cpu>();

  const TBlob& data = inputs[0];
  const TShape& dshape = data.shape_;
  const TBlob& mask = inputs[1];
  const TShape& mshape = mask.shape_;
  const int start_axis = std::stoi(common::attr_value_string(attrs, "start_axis", "0"));
  // Get valid_num
  size_t valid_num = 0;
  size_t mask_size = mask.shape_.Size();
  std::vector<size_t> prefix_sum(mask_size + 1, 0);
  MSHADOW_TYPE_SWITCH_WITH_BOOL(mask.type_flag_, MType, {
    valid_num = GetValidNumCPU(mask.dptr<MType>(), prefix_sum.data(), mask_size);
  });
  // If there's no True in mask, return directly
  if (valid_num == 0) return;

  if (inputs.size() == 3U) {
    // tensor case
    const TShape& vshape = inputs.at(2).shape_;
    if (inputs[2].shape_.Size() != 1) {
      auto vndim = vshape.ndim();
      auto dndim = dshape.ndim();
      auto mndim = mshape.ndim();
      CHECK(vndim <= (dndim - mndim + 1));
      if ((vndim == (dndim - mndim + 1)) && (vshape[start_axis] != 1)) {
        // tensor case, check tensor size equal to or broadcastable with valid_num
        CHECK_EQ(static_cast<size_t>(valid_num), vshape[start_axis])
          << "boolean array indexing assignment cannot assign " << vshape
          << " input values to the " << valid_num << " output values where the mask is true"
          << std::endl;
      }
    }
  }

  size_t leading = 1U;
  size_t middle = mask_size;
  size_t trailing = 1U;

  for (int i = 0; i < dshape.ndim(); ++i) {
    if (i < start_axis) {
      leading *= dshape[i];
    }
    if (i >= start_axis + mshape.ndim()) {
      trailing *= dshape[i];
    }
  }

  if (inputs.size() == 3U) {
    // tensor case
    const TShape& vshape = inputs.at(2).shape_;
    MSHADOW_TYPE_SWITCH_WITH_BOOL(data.type_flag_, DType, {
      if (inputs[2].shape_.Size() == 1) {
        Kernel<BooleanAssignCPUKernel<true>, cpu>::Launch(
          s, valid_num, data.dptr<DType>(), prefix_sum.data(), prefix_sum.size(),
          leading, middle, valid_num, trailing, inputs[2].dptr<DType>());
      } else {
        bool need_broadcast = (vshape.ndim() == (dshape.ndim() - mshape.ndim() + 1)) ?
                              (vshape[start_axis] == 1) :
                              true;
        Kernel<BooleanAssignCPUKernel<false>, cpu>::Launch(
          s, valid_num, data.dptr<DType>(), prefix_sum.data(), prefix_sum.size(),
          leading, middle, valid_num, trailing, inputs[2].dptr<DType>(), need_broadcast);
      }
    });
  } else {
    // scalar case
    CHECK(attrs.dict.find("value") != attrs.dict.end()) << "value needs be provided";
    MSHADOW_TYPE_SWITCH_WITH_BOOL(data.type_flag_, DType, {
      Kernel<BooleanAssignCPUKernel<true>, cpu>::Launch(
        s, valid_num, data.dptr<DType>(), prefix_sum.data(), prefix_sum.size(),
        leading, middle, trailing, static_cast<DType>(std::stod(attrs.dict.at("value"))));
    });
  }
}

NNVM_REGISTER_OP(_npi_boolean_mask_assign_scalar)
.describe(R"code(Scalar version of boolean assign)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "mask"};
})
.set_attr<mxnet::FInferShape>("FInferShape", BooleanAssignShape)
.set_attr<nnvm::FInferType>("FInferType", BooleanAssignType)
.set_attr<FCompute>("FCompute<cpu>", NumpyBooleanAssignForwardCPU)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "input")
.add_argument("mask", "NDArray-or-Symbol", "mask")
.add_argument("value", "float", "value to be assigned to masked positions")
.add_argument("start_axis", "int", "starting axis of boolean mask");

NNVM_REGISTER_OP(_npi_boolean_mask_assign_tensor)
.describe(R"code(Tensor version of boolean assign)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "mask", "value"};
})
.set_attr<mxnet::FInferShape>("FInferShape", BooleanAssignShape)
.set_attr<nnvm::FInferType>("FInferType", BooleanAssignType)
.set_attr<FCompute>("FCompute<cpu>", NumpyBooleanAssignForwardCPU)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "input")
.add_argument("mask", "NDArray-or-Symbol", "mask")
.add_argument("value", "NDArray-or-Symbol", "assignment")
.add_argument("start_axis", "int", "starting axis of boolean mask");

}  // namespace op
}  // namespace mxnet
