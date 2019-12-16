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
 * \file im2col.cc
 * \brief
 * \author Jiajun Wang
*/

#include "./im2col-inl.h"
#include "../operator_common.h"
#include "mxnet/op_attr_types.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(Im2colParam);
DMLC_REGISTER_PARAMETER(Col2imParam);

template<typename PType>
void SlidingParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  PType param_;
  try {
    param_.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  if (param_.kernel.ndim() == 1) {
    if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
    if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
  } else if (param_.kernel.ndim() == 2) {
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
  } else {
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D convolution not supported";
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
  }
  CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
    << "Stride must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while stride is "
    << param_.stride;
  CHECK_EQ(param_.kernel.ndim(), param_.dilate.ndim())
    << "Dilate must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while dilate is "
    << param_.dilate;
  CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
    << "Padding must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while padding is "
    << param_.pad;
  attrs->parsed = std::move(param_);
}

NNVM_REGISTER_OP(im2col)
.describe(R"(Extract sliding blocks from input array.

This operator is used in vanilla convolution implementation to transform the sliding
blocks on image to column matrix, then the convolution operation can be computed
by matrix multiplication between column and convolution weight. Due to the close
relation between im2col and convolution, the concept of **kernel**, **stride**,
**dilate** and **pad** in this operator are inherited from convolution operation.

Given the input data of shape :math:`(N, C, *)`, where :math:`N` is the batch size,
:math:`C` is the channel size, and :math:`*` is the arbitrary spatial dimension,
the output column array is always with shape :math:`(N, C \times \prod(\text{kernel}), W)`,
where :math:`C \times \prod(\text{kernel})` is the block size, and :math:`W` is the
block number which is the spatial size of the convolution output with same input parameters.
Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.

)" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(SlidingParser<Im2colParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U);
  const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
  if (mxnet::op::shape_is_none(in_shape->at(0))) {
    return false;
  }

  CHECK_GT(param.kernel.Size(), 0U) \
    << "incorrect kernel size: " << param.kernel;
  CHECK_GT(param.stride.Size(), 0U) \
    << "incorrect stride size: " << param.stride;
  CHECK_GT(param.dilate.Size(), 0U) \
    << "incorrect dilate size: " << param.dilate;

  index_t out_dim = 1;
  mxnet::TShape dshape(in_shape->at(0));
  for (int i = 0; i < param.kernel.ndim(); ++i) {
    const index_t pad_size = dshape[i + 2] + 2 * param.pad[i];
    const index_t dilated_kernel_size = param.DilatedKernelSize(i);
    CHECK_LE(dilated_kernel_size, pad_size)
      << "kernel size exceed input";
    const index_t output_size = (pad_size - dilated_kernel_size) / param.stride[i] + 1;
    out_dim *= output_size;
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape3(dshape[0], dshape[1] * param.kernel.Size(), out_dim));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1U);
  if (mxnet::op::type_is_none(in_type->at(0))) {
    return false;
  }

  int dtype = in_type->at(0);
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
})
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", Im2colCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_im2col"})
.add_argument("data", "NDArray-or-Symbol", "Input array to extract sliding blocks.")
.add_arguments(Im2colParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_im2col)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(SlidingParser<Im2colParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", Im2colGradCompute<cpu>);

NNVM_REGISTER_OP(col2im)
.describe(R"(Combining the output column matrix of im2col back to image array.

Like :class:`~mxnet.ndarray.im2col`, this operator is also used in the vanilla convolution
implementation. Despite the name, col2im is not the reverse operation of im2col. Since there
may be overlaps between neighbouring sliding blocks, the column elements cannot be directly
put back into image. Instead, they are accumulated (i.e., summed) in the input image
just like the gradient computation, so col2im is the gradient of im2col and vice versa.

Using the notation in im2col, given an input column array of shape
:math:`(N, C \times  \prod(\text{kernel}), W)`, this operator accumulates the column elements
into output array of shape :math:`(N, C, \text{output_size}[0], \text{output_size}[1], \dots)`.
Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.

)" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(SlidingParser<Col2imParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U);
  const Col2imParam& param = nnvm::get<Col2imParam>(attrs.parsed);
  if (mxnet::op::shape_is_none(in_shape->at(0))) {
    return false;
  }

  CHECK_EQ(param.kernel.ndim(), param.output_size.ndim())
    << "Output size must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param.kernel << " while output size is "
    << param.output_size;

  CHECK_GT(param.output_size.Size(), 0U) \
    << "incorrect output size: " << param.output_size;
  CHECK_GT(param.kernel.Size(), 0U) \
    << "incorrect kernel size: " << param.kernel;
  CHECK_GT(param.stride.Size(), 0U) \
    << "incorrect stride size: " << param.stride;
  CHECK_GT(param.dilate.Size(), 0U) \
    << "incorrect dilate size: " << param.dilate;

  const int spatial_size = param.kernel.ndim();
  mxnet::TShape dshape(in_shape->at(0));

  index_t out_dim = 1;
  for (int i = 0; i < spatial_size; ++i) {
    const index_t pad_size = param.output_size[i] + 2 * param.pad[i];
    const index_t dilated_kernel_size = param.DilatedKernelSize(i);
    CHECK_LE(dilated_kernel_size, pad_size)
      << "kernel size exceed output size";
    const index_t output_size = (pad_size - dilated_kernel_size) / param.stride[i] + 1;
    out_dim *= output_size;
  }

  CHECK_EQ(dshape[2], out_dim)
    << "output size does not match convolution parameters";
  CHECK_EQ(dshape[1] % param.kernel.Size(), 0)
    << "the second dim of input shape should be multiples of kernel size";

  mxnet::TShape oshape(param.kernel.ndim() + 2, 1);
  oshape[0] = dshape[0];
  oshape[1] = dshape[1] / param.kernel.Size();
  for (int i = 0; i < spatial_size; ++i) {
    oshape[i + 2] = param.output_size[i];
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1U);
  if (mxnet::op::type_is_none(in_type->at(0))) {
    return false;
  }

  int dtype = in_type->at(0);
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", Col2imCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_col2im"})
.add_argument("data", "NDArray-or-Symbol", "Input array to combine sliding blocks.")
.add_arguments(Col2imParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_col2im)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(SlidingParser<Col2imParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", Col2imGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
