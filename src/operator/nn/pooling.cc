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
 * \file pooling.cc
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/
#include "./pooling-inl.h"
#include "../elemwise_op_common.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_pooling-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {

static void PoolingParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  PoolingParam param_;
  param_.Init(attrs->dict);
  if (param_.kernel.ndim() == 1) {
    if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
    if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
  } else if (param_.kernel.ndim() == 2) {
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
  } else {
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D pooling not supported";
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
  }
  CHECK_EQ(param_.stride.ndim(), param_.kernel.ndim())
    << "stride and kernel should have the same length";
  CHECK_EQ(param_.pad.ndim(), param_.kernel.ndim())
    << "pad and kernel should have the same length";
  attrs->parsed = std::move(param_);
}

static bool PoolingShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_shape, std::vector<TShape> *out_shape) {
  const PoolingParam& param_ = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  const TShape &dshape = (*in_shape)[0];
  CHECK_GE(dshape.ndim(), 3U) << "Pooling: Input data should be  3D in (batch, channel, x)"
    << " Or 4D in (batch, channel, y, x) "
    << " Or 5D in (batch, channel, d, y, x)";
  TShape oshape = dshape;
  if (dshape.ndim() ==  0) return false;
  if (param_.kernel.ndim() == 1) {
    CHECK_EQ(dshape.ndim(), 3U) << "Pooling: Input data should be 3D in (batch, channel, x)";
    if (param_.global_pool) {
      oshape[2] = 1;
    } else {
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
        << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
        << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
          param_.stride[0];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[2] + 2 * param_.pad[0] -
                param_.kernel[0]) / param_.stride[0]));
      }
    }
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
  } else if (param_.kernel.ndim() == 2) {
    CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
    if (param_.global_pool) {
      oshape[2] = 1;
      oshape[3] = 1;
    } else {
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
        << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
        << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
      CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
        << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
        << " padded to " << (dshape[3] + 2*param_.pad[1]) << ")";
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
          param_.stride[0];
        oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
          param_.stride[1];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[2] + 2 * param_.pad[0] -
                param_.kernel[0]) / param_.stride[0]));
        oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[3] + 2 * param_.pad[1] -
                param_.kernel[1]) / param_.stride[1]));
      }
    }
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
  } else if (param_.kernel.ndim() == 3) {
    CHECK_EQ(dshape.ndim(), 5U)
      << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
    CHECK_LE(param_.kernel[0], dshape[2] + 2 * param_.pad[0]) << "kernel size exceeds input";
    CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1]) << "kernel size exceeds input";
    CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2]) << "kernel size exceeds input";
    if (param_.global_pool) {
      oshape[2] = 1;
      oshape[3] = 1;
      oshape[4] = 1;
    } else {
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
          param_.stride[0];
        oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
          param_.stride[1];
        oshape[4] = 1 + (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) /
          param_.stride[2];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[2] + 2 * param_.pad[0] -
                param_.kernel[0]) / param_.stride[0]));
        oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[3] + 2 * param_.pad[1] -
                param_.kernel[1]) / param_.stride[1]));
        oshape[4] = 1 + static_cast<int>(ceil(static_cast<float>(
                dshape[4] + 2 * param_.pad[2] -
                param_.kernel[2]) / param_.stride[2]));
      }
    }

    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
  }
  return true;
}

struct PoolingGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[pool_enum::kOut]);
    heads.push_back(n->inputs[pool_enum::kData]);
    heads.emplace_back(nnvm::NodeEntry{n, pool_enum::kOut, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(PoolingParam);

NNVM_REGISTER_OP(Pooling)
.describe(R"code(Performs pooling on the input.

The shapes for 1-D pooling are

- **data**: *(batch_size, channel, width)*,
- **out**: *(batch_size, num_filter, out_width)*.

The shapes for 2-D pooling are

- **data**: *(batch_size, channel, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The definition of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor((x+2*p-k)/s)+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil((x+2*p-k)/s)+1

But ``global_pool`` is set to be true, then do a global pooling, namely reset
``kernel=(height, width)``.

Three pooling options are supported by ``pool_type``:

- **avg**: average pooling
- **max**: max pooling
- **sum**: sum pooling

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(PoolingParamParser)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInferShape>("FInferShape", PoolingShape)
.set_attr<FCompute>("FCompute<cpu>", PoolingCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_Pooling"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Pooling)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
#if MXNET_USE_CUDNN == 1
  return std::vector<std::pair<int, int> >();
#else
  return std::vector<std::pair<int, int> >{{1, 0}};
#endif
})
.set_attr_parser(PoolingParamParser)
.set_attr<FCompute>("FCompute<cpu>", PoolingGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
