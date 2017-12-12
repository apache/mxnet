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
#include "../elemwise_op_common.h"
#include "./pooling-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_pooling-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_pooling-inl.h"
#endif  // MXNET_USE_MKLDNN

namespace mxnet {
namespace op {

static void PoolingParamParser(nnvm::NodeAttrs *attrs) {
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
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim()
                                       << "D pooling not supported";
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
  }
  CHECK_EQ(param_.stride.ndim(), param_.kernel.ndim())
      << "stride and kernel should have the same length";
  CHECK_EQ(param_.pad.ndim(), param_.kernel.ndim())
      << "pad and kernel should have the same length";
  attrs->parsed = std::move(param_);
}

int GetNumOutputs(const PoolingParam &param) {
#if MXNET_USE_MKLDNN == 1
  return MKLDNNRequireWorkspace(param) && SupportMKLDNNPooling(param) ? 2 : 1;
#else
  return 1;
#endif
}

int GetNumBackInputs(const PoolingParam &param) {
#if MXNET_USE_MKLDNN == 1
  return MKLDNNRequireWorkspace(param) && SupportMKLDNNPooling(param) ? 5 : 3;
#else
  return 3;
#endif
}

static bool PoolingType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  out_attrs->at(0) = in_attrs->at(0);
#if MXNET_USE_MKLDNN == 1
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  if (MKLDNNRequireWorkspace(param) && SupportMKLDNNPooling(param)) {
    CHECK_GT(out_attrs->size(), 1U);
    out_attrs->at(1) = mshadow::kInt32;
  }
#endif
  return true;
}

static bool PoolingShape(const nnvm::NodeAttrs &attrs,
                         std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) {
  const PoolingParam &param_ = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  const TShape &dshape = (*in_shape)[0];
  CHECK_GE(dshape.ndim(), 3U)
      << "Pooling: Input data should be  3D in (batch, channel, x)"
      << " Or 4D in (batch, channel, y, x) "
      << " Or 5D in (batch, channel, d, y, x)";
  TShape oshape = dshape;
  if (dshape.ndim() == 0) return false;
  if (param_.kernel.ndim() == 1) {
    CHECK_EQ(dshape.ndim(), 3U)
        << "Pooling: Input data should be 3D in (batch, channel, x)";
    if (param_.global_pool) {
      oshape[2] = 1;
    } else {
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
          << "kernel size (" << param_.kernel[0] << ") exceeds input ("
          << dshape[2] << " padded to " << (dshape[2] + 2 * param_.pad[0])
          << ")";
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 +
                    (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                        param_.stride[0];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[2] + 2 * param_.pad[0] -
                                               param_.kernel[0]) /
                            param_.stride[0]));
      }
    }
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_MKLDNN == 1
    if (MKLDNNRequireWorkspace(param_) && SupportMKLDNNPooling(param_))
      out_shape->push_back(oshape);   // for workspace
#endif
  } else if (param_.kernel.ndim() == 2) {
    CHECK_EQ(dshape.ndim(), 4U)
        << "Pooling: Input data should be 4D in (batch, channel, y, x)";
    if (param_.global_pool) {
      oshape[2] = 1;
      oshape[3] = 1;
    } else {
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
          << "kernel size (" << param_.kernel[0] << ") exceeds input ("
          << dshape[2] << " padded to " << (dshape[2] + 2 * param_.pad[0])
          << ")";
      CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
          << "kernel size (" << param_.kernel[1] << ") exceeds input ("
          << dshape[3] << " padded to " << (dshape[3] + 2 * param_.pad[1])
          << ")";
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 +
                    (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                        param_.stride[0];
        oshape[3] = 1 +
                    (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                        param_.stride[1];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[2] + 2 * param_.pad[0] -
                                               param_.kernel[0]) /
                            param_.stride[0]));
        oshape[3] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[3] + 2 * param_.pad[1] -
                                               param_.kernel[1]) /
                            param_.stride[1]));
      }
    }
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_MKLDNN == 1
    if (MKLDNNRequireWorkspace(param_) && SupportMKLDNNPooling(param_))
      out_shape->push_back(oshape);   // for workspace
#endif
  } else if (param_.kernel.ndim() == 3) {
    CHECK_EQ(dshape.ndim(), 5U)
        << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
    CHECK_LE(param_.kernel[0], dshape[2] + 2 * param_.pad[0])
        << "kernel size exceeds input";
    CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1])
        << "kernel size exceeds input";
    CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2])
        << "kernel size exceeds input";
    if (param_.global_pool) {
      oshape[2] = 1;
      oshape[3] = 1;
      oshape[4] = 1;
    } else {
      if (param_.pooling_convention == pool_enum::kValid) {
        oshape[2] = 1 +
                    (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                        param_.stride[0];
        oshape[3] = 1 +
                    (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                        param_.stride[1];
        oshape[4] = 1 +
                    (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) /
                        param_.stride[2];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[2] + 2 * param_.pad[0] -
                                               param_.kernel[0]) /
                            param_.stride[0]));
        oshape[3] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[3] + 2 * param_.pad[1] -
                                               param_.kernel[1]) /
                            param_.stride[1]));
        oshape[4] = 1 + static_cast<int>(ceil(
                            static_cast<float>(dshape[4] + 2 * param_.pad[2] -
                                               param_.kernel[2]) /
                            param_.stride[2]));
      }
    }

    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_MKLDNN == 1
    if (MKLDNNRequireWorkspace(param_) && SupportMKLDNNPooling(param_))
      out_shape->push_back(oshape);   // for workspace
#endif
  }
  return true;
}

void PoolingCompute_CPU(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
#if MXNET_USE_MKLDNN == 1
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  const NDArray *workspace = nullptr;
  if (MKLDNNRequireWorkspace(param)) {
    CHECK_GT(outputs.size(), 1U);
    workspace = &outputs[1];
  }
  if (SupportMKLDNN(inputs[0])
      && SupportMKLDNNPooling(param, inputs[0].shape())) {
    MKLDNNPooling_Forward(ctx, param, inputs[0], req[0], outputs[0],
                          workspace);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) in_blobs[i] = inputs[i].data();
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  PoolingCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

void PoolingGradCompute_CPU(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
#if MXNET_USE_MKLDNN == 1
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  const NDArray &out_grad = inputs[0];
  const NDArray *workspace = nullptr;
  const NDArray *in_data = nullptr;
  if (MKLDNNRequireWorkspace(param)) {
    // The first two elements are the gradient of the outputs in forward.
    // The third is the input of forward.
    // The fourth and the fifth are the outputs of forward.
    CHECK_EQ(inputs.size(), 5U);
    in_data = &inputs[2];
    workspace = &inputs[4];
  } else {
    CHECK_EQ(inputs.size(), 3U);
    in_data = &inputs[1];
  }
  const NDArray &in_grad = outputs[0];
  if (SupportMKLDNN(inputs[0])
      && SupportMKLDNNPooling(param, inputs[0].shape())) {
    MKLDNNPooling_Backward(ctx, param, out_grad, *in_data, workspace,
                           req[0], in_grad);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  PoolingGradCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

struct PoolingGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(
      const nnvm::NodePtr &n,
      const std::vector<nnvm::NodeEntry> &ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[pool_enum::kOut]);
    heads.push_back(n->inputs[pool_enum::kData]);
    heads.emplace_back(nnvm::NodeEntry{n, pool_enum::kOut, 0});
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

inline static bool PoolingStorageType(const nnvm::NodeAttrs &attrs,
                                      const int dev_mask,
                                      DispatchMode *dispatch_mode,
                                      std::vector<int> *in_attrs,
                                      std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);

#if MXNET_USE_MKLDNN == 1
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  auto expected = MKLDNNRequireWorkspace(param) && SupportMKLDNNPooling(param) ? 2 : 1;
  CHECK_EQ(out_attrs->size(), expected);
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNPooling(param)
      // There is no reason to use MKLDNN pooling if the input isn't in
      // MKLDNN format.
      && in_attrs->at(0) == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    for (size_t i = 0; i < out_attrs->size(); i++)
      (*out_attrs)[i] = kMKLDNNStorage;
    return true;
  }
#else
  CHECK_EQ(out_attrs->size(), 1);
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (size_t i = 0; i < out_attrs->size(); i++)
    (*out_attrs)[i] = kDefaultStorage;
  return true;
}

inline static bool backward_PoolingStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), GetNumBackInputs(param));
  CHECK_EQ(out_attrs->size(), 1);

#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNPooling(param)
      // There is no reason to use MKLDNN pooling if the input isn't in
      // MKLDNN format.
      && in_attrs->at(0) == kMKLDNNStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    for (size_t i = 0; i < out_attrs->size(); i++)
      (*out_attrs)[i] = kMKLDNNStorage;
    return true;
  }
#else
  CHECK_EQ(in_attrs->size(), 3);
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (size_t i = 0; i < out_attrs->size(); i++)
    (*out_attrs)[i] = kDefaultStorage;
  return true;
}

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
.set_num_outputs([](const NodeAttrs& attrs) {
  const PoolingParam &param = nnvm::get<PoolingParam>(attrs.parsed);
  return GetNumOutputs(param);
})
#if MXNET_USE_MKLDNN == 1
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                    [](const NodeAttrs& attrs) { return 1; })
#endif
.set_attr_parser(PoolingParamParser)
.set_attr<FInferStorageType>("FInferStorageType", PoolingStorageType)
.set_attr<nnvm::FInferType>("FInferType", PoolingType)
.set_attr<nnvm::FInferShape>("FInferShape", PoolingShape)
.set_attr<FCompute>("FCompute<cpu>", PoolingCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", PoolingCompute_CPU)
.set_attr<nnvm::FGradient>("FGradient",
                           ElemwiseGradUseInOut{"_backward_Pooling"})
.add_argument("data", "NDArray-or-Symbol",
              "Input data to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Pooling)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption",
    [](const NodeAttrs &attrs) {
#if MXNET_USE_CUDNN == 1
  return std::vector<std::pair<int, int> >();
#else
  return std::vector<std::pair<int, int> >{{1, 0}};
#endif
})
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr<FInferStorageType>("FInferStorageType",
                             backward_PoolingStorageType)
.set_attr_parser(PoolingParamParser)
.set_attr<FCompute>("FCompute<cpu>", PoolingGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", PoolingGradCompute_CPU);

}  // namespace op
}  // namespace mxnet
