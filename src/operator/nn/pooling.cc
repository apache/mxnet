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
 * \file pooling.cc
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
 */
#include "../elemwise_op_common.h"
#include "./pooling-inl.h"
#include "../../common/alm.h"
#if MXNET_USE_ONEDNN == 1
#include "./dnnl/dnnl_base-inl.h"
#include "./dnnl/dnnl_pooling-inl.h"
#endif  // MXNET_USE_ONEDNN
namespace mxnet {
namespace op {

void PoolingParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  PoolingParam param;
  param.Init(attrs->dict);
  // Set default layout if it can be inferred from kernel shape.
  if (param.kernel.ndim() > 0)
    param.layout = param.GetLayout(param.kernel.ndim() + 2);
  if (param.kernel.ndim() == 1) {
    if (param.stride.ndim() == 0)
      param.stride = Shape1(1);
    if (param.pad.ndim() == 0)
      param.pad = Shape1(0);
  } else if (param.kernel.ndim() == 2) {
    if (param.stride.ndim() == 0)
      param.stride = Shape2(1, 1);
    if (param.pad.ndim() == 0)
      param.pad = Shape2(0, 0);
  } else {
    // ignore kernel size only if global_pool not assigned false
    if (param.global_pool == false && !param.IsAdaptivePooling()) {
      CHECK_EQ(param.kernel.ndim(), 3U) << param.kernel.ndim() << "D pooling not supported";
    }
    if (param.stride.ndim() == 0)
      param.stride = Shape3(1, 1, 1);
    if (param.pad.ndim() == 0)
      param.pad = Shape3(0, 0, 0);
  }

  attrs->parsed = std::move(param);
}

int GetNumOutputs(const PoolingParam& param) {
#if MXNET_USE_ONEDNN == 1
  return DNNLRequireWorkspace(param) && SupportDNNLPooling(param) ? 2 : 1;
#else
  return 1;
#endif
}

int GetNumBackInputs(const PoolingParam& param) {
#if MXNET_USE_ONEDNN == 1
  return DNNLRequireWorkspace(param) && SupportDNNLPooling(param) ? 5 : 3;
#else
  return 3;
#endif
}

static bool PoolingType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  out_attrs->at(0) = in_attrs->at(0);
#if MXNET_USE_ONEDNN == 1
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param)) {
    CHECK_GT(out_attrs->size(), 1U);
    out_attrs->at(1) = mshadow::kInt32;
  }
#endif
  return true;
}

static bool PoolingShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_shape,
                         mxnet::ShapeVector* out_shape) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  const mxnet::TShape& dshape = (*in_shape)[0];
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  if (param.pool_type == pool_enum::kLpPooling) {
    CHECK(param.p_value.has_value());
  }

  if (param.pooling_convention == pool_enum::kSame) {
    CHECK_EQ(dshape.ndim(), 3U) << "Pooling: Input data should be 3D in (batch, channel, x)"
                                << ". Currently 'same' supports Max Pooling 1-D";
    CHECK(param.pad[0] == 0 && param.pad[1] == 0 && param.pad[2] == 0)
        << "Same pooling convention disables the use of pad parameter.";
  }
  CHECK_GE(dshape.ndim(), 3) << "Pooling: Input data should be  3D in (batch, channel, x)"
                             << " Or 4D in (batch, channel, y, x) "
                             << " Or 5D in (batch, channel, d, y, x)";
  CHECK_LE(dshape.ndim(), 5) << "Pooling: Input data should be  3D in (batch, channel, x)"
                             << " Or 4D in (batch, channel, y, x) "
                             << " Or 5D in (batch, channel, d, y, x)";

  for (int i = 0; i < dshape.ndim(); i++) {
    CHECK_LT(dshape[i], INT32_MAX) << "Pooling does not support large"
                                   << " dimensions (>= 2^31).";
  }

  int layout = param.GetLayout(dshape.ndim());
  if (param.global_pool) {
    mxnet::TShape oshape = dshape;
    int c_index          = 0;
    switch (layout) {
      case mshadow::kNCW:
      case mshadow::kNCHW:
      case mshadow::kNCDHW:
        c_index = 1;
        break;
      case mshadow::kNWC:
      case mshadow::kNHWC:
      case mshadow::kNDHWC:
        c_index = dshape.ndim() - 1;
        break;
      default:
        LOG(FATAL) << "Unsupported tensor layout " << param.layout.value();
    }
    for (int i = 1; i < dshape.ndim(); i++)
      if (i != c_index)
        oshape[i] = 1;
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_ONEDNN == 1
    if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param))
      out_shape->push_back(oshape);  // for workspace
#endif
  } else if (param.kernel.ndim() == 0) {
    return false;
  } else if (param.kernel.ndim() == 1) {
    CHECK_EQ(dshape.ndim(), 3U) << "Pooling: Input data should be 3D in (batch, channel, x)";
    CHECK(layout == mshadow::kNCW || layout == mshadow::kNWC) << "Need 1D layout";
    // Perform shape calculations in a standard (NCW) layout space
    mshadow::Shape<3> dshape_ncw =
        (layout == mshadow::kNWC) ? ConvertLayout(dshape.get<3>(), mshadow::kNWC, mshadow::kNCW) :
                                    dshape.get<3>();
    mshadow::Shape<3> oshape_ncw = dshape_ncw;
    CHECK(param.kernel[0] <= dshape_ncw[2] + 2 * param.pad[0])
        << "kernel size (" << param.kernel[0] << ") exceeds input (" << dshape[2] << " padded to "
        << (dshape_ncw[2] + 2 * param.pad[0]) << ")";
    if (param.pooling_convention == pool_enum::kValid) {
      oshape_ncw[2] = 1 + (dshape_ncw[2] + 2 * param.pad[0] - param.kernel[0]) / param.stride[0];
    } else if (param.pooling_convention == pool_enum::kFull) {
      oshape_ncw[2] =
          1 + static_cast<int>(
                  std::ceil(static_cast<float>(dshape_ncw[2] + 2 * param.pad[0] - param.kernel[0]) /
                            param.stride[0]));
    } else {
      oshape_ncw[2] = static_cast<int>(
          std::ceil(static_cast<float>(dshape_ncw[2] + 2 * param.pad[0]) / param.stride[0]));
    }
    // Convert back from standard (NCW) layout space to the actual layout type
    mxnet::TShape oshape = (layout == mshadow::kNWC) ?
                               ConvertLayout(oshape_ncw, mshadow::kNCW, mshadow::kNWC) :
                               oshape_ncw;
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_ONEDNN == 1
    if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param))
      out_shape->push_back(oshape);  // for workspace
#endif
  } else if (param.kernel.ndim() == 2) {
    CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
    CHECK(layout == mshadow::kNCHW || layout == mshadow::kNHWC) << "Need 2D layout";
    // Perform shape calculations in a standard (NCHW) layout space
    mshadow::Shape<4> dshape_nchw =
        (layout == mshadow::kNHWC) ?
            ConvertLayout(dshape.get<4>(), mshadow::kNHWC, mshadow::kNCHW) :
            dshape.get<4>();
    mshadow::Shape<4> oshape_nchw = dshape_nchw;
    CHECK(param.kernel[0] <= dshape_nchw[2] + 2 * param.pad[0])
        << "kernel size (" << param.kernel[0] << ") exceeds input (" << dshape_nchw[2]
        << " padded to " << (dshape_nchw[2] + 2 * param.pad[0]) << ")";
    CHECK(param.kernel[1] <= dshape_nchw[3] + 2 * param.pad[1])
        << "kernel size (" << param.kernel[1] << ") exceeds input (" << dshape_nchw[3]
        << " padded to " << (dshape_nchw[3] + 2 * param.pad[1]) << ")";
    if (param.pooling_convention == pool_enum::kValid) {
      oshape_nchw[2] = 1 + (dshape_nchw[2] + 2 * param.pad[0] - param.kernel[0]) / param.stride[0];
      oshape_nchw[3] = 1 + (dshape_nchw[3] + 2 * param.pad[1] - param.kernel[1]) / param.stride[1];
    } else {
      oshape_nchw[2] =
          1 + static_cast<int>(
                  ceil(static_cast<float>(dshape_nchw[2] + 2 * param.pad[0] - param.kernel[0]) /
                       param.stride[0]));
      oshape_nchw[3] =
          1 + static_cast<int>(
                  ceil(static_cast<float>(dshape_nchw[3] + 2 * param.pad[1] - param.kernel[1]) /
                       param.stride[1]));
    }
    // Convert back from standard (NCHW) layout space to the actual layout type
    mxnet::TShape oshape = (layout == mshadow::kNHWC) ?
                               ConvertLayout(oshape_nchw, mshadow::kNCHW, mshadow::kNHWC) :
                               oshape_nchw;
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_ONEDNN == 1
    if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param))
      out_shape->push_back(oshape);  // for workspace
#endif
  } else if (param.kernel.ndim() == 3) {
    CHECK_EQ(dshape.ndim(), 5U) << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
    CHECK(layout == mshadow::kNCDHW || layout == mshadow::kNDHWC) << "Need 3D layout";
    // Perform shape calculations in a standard (NCDHW) layout space
    mshadow::Shape<5> dshape_ncdhw =
        (layout == mshadow::kNDHWC) ?
            ConvertLayout(dshape.get<5>(), mshadow::kNDHWC, mshadow::kNCDHW) :
            dshape.get<5>();
    mshadow::Shape<5> oshape_ncdhw = dshape_ncdhw;
    CHECK_LE(param.kernel[0], dshape_ncdhw[2] + 2 * param.pad[0]) << "kernel size exceeds input";
    CHECK_LE(param.kernel[1], dshape_ncdhw[3] + 2 * param.pad[1]) << "kernel size exceeds input";
    CHECK_LE(param.kernel[2], dshape_ncdhw[4] + 2 * param.pad[2]) << "kernel size exceeds input";
    if (param.pooling_convention == pool_enum::kValid) {
      oshape_ncdhw[2] =
          1 + (dshape_ncdhw[2] + 2 * param.pad[0] - param.kernel[0]) / param.stride[0];
      oshape_ncdhw[3] =
          1 + (dshape_ncdhw[3] + 2 * param.pad[1] - param.kernel[1]) / param.stride[1];
      oshape_ncdhw[4] =
          1 + (dshape_ncdhw[4] + 2 * param.pad[2] - param.kernel[2]) / param.stride[2];
    } else {
      oshape_ncdhw[2] =
          1 + static_cast<int>(
                  ceil(static_cast<float>(dshape_ncdhw[2] + 2 * param.pad[0] - param.kernel[0]) /
                       param.stride[0]));
      oshape_ncdhw[3] =
          1 + static_cast<int>(
                  ceil(static_cast<float>(dshape_ncdhw[3] + 2 * param.pad[1] - param.kernel[1]) /
                       param.stride[1]));
      oshape_ncdhw[4] =
          1 + static_cast<int>(
                  ceil(static_cast<float>(dshape_ncdhw[4] + 2 * param.pad[2] - param.kernel[2]) /
                       param.stride[2]));
    }
    // Convert back from standard (NCDHW) layout space to the actual layout type
    mxnet::TShape oshape = (layout == mshadow::kNDHWC) ?
                               ConvertLayout(oshape_ncdhw, mshadow::kNCDHW, mshadow::kNDHWC) :
                               oshape_ncdhw;
    out_shape->clear();
    out_shape->push_back(oshape);  // save output shape
#if MXNET_USE_ONEDNN == 1
    if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param))
      out_shape->push_back(oshape);  // for workspace
#endif
  }

  return true;
}

static bool PoolChangeLayout(nnvm::NodeAttrs* attrs,
                             mshadow::LayoutFlag targetLayout,
                             std::vector<alm::Transpose>* inpTransposes,
                             std::vector<alm::Transpose>* outTransposes) {
  CHECK_EQ(targetLayout, mshadow::kUNKNOWN);
  const auto& param = nnvm::get<PoolingParam>(attrs->parsed);
  CHECK(param.layout) << "Current layout of pooling should be known: " << attrs->name;
  auto layout = static_cast<mshadow::LayoutFlag>(param.layout.value());
  auto t      = alm::FactorCommonTranspose(inpTransposes);
  if (alm::IsIdentity(t))
    return false;
  outTransposes->assign(1, t);
  attrs->dict["layout"] = mshadow::toString(alm::ApplyTranspose(layout, alm::Reverse(t)));
  return true;
}

#if MXNET_USE_ONEDNN == 1
void PoolingComputeExCPU(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);

  if (SupportDNNLPooling(param, inputs[0])) {
    DNNL_OPCHECK_INIT(false, 1, inputs, outputs);
    DNNLRun(DNNLPoolingCompute, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(PoolingCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(PoolingCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

void PoolingGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);

  // Pooling does not currently support working with views
  if (inputs[0].IsView() || outputs[0].IsView()) {
    FallBackCompute(PoolingGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }

  if (SupportDNNLPooling(param, inputs[0])) {
    DNNL_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    DNNLRun(DNNLPoolingGradCompute, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(PoolingGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(PoolingGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool PoolingStorageType(const nnvm::NodeAttrs& attrs,
                                      const int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  bool support_dnnl_pool    = SupportDNNLPooling(param);

  return DNNLStorageType(attrs, dev_mask, support_dnnl_pool, dispatch_mode, in_attrs, out_attrs);
}

inline static bool BackwardPoolingStorageType(const nnvm::NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), GetNumBackInputs(param));
  CHECK_EQ(out_attrs->size(), 1);
  bool support_dnnl_pool = SupportDNNLPooling(param);

  return DNNLStorageType(attrs, dev_mask, support_dnnl_pool, dispatch_mode, in_attrs, out_attrs);
}
#endif

DMLC_REGISTER_PARAMETER(PoolingParam);

NNVM_REGISTER_OP(Pooling)
    .add_alias("_npx_pooling")
    .describe(R"code(Performs pooling on the input.

The shapes for 1-D pooling are

- **data** and **out**: *(batch_size, channel, width)* (NCW layout) or
  *(batch_size, width, channel)* (NWC layout),

The shapes for 2-D pooling are

- **data** and **out**: *(batch_size, channel, height, width)* (NCHW layout) or
  *(batch_size, height, width, channel)* (NHWC layout),

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The definition of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor((x+2*p-k)/s)+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil((x+2*p-k)/s)+1

When ``global_pool`` is set to be true, then global pooling is performed. It will reset
``kernel=(height, width)`` and set the appropiate padding to 0.

Three pooling options are supported by ``pool_type``:

- **avg**: average pooling
- **max**: max pooling
- **sum**: sum pooling
- **lp**: Lp pooling

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data and output will have shape *(batch_size, channel, depth,
height, width)* (NCDHW layout) or *(batch_size, depth, height, width, channel)* (NDHWC layout).

Notes on Lp pooling:

Lp pooling was first introduced by this paper: https://arxiv.org/pdf/1204.3968.pdf.
L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.
We can see that Lp pooling stands between those two, in practice the most common value for p is 2.

For each window ``X``, the mathematical expression for Lp pooling is:

:math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`

)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs([](const NodeAttrs& attrs) {
      const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
      return GetNumOutputs(param);
    })
#if MXNET_USE_ONEDNN == 1
    .set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                        [](const NodeAttrs& attrs) { return 1; })
#endif
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        const PoolingParam& param =
                                            nnvm::get<PoolingParam>(attrs.parsed);
                                        if (GetNumOutputs(param) == 2)
                                          return std::vector<std::string>{"output", "workspace"};
                                        else
                                          return std::vector<std::string>{"output"};
                                      })
    .set_attr_parser(PoolingParamParser)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FInferStorageType>("FInferStorageType", PoolingStorageType)
#endif
    .set_attr<nnvm::FInferType>("FInferType", PoolingType)
    .set_attr<mxnet::FInferShape>("FInferShape", PoolingShape)
    .set_attr<mxnet::alm::FChangeLayout>("FChangeLayout", PoolChangeLayout)
    .set_attr<FCompute>("FCompute<cpu>", PoolingCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", PoolingComputeExCPU)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_Pooling"})
    .add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
    .add_arguments(PoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Pooling)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
      // 1 input to fwd op and 2 * outputs from fwd op (fwd outputs and gradient inputs)
      return 1 + 2 * GetNumOutputs(param);
    })
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
// Different backend requires different FInplaceOption
#if MXNET_USE_ONEDNN == 1
                                      const PoolingParam& param =
                                          nnvm::get<PoolingParam>(attrs.parsed);
                                      if (DNNLRequireWorkspace(param) && SupportDNNLPooling(param))
                                        return std::vector<std::pair<int, int> >{{1, 0}};
#endif
                                      return std::vector<std::pair<int, int> >();
                                    })
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FInferStorageType>("FInferStorageType", BackwardPoolingStorageType)
#endif
    .set_attr_parser(PoolingParamParser)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", PoolingGradComputeExCPU)
#endif
    .set_attr<FCompute>("FCompute<cpu>", PoolingGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
