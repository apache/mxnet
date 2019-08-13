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
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
#include "./mkldnn/mkldnn_ops-inl.h"
#include "./mkldnn/mkldnn_base-inl.h"
#if MXNET_USE_NNPACK == 1
#include "../nnpack/nnpack_fully_connected-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {

bool SupportMKLDNNFC(const NDArray& input) {
  int ndim = input.shape().ndim();
  return input.dtype() == mshadow::kFloat32 && (ndim >= 1 && ndim <= 4) &&
         input.storage_type() == kDefaultStorage;
}

static bool FullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector *in_shape,
                                mxnet::ShapeVector *out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  if (!param.no_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  mxnet::TShape dshape = (*in_shape)[fullc::kData];
  mxnet::TShape oshape = (*out_shape)[0];
  // require data to be known
  if (!mxnet::ndim_is_known(dshape)) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim()-1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }
  SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param.num_hidden, num_input));
  if (!param.no_bias) {
    if (!shape_assign(&(*in_shape)[fullc::kBias], Shape1(param.num_hidden)) &&
        !shape_assign(&(*in_shape)[fullc::kBias], Shape2(param.num_hidden, 1))) {
      LOG(FATAL) << "Unexpected shape for bias " << (*in_shape)[fullc::kBias];
    }
  }

  if (!param.flatten) {
    mxnet::TShape result_shape(dshape);
    result_shape[dshape.ndim()-1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  if (oshape.ndim() > 0) {
    dshape[0] = oshape[0];
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kData, dshape);
  }
  return true;
}

void FullyConnectedComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const bool valid_data = inputs[0].storage_type() == kDefaultStorage;
  const bool valid_weight = inputs[1].storage_type() == kDefaultStorage ||
                            inputs[1].storage_type() == kRowSparseStorage;
  const bool valid_out = outputs[0].storage_type() == kDefaultStorage;
  bool valid_bias = true;
  if (!param.no_bias) {
    valid_bias = inputs[2].storage_type() == kDefaultStorage ||
                 inputs[2].storage_type() == kRowSparseStorage;
  }
#if MXNET_USE_MKLDNN == 1
  if (common::ContainsOnlyStorage(inputs, kDefaultStorage) &&
      common::ContainsOnlyStorage(outputs, kDefaultStorage)) {
    if (SupportMKLDNNFC(inputs[0])) {
      MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
      MKLDNNFCForward(attrs, ctx, inputs, req, outputs);
      MKLDNN_OPCHECK_RUN(FullyConnectedCompute<cpu>, attrs, ctx, inputs, req,
                         outputs);
    } else {
      FallBackCompute(FullyConnectedCompute<cpu>, attrs, ctx, inputs, req, outputs);
    }
    return;
  } else if (valid_data && valid_weight && valid_bias && valid_out) {
    // inputs
    std::vector<NDArray> temp_ndarrays;
    std::vector<TBlob> in_blobs;
    for (const NDArray& in : inputs) {
      // if ndarray is in default storage and MKLDNN is available,
      // need to make sure cpu layout data is used, instead of MKL layout
      if (in.storage_type() == kDefaultStorage) {
        temp_ndarrays.push_back(in.Reorder2Default());
        in_blobs.emplace_back(temp_ndarrays.back().data());
      } else {
        in_blobs.emplace_back(in.data());
      }
    }
    // output
    FullyConnectedCompute<cpu>(attrs, ctx, in_blobs, req, {outputs[0].data()});
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
#else
  if (valid_data && valid_weight && valid_bias && valid_out) {
    std::vector<TBlob> in_blobs(inputs.size());
    for (size_t i = 0; i < in_blobs.size(); i++) in_blobs[i] = inputs[i].data();
    std::vector<TBlob> out_blobs(outputs.size());
    for (size_t i = 0; i < out_blobs.size(); i++) out_blobs[i] = outputs[i].data();
    FullyConnectedCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
#endif
}

#if MXNET_USE_MKLDNN == 1
void FullyConnectedGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  if (SupportMKLDNNFC(inputs[0])) {
    MKLDNN_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    MKLDNNFCBackward(attrs, ctx, inputs, req, outputs);
    MKLDNN_OPCHECK_RUN(FullyConnectedGradCompute<cpu>, attrs, ctx, inputs, req,
                       outputs);
    return;
  }
  FallBackCompute(FullyConnectedGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

static bool FullyConnectedType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_GE(in_type->size(), 1U);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_type, out_type, -1);
}

struct FullyConnectedGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[fullc::kData]);
    heads.push_back(n->inputs[fullc::kWeight]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

inline static bool FCStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const bool valid_data = in_attrs->at(0) == kDefaultStorage;
  const bool valid_weight = in_attrs->at(1) == kDefaultStorage ||
                            in_attrs->at(1) == kRowSparseStorage;
  bool valid_bias = true;
  uint32_t in_expected = 2;
  if (!param.no_bias) {
    in_expected = 3;
    valid_bias = in_attrs->at(2) == kDefaultStorage || in_attrs->at(2) == kRowSparseStorage;
  }
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), 1);
  // dispatch to kFComputeEx is fine even if all inputs are dense and no MKL is present
  bool dispatched = false;
  if (!dispatched && valid_data && valid_weight && valid_bias) {
    dispatched = storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
#if MXNET_USE_MKLDNN == 1
  if (!MKLDNNEnvSet())
    *dispatch_mode = DispatchMode::kFComputeFallback;
#endif

  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

inline static bool BackwardFCStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), out_expected);
  // TODO(zhengda) let's disable MKLDNN for FullyConnected for now.
  // It seems there is a bug.
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, mxnet::kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && common::ContainsStorageType(*in_attrs, mxnet::kRowSparseStorage)) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  if (!dispatched) {
    dispatched = storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
#if MXNET_USE_MKLDNN == 1
  if (!MKLDNNEnvSet())
    *dispatch_mode = DispatchMode::kFComputeFallback;
#endif
  return dispatched;
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

NNVM_REGISTER_OP(FullyConnected)
MXNET_ADD_SPARSE_OP_ALIAS(FullyConnected)
.add_alias("_npx_fully_connected")
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

If ``flatten`` is set to be true, then the shapes are:

- **data**: `(batch_size, x1, x2, ..., xn)`
- **weight**: `(num_hidden, x1 * x2 * ... * xn)`
- **bias**: `(num_hidden,)`
- **out**: `(batch_size, num_hidden)`

If ``flatten`` is set to be false, then the shapes are:

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(num_hidden, input_dim)`
- **bias**: `(num_hidden,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

.. Note::

    The sparse support for FullyConnected is limited to forward evaluation with `row_sparse`
    weight and bias, where the length of `weight.indices` and `bias.indices` must be equal
    to `num_hidden`. This could be useful for model inference with `row_sparse` weights
    trained with importance sampling or noise contrastive estimation.

    To compute linear transformation with 'csr' sparse data, sparse.dot is recommended instead
    of sparse.FullyConnected.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<FInferStorageType>("FInferStorageType", FCStorageType)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  if (!params.no_bias) {
    return std::vector<std::string>{"data", "weight", "bias"};
  } else {
    return std::vector<std::string>{"data", "weight"};
  }
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr<mxnet::FInferShape>("FInferShape", FullyConnectedShape)
.set_attr<nnvm::FInferType>("FInferType", FullyConnectedType)
.set_attr<FCompute>("FCompute<cpu>", FullyConnectedCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FullyConnectedComputeExCPU)
.set_attr<nnvm::FGradient>("FGradient", FullyConnectedGrad{"_backward_FullyConnected"})
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_FullyConnected)
.set_num_inputs(3)
.set_num_outputs([](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr<FInferStorageType>("FInferStorageType", BackwardFCStorageType)
.set_attr_parser(ParamParser<FullyConnectedParam>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", FullyConnectedGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", FullyConnectedGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
