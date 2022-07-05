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
 * \file concat.cc
 * \brief
 * \author Bing Xu
 */

#include "../../common/utils.h"
#include "./concat-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_concat-inl.h"
#endif  // MXNET_USE_ONEDNN == 1

namespace mxnet {
namespace op {

bool ConcatShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector* in_shape,
                 mxnet::ShapeVector* out_shape) {
  using namespace mshadow;
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  mxnet::TShape dshape;
  dim_t size                = 0;
  int param_dim             = param_.dim.has_value() ? param_.dim.value() : 0;
  bool has_unknown_dim_size = false;
  int axis                  = -1;
  if (!param_.dim.has_value()) {
    for (int i = 0; i < param_.num_args; ++i) {
      (*in_shape)[i] = Shape1((*in_shape)[i].Size());
    }
  }
  for (int i = 0; i < param_.num_args; ++i) {
    mxnet::TShape tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      axis                 = CheckAxis(param_dim, tmp.ndim());
      has_unknown_dim_size = !mxnet::dim_size_is_known(tmp, axis) || has_unknown_dim_size;
      size += tmp[axis];
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis      = CheckAxis(param_dim, tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == -1)
    return false;
  CHECK_NE(dshape.ndim(), 0) << "zero-dimensional arrays cannot be concatenated";

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_unknown_dim_size)
    dshape[axis] = size;
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return shape_is_known(dshape);
}

// Concat for RNN param deals with the reverse shape inference from output
// for the special case of concatenating RNN parameters.
// The first (and sometimes the second) input may be unknown on the target axis.
// If the two inputs are unknown, they always have the same shape.
static bool RNNParamConcatShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector* in_shape,
                                mxnet::ShapeVector* out_shape) {
  using namespace mshadow;
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  mxnet::TShape dshape;
  index_t size = 0;
  std::vector<int> zero_indices;
  int axis      = -1;
  int param_dim = param_.dim.has_value() ? param_.dim.value() : 0;
  for (int i = 0; i < param_.num_args; ++i) {
    mxnet::TShape tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      axis = CheckAxis(param_dim, tmp.ndim());
      if (!mxnet::dim_size_is_known(tmp, axis)) {
        zero_indices.emplace_back(i);
      } else {
        CHECK_GE(tmp[axis], 0);
        size += tmp[axis];
      }
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis      = CheckAxis(param_dim, tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (!mxnet::ndim_is_known(dshape))
    return false;

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (zero_indices.empty())
    dshape[axis] = size;
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];
  if ((*out_shape)[0][axis] != -1 && !zero_indices.empty()) {
    int residual = (*out_shape)[0][axis] - size;
    CHECK_GE(residual, 0) << "Input size already exceeds output size. Residual: " << residual;
    CHECK(zero_indices.size() <= 2 && zero_indices.size() > 0)
        << "Expecting 1 or 2 inputs that need shape inference. Got: " << zero_indices.size();
    bool need_infer = !shape_is_known((*out_shape)[0]);
    for (int i : zero_indices) {
      (*in_shape)[i][axis] = residual / zero_indices.size();
      need_infer           = need_infer || !shape_is_known((*in_shape)[i]);
    }
    return !need_infer;
  }

  return shape_is_known(dshape);
}

bool ConcatType(const nnvm::NodeAttrs& attrs,
                std::vector<int>* in_type,
                std::vector<int>* out_type) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  int dtype                 = -1;

  // checks uniformity of input
  for (size_t i = 0; i < in_type->size(); ++i) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    } else {
      CHECK(in_type->at(i) == dtype || in_type->at(i) == -1)
          << "Non-uniform data type in " << attrs.op->name << ", expected data type "
          << mxnet::op::type_string(dtype) << ", got data type "
          << mxnet::op::type_string(in_type->at(i)) << " for input " << i;
    }
  }

  size_t nin = param_.num_args;

  // if in types are known out types are unknown
  if (dtype != -1 && (*out_type)[0] == -1) {
    (*out_type)[0] = dtype;
    in_type->clear();
    for (size_t i = 0; i < nin; ++i) {
      in_type->push_back(dtype);
    }
    // if out types are known in types are unknown
  } else if ((*out_type)[0] != -1 && dtype == -1) {
    in_type->clear();
    for (size_t i = 0; i < nin; ++i) {
      in_type->push_back((*out_type)[0]);
    }
    // if both out_types and in_types are known, and different
  } else if ((*out_type)[0] != -1 && dtype != -1 && ((*out_type)[0] != dtype)) {
    std::ostringstream os;
    os << "Type inconsistent, Provided output type = " << mxnet::op::type_string((*out_type)[0])
       << ',' << " inferred type = " << mxnet::op::type_string(dtype);
    throw mxnet::op::InferTypeError(os.str(), 0);
  }
  return true;
}

inline static bool ConcatForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int>* in_attrs,
                                                 std::vector<int>* out_attrs) {
  CHECK(!in_attrs->empty());
  CHECK_EQ(out_attrs->size(), 1U);
  auto& out_stype          = out_attrs->at(0);
  bool dispatched          = false;
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int param_dim            = param.dim.has_value() ? param.dim.value() : 0;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kCSRStorage) && param_dim == 0) {
    dispatched =
        storage_type_assign(&out_stype, kCSRStorage, dispatch_mode, DispatchMode::kFComputeEx);
  }
#if MXNET_USE_ONEDNN == 1
  if (!dispatched && dev_mask == mshadow::cpu::kDevMask &&
      common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched =
        storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode, DispatchMode::kFComputeEx);
  }
#endif  // MXNET_USE_ONEDNN == 1
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched =
        storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
#if MXNET_USE_ONEDNN == 1
  if (!DNNLEnvSet())
    *dispatch_mode = DispatchMode::kFComputeFallback;
#endif  // MXNET_USE_ONEDNN == 1
  return dispatched;
}

inline static bool BackwardConcatStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int>* in_attrs,
                                             std::vector<int>* out_attrs) {
  DispatchMode wanted_mode;
#if MXNET_USE_ONEDNN == 1
  CHECK_EQ(out_attrs->size(), in_attrs->size() - 1);
  if (dev_mask == mshadow::cpu::kDevMask && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage))
    wanted_mode = DispatchMode::kFComputeEx;
  else
#endif  // MXNET_USE_ONEDNN == 1
    wanted_mode = DispatchMode::kFCompute;
#if MXNET_USE_ONEDNN == 1
  if (!DNNLEnvSet())
    wanted_mode = DispatchMode::kFComputeFallback;
#endif  // MXNET_USE_ONEDNN == 1
  return storage_type_assign(out_attrs, mxnet::kDefaultStorage, dispatch_mode, wanted_mode);
}
#if MXNET_USE_ONEDNN == 1
// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_concat.html
bool SupportDNNLConcat(const std::vector<NDArray>& arrs) {
  for (auto& arr : arrs) {
    if (arr.IsView() || !SupportDNNL<2, 12, AllTypes>(arr))
      return false;
    int ndim             = arr.shape().ndim();
    const int dnnl_ndims = arr.GetDNNLData()->get_desc().data.ndims;
    if (ndim != dnnl_ndims) {
      return false;
    }
  }
  return true;
}
#endif  // MXNET_USE_ONEDNN == 1

static void ConcatComputeExCPU(const nnvm::NodeAttrs& attrs,
                               const OpContext& op_ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp)
    return;
  if (common::ContainsOnlyStorage(inputs, kCSRStorage) &&
      outputs[0].storage_type() == kCSRStorage) {
    ConcatCSRImpl<cpu>(attrs, op_ctx, inputs, req, outputs);
#if MXNET_USE_ONEDNN == 1
  } else if (SupportDNNLConcat(inputs)) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLConcatForward, attrs, op_ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(ConcatCompute<cpu>, attrs, op_ctx, inputs, req, outputs);
  } else if (common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    FallBackCompute(ConcatCompute<cpu>, attrs, op_ctx, inputs, req, outputs);
#endif  // MXNET_USE_ONEDNN == 1
  } else {
    LogUnimplementedOp(attrs, op_ctx, inputs, req, outputs);
  }
}

#if MXNET_USE_ONEDNN == 1
static void ConcatGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  if (SupportDNNLConcat(inputs)) {
    DNNL_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
    DNNLRun(DNNLConcatBackward, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(ConcatGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(ConcatGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif  // MXNET_USE_ONEDNN == 1

struct ConcatGrad {
  const char* op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
#if MXNET_USE_ONEDNN == 1
    for (size_t i = 0; i < n->inputs.size(); i++) {
      heads.push_back(n->inputs[i]);
    }
#endif  // MXNET_USE_ONEDNN == 1
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(ConcatParam);

#define CONCAT_FORWARD_ATTRS                                                                      \
  .set_num_inputs([](const NodeAttrs& attrs) {                                                    \
    const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);                             \
    return params.num_args;                                                                       \
  })                                                                                              \
      .set_num_outputs(1)                                                                         \
      .set_attr_parser(ParamParser<ConcatParam>)                                                  \
      .set_attr<nnvm::FListInputNames>("FListInputNames",                                         \
                                       [](const NodeAttrs& attrs) {                               \
                                         const ConcatParam& params =                              \
                                             nnvm::get<ConcatParam>(attrs.parsed);                \
                                         std::vector<std::string> ret;                            \
                                         ret.reserve(params.num_args);                            \
                                         for (int i = 0; i < params.num_args; ++i) {              \
                                           ret.push_back(std::string("arg") + std::to_string(i)); \
                                         }                                                        \
                                         return ret;                                              \
                                       })                                                         \
      .set_attr<nnvm::FListOutputNames>(                                                          \
          "FListOutputNames",                                                                     \
          [](const NodeAttrs& attrs) { return std::vector<std::string>{"output"}; })              \
      .set_attr<nnvm::FInferType>("FInferType", ConcatType)                                       \
      .set_attr<FInferStorageType>("FInferStorageType", ConcatForwardInferStorageType)            \
      .set_attr<FCompute>("FCompute<cpu>", ConcatCompute<cpu>)                                    \
      .set_attr<FComputeEx>("FComputeEx<cpu>", ConcatComputeExCPU)                                \
      .set_attr<nnvm::FGradient>("FGradient", ConcatGrad{"_backward_Concat"})                     \
      .set_attr<std::string>("key_var_num_args", "num_args")

NNVM_REGISTER_OP(Concat)
MXNET_ADD_SPARSE_OP_ALIAS(concat)
    .add_alias("concat")
    .add_alias("_npi_concatenate")
    .describe(R"code(Joins input arrays along a given axis.

.. note:: `Concat` is deprecated. Use `concat` instead.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated. With dimension parameter ``None`` input 
arrays are flattened before concatenating them along axis 0.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

The storage type of ``concat`` output depends on storage types of inputs

- concat(csr, csr, ..., csr, dim=0) = csr
- otherwise, ``concat`` generates output with default storage

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concat(x,y,z,dim=0) = [[ 1.,  1.],
                          [ 2.,  2.],
                          [ 3.,  3.],
                          [ 4.,  4.],
                          [ 5.,  5.],
                          [ 6.,  6.],
                          [ 7.,  7.],
                          [ 8.,  8.]]

   concat(x,y,z,dim=None) = [1., 1., 2., 2., 
                             3., 3., 4., 4.,
                             5., 5., 6., 6.,
                             7., 7., 8., 8.]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                         [ 4.,  4.,  7.,  7.],
                         [ 5.,  5.,  8.,  8.]]

)code" ADD_FILELINE)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<bool>("TIsDNNL", true)
#endif  // MXNET_USE_ONEDNN == 1
        CONCAT_FORWARD_ATTRS.set_attr<mxnet::FInferShape>("FInferShape", ConcatShape)
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
    .add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Concat)
    .add_alias("_backward_np_concat")
    .set_num_inputs([](const NodeAttrs& attrs) {
#if MXNET_USE_ONEDNN == 1
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return 1 + params.num_args;
#else
      return 1;
#endif
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return params.num_args;
    })
    .set_attr_parser(ParamParser<ConcatParam>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif  // MXNET_USE_ONEDNN == 1
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FInferStorageType>("FInferStorageType", BackwardConcatStorageType)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", ConcatGradComputeExCPU)
#endif  // MXNET_USE_ONEDNN == 1
    .set_attr<FCompute>("FCompute<cpu>", ConcatGradCompute<cpu>);

// _rnn_param_concat is a custom concat op with specialized infer_shape,
// which handles the case where the first one or two inputs may have
// unknown shape that can be inferred from output shape.
NNVM_REGISTER_OP(_rnn_param_concat)
    .add_alias("_npi_rnn_param_concat")
#if MXNET_USE_ONEDNN == 1
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif  // MXNET_USE_ONEDNN == 1
        CONCAT_FORWARD_ATTRS.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<mxnet::FInferShape>("FInferShape", RNNParamConcatShape)
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
    .add_arguments(ConcatParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
