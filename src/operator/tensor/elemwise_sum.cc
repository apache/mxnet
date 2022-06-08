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
 * \file elemwise_sum.cc
 * \brief CPU implementation of elementwise sum operator
 */
#include "./elemwise_sum.h"

#include "../../common/utils.h"
#include "../../ndarray/ndarray_function.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_sum-inl.h"
#endif  // MXNET_USE_ONEDNN == 1

namespace mxnet {
namespace op {

struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).describe("Number of inputs to be summed.");
  }
};

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

std::vector<nnvm::NodeEntry> ElementWiseSumGrad(const nnvm::ObjectPtr& n,
                                                const std::vector<nnvm::NodeEntry>& ograds) {
  // identity constraints in the beginning for easier shape inference.
  const nnvm::Op* copy_op = nnvm::Op::Get("identity");
  CHECK_EQ(ograds.size(), 1);
  std::vector<nnvm::NodeEntry> ret;
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    nnvm::ObjectPtr node = nnvm::Node::Create();
    node->attrs.op       = copy_op;
    node->inputs         = {ograds[0]};
    ret.emplace_back(std::move(node));
  }
  return ret;
}

bool ElementWiseSumShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<mxnet::TShape, shape_is_none, shape_assign, true, shape_string>(
      attrs, in_attrs, out_attrs, mxnet::TShape());
}

bool ElementWiseSumType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_attrs, out_attrs, -1);
}

bool ElementWiseSumForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  CHECK(!in_attrs->empty());
  CHECK_EQ(out_attrs->size(), 1U);
  bool ret =
      ElemwiseStorageAttr<false, true, false>(attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
#if MXNET_USE_ONEDNN == 1
  // We should always use FComputeEx.
  if (dev_mask == mshadow::cpu::kDevMask &&
      common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) &&
      out_attrs->at(0) == kDefaultStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

#if MXNET_USE_ONEDNN == 1
static inline bool IsDNNLData(const std::vector<NDArray>& arrs) {
  for (auto& arr : arrs) {
    if (!arr.IsDNNLData())
      return false;
  }
  return true;
}
#endif

void ElementWiseSumComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp)
    return;
#if MXNET_USE_ONEDNN == 1
  if (IsDNNLData(inputs)) {
    DNNLRun(DNNLSumForward, attrs, ctx, inputs, req, outputs);
  } else if (common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    FallBackCompute(ElementWiseSumCompute<cpu>, attrs, ctx, inputs, req, outputs);
  }
#endif
  else if (common::ContainsOnlyStorage(inputs, kRowSparseStorage) ||  // NOLINT(*)
           (inputs.size() == 3U && inputs[0].storage_type() == kDefaultStorage &&
            inputs[1].storage_type() == kCSRStorage &&
            inputs[2].storage_type() == kDefaultStorage) ||
           (inputs.size() > 4U && common::ContainsStorageType(inputs, kDefaultStorage) &&
            outputs[0].storage_type() == kDefaultStorage)) {
    mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
    Resource rsc            = ResourceManager::Get()->Request(ctx.run_ctx.get_ctx(),
                                                   ResourceRequest(ResourceRequest::kTempSpace));
    NDArray out_nd          = outputs[0];
    mxnet::ndarray::ElementwiseSum<cpu>(s, rsc, inputs, &out_nd);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(add_n)
MXNET_ADD_SPARSE_OP_ALIAS(add_n)
MXNET_ADD_SPARSE_OP_ALIAS(ElementWiseSum)
    .add_alias("ElementWiseSum")
    .add_alias("_npx_add_n")
    .describe(R"doc(Adds all input arguments element-wise.

.. math::
   add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

``add_n`` is potentially more efficient than calling ``add`` by `n` times.

The storage type of ``add_n`` output depends on storage types of inputs

- add_n(row_sparse, row_sparse, ..) = row_sparse
- add_n(default, csr, default) = default
- add_n(any input combinations longer than 4 (>4) with at least one default type) = default
- otherwise, ``add_n`` falls all inputs back to default storage and generates default storage

)doc" ADD_FILELINE)
    .set_attr_parser(ParamParser<ElementWiseSumParam>)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      uint32_t ret = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
      return ret;
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       uint32_t num_args =
                                           dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
                                       std::vector<std::string> ret;
                                       for (uint32_t i = 0; i < num_args; ++i) {
                                         ret.push_back(std::string("arg") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<FCompute>("FCompute<cpu>", ElementWiseSumCompute<cpu>)
    .set_attr<FComputeEx>("FComputeEx<cpu>", ElementWiseSumComputeExCPU)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
#endif
    .set_attr<mxnet::FInferShape>("FInferShape", ElementWiseSumShape)
    .set_attr<nnvm::FInferType>("FInferType", ElementWiseSumType)
    .set_attr<FInferStorageType>("FInferStorageType", ElementWiseSumForwardInferStorageType)
    .set_attr<nnvm::FGradient>("FGradient", ElementWiseSumGrad)
    .add_argument("args", "NDArray-or-Symbol[]", "Positional input arguments");

}  // namespace op
}  // namespace mxnet
