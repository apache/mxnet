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
 * Copyright (c) 2018 Intel Corporation
 * \file ngraph.cc
 * \brief ngraph subgraph property for mxnet
*/

#if MXNET_USE_NGRAPH
#include <mxnet/ndarray.h>
#include <ngraph_graph.h>
#include <ngraph_imperative.h>
#include <ngraph_nnvm_ops.h>
#include <ngraph_sgcompiler_utils.h>

#include "../subgraph/common.h"
#include "../subgraph/subgraph_property.h"
#include "./ngraph-inl.h"

namespace mxnet {
namespace op {

std::shared_ptr<ngraph_bridge::Graph> get_ngraph(const NodeAttrs &attrs) {
  auto compiler =
      nnvm::get<std::shared_ptr<ngraph_bridge::Compiler>>(attrs.parsed);
  return compiler->GetNgraph();
}

class NgraphSubgraphOperator {
 public:
  explicit NgraphSubgraphOperator(std::shared_ptr<ngraph_bridge::Graph> ngraph)
      : ngraph_(ngraph) {}
  void Forward(const OpContext &ctx, const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);
  void Backward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs);

 private:
  std::shared_ptr<ngraph_bridge::Graph> ngraph_;
};

void NgraphSubgraphOperator::Forward(const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  compute_forward(ctx, ngraph_, inputs, req, outputs);
}

void NgraphSubgraphOperator::Backward(const OpContext &ctx,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<NDArray> &outputs) {
  compute_backward(ctx, ngraph_, inputs, req, outputs);
}

OpStatePtr CreateNgraphSubgraphOpState(const NodeAttrs &attrs, Context ctx,
                                       const std::vector<TShape> &in_shapes,
                                       const std::vector<int> &in_types) {
  return OpStatePtr::Create<NgraphSubgraphOperator>(get_ngraph(attrs));
}

void NgraphSubgraphOpForward(const OpStatePtr &state_ptr, const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
  NgraphSubgraphOperator &op = state_ptr.get_state<NgraphSubgraphOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

void NgraphSubgraphOpBackward(const OpStatePtr &state_ptr, const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  NgraphSubgraphOperator &op = state_ptr.get_state<NgraphSubgraphOperator>();
  op.Backward(ctx, inputs, req, outputs);
}

std::vector<nnvm::NodeEntry> NgraphSubgraphGradient(
    const nnvm::NodePtr &n, const std::vector<nnvm::NodeEntry> &ograds) {
  auto graph = get_ngraph(n->attrs);
  const bool zero_grad = check_zero_grad(graph);
  graph->zero_grad = zero_grad;
  auto is_loss = graph->is_loss;
  auto p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward_ngraph_subgraph_op");
  p->attrs.parsed = n->attrs.parsed;
  if (std::find(begin(is_loss), end(is_loss), true) == end(is_loss) &&
      zero_grad && graph->num_outputs_ == 1) {
    return mxnet::op::MakeZeroGradNodes(n, ograds);
  }
  if (!graph->need_grad) {
    LOG(FATAL)
        << "NGRAPH_BRIDGE: This graph was compiled as inference but "
        << "is called in training";
  }
  p->attrs.name = n->attrs.name + "_backward";
  p->attrs.dict = n->attrs.dict;
  p->control_deps.emplace_back(n);
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  if (!zero_grad) {
    for (size_t i = 0; i < graph->num_adjoints_; ++i) {
      if (!is_loss[i]) {
        p->inputs.push_back(ograds[i]);
      }
    }
  }
  p->inputs.insert(p->inputs.end(), n->inputs.begin(), n->inputs.end());
  for (unsigned i = graph->outputs_.size();
       i < graph->fprop_cache->fprop->get_results().size(); ++i) {
    p->inputs.emplace_back(nnvm::NodeEntry{n, i, 0});
  }
  std::vector<nnvm::NodeEntry> ret;
  for (unsigned i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(nnvm::NodeEntry{p, i, 0});
  }
  return ret;
}

std::vector<std::string> NgraphSubgraphListNodeNames(
    const std::vector<ngraph_bridge::NodePtr> &nodes) {
  std::vector<std::string> names;
  for (const auto &n : nodes) {
    names.emplace_back(n->name_);
  }
  return names;
}
std::vector<std::string> NgraphSubgraphListInputNames(
    const nnvm::NodeAttrs &attrs) {
  auto graph = get_ngraph(attrs);
  return NgraphSubgraphListNodeNames(graph->inputs_);
}
std::vector<std::string> NgraphSubgraphListOutputNames(
    const nnvm::NodeAttrs &attrs) {
  auto graph = get_ngraph(attrs);
  auto names = NgraphSubgraphListNodeNames(graph->outputs_);
  for (size_t i = names.size(); i < graph->get_results().size(); ++i) {
    names.push_back(graph->name_ + "_output_" + std::to_string(i));
  }
  return names;
}
bool NgraphSubgraphInferShape(const nnvm::NodeAttrs &attrs,
                              std::vector<nnvm::TShape> *in_attrs,
                              std::vector<nnvm::TShape> *out_attrs) {
  auto compiler =
      nnvm::get<std::shared_ptr<ngraph_bridge::Compiler>>(attrs.parsed);
  auto graph = get_ngraph(attrs);
  if ((graph->inputs_.size() > 0) &&
      (*in_attrs)[0] != graph->inputs_[0]->shape_) {
    compiler->ReshapeGraph(*in_attrs);
    graph = compiler->GetNgraph();
  }
  for (size_t i = 0; i < graph->inputs_.size(); ++i) {
    (*in_attrs)[i] = graph->inputs_[i]->shape_;
  }
  size_t i = 0;
  for (const auto& output : graph->get_results()) {
    auto tmp_shape = ngraph_bridge::NShape_to_TShape(output->get_shape());
    (*out_attrs)[i] = tmp_shape;
    i += 1;
  }
  return true;
}
bool NgraphSubgraphInferType(const nnvm::NodeAttrs &attrs,
                             std::vector<int> *iattr, std::vector<int> *oattr) {
  auto graph = get_ngraph(attrs);
  for (size_t i = 0; i < graph->inputs_.size(); ++i) {
    (*iattr)[i] = graph->inputs_[i]->dtype_;
  }
  std::vector<int> dtypes;
  for (const auto& output : graph->get_results()) {
    dtypes.push_back(ngraph_bridge::getType(output->get_element_type()));
  }
  for (size_t i = 0; i < dtypes.size(); ++i) {
    mxnet::op::type_assign(&((*oattr)[i]), dtypes[i]);
  }
  return true;
}

bool NgraphSubgraphInferStorageType(const nnvm::NodeAttrs &attrs,
                                    const int dev_mask,
                                    mxnet::DispatchMode *dispatch_mode,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  if (in_attrs->size() > 0)
    mxnet::op::storage_type_assign(in_attrs, mxnet::kDefaultStorage,
                                   dispatch_mode,
                                   mxnet::DispatchMode::kFComputeEx);
  return mxnet::op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                        dispatch_mode,
                                        mxnet::DispatchMode::kFComputeEx);
}
bool NgraphSubgraphBackwardInferStorageType(const nnvm::NodeAttrs &attrs,
                                            const int dev_mask,
                                            mxnet::DispatchMode *dispatch_mode,
                                            std::vector<int> *in_attrs,
                                            std::vector<int> *out_attrs) {
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  mxnet::op::storage_type_assign(in_attrs, mxnet::kDefaultStorage,
                                 dispatch_mode,
                                 mxnet::DispatchMode::kFComputeEx);
  return mxnet::op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                        dispatch_mode,
                                        mxnet::DispatchMode::kFComputeEx);
}
std::vector<uint32_t> NGraphSubgraphMutateInputs(const nnvm::NodeAttrs &attrs) {
  auto graph = get_ngraph(attrs);
  std::vector<uint32_t> mutate_vars;
  for (size_t i = 0; i < graph->inputs_.size(); ++i) {
    if (graph->inputs_[i]->type_ == ngraph_bridge::NodeType::kAux) {
      mutate_vars.emplace_back(i);
    }
  }
  return mutate_vars;
}

NNVM_REGISTER_OP(_ngraph_subgraph_op)
    .describe(R"code(_ngraph_subgraph_op)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs &attrs) {
      auto graph = get_ngraph(attrs);
      return graph->inputs_.size();
    })
    .set_num_outputs([](const NodeAttrs &attrs) {
      auto graph = get_ngraph(attrs);
      return graph->get_results().size();
    })
    .set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                        [](const NodeAttrs& attrs) {
                                          auto graph = get_ngraph(attrs);
                                          return graph->outputs_.size();
                                        })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     NgraphSubgraphListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      NgraphSubgraphListOutputNames)
    .set_attr<FCreateOpState>("FCreateOpState", CreateNgraphSubgraphOpState)
    .set_attr<nnvm::FInferShape>("FInferShape", NgraphSubgraphInferShape)
    .set_attr<nnvm::FInferType>("FInferType", NgraphSubgraphInferType)
    .set_attr<FInferStorageType>("FInferStorageType",
                                 NgraphSubgraphInferStorageType)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                  NgraphSubgraphOpForward)
    .set_attr<nnvm::FGradient>("FGradient", NgraphSubgraphGradient)
    .set_attr<nnvm::FMutateInputs>("FMutateInputs", NGraphSubgraphMutateInputs)
    .set_attr<std::string>("key_var_num_args", "num_args")
    .add_argument("data", "NDArray-or-Symbol[]", "input data list");

NNVM_REGISTER_OP(_backward_ngraph_subgraph_op)
    .set_num_inputs([](const NodeAttrs &attrs) {
      auto graph = get_ngraph(attrs);
      int mode = static_cast<int>(ngraph_bridge::GraphExeMode::kTrain);
      return graph->fprop_cache->bprop->get_parameters().size() +
             graph->cached_aux_positions[mode].size();
    })
    .set_num_outputs([](const NodeAttrs &attrs) {
      auto graph = get_ngraph(attrs);
      return graph->fprop_cache->bprop->get_results().size();
    })
    .set_attr<bool>("TIsBackward", true)
    .set_attr<bool>("TIsLayerOpBackward", true)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                                  NgraphSubgraphOpBackward)
    .set_attr<FInferStorageType>("FInferStorageType",
                                 NgraphSubgraphBackwardInferStorageType);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ngraph, SgNgraphProperty);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_NGRAPH
