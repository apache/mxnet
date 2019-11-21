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

#include <tuple>

#include "./fused_op.h"
#include "../operator_common.h"
#include "../../executor/exec_pass.h"

#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

namespace mxnet {

DMLC_REGISTER_PARAMETER(FusedOpConfig);

std::mutex FusedOp::mutex_;

void FusedOpParamParser(nnvm::NodeAttrs* attrs) {
  FusedOpConfig param;
  try {
    param.Init(attrs->dict);
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
  attrs->parsed = FusedOpPtr(new FusedOp(attrs, param));
}

FusedOp::FusedOp(const nnvm::NodeAttrs* attrs, const FusedOpConfig& config) :
    initialized_(false),
    kernel_function_dev_id_(-1) {
  inputs_ = std::vector<FusedOpEntry>(config.num_inputs);
  outputs_ = std::vector<FusedOpEntry>(config.num_outputs);
  subgraph_ = nnvm::Graph();
  subgraph_.outputs = attrs->subgraphs[0]->outputs;
}

bool FusedOp::InferShape(const nnvm::NodeAttrs &attrs,
                         std::vector<mxnet::TShape> *in_attrs,
                         std::vector<mxnet::TShape> *out_attrs) {
  subgraph_.attrs.erase("shape");
  subgraph_.attrs.erase("shape_inputs");
  std::vector<mxnet::TShape> input_shapes(*in_attrs);
  subgraph_ = mxnet::exec::InferShape(std::move(subgraph_),
                                      std::move(input_shapes),
                                      "__shape__");

  const auto& g = subgraph_.indexed_graph();
  const auto& input_nids = g.input_nodes();

  std::vector<mxnet::TShape> out_shapes;
  const std::vector<mxnet::TShape> shapes = subgraph_.GetAttr<mxnet::ShapeVector>("shape");
  for (auto& e : g.outputs()) {
    out_shapes.push_back(shapes[g.entry_id(e)]);
  }
  CHECK_EQ(out_shapes.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::shape_assign(&(out_attrs->at(i)), out_shapes[i]);
  }

  // assign to in_attrs
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    const auto eid = g.entry_id(input_nids[i], 0);
    SHAPE_ASSIGN_CHECK(*in_attrs, i, shapes[eid]);
  }

  bool inferred = true;
  for (const auto& attr : *in_attrs) {
    inferred = inferred && !op::shape_is_none(attr);
  }
  for (const auto& attr : *out_attrs) {
    inferred = inferred && !op::shape_is_none(attr);
  }
  if (inferred) {
    std::lock_guard<std::mutex> lock(my_mutex_);
    intermediate_shapes_.push_back({*in_attrs, *out_attrs, shapes});
  }
  return inferred;
}

bool FusedOp::InferType(const nnvm::NodeAttrs &attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  subgraph_.attrs.erase("dtype");
  subgraph_.attrs.erase("dtype_inputs");
  std::vector<int> input_types(*in_attrs);
  subgraph_ = mxnet::exec::InferType(std::move(subgraph_),
                                     std::move(input_types),
                                     "__dtype__");

  const auto& g = subgraph_.indexed_graph();
  const auto& input_nids = g.input_nodes();

  std::vector<int> out_types;
  const std::vector<int> types = subgraph_.GetAttr<nnvm::DTypeVector>("dtype");
  for (auto& e : g.outputs()) {
    out_types.push_back(types[g.entry_id(e)]);
  }
  CHECK_EQ(out_types.size(), out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); ++i) {
    op::type_assign(&(out_attrs->at(i)), out_types[i]);
  }

  // assign to in_attrs
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    const auto eid = g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_attrs, i, types[eid]);
  }

  bool inferred = true;
  for (const auto& attr : *in_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  for (const auto& attr : *out_attrs) {
    inferred = inferred && !op::type_is_none(attr);
  }
  if (inferred) {
    std::lock_guard<std::mutex> lock(my_mutex_);
    intermediate_dtypes_.push_back({*in_attrs, *out_attrs, types});
  }
  return inferred;
}

template <typename Attr>
std::tuple<const nnvm::NodePtr,
           std::vector<Attr>,
           std::vector<Attr>>
FusedOp::GetAttrs(const std::string& attr_name, const uint32_t node_id) {
  const auto& g = subgraph_.indexed_graph();
  const std::vector<Attr> attrs = subgraph_.GetAttr<std::vector<Attr>>(attr_name);
  const auto& node = g[node_id];
  std::vector<Attr> inputs, outputs;
  for (const auto& e : node.inputs) {
    inputs.emplace_back(attrs[g.entry_id(e)]);
  }
  outputs.resize(node.source->num_outputs());
  for (size_t i = 0; i < g.num_nodes(); ++i) {
    if (i == node_id) continue;
    const auto& other_node = g[i];
    for (const auto& e : other_node.inputs) {
      if (e.node_id == node_id) {
        outputs[e.index] = attrs[g.entry_id(e)];
      }
    }
  }
  for (const auto& e : g.outputs()) {
    if (e.node_id == node_id) {
      outputs[e.index] = attrs[g.entry_id(e)];
    }
  }

  return std::make_tuple(node.weak_ref.lock(),
                         inputs,
                         outputs);
}

bool FusedOpInferShape(const nnvm::NodeAttrs& attrs,
                       std::vector<mxnet::TShape> *in_attrs,
                       std::vector<mxnet::TShape> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferShape(attrs, in_attrs, out_attrs);
}

bool FusedOpInferType(const nnvm::NodeAttrs& attrs,
                      std::vector<int> *in_attrs,
                      std::vector<int> *out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  return op->InferType(attrs, in_attrs, out_attrs);
}

void FusedOpProvideShape(const nnvm::NodeAttrs& attrs,
                         const std::vector<nnvm::NodePtr>& nodes,
                         const std::vector<std::vector<mxnet::TShape>> &in_attrs,
                         const std::vector<std::vector<mxnet::TShape>> &out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->ProvideShape(nodes, in_attrs, out_attrs);
}

void FusedOpProvideType(const nnvm::NodeAttrs& attrs,
                        const std::vector<nnvm::NodePtr>& nodes,
                        const std::vector<std::vector<int>> &in_attrs,
                        const std::vector<std::vector<int>> &out_attrs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->ProvideType(nodes, in_attrs, out_attrs);
}

void FusedOpProvideStorageType(const nnvm::NodeAttrs& attrs,
                               const std::vector<nnvm::NodePtr>& nodes,
                               const std::vector<std::vector<int>> &in_attrs,
                               const std::vector<std::vector<int>> &out_attrs) {}

NNVM_REGISTER_OP(_FusedOp)
.set_attr<exec::TIsFusion>("TIsFusion", true)
.set_num_inputs([](const NodeAttrs& attrs) {
    const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
    const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
    const auto num_inputs = op->num_inputs();
    const auto num_outputs = op->num_outputs();
    std::vector<std::pair<int, int> > ret;
    for (unsigned int i = 0; i < num_inputs; ++i) {
      for (unsigned int j = 0; j < num_outputs; ++j) {
        ret.emplace_back(i, j);
      }
    }
    return ret;
    })
.set_attr<exec::FProvideSubgraphShape>("FProvideSubgraphShape", FusedOpProvideShape)
.set_attr<exec::FProvideSubgraphType>("FProvideSubgraphType", FusedOpProvideType)
.set_attr<exec::FProvideSubgraphStorageType>("FProvideSubgraphStorageType",
                                             FusedOpProvideStorageType)
.set_attr<mxnet::FInferShape>("FInferShape", FusedOpInferShape)
.set_attr<nnvm::FInferType>("FInferType", FusedOpInferType)
.set_attr_parser(FusedOpParamParser)
.add_argument("data", "NDArray-or-Symbol[]", "Data");

std::tuple<const nnvm::NodePtr,
           std::vector<mxnet::TShape>,
           std::vector<mxnet::TShape>>
FusedOpHelperShape(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAttrs<mxnet::TShape>("shape", node_id);
}

std::tuple<const nnvm::NodePtr,
           std::vector<int>,
           std::vector<int>>
FusedOpHelperType(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAttrs<int>("dtype", node_id);
}

NNVM_REGISTER_OP(_FusedOpHelper)
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<nnvm::TIsGhost>("TIsGhost", true)
.set_attr<exec::TIsFusionHelper>("TIsFusionHelper", true)
.set_attr<exec::FAccessSubgraphShape>("FAccessSubgraphShape", FusedOpHelperShape)
.set_attr<exec::FAccessSubgraphType>("FAccessSubgraphType", FusedOpHelperType);


std::tuple<const nnvm::NodePtr,
           std::vector<mxnet::TShape>,
           std::vector<mxnet::TShape>>
FusedOpOutHelperShape(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAuxShape(node_id);
}

std::tuple<const nnvm::NodePtr,
           std::vector<int>,
           std::vector<int>>
FusedOpOutHelperType(const NodeAttrs& attrs) {
  const auto& p = nnvm::get<FusedOpHelperParamPtr>(attrs.parsed);
  const auto& op = p->op;
  const auto& node_id = p->node_id;
  return op->GetAuxType(node_id);
}

NNVM_REGISTER_OP(_FusedOpOutHelper)
.set_num_inputs(0)
.set_num_outputs(0)
.set_attr<nnvm::TIsGhost>("TIsGhost", true)
.set_attr<exec::TIsFusionHelper>("TIsFusionHelper", true)
.set_attr<exec::FAccessSubgraphShape>("FAccessSubgraphShape", FusedOpOutHelperShape)
.set_attr<exec::FAccessSubgraphType>("FAccessSubgraphType", FusedOpOutHelperType);

}  // namespace mxnet

#endif  // MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
