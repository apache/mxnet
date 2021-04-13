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

#ifndef MXNET_IMPERATIVE_CACHED_OP_H_
#define MXNET_IMPERATIVE_CACHED_OP_H_

#include <mxnet/imperative.h>
#include <vector>
#include <numeric>
#include <atomic>
#include <utility>
#include <string>
#include <unordered_map>
#include <map>
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"
#include "./imperative_utils.h"
#include "../nnvm/error.h"

namespace mxnet {
namespace {

  static const char FULL[] = "full";
  static const char FORWARD[] = "forward";
  static const char BACKWARD[] = "backward";
  static const char REF_COUNT[] = "ref_count";
  static const char MEM_PLAN[] = "mem_plan";
  static const char STORAGE_PLAN[] = "storage_plan";

std::string AddPrefix(const std::string& prefix,
                      const std::string& s) {
  return prefix + "_" + s;
}

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  static const Op* zeros_like_op = Op::Get("zeros_like");

  if (v.empty()) {
    nnvm::ObjectPtr ng = nnvm::Node::Create();
    ng->attrs.op = Op::Get("_zeros_without_dtype");
    ng->attrs.name = "zeros_without_dtype";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry(std::move(ng), 0, 0);
  }

  // remove zero in the sum. at least keep 1.
  auto begin = std::remove_if(v.begin(), v.end(), [](const nnvm::NodeEntry& nodeEntry) {
     CHECK(nodeEntry.node);
     return nodeEntry.node->op() == zeros_op || nodeEntry.node->op() == zeros_like_op;
  });
  if (begin == v.begin()) ++begin;
  v.erase(begin, v.end());
  CHECK(!v.empty());

  if (v.size() == 1) {
    return std::move(v[0]);
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::ObjectPtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry(std::move(sum_node), 0, 0);
    } else {
      // use a stream line of plus instead
      nnvm::NodeEntry ret = v[0];
      for (size_t i = 1; i < v.size(); ++i) {
        // Add control flow dependency from to previous node
        // This enforces the gradient sum order will be in the inverse
        // order of forward traversal
        // NOTE: adding control dependency can be dangerous and cause cycle in the dep.
        // The curent usage is correct, because of the following invariant:
        // assert: v[i-1] do not depend on v[i]
        // To put in plain text: v is gradient vector that get pushed in the order
        // that can generate them, which means if v[i] is not yet pushed,
        // all previous gradient cannot depend on it.
        // Note: For a symbol like the following:
        // data = mx.sym.Variable('data')
        // sym = data + data + data + data + data + data + data
        // the node entries v passed in here are of the same node of
        // op _identity_with_attr_like_rhs. We should skip adding a node
        // to its own control_deps.
        if (v[i-1].node != v[i].node) {
          v[i].node->control_deps.push_back(ret.node);
        }

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::ObjectPtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry(std::move(x), 0, 0);
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::ObjectPtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}


/* \brief collect pointers to input and output ndarrays
 * into a single data structure, this data structure can
 * be used for Memory allocation pass*/

void CollectInputOutputNDRefs(const nnvm::Graph& g,
                              const std::vector<NDArray*>& inputs,
                              const std::vector<size_t>& input_map,
                              const std::vector<NDArray*>& outputs,
                              std::vector<NDArray*>* arrays) DMLC_ATTRIBUTE_UNUSED;
void CollectInputOutputNDRefs(const nnvm::Graph& g,
                              const std::vector<NDArray*>& inputs,
                              const std::vector<size_t>& input_map,
                              const std::vector<NDArray*>& outputs,
                              std::vector<NDArray*>* arrays) {
  const auto& idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();
  for (size_t i = 0; i < num_inputs; ++i) {
    (*arrays)[idx.entry_id(idx.input_nodes()[i], 0)] = inputs[input_map[i]];
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    if (!(*arrays)[eid]->is_none())
      *outputs[i] = (*arrays)[eid]->Detach();
    (*arrays)[eid] = outputs[i];
  }
}

/* \brief create ndarrays for the intermediate outputs and final outputs
 * from the allocated storage (happens in MXPlanMemory NNVM pass)*/
void CreateGraphNDs(const nnvm::Graph& g,
                    const mxnet::Context& default_ctx,
                    const mxnet::imperative::MemoryPlanVector& mem_plan,
                    std::vector<OpReqType>* array_reqs,
                    std::vector<NDArray*>* arrays) DMLC_ATTRIBUTE_UNUSED;
void CreateGraphNDs(const nnvm::Graph& g,
                    const mxnet::Context& default_ctx,
                    const mxnet::imperative::MemoryPlanVector& mem_plan,
                    std::vector<OpReqType>* array_reqs,
                    std::vector<NDArray*>* arrays) {
  const auto& idx = g.indexed_graph();
  mxnet::imperative::AllocateMemory(g, idx, default_ctx, 0,
                                    idx.num_node_entries(), mem_plan, *arrays,
                                    array_reqs);
  const auto &dtypes = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto &shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto &stypes = g.GetAttr<mxnet::StorageTypeVector>("storage_type");
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    if (!(*arrays)[eid]->is_none())
      continue;
    *((*arrays)[eid]) = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                                shapes[eid], default_ctx, true, dtypes[eid]);
    const nnvm::NodeAttrs& attrs = idx[idx.outputs()[i].node_id].source->attrs;
    (*arrays)[eid]->AssignStorageInfo(
        common::NodeAttrsGetProfilerScope(attrs),
        attrs.name);
  }
}

/* \brief create a forward graph from they Symbol */
void CreateForwardGraph(const nnvm::Symbol &sym, nnvm::Graph *fwd_graph) {
  using namespace nnvm;
  static const auto _copy_op = Op::Get("_copy");
  NodeEntryMap<size_t> dedup_out;
  // Iterate through all node entries, emplace node entry outputs of symbol
  // to graph outputs. Since node entry stores information about the node
  // as well as the input node of the graph, a graph can be recreated from a
  // symbol by just copying the outputs
  for (const NodeEntry &nodeEntry : sym.outputs) {
    if (dedup_out.find(nodeEntry) != dedup_out.end()) {
      ObjectPtr copy_node = Node::Create();
      copy_node->attrs.op = _copy_op;
      copy_node->attrs.name = nodeEntry.node->attrs.name + "_copy" +
                              std::to_string(dedup_out[nodeEntry]++);
      copy_node->inputs.emplace_back(nodeEntry);
      if (_copy_op->attr_parser != nullptr) {
        _copy_op->attr_parser(&(copy_node->attrs));
      }
      fwd_graph->outputs.emplace_back(std::move(copy_node));
    } else {
      dedup_out.emplace(nodeEntry, 0);
      fwd_graph->outputs.push_back(nodeEntry);
    }
  }
}

/* \brief construct grad_graph from fwd_graph and ograd_entries*/
void CreateBackwardGraph(nnvm::Graph* fwd_graph,
                         nnvm::Graph* grad_graph,
                         std::vector<nnvm::NodeEntry>* ograd_entries,
                         std::unordered_map<uint32_t, uint32_t>* fwd_input_to_grad_output) {
  using namespace nnvm;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  ograd_entries->reserve(fwd_graph->outputs.size());
  for (size_t i = 0; i < fwd_graph->outputs.size(); ++i) {
    nnvm::ObjectPtr np = Node::Create();
    const nnvm::NodeAttrs& attrs = fwd_graph->outputs[i].node->attrs;
    np->attrs.name = attrs.name + "_head_grad";
    np->attrs.dict["__profiler_scope__"] = common::NodeAttrsGetProfilerScope(attrs);
    ograd_entries->emplace_back(np);
  }

  std::vector<NodeEntry> xs;
  const IndexedGraph &indexed_graph = fwd_graph->indexed_graph();
  // Create vector of inputs to be passed to the gradient pass
  for (size_t i = 0; i < indexed_graph.input_nodes().size(); ++i) {
    const uint32_t node_id = indexed_graph.input_nodes()[i];
    // skip the mutable nodes, which store the auxiliary states,
    // since we don't need to compute gradient w.r.t auxiliary states
    if (indexed_graph.mutable_input_nodes().count(node_id))
      continue;
    // Hold a mapping of the node id to its igrad position
    // Need this mapping in StaticBackward, to obtain the igrad node,
    // corresponding to a fwd_graph node.
    (*fwd_input_to_grad_output)[i] = xs.size();
    xs.emplace_back(indexed_graph[node_id].weak_ref.lock());
  }

  // There are inputs in computation graph that require gradients
  if (!xs.empty()) {
    try {
      *grad_graph = pass::MXGradient(
           *fwd_graph, fwd_graph->outputs, xs, *ograd_entries,
           mxnet::AggregateGradient, nullptr,
           zero_ops, "_copy");
    } catch (const nnvm::pass::InvalidGraphError &e) {
      *grad_graph = nnvm::Graph();
    }
  } else {
    *grad_graph = nnvm::Graph();
  }
}

/* \brief construct fwd_graph, grad_graph and full_graph from symbol */
void CreateFullGraph(const nnvm::Symbol& sym,
                     nnvm::Graph* fwd_graph,
                     nnvm::Graph* grad_graph,
                     nnvm::Graph* full_graph,
                     std::vector<nnvm::NodeEntry>* ograd_entries,
                     std::unordered_map<uint32_t, uint32_t>* fwd_input_to_grad_output) {
  using namespace nnvm;
  CreateForwardGraph(sym, fwd_graph);

  bool do_elim_common_expr = dmlc::GetEnv("MXNET_ELIMINATE_COMMON_EXPR", true);
  if (do_elim_common_expr)
    *fwd_graph = exec::EliminateCommonExpr(std::move(*fwd_graph));

  // construct backward graph
  CreateBackwardGraph(fwd_graph, grad_graph, ograd_entries,
                      fwd_input_to_grad_output);

  full_graph->outputs = fwd_graph->outputs;
  // add backward graph outputs to full graph
  for (const auto &i : grad_graph->outputs) {
    full_graph->outputs.emplace_back(i);
  }
}

/* \brief Set Ref counts for node entries for forward graph */
void SetForwardRefCounts(nnvm::Graph *fwd_graph) {
  const auto& idx = fwd_graph->indexed_graph();

  std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
  for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
  for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
  }

  fwd_graph->attrs[AddPrefix(FORWARD, REF_COUNT)] =
      std::make_shared<dmlc::any>(std::move(ref_count));
}

/* \brief Set Ref counts for node entries for forward graph and full graph */
void SetRefCounts(nnvm::Graph* fwd_graph, const nnvm::Graph& full_graph) {
  const auto& idx = fwd_graph->indexed_graph();
  SetForwardRefCounts(fwd_graph);

  size_t num_forward_nodes = idx.num_nodes();
  size_t num_forward_entries = idx.num_node_entries();

  const auto& full_idx = full_graph.indexed_graph();

  std::vector<uint32_t> temp_ref_count(full_idx.num_node_entries(), 0);
  for (size_t i = num_forward_nodes; i < full_idx.num_nodes(); ++i) {
    for (const auto& j : full_idx[i].inputs) {
       ++temp_ref_count[full_idx.entry_id(j)];
    }
  }

  auto full_ref_count = fwd_graph->GetAttr<std::vector<uint32_t> >(AddPrefix(FORWARD,
                                                                             REF_COUNT));
  for (size_t i = 0; i < num_forward_entries; ++i) full_ref_count.at(i) += temp_ref_count[i];
  fwd_graph->attrs[AddPrefix(FULL, REF_COUNT)] =
      std::make_shared<dmlc::any>(std::move(full_ref_count));
}

void OptimizeGraph(nnvm::Graph* full_graph, nnvm::Graph* fwd_graph, nnvm::Graph* grad_graph,
                   std::vector<size_t>* input_map, const Context& context,
                   size_t num_forward_outputs, const bool inlining) {
  input_map->resize(full_graph->indexed_graph().input_nodes().size());
  std::iota(input_map->begin(), input_map->end(), 0);
#if MXNET_USE_CUDA && !defined(_WIN32)
  if (context.dev_mask() == kGPU &&
      !inlining &&
      dmlc::GetEnv("MXNET_USE_FUSION", true)) {
    nnvm::Graph unoptimized_graph;
    common::CopyGraph(&unoptimized_graph, *full_graph, false);

    if (common::CheckForInputNameDuplicates(unoptimized_graph.indexed_graph())) {
      *full_graph = exec::FusePointwise(*full_graph, num_forward_outputs);
      // Fill in input_map - mapping from the new to the original input indices.
      const auto &original_inputs = unoptimized_graph.indexed_graph().input_nodes();
      const auto &new_inputs = full_graph->indexed_graph().input_nodes();
      if (original_inputs.size() != new_inputs.size()) {
        LOG(WARNING)
          << "Number of inputs after fusion does not match original number of inputs. "
          << "This is most probably a bug. Disabling fusion for this run.";
        *full_graph = unoptimized_graph;
      } else {
        std::unordered_map<std::string, size_t> original_input_map;
        for (size_t i = 0; i < original_inputs.size(); ++i) {
          auto r = original_input_map.insert(std::make_pair(
              unoptimized_graph.indexed_graph()[original_inputs[i]].source->attrs.name, i));
          CHECK(r.second);
        }
        for (size_t i = 0; i < new_inputs.size(); ++i) {
          auto it = original_input_map.find(
              full_graph->indexed_graph()[new_inputs[i]].source->attrs.name);
          CHECK(it != original_input_map.end());
          (*input_map)[i] = it->second;
        }
      }
    } else {
      LOG(WARNING)
        << "Graph contains duplicate names for some of its inputs - fusion is NOT enabled!";
     }
  }
#else
  // Only warn user if MXNET_USE_FUSION env var is explicitly set
  if (context.dev_mask() == kGPU && !inlining &&
      dmlc::GetEnv("MXNET_USE_FUSION", false)) {
    exec::WarnFusionNotSupported();
  }
#endif  // MXNET_USE_CUDA && !defined(_WIN32)

  *fwd_graph = nnvm::Graph();
  fwd_graph->outputs = std::vector<nnvm::NodeEntry>(full_graph->outputs.begin(),
                                                    full_graph->outputs.begin() +
                                                    num_forward_outputs);
  *grad_graph = nnvm::Graph();
  grad_graph->outputs = std::vector<nnvm::NodeEntry>(full_graph->outputs.begin() +
                                                     num_forward_outputs,
                                                     full_graph->outputs.end());
  SetRefCounts(fwd_graph, *full_graph);
}

/* \brief Check if param indices and data indices are set, if not then set data indices */
void SetInputIndices(const nnvm::Graph& fwd_graph,
                     const mxnet::Tuple<uint32_t>& param_indices,
                     mxnet::Tuple<uint32_t>* data_indices) DMLC_ATTRIBUTE_UNUSED;
void SetInputIndices(const nnvm::Graph& fwd_graph,
                     const mxnet::Tuple<uint32_t>& param_indices,
                     mxnet::Tuple<uint32_t>* data_indices) {
  const auto& indexed_graph = fwd_graph.indexed_graph();
  if (data_indices->ndim() || param_indices.ndim()) {
    CHECK_EQ(data_indices->ndim() + param_indices.ndim(),
             static_cast<const int>(indexed_graph.input_nodes().size()));
  } else {
    std::vector<uint32_t> tmp;
    tmp.reserve(indexed_graph.input_nodes().size());
    for (size_t i = 0; i < indexed_graph.input_nodes().size(); ++i) {
      tmp.emplace_back(i);
    }
    data_indices->assign(tmp.begin(), tmp.end());
  }
}

}  // namespace

/*! \brief CachedOp Parameters */
struct CachedOpConfig : public dmlc::Parameter<CachedOpConfig> {
  uint32_t inline_limit;
  uint32_t forward_bulk_size;
  uint32_t backward_bulk_size;
  bool static_alloc;
  bool static_shape;
  bool is_dynamic;
  mxnet::Tuple<uint32_t> data_indices;
  mxnet::Tuple<uint32_t> param_indices;
  std::string subgraph;
  DMLC_DECLARE_PARAMETER(CachedOpConfig) {
    DMLC_DECLARE_FIELD(static_alloc)
    .set_default(false)
    .describe("Statically allocate memory to improve speed. "
              "Memory usage may increase.");
    DMLC_DECLARE_FIELD(static_shape)
    .set_default(false)
    .describe("Optimize for invariant input shapes between iterations. "
              "Must also set static_alloc to True. "
              "Change of input shapes is still allowed but slower.");
    DMLC_DECLARE_FIELD(inline_limit)
    .set_default(2)
    .describe("Maximum number of operators that can be inlined.");
    DMLC_DECLARE_FIELD(forward_bulk_size)
    .set_default(Imperative::BulkExecMaxNodeTrainFwd())
    .describe("Segment size of bulk execution during forward pass.");
    DMLC_DECLARE_FIELD(backward_bulk_size)
    .set_default(Imperative::BulkExecMaxNodeTrainBwd())
    .describe("Segment size of bulk execution during backward pass.");
    DMLC_DECLARE_FIELD(data_indices)
    .set_default(mxnet::Tuple<uint32_t>())
    .describe("Position of argument variables.");
    DMLC_DECLARE_FIELD(param_indices)
    .set_default(mxnet::Tuple<uint32_t>())
    .describe("Position of parameters.");
    DMLC_DECLARE_FIELD(subgraph)
    .set_default(std::string(""))
    .describe("JSON string of a subgraph.");
    DMLC_DECLARE_FIELD(is_dynamic)
    .set_default(false)
    .describe("Whether the graph contains dynamic shape operators.");
  }
};

namespace io {
class LazyTransformDataset;
}

class CachedOp {
  using CachedOpMonCallback =
      std::function<void(const char *, const char *, void *)>;

 public:
  CachedOp(
      const nnvm::Symbol& sym,
      const std::vector<std::pair<std::string, std::string> >& flags);
  virtual ~CachedOp();
  nnvm::Symbol GetOptimizedSymbol() const;
  uint32_t num_inputs() const {
    return fwd_graph_.indexed_graph().input_nodes().size();
  }
  uint32_t num_outputs() const {
    return fwd_graph_.outputs.size();
  }
  uint32_t num_backward_inputs() const {
    return bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size();
  }
  uint32_t num_backward_outputs() const {
    auto &idx = fwd_graph_.indexed_graph();
    return idx.input_nodes().size() - idx.mutable_input_nodes().size();
  }
  std::vector<bool>& save_inputs() {
    return save_inputs_;
  }
  std::vector<bool>& save_outputs() {
    return save_outputs_;
  }
  const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return fwd_graph_.indexed_graph().mutable_input_nodes();
  }
  virtual std::vector<nnvm::NodeEntry> Gradient(
      const nnvm::ObjectPtr& node,
      const std::vector<nnvm::NodeEntry>& ograds) const;
  virtual OpStatePtr Forward(
      const std::shared_ptr<CachedOp>& op_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs,
      const Context &default_context);
  virtual void Backward(
      const bool retain_graph,
      const OpStatePtr& state,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);
  // backward storage type inference
  virtual bool BackwardStorageType(
      const nnvm::NodeAttrs& attrs,
      const int dev_mask,
      DispatchMode* dispatch_mode,
      std::vector<int> *in_attrs,
      std::vector<int> *out_attrs);
  std::vector<std::string> ListForwardInputNames() const {
    nnvm::Symbol sym = GetForwardSym();
    return sym.ListInputNames(nnvm::Symbol::kAll);
  }
  std::vector<std::string> ListForwardOutputNames() const {
    nnvm::Symbol sym = GetForwardSym();
    return sym.ListOutputNames();
  }
  nnvm::Symbol GetForwardSym() const {
    nnvm::Symbol sym;
    sym.outputs = fwd_graph_.outputs;
    return sym;
  }
  void RegisterOpHook(const CachedOp::CachedOpMonCallback& callback,
                      bool monitor_all = false);

 protected:
  struct GraphInfo {
    nnvm::Graph fwd_graph;
    nnvm::Graph grad_graph;
    nnvm::Graph full_graph;
    std::vector<size_t> input_map;  // the original index of an input
    std::vector<nnvm::NodeEntry> ograd_entries;
    std::unordered_map<uint32_t, uint32_t> fwd_input_to_grad_output;
    std::vector<OpReqType> bwd_output_reqs;
    std::vector<uint32_t> bwd_input_eid;
  };

  struct CachedOpState {
    CachedOpState(const Context &context_, const nnvm::Graph &fwd_graph_,
                  const nnvm::Graph &full_graph_, const bool inlining_) {
      context = context_;
      nnvm::Symbol sym;
      sym.outputs = fwd_graph_.outputs;
      CreateFullGraph(sym.Copy(), &info.fwd_graph, &info.grad_graph,
                      &info.full_graph, &info.ograd_entries,
                      &info.fwd_input_to_grad_output);

      OptimizeGraph(&info.full_graph, &info.fwd_graph, &info.grad_graph, &info.input_map,
                    context_, fwd_graph_.outputs.size(), inlining_);

      size_t max_nodes = info.full_graph.indexed_graph().num_nodes();
      size_t max_entries = info.full_graph.indexed_graph().num_node_entries();
      info.fwd_graph.attrs["context"] =
          std::make_shared<dmlc::any>(std::vector<Context>(
              info.fwd_graph.indexed_graph().num_nodes(), context));
      info.full_graph.attrs["context"] =
          std::make_shared<dmlc::any>(std::vector<Context>(max_nodes, context));

      buff.resize(max_entries);
      arrays.resize(max_entries);
      array_reqs.resize(max_entries);
      dynamic_entries.resize(max_entries, false);
      op_states.resize(max_nodes);
      execs.resize(max_nodes);
      opr_segs.resize(max_nodes);
    }

    std::mutex mutex;
    Context context;
    GraphInfo info;

    bool recording = false;
    bool fwd_alloc = false;
    bool bwd_alloc = false;
    bool fwd_exec_init = false;
    bool bwd_exec_init = false;

    std::vector<NDArray> buff;
    std::vector<NDArray *> arrays;
    std::vector<NDArray *> arrays_with_in_out;
    std::vector<OpReqType> array_reqs;

    std::vector<OpStatePtr> op_states;
    std::vector<std::shared_ptr<exec::OpExecutor>> execs;
    std::vector<imperative::EngineOprSeg> opr_segs;

    std::vector<bool> dynamic_entries;
    std::multimap<size_t, NDArray> fwd_reuse_pool;
    std::multimap<size_t, NDArray> bwd_reuse_pool;
  };

  OpStatePtr GetCachedOpState(const Context& ctx);
  bool SetForwardGraph(
      const Context& default_ctx,
      GraphInfo* info,
      const bool recording,
      const std::vector<NDArray*>& inputs);
  bool SetBackwardGraph(
      GraphInfo* info,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& inputs,
      bool detect_inplace_addto = false);
  bool CheckDynamicShapeExists(
      const Context& default_ctx,
      const std::vector<NDArray*>& inputs,
      bool erase_result);
  void StaticAllocMemory(
      const OpStatePtr& state_ptr,
      bool recording,
      bool keep_fwd);
  void StaticInitExec(
      const OpStatePtr& state_ptr,
      bool recording,
      bool keep_fwd);
  void StaticRunOps(
      const Context& default_ctx,
      const nnvm::Graph& g,
      const OpStatePtr& state_ptr,
      const std::vector<NDArray *> &state_arrays,
      size_t start_nid,
      size_t end_nid);
  OpStatePtr StaticForward(
      const Context& default_ctx,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs);
  struct DynamicRuntime;

 private:
  OpStatePtr DynamicForward(
      const Context& default_ctx,
      const std::vector<NDArray*>& inputs,
      const std::vector<NDArray*>& outputs,
      bool use_naive_run = false);
  void DynamicBackward(
      const bool retain_graph,
      const OpStatePtr& op_state,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);
  void StaticBackward(
      const bool retain_graph,
      const OpStatePtr& state_ptr,
      const std::vector<NDArray*>& inputs,
      const std::vector<OpReqType>& reqs,
      const std::vector<NDArray*>& outputs);
  size_t BwdOriginalInput(const std::vector<size_t>& input_map, size_t new_i);

  CachedOpConfig config_;
  nnvm::Graph fwd_graph_;
  nnvm::Graph full_graph_;
  bool inlining_;
  bool dynamic_shape_checked_;
  std::vector<nnvm::NodeEntry> ograd_entries_;
  std::vector<uint32_t> bwd_in_dep_, bwd_out_dep_, bwd_ograd_dep_;
  std::vector<bool> save_inputs_, save_outputs_;
  std::vector<OpReqType> bwd_output_reqs_;

  std::function<void(const char*, const char*, NDArrayHandle)> monitor_callback_{nullptr};
  bool monitor_all_{false};

  std::mutex mutex_;
  std::unordered_map<Context, std::vector<OpStatePtr> > cached_op_states_;

  friend class ::mxnet::io::LazyTransformDataset;
  nnvm::Symbol sym_;
  std::vector<std::pair<std::string, std::string> > flags_;
};

struct CachedOp::DynamicRuntime {
  GraphInfo info;
  std::vector<NDArray> buff;
  std::vector<OpStatePtr> op_states;
};

using CachedOpPtr = std::shared_ptr<CachedOp>;

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_CACHED_OP_H_
