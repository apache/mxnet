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
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <mxnet/base.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "error.h"
#include "../imperative/exec_pass.h"

namespace nnvm {
namespace pass {

extern size_t MXGetDTypeSize(const int type_flag);  // defined in plan_memory.cc

namespace {

/*! Auxiliary Data Structure for Gradient Entries */
struct GradEntry {
  NodeEntry sum = NodeEntry(nullptr, 0, 0);
  std::vector<NodeEntry> grads;
};


/*!
 * \brief Build the backward graph from the mirror map. This function will be
 *        invoked twice if backward mirroring has been enabled.
 */
Graph BuildGradientGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<ObjectPtr>& topo_order,
    std::unordered_map<const Node*, std::vector<GradEntry> > output_grads,
    std::function<int(const Node&)> mirror_fun,
    const std::unordered_map<const Node*, ObjectPtr>& mirror_map);

/*!
 * \brief Auxiliary function that maps the forward node of the source graph to
 *        its corresponding node on the mirror path.
 */
inline const ObjectPtr& MapFwdNodeToMirrorPath(
    const ObjectPtr& n,
    const std::unordered_map<const Node*, ObjectPtr>& mirror_map) {
  auto iter = mirror_map.find(n.get());
  if (iter == mirror_map.end() ||
      iter->second == nullptr) {
    return n;
  }
  return iter->second;
}


Graph Gradient(Graph src) {
  CHECK_NE(src.attrs.count("grad_ys"), 0U)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0U)
      << "Gradient require grad_xs to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0U)
      << "Gradient require grad_ys_out_grad to be presented.";
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  CHECK_EQ(ys.size(), ys_out_grad.size());

  // initialize a topological order of the graph nodes and `output_grads`
  // that maps every operator node to its gradient entries
  std::vector<ObjectPtr> topo_order;
  std::unordered_map<const Node*, std::vector<GradEntry> > output_grads;

  DFSVisit(ys,
           [&](const ObjectPtr& node) {
             if (output_grads.count(node.get()) == 0) {
               output_grads[node.get()].resize(node->num_outputs());
             }
             topo_order.push_back(node);
           });

  for (size_t i = 0; i < ys.size(); ++i) {
    output_grads[ys[i].node.get()][ys[i].index].grads = {ys_out_grad[i]};
  }

  // check that all xs are reachable from ys
  for (size_t i = 0; i < xs.size(); ++i) {
    CHECK(output_grads.find(xs[i].node.get()) != output_grads.end())
        << "Cannot differentiate with respect to the "
        << (i + 1) << "-th variable "
        << "because it is unreachable from the outputs.";
  }

  using MirrorFun = std::function<int (const Node&)>;
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("mirror_fun");
  }
  std::unordered_map<const Node*, ObjectPtr> mirror_map;

  // complete the backward graph of the src, but without backward mirroring
  nnvm::Graph gsrc = BuildGradientGraph(src, xs, topo_order,
                                        output_grads,
                                        nullptr, mirror_map);
  if (mirror_fun == nullptr) {
    return gsrc;  // Gradient pass without mirroring ends here.
  }
  const IndexedGraph& idx = src.indexed_graph(),
                    & gidx = gsrc.indexed_graph();
  // ===========================================================================
  // ----- Gradient Pass w/ Backward Mirroring -----
  // ===========================================================================
  // Record, for each node entry ∈ gsrc, the nodes that reference it as inputs.
  // It is important to note that since the node entry reference mapping is
  // constructed from gradient graph, it can only be indexed using gidx entry ID.
  std::vector<std::unordered_set<const Node*> > node_entry_ref_map(
      gidx.num_node_entries());
  static const auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  for (uint32_t gnid = 0; gnid < gidx.num_nodes(); ++gnid) {
    const IndexedGraph::Node& inode = gidx[gnid];
    if (inode.source->is_variable()) {
      continue;
    }
    for (uint32_t i = 0; i < inode.inputs.size(); ++i) {
      if (fignore_inputs.count(inode.source->op()) != 0) {
        std::vector<uint32_t> ignore_inputs =
            fignore_inputs[inode.source->op()](inode.source->attrs);
        if (std::find(ignore_inputs.begin(), ignore_inputs.end(), i)
            != ignore_inputs.end()) {
          continue;
        }
      }
      node_entry_ref_map[gidx.entry_id(inode.inputs[i])].insert(inode.source);
    }
  }  // for (gnid ∈ gidx.num_nodes())
  // Inference the shapes and data types of the gradient graphs. Those
  // information is needed in later stages to determine whether putting a node
  // on the mirror path can be beneficial or not.
  using mxnet::ShapeVector;
  ShapeVector in_arg_shapes = src.GetAttr<ShapeVector>("in_arg_shapes");
  DTypeVector in_arg_dtypes = src.GetAttr<DTypeVector>("in_arg_dtypes");
  src = mxnet::exec::InferShape(std::move(src), std::move(in_arg_shapes), "__shape__");
  src = mxnet::exec::InferType(std::move(src), std::move(in_arg_dtypes), "__dtype__");
  CHECK(src.GetAttr<size_t>("shape_num_unknown_nodes") == 0U);
  CHECK(src.GetAttr<size_t>("dtype_num_unknown_nodes") == 0U);
  const ShapeVector& src_shapes = src.GetAttr<ShapeVector>("shape");
  const DTypeVector& src_dtypes = src.GetAttr<DTypeVector>("dtype");

  std::queue<const Node*> worklist;
  // initialize the worklist to the output nodes
  for (const NodeEntry& e : src.outputs) {
    worklist.push(e.node.get());
  }
  for (; !worklist.empty(); worklist.pop()) {
    const Node* const workitem = worklist.front();
    // skip the current node if it has already been recorded in the mirror map
    if (mirror_map.find(workitem) != mirror_map.end()) {
      continue;
    }

    // subgraph and its frontier and topological-sorted view
    std::unordered_set<const Node*> subgraph;
    // The associated boolean variable is used for marking forward propagation.
    std::unordered_map<const Node*, bool> subgraph_frontier;
    std::deque<const Node*> subgraph_topo_order;
    // =========================================================================
    // --- Backward Pass ---
    // =========================================================================
    // The sub-worklist is used for constructing the subgraph. It is initialized
    // to have the current workitem node.
    std::queue<const Node*> subworklist;
    subworklist.push(workitem);
    // Local auxiliary function that does backpropagation on the subworklist
    // items to construct the subgraph. E.g.,
    // A subworklist = {A}
    // ↑
    // B
    // After invoking this function. `subgraph` will become {A, B}.
    // Note that this function will be invoked multiple times.
    auto subworklist_backprop = [&subworklist, &subgraph,
                                 &subgraph_topo_order,
                                 &mirror_fun, &worklist]() {
          std::deque<const Node*> subworklist_topo_order;
          for (; !subworklist.empty(); subworklist.pop()) {
            const Node* const subworkitem = subworklist.front();
            if (subgraph.find(subworkitem) == subgraph.end()) {
              subgraph.insert(subworkitem);
              subworklist_topo_order.push_front(subworkitem);
            }
            for (const NodeEntry& e : subworkitem->inputs) {
              if (!mirror_fun(*(e.node))) {
                worklist.push(e.node.get());
              } else {
                subworklist.push(e.node.get());
              }
            }
            for (const ObjectPtr& n : subworkitem->control_deps) {
              if (!mirror_fun(*n)) {
                worklist.push(n.get());
              } else {
                subworklist.push(n.get());
              }
            }
          }  // for (subworkitem ∈ subworklist)
          // please refer to later comments for why the topological order of the
          // subworklist should be directly appended to that of the subgraph
          subgraph_topo_order.insert(subgraph_topo_order.end(),
                                     subworklist_topo_order.begin(),
                                     subworklist_topo_order.end());
        };
    // Start propagating from the current workitem node backward until the
    // mirroring function returns false (indicating that a compute-heavy layer
    // has been hit), in which case we put the node that fails the mirroring
    // function into the worklist as the new head. During the traversal, we
    // build up the subgraph and its topological order at the same time.
    subworklist_backprop();

    // Forward propagate the subgraph nodes in topological order and make sure
    // that all the node entries that are part of the forward propagation belong
    // to the same subgraph. This process continues until all the node entries
    // have been included, in which case we say that the subgraph has converged.
    //
    // The reason why this step is needed is because, consider the example below:
    // A  B  C  subworklist = {A}
    // ↑  ↑  ↑
    // ↖  ↑  ↗
    //    D
    // Without loss of generality, suppose that the previous backpropagation
    // starts from node A, then the subgraph will only contain branch D → A.
    // However, we want to include branch D → B adn D → C as well since all
    // three branches share the same node entries (i.e., the outputs of D) and
    // hence they are all affected by the decision on whether D should be put
    // onto the mirror path or not.
    bool has_subgraph_converged;
    do {
      has_subgraph_converged = true;
      for (const Node* const subgraph_node : subgraph_topo_order) {
        for (const NodeEntry& subgraph_node_entry :
             subgraph_node->inputs) {
          const std::unordered_set<const Node*> ref_nodes =
              node_entry_ref_map[gidx.entry_id(subgraph_node_entry)];

          for (const Node* const ref_node : ref_nodes) {
            // If there are other nodes that reference the node entry and that
            // node satisfies the following conditions:
            //   (1) belongs to the forward graph, and
            //   (2) is not part of the subgraph
            // We add that node to the subgraph and adjust the topological order
            // accordingly.
            if (ref_node != subgraph_node && idx.exist(ref_node) &&
                subgraph.find(ref_node) == subgraph.end()) {
              // Forward propagate from the reference node until the mirroring
              // function returns false. This indicates that the head of the
              // branch has been reached (i.e., B or C in our previously
              // illustrated example), and we add it to the subworklist for
              // another backpropagation.
              std::queue<const Node*> ref_node_heads;
              ref_node_heads.push(ref_node);
              for (; !ref_node_heads.empty(); ref_node_heads.pop()) {
                const Node* const ref_node_head = ref_node_heads.front();
                bool is_ref_node_head_output = false;
                for (const NodeEntry& y : ys) {
                  if (ref_node_head == y.node.get()) {
                    is_ref_node_head_output = true;
                  }
                }
                if (!mirror_fun(*ref_node_head) || is_ref_node_head_output) {
                  subworklist.push(ref_node_head);
                  continue;
                }

                uint32_t gnid = gidx.node_id(ref_node_head);
                for (uint32_t oid = 0; oid < ref_node_head->num_outputs(); ++oid) {
                  uint32_t geid = gidx.entry_id(gnid, oid);
                  for (const Node* const n : node_entry_ref_map[geid]) {
                    if (idx.exist(n)) {
                      ref_node_heads.push(n);
                    }
                  }
                }  // for (oid ∈ [0, ref_node_head->num_outputs()))
              }  // for (ref_node_head ∈ ref_node_heads)
              // Do the backpropagation again. The topological order of the
              // subworklist can be directly appended to the end of the existing
              // order. E,g, in our previous example, we expect to have
              // `subgraph_topo_order` = {D, A} + {B} + {C}
              subworklist_backprop();
              // indicate that the subgraph has not changed the quit the loop
              has_subgraph_converged = false;
              break;
            }  // if (ref_node != subgraph_node && idx.exist(ref_node) &&
               //     subgraph.find(ref_node) == subgraph.end()
          }  // for (ref_node ∈ ref_nodes)
          if (!has_subgraph_converged) {
            break;
          }
        }  // for (subgraph_node_entry ∈ subgraph_node->inputs)
        if (!has_subgraph_converged) {
          break;
        }
      }  // for (subgraph_node ∈ subgraph_topo_order)
    } while (!has_subgraph_converged);
    // =========================================================================
    // --- Forward Pass ---
    // =========================================================================
    // Now that the subgraph is complete, we start by assuming that all the
    // nodes in the subgraph can be mirrored, and forward propagate starting
    // from the subgraph frontier. The propagation is successful if the amount
    // of storage released by removing the frontier nodes off the mirror path is
    // greater or equal to the storage allocated.
    do {
      has_subgraph_converged = true;
      // Obtain the subgraph frontier. The subgraph frontier denotes a group of
      // nodes whose inputs satisfy the following conditions:
      //  (1) fails the mirroring function, or
      //  (2) has been marked as NOT on the mirror path, i.e.,
      //      `mirror_map[input_node] == nullptr`
      // E.g., consider the subgraph below:
      // A
      // ↑
      // B
      // ↑
      // C
      // The subgraph frontier in this example is {C}. As C is the place where
      // the mirror path (and hence the forward propagation) starts.
      subgraph_frontier.clear();
      for (const Node* const subgraph_node : subgraph) {
        if (!mirror_fun(*subgraph_node)) {
          mirror_map[subgraph_node] = nullptr;
          continue;
        }
        if (mirror_map.find(subgraph_node) != mirror_map.end()) {
          continue;
        }
        bool is_frontier = true;
        for (const NodeEntry& e : subgraph_node->inputs) {
          auto iter = mirror_map.find(e.node.get());
          if (mirror_fun(*(e.node)) &&
              !(iter != mirror_map.end() && iter->second == nullptr)) {
            is_frontier = false;
          }
        }
        for (const ObjectPtr& n : subgraph_node->control_deps) {
          auto iter = mirror_map.find(n.get());
          if (mirror_fun(*n) &&
              !(iter != mirror_map.end() && iter->second == nullptr)) {
            is_frontier = false;
          }
        }
        if (is_frontier) {
          subgraph_frontier.emplace(subgraph_node, false);
        }
      }  // for (subgraph_node ∈ subgraph)
      for (std::pair<const Node* const, bool>& frontier_node : subgraph_frontier) {
        if (frontier_node.second) {
          // If the frontier node has been marked as true, then this indicates
          // that the node has been forward propagated before (by other nodes
          // that share the same input).
          continue;
        }
        // As we do the forward propagation, we not only propagate the current
        // frontier node individually, but all the frontier nodes that share the
        // same input with the current one. This is a recursive progress because
        // it is possible for A to share the same input with B and B, at the
        // same time, to share the same input with C, like in the graph below:
        //    D       E
        //    ↑       ↑
        //  ↗   ↖   ↗   ↖
        // A      B       C
        std::unordered_set<const Node*> forward_candidates{frontier_node.first};
        frontier_node.second = true;
        bool has_forward_candidates_converged;
        do {
          has_forward_candidates_converged = true;
          for (const Node* const candidate : forward_candidates) {
            for (const NodeEntry& candidate_input : candidate->inputs) {
              uint32_t geid = gidx.entry_id(candidate_input);
              const std::unordered_set<const Node*>& ref_nodes = node_entry_ref_map[geid];
              for (const Node* const ref_node : ref_nodes) {
                if (ref_node != frontier_node.first &&
                    subgraph_frontier.find(ref_node) != subgraph_frontier.end() &&
                    forward_candidates.find(ref_node) == forward_candidates.end()) {
                  subgraph_frontier[ref_node] = true;
                  forward_candidates.insert(ref_node);
                  has_forward_candidates_converged = false;
                }
              }  // for (ref_node ∈ ref_nodes)
              if (!has_forward_candidates_converged) {
                break;
              }
            }  // for (candidate_input ∈ candidate->inputs)
            if (!has_forward_candidates_converged) {
              break;
            }
          }  // for (candidate ∈ forward_candidates)
        } while (!has_forward_candidates_converged);
        // Record the node entries that are newly allocated and those that are
        // released. A node entry can be released if all its referencing nodes
        // are part of the subgraph frontier. Otherwise, it is in the allocated set.
        std::unordered_set<uint32_t> newly_allocated_node_entries,
                                     released_node_entries;
        for (const Node* const candidate : forward_candidates) {
          uint32_t nid = idx.node_id(candidate),
                   gnid = gidx.node_id(candidate);
          for (uint32_t oid = 0; oid < candidate->num_outputs(); ++oid) {
            uint32_t eid = idx.entry_id(nid, oid),
                     geid = gidx.entry_id(gnid, oid);
            if (node_entry_ref_map[geid].size() != 0) {
              newly_allocated_node_entries.insert(eid);
            }
          }
          for (const NodeEntry& candidate_input : candidate->inputs) {
            uint32_t eid = idx.entry_id(candidate_input),
                     geid = gidx.entry_id(candidate_input);
            const std::unordered_set<const Node*>& ref_nodes = node_entry_ref_map[geid];
            bool can_be_released = true;
            for (const Node* const ref_node : ref_nodes) {
              if (subgraph_frontier.find(ref_node) == subgraph_frontier.end()) {
                newly_allocated_node_entries.insert(eid);
                can_be_released = false;
              }
            }
            if (can_be_released) {
              released_node_entries.insert(eid);
            }
          }  // for (candidate_input ∈ candidate->input)
        }  // for (candidate ∈ forward_candidates)

        // Now, compare the total amount of newly allocated storage versus the
        // released storage, if the latter is greater or equal to the former,
        // then we remove the current node from the frontier. Otherwise all the
        // forward candidate nodes are marked as on the mirror path.
        size_t newly_allocated_storage = 0, released_storage = 0;
        for (const uint32_t eid : newly_allocated_node_entries) {
          newly_allocated_storage += src_shapes[eid].Size() *
                                     MXGetDTypeSize(src_dtypes[eid]);
        }
        for (const uint32_t eid : released_node_entries) {
          released_storage += src_shapes[eid].Size() * MXGetDTypeSize(src_dtypes[eid]);
        }
        if (released_storage >= newly_allocated_storage) {
          for (const Node* const candidate : forward_candidates) {
            CHECK(subgraph_frontier.find(candidate) != subgraph_frontier.end());
            subgraph_frontier.erase(candidate);
            mirror_map[candidate] = nullptr;
          }
          has_subgraph_converged = false;
          break;
        }  // if (released_storage >= newly_allocated_storage)
      }  // for (frontier_node ∈ subgraph_frontier)
    } while (!has_subgraph_converged);

    // Finally, mark all the remaining nodes of the subgraph as on the mirror path.
    for (const Node* const subgraph_node : subgraph_topo_order) {
      if (mirror_map.find(subgraph_node) != mirror_map.end()) {
        continue;
      }
      ObjectPtr subgraph_node_mirror = Node::Create();
      *subgraph_node_mirror = *subgraph_node;
      subgraph_node_mirror->attrs.name += "_mirror";
      for (NodeEntry& e : subgraph_node_mirror->inputs) {
        e.node = MapFwdNodeToMirrorPath(e.node, mirror_map);
      }
      for (ObjectPtr& n : subgraph_node_mirror->control_deps) {
        n = MapFwdNodeToMirrorPath(n, mirror_map);
      }
      mirror_map[subgraph_node] = subgraph_node_mirror;
    }
  }  // for (workitem ∈ worklist)
  DFSVisit(ys,
           [&](const ObjectPtr& node) {
             if (mirror_map.at(node.get()) != nullptr) {
               node->attrs.dict["__mirror_stage__"] = "1";
             } else {
               node->attrs.dict["__mirror_stage__"] = "0";
             }
           });
  return BuildGradientGraph(src, xs, topo_order,
                            output_grads,
                            mirror_fun, mirror_map);
}


/*!
 * \brief Auxiliary function that checks whether all the gradients are zero or not.
 */
inline bool CheckGradAllZero(const std::vector<NodeEntry>& grads,
                             const std::vector<const Op*>& zero_ops) {
  if (!grads.size() || !zero_ops.size()) return false;
  for (const auto& g : grads) {
    bool found = false;
    for (const auto& op : zero_ops) {
      if (g.node->op() == op) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}


Graph BuildGradientGraph(
    const Graph& src,
    const std::vector<NodeEntry>& xs,
    const std::vector<ObjectPtr>& topo_order,
    std::unordered_map<const Node*, std::vector<GradEntry> > output_grads,
    std::function<int(const Node&)> mirror_fun,
    const std::unordered_map<const Node*, ObjectPtr>& mirror_map) {
  static auto& grad_fun_map = Op::GetAttr<nnvm::FGradient>("FGradient");

  // gradient aggregation function
  using AggFun = std::function<NodeEntry (std::vector<NodeEntry>&&)>;
  AggFun agg_fun = [](std::vector<NodeEntry>&& v)->NodeEntry {
        if (v.size() == 1) {
          return std::move(v[0]);
        } else if (v.size() == 0) {
          ObjectPtr zero_grad_node = Node::Create();
          zero_grad_node->attrs.op = Op::Get("zeros");
          zero_grad_node->attrs.name = "zero_grad";
          zero_grad_node->attrs.op->attr_parser(&(zero_grad_node->attrs));
          return NodeEntry{zero_grad_node, 0, 0};
        } else {
          ObjectPtr grad_sum_node = Node::Create();
          grad_sum_node->attrs.op = Op::Get("elemwise_sum");
          grad_sum_node->inputs = std::move(v);
          grad_sum_node->attrs.name = "grad_sum";
          grad_sum_node->attrs.dict["num_args"] =
              std::to_string(grad_sum_node->inputs.size());
          grad_sum_node->attrs.op->attr_parser(&(grad_sum_node->attrs));
          return NodeEntry{grad_sum_node, 0, 0};
        }
      };
  if (src.attrs.count("grad_aggregate_fun") != 0) {
    agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }

  // zero and copy operators
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }
  const Op* copy_op = (src.attrs.count("copy_op_str") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op_str")) : nullptr;

  std::vector<NodeEntry> out_agg_grads;
  for (auto topo_order_rit = topo_order.rbegin();
       topo_order_rit != topo_order.rend(); ++topo_order_rit) {
    const ObjectPtr& src_fwd_node = *topo_order_rit;
    if (src_fwd_node->is_variable()) continue;

    // gather all the output gradient entries and apply the aggregation function
    out_agg_grads.clear();
    auto& out_grad_vec = output_grads.at(src_fwd_node.get());
    for (auto & e : out_grad_vec) {
      e.sum = agg_fun(std::move(e.grads));
      out_agg_grads.push_back(e.sum);
    }
    if (src_fwd_node->inputs.size() != 0) {
      // If the current node has inputs, the gradients need to be further
      // propagated backward.
      ObjectPtr fwd_node = MapFwdNodeToMirrorPath(src_fwd_node, mirror_map);
      // calculate the input gradients
      std::vector<NodeEntry> input_grads;
      if (grad_fun_map.count(src_fwd_node->op())) {
        input_grads = grad_fun_map[src_fwd_node->op()](fwd_node, out_agg_grads);
        CHECK_EQ(src_fwd_node->inputs.size(), input_grads.size())
            << "The Gradient function is not returning enough gradients.";
        // If the operator node fails the mirror function, it is however still
        // possible for its feature maps to be recomputed without incurring
        // significant runtime overhead. The reason is because some operators
        // have their feature maps sit on the inputs rather than the outputs.
        // E.g., the fully-connected layer (Y=XW^T), whose gradients are given
        // by dX = dYW, dW = dY^TX and hence have no data dependency on Y.
        if (mirror_fun != nullptr && !mirror_fun(*fwd_node)) {
          for (NodeEntry& input_grad : input_grads) {
            for (NodeEntry& grad_input : input_grad.node->inputs) {
              const ObjectPtr& grad_input_node_mirrored = MapFwdNodeToMirrorPath(
                  grad_input.node, mirror_map);
              grad_input = NodeEntry(
                  grad_input_node_mirrored,
                  grad_input.index,
                  grad_input.version);
            }  // for (grad_input ∈ input_grad.node->inputs)
          }  // for (input_grad ∈ input_grads)
        }  // if (mirror_fun != nullptr && !mirror_fun(*fwd_node))
      } else if (CheckGradAllZero(out_agg_grads, zero_ops)) {
        for (size_t i = 0; i < src_fwd_node->num_inputs(); ++i) {
          std::ostringstream os;
          if (1 == src_fwd_node->num_inputs()) {
            os << fwd_node->attrs.name << "_backward";
          } else {
            os << fwd_node->attrs.name << "_in" << i << "_backward";
          }
          auto p = Node::Create();
          p->attrs.op = zero_ops[0];
          p->attrs.name = os.str();
          p->inputs.push_back(fwd_node->inputs[i]);
          p->control_deps.emplace_back(fwd_node);
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          input_grads.emplace_back(p, 0, 0);
        }  // for (i ∈ src_fwd_node->num_inputs())
      } else {
        std::string message = "Operator " + std::string(src_fwd_node->op()->name)
          + "is non-differentiable because it didn't register FGradient attribute.";
        throw nnvm::pass::InvalidGraphError(message);
      }
      for (const auto& e : input_grads) {
        CHECK(e.node);
      }
      auto input_grad_iter = input_grads.begin();
      CHECK(src_fwd_node->inputs.size() <= input_grads.size());
      for (auto input_iter = src_fwd_node->inputs.begin();
           input_iter != src_fwd_node->inputs.end();
           ++input_iter, ++input_grad_iter) {
        // propagate the input gradients to the output gradients of the input nodes
        output_grads[input_iter->node.get()][input_iter->index]
            .grads.emplace_back(std::move(*input_grad_iter));
      }
    }  // if (src_fwd_node->inputs.size() != 0)
  }  // for (topo_order_rit ∈ reverse(topo_order))
  // take out the xs' grads
  Graph ret;
  ret.outputs.resize(xs.size());
  NodeEntryMap<std::pair<size_t, size_t> > unique_grads;
  size_t counter = 0;
  for (const NodeEntry& e : xs) {
    GradEntry& entry = output_grads[e.node.get()][e.index];
    // aggregate sum if there haven't been
    if (entry.sum.node.get() == nullptr) {
      entry.sum = agg_fun(std::move(entry.grads));
    }
    if (copy_op != nullptr) {
      auto kv = unique_grads.find(entry.sum);
      if (kv == unique_grads.end()) {
        unique_grads.emplace(std::move(entry.sum), std::make_pair(1, counter));
      } else {
        ObjectPtr copy_node = Node::Create();
        std::ostringstream os;
        os << entry.sum.node->attrs.name << "_" << kv->second.first << "_copy";
        kv->second.first++;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->inputs.emplace_back(entry.sum);
        if (copy_node->attrs.op->attr_parser != nullptr) {
          copy_node->attrs.op->attr_parser(&(copy_node->attrs));
        }
        unique_grads.emplace(NodeEntry{std::move(copy_node), 0, 0},
                             std::make_pair(1, counter));
      }
    } else {
      ret.outputs[counter] = entry.sum;
    }
    ++counter;
  }  // for (e ∈ xs)
  if (copy_op != nullptr) {
    for (const auto& kv : unique_grads) {
      ret.outputs[kv.second.second] = kv.first;
    }
  }
  return ret;
}


// register pass
NNVM_REGISTER_PASS(MXGradient)
.describe(R"(Return a gradient graph of src.attrs["ys"] wrt src.attrs["xs"])")
.set_body(Gradient)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("in_arg_shapes")
.depend_graph_attr("in_arg_dtypes")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace

}  // namespace pass
}  // namespace nnvm
