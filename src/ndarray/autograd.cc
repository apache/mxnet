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
 * \file autograd.cc
 * \brief Implementation of AutogradRuntime module.
 */

#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/pass_functions.h>
#include <unordered_set>
#include <iostream>
#include "../executor/graph_executor.h"
#include "./autograd.h"

namespace mxnet {
namespace autograd {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;
using exec::GraphExecutor;

#if DMLC_CXX11_THREAD_LOCAL
thread_local bool AutogradRuntime::is_train_ = false;
thread_local bool AutogradRuntime::is_recording_ = false;
#else
MX_THREAD_LOCAL bool AutogradRuntime::is_train_ = false;
MX_THREAD_LOCAL bool AutogradRuntime::is_recording_ = false;
#endif

template<typename FVisit>
inline void AGDFSVisit(const std::vector<AGNodeEntry>& heads,
                       FVisit fvisit) {
  typedef const AGNodePtr* GNode;
  std::vector<GNode> head_nodes(heads.size());
  std::transform(heads.begin(), heads.end(), head_nodes.begin(),
                 [](const AGNodeEntry& e)->GNode {
                   return &e.ag_node;
                 });
  nnvm::PostOrderDFSVisit<GNode, AGNode*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->AGNode* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t { return (*n)->inputs.size(); },
      [](GNode n, uint32_t index)->GNode { return &(*n)->inputs.at(index).ag_node; });
}

nnvm::NodeEntry AGNodeEntry::nn_entry() const {
  return nnvm::NodeEntry{ag_node->nn_node, index, version};
}

bool AGNodeEntry::is_none() const {
  return ag_node == nullptr || ag_node->outputs.empty();
}

AutogradRuntime::AutogradRuntime() {}

void AutogradRuntime::MarkVariables(
    const std::vector<NDArray*>& variables,
    const std::vector<mx_uint>& grad_reqs,
    const std::vector<NDArray*>& gradients) {
  for (uint32_t i = 0; i < variables.size(); ++i) {
    std::string str_c(std::to_string(variable_count_++));

    AGNodeEntry e{
      AGNode::Create(
        nnvm::Symbol::CreateVariable("var" + str_c).outputs[0].node), 0, 0};
    variables[i]->entry_.clear();
    e.ag_node->outputs.emplace_back(*variables[i]);

    AGNodeEntry ge{
      AGNode::Create(
        nnvm::Symbol::CreateVariable("grad" + str_c).outputs[0].node), 0, 0};
    gradients[i]->entry_.clear();
    ge.ag_node->outputs.emplace_back(*gradients[i]);
    gradients[i]->entry_ = std::move(ge);
    e.ag_node->out_grads.emplace_back(*gradients[i]);

    e.ag_node->grad_req = static_cast<OpReqType>(grad_reqs[i]);
    variables[i]->entry_ = std::move(e);  // assign last to prevent cyclic reference
  }
}

void AutogradRuntime::RecordImperativeFCompute(const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, OpStatePtr());
}

void AutogradRuntime::RecordImperativeOperator(const OpStatePtr& state,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, state);
}

std::shared_ptr<AutogradRuntime> AutogradRuntime::_GetSharedRef() {
  static std::shared_ptr<AutogradRuntime> inst(new AutogradRuntime());
  return inst;
}

AutogradRuntime* AutogradRuntime::Get() {
  static AutogradRuntime *ptr = _GetSharedRef().get();
  return ptr;
}

AGNodePtr AutogradRuntime::RecordOp(const nnvm::Op* op,
                                    const nnvm::NodeAttrs& attrs,
                                    std::vector<NDArray> *p_inputs,
                                    std::vector<NDArray> *p_outputs,
                                    const OpStatePtr& state) {
  std::vector<NDArray>& inputs  = *p_inputs;
  std::vector<NDArray>& outputs = *p_outputs;

  NodePtr nn_node = Node::Create();
  nn_node->attrs = attrs;
  nn_node->attrs.name = "node_" + std::to_string(node_count_++);

  AGNodePtr ag_node = AGNode::Create(nn_node);
  ag_node->state = state;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].entry_.is_none()) {
      AGNodeEntry e{
        AGNode::Create(
          nnvm::Symbol::CreateVariable(
            "null" + std::to_string(variable_count_++)).outputs[0].node), 0, 0};
      e.ag_node->outputs.emplace_back(inputs[i]);
      e.ag_node->out_grads.emplace_back();
      inputs[i].entry_ = std::move(e);  // assign last to prevent cyclic reference
    }
    nn_node->inputs.push_back(inputs[i].entry_.nn_entry());
    ag_node->inputs.push_back(inputs[i].entry_);
  }

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    CHECK(outputs[i].entry_.is_none())
      << "Inplace operations (+=, -=, x[:]=, etc) are not supported when "
      << "recording with autograd. "
      << "Assigning to NDArrays that are already in a computational graph "
      << "will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a record section. ";
    outputs[i].entry_.clear();
    ag_node->outputs.push_back(outputs[i]);
    outputs[i].entry_ = AGNodeEntry{ag_node, i, 0};
  }

  return ag_node;
}

void AutogradRuntime::ComputeGradient(const std::vector<NDArray>& outputs,
                                      const std::vector<NDArray>& ograds,
                                      bool retain_graph, bool is_train) {
  static auto& fmutate_inputs = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  std::vector<AGNodeEntry> heads;
  Symbol sym;
  NodeEntryMap<NDArray> feed_dict;
  for (const auto& i : outputs) {
    CHECK(!i.entry_.is_none())
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_recording to true or use autograd.record() to save "
      << "computational graphs for backward. If you want to differentiate the same "
      << "graph twice, you need to pass retain_graph=True to backward.";
    heads.emplace_back(i.entry_);
    sym.outputs.emplace_back(i.entry_.nn_entry());
  }

  std::unordered_set<AGNode*> mutable_set;
  std::vector<AGNodePtr> vlist;
  std::vector<NDArray> args, args_grad;
  std::vector<NDArray> aux_states;
  std::vector<OpReqType> grad_reqs;
  std::unordered_map<const nnvm::Node*, OpStatePtr> saved_states;
  AGDFSVisit(heads, [&](const AGNodePtr& n) {
      CHECK(n->nn_node != nullptr)
          << "Node is differentiated twice without retaining graph the first time. "
          << "This usually happens when you want to differentiate a graph twice but "
          << "forgot to set retain_graph=True the first time. If you are training "
          << "recurrent model (like LSTMs) maybe you forgot to detach the hidden "
          << "state from the previous iteration before feeding it to the next iteration.";
      if (n->nn_node->is_variable()) {
        vlist.push_back(n);
      } else {
        if (n->state) {
          saved_states.insert({n->nn_node.get(), n->state});
        }
        if (fmutate_inputs.count(n->nn_node->op())) {
          for (uint32_t i : fmutate_inputs[n->nn_node->op()](n->nn_node->attrs)) {
            mutable_set.insert(n->inputs[i].ag_node.get());
          }
        }
      }
      for (uint32_t i = 0; i < n->outputs.size(); ++i) {
        feed_dict.insert({NodeEntry{n->nn_node, i, 0}, n->outputs[i]});
      }
    });

  bool has_writeto = false;
  for (const auto& n : vlist) {
    if (mutable_set.count(n.get())) {
      aux_states.push_back(n->outputs[0]);
    } else {
      if (n->grad_req != kNullOp) {
        n->fresh_out_grad = true;
      }
      args.push_back(n->outputs[0]);
      args_grad.push_back(n->out_grads[0]);
      grad_reqs.push_back(n->grad_req);
      has_writeto = has_writeto || n->grad_req == kWriteTo;
    }
  }

  if (args.size()) {
    std::map<std::string, Context> ctx_map;
    auto exec = new exec::GraphExecutor();
    // (TODO) too hack here
    exec->saved_states_ = saved_states;
    exec->Init(sym, args[0].ctx(), ctx_map,
               args, args_grad, grad_reqs,
               aux_states, nullptr, feed_dict);

    std::vector<NDArray> head_grads;
    head_grads.reserve(exec->head_grad_array_.size());
    CHECK_EQ(ograds.size(), exec->output_arrays_.size());

    for (size_t i = 0; i < ograds.size(); ++i) {
      if (ograds[i].is_none()) {
        head_grads.emplace_back(
          exec->output_arrays_[i].shape(), exec->output_arrays_[i].ctx(),
          false, exec->output_arrays_[i].dtype());
        head_grads.back() = static_cast<real_t>(1.0);
      } else {
        head_grads.emplace_back(ograds[i]);
      }
    }

    exec->Backward(head_grads, is_train);
    delete exec;
  }

  if (!retain_graph) {
    for (auto& i : heads) {
      i.ag_node->clear_history();
    }
  } else if (has_writeto) {
    LOG(INFO)
        << "Warning: when calling backward with retain_graph=True, grad_req for "
        << "Parameters should be set to 'add'. Otherwise the second backward "
        << "will over-write gradients from the first backward. Also remember "
        << "to manually set gradients to zero with zero_grad before starting the "
        << "next iteration.";
  }
}

}  // namespace autograd
}  // namespace mxnet
