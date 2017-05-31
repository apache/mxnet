/*!
 *  Copyright (c) 2017 by Contributors
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
thread_local bool AutogradRuntime::is_train_;
#else
MX_THREAD_LOCAL bool AutogradRuntime::is_train_;
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

    AGNodeEntry e{AGNode::Create(Node::Create()), 0, 0};
    variables[i]->entry_.clear();
    e.ag_node->outputs.emplace_back(*variables[i]);

    AGNodeEntry ge{AGNode::Create(Node::Create()), 0, 0};
    gradients[i]->entry_.clear();
    ge.ag_node->outputs.emplace_back(*gradients[i]);
    ge.ag_node->nn_node->attrs.name = "grad" + str_c;
    gradients[i]->entry_ = std::move(ge);
    e.ag_node->out_grads.emplace_back(*gradients[i]);

    e.ag_node->grad_req = static_cast<OpReqType>(grad_reqs[i]);
    e.ag_node->nn_node->attrs.name = "var" + str_c;
    variables[i]->entry_ = std::move(e);  // assign last to prevent cyclic reference
  }
}

void AutogradRuntime::RecordImperativeFCompute(const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, nullptr);
}

void AutogradRuntime::RecordImperativeOperator(const std::shared_ptr<Operator>& opr,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, opr);
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
                                    const std::shared_ptr<Operator>& opr) {
  std::vector<NDArray>& inputs  = *p_inputs;
  std::vector<NDArray>& outputs = *p_outputs;

  NodePtr nn_node = Node::Create();
  nn_node->attrs = attrs;
  nn_node->attrs.name = "node_" + std::to_string(node_count_++);

  AGNodePtr ag_node = AGNode::Create(nn_node);
  ag_node->opr = opr;

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    CHECK(outputs[i].entry_.is_none())
      << "Output NDArray is non-empty and already in another computation graph. "
      << "Assigning to it will cause undefined behavior when evaluating gradients. "
      << "Please call backward first to clear the graph or do this out side of "
      << "a train section. ";
    outputs[i].entry_.clear();
    ag_node->outputs.push_back(outputs[i]);
    outputs[i].entry_ = AGNodeEntry{ag_node, i, 0};
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].entry_.is_none()) {
      AGNodeEntry e{AGNode::Create(Node::Create()), 0, 0};
      e.ag_node->outputs.emplace_back(inputs[i]);
      e.ag_node->out_grads.emplace_back();
      e.ag_node->nn_node->attrs.name = "var_" + std::to_string(variable_count_++);
      inputs[i].entry_ = std::move(e);  // assign last to prevent cyclic reference
    }
    nn_node->inputs.push_back(inputs[i].entry_.nn_entry());
    ag_node->inputs.push_back(inputs[i].entry_);
  }

  return ag_node;
}

void AutogradRuntime::ComputeGradient(const std::vector<NDArray>& outputs,
                                      const std::vector<NDArray>& ograds,
                                      bool retain_graph) {
  static auto& fmutate_inputs = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  std::vector<AGNodeEntry> heads;
  Symbol sym;
  NodeEntryMap<NDArray> feed_dict;
  for (const auto& i : outputs) {
    CHECK(!i.entry_.is_none())
      << "Cannot differentiate node because it is not in a computational graph. "
      << "You need to set is_training to true or use a train_section to save "
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
  std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>> saved_opr;
  AGDFSVisit(heads, [&](const AGNodePtr& n) {
      if (n->nn_node->is_variable()) {
        vlist.push_back(n);
      } else {
        if (n->opr != nullptr) {
          saved_opr.insert({n->nn_node.get(), n->opr});
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
    }
  }

  if (args.size()) {
    std::map<std::string, Context> ctx_map;
    auto exec = new exec::GraphExecutor();
    // (TODO) too hack here
    exec->saved_opr_ = saved_opr;
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

    exec->Backward(head_grads);
    delete exec;
  }

  if (!retain_graph) {
    for (auto& i : heads) {
      i.ag_node->clear_history();
    }
  }
}

}  // namespace autograd
}  // namespace mxnet
