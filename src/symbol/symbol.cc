/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief symbol of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mxnet {
/*!
 * \brief Node is represents node of an operator in the symbolic graph.
 *
 * It stores connection to the inputs to function represented by OperatorProperty
 * NOTE on data structure: there are three types of node:
 * - Normal node: contains all the necessary elements of a graph.
 * - OperatorProperty: the inputs_ is empty, represents an OperatorProperty that has not been applied.
 * - Variable: the sym_ is nullptr, represents an named Variable of tensors that can be composed.
 */
struct Symbol::Node {
  /*! \brief Operator of this node */
  std::unique_ptr<OperatorProperty> op;
  /*! \brief name of the node */
  std::string name;
  /*! \brief inputs to this node */
  std::vector<DataEntry> inputs;
  /*!
   * \brief constructor
   * \param op the OperatorProperty to construct the Node
   * \param name the name of the symbol
   */
  explicit Node(OperatorProperty* op = nullptr, const std::string& name = "")
      : op(op), name(name) {
  }
  /*! \return Whether the symbol is atomic */
  inline bool is_atomic() const {
    return inputs.size() == 0 && op != nullptr;
  }
  /*! \return Whether it is unit variable */
  inline bool is_variable() const {
    return op == nullptr;
  }
};

/*! \return whwther the symbol is atomic */
inline bool Symbol::is_atomic() const {
  return heads_.size() == 1 && heads_[0].source->is_atomic();
}
// implementation of template functions
template<typename FVisit>
inline void Symbol::DFSVisit(FVisit fvisit) const {
  std::vector<Node*> stack;
  std::unordered_set<Node*> visited;
  // put the head into the graph
  for (auto &head : heads_) {
    Node *ptr = head.source.get();
    if (visited.count(ptr) == 0) {
      stack.push_back(ptr);
      visited.insert(ptr);
    }
  }
  while (!stack.empty()) {
    Node* back = stack.back();
    stack.pop_back();
    fvisit(back);
    for (auto it = back->inputs.rbegin(); it != back->inputs.rend(); ++it) {
      Node *ptr = it->source.get();
      if (visited.count(ptr) == 0) {
        stack.push_back(ptr);
        visited.insert(ptr);
      }
    }
  }
}

int Symbol::FindDuplicateArgs(std::unordered_map<std::string, int> *out) const {
  out->clear();
  int max_dup = 1;
  this->DFSVisit([out, &max_dup](Node *node) {
      if (node->is_variable()) {
        auto iter = out->find(node->name);
        if (iter == out->end()) {
          (*out)[node->name] = 1;
        } else {
          ++iter->second;
          max_dup = std::max(max_dup, iter->second);
        }
      }
    });
  return max_dup;
}

// public functions
Symbol Symbol::Copy() const {
  std::unordered_map<Node*, std::shared_ptr<Node> > old_new;
  // use DFSVisit to copy all the nodes
  this->DFSVisit([&old_new](Node *node) {
      if (node->op == nullptr) {
        old_new[node] = std::make_shared<Node>(nullptr, node->name);
      } else {
        old_new[node] =  std::make_shared<Node>(node->op->Copy(), node->name);
      }
    });
  // connect nodes of new graph
  for (const auto &kv : old_new) {
    for (const DataEntry& n : kv.first->inputs) {
      Node *ptr = n.source.get();
      kv.second->inputs.push_back(DataEntry(old_new[ptr], n.index));
    }
  }
  // set the head
  Symbol s;
  for (auto &head : heads_) {
    s.heads_.push_back(DataEntry(old_new[head.source.get()], head.index));
  }
  return s;
}

void Symbol::Print(std::ostream &os) const {
  if (this->is_atomic()) {
    os << "AtomicFunction "<< " Type:" << heads_[0].source->op->TypeString() << '\n'
       << "Inputs:";
    std::vector<std::string> args = this->ListArguments();
    for (size_t i = 0; i < args.size(); ++i) {
      os << "\targ[" << i << "]=" << args[i] << "\n";
    }
  } else {
    // use DFSVisit to copy all the nodes
    os << "Outputs:\n";
    for (size_t i = 0; i < heads_.size(); ++i) {
      os << "\toutput[" << i << "]=" << heads_[i].source->name
         << '(' << heads_[i].index << ")\n";
    }
    this->DFSVisit([&os](Node *node) {
        if (node->is_variable()) {
          os << "Variable:" << node->name << '\n';
        } else {
          os << "Name: " << node->name << " Type:" << node->op->TypeString() << '\n'
             << "Inputs:\n";
          for (size_t i = 0; i < node->inputs.size(); ++i) {
            os << "\targ[" << i << "]=" << node->inputs[i].source->name
               << '(' << node->inputs[i].index << ")\n";
          }
        }
      });
  }
}

std::vector<std::string> Symbol::ListArguments() const {
  std::vector<std::string> ret;
  if (this->is_atomic()) {
    return heads_[0].source->op->ListArguments();
  } else {
    this->DFSVisit([&ret](Node *node) {
        if (node->is_variable()) {
          ret.push_back(node->name);
        }
      });
    return ret;
  }
}

std::vector<std::string> Symbol::ListReturns() const {
  std::vector<std::string> ret;
  for (auto &head : heads_) {
    if (head.source->is_variable()) {
      ret.push_back(head.source->name);
    } else {
      // TODO(bing) rethink about output naming
      auto &hname = head.source->name;
      std::string rname = head.source->op->ListReturns()[head.index];
      if (hname.length() == 0) {
        ret.push_back(std::move(rname));
      } else {
        ret.push_back(hname + '_' + rname);
      }
    }
  }
  return std::move(ret);
}

Symbol Symbol::operator[] (size_t index) const {
  size_t nreturn = NumReturns();
  CHECK_LT(index, nreturn) << "Symbol only accept nonnegative index";
  if (nreturn == 1) {
    return *this;
  } else {
    Symbol s;
    s.heads_.push_back(heads_[index]);
    return s;
  }
}

void Symbol::Compose(const std::vector<Symbol>& args,
                     const std::string& name) {
  CHECK_EQ(NumReturns(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_NE(args[i].NumReturns(), 1)
        << "Argument " << i << " is a tuple, scalar is required";
  }
  // positional arguments requires all arguments for now.
  // TODO(bing) consider partial assignments
  if (this->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
    CHECK_EQ(args.size(), req_args.size())
        << "Incorrect number of arguments, requires " << req_args.size()
        << ", provided " << args.size();
    heads_[0].source->inputs.resize(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      heads_[0].source->inputs[i] = args[i].heads_[0];
    }
  } else {
    // find all the place holders
    size_t arg_counter = 0;
    std::unordered_map<Node*, const DataEntry*> replace_map;
    std::vector<std::pair<DataEntry*, const DataEntry*> > replace_plan;
    // replace map stores the existing replacement plan for arguments node
    this->DFSVisit([&arg_counter, &replace_map, &replace_plan, &args](Node *node) {
        // visit all the childs, find possible replacement
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          DataEntry *e = &(node->inputs[i]);
          if (e->source->is_variable()) {
            const DataEntry *target = nullptr;
            auto iter = replace_map.find(e->source.get());
            if (iter == replace_map.end()) {
              if (arg_counter < args.size()) {
                target = &(args[arg_counter].heads_[0]);
                replace_map[e->source.get()] = target;
              }
              ++arg_counter;
            } else {
              target = iter->second;
            }
            replace_plan.push_back(std::make_pair(e, target));
          }
        }
      });
    CHECK_EQ(args.size(), arg_counter)
        << "Incorrect number of arguments, requires " << arg_counter
        << ", provided " << args.size();
    // now run the replacement
    for (const auto& kv : replace_plan) {
      *(kv.first) = *(kv.second);
    }
  }
}

void Symbol::Compose(const std::unordered_map<std::string, Symbol>& kwargs,
                     const std::string& name) {
  CHECK_EQ(NumReturns(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (const auto& kv : kwargs) {
    CHECK_EQ(kv.second.NumReturns(), 1)
        << "Keyword Argument " << kv.first << " is a tuple, scalar is required";
  }
  size_t nmatched = 0;
  if (this->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
    heads_[0].source->inputs.resize(req_args.size());
    for (size_t i = 0; i < req_args.size(); ++i) {
      auto iter = kwargs.find(req_args[i]);
      if (iter != kwargs.end()) {
        heads_[0].source->inputs[i] = iter->second.heads_[0];
        ++nmatched;
      } else {
        // create a variable node
        // TODO(bing): think of naming convention
        if (name.length() == 0) {
          heads_[0].source->inputs[i] = DataEntry(
              std::make_shared<Node>(nullptr, req_args[i]), 0);
        } else {
          heads_[0].source->inputs[i] = DataEntry(
              std::make_shared<Node>(nullptr, name + '_' + req_args[i]), 0);
        }
      }
    }
    // if things goes wrong recover the old state
    if (nmatched != kwargs.size()) {
      heads_[0].source->inputs.clear();
    }
  } else {
    // find all the arguments positions
    std::unordered_map<std::string, int> dup_args;
    int max_dup = this->FindDuplicateArgs(&dup_args);
    if (max_dup > 1) {
      for (const auto& kv : dup_args) {
        CHECK_EQ(kv.second, 1)
            << " Argument name=\"" << kv.first << "\" occured in "
            << kv.second << " places in the Symbol, "
            << "Keyword argument call is not supported because this duplication.";
      }
    }
    CHECK_EQ(max_dup, 1);
    std::vector<std::pair<DataEntry*, const DataEntry*> > replace_plan;
    std::unordered_set<Node *> visited;
    // replace map stores the existing replacement plan for arguments node
    this->DFSVisit([&nmatched, &visited, &kwargs, &replace_plan](Node *node) {
        // visit all the childs, find possible replacement
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          DataEntry *e = &(node->inputs[i]);
          if (e->source->is_variable()) {
            const DataEntry *target = nullptr;
            auto iter = kwargs.find(e->source->name);
            if (iter != kwargs.end()) {
              target = &(iter->second.heads_[0]);
              // count how many arguments have been matched.
              if (visited.count(e->source.get()) == 0) {
                visited.insert(e->source.get());
                ++nmatched;
              }
              replace_plan.push_back(std::make_pair(e, target));
            }
          }
        }
      });
    if (nmatched == kwargs.size()) {
      for (const auto& kv : replace_plan) {
        *(kv.first) = *(kv.second);
      }
    }
  }
  if (nmatched != kwargs.size()) {
    // Error message handling
    std::vector<std::string> req_args = this->ListArguments();
    std::unordered_set<std::string> keys(req_args.begin(), req_args.end());
    std::ostringstream msg;
    msg << "\nCandidate arguments:\n";
    for (size_t i = 0; i < req_args.size(); ++i) {
      msg << "\t[" << i << ']' << req_args[i] << '\n';
    }
    for (const auto& kv : kwargs) {
      CHECK_NE(keys.count(kv.first), 0)
          << "Keyword Argument " << kv.first << " not found in arguments."
          << msg.str();
    }
  }
}

Symbol Symbol::operator () (const std::vector<Symbol>& args,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(args, name);
  return s;
}

Symbol Symbol::operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                            const std::string& name) const {
  Symbol s = this->Copy();
  s.Compose(kwargs, name);
  return s;
}

bool Symbol::InferShape(std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  return g.InferShape(in_shape, out_shape);
}

Symbol Symbol::Create(OperatorProperty *op)  {
  // use special representation for atomic symbol
  auto node = std::make_shared<Node>(op, "");
  size_t nret = op->NumReturns();
  Symbol s;
  for (uint32_t i = 0; i < nret; ++i) {
    s.heads_.push_back(DataEntry(node, i));
  }
  return s;
}

Symbol Symbol::CreateGroup(const std::vector<Symbol> &symbols) {
  Symbol ret;
  for (const auto &s : symbols) {
    ret.heads_.insert(ret.heads_.end(), s.heads_.begin(), s.heads_.end());
  }
  return std::move(ret);
}

Symbol Symbol::CreateVariable(const std::string &name) {
  Symbol s;
  s.heads_.push_back(DataEntry(std::make_shared<Node>(nullptr, name), 0));
  return std::move(s);
}

void Symbol::ToStaticGraph(StaticGraph *out_graph) const {
  // TODO(bing): Check unique name
  std::vector<Node*> node_order;
  std::unordered_map<Node*, uint32_t> node_index;
  auto &arg_nodes = out_graph->arg_nodes;
  arg_nodes.clear();

  this->DFSVisit([&node_order, &node_index, &arg_nodes](Node *n) {
      uint32_t nid = static_cast<uint32_t>(node_index.size());
      node_index[n] = nid;
      if (n->is_variable()) {
        arg_nodes.push_back(nid);
      }
      node_order.push_back(n);
    });
  // setup nodes
  out_graph->nodes.resize(node_index.size());
  for (uint32_t nid = 0; nid < node_order.size(); ++nid) {
    if (node_order[nid]->op != nullptr) {
      out_graph->nodes[nid].op.reset(node_order[nid]->op->Copy());
    } else {
      out_graph->nodes[nid].op.reset(nullptr);
    }
    out_graph->nodes[nid].name = node_order[nid]->name;
    auto &inputs = out_graph->nodes[nid].inputs;
    inputs.clear();
    for (const DataEntry &src : node_order[nid]->inputs) {
      StaticGraph::DataEntry e;
      e.index = src.index;
      e.source_id = node_index[src.source.get()];
      inputs.push_back(e);
    }
  }
  // setup heads
  out_graph->outputs.clear();
  for (auto &head : heads_) {
    StaticGraph::DataEntry e;
    e.source_id = node_index[head.source.get()];
    e.index = head.index;
    out_graph->outputs.push_back(e);
  }
}
}  // namespace mxnet
