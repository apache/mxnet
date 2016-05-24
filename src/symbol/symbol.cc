/*!
 * Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief symbol of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbolic.h>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "./static_graph.h"

namespace mxnet {

namespace symbol_constants {
const char *kShapeKey = "__shape__";
const char *kNamespaceSeparator = "_";
}  // namespace symbol_constants

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
  /*! \brief source node of the current node */
  std::shared_ptr<Symbol::Node> backward_source_node;
  /*!
   * \brief additional attributes about the node,
   *  Use pointer to save space, as attr can be accessed in a slow way,
   *  not every node will have attributes.
   */
  std::unique_ptr<std::map<std::string, std::string> > attr;
  /*!
    *\brief constructor
    *\param op the OperatorProperty to construct the Node
    *\param name the name of the symbol
   */
  explicit Node(OperatorProperty *op,
                const std::string& name)
      : op(op), name(name) {}
  /*!
    *\brief copy constructor constructor
   */
  explicit Node(const Node& other)
      : name(other.name) {
    if (other.op != nullptr) {
      op.reset(other.op->Copy());
    }
    if (other.attr.get() != nullptr) {
      attr.reset(new std::map<std::string, std::string>(*(other.attr)));
    }
  }
  /*! \return Whether the symbol is atomic */
  inline bool is_atomic() const {
    return inputs.size() == 0 && op != nullptr;
  }
  /*! \return Whether it is unit variable */
  inline bool is_variable() const {
    return op == nullptr && !backward_source_node;
  }
  /*! \return Whether it is backward op */
  inline bool is_backward() const {
    return backward_source_node.get() != nullptr;
  }
};

/*! \return whwther the symbol is atomic */
inline bool Symbol::is_atomic() const {
  return heads_[0].source->is_atomic();
}
// implementation of template functions
template<typename FVisit>
inline void Symbol::DFSVisit(FVisit fvisit) const {
  std::vector<std::pair<const std::shared_ptr<Node>*, uint32_t> > stack;
  std::unordered_set<Node*> visited;
  // put the head into the graph
  for (auto &head : heads_) {
    Node* ptr = head.source.get();
    if (visited.count(ptr) == 0) {
      stack.push_back(std::make_pair(&head.source, 0));
      visited.insert(ptr);
    }
    while (!stack.empty()) {
      std::pair<const std::shared_ptr<Node> *, uint32_t>& back = stack.back();
      if (back.second == back.first->get()->inputs.size()) {
        fvisit(*(back.first));
        stack.pop_back();
      } else {
        std::vector<Symbol::DataEntry>& inputs = back.first->get()->inputs;
        Symbol::DataEntry& input = inputs.at(back.second++);
        Node* ptr = input.source.get();
        if (visited.count(ptr) == 0) {
          stack.push_back(std::make_pair(&input.source, 0));
          visited.insert(ptr);
        }
      }
    }
  }
}

// helper function to handle keyword argument mismatch
// throw approperiate messages
inline void KeywordArgumentMismatch(const char *source,
                                    const std::vector<std::string> &user_args,
                                    const std::vector<std::string> &args) {
  std::unordered_set<std::string> keys(args.begin(), args.end());
  std::ostringstream head, msg;
  msg << "\nCandidate arguments:\n";
  for (size_t i = 0; i < args.size(); ++i) {
    msg << "\t[" << i << ']' << args[i] << '\n';
  }

  for (const auto& key : user_args) {
    if (keys.count(key) == 0) {
      LOG(FATAL) << source
                 << "Keyword argument name " << key << " not found."
                 << msg.str();
    }
  }
}

int Symbol::FindDuplicateArgs(std::unordered_map<std::string, int> *out) const {
  out->clear();
  int max_dup = 1;
  this->DFSVisit([out, &max_dup](const std::shared_ptr<Node> &node) {
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
  this->DFSVisit([&old_new](const std::shared_ptr<Node> &node) {
      old_new[node.get()] =  std::make_shared<Node>(*node);
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
    this->DFSVisit([&os](const std::shared_ptr<Node> &node) {
        if (node->is_variable()) {
          os << "Variable:" << node->name << '\n';
        } else {
          std::string type_string;
          if (!node->backward_source_node) {
            type_string = node->op->TypeString();
          } else {
            type_string = node->backward_source_node->op->TypeString();
          }
          os << "Name: " << node->name << " Type:" << type_string << '\n'
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
    this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
        if (node->is_variable()) {
          ret.push_back(node->name);
        }
      });
    return ret;
  }
}

std::vector<std::string> Symbol::ListOutputs() const {
  std::vector<std::string> ret;
  for (auto &head : heads_) {
    if (head.source->is_variable()) {
      ret.push_back(head.source->name);
    } else {
      auto &hname = head.source->name;
      std::string rname;
      if (head.source->is_backward()) {
        rname = head.source->backward_source_node->op->ListArguments()[head.index];
      } else {
        rname = head.source->op->ListOutputs()[head.index];
      }
      if (hname.length() == 0) {
        ret.push_back(std::move(rname));
      } else {
        ret.push_back(hname + '_' + rname);
      }
    }
  }
  return ret;
}

std::vector<std::string> Symbol::ListAuxiliaryStates() const {
  std::vector<std::string> ret;
  if (this->is_atomic()) {
    return heads_[0].source->op->ListAuxiliaryStates();
  } else {
    this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
        if (node->op != nullptr) {
          auto aux_args = node->op->ListAuxiliaryStates();
          if (aux_args.size() > 0) {
            auto &hname = node->name;
            for (auto const &aux : aux_args) {
              ret.push_back(hname + '_' + aux);
            }
          }
        }
      });
    return ret;
  }
}

Symbol Symbol::operator[] (size_t index) const {
  size_t nreturn = NumOutputs();
  CHECK_LT(index, nreturn) << "Symbol only accept nonnegative index";
  if (nreturn == 1) {
    return *this;
  } else {
    Symbol s;
    s.heads_.push_back(heads_[index]);
    return s;
  }
}

Symbol Symbol::GetInternals() const {
  Symbol ret;
  this->DFSVisit([&ret](const std::shared_ptr<Node> &node) {
      Node* n = node.get();
      uint32_t nout;
      if (n->is_variable()) {
        nout = 1;
      } else if (n->is_backward()) {
        nout = static_cast<uint32_t>(n->backward_source_node->inputs.size());
      } else {
        nout = n->op->NumVisibleOutputs();
      }
      for (uint32_t i = 0; i < nout; ++i) {
        ret.heads_.push_back(DataEntry(node, i));
      }
    });
  return ret;
}

// create a default variable name
inline std::string DefaultVarName(const std::string &op_name,
                                  const std::string &arg_name) {
  if (op_name.length() == 0) {
    return arg_name;
  } else {
    return op_name + '_' + arg_name;
  }
}

void Symbol::Compose(const std::vector<Symbol>& args,
                     const std::string& name) {
  // CHECK_EQ(NumOutputs(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_EQ(args[i].NumOutputs(), 1)
        << "Argument " << i << " is a tuple with " <<  args[i].NumOutputs()
        << " elements, scalar is required";
  }
  // positional arguments requires all arguments for now.
  // TODO(bing) consider partial assignments
  if (this->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = heads_[0].source->op->ListArguments();
    CHECK_LE(args.size(), req_args.size())
        << "Incorrect number of arguments, requires " << req_args.size()
        << ", provided " << args.size();
    heads_[0].source->inputs.resize(req_args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      heads_[0].source->inputs[i] = args[i].heads_[0];
    }
    for (size_t i = args.size(); i < req_args.size(); ++i) {
      heads_[0].source->inputs[i] = DataEntry(
          std::make_shared<Node>(nullptr, DefaultVarName(name, req_args[i])), 0);
      // also copy attribute of operator over to automatically created variable
      if (heads_[0].source->attr.get() != nullptr) {
        heads_[0].source->inputs[i].source->attr.reset(
            new std::map<std::string, std::string>(*(heads_[0].source->attr)));
      }
    }
  } else {
    // find all the place holders
    size_t arg_counter = 0;
    std::unordered_map<Node*, const DataEntry*> replace_map;
    std::vector<std::pair<DataEntry*, const DataEntry*> > replace_plan;
    // replace map stores the existing replacement plan for arguments node
    this->DFSVisit([&arg_counter, &replace_map, &replace_plan, &args]
                   (const std::shared_ptr<Node> &node) {
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
  // CHECK_EQ(NumOutputs(), 1) << "Only composition of value function is supported currently";
  CHECK(!heads_[0].source->is_variable()) << "Variable cannot be composed";
  heads_[0].source->name = name;
  for (const auto& kv : kwargs) {
    CHECK_EQ(kv.second.NumOutputs(), 1)
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
        heads_[0].source->inputs[i] = DataEntry(
            std::make_shared<Node>(nullptr, DefaultVarName(name, req_args[i])), 0);
        // also copy attribute of operator over to automatically created variable
        if (heads_[0].source->attr.get() != nullptr) {
          heads_[0].source->inputs[i].source->attr.reset(
              new std::map<std::string, std::string>(*(heads_[0].source->attr)));
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
    this->DFSVisit([&nmatched, &visited, &kwargs, &replace_plan]
                   (const std::shared_ptr<Node> &node) {
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
    std::vector<std::string> keys(kwargs.size());
    std::transform(kwargs.begin(), kwargs.end(), keys.begin(),
                   [](decltype(*kwargs.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.Compose", keys, ListArguments());
  }
}

bool Symbol::GetName(std::string* out) {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.GetName only works for non-grouped symbol";
  }
  *out = node->name;
  return true;
}

void Symbol::SetAttr(const std::string &key, const std::string& value) {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.SetAttr only works for non-grouped symbol";
  }
  if (node->attr.get() == nullptr) {
    node->attr.reset(new std::map<std::string, std::string>());
  }
  (*node->attr)[key] = value;
}

bool Symbol::GetAttr(const std::string& key, std::string* out) {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.GetAttr only works for non-grouped symbol";
  }
  if (node->attr.get() == nullptr) return false;
  auto it = node->attr->find(key);
  if (it == node->attr->end()) return false;
  *out = it->second;
  return true;
}

std::map<std::string, std::string> Symbol::ListAttr() {
  std::map<std::string, std::string> ret;
  this->DFSVisit([&ret](const std::shared_ptr<Node> &n) {
      if (n->attr.get() == nullptr) return;
      for (const auto &it : *(n->attr.get())) {
        ret[n->name + symbol_constants::kNamespaceSeparator + it.first] = it.second;
      }
      // Also propagate attributes of each node to its auxiliary states.
      // this is a hack to enable correct allocation of auxiliary state
      // easily in multiple devices. This behavior should be helpful in current setting,
      // but can be changed when needed in future.
      if (n->op.get() != nullptr) {
        for (const auto& aux : n->op->ListAuxiliaryStates()) {
          for (const auto &it : *(n->attr.get())) {
            ret[n->name + '_'  + aux +
                symbol_constants::kNamespaceSeparator + it.first] = it.second;
          }
        }
      }
    });
  return ret;
}

std::map<std::string, std::string> Symbol::ListAttrShallow() {
  Node* node = heads_[0].source.get();
  for (const DataEntry& e : heads_) {
    CHECK(node == e.source.get())
        << "Symbol.ListAttrShallow only works for non-grouped symbol";
  }
  if (node->attr.get() == nullptr) return std::map<std::string, std::string>();
  return *node->attr.get();
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

Symbol Symbol::Grad(const std::vector<std::string>& wrt) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  uint32_t num_nodes = g.nodes.size();
  std::vector<uint32_t> head_grad_nodes;
  std::vector<StaticGraph::DataEntry> arg_grads;
  // mirror is need to be disabled here.
  std::map<uint32_t, uint32_t> mirror;
  g.MakeBackwardPass(&head_grad_nodes, &arg_grads, &mirror);

  std::vector<std::shared_ptr<Node> > shared_node;
  this->DFSVisit([&shared_node](const std::shared_ptr<Node> &n) {
      shared_node.push_back(n);
    });
  for (std::vector<StaticGraph::Node>::const_iterator it = g.nodes.begin() + num_nodes;
       it != g.nodes.end(); ++it) {
    auto sym_node = std::make_shared<Symbol::Node>(nullptr, it->name);
    if (it->backward_source_id != -1) {
      sym_node->backward_source_node = shared_node[it->backward_source_id];
    }
    shared_node.push_back(sym_node);
    for (auto e : it->inputs) {
      Symbol::DataEntry entry(shared_node[e.source_id], e.index);
      sym_node->inputs.push_back(std::move(entry));
    }
  }
  // make arg lookup dict
  auto arg_list = ListArguments();
  std::unordered_map<std::string, uint32_t> arg_index;
  for (uint32_t i = 0; i < arg_list.size(); ++i) {
    arg_index[arg_list[i]] = i;
  }
  // generate the heads
  Symbol ret;
  for (const std::string& name : wrt) {
    if (arg_index.find(name) != arg_index.end()) {
      uint32_t index = arg_index[name];
      Symbol::DataEntry entry(shared_node[arg_grads[index].source_id], arg_grads[index].index);
      ret.heads_.push_back(entry);
    } else {
      KeywordArgumentMismatch("Symbol.Grad ", wrt, arg_list);
    }
  }
  return ret;
}

bool Symbol::InferShape(std::vector<TShape> *arg_shapes,
                        std::vector<TShape> *out_shapes,
                        std::vector<TShape> *aux_shapes,
                        bool partial_infer) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  return g.InferShape(arg_shapes, out_shapes, aux_shapes, partial_infer);
}

bool Symbol::InferShape(const std::unordered_map<std::string, TShape>& known_arg_shapes,
                        std::vector<TShape> *arg_shapes,
                        std::vector<TShape> *out_shapes,
                        std::vector<TShape> *aux_shapes,
                        bool partial_infer) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  arg_shapes->clear();
  arg_shapes->resize(g.arg_nodes.size(), TShape());
  size_t nmatched = 0;
  for (size_t i = 0; i < g.arg_nodes.size(); ++i) {
    const std::string& name = g.nodes[g.arg_nodes[i]].name;
    auto it = known_arg_shapes.find(name);
    if (it != known_arg_shapes.end()) {
      arg_shapes->at(i) = it->second;
      ++nmatched;
    } else if (g.nodes[g.arg_nodes[i]].is_variable()) {
      arg_shapes->at(i) = g.nodes[g.arg_nodes[i]].get_attr(symbol_constants::kShapeKey, TShape());
    }
  }
  if (nmatched != known_arg_shapes.size()) {
    std::vector<std::string> keys(known_arg_shapes.size());
    std::transform(known_arg_shapes.begin(), known_arg_shapes.end(), keys.begin(),
                   [](decltype(*known_arg_shapes.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.InferShape", keys, ListArguments());
  }
  return g.InferShape(arg_shapes, out_shapes, aux_shapes, partial_infer);
}

bool Symbol::InferType(std::vector<int> *arg_types,
                        std::vector<int> *out_types,
                        std::vector<int> *aux_types) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  return g.InferType(arg_types, out_types, aux_types);
}

bool Symbol::InferType(const std::unordered_map<std::string, int>& known_arg_types,
                        std::vector<int> *arg_types,
                        std::vector<int> *out_types,
                        std::vector<int> *aux_types) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  arg_types->clear();
  arg_types->resize(g.arg_nodes.size(), -1);
  size_t nmatched = 0;
  for (size_t i = 0; i < g.arg_nodes.size(); ++i) {
    const std::string& name = g.nodes[g.arg_nodes[i]].name;
    auto it = known_arg_types.find(name);
    if (it != known_arg_types.end()) {
      arg_types->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_types.size()) {
    std::vector<std::string> keys(known_arg_types.size());
    std::transform(known_arg_types.begin(), known_arg_types.end(), keys.begin(),
                   [](decltype(*known_arg_types.begin())& kv)->std::string { return kv.first; });
    KeywordArgumentMismatch("Symbol.InferType", keys, ListArguments());
  }
  return g.InferType(arg_types, out_types, aux_types);
}


void Symbol::Save(dmlc::JSONWriter *writer) const {
  StaticGraph g;
  this->ToStaticGraph(&g);
  g.Save(writer);
}

void Symbol::Load(dmlc::JSONReader *reader) {
  StaticGraph g;
  g.Load(reader);
  this->FromStaticGraph(g);
}

Symbol Symbol::Create(OperatorProperty *op)  {
  // use special representation for atomic symbol
  auto node = std::make_shared<Node>(op, "");
  size_t nret = op->NumVisibleOutputs();
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
  return ret;
}

Symbol Symbol::CreateVariable(const std::string &name) {
  Symbol s;
  s.heads_.push_back(DataEntry(std::make_shared<Node>(nullptr, name), 0));
  return s;
}

void Symbol::ToStaticGraph(StaticGraph *out_graph) const {
  std::vector<Node*> node_order;
  std::unordered_map<Node*, uint32_t> node_index;
  auto &arg_nodes = out_graph->arg_nodes;
  arg_nodes.clear();

  this->DFSVisit([&node_order, &node_index, &arg_nodes](const std::shared_ptr<Node> &n) {
      uint32_t nid = static_cast<uint32_t>(node_index.size());
      node_index[n.get()] = nid;
      if (n->is_variable()) {
        arg_nodes.push_back(nid);
      }
      node_order.push_back(n.get());
    });
  // setup nodes
  out_graph->nodes.resize(node_index.size());
  for (uint32_t nid = 0; nid < node_order.size(); ++nid) {
    if (node_order[nid]->op != nullptr) {
      out_graph->nodes[nid].op.reset(node_order[nid]->op->Copy());
    } else {
      out_graph->nodes[nid].op.reset(nullptr);
    }
    // backward source
    if (node_order[nid]->backward_source_node) {
      out_graph->nodes[nid].backward_source_id =
          node_index[node_order[nid]->backward_source_node.get()];
    } else {
      out_graph->nodes[nid].backward_source_id = -1;
    }
    if (node_order[nid]->attr.get() != nullptr) {
      out_graph->nodes[nid].attr = *(node_order[nid]->attr);
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
  out_graph->heads.clear();
  for (auto &head : heads_) {
    StaticGraph::DataEntry e;
    e.source_id = node_index[head.source.get()];
    e.index = head.index;
    out_graph->heads.push_back(e);
  }
}

void Symbol::FromStaticGraph(const StaticGraph &graph) {
  std::unordered_map<uint32_t, std::shared_ptr<Node> > nodes;
  std::vector<uint32_t> topo_order = graph.TopoSort();
  // copy ver nodes in topo order
  for (uint32_t nid : topo_order) {
    auto &gnode = graph.nodes[nid];
    auto sym_node = std::make_shared<Symbol::Node>(nullptr, gnode.name);
    if (gnode.op.get() != nullptr) {
      sym_node->op.reset(gnode.op->Copy());
    }
    if (gnode.backward_source_id != -1) {
      sym_node->backward_source_node = nodes.at(gnode.backward_source_id);
    }
    if (gnode.attr.size() != 0) {
      sym_node->attr.reset(new std::map<std::string, std::string>(gnode.attr));
    }
    for (const StaticGraph::DataEntry& e : gnode.inputs) {
      Symbol::DataEntry entry(nodes.at(e.source_id), e.index);
      sym_node->inputs.push_back(std::move(entry));
    }
    nodes[nid] = sym_node;
  }
  // generate the heads
  heads_.clear();
  for (const StaticGraph::DataEntry& e : graph.heads) {
    Symbol::DataEntry entry(nodes.at(e.source_id), e.index);
    heads_.push_back(std::move(entry));
  }
}
}  // namespace mxnet
