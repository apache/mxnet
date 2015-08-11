/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief symbol of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbol.h>
#include <mxnet/registry.h>
#include <mxnet/static_graph.h>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mxnet {
// copy the symbol
Symbol Symbol::Copy() const {
  std::unordered_map<Node*, std::shared_ptr<Node> > old_new;
  // use DFSVisit to copy all the nodes
  this->DFSVisit([&old_new](Node *node) {
      if (node->sym == nullptr) {
        old_new[node] = std::make_shared<Node>(nullptr, node->name);
      } else {
        old_new[node] =  std::make_shared<Node>(node->sym->Copy(), node->name);
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
  s.head_ = DataEntry(old_new[head_.source.get()], head_.index);
  return s;
}

void Symbol::Print(std::ostream &os) const {
  if (head_.source->is_atomic()) {
    os << "AtomicSymbol "<< " Type:" << head_.source->sym->TypeString() << '\n'
       << "Inputs:";
    std::vector<std::string> args = this->ListArguments();
    for (size_t i = 0; i < args.size(); ++i) {
      os << "\targ[" << i << "]=" << args[i] << "\n";
    }
  } else {
    // use DFSVisit to copy all the nodes
    this->DFSVisit([&os](Node *node) {
        if (node->is_variable()) {
          os << "Variable:" << node->name << '\n';
        } else {
          os << "Name: " << node->name << " Type:" << node->sym->TypeString() << '\n'
             << "Inputs:\n";
          for (size_t i = 0; i < node->inputs.size(); ++i) {
            os << "\targ[" << i << "]=" << node->inputs[i].source->name
               << '(' << node->inputs[i].index << ")\n";
          }
        }
      });
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

void Symbol::Compose(const std::vector<Symbol>& args,
                     const std::string& name) {
  CHECK(!head_.source->is_variable()) << "PlaceHolder cannot be composed";
  head_.source->name = name;
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_NE(args[i].head_.index, -1)
        << "Argument " << i << " is a tuple, scalar is required";
  }
  // positional arguments requires all arguments for now.
  // TODO(bing) consider partial assignments
  if (head_.source->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = head_.source->sym->ListArguments();
    CHECK_EQ(args.size(), req_args.size())
        << "Incorrect number of arguments, requires " << req_args.size()
        << ", provided " << args.size();
    head_.source->inputs.resize(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      head_.source->inputs[i] = args[i].head_;
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
                target = &(args[arg_counter].head_);
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
  CHECK(!head_.source->is_variable()) << "PlaceHolder cannot be composed";
  head_.source->name = name;
  for (const auto& kv : kwargs) {
    CHECK_NE(kv.second.head_.index, -1)
        << "Keyword Argument " << kv.first << " is a tuple, scalar is required";
  }
  size_t nmatched = 0;
  if (head_.source->is_atomic()) {
    // atomic symbol do not have place holder for all the arguments
    std::vector<std::string> req_args = head_.source->sym->ListArguments();
    head_.source->inputs.resize(req_args.size());
    for (size_t i = 0; i < req_args.size(); ++i) {
      auto iter = kwargs.find(req_args[i]);
      if (iter != kwargs.end()) {
        head_.source->inputs[i] = iter->second.head_;

        ++nmatched;
      } else {
        // create a variable node
        // TODO(bing): think of naming convention
        if (name.length() == 0) {
          head_.source->inputs[i] = DataEntry(
              std::make_shared<Node>(nullptr, req_args[i]), 0);
        } else {
          head_.source->inputs[i] = DataEntry(
              std::make_shared<Node>(nullptr, name + '_' + req_args[i]), 0);
        }
      }
    }
    // if things goes wrong recover the old state
    if (nmatched != kwargs.size()) {
      head_.source->inputs.clear();
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
              target = &(iter->second.head_);
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

Symbol Symbol::operator[] (int index) const {
  CHECK_EQ(head_.index, -1) << "Current symbol can't be indexed because it returns a scalar.";
  CHECK_GE(index, 0) << "Symbol only accept nonnegative index";
  Symbol s = *this;
  s.head_.index = index;
  return s;
}

std::vector<std::string> Symbol::ListArguments() const {
  std::vector<std::string> ret;
  if (head_.source->is_atomic()) {
    return head_.source->sym->ListArguments();
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
  return head_.source->sym->ListReturns();
}

Symbol Symbol::Create(AtomicSymbol *atomic_symbol)  {
  // use special representation for atomic symbol
  Symbol s;
  s.head_ = DataEntry(std::make_shared<Node>(atomic_symbol, ""),
                      atomic_symbol->NumReturns() > 1 ? -1 : 0);
  return s;
}

void Symbol::Convert(const std::vector<Symbol> &heads, StaticGraph *out_graph) {
  // TODO(bing): Check unique name
  std::vector<Node*> node_order;
  std::unordered_map<Node*, uint32_t> node_index;
  auto &arg_nodes = out_graph->arg_nodes;
  arg_nodes.clear();

  DFSVisit(heads, [&node_order, &node_index, &arg_nodes](Node *n) {
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
    if (node_order[nid]->sym != nullptr) {
      out_graph->nodes[nid].sym.reset(node_order[nid]->sym->Copy());
    } else {
      out_graph->nodes[nid].sym.reset(nullptr);
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
  for (auto &head : heads) {
    StaticGraph::DataEntry e;
    e.source_id = node_index[head.head_.source.get()];
    if (head.head_.index == -1) {
      int nout = head.head_.source->sym->NumReturns();
      for (int i = 0; i < nout; ++i) {
        e.index = i;
        out_graph->outputs.push_back(e);
      }
    } else {
      e.index = head.head_.index;
      out_graph->outputs.push_back(e);
    }
  }
}
}  // namespace mxnet
