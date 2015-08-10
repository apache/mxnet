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

inline void Symbol::Toposort(const std::vector<DataEntry> &heads,
                            std::vector<Node*> *ret) {
  std::unordered_map<Node*, int> out_degree;
  std::queue<Node*> queue;
  ret->clear();
  size_t idx = 0;
  DFSVisit(heads,
          [&out_degree](Node* node) {
      for (auto &entry : node->inputs) {
        Node *ptr = entry.source.get();
        auto iter = out_degree.find(ptr);
        if (iter == out_degree.end()) {
          out_degree[ptr] = 0;
        } else {
          iter->second += 1;
        }
      }
    });
  for (auto &entry : heads) {
    queue.push(entry.source.get());
  }
  idx = out_degree.size();
  ret->resize(idx);
  --idx;
  while (queue.size() > 0) {
    Node *node = queue.front();
    queue.pop();
    ret->at(idx--) = node;
    for (auto it = node->inputs.rbegin(); it != node->inputs.rend(); ++it) {
      Node *ptr = it->source.get();
      out_degree[ptr] -= 1;
      if (out_degree[ptr] == 0) {
        queue.push(ptr);
      }
    }
  }
}

bool Symbol::InferShape(std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape) {
  bool success = true;
  StaticGraph graph;
  auto input_args = this->ListArguments();
  std::vector<Symbol> tmp_arg = {*this};
  CHECK(in_shape->size() == input_args.size()) << "Input shape should be same to arguments";
  out_shape->clear();
  Convert(tmp_arg, &graph);
  for (size_t i = 0; i < in_shape->size(); ++i) {
    graph.nodes[graph.in_args_node_id[i]].in_shape.push_back(in_shape->at(i));
  }
  for (auto &nd : graph.nodes) {
    success &= nd.sym->InferShape(&nd.in_shape, &nd.out_shape);
  }
  //  copy result back
  for (size_t i = 0; i < in_shape->size(); ++i) {
    in_shape->at(i) = graph.nodes[graph.in_args_node_id[i]].in_shape[0];
  }
  for (auto i : graph.return_node_id) {
    for (auto sp : graph.nodes[i].out_shape) {
      out_shape->push_back(sp);
    }
  }
  return success;
}

std::vector<std::string> Symbol::ListReturns() const {
  return head_.source->sym->ListReturns();
}

Symbol Symbol::Create(AtomicSymbol *atomic_symbol)  {
  // use special representation for atomic symbol
  Symbol s;
  s.head_ = DataEntry(std::make_shared<Node>(atomic_symbol, ""),
                      atomic_symbol->ListReturns().size() > 1 ? -1 : 0);
  return s;
}

void Symbol::Convert(const std::vector<Symbol> &heads, StaticGraph *out_graph) {
  // TODO(bing): Check unique name
  std::vector<Node*> nodes;
  std::unordered_map<Node*, int> node_id_dic;
  std::vector<DataEntry> arg(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    arg[i] = heads[i].head_;
  }
  Toposort(arg, &nodes);
  out_graph->nodes.resize(nodes.size());
  //  set up dict
  for (size_t i = 0; i < nodes.size(); ++i) {
    node_id_dic[nodes[i]] = i;
  }
  //  copy
  for (size_t i = 0; i < nodes.size(); ++i) {
    out_graph->name_id_map[nodes[i]->name] = i;
    if (!nodes[i]->is_variable()) {
      out_graph->nodes[i].sym.reset(nodes[i]->sym->Copy());
    }
    out_graph->nodes[i].name = nodes[i]->name;
    for (auto &entry : nodes[i]->inputs) {
      out_graph->nodes[i].inputs_index.push_back(node_id_dic[entry.source.get()]);
      out_graph->nodes[node_id_dic[entry.source.get()]].outputs_index.push_back(i);
    }
  }
  // set input map
  for (auto const &head : heads) {
    auto input_args = head.ListArguments();
    out_graph->in_args_node_id.resize(input_args.size());
    for (size_t i = 0; i < input_args.size(); ++i) {
      out_graph->in_args_node_id[i] = out_graph->name_id_map[input_args[i]];
    }
  }
  // set output map
  out_graph->return_node_id.resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    out_graph->return_node_id[i] = out_graph->name_id_map[heads[i].head_.source->name];
  }
}

}  // namespace mxnet
