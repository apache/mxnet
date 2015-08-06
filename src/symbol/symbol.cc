/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief symbol of mxnet
 */
#include <dmlc/logging.h>
#include <mxnet/symbol.h>
#include <mxnet/registry.h>
#include <mxnet/static_graph.h>
#include <iterator>

namespace mxnet {

void Symbol::FindArgUsers() {
  arg_users_.reset(new std::vector<std::pair<Node*, int> >);
  // depth first traversing
  std::vector<std::pair<Node*, size_t> > stk;
  stk.push_back({head_.get(), 0});
  while (!stk.empty()) {
    std::pair<Node*, size_t>& back = stk.back();
    if (back.first->in_symbol_.size() == back.second) {
      stk.pop_back();
    } else {
      Node* next_level = back.first->in_symbol_[back.second].get();
      if (next_level->sym_) {
        stk.push_back({next_level, 0});
      } else {  // back uses next_level which is a placeholder
        arg_users_->push_back({back.first, back.second});
      }
      back.second += 1;
    }
  }
}

Symbol Symbol::Copy() const {
  Symbol s;
  std::unordered_map<Node*, std::shared_ptr<Node> > old_new;
  std::vector<Node*> stk;
  stk.push_back(head_.get());
  // copy nodes
  while (!stk.empty()) {
    Node* back = stk.back();
    stk.pop_back();
    if (old_new.count(back) == 0) {
      if (back->sym_) {
        old_new[back] = std::make_shared<Node>(back->sym_->Copy(), back->name_);
      } else {
        old_new[back] = std::make_shared<Node>(nullptr, back->name_);
      }
    }
    for (const std::shared_ptr<Node>& n : back->in_symbol_) {
      if (old_new.count(n.get()) == 0) {
        stk.push_back(n.get());
      }
    }
  }
  // connect nodes
  for (auto kv : old_new) {
    for (const std::shared_ptr<Node>& n : kv.first->in_symbol_) {
      kv.second->in_symbol_.push_back(old_new[n.get()]);
    }
  }
  s.head_ = old_new[this->head_.get()];
  // copy arg_users_
  if (arg_users_) {
    s.arg_users_.reset(new std::vector<std::pair<Node*, int> >);
    std::transform(arg_users_->begin(), arg_users_->end(), std::back_inserter(*s.arg_users_),
        [&old_new](const std::pair<Node*, int>& n) -> std::pair<Node*, int> {
          return { old_new[n.first].get(), n.second };
        });
  }
  return s;
}

Symbol Symbol::operator () (const std::vector<Symbol>& args) const {
  Symbol s = this->Copy();
  if (!s.arg_users_) {  // if arg_users_ has not been populated
    s.FindArgUsers();
  }
  CHECK_LT(args.size(), s.arg_users_->size()) << "Too many args, requires " << s.arg_users_->size()
      << " provided " << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    const std::pair<Node*, int>& arg_user = (*s.arg_users_)[i];
    arg_user.first->in_symbol_[arg_user.second] = args[i].head_;
    CHECK_NE(args[i].index_, -1) << "Argument " << i << " is a tuple, scalar is required";
    arg_user.first->in_index_[arg_user.second] = args[i].index_;
  }
  s.arg_users_.reset();
  return s;
}

Symbol Symbol::operator () (const std::unordered_map<std::string, Symbol>& kwargs) const {
  Symbol s = this->Copy();
  if (!s.arg_users_) {  // if arg_users_ has not been populated
    s.FindArgUsers();
  }
  CHECK_LT(kwargs.size(), s.arg_users_->size()) << "Too many args, requires "
      << s.arg_users_->size() << " provided " << kwargs.size();
  for (size_t i = 0; i < s.arg_users_->size(); ++i) {
    const std::pair<Node*, int>& arg_user = (*s.arg_users_)[i];
    const std::string& name = arg_user.first->name_;
    if (!(name == "") && kwargs.count(name) != 0) {
      const Symbol& bind = kwargs.at(name);
      arg_user.first->in_symbol_[arg_user.second] = bind.head_;
      CHECK_NE(bind.index_, -1) << "Argument " << name << " is a tuple, scalar is required";
      arg_user.first->in_index_[arg_user.second] = bind.index_;
    }
  }
  s.arg_users_.reset();
  // TODO(linmin): report error if kwargs contains non-existing keys
  return s;
}

Symbol Symbol::operator[] (int index) const {
  CHECK_EQ(index_, -1) << "Current symbol can't be indexed because it returns a scalar.";
  Symbol s = *this;
  s.index_ = index;
  return s;
}

std::vector<std::string> Symbol::ListArgs() {
  std::vector<std::string> ret;
  if (!arg_users_) {
    FindArgUsers();
  }
  std::transform(arg_users_->begin(), arg_users_->end(), std::back_inserter(ret),
      [&](const std::pair<Node*, int>& n) -> std::string {
        return n.first->in_symbol_[n.second]->name_;
      });
  return ret;
}

Symbol Symbol::Create(AtomicSymbol *atomic_symbol)  {
  Symbol s;
  std::vector<std::string> args = atomic_symbol->DescribeArguments();
  std::vector<std::string> rets = atomic_symbol->DescribeReturns();
  // set head_
  s.head_ = std::make_shared<Node>(atomic_symbol, "");
  // set index_
  s.index_ = rets.size() > 1 ? -1 : 0;
  // set head_->in_index_
  s.head_->in_index_ = std::vector<int>(args.size(), 0);
  // set head_->in_symbol_
  for (auto name : args) {
    s.head_->in_symbol_.push_back(std::make_shared<Node>(nullptr, name));
  }
  // set head_->out_shape_
  s.head_->out_shape_ = std::vector<TShape>(rets.size());
  return s;
}

Symbol Symbol::Create(const std::string& type_name,
                      const std::vector<std::pair<std::string, std::string> >& param) {
  const AtomicSymbolEntry *entry = Registry<AtomicSymbolEntry>::Find(type_name);
  CHECK_NE(entry, NULL) << type_name << " is not a valid Symbol type";
  AtomicSymbol* atomic_symbol = (*entry)();
  for (auto p : param) {
    atomic_symbol->SetParam(p.first.c_str(), p.second.c_str());
  }
  return Create(atomic_symbol);
}

StaticGraph Symbol::ToStaticGraph() {
  StaticGraph graph;
  Dfs(this->head_, graph);
  return graph;
}

CompositeOperator* Symbol::Bind(Context ctx, const std::unordered_map<std::string, NArray>& in) {
  StaticGraph graph = this->ToStaticGraph();
  return NULL;
  //TODO: pass the graph and in to initlialize a composite op.
}

void Symbol::Dfs(const std::shared_ptr<Node> node, StaticGraph& graph) {
  int id = graph.FindNodeByName(node->name_, node->sym_);
  for (size_t i = 0; i < node->in_symbol_.size(); ++i) {
    std::shared_ptr<Node> parent = node->in_symbol_[i];
    int parent_id = graph.FindNodeByName(parent->name_, parent->sym_);
    graph.connected_graph[parent_id].push_back(id);
    graph.output_index[parent_id].push_back(node->in_index_[i]);
    if (parent->sym_) {
      Dfs(parent, graph);
    }
  }
}

}  // namespace mxnet
