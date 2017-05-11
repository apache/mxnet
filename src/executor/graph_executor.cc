/*!
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <vector>
#include <algorithm>

#include "./exec_pass.h"
#include "./graph_executor.h"
#include "../engine/profiler.h"

namespace mxnet {
namespace exec {
GraphExecutor::~GraphExecutor() {
  for (auto& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
  // clean up seg ops
  for (auto& seg : cached_seg_opr_) {
    if (seg.opr != nullptr) {
      Engine::Get()->DeleteOperator(seg.opr);
    }
  }
}

void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray>& head_grads) {
  const auto& idx = graph_.indexed_graph();
  if (num_forward_inputs_ != idx.input_nodes().size()) {
    for (size_t i = 0; i < head_grad_array_.size(); ++i) {
      if (!head_grad_array_[i].is_none()) {
        CHECK(i < head_grads.size() && !head_grads[i].is_none())
            << "Because the last operator is not Loss function, "
            << "head_gradient is required when calling backward. "
            << "If you are attempting to minimize the output as "
            << "an objective, please modify your network and "
            << "pass it through the make_loss symbol.";
        CopyFromTo(head_grads[i], &(head_grad_array_[i]));
      }
    }
  }
  RunOps(true, num_forward_nodes_, idx.num_nodes());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  nnvm::Symbol s; s.outputs = graph_.outputs;
  s.Print(os);
  // message to be backward compatible with the memonger
  size_t total_bytes = graph_.GetAttr<size_t>("storage_allocated_bytes");
  os << "Total " << (total_bytes >> 20UL) <<" MB allocated\n";
  os << "Total " << 11 << " TempSpace resource requested\n";
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback) {
  CHECK(callback) << "invalid callback";
  monitor_callback_ = callback;
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  return output_arrays_;
}

nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like) {
  static const Op* id_like = Op::Get("_identity_with_attr_like_rhs");
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = id_like;
  n->attrs.name = src.node->attrs.name + "_id";
  n->inputs = {src, like};
  return nnvm::NodeEntry{n, 0, 0};
}

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  static const Op* zeros_like_op = Op::Get("zeros_like");

  if (v.size() == 0) {
    nnvm::NodePtr ng = nnvm::Node::Create();
    ng->attrs.op = zeros_op;
    ng->attrs.name = "zeros";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry{ng, 0, 0};
  }

  // remove zero in the sum. at least keep 1.
  size_t begin = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i].node->op() != zeros_op && v[i].node->op() != zeros_like_op) {
      if (begin != i) {
        v[begin] = std::move(v[i]);
      }
      ++begin;
    }
  }
  if (begin == 0) begin = 1;
  v.resize(begin);

  if (v.size() == 1) {
    return std::move(v[0]);
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::NodePtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry{sum_node, 0, 0};
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
        v[i].node->control_deps.push_back(ret.node);

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::NodePtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry{x, 0, 0};
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::NodePtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

nnvm::Graph GraphExecutor::InitFullGraph(
    nnvm::Symbol symbol,
    const std::vector<OpReqType>& grad_req_type,
    const std::vector<NDArray>& arg_grad_store) {
  using nnvm::NodePtr;
  using nnvm::NodeEntry;
  // initial information
  num_forward_outputs_ = symbol.outputs.size();
  num_forward_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();

  nnvm::Graph g;
  g.outputs = symbol.outputs;
  bool need_grad = false;
  for (OpReqType req : grad_req_type) {
    if (req != kNullOp) need_grad = true;
  }
  if (!need_grad) return g;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    NodeEntry ngrad{nnvm::Node::Create(), 0, 0};
    head_grad_entry_.emplace_back(AttrHint(ngrad, g.outputs[i]));
    head_grad_map_[ngrad.node.get()] = i;
  }
  std::vector<NodePtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  std::vector<NodeEntry> xs;
  for (size_t i = 0; i < grad_req_type.size(); ++i) {
    if (grad_req_type[i] != kNullOp) {
      grad_store_.emplace_back(
          std::make_pair(grad_req_type[i], arg_grad_store[i]));
      xs.emplace_back(NodeEntry{args[i], 0, 0});
    }
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "__force_mirroring__", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "CuDNNBatchNorm") return false;
    return true;
  };

  std::vector<const nnvm::Op*> zero_ops;
  zero_ops.push_back(nnvm::Op::Get("zeros_like"));
  zero_ops.push_back(nnvm::Op::Get("_zeros"));

  // take gradient
  nnvm::Graph g_grad = nnvm::pass::Gradient(
      g, symbol.outputs, xs, head_grad_entry_,
      AggregateGradient, need_mirror, nullptr,
      zero_ops);
  CHECK_EQ(g_grad.outputs.size(), xs.size());
  for (const auto &e : g_grad.outputs) {
    g.outputs.push_back(e);
  }
  return g;
}

// pass to assign context to the graph
Graph AssignContext(Graph g,
                    const Context& default_ctx,
                    const std::map<std::string, Context>& ctx_map,
                    const std::vector<NDArray>& in_args,
                    const std::vector<std::pair<OpReqType, NDArray> >& grad_store,
                    const std::vector<NDArray>& aux_states,
                    size_t num_forward_inputs,
                    size_t num_forward_outputs) {
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
  // default use default context.
  if (ctx_map.size() == 0) {
    g.attrs["context"] = std::make_shared<nnvm::any>(
        ContextVector(idx.num_nodes(), default_ctx));
    for (const auto& x : in_args) {
      CHECK(x.ctx() == default_ctx)
        << "Input array is in " << x.ctx() << " while binding with ctx=" << default_ctx
        << ". All arguments must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    for (const auto& x : grad_store) {
      CHECK(x.second.ctx() == default_ctx)
        << "Gradient array is in " << x.second.ctx() << " while binding with ctx="
        << default_ctx << ". All gradients must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    return g;
  }
  // otherwise, use context assignment.
  std::map<Context, int> ctx2id;
  std::vector<Context> ctx_list;
  nnvm::DeviceVector device(idx.num_nodes(), -1);
  nnvm::DeviceAssignMap device_map;

  for (auto &kv : ctx_map) {
    if (ctx2id.count(kv.second) == 0) {
      ctx2id[kv.second] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(kv.second);
    }
    device_map[kv.first] = ctx2id.at(kv.second);
  }

  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    Context ctx;
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      ctx = aux_states[aux_top].ctx();
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      ctx = in_args[arg_top].ctx();
      ++arg_top;
    }
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    device[nid] = ctx2id.at(ctx);
  }
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i) {
    const uint32_t nid = idx.outputs()[i].node_id;
    Context ctx = grad_store[i - num_forward_outputs].second.ctx();
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    int devid = ctx2id.at(ctx);
    if (device[nid] != -1) {
      CHECK_EQ(device[nid], devid) << "device of same output not equal to each other";
    } else {
      device[nid] = devid;
    }
  }
  g.attrs["device"] = std::make_shared<dmlc::any>(std::move(device));
  g = nnvm::pass::PlaceDevice(g, "__ctx_group__", device_map, "_CrossDeviceCopy");
  const auto& assigned_device = g.GetAttr<nnvm::DeviceVector>("device");

  ContextVector vcontext;
  for (size_t i = 0; i < assigned_device.size(); ++i) {
    if (assigned_device[i] == -1) {
      vcontext.push_back(default_ctx);
    } else {
      vcontext.push_back(ctx_list[assigned_device[i]]);
    }
  }
  g.attrs["context"] = std::make_shared<nnvm::any>(std::move(vcontext));
  return g;
}

void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_type,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec,
                         const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  nnvm::Graph g = InitGraph(symbol, default_ctx,
                            ctx_map, in_args, arg_grad_store,
                            grad_req_type, aux_states, feed_dict);
  g.attrs["saved_opr"] = std::make_shared<nnvm::any>(std::move(saved_opr_));
  g = AttachOpExecs(g);
  g = AttachOpResources(g);
  graph_ = std::move(g);
  if (shared_exec != nullptr) {
    this->InitDataEntryMemory(&(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_));
  } else {
    this->InitDataEntryMemory(nullptr);
  }
  {
    // initialize output arrays
    auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      auto& e = idx.outputs()[i];
      output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
    }
    // initialize head gradient array
    head_grad_array_.resize(symbol.outputs.size());
    for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
      uint32_t nid = idx.input_nodes().at(i);
      uint32_t oid = head_grad_map_.at(idx[nid].source);
      head_grad_array_[oid] = data_entry_[idx.entry_id(nid, 0)];
    }
  }
  this->InitCachedOps();
  this->InitOpSegs();
}

Graph GraphExecutor::InitGraph(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& ctx_map,
                               const std::vector<NDArray>& in_args,
                               const std::vector<NDArray>& arg_grad_store,
                               const std::vector<OpReqType>& grad_req_type,
                               const std::vector<NDArray>& aux_states,
                               const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  // setup gradient
  nnvm::Graph g = InitFullGraph(symbol, grad_req_type, arg_grad_store);
  g = AssignContext(g, default_ctx, ctx_map,
                    in_args,
                    grad_store_,
                    aux_states,
                    num_forward_inputs_,
                    num_forward_outputs_);
  const auto& idx = g.indexed_graph();
  // get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_forward_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  // Setup data entry, shape and type.
  data_entry_.resize(idx.num_node_entries());
  auto mutable_nodes = idx.mutable_input_nodes();
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_types;
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    if (mutable_nodes.count(nid)) {
      CHECK_LT(aux_top, aux_states.size());
      data_entry_[idx.entry_id(nid, 0)] = aux_states[aux_top];
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_types.push_back(aux_states[aux_top].dtype());
      ++aux_top;
    } else {
      CHECK_LT(arg_top, in_args.size());
      data_entry_[idx.entry_id(nid, 0)] = in_args[arg_top];
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_types.push_back(in_args[arg_top].dtype());
      ++arg_top;
    }
  }
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    data_entry_[idx.entry_id(idx.outputs()[j])]
        = grad_store_[j - num_forward_outputs_].second;
  }
  arg_shapes.resize(idx.input_nodes().size(), TShape());
  arg_types.resize(idx.input_nodes().size(), -1);
  // other initializations
  g = nnvm::pass::InferShape(g, arg_shapes, "__shape__");
  g = nnvm::pass::InferType(g, arg_types, "__dtype__");

  {
    // memory allocator
    const int kBadStorageID = -1;
    const int kExternalStorageID = -2;
    nnvm::StorageVector arg_storage_id(idx.num_node_entries(), kBadStorageID);
    for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
      arg_storage_id[idx.entry_id(idx.outputs()[j])] = kExternalStorageID;
    }
    for (const auto& kv : feed_dict) {
      uint32_t eid = idx.entry_id(kv.first);
      data_entry_[eid] = kv.second;
      arg_storage_id[eid] = kExternalStorageID;
    }
    g.attrs["storage"] = std::make_shared<dmlc::any>(std::move(arg_storage_id));
    g = nnvm::ApplyPass(g, "PlanMemory");
  }
  g = DetectInplaceAddTo(g);
  return g;
}

// initialize the memory of each entries
void GraphExecutor::InitDataEntryMemory(std::vector<NDArray>* shared_pool) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());
  CHECK_EQ(data_entry_.size(), vshape.size());
  std::vector<Context> data_context(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    for (uint32_t i = 0; i < idx[nid].source->num_outputs(); ++i) {
      data_context[idx.entry_id(nid, i)] = vctx[nid];
    }
  }

  // information about the pool
  using PoolEntry = std::pair<Context, size_t>;
  std::vector<PoolEntry> pool_info;

  // assign array to head gradient
  for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) {
    uint32_t nid = idx.input_nodes().at(i);
    uint32_t oid = head_grad_map_.at(idx[nid].source);
    uint32_t eid = idx.entry_id(idx.outputs()[oid]);
    CHECK_NE(vshape[eid].ndim(), 0U);
    CHECK_NE(vdtype[eid], -1);
    data_entry_[idx.entry_id(nid, 0)] =
        NDArray(vshape[eid], data_context[eid], false, vdtype[eid]);
  }
  // get maximum bytes in each pool
  for (size_t i = 0; i < vshape.size(); ++i) {
    if (!data_entry_[i].is_none()) continue;
    size_t bytes = vshape[i].Size() * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];
    if (storage_id < 0) continue;
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_info.size()) {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0)});
    }
    PoolEntry& info = pool_info[sid];
    if (info.second == 0) {
      info = PoolEntry{data_context[i], bytes};
    } else {
      info.second = std::max(info.second, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  if (shared_pool != nullptr) {
    for (const NDArray& nd : *shared_pool) {
      size_t bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
      free_pool.insert(std::make_pair(bytes, nd));
    }
  }
  // remake the data pool
  data_pool_.clear();
  data_pool_.resize(pool_info.size());

  // sort the pool info the descending order before allocating memory
  std::vector<size_t> sorted_pool_index;
  for (size_t i = 0; i < pool_info.size(); i++) {
    sorted_pool_index.push_back(i);
  }
  auto pool_comparator = [&pool_info](int lhs, int rhs){
    return pool_info[lhs].second > pool_info[rhs].second;
  };
  std::sort(sorted_pool_index.begin(), sorted_pool_index.end(), pool_comparator);

  for (size_t i : sorted_pool_index) {
    const Context& ctx = pool_info[i].first;
    size_t bytes = pool_info[i].second;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) {
      if (it->second.ctx() == ctx && it->first >= bytes) {
        data_pool_[i] = it->second;
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<index_t>::max());
      // allocate float arrays
      TShape shape{index_t(nword)};
      NDArray nd(shape, ctx);
      data_pool_[i] = nd;
      // put the new allocated arrays to shared pool
      if (shared_pool != nullptr)  {
        shared_pool->push_back(nd);
      }
    }
  }
  CHECK_EQ(data_pool_.size(), pool_info.size());
  // assign the data entries

  for (size_t i = 0; i < data_entry_.size(); ++i) {
    // avoid pre-allocated arrays
    if (!data_entry_[i].is_none()) continue;
    // assign allocated array by storage id
    int storage_id = vstorage[i];
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op yet";
    const NDArray& src = data_pool_.at(storage_id);
    data_entry_[i] = src.AsArray(vshape[i], vdtype[i]);
  }
}


void GraphExecutor::InitCachedOps() {
  // get the graph
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace =
      graph_.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs =
      graph_.GetAttr<OpExecVector>("op_execs");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  const auto& addto_entry = graph_.GetAttr<std::vector<int> >("addto_entry");
  const auto& skip_plus_node = graph_.GetAttr<std::vector<int> >("skip_plus_node");

  op_nodes_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
#if MXNET_USE_PROFILER
    op_nodes_[nid].opr_name = inode.source->op()->name.c_str();
#else
    op_nodes_[nid].opr_name = nullptr;
#endif
    if (skip_plus_node.at(nid)) {
      op_nodes_[nid].skip_exec_node = true; continue;
    }

    op_nodes_[nid].exec = op_execs[nid];
    op_nodes_[nid].ctx = vctx[nid];
    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0U);
    CHECK_EQ(exec->out_array.size(), 0U);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(data_entry_[eid]);
      if (addto_entry.at(eid) != 0) {
        exec->req.push_back(kAddTo);
      } else if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
      } else if (vstorage_inplace[eid] == -2) {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
  }
  // Note that this modifies the requirment of kWriteInplace
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
    auto& e = idx.outputs()[j];
    op_nodes_[e.node_id].exec->req[e.index] =
        grad_store_[j - num_forward_outputs_].first;
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (op_nodes_[nid].skip_exec_node) continue;
    auto& exec = op_nodes_[nid].exec;
    bool is_async = op_nodes_[nid].exec->exec_type() == Operator::kAsync;
    bool is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;

    // the variables
    std::vector<Engine::VarHandle> use_vars, mutate_vars;
    for (size_t i = 0; i < exec->in_array.size(); ++i) {
      auto& nd = exec->in_array[i];
      use_vars.push_back(nd.var());
    }
    for (auto& r : exec->op_ctx.requested) {
      mutate_vars.push_back(r.var);
    }
    for (auto& nd : exec->out_array) {
      mutate_vars.push_back(nd.var());
    }
    // dedup vars
    Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
    // all vars include both mutate vars and use vars
    std::vector<Engine::VarHandle> all_vars(use_vars);
    std::copy(mutate_vars.begin(), mutate_vars.end(),
              std::inserter(all_vars, all_vars.end()));
    // setup exec vars
    Engine::Get()->PushSync([exec](RunContext rctx) {
        exec->Setup();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, 0,
      PROFILER_MESSAGE("SetupExec"));
    auto exec_fun = [exec, is_async, is_gpu] (
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
      if (is_async) {
        exec->op_ctx.async_on_complete = on_complete;
      }
      exec->Run(ctx);
      // call on complete only if it is async op
      if (!is_async) {
        if (is_gpu) {
        #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();
        #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
        }
        on_complete();
      }
    };
    // setup the vars
    op_nodes_[nid].cached_opr = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
        PROFILER_MESSAGE(op_nodes_[nid].opr_name));
    op_nodes_[nid].mutate_vars = mutate_vars;
    op_nodes_[nid].use_vars = use_vars;
  }
}

void GraphExecutor::InitOpSegs() {
  size_t total_num_nodes = graph_.indexed_graph().num_nodes();
  cached_seg_opr_.clear();
  CachedSegOpr p;
  cached_seg_opr_.resize(total_num_nodes, p);
  if (monitor_callback_) return;

  // Generate segments based on the graph structure
  bool prefer_bulk_exec_inference = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_INFERENCE", true);
  if (prefer_bulk_exec_inference && num_forward_nodes_ == total_num_nodes) {
    // bulk the whole graph for inference
    cached_seg_opr_[0] = this->CreateCachedSegOpr(0, num_forward_nodes_);
    return;
  }

  // Whether to perform bulk exec for training
  bool prefer_bulk_exec = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_TRAIN", 1);
  // The maximum number of node in a segment executed in bulk
  size_t num_nodes_threshold = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15);
  // create forward segments for training
  if (prefer_bulk_exec > 0) {
    size_t topo_start = 0;
    for (size_t nid = 0; nid < num_forward_nodes_; nid++) {
      auto &node = graph_.indexed_graph()[nid].source;
      auto &op_node = op_nodes_[nid];
      // check if the segment relies on external input, or exceeds maxinum number of node,
      // or requires async ops
      if (node->is_variable() || nid - topo_start > num_nodes_threshold ||
          op_node.exec->exec_type() != Operator::kSync) {
        // create a new segment for the previous nodes if the current one cannot be bulked
        cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
        topo_start = nid + 1;
      }
    }
    // the last segmenet
    if (topo_start != num_forward_nodes_) {
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, num_forward_nodes_);
    }
  }

  // create backward segments for training
  if (prefer_bulk_exec) {
    // get all gradient variables
    std::unordered_set<engine::VarHandle> grad_vars;
    for (auto &kv : grad_store_) {
      grad_vars.insert(kv.second.var());
    }
    auto &idx = graph_.indexed_graph();
    size_t topo_start = num_forward_nodes_;
    for (size_t nid = num_forward_nodes_; nid < total_num_nodes; nid++) {
      auto &op_node = op_nodes_[nid];
      if (op_node.skip_exec_node || op_node.exec == nullptr) {
        continue;
      }
      if (idx[nid].source->is_variable() || nid - topo_start > num_nodes_threshold ||
          op_node.exec->exec_type() != Operator::kSync) {
        cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
        topo_start = nid + 1;
      } else {
        // If it produces output gradient, don't include it in the segment
        bool output_gradient = false;
        for (auto &out_arr : op_node.exec->out_array) {
          if (grad_vars.find(out_arr.var()) != grad_vars.end()) {
            output_gradient = true;
          }
        }
        if (output_gradient) {
          cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
          topo_start = nid + 1;
        }
      }
    }
    // last segment for backward
    if (topo_start < total_num_nodes) {
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, total_num_nodes);
    }
  }
  return;
}

void GraphExecutor::ExecuteMonCallback(size_t nid) {
  static const auto& flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  const auto& idx = graph_.indexed_graph();
  std::vector<std::string> output_names;
  OpNode& opnode = op_nodes_[nid];
  const auto& inode = idx[nid];
  const auto& node = idx[nid].source;
  if (flist_outputs.count(node->op())) {
    output_names = flist_outputs[node->op()](node->attrs);
  } else {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      output_names.emplace_back(std::to_string(i));
    }
  }
  for (index_t i = 0; i < opnode.exec->out_array.size(); ++i) {
    NDArray *cpy = new NDArray(opnode.exec->out_array[i]);
    std::string name = inode.source->attrs.name + "_" + output_names[i];
    this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  // Update context
  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    OpNode& opnode = op_nodes_[nid];
    if (opnode.skip_exec_node) continue;
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    opnode.exec->op_ctx.is_train = is_train;
  }

  // Push Ops
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    auto seg_op = cached_seg_opr_[nid];
    // Check segments first
    if (monitor_callback_ == nullptr && seg_op.opr != nullptr && seg_op.topo_end <= topo_end) {
#if MXNET_USE_PROFILER
      bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
      bool profiling = false;
#endif
      Engine::Get()->Push(seg_op.opr, seg_op.ctx, 0, profiling);
      nid = seg_op.topo_end - 1;
      continue;
    }
    // Normal mode
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    if (op_nodes_[nid].skip_exec_node) continue;
    opnode.exec->op_ctx.is_train = is_train;
    if (opnode.exec->exec_type() == Operator::kCrossDeviceCopy) {
      CHECK_EQ(inode.inputs.size(), 1U);
      CHECK_EQ(opnode.exec->in_array.size(), 1U);
      CHECK_EQ(opnode.exec->out_array.size(), 1U);
      CopyFromTo(opnode.exec->in_array[0], &(opnode.exec->out_array[0]));
    } else if (opnode.cached_opr != nullptr) {
#if MXNET_USE_PROFILER
      bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
      bool profiling = false;
#endif
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);
    } else {
      LOG(FATAL) << "Not accessed";
    }
    // Monitor callbacks
    if (monitor_callback_) {
      ExecuteMonCallback(nid);
    }
  }
}

GraphExecutor::CachedSegOpr GraphExecutor::CreateCachedSegOpr(size_t topo_start, size_t topo_end) {
  std::vector<Engine::VarHandle> use_vars;
  std::vector<Engine::VarHandle> mutate_vars;
  Context *pctx = nullptr;
  GraphExecutor::CachedSegOpr ret;
  ret.topo_start = topo_start;
  ret.topo_end = topo_end;
  auto& exec_list = ret.exec_list;
  // invalid segment
  if (topo_end <= topo_start) {
    return ret;
  }
#if MXNET_USE_PROFILER
  std::string opr_names = "[";
#else
  std::string opr_names = "Bulk Execution";
#endif

  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    std::vector<Engine::VarHandle> all_vars;
    const auto& inode = idx[nid];
    OpNode& op_node = op_nodes_[nid];
    if (op_node.skip_exec_node) continue;
    if (inode.source->is_variable()) continue;
    if (op_node.exec->exec_type() != Operator::kSync) {
      return ret;
    }
    if (pctx == nullptr) pctx = &(op_node.ctx);
    if (*pctx != op_node.ctx) {
      return ret;
    }
    auto& exec = op_nodes_[nid].exec;
    std::copy(op_node.mutate_vars.begin(), op_node.mutate_vars.end(),
              std::inserter(mutate_vars, mutate_vars.end()));
    std::copy(op_node.use_vars.begin(), op_node.use_vars.end(),
              std::inserter(use_vars, use_vars.end()));
    ret.exec_list.push_back(exec.get());
#if MXNET_USE_PROFILER
    opr_names += inode.source->op()->name + ",";
#endif
  }

  if (pctx == nullptr) return ret;
  ret.ctx = *pctx;
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);

  bool is_gpu = pctx->dev_mask() == gpu::kDevMask;
  auto exec_fun = [exec_list, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    // Run all opr in the sub-graph
    for (auto &exec : exec_list) {
      exec->Run(ctx);
    }
    if (is_gpu) {
#if MXNET_USE_CUDA
      // Wait GPU kernel to finish.
      ctx.get_stream<gpu>()->Wait();
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    on_complete();
  };
#if MXNET_USE_PROFILER
    opr_names.pop_back();
    opr_names += "]";
    // the lifetime of `opr_names.c_str()` is same with opr_names
    // you need to copy it out. (potential memory leak risk)
    char *p_opr_name = new char[opr_names.size() + 1];
    memcpy(p_opr_name, opr_names.c_str(), opr_names.size() + 1);
#endif
  ret.opr = Engine::Get()->NewOperator(
      exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
      PROFILER_MESSAGE(p_opr_name));
  return ret;
}
}  // namespace exec

Executor *Executor::Bind(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states,
                         Executor* shared_exec) {
  auto exec = new exec::GraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_args, arg_grad_store, grad_req_type, aux_states,
             reinterpret_cast<Executor*>(shared_exec));
  return exec;
}
}  // namespace mxnet
