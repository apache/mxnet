/*!
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <vector>
#include <algorithm>
#include "./graph_executor.h"

namespace mxnet {
namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {
GraphExecutor::~GraphExecutor() {
  for (OpNode& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
}

void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_type,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec) {
  this->InitGraph(symbol, default_ctx,
                  ctx_map, in_args, arg_grad_store,
                  grad_req_type, aux_states);
  this->InitOpExecs();
  this->InitResources();
  if (shared_exec != nullptr) {
    this->InitDataEntryMemory(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_);
  } else {
    this->InitDataEntryMemory({});
  }
  this->InitCachedOps();
}

void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  LOG(FATAL) << "not implemented";
}

void GraphExecutor::Backward(const std::vector<NDArray> &head_grads) {
  const auto& idx = graph_.indexed_graph();
  for (size_t i = num_inputs_; i < idx.input_nodes().size(); ++i) {
    uint32_t nid = idx.input_nodes().at(i);
    uint32_t oid = head_grad_map_.at(idx[nid].source);
    CopyFromTo(head_grads[oid], &(data_entry_[idx.entry_id(nid, 0)]));
  }
  RunOps(true, num_forward_nodes_, op_nodes_.size());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  LOG(FATAL) << "not implemented";
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback) {
  LOG(FATAL) << "not implemented";
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  return output_arrays_;
}

nnvm::Graph GraphExecutor::InitGradGraph(
    nnvm::Symbol symbol,
    const std::vector<OpReqType>& grad_req_type,
    const std::vector<NDArray>& arg_grad_store) {
  using nnvm::NodePtr;
  using nnvm::NodeEntry;
  nnvm::Graph g;
  g.outputs = symbol.outputs;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    head_grad_entry_.emplace_back(NodeEntry{nnvm::Node::Create(), 0, 0});
    head_grad_map_[head_grad_entry_.back().node.get()] = i;
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
  nnvm::Graph g_grad = nnvm::pass::Gradient(g, symbol.outputs, xs, head_grad_entry_);
  CHECK_EQ(g_grad.outputs.size(), xs.size());
  g.outputs.insert(g.outputs.end(), g_grad.outputs.begin(), g_grad.outputs.end());
  return g;
}

void GraphExecutor::InitGraph(nnvm::Symbol symbol,
                              const Context& default_ctx,
                              const std::map<std::string, Context>& ctx_map,
                              const std::vector<NDArray>& in_args,
                              const std::vector<NDArray>& arg_grad_store,
                              const std::vector<OpReqType>& grad_req_type,
                              const std::vector<NDArray>& aux_states) {
  // setup gradient
  bool need_grad = false;
  for (OpReqType req : grad_req_type) {
    if (req != kNullOp) need_grad = true;
  }
  num_outputs_ = symbol.outputs.size();
  num_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();
  if (need_grad) {
    graph_ = InitGradGraph(symbol, grad_req_type, arg_grad_store);
  } else {
    graph_.outputs = symbol.outputs;
  }

  CHECK_EQ(ctx_map.size(), 0);
  // other works
  const auto& idx = graph_.indexed_graph();
  // TODO(tqchen) assign context
  data_context_.resize(idx.num_node_entries(), default_ctx);
  op_nodes_.resize(idx.num_nodes());
  for (OpNode& n : op_nodes_) {
    n.ctx = default_ctx;
  }
  // setup the known data entries
  data_entry_.resize(idx.num_node_entries());
  auto mutable_nodes = idx.mutable_input_nodes();
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_types;


  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_inputs_; ++i) {
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

  arg_shapes.resize(idx.input_nodes().size(), TShape());
  arg_types.resize(idx.input_nodes().size(), 0);

  // other initializations
  graph_ = nnvm::pass::InferShape(std::move(graph_), arg_shapes, "__shape__");
  graph_ = nnvm::pass::InferType(std::move(graph_), arg_types);
  graph_ = nnvm::ApplyPass(std::move(graph_), {"PlanMemory"});

  // setup number of nodes
  num_forward_nodes_ = 0;
  for (size_t i = 0; i < num_outputs_; ++i) {
    num_forward_nodes_ = std::max(
        num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  // set the data entry of
  for (size_t j = num_outputs_; j < idx.outputs().size(); ++j) {
    data_entry_[idx.entry_id(idx.outputs()[j])] = grad_store_[j - num_outputs_].second;
  }
}

class GraphExecutor::ForwardOpExecutor
    : public GraphExecutor::OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
  }

  void Init() override {
    in_data_.clear(); aux_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_data_.push_back(in_array[i].data());
      } else {
        aux_data_.push_back(in_array[i].data());
      }
    }
    out_data_.resize(out_array.size());
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  explicit ForwardOpExecutor(Operator* op, std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend class GraphExecutor;
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class GraphExecutor::BackwardOpExecutor
    : public GraphExecutor::OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
  }

  void Init() override {
    size_t arg_top = 0, aux_top = 0;
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        *arg_data_ptr_[arg_top++] = in_array[i].data();
      } else {
        aux_data_[aux_top++] = in_array[i].data();
      }
    }
    CHECK_EQ(out_array.size(), in_grad_.size());
    std::transform(out_array.begin(), out_array.end(),
                   in_grad_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  explicit BackwardOpExecutor(std::shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());

    std::vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    std::vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    std::vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  std::vector<TBlob*> arg_data_ptr_;
};


void GraphExecutor::InitOpExecs() {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;
  auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");
  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) continue;
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(inode.source->op())) {
      mutate_index = fmutate_inputs[inode.source->op()](inode.source->attrs);
    }
    if (fcreate_layer_op.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      op_nodes_[i].exec.reset(
          new ForwardOpExecutor(fcreate_layer_op[inode.source->op()](
              inode.source->attrs, op_nodes_[i].ctx, ishape, itype), mutate_index));
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(op_nodes_[fwd_id].exec != nullptr);
      op_nodes_[i].exec.reset(
          new BackwardOpExecutor(
              static_cast<ForwardOpExecutor*>(op_nodes_[fwd_id].exec.get())->op_,
              op::OpPropGetOpProperty(inode.source->attrs),
              mutate_index));
    } else {
      LOG(INFO) << "FCompute not registered " << inode.source->op()->name;
    }
  }
}

void GraphExecutor::InitResources() {
  auto& fresource =
      nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
  auto& idx = graph_.indexed_graph();
  // Use global resource pool for each executor for now.
  std::map<Context, Resource> cached_temp;
  total_allocated_temp_ = 0;
  // Resource allocation
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (fresource.count(inode.source->op()) == 0) continue;
    auto reqs = fresource[inode.source->op()](inode.source->attrs);
    auto& requested = op_nodes_[nid].exec->op_ctx.requested;
    requested.clear();
    // Get the resource of temporal space.
    for (const ResourceRequest& req : reqs) {
      const Context &ctx = op_nodes_[nid].ctx;
      if (req.type == ResourceRequest::kTempSpace) {
        if (cached_temp.count(ctx) != 0) {
          requested.push_back(cached_temp.at(ctx));
        } else {
          Resource r = ResourceManager::Get()->Request(ctx, req);
          requested.push_back(r);
          cached_temp[ctx] = r;
          ++total_allocated_temp_;
        }
      } else if (req.type == ResourceRequest::kRandom) {
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
      } else {
        LOG(FATAL) << "resource type not yet supported";
      }
    }
  }
}

// initialize the memory of each entries
void GraphExecutor::InitDataEntryMemory(const std::vector<NDArray>& shared_pool) {
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());

  // information about the pool
  using PoolEntry = std::pair<Context, size_t>;
  std::vector<PoolEntry> pool_info;

  // assign array to head gradient
  for (size_t i = num_inputs_; i < idx.input_nodes().size(); ++i) {
    uint32_t nid = idx.input_nodes().at(i);
    uint32_t oid = head_grad_map_.at(idx[nid].source);
    uint32_t eid = idx.entry_id(idx.outputs()[oid]);
    CHECK(vshape[eid].ndim() != 0);
    CHECK(vdtype[eid] != -1);
    data_entry_[idx.entry_id(nid, 0)] = NDArray(vshape[eid], data_context_[eid], vdtype[eid]);
  }

  // get maximum bytes in each pool
  for (size_t i = 0; i < vshape.size(); ++i) {
    size_t bytes = vshape[i].Size() * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];
    if (storage_id < 0) continue;
    if (!data_entry_[i].is_none()) continue;
    size_t sid = static_cast<size_t>(storage_id);
    if (sid <= pool_info.size()) {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0)});
    }
    PoolEntry& info = pool_info[sid];
    if (info.second == 0) {
      info = PoolEntry{data_context_[i], bytes};
    } else {
      info.second = std::max(info.second, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  for (const NDArray& nd : shared_pool) {
    size_t bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
    free_pool.insert(std::make_pair(bytes, nd));
  }
  // remake the data pool
  data_pool_.clear();
  for (size_t i = 0; i < pool_info.size(); ++i) {
    const Context& ctx = pool_info[i].first;
    size_t bytes = pool_info[i].second;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) {
      if (it->second.ctx() == ctx && it->first >= bytes) {
        data_pool_.push_back(it->second);
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated) {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<index_t>::max());
      // allocate float arrays
      data_pool_.emplace_back(NDArray(TShape({index_t(nword)}), ctx));
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

  // initialize output arrays
  for (size_t i = 0; i < num_outputs_; ++i) {
    auto& e = idx.outputs()[i];
    output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
  }
}

void GraphExecutor::InitCachedOps() {
  // get the graph
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace
      = graph_.GetAttr<std::vector<int> >("storage_inplace_index");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0);
    CHECK_EQ(exec->out_array.size(), 0);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    std::vector<uint32_t> inplace_inputs;
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(data_entry_[eid]);
      if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
        inplace_inputs.push_back(vstorage_inplace[eid]);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
    std::sort(inplace_inputs.begin(), inplace_inputs.end());

    bool is_async = false;
    bool is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;
    // the variable
    std::vector<Engine::VarHandle> use_vars, mutate_vars, all_vars;
    for (size_t i = 0; i < exec->in_array.size(); ++i) {
      if (!std::binary_search(inplace_inputs.begin(), inplace_inputs.end(), i)) {
        auto& nd = exec->in_array[i];
        all_vars.push_back(nd.var());
        use_vars.push_back(nd.var());
      }
    }
    for (auto& nd : exec->out_array) {
      all_vars.push_back(nd.var());
      mutate_vars.push_back(nd.var());
    }
    Engine::Get()->PushSync([exec](RunContext rctx) {
        exec->Init();
      }, Context::CPU(), {}, all_vars);

    auto exec_fun = [exec, is_async, is_gpu](
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
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
    op_nodes_[nid].cached_opr =  Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal);
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  const auto& idx = graph_.indexed_graph();
  for (size_t nid = topo_start; nid < topo_end; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    opnode.exec->op_ctx.is_train = is_train;
    if (opnode.cached_opr != nullptr) {
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx);
    } else {
      LOG(FATAL) << "TODO";
    }
  }
}

}  // namespace exec
}  // namespace mxnet
