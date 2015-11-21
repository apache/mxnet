/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief Executor to execute the Graph.
 */
#include <dmlc/logging.h>
#include <mxnet/resource.h>
#include <mxnet/symbolic.h>
#include <memory>
#include <map>
#include "./graph_executor.h"
#include "./graph_algorithm.h"

namespace mxnet {
/*!
 * \brief wrapper class that wraps Backward operation as Forward.
 */
class GraphExecutor::BackwardOpWrapper : public Operator {
 public:
  /*!
   * \brief create a backward Operator wrapper given forward op.
   * \param prop pointer to the property of forward wrapper
   * \param forward_op the shared ptr to Forward operator
   * \return the created wrapper.
   */
  explicit BackwardOpWrapper(const OperatorProperty *prop,
                             std::shared_ptr<Operator> forward_op)
      : op_(forward_op) {
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
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
  // implement forward
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    // set things correctly
    CHECK(arg_data_ptr_.size() == in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      *(arg_data_ptr_[i]) = in_data[i];
    }
    // redirect internally
    op_->Backward(ctx, out_grad_, in_data_, out_data_, req, out_data, aux_states);
  }
  virtual ExecType exec_type() const {
    return op_->exec_type();
  }

 private:
  /*! \brief internal forward operator */
  std::shared_ptr<Operator> op_;
  /*! \brief internal space for out_grad */
  std::vector<TBlob> out_grad_;
  /*! \brief internal space for in_data */
  std::vector<TBlob> in_data_;
  /*! \brief internal space for out_data */
  std::vector<TBlob> out_data_;
  /*!
   * \brief pointer to places in the internal space.
   *  arg_data_ptr_ maps in_data in Forward to the internal space.
   */
  std::vector<TBlob*> arg_data_ptr_;
};

// get resource
inline std::vector<ResourceRequest>
GraphExecutor::GetResource(uint32_t node_id) const {
  const StaticGraph::Node &node = graph_.nodes[node_id];
  // use input shape
  std::vector<TShape> in_shapes;
  for (StaticGraph::DataEntry e : node.inputs) {
    in_shapes.push_back(op_nodes_[e.source_id].outputs[e.index].shape);
  }

  if (node.is_forward()) {
    return node.op->ForwardResource(in_shapes);
  } else {
    CHECK(node.is_backward());
    return graph_.nodes[node.backward_source_id]
        .op->BackwardResource(in_shapes);
  }
}

inline int GraphExecutor::GetNumOutputs(uint32_t node_id) const {
  const StaticGraph::Node &node = graph_.nodes[node_id];
  if (node.is_forward()) {
    return node.op->NumOutputs();
  } else if (node.is_backward()) {
    return static_cast<int>(
        graph_.nodes[node.backward_source_id].op->ListArguments().size());
  } else {
    CHECK(node.is_variable());
    return 1;
  }
}

// implement get input option
template<typename T>
inline std::vector<std::pair<T, T> > GraphExecutor::GetInplaceOption(
    uint32_t node_id,
    const std::vector<T> &in_data,
    const std::vector<T> &out_data) const {
  // get the node
  const StaticGraph::Node &node = graph_.nodes[node_id];

  if (node.is_forward()) {
    std::vector<int> in_data_index(in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      in_data_index[i] = static_cast<int>(i);
    }
    std::vector<void*> out_data_ptr(out_data.size());
    for (size_t i = 0; i < out_data.size(); ++i) {
      out_data_ptr[i] = (void*)&out_data[i];  // NOLINT(*)
    }
    auto rmap_index = node.op->ForwardInplaceOption(in_data_index, out_data_ptr);
    std::vector<std::pair<T, T> > remap(rmap_index.size());
    for (size_t i = 0; i < remap.size(); ++i) {
      remap[i].first = in_data[rmap_index[i].first];
      remap[i].second = *static_cast<const T*>(rmap_index[i].second);
    }
    return remap;
  } else {
    CHECK(node.is_backward());
    // forward property
    const OperatorProperty *fwd = graph_.nodes[node.backward_source_id].op.get();

    std::vector<int> out_grad_index(fwd->NumVisibleOutputs());
    std::vector<int> in_data_index(fwd->ListArguments().size());
    std::vector<int> out_data_index(fwd->NumOutputs());
    CHECK_EQ(in_data_index.size(), out_data.size());
    int counter = 0;
    for (size_t i = 0; i < out_grad_index.size(); ++i) {
      out_grad_index[i] = counter++;
    }
    for (size_t i = 0; i < in_data_index.size(); ++i) {
      in_data_index[i] = counter++;
    }
    for (size_t i = 0; i < out_data_index.size(); ++i) {
      out_data_index[i] = counter++;
    }
    auto args_index = fwd->DeclareBackwardDependency(
        out_grad_index, in_data_index, out_data_index);
    std::vector<const T*> args_array(counter, nullptr);
    CHECK_EQ(args_index.size(), in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      args_array[args_index[i]] = &in_data[i];
    }
    std::vector<void*> in_grad_ptr(out_data.size());
    for (size_t i = 0; i < in_grad_ptr.size(); ++i) {
      in_grad_ptr[i] = (void*)&out_data[i];  // NOLINT(*)
    }
    auto remap_index = fwd->BackwardInplaceOption(
        out_grad_index, in_data_index, out_data_index, in_grad_ptr);
    std::vector<std::pair<T, T> > remap(remap_index.size());
    for (size_t i = 0; i < remap_index.size(); ++i) {
      if (args_array[remap_index[i].first] == nullptr) {
        LOG(FATAL) << "BackwardInplaceOption not consistent with DeclareBackwardDependency";
      }
      remap[i].first = *args_array[remap_index[i].first];
      remap[i].second = *static_cast<T*>(remap_index[i].second);
    }
    return remap;
  }
}

inline GraphExecutor::OpExecEntry
GraphExecutor::GetOpExecEntry(uint32_t nid) {
  OpNode& op_node = op_nodes_[nid];
  std::vector<OpReqType> req;
  std::vector<NDArray> in_array, out_array, aux_array;
  in_array.reserve(graph_.nodes[nid].inputs.size());
  out_array.reserve(op_node.outputs.size());
  req.reserve(op_node.outputs.size());
  aux_array.reserve(op_node.aux_states.size());

  OpExecEntry exec;
  // output
  for (const DataEntryInfo& out : op_node.outputs) {
    out_array.push_back(out.data);
    exec.mutate_vars.push_back(out.data.var());
    req.push_back(out.op_req);
  }
  // aux
  for (const DataEntryInfo& aux : op_node.aux_states) {
    aux_array.push_back(aux.data);
    exec.mutate_vars.push_back(aux.data.var());
  }
  // input
  for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
    const DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    in_array.push_back(info.data);
    // skip inplace since they already appear in mutate vars
    if (info.inplace_op_id != static_cast<int>(nid)) {
      exec.use_vars.push_back(info.data.var());
    }
  }
  // de-duplicate the used vars
  std::sort(exec.use_vars.begin(), exec.use_vars.end());
  exec.use_vars.resize(std::unique(exec.use_vars.begin(), exec.use_vars.end()) -
                       exec.use_vars.begin());

  // start setup exec function.
  for (const Resource& r : op_node.op_ctx.requested) {
    exec.mutate_vars.push_back(r.var);
  }

  Operator* op = op_node.op.get();
  OpContext* op_ctx_ptr = &op_node.op_ctx;
  bool is_gpu = op_node.ctx.dev_mask() == gpu::kDevMask;
  bool is_async = op->exec_type() == Operator::kAsync;
  exec.exec_fun = [op, is_gpu, is_async, op_ctx_ptr, in_array, req, out_array, aux_array]
      (RunContext ctx, Engine::CallbackOnComplete on_complete) {
    std::vector<TBlob> in_data(in_array.size());
    std::vector<TBlob> out_data(out_array.size());
    std::vector<TBlob> aux_data(aux_array.size());
    std::transform(in_array.begin(), in_array.end(), in_data.begin(), [](const NDArray& nd) {
      return nd.data();
      });
    std::transform(out_array.begin(), out_array.end(), out_data.begin(), [](const NDArray& nd) {
        return nd.data();
      });
    std::transform(aux_array.begin(), aux_array.end(), aux_data.begin(), [](const NDArray& nd) {
        return nd.data();
      });
    op_ctx_ptr->run_ctx = ctx;
    if (is_async) {
      op_ctx_ptr->async_on_complete = on_complete;
    }
    op->Forward(*op_ctx_ptr, in_data, req, out_data, aux_data);
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
  return exec;
}

GraphExecutor::~GraphExecutor() {
  Engine::Get()->WaitForAll();
  // need to delete the operators before delete the NDArray they referenced.
  for (OpNode& node : op_nodes_) {
    node.DeleteOperator();
  }
}

void GraphExecutor::InitGraph(const Symbol &symbol, Context ctx, bool need_backward) {
  // initialize all internal data structures
  graph_.FromSymbol(symbol);
  num_forward_nodes_  = graph_.nodes.size();
  if (need_backward) {
    graph_.MakeBackwardPass(&head_grad_nodes_, &arg_grads_);
  }
  // reorganize so backward node always follow forward
  // note that this may not be the case, because existence of head_grad_nodes
  std::vector<uint32_t> topo = graph_.TopoSort();
  std::vector<uint32_t>  backward;
  for (uint32_t nid : topo) {
    if (nid < num_forward_nodes_) {
      topo_order_.push_back(nid);
    } else {
      backward.push_back(nid);
    }
  }
  topo_order_.insert(topo_order_.end(), backward.begin(), backward.end());
  // setup all the operator nodes data structure
  op_nodes_.resize(graph_.nodes.size());
  for (size_t i = 0; i < graph_.nodes.size(); ++i) {
    op_nodes_[i].ctx = ctx;
    op_nodes_[i].outputs.resize(GetNumOutputs(i));
  }
}

void GraphExecutor::InitDataEntryInfo(const std::vector<NDArray> &in_args,
                                      const std::vector<NDArray> &arg_grad_store,
                                      const std::vector<OpReqType> &grad_req_type,
                                      const std::vector<NDArray> &aux_states) {
  CHECK_EQ(arg_grad_store.size(), grad_req_type.size());
  CHECK_EQ(in_args.size(), graph_.arg_nodes.size());
  // bind inputs
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    DataEntryInfo &info = op_nodes_[graph_.arg_nodes[i]].outputs[0];
    info.type = kBindByExternal;
    info.data = in_args[i];
  }
  // setup ref for head nodes
  for (StaticGraph::DataEntry e : graph_.heads) {
    DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    ++info.ref_count;
    op_nodes_[e.source_id].activated = true;
  }
  // need Backward pass
  if (arg_grads_.size() != 0) {
    CHECK_EQ(arg_grads_.size(), arg_grad_store.size());
    CHECK_EQ(arg_grads_.size(), grad_req_type.size());
    // setup gradient placeholders
    for (size_t i = 0; i < arg_grads_.size(); ++i) {
      if (grad_req_type[i] == kNullOp) continue;
      CHECK_NE(grad_req_type[i], kWriteInplace)
          << "Gradient request can only be nullop, add, write";
      StaticGraph::DataEntry &grad_source = arg_grads_[i];
      DataEntryInfo &info = op_nodes_[grad_source.source_id].outputs[grad_source.index];
      info.type = kBindByExternal;
      info.op_req = grad_req_type[i];
      info.data = arg_grad_store[i];
      ++info.ref_count;
      op_nodes_[grad_source.source_id].activated = true;
    }
    // setup head gradient
    for (uint32_t nid : head_grad_nodes_) {
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      info.type = kTobeBindByExternal;
    }
  }
  // update ref counters for all other nodes, in reverse topo order
  for (auto it = topo_order_.rbegin(); it != topo_order_.rend(); ++it) {
    uint32_t nid = *it;
    if (op_nodes_[nid].activated) {
      for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
        DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
        ++info.ref_count;
        op_nodes_[e.source_id].activated = true;
      }
    }
  }

  // shape inference
  std::vector<std::vector<TShape> > out_shapes(op_nodes_.size());
  std::vector<std::vector<TShape> > aux_shapes(op_nodes_.size());
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    out_shapes[i].resize(op_nodes_[i].outputs.size());
  }
  for (size_t i = 0; i < graph_.arg_nodes.size(); ++i) {
    out_shapes[graph_.arg_nodes[i]][0] = in_args[i].shape();
  }
  CHECK(graph_.InferNodeShapes(topo_order_, &out_shapes, &aux_shapes))
      << "Shape inference cannot be complete in bind";
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    for (size_t j = 0; j < out_shapes[i].size(); ++j) {
      op_nodes_[i].outputs[j].shape = out_shapes[i][j];
    }
  }
  // bind aux args
  size_t aux_ndarray_idx = 0;
  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    op_nodes_[i].aux_states.resize(aux_shapes[i].size());
    for (size_t j = 0; j < aux_shapes[i].size(); ++j) {
      DataEntryInfo &info = op_nodes_[i].aux_states[j];
      info.shape = aux_shapes[i][j];
      info.type = kBindByExternal;
      if (graph_.nodes[i].backward_source_id == -1) {
        info.data = aux_states[aux_ndarray_idx++];
      } else {
        CHECK_NE(graph_.nodes[i].backward_source_id, -1)
          << "Input auxiliary NDArray is less than required";
        info.data = op_nodes_[graph_.nodes[i].backward_source_id].aux_states[j].data;
      }
      CHECK_EQ(info.data.data().shape_, info.shape)
        << "Incorrect NDArray shape"
        << " Input: " << info.data.data().shape_
        << " Desired: " << info.shape;
    }
  }
}

void GraphExecutor::InitDataEntryMemory() {
  // setup the temp ref counter for allocator algorithms
  for (OpNode &op : op_nodes_) {
    for (DataEntryInfo &node : op.outputs) {
      node.temp_ref_count = node.ref_count;
    }
  }

  // use allocator to allocate memory.
  GraphStorageAllocator allocator(&graph_, topo_order_);
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;

    // check inplace option
    std::vector<DataEntryInfo*> in_data;
    in_data.reserve(graph_.nodes[nid].inputs.size());
    // check inputs are ready.
    for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
      DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
      CHECK_NE(info.type, kNotInitialized);
      CHECK_NE(info.temp_ref_count, 0);
      in_data.push_back(&info);
    }
    std::vector<DataEntryInfo*> out_data(op_nodes_[nid].outputs.size());
    for (size_t i = 0; i < op_nodes_[nid].outputs.size(); ++i) {
      out_data[i] = &op_nodes_[nid].outputs[i];
      CHECK_NE(out_data[i]->type, kInternalAllocated);
    }
    auto inplace = GetInplaceOption(nid, in_data, out_data);

    for (std::pair<DataEntryInfo*, DataEntryInfo*> kv : inplace) {
      DataEntryInfo* in = kv.first;
      DataEntryInfo* out = kv.second;
      if (enable_inplace_allocation_ &&
          in->temp_ref_count == 1 &&
          in->type == kInternalAllocated &&
          out->type == kNotInitialized) {
        // we can only do inplace if we are last user of in
        // and out is not initialized.
        out->type = kInternalAllocated;
        out->op_req = kWriteInplace;
        out->storage_id = in->storage_id;
        // set inplace op id
        in->temp_ref_count = 0;
        in->inplace_op_id = static_cast<int>(nid);
      }
    }
    // allocate output,
    for (DataEntryInfo *out : out_data) {
      if (out->op_req == kNullOp && out->temp_ref_count != 0) {
        out->op_req = kWriteTo;
      }
      if (out->type == kNotInitialized) {
        out->storage_id = allocator.Request(
            op_nodes_[nid].ctx, out->shape, nid);
        out->type = kInternalAllocated;
      }
    }
    // then free inputs
    for (DataEntryInfo *in : in_data) {
      // temp_ref_count == 0 means it is taken by inplace op
      if (in->temp_ref_count == 0) {
        CHECK_EQ(in->inplace_op_id, static_cast<int>(nid));
        continue;
      }
      // if we decrease it to zero, means we are ready to relase
      --in->temp_ref_count;
      if (in->temp_ref_count == 0 && in->type == kInternalAllocated) {
        allocator.Release(in->storage_id, nid);
      }
    }
    // check out again, if there is temp_ref_count == 0, release it
    for (DataEntryInfo *out : out_data) {
      if (out->temp_ref_count == 0 && out->type == kInternalAllocated) {
        allocator.Release(out->storage_id, nid);
      }
    }
  }
  // one pass complete, allocate real memory
  this->total_allocated_reals_ = allocator.InitStorages();
  // get the real data NDArray into the DataEntryInfo
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    for (DataEntryInfo &out : op_nodes_[nid].outputs) {
      CHECK_NE(out.type, kNotInitialized);
      if (out.type == kInternalAllocated) {
        out.data = allocator.Get(out.storage_id, out.shape);
      }
    }
  }
  // setup heads
  for (StaticGraph::DataEntry e : graph_.heads) {
    DataEntryInfo &info = op_nodes_[e.source_id].outputs[e.index];
    CHECK_EQ(info.type, kInternalAllocated);
    heads_ndarray_.push_back(info.data);
  }
}

void GraphExecutor::InitResources() {
  // prepare for temp space allocation
  std::vector<uint32_t> req_temp_cnt(topo_order_.size(), 0);
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    uint32_t cnt = 0;
    for (const ResourceRequest& req : GetResource(nid)) {
      if (req.type == ResourceRequest::kTempSpace) ++cnt;
    }
    CHECK_LE(cnt, 1) << "Node can only have one temp space request";
    req_temp_cnt[nid] = cnt;
  }

  uint32_t num_color = static_cast<uint32_t>(common::GetExecNumMatchColor());
  std::vector<uint32_t> req_temp_color;
  // use graph coloring to find node that won't run in parallel
  num_color = graph::ColorNodeGroup(graph_, topo_order_, req_temp_cnt,
                                    num_color, &req_temp_color);

  // cached resources temp space
  std::map<Context, std::map<uint32_t, Resource> > cached_temp;
  total_allocated_temp_ = 0;

  // Resource allocation
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    const std::vector<ResourceRequest>& reqs = GetResource(nid);
    auto& requested = op_nodes_[nid].op_ctx.requested;
    requested.clear();
    // Get the resource of temporal space.
    for (const ResourceRequest& req : reqs) {
      const Context &ctx = op_nodes_[nid].ctx;
      if (req.type == ResourceRequest::kTempSpace) {
        uint32_t color = req_temp_color[nid];
        // try to reuse graph in same color
        std::map<uint32_t, Resource> &cmap = cached_temp[ctx];
        if (cmap.count(color) != 0) {
          requested.push_back(cmap.at(color));
        } else {
          Resource r = ResourceManager::Get()->Request(ctx, req);
          requested.push_back(r);
          cmap[color] = r;
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

void GraphExecutor::InitOpNodes() {
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    OpNode& op_node = op_nodes_[nid];
    if (graph_.nodes[nid].is_forward()) {
      op_node.op.reset(graph_.nodes[nid].op->CreateOperator(op_node.ctx));
    } else {
      CHECK(graph_.nodes[nid].is_backward());
      op_node.op.reset(new BackwardOpWrapper(
          graph_.nodes[graph_.nodes[nid].backward_source_id].op.get(),
          op_nodes_[graph_.nodes[nid].backward_source_id].op));
    }
    bool allow_cache = true;
    for (StaticGraph::DataEntry e : graph_.nodes[nid].inputs) {
      DataEntryInfo& info = op_nodes_[e.source_id].outputs[e.index];
      if (info.type == kTobeBindByExternal) allow_cache = false;
    }
    for (DataEntryInfo& info : op_node.outputs) {
      if (info.type == kTobeBindByExternal) allow_cache = false;
    }
    if (allow_cache) {
      op_node.cached_exec = GetOpExecEntry(nid);
      op_node.cached_opr = Engine::Get()->NewOperator(
          op_node.cached_exec.exec_fun,
          op_node.cached_exec.use_vars,
          op_node.cached_exec.mutate_vars,
          FnProperty::kNormal);
    }
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
  for (size_t i = topo_start; i < topo_end; ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    if (graph_.nodes[nid].is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    opnode.op_ctx.is_train = is_train;
    if (opnode.cached_opr != nullptr) {
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx);
    } else {
      auto exec = GetOpExecEntry(nid);
      Engine::Get()->PushAsync(
          exec.exec_fun,
          opnode.ctx,
          exec.use_vars,
          exec.mutate_vars,
          FnProperty::kNormal);
    }
  }
}

void GraphExecutor::Print(std::ostream &os) const {
  os << "num_forward_nodes=" << num_forward_nodes_ << '\n';
  for (size_t i = 0; i < topo_order_.size(); ++i) {
    uint32_t nid = topo_order_[i];
    if (!op_nodes_[nid].activated) continue;
    os << "Op " << i << ":" << graph_.nodes[nid].name << '\n';
    for (size_t j = 0; j < op_nodes_[nid].outputs.size(); ++j) {
      const DataEntryInfo &info = op_nodes_[nid].outputs[j];
      os << "\toutput[" << j << "]: shape=" << info.shape;
      if (info.storage_id != GraphStorageAllocator::kBadStorageID) {
        os << ", storage_id=" << info.storage_id;
      }
      if (info.inplace_op_id != -1) {
        os << ", inplace_consumer=" << graph_.nodes[info.inplace_op_id].name;
      }
      os << '\n';
    }
    for (size_t j = 0; j < op_nodes_[nid].op_ctx.requested.size(); ++j) {
      const Resource& resource = op_nodes_[nid].op_ctx.requested[j];
      os << "\tresource[" << j << "]: ";
      if (resource.req.type == ResourceRequest::kTempSpace) {
        os << "type=TempSpace, id=" << resource.id;
      } else if (resource.req.type == ResourceRequest::kRandom) {
        os << "type=RandomNumber";
      }
      os << '\n';
    }
  }
  os << "Total " << (total_allocated_reals_ >> 18UL) <<" MB allocated\n";
  os << "Total " << total_allocated_temp_ <<" TempSpace resource requested\n";
}

void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::Backward(const std::vector<NDArray> &head_grads) {
  if (head_grads.size() != 0) {
    // TODO(bing, min): consider pass a map for backward
    CHECK_EQ(head_grad_nodes_.size(), head_grads.size());
    for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
      uint32_t nid = head_grad_nodes_[i];
      CHECK(graph_.nodes[nid].is_variable());
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      CHECK_EQ(info.type, kTobeBindByExternal);
      info.data = head_grads[i];
    }
  } else {
    // check all the head_grad_nodes need to have zero ref_count
    // loss function do not need out_grad
    for (size_t i = 0; i < head_grad_nodes_.size(); ++i) {
      uint32_t nid = head_grad_nodes_[i];
      DataEntryInfo &info = op_nodes_[nid].outputs[0];
      CHECK_EQ(info.ref_count, 0)
          << "Because the last operator is not Loss function, "
          << "head_gradient is required in calling backward.";
    }
  }
  RunOps(true, num_forward_nodes_, topo_order_.size());
}

Executor *Executor::Bind(Symbol symbol,
                         Context ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states) {
  GraphExecutor *exec = new GraphExecutor();
  exec->Init(symbol, ctx, in_args, arg_grad_store, grad_req_type, aux_states);
  return exec;
}
}  // namespace mxnet
