/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.h
 * \brief Executor to execute the Forward and Backward on Composition Graph.
*/
#ifndef MXNET_SYMBOL_GRAPH_EXECUTOR_H_
#define MXNET_SYMBOL_GRAPH_EXECUTOR_H_

#include <mxnet/c_api.h>
#include <mxnet/symbolic.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "./static_graph.h"
#include "./graph_memory_allocator.h"

namespace mxnet {
/*!
 * \brief Executor of a computation graph.
 */
class GraphExecutor : public Executor {
 public:
  GraphExecutor() {}
  virtual ~GraphExecutor();
  void Forward(bool is_train) override;
  void PartialForward(bool is_train, int step, int *step_left) override;
  void Backward(const std::vector<NDArray> &head_grads) override;
  const std::vector<NDArray> &outputs() const override {
    return heads_ndarray_;
  }
  void Print(std::ostream &os) const override; // NOLINT(*)
  // install callback
  void SetMonitorCallback(const MonitorCallback& callback) {
    CHECK(callback) << "invalid callback";
    monitor_callback_ = callback;
  }
  // implement Executor::Bind, only call it once.
  inline void Init(Symbol symbol,
                   const Context& default_ctx,
                   const std::map<std::string, Context>& ctx_map,
                   const std::vector<NDArray> &in_args,
                   const std::vector<NDArray> &arg_grad_store,
                   const std::vector<OpReqType> &grad_req_type,
                   const std::vector<NDArray> &aux_states,
                   Executor* shared_exec = nullptr) {
    enable_inplace_allocation_ = dmlc::GetEnv("MXNET_EXEC_ENABLE_INPLACE", true);
    prefer_bulk_execution_ = dmlc::GetEnv("MXNET_EXEC_PREFER_BULK_EXEC", true);
    if (shared_exec != NULL) {
      GraphExecutor* gexec = dynamic_cast<GraphExecutor*>(shared_exec);
      CHECK(gexec) << "Input executor for sharing memory must have GraphExecutor type.";
      shared_mem_ = gexec->shared_mem_;
    } else {
      shared_mem_ = std::make_shared<GraphStoragePool>();
    }

    CHECK_EQ(grad_req_type.size(), arg_grad_store.size());
    bool need_backward = false;
    for (auto req : grad_req_type) {
      if (req != kNullOp) need_backward = true;
    }
    this->InitGraph(symbol, default_ctx, ctx_map,
                    in_args, arg_grad_store, grad_req_type,
                    need_backward);
    this->InitDataEntryInfo(in_args, arg_grad_store, grad_req_type, aux_states);
    this->InitOperators();
    this->InitDataEntryMemory();
    this->InitResources();
    this->InitCachedOps();
    this->InitOpSegs();
  }

 protected:
  // internal class of wrapping BackwardOp as ForwardOp
  class BackwardOpWrapper;
  // type of data entry
  enum DataEntryType {
    // memory is bound by external NDArray in Bind
    kBindByExternal,
    // to be bound by external NDArray in Forward and Backward
    kTobeBindByExternal,
    // internal memory, allocated
    kInternalAllocated,
    // internal memory, to be allocated
    kNotInitialized
  };
  // Additional information about each data entry
  struct DataEntryInfo {
    // the actual data for the entry
    NDArray data;
    // write request to this entry
    OpReqType op_req;
    // the operatio node that will take
    // this DataEntry as inplace input
    int inplace_op_id;
    // data entry type
    DataEntryType type;
    // shape of this entry
    TShape shape;
    // data type of this entry
    int type_flag;
    // storage id from allocator if it is internal allocation.
    GraphStorageAllocator::StorageID storage_id;
    // reference count on how many times this entry is being used.
    // That is how many operators and heads need this DataEntry
    // this is a temporal variable that is used during initialization.
    uint32_t temp_ref_count;
    // real permanent ref count
    uint32_t ref_count;
    // constructor
    DataEntryInfo()
        : op_req(kNullOp),
          inplace_op_id(-1),
          type(kNotInitialized),
          storage_id(GraphStorageAllocator::kBadStorageID),
          temp_ref_count(0), ref_count(0) {}
  };
  // all the information needed to push the op to engine
  struct OpExecEntry {
    // execution function for
    Engine::AsyncFn exec_fun;
    // variables to read from
    std::vector<Engine::VarHandle> use_vars;
    // variables to mutate
    std::vector<Engine::VarHandle> mutate_vars;
    // constructor
    OpExecEntry() : exec_fun(nullptr) {}
  };
  // Information about operational node
  struct OpNode {
    // whether this op node is activated
    bool activated;
    // the context of the node
    Context ctx;
    // data entry information about outputs of op
    std::vector<DataEntryInfo> outputs;
    // auxiliary data information of op
    std::vector<DataEntryInfo> aux_states;
    // The following parts are constructed in InitOpNodes
    // the real operator
    std::shared_ptr<Operator> op;
    // op context, that is defined for this op.
    OpContext op_ctx;
    // executor, this is only allocated for nodes
    // whose inputs, outputs are pre-defined.
    // otherwise cached_exec.exec_fun == nullptr
    OpExecEntry cached_exec;
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
    // constructor
    OpNode() : activated(false) {}
    // Manual option for delete operator
    // need to do this before delete NDArrays
    inline void DeleteOperator() {
      if (cached_opr != nullptr) {
        Engine::Get()->DeleteOperator(cached_opr);
        cached_opr = nullptr;
      }
    }
  };
  // a cached segment operator that executes a segment
  struct CachedSegOpr {
    // context of the operator
    Context ctx;
    // begin in topo order
    size_t topo_begin;
    // end in topo order
    size_t topo_end;
    // the cached operator
    Engine::OprHandle opr;
  };
  /*!
   * \brief Get input option of a node.
   *  This function is overriden for both Forward and Backward node.
   *
   * \param node_id node index of node in StaticGraph
   * \param in_data the input data entry to the node
   * \param out_data the output data entry in the graph
   * \return the paired inplace option.
   */
  template<typename T>
  inline std::vector<std::pair<T, T> > GetInplaceOption(
      uint32_t node_id,
      const std::vector<T> &in_data,
      const std::vector<T> &out_data) const;
  /*!
   * \brief Get resource requirement of a node.
   *  This function is overriden for both Forward and Backward node.
   * \param node_id node index of node in StaticGraph
   * \return the desired resource request.
   */
  inline std::vector<ResourceRequest> GetResource(uint32_t node_id) const;
  /*!
   * \brief Get number of outputs of a node.
   *  This function is overriden for both Forward and Backward node.
   * \param node_id node index of node in StaticGraph
   * \return the number of outputs of the node.
   */
  inline int GetNumOutputs(uint32_t node_id) const;
  /*!
   * \brief get execution entry for an OpNode.
   *  This function can only be called after initialization is done.
   * \param node_id the id of operational node.
   * \return the execution entry.
   */
  inline OpExecEntry GetOpExecEntry(uint32_t node_id);
  /*!
   * \brief Try to create a cached operator to run segments between start and end
   * \param topo_start beginning of segment
   * \param topo_end end of segment
   * \return the cached operator.
   * The ret.opr can be nullptr if tyhe creation failed
   */
  CachedSegOpr CreateCachedSegOpr(size_t topo_start, size_t topo_end);
  // initialize the internal graph structure
  void InitGraph(const Symbol &symbol,
                 const Context& default_ctx,
                 const std::map<std::string, Context>& ctx_map,
                 const std::vector<NDArray> &in_args,
                 const std::vector<NDArray> &arg_grad_store,
                 const std::vector<OpReqType> &grad_req_type,
                 bool need_backward);
  // initialize internal DataEntryInfo, reference counting
  void InitDataEntryInfo(const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states);
  // initialize internal data entries NDArray
  void InitDataEntryMemory();
  // initialize the internal resources for each op
  void InitResources();
  // initialize OpNode data structure
  void InitOperators();
  // initialize OpNode data structure
  void InitCachedOps();
  // initialize segments of code to run together as a group.
  void InitOpSegs();
  // assign context to the graph, this will mutate the graph.
  void AssignContext(const Context default_ctx,
                     const std::map<std::string, Context>& ctx_map,
                     const std::vector<NDArray> &in_args,
                     const std::vector<NDArray> &arg_grad_store,
                     const std::vector<OpReqType> &grad_req_type,
                     std::vector<Context> *ctx_plan);
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
  // internal computational graph
  StaticGraph graph_;
  // topological order of nodes in computation graph
  // backward nodes always follow forward nodes
  std::vector<uint32_t> topo_order_;
  // whether to enable inplace space
  bool enable_inplace_allocation_;
  // total allocated space in bytes
  size_t total_allocated_bytes_;
  // total allocated temp space
  size_t total_allocated_temp_;
  // number of forward nodes in the graph
  size_t num_forward_nodes_;
  // whether to enable bulk execution
  bool prefer_bulk_execution_;
  // head gradient node in the graph, if there is backward pass
  std::vector<uint32_t> head_grad_nodes_;
  // mirror map of nodes, experimental feature, normally can be ignored.
  std::map<uint32_t, uint32_t> mirror_source_map_;
  // argument node in the graph, if there is backward pass
  std::vector<StaticGraph::DataEntry> arg_grads_;
  // operational nodes
  std::vector<OpNode> op_nodes_;
  // head NDArrays
  std::vector<NDArray> heads_ndarray_;
  // shared NDArrays
  std::shared_ptr<GraphStoragePool> shared_mem_;
  // monitor call back
  std::function<void(const char*, void*)> monitor_callback_;
  // cached segment operator
  std::vector<CachedSegOpr> cached_seg_opr_;
};  // class GraphExecutor
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_EXECUTOR_H_
