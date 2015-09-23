/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.h
 * \brief Executor to execute the Forward and Backward on Composition Graph.
*/
#ifndef MXNET_SYMBOL_GRAPH_EXECUTOR_H_
#define MXNET_SYMBOL_GRAPH_EXECUTOR_H_

#include <mxnet/symbolic.h>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "./static_graph.h"
#include "./graph_memory_allocator.h"

namespace mxnet {
/*!
 * \brief Executor of a computation graph.
 */
class GraphExecutor : public Executor {
 public:
  virtual ~GraphExecutor();
  void Forward(bool is_train) override;
  void Backward(const std::vector<NDArray> &head_grads) override;
  const std::vector<NDArray> &outputs() const override {
    return heads_ndarray_;
  }
  void Print(std::ostream &os) const override; // NOLINT(*)
  // implement Executor::Bind, only call it once.
  inline void Init(Symbol symbol,
                   Context ctx,
                   const std::vector<NDArray> &in_args,
                   const std::vector<NDArray> &arg_grad_store,
                   const std::vector<OpReqType> &grad_req_type,
                   const std::vector<NDArray> &aux_states) {
    enable_inplace_allocation_ = dmlc::GetEnv("MXNET_EXEC_ENABLE_INPLACE", true);

    CHECK_EQ(grad_req_type.size(), arg_grad_store.size());
    bool need_backward = false;
    for (auto req : grad_req_type) {
      if (req != kNullOp) need_backward = true;
    }
    this->InitGraph(symbol, ctx, need_backward);
    this->InitDataEntryInfo(in_args, arg_grad_store, grad_req_type, aux_states);
    this->InitDataEntryMemory();
    this->InitResources();
    this->InitOpNodes();
  }

 protected:
  // internal class of wrapping BackwardOp as ForwardOp
  class BackwardOpWrapper;
  // type of data entry
  enum DataEntryType {
    // memory is binded by external NDArray in Bind
    kBindByExternal,
    // to be binded by external NDArray in Forward and Backward
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
    ~OpNode() {
      if (cached_opr != nullptr) {
        Engine::Get()->DeleteOperator(cached_opr);
      }
    }
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
  // initialize the internal graph structure
  void InitGraph(const Symbol &symbol, Context ctx, bool need_backward);
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
  void InitOpNodes();
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
  // internal computational graph
  StaticGraph graph_;
  // topological order of nodes in computation graph
  // backward nodes always follow forward nodes
  std::vector<uint32_t> topo_order_;
  // whether to enable inplace space
  bool enable_inplace_allocation_;
  // total allocated space in #reals
  size_t total_allocated_reals_;
  // total allocated temp space
  size_t total_allocated_temp_;
  // number of forward nodes in the graph
  size_t num_forward_nodes_;
  // head gradient node in the graph, if there is backward pass
  std::vector<uint32_t> head_grad_nodes_;
  // argument node in the graph, if there is backward pass
  std::vector<StaticGraph::DataEntry> arg_grads_;
  // operational nodes
  std::vector<OpNode> op_nodes_;
  // head NDArrays
  std::vector<NDArray> heads_ndarray_;
};  // class GraphExecutor
}  // namespace mxnet
#endif  // MXNET_SYMBOL_GRAPH_EXECUTOR_H_
