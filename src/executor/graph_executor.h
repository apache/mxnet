/*!
 * Copyright (c) 2016 by Contributors
 * \file graph_executor.h
 * \brief Executor to execute the computation graph.
 */
#ifndef MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
#define MXNET_EXECUTOR_GRAPH_EXECUTOR_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./exec_pass.h"

namespace mxnet {

using NodeOperatorMap = std::unordered_map<const nnvm::Node*,
    std::shared_ptr<Operator>>;

// forward declaration
namespace exec {
class GraphExecutor;
}

// forward declaration
namespace autograd {
class AutogradRuntime;
}

namespace exec {

using nnvm::Graph;

// graph executors
class GraphExecutor : public Executor {
 public:
  friend class autograd::AutogradRuntime;
  using Executor::MonitorCallback;

  virtual ~GraphExecutor();
  void Forward(bool is_train) override;
  void PartialForward(bool is_train, int step, int *step_left) override;
  void Backward(const std::vector<NDArray> &head_grads) override;
  const std::vector<NDArray>& outputs() const override;
  void Print(std::ostream &os) const override; // NOLINT(*)
  void SetMonitorCallback(const MonitorCallback& callback) override;
  // initialized the executor
  void Init(nnvm::Symbol symbol,
            const Context& default_ctx,
            const std::map<std::string, Context>& ctx_map,
            const std::vector<NDArray>& in_args,
            const std::vector<NDArray>& arg_grad_store,
            const std::vector<OpReqType>& grad_req_type,
            const std::vector<NDArray>& aux_states,
            Executor* shared_exec = nullptr,
            const nnvm::NodeEntryMap<NDArray>& feed_dict
              = nnvm::NodeEntryMap<NDArray>());

 protected:
  // Information about operational node
  struct OpNode {
    // The name of the operator
    const char* opr_name;
    // the context of the node
    Context ctx;
    // The executor
    std::shared_ptr<OpExecutor> exec;
    // skip the execution of this node
    bool skip_exec_node{false};
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
    // cached const vars, used for seg ops creation
    std::vector<Engine::VarHandle> use_vars;
    // cached mutate vars, used for seg ops creation
    std::vector<Engine::VarHandle> mutate_vars;
  };
  // a cached segment operator that executes a segment
  struct CachedSegOpr {
    // context of the operator
    Context ctx;
    // begin in topo order
    size_t topo_start;
    // end in topo order
    size_t topo_end;
    // the cached operator
    Engine::OprHandle opr = nullptr;
    // list of op executors
    std::vector<OpExecutor*> exec_list;
  };

  // internal initialization of the graph.
  Graph InitGraph(nnvm::Symbol symbol,
                  const Context& default_ctx,
                  const std::map<std::string, Context>& ctx_map,
                  const std::vector<NDArray>& in_args,
                  const std::vector<NDArray>& arg_grad_store,
                  const std::vector<OpReqType>& grad_req_type,
                  const std::vector<NDArray>& aux_states,
                  const nnvm::NodeEntryMap<NDArray>& feed_dict
                    = nnvm::NodeEntryMap<NDArray>());
  // initialize the full graph, including gradient.
  Graph InitFullGraph(nnvm::Symbol symbol,
                      const std::vector<OpReqType>& grad_req_type,
                      const std::vector<NDArray>& arg_grad_store);
  // initialize the cached operator
  void InitCachedOps();
  // initialize the opr segments for bulk exec
  void InitOpSegs();
  // initialize the resources in the graph
  // initialize the memory of data entries
  // shared_pool: extra memory shared from other parts
  void InitDataEntryMemory(std::vector<NDArray>* shared_pool);
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
  /*!
   * \brief Try to create a cached operator to run segments between start and end
   * \param topo_start beginning of segment
   * \param topo_end end of segment
   * \return the cached operator.
   *  ret.opr Can be nullptr if creation failed.
  */
  CachedSegOpr CreateCachedSegOpr(size_t topo_start, size_t topo_end);
  // run the monitor callback for node `nid`
  void ExecuteMonCallback(size_t nid);

  // internal graph
  nnvm::Graph graph_;
  // operator node
  std::vector<OpNode> op_nodes_;
  // internal data entry of each node
  std::vector<NDArray> data_entry_;
  // internal data pool of allocated entries
  std::vector<NDArray> data_pool_;
  // output arrays
  std::vector<NDArray> output_arrays_;
  // gradient store
  std::vector<std::pair<OpReqType, NDArray> > grad_store_;
  // array to hold head gradient.
  std::vector<NDArray> head_grad_array_;
  // entry to hold head gradient
  std::vector<nnvm::NodeEntry> head_grad_entry_;
  // the index map of entry to map.
  std::unordered_map<const nnvm::Node*, size_t> head_grad_map_;
  // number of outputs.
  size_t num_forward_outputs_{0};
  // number of inputs
  size_t num_forward_inputs_{0};
  // number of forward nodes
  size_t num_forward_nodes_{0};
  // saved operator for autograd
  NodeOperatorMap saved_opr_;
  // monitor call back
  std::function<void(const char*, void*)> monitor_callback_{nullptr};
  // whether to enable bulk execution
  bool prefer_bulk_execution_;
  // cached segment operator
  std::vector<CachedSegOpr> cached_seg_opr_;
};

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
