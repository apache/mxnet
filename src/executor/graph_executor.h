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
namespace exec {

using nnvm::Graph;

// graph executors
class GraphExecutor : public Executor {
 public:
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
            Executor* shared_exec = nullptr);

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
  };
  // internal initialization of the graph.
  Graph InitGraph(nnvm::Symbol symbol,
                  const Context& default_ctx,
                  const std::map<std::string, Context>& ctx_map,
                  const std::vector<NDArray>& in_args,
                  const std::vector<NDArray>& arg_grad_store,
                  const std::vector<OpReqType>& grad_req_type,
                  const std::vector<NDArray>& aux_states);
  // initialize the full graph, including gradient.
  Graph InitFullGraph(nnvm::Symbol symbol,
                      const std::vector<OpReqType>& grad_req_type,
                      const std::vector<NDArray>& arg_grad_store);
  // initialize the cached operator
  void InitCachedOps();
  // initialize the resources in the graph
  // initialize the memory of data entries
  // shared_pool: extra memory shared from other parts
  void InitDataEntryMemory(const std::vector<NDArray>& shared_pool);
  // run ops from topo order start to end
  void RunOps(bool is_train, size_t topo_start, size_t topo_end);
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
  // monitor call back
  std::function<void(const char*, void*)> monitor_callback_{nullptr};
};

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
