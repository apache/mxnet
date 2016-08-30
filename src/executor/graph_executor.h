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
#include <mxnet/symbolic.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass_functions.h>
#include <map>
#include <string>
#include <vector>

namespace mxnet {
namespace exec {

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
  // executor to execute an operator.
  class OpExecutor {
   public:
    // virtual destructor
    virtual ~OpExecutor() {}
    // initialize the operator
    virtual void Init() = 0;
    // run the operator given runtime context
    virtual void Run(RunContext rctx) = 0;
    // input arrays
    std::vector<NDArray> in_array;
    // out arrays
    std::vector<NDArray> out_array;
    // the output requirement
    std::vector<OpReqType> req;
    // runtime op context, contains allocated resources.
    OpContext op_ctx;
  };
  class ForwardOpExecutor;
  class BackwardOpExecutor;
  // Information about operational node
  struct OpNode {
    // the context of the node
    Context ctx;
    // The executor
    std::shared_ptr<OpExecutor> exec;
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
  };
  // internal initialization of the graph.
  void InitGraph(nnvm::Symbol symbol,
                 const Context& default_ctx,
                 const std::map<std::string, Context>& ctx_map,
                 const std::vector<NDArray>& in_args,
                 const std::vector<NDArray>& arg_grad_store,
                 const std::vector<OpReqType>& grad_req_type,
                 const std::vector<NDArray>& aux_states);
  nnvm::Graph InitGradGraph(nnvm::Symbol symbol,
                            const std::vector<OpReqType>& grad_req_type,
                            const std::vector<NDArray>& arg_grad_store);
  // intitialize the operator executors on each node
  void InitOpExecs();
  // intitialize the operator executors on each node
  void InitResources();
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
  // data structure about entries
  std::vector<OpNode> op_nodes_;
  // context of each data entry
  std::vector<Context> data_context_;
  // internal data entry of each node
  std::vector<NDArray> data_entry_;
  // internal data pool of allocated entries
  std::vector<NDArray> data_pool_;
  // output arrays
  std::vector<NDArray> output_arrays_;
  // gradient store
  std::vector<std::pair<OpReqType, NDArray> > grad_store_;
  // head gradient entry
  std::vector<nnvm::NodeEntry> head_grad_entry_;
  std::unordered_map<const nnvm::Node*, size_t> head_grad_map_;
  // number of outputs.
  size_t num_outputs_;
  // number of inputs
  size_t num_inputs_;
  // number of forward nodes
  size_t num_forward_nodes_;
  // total number of allocated temp space.
  size_t total_allocated_temp_{0};
};

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_GRAPH_EXECUTOR_H_
