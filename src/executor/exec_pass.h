/*!
 * Copyright (c) 2016 by Contributors
 * \file exec_pass.h
 * \brief All the execution related pass and data structures.
 */
#ifndef MXNET_EXECUTOR_EXEC_PASS_H_
#define MXNET_EXECUTOR_EXEC_PASS_H_

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <vector>
#include <memory>

namespace mxnet {
namespace exec {

/*! \brief reuse graph definition */
using nnvm::Graph;

/*!
 * \brief executor to execute an operator
 * This is a graph executor dependent interface
 * that unifies all the operator
 */
class OpExecutor {
 public:
  /*! \brief input arrays */
  std::vector<NDArray> in_array;
  /*! \brief output data arrays */
  std::vector<NDArray> out_array;
  /*! \brief output requirement on each array */
  std::vector<OpReqType> req;
  /*! \brief runtime op context, contains allocated resources */
  OpContext op_ctx;
  /*! \brief virtual destructor */
  virtual ~OpExecutor() {}
  /*!
   * \brief Setup the executor for given NDArray member
   * this can be called multiple times if NDArray changed during reshape.
   *  It is safe to call it via asynchronize engine lambda
   */
  virtual void Setup() = 0;
  /*!
   * \brief run the operator given runtime context on device.
   *  This function call do not synchronize the stream.
   * \param rctx The runtime context passed in by environment.
   */
  virtual void Run(RunContext rctx) = 0;
  /*! \return the execution type */
  virtual Operator::ExecType exec_type() const = 0;
};

/*!
 * \brief per node vector of operator executors.
 * \note stored under attribute "op_exec"
 */
using OpExecVector = std::vector<std::shared_ptr<OpExecutor> >;

/*!
 * \brief per node context vector
 * \node stored under "context"
 */
using ContextVector = std::vector<Context>;

/*!
 * \brief Attach OpExecutor to the graph attributes.
 *
 * \param g input graph
 * \return graph with new attribute "op_exec" of type OpExecVector
 *  The fields on the OpExecVector are not yet been setup.
 */
Graph AttachOpExecs(Graph g);

/*!
 * \brief Attach Resource to the OpExecVector of the graph.
 *
 * \param g input graph need to contain op_exec attribute.
 *
 * \return graph with new attribute "op_exec" of type OpExecVector
 *  The fields on the OpExecVector are not yet been setup.
 */
Graph AttachOpResources(Graph g);

/*!
 * \brief Discover chance of inplace addto operators.
 *  i.e. z = plus(z, source_op), and encourage it to become z += source_op.
 *
 * This optimization is coupled with executor. This is helpful to reduce memory
 * and computation for gradient aggregation of RNN.
 *
 * Require storage placement to be already finished.
 *
 * \param g input graph need to contain op_exec attribute.
 *
 * \return graph two new attributes, changes attribute "storage_id".
 *  - "addto_entry", std::vector<bool> size=g.num_node_entries()
 *    - addto_entry[eid] == 1, the corresponding op need to be performed using req=kAddTo
 *  - "skip_plus_node", std::vector<int> if set to 1, current op's execution is skiped.
 */
Graph DetectInplaceAddTo(Graph g);

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_EXEC_PASS_H_
