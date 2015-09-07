/*!
 * Copyright (c) 2015 by Contributors
 * \file dag_engine.h
 * \brief DAG engine that schedules data.
 */
#ifndef MXNET_DAG_ENGINE_H_
#define MXNET_DAG_ENGINE_H_
#include <dmlc/base.h>

#if DMLC_USE_CXX11 == 0
#error "C++11 was required for DAG engine module."
#endif

#include <functional>
#include <vector>
#include "base.h"
#include "context.h"

namespace mxnet {

/*!
 * \brief Namespace of engine implementation.
 */
namespace engine {

/*!
 * \brief Inner representation of variable.
 */
struct Var;

/*!
 * \brief Inner representation of operator.
 */
struct Opr;

}  // namespace engine

/*!
 * \brief Dynamic dataflow DAG engine that schedules operations.
 */
class DAGEngine {
 public:
  /*!
   * \brief Operation to pass to DAG engine.
   */
  using Fn = std::function<void(RunContext)>;
  /*!
   * \brief Callback function to notify operation complete.
   */
  using Callback = std::function<void()>;
  /*!
   * \brief Asynchronous operation to pass to DAG engine.
   */
  using AsyncFn = std::function<void(RunContext, Callback)>;
  /*!
   * \brief Variable of dag engine, used to specify dependencies defined to be a
   *        pointer, that points to an internal data structure of the engine
   *        itself.
   */
  using Variable = engine::Var*;
  /*!
   * \brief Operator of the engine.
   */
  using OprHandle = engine::Opr*;
  /*!
   * \brief Allocate a new variable, the variable can then
   *        be used to schedule the operation concurrently via dependency
   *        patterns.
   * \return The new variable allocated.
   */
  virtual Variable NewVar() = 0;
  /*!
   * \brief Create a new operator. The returned operator could be saved
   *        externally so that it could be resued for scheduling.
   * \param fn The execution function.
   * \param use_vars The variables that current operation will use but not
   *                 mutate.
   * \param mutate_vars Teh variables that current operation will mutate.
   * \return The new operator allocated.
   */
  virtual OprHandle NewOperator(AsyncFn fn,
                                std::vector<Variable> const& use_vars,
                                std::vector<Variable> const& mutate_vars) = 0;
  /*!
   * \brief Delete the given operator.
   * \param op The operator to delete.
   */
  virtual void DeleteOperator(OprHandle op) = 0;
  /*!
   * \brief Push an operator to the engine.
   * \param op The operator to push.
   * \param exec_ctx Execution context.
   */
  virtual void Push(OprHandle op, Context exec_ctx) = 0;
  /*!
   * \brief Push an synchronous operation to the DAG engine.
   * \param exec_fun Execution function that executes the operation.
   * \param exec_ctx Execution context.
   * \param use_vars The variables that current operation will use but not
   *                 mutate.
   * \param mutate_vars The variables that current operation will mutate.
   */
  virtual void Push(Fn exec_fun, Context exec_ctx,
                    std::vector<Variable> const& use_vars,
                    std::vector<Variable> const& mutate_vars) = 0;
  /*!
   * \brief Push an asynchronous operation to the DAG engine.
   * \param exec_fun Execution function, this function takes a parameter
   *                 on_complete that must be called when the execution
   *                 completes.
   * \param exec_ctx Execution context.
   * \param use_vars The variables that current operation will use but not
   *                 mutate.
   * \param mutate_vars The variables that current operation will mutate.
   */
  virtual void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                         std::vector<Variable> const& use_vars,
                         std::vector<Variable> const& mutate_vars) = 0;
  /*!
   * \brief Schedule the delete of a variable.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations depending on var is completed.
   *
   * \param delete_fun A function that will be called after the variable is
   *                   deleted.
   * \param exec_ctx Execution context.
   * \param var The variable to be deleted.
   */
  virtual void PushDelete(Fn delete_fun, Context exec_ctx, Variable var) = 0;
  /*!
   * \brief Wait to read a variable.
   *
   *  The caller should read the content immediately in a synchronized way,
   *  before any subsequent write operations are issued.
   *  The subsequent write operations to the variable can destroy the content.
   *
   * \param var The variable we should wait for,
   *            This function returns when all the write operations to this
   *            var has been completed.
   */
  virtual void WaitToRead(Variable var) = 0;
  /*!
   * \brief Wait to write a variable.
   *
   *  The caller should rwrite the content immediately in a synchronized way,
   *  before any subsequent write operations are issued.
   *  The subsequent write operations to the variable can destroy the content.
   *
   * \param var The variable we should wait for,
   *            This function returns when all the read/write operations
   *            on var has been completed.
   */
  virtual void WaitToWrite(Variable var) = 0;
  /*!
   * \brief Wait until all the activity of dag engine finishes.
   */
  virtual void WaitForAll() = 0;
  /*!
   * \brief Virtual destructor.
   */
  virtual ~DAGEngine() noexcept(false) {}
  /*!
   * \return DAG engine singleton.
   */
  static DAGEngine* Get();

  // remove DISALLOW_COPY_AND_ASSIGN since this is virtual class.
};  // class DAGEngine

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_H_
