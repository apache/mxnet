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
class Variable;

/*!
 * \brief Inner representation of operator.
 */
class Operator;

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
  using Variable = engine::Variable*;
  /*!
   * \brief Operator of the engine.
   */
  using Operator = engine::Operator*;
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
  virtual Operator NewOperator(AsyncFn fn,
                               std::vector<Variable> const& use_vars,
                               std::vector<Variable> const& mutate_vars) = 0;
  /*!
   * \brief Delete the given operator.
   * \param op The operator to delete.
   */
  virtual void DeleteOperator(Operator op) = 0;
  /*!
   * \brief Push an operator to the engine.
   * \param op The operator to push.
   * \param exec_ctx Execution context.
   */
  virtual void Push(Operator op, Context exec_ctx) = 0;
  /*!
   * \brief Push an synchronous operation to the DAG engine.
   * \param exec_fun Execution function that executes the operation.
   * \param exec_ctx Execution context.
   * \param use_vars The variables that current operation will use but not
   *                 mutate.
   * \param mutate_vars The variables that current operation will mutate.
   */
  void Push(Fn exec_fun, Context exec_ctx,
            std::vector<Variable> const& use_vars,
            std::vector<Variable> const& mutate_vars);
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
   * \brief Wait for variable.
   * \param var The variable we should wait for, this function returns when all
   *            the operations related to var has been completed.
   */
  virtual void WaitForVar(Variable var) = 0;
  /*!
   * \brief Wait until all the activity of dag engine finishes.
   */
  virtual void WaitForAll() = 0;
  /*!
   * \brief Virtual destructor.
   */
  virtual ~DAGEngine();
  /*!
   * \return DAG engine singleton.
   */
  static DAGEngine* Get();

 protected:
  /*!
   * \brief Hidden constructors.
   */
  DAGEngine();

 private:
  DISALLOW_COPY_AND_ASSIGN(DAGEngine);
};

}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_H_
