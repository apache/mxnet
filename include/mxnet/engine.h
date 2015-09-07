/*!
 * Copyright (c) 2015 by Contributors
 * \file engine.h
 * \brief Engine that schedules data.
 */
#ifndef MXNET_ENGINE_H_
#define MXNET_ENGINE_H_
#include <dmlc/base.h>

#if DMLC_USE_CXX11 == 0
#error "C++11 was required for engine module."
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
 * \brief Dynamic dataflow engine that schedules operations.
 */
class Engine {
 public:
  /*!
   * \brief Operation to pass to engine.
   */
  using Fn = std::function<void(RunContext)>;
  /*!
   * \brief Callback function to notify operation complete.
   */
  using Callback = std::function<void()>;
  /*!
   * \brief Asynchronous operation to pass to engine.
   */
  using AsyncFn = std::function<void(RunContext, Callback)>;
  /*!
   * \brief Variable of engine, used to specify dependencies defined to be a
   *        pointer, that points to an internal data structure of the engine
   *        itself.
   */
  using VarHandle = engine::Var*;
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
  virtual VarHandle NewVariable() = 0;
  /*!
   * \brief Create a new operator. The returned operator could be saved
   *        externally so that it could be resued for scheduling.
   * \param fn The execution function.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   * \return The new operator allocated.
   */
  virtual OprHandle NewOperator(AsyncFn fn,
                                std::vector<VarHandle> const& const_vars,
                                std::vector<VarHandle> const& mutable_vars) = 0;
  /*!
   * \brief Delete the given operator.
   * \param op The operator to delete.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations using this operator are completed.
   */
  virtual void DeleteOperator(OprHandle op) = 0;
  /*!
   * \brief Push an operator to the engine.
   * \param op The operator to push.
   * \param exec_ctx Execution context.
   */
  virtual void Push(OprHandle op, Context exec_ctx) = 0;
  /*!
   * \brief Push an synchronous operation to the engine.
   * \param exec_fun Execution function that executes the operation.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   */
  virtual void Push(Fn exec_fun, Context exec_ctx,
                    std::vector<VarHandle> const& const_vars,
                    std::vector<VarHandle> const& mutable_vars) = 0;
  /*!
   * \brief Push an asynchronous operation to the engine.
   * \param exec_fun Execution function, this function takes a parameter
   *                 on_complete that must be called when the execution
   *                 completes.
   * \param exec_ctx Execution context.
   * \param const_vars The variables that current operation will use but not
   *                   mutate.
   * \param mutable_vars The variables that current operation will mutate.
   */
  virtual void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                         std::vector<VarHandle> const& const_vars,
                         std::vector<VarHandle> const& mutable_vars) = 0;
  /*!
   * \brief Schedule the delete of a variable.
   *
   * The delete will not happen immediately, but will wait until all the
   * operations depending on var are completed.
   *
   * \param delete_fun A function that will be called after the variable is
   *                   deleted.
   * \param exec_ctx Execution context.
   * \param var The variable to be deleted.
   */
  virtual void DeleteVariable(Fn delete_fun, Context exec_ctx,
                              VarHandle var) = 0;
  /*!
   * \brief Wait for a variable.
   * \param var The variable we should wait for. This function returns when the
   *            variable is ready.
   */
  virtual void WaitForVar(VarHandle var) = 0;
  /*!
   * \brief Wait until all the activity of engine finishes.
   */
  virtual void WaitForAll() = 0;
  /*!
   * \brief Virtual destructor.
   */
  virtual ~Engine() noexcept(false);
  /*!
   * \return Engine singleton.
   */
  static Engine* Get();
};  // class Engine

}  // namespace mxnet

#endif  // MXNET_ENGINE_H_
