/*!
 *  Copyright (c) 2015 by Contributors
 * \file dag_engine.h
 * \brief dynamic data-flow dag engine that schedules
 *  operations in a concurrent way
 */
#ifndef MXNET_DAG_ENGINE_H_
#define MXNET_DAG_ENGINE_H_
#include <dmlc/base.h>
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for narray module"
#endif

#include <functional>
#include <vector>
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {
/*!
 * \brief dynamic data-flow dag engine that schedules
 *  operations in a concurrent way
 */
class DAGEngine {
 public:
  /*!
   * \brief operation to pass to DAG engine
   * \param ctx runtime context
   */
  typedef std::function<void(RunContext rctx)> Op;
  /*! \brief callback function to notify operation complete */
  typedef std::function<void()> Callback;
  /*!
   * \brief operation to pass to DAG engine
   * \param ctx runtime context
   * \param on_complete a callback function used to notify the engine the action completes
   */
  typedef std::function<void(RunContext ctx, Callback on_complete)> AsyncOp;
  /*!
   * \brief variable of dag engine, used to specify dependencies
   *  defined to be a pointer, that can points to a internal data structure
   *  of the engine itself
   *
   *  Design detail: we choose pointer instead of some ID to avoid
   *  indirect map lookup. usually, Variable directly points to the content we need
   */
  typedef void *Variable;
  /*!
   * \brief Push an asynchronize operation to the DAG engine
   * \param exec_fun execution funtion, this function takes a parameter on_complete
   *  that must be called when the execution completes. For synchronize operations
   * \param exec_ctx execution context
   * \param use_vars the variables that current operation will use(but not mutate)
   * \param mutate_vars the variables that current operation will mutate
   */
  virtual void Push(AsyncOp exec_fun,
                    Context exec_ctx,
                    const std::vector<Variable> &use_vars,
                    const std::vector<Variable> &mutate_vars) = 0;
  /*!
   * \brief Push an synchronize operation to the DAG engine
   * \param exec_fun execution funtion that executes the operation
   * \param exec_ctx execution context
   * \param use_vars the variables that current operation will use(but not mutate)
   * \param mutate_vars the variables that current operation will mutate
   */
  virtual void Push(Op exec_fun,
                    Context exec_ctx,
                    const std::vector<Variable> &use_vars,
                    const std::vector<Variable> &mutate_vars) {
    this->Push([exec_fun](RunContext ctx, Callback on_complete) {
        exec_fun(ctx); on_complete();
      }, exec_ctx, use_vars, mutate_vars);
  }
  /*!
   * \brief schedule the delete of variable var,
   * The delete will not happen immediately, but will wait until all the operations
   * depending on var is completed
   *
   * \param delete_fun a function that will be called after var is deleted
   * \param exec_ctx execution context
   * \param var the variable to be deleted
   */
  virtual void PushDelete(Op delete_fun,
                          Context exec_ctx,
                          Variable var) = 0;
  /*!
   * \brief allocate a new variable, the variable can then
   *  be used to schedul the operation concurrently via dependency patterns
   * \return thew new variable allocated
   */
  virtual Variable NewVar() = 0;
  /*!
   * \brief wait for variable var
   * \param var the variable we should wait for, this function returns when all the operations
   *  related to var has been completed
   */
  virtual void WaitForVar(Variable var) {}
  /*! \brief wait until all the activity of dag engine finishes */
  virtual void WaitForAll() {}
  /*! \return DAG engine singleton */
  static DAGEngine *Get();
};
}  // namespace mxnet
#endif  // MXNET_DAG_ENGINE_H_
