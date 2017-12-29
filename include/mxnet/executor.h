/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file executor.h
 * \brief Symbolic executor interface of mxnet.
 * \author Min Lin, Bing Xu
 */
#ifndef MXNET_EXECUTOR_H_
#define MXNET_EXECUTOR_H_

#include <dmlc/base.h>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include "./base.h"
#include "./c_api.h"
#include "./ndarray.h"
#include "./operator.h"

// check c++11
#if DMLC_USE_CXX11 == 0
#error "CXX11 was required for symbolic module"
#endif

namespace mxnet {
/*! \brief use symbolic graph from NNVM */
using nnvm::Symbol;

/*!
 * \brief Executor of a computation graph.
 *  Executor can be created by Binding a symbol.
 */
class Executor {
 public:
  /*! \brief destructor */
  virtual ~Executor() {}
  /*!
   * \brief Perform a Forward operation of Operator
   *  After this operation, user can get the result by using function head.
   */
  virtual void Forward(bool is_train) = 0;
  /*!
   * \brief Perform a Partial Forward operation of Operator.
   *  Only issue operation specified by step.
   *  The caller must keep calling PartialForward with increasing steps, until step_left=0.
   * \param is_train Whether this is training phase.
   * \param step current step, user can always start from 0
   * \param step_left Number of steps left to finish the forward.
   */
  virtual void PartialForward(bool is_train, int step, int *step_left) = 0;
  /*!
   * \brief Perform a Backward operation of the Operator.
   *  This must be called after Forward.
   *  After this operation, NDArrays specified by grad_in_args_store will be updated accordingly.
   *  User is allowed to pass in an empty Array if the head node is
   *  loss function and head gradeitn is not needed.
   *
   * \param head_grads the gradient of head nodes to be backproped.
   */
  virtual void Backward(const std::vector<NDArray> &head_grads, bool is_train = true) = 0;
  /*!
   * \brief print the execution plan info to output stream.
   * \param os the output stream we like to print to.
   */
  virtual void Print(std::ostream &os) const {} // NOLINT(*)
  /*!
   * \brief get array of outputs in the executor.
   * \return array of outputs in the executor.
   */
  virtual const std::vector<NDArray> &outputs() const = 0;
  /*!
   * \brief get input argument map, key is arg name, value is arg's NDArray.
   * \return input argument map in the executor.
   */
  virtual const std::unordered_map<std::string, NDArray>& in_arg_map() const = 0;
  /*!
   * \brief get input argument graident map, key is arg name, value is gradient's NDArray.
   * \return input argument gradient map in the executor.
   */
  virtual const std::unordered_map<std::string, NDArray>& arg_grad_map() const = 0;
  /*!
   * \brief get aux state map, key is arg name, value is aux state's NDArray.
   * \return aux state map in the executor.
   */
  virtual const std::unordered_map<std::string, NDArray>& aux_state_map() const = 0;
  /*!
   * \brief Create an operator by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *
   * \param default_ctx the default context of binding.
   * \param group2ctx Context mapping group to context.
   * \param symbol the symbol that specifies the output of Forward pass.
   * \param in_args the NDArray that stores the input arguments to the symbol.
   * \param arg_grad_store NDArray that is used to store the gradient output of the input arguments.
   * \param grad_req_type requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   * \param aux_states NDArray that is used as internal state in op
   * \param shared_exec input executor to share memory with.
   * \return a new executor.
   */
  static Executor *Bind(nnvm::Symbol symbol,
                        const Context& default_ctx,
                        const std::map<std::string, Context>& group2ctx,
                        const std::vector<NDArray> &in_args,
                        const std::vector<NDArray> &arg_grad_store,
                        const std::vector<OpReqType> &grad_req_type,
                        const std::vector<NDArray> &aux_states,
                        Executor* shared_exec = NULL);

  static Executor* SimpleBind(nnvm::Symbol symbol,
                              const Context& default_ctx,
                              const std::map<std::string, Context>& group2ctx,
                              const std::vector<Context>& in_arg_ctxes,
                              const std::vector<Context>& arg_grad_ctxes,
                              const std::vector<Context>& aux_state_ctxes,
                              const std::unordered_map<std::string, TShape>& arg_shape_map,
                              const std::unordered_map<std::string, int>& arg_dtype_map,
                              const std::unordered_map<std::string, int>& arg_stype_map,
                              const std::vector<OpReqType>& grad_req_types,
                              const std::unordered_set<std::string>& param_names,
                              std::vector<NDArray>* in_args,
                              std::vector<NDArray>* arg_grads,
                              std::vector<NDArray>* aux_states,
                              std::unordered_map<std::string, NDArray>*
                                shared_data_arrays = nullptr,
                              Executor* shared_exec = nullptr);
  /*!
   * \brief the prototype of user-defined monitor callback
   */
  typedef std::function<void(const char*, void*)> MonitorCallback;
  /*!
   * \brief Install a callback to notify the completion of operation.
   */
  virtual void SetMonitorCallback(const MonitorCallback& callback) {}
};  // class executor
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_H_
