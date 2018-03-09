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
 *  Copyright (c) 2016 by Contributors
 * \file op_attr_types.h
 * \brief Additional operator attributes
 *  beside the ones provided by NNVM
 */
#ifndef MXNET_OP_ATTR_TYPES_H_
#define MXNET_OP_ATTR_TYPES_H_

#include <mshadow/tensor.h>
#include <nnvm/op_attr_types.h>

#include <vector>
#include <functional>

#include "./base.h"
#include "./ndarray.h"
#include "./engine.h"
#include "./resource.h"

namespace mxnet {

using nnvm::NodeAttrs;

/*! \brief operation request type to Forward and Backward */
enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
   * \brief perform an inplace write,
   * Target shares memory with one of input arguments.
   * This option only happen when
   */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

/*!
 * \brief All the possible information needed by Operator.Forward and Backward
 *  This is the superset of RunContext.
 *  We use this data structure to bookkeep everything needed by Forward and Backward.
 * \sa Resource
 */
struct OpContext {
  /*! \brief whether it is training phase */
  int is_train;
  /*! \brief RunContext related resources */
  RunContext run_ctx;
  /*! \brief the callback when operation completes, used by asynchronize ops */
  engine::CallbackOnComplete async_on_complete;
  /*! \brief Resources requested by the operator */
  std::vector<Resource> requested;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return run_ctx.get_stream<xpu>();
  }
};

/*! \brief the execution type of the operator */
enum class ExecType {
  /*! \brief Forward/Backward are synchronize calls */
  kSync,
  /*!
   * \brief Forward/Backward are asynchronize,
   *  will call OpContext.async_on_complete when operation finishes.
   */
  kAsync,
  /*!
   * \brief Cross device copy operation, this is a special operator
   *  That indicates copy across devices, the input and output can sit on different device.
   *  In current implementation, copy operator is specially handled by executor.
   *  This flag is used for special case treatment and future extension of different copy ops.
   */
  kCrossDeviceCopy
};

/*! \brief the dispatch mode of the operator */
enum class DispatchMode {
  kUndefined = -1,
  // dispatch on FCompute or FStatefulCompute
  kFCompute,
  // dispatch on FComputeEx or FStatefulComputeEx, if available
  kFComputeEx,
  // dispatch on FCompute or FStatefulCompute, and performs storage fallback
  kFComputeFallback,
  // special dispatch mode for variables
  kVariable,
};

/*!
 * \brief Operator state. This is a pointer type, its content is mutable
 *  even if OpStatePtr is const.
 */
class OpStatePtr {
 public:
  /* \brief Create a OpStatePtr with state of type T.
   * \param args Arguments passed to T's constructor.
   */
  template<typename T, typename... Args>
  static OpStatePtr Create(Args&&... args) {
    OpStatePtr ret;
    ret.ptr_ = std::make_shared<OpState>();
    ret.ptr_->var_ = Engine::Get()->NewVariable();
    ret.ptr_->state_.construct<T>(std::forward<Args>(args)...);

    return ret;
  }
  /* \brief Get engine variable associated with this state */
  engine::VarHandle get_var() const {
    return ptr_->var_;
  }
  /* \brief Get state of type T */
  template<typename T>
  T& get_state() const {
    return dmlc::get<T>(ptr_->state_);
  }
  /* \brief clear state */
  void reset() {
    ptr_.reset();
  }
  /* \brief Whether state is empty */
  explicit operator bool() const {
    return ptr_ ? true : false;
  }

 private:
  /* \brief state structure */
  struct OpState {
    OpState() {}
    OpState(const OpState& other) = delete;
    OpState& operator=(const OpState& other) = delete;

    ~OpState() {
      Engine::Get()->DeleteVariable([](RunContext s) {}, Context::CPU(), var_);
    }

    engine::VarHandle var_;
    dmlc::any state_;
  };
  /* \brief shared pointer to state */
  std::shared_ptr<OpState> ptr_;
};

/*!
 * \brief Create a Layer style, forward/backward operator.
 *  This is easy to write code that contains state.
 *  OpStatePtr is a pointer type, it's content is mutable even if
 *  OpStatePtr is constant.
 *
 *
 *  This is not the only way to register an op execution function.
 *  More simpler or specialized operator form can be registered
 *
 *  \note Register under "FCreateLayerOp"
 */
using FCreateOpState = std::function<OpStatePtr (const NodeAttrs& attrs,
                                                 Context ctx,
                                                 const std::vector<TShape>& in_shape,
                                                 const std::vector<int>& in_type)>;
/*!
 * \brief Execution mode of this operator.
 */
using FExecType = std::function<ExecType (const NodeAttrs& attrs)>;
/*!
 * \brief Resiger a compute function for stateful operator.
 *  OpStatePtr is a pointer type, it's content is mutable even if
 *  OpStatePtr is constant.
 *
 * \note Register under "FStatefulCompute<cpu>" and "FStatefulCompute<gpu>"
 */
using FStatefulCompute = std::function<void (const OpStatePtr& state,
                                             const OpContext& ctx,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs)>;
/*!
 * \brief Resiger a compute function for stateful operator using NDArray interface.
 *  OpStatePtr is a pointer type, it's content is mutable even if
 *  OpStatePtr is constant.
 *
 * \note Register under "FStatefulComputeEx<cpu>" and "FStatefulComputeEx<gpu>"
 */
using FStatefulComputeEx = std::function<void (const OpStatePtr& state,
                                               const OpContext& ctx,
                                               const std::vector<NDArray>& inputs,
                                               const std::vector<OpReqType>& req,
                                               const std::vector<NDArray>& outputs)>;
/*!
 * \brief The resource request from the operator
 *
 * \note Register under "FResourceRequest"
 */
using FResourceRequest = std::function<
  std::vector<ResourceRequest> (const NodeAttrs& n)>;
/*!
 * \brief Register an operator called as a NDArray function
 *
 * \note Register under "FNDArrayFunction"
 */
using FNDArrayFunction = std::function<void (const nnvm::NodeAttrs& attrs,
                                             const std::vector<NDArray>& inputs,
                                             std::vector<NDArray>* outputs)>;
/*!
 * \brief Resiger a compute function for simple stateless forward only operator
 *
 * \note Register under "FCompute<cpu>" and "FCompute<gpu>"
 */
using FCompute = std::function<void (const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs)>;
/*!
 * \brief Resiger an NDArray compute function for simple stateless forward only operator
 *
 * \note Register under "FComputeEx<xpu>" and "FComputeEx<xpu>"
 *       Dispatched only when inferred dispatch_mode is FDispatchComputeEx
 */
using FComputeEx = std::function<void (const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs)>;

/*!
 * \brief Resiger a storage and dispatch mode inference function based on
 *        storage types of the inputs and outputs, and the dev_mask for the operator.
 *
 * \note Register under "FInferStorageType"
 */
using FInferStorageType = std::function<bool (const NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs)>;

}  // namespace mxnet

#endif  // MXNET_OP_ATTR_TYPES_H_
