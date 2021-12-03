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
#include <string>

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
   * This option only happen when
   * Target shares memory with one of input arguments.
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
  /*! \brief whether there is a backward phase to compute gradients. */
  bool need_grad;
  /*! \brief whether it is training phase */
  bool is_train;
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
#if MXNET_USE_CUDA
  /*!
   * \brief get auxilary gpu stream auto-syncing object from Context
   * \return the aux stream auto-syncing object
   */
  inline SyncedGPUAuxStream get_gpu_aux_stream() const {
    return run_ctx.get_gpu_aux_stream();
  }
#endif
};

/*! \brief the execution type of the operator */
enum class ExecType {
  /*! \brief Forward/Backward are synchronous calls */
  kSync,
  /*!
   * \brief Forward/Backward are asynchronous,
   *  will call OpContext.async_on_complete when operation finishes.
   */
  kAsync,
  /*!
   * \brief Cross device copy operation, this is a special operator that indicates it will copy
   * across devices. For example the input and output for this type of operator can potentially
   * reside on different devices.  In the current implementation, a copy operator is specially
   * handled by an executor. This flag is used for special case treatment and future extension of
   * different copy ops.
   */
  kCrossDeviceCopy,
  /*!
   * \brief A subgraph execution should happen in the main thread, instead of
   *  in the execution engine.
   */
  kSubgraphExec,
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

/*! \brief the quantization type of the operator */
enum class QuantizeType {
  // This operator doesn't support quantization
  kNone = 0,
  // This operator can get huge benefit from quantization, thus must be quantized
  kMust,
  // This operator support quantization, but will be decided depending on the connection
  kSupport,
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
    auto state = new T(std::forward<Args>(args)...);
    auto var = Engine::Get()->NewVariable();
    ret.ptr_.reset(
      new OpState(var, state),
      [](OpState* p) {
        Engine::Get()->DeleteVariable([](RunContext s) {}, Context::CPU(), p->var);
        delete reinterpret_cast<T*>(p->state);
        delete p;
      });

    return ret;
  }
  /* \brief Get engine variable associated with this state */
  engine::VarHandle get_var() const {
    return ptr_->var;
  }
  /* \brief Get state of type T */
  template<typename T>
  T& get_state() const {
    return *reinterpret_cast<T*>(ptr_->state);
  }
  /* \brief clear state */
  void reset() {
    ptr_.reset();
  }
  /* \brief checks whether the managed object is managed only by the current
            OpStatePtr instance */
  bool unique() const {
    return ptr_.unique();
  }
  /* \brief Whether state is empty */
  explicit operator bool() const {
    return ptr_ ? true : false;
  }

 private:
  /* \brief state structure */
  struct OpState {
    engine::VarHandle var;
    void* state;

    OpState(engine::VarHandle var_, void* state_) : var(var_), state(state_) {}
    OpState(const OpState& other) = delete;
    OpState& operator=(const OpState& other) = delete;
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
                                                 const mxnet::ShapeVector& in_shape,
                                                 const std::vector<int>& in_type)>;

/*!
 * \brief Whether the operator always produces the same
 *        output given the same input.
 *        This enables certain optimizations
 *        like common expression elimination.
 *
 * \note Register under "THasDeterministicOutput"
 */
using THasDeterministicOutput = bool;

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
 * \brief The resource request from the operator.
 *        An operator could register ResourceRequestEx, or ResourceRequest, or neither.
 *
 * \note Register under "FResourceRequest"
 */
using FResourceRequest = std::function<
  std::vector<ResourceRequest> (const NodeAttrs& n)>;
/*!
 * \brief The resource request from the operator.
 *        An operator could register ResourceRequestEx, or ResourceRequest, or neither.
 *        If an operator registers both ResourceRequestEx and ResourceRequest,
 *        ResourceRequest is ignored.
 *
 * \note Register under "FResourceRequestEx"
 */
using FResourceRequestEx = std::function<
  std::vector<ResourceRequest> (const NodeAttrs& n,
                                const int dev_mask,
                                const DispatchMode dispatch_mode)>;
/*!
 * \brief Register an operator called as a NDArray function
 *
 * \note Register under "FNDArrayFunction"
 */
using FNDArrayFunction = std::function<void (const nnvm::NodeAttrs& attrs,
                                             const std::vector<NDArray>& inputs,
                                             std::vector<NDArray>* outputs)>;
/*!
 * \brief Register a compute function for simple stateless forward only operator
 *
 * \note Register under "FCompute<cpu>" and "FCompute<gpu>"
 */
using FCompute = std::function<void (const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs)>;
/*!
 * \brief Register an NDArray compute function for simple stateless forward only operator
 * \note Register under "FComputeEx<xpu>" and "FComputeEx<xpu>"
 *       Dispatched only when inferred dispatch_mode is FDispatchComputeEx
 */
using FComputeEx = std::function<void (const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs)>;

/*!
 * \brief Register a storage and dispatch mode inference function based on
 *        storage types of the inputs and outputs, and the dev_mask for the operator.
 *
 * \note Register under "FInferStorageType"
 */
using FInferStorageType = std::function<bool (const NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs)>;

/*!
 * \brief Register a quantized node creation function based on the attrs of the node
 * \note Register under "FQuantizedOp" for non-quantized operators
 */
using FQuantizable = std::function<QuantizeType (const NodeAttrs& attrs)>;

/*!
 * \brief Register a quantized node creation function based on the attrs of the node
 * \note Register under "FQuantizedOp" for non-quantized operators
 */
using FQuantizedOp = std::function<nnvm::ObjectPtr (const NodeAttrs& attrs)>;

/*!
 * \brief Register a function to determine if the output of a quantized operator
 * needs to be requantized. This is usually used for the operators
 * taking int8 data types while accumulating in int32, e.g. quantized_conv.
 * \note Register under "FNeedRequantize" for non-quantized operators
 */
using FNeedRequantize = std::function<bool (const NodeAttrs& attrs)>;

/*!
 * \brief Register a function to determine if the input of a quantized operator
 * needs to be quantized. This is usually used for the quantized operators
 * which can handle fp32 inputs directly.
 */
using FAvoidQuantizeInput = std::function<bool (const NodeAttrs& attrs,
                                                const size_t index,
                                                const std::string quantize_granularity)>;

/*!
 * \brief Register a function to determine if the input of a quantized operator
 * needs to be quantized asymmetrically.
 */
using FNeedAsymQuantizeInput = std::function<bool (const NodeAttrs& attrs,
                                                   const size_t index)>;

/*!
 * \brief Register a function to determine if the output of a quantized operator
 * needs to be dequantized. This is usually used for the quantized operators
 * which can produce fp32 outputs directly.
 */
using FAvoidDequantizeOutput = std::function<bool (const NodeAttrs& attrs,
                                                   const size_t index)>;

/*!
 * \brief Register a function to determine if the input of a quantized operator
 * needs to be calibrated. This is usually used for the quantized operators
 * which need calibration on its input.
 */
using FNeedCalibrateInput = std::function<std::vector<int> (const NodeAttrs& attrs)>;

/*!
 * \brief Register a function to determine if the output of a quantized operator
 * needs to be calibrated. This is usually used for the quantized operators
 * which need calibration on its output.
 */
using FNeedCalibrateOutput = std::function<std::vector<int> (const NodeAttrs& attrs)>;

#if MXNET_USE_CUDA

/*!
 * \brief Register a function to determine if
 * the operator implementation is compatible
 * with CUDA graphs. This requires the execution
 * to stay the same as long as the shape and type
 * of input stays the same.
 */
using FIsCUDAGraphsCompatible = std::function<bool (const NodeAttrs& attrs, const bool is_train)>;

#endif

}  // namespace mxnet

#endif  // MXNET_OP_ATTR_TYPES_H_
