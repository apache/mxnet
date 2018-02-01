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
#ifndef TEST_CORE_OP_H_
#define TEST_CORE_OP_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include "./test_op.h"
#include "../../../src/imperative/imperative_utils.h"

namespace mxnet {
namespace test {
namespace op {

// Tried making this a struct w/constexpr, but getting undefined reference on gcc 5.4.1
#define COREOP_FWD_OP_NAME_KEY          "fwd_op_name"
#define COREOP_BWD_OP_NAME_KEY          "bwd_op_name"
#define COREOP_BWD_OP_NAME_VALUE_NONE   "[none]"

enum TimingDirection {
  kForward,
  kBackward
};

inline const char *TimingDirectionAsString(const TimingDirection td) {
  switch (td) {
    case kForward:
      return "Forward";
    case kBackward:
      return "Backward";
    default:
      CHECK(false) << "Unknown timing direction: " << static_cast<int>(td);
      return "<unknown>";
  }
}

/*!
 * Low-noise operator executor
 * @tparam DType Data type for the operator executions
 */
template<typename DType>
class CoreOpExecutor : public test::op::OperatorDataInitializer<DType>
  , public test::op::OperatorExecutorTiming {
  /*! \brief Performance timing categories */
  /*!
   * \brief Access data blob as if on the CPU via a callback
   * \tparam Type of callback Function to call with CPU-data NDArray
   * \param src Source NDArray (on GPU or CPU)
   * \param run_ctx Run context
   * \param cb Callback Function to call with CPU-data NDArray
   */
  template <typename CallbackFunction>
  static inline void AccessAsCPU(const NDArray &src,
                                 const RunContext &run_ctx,
                                 CallbackFunction cb) {
#if MXNET_USE_CUDA
    if (src.ctx().dev_type == Context::kCPU) {
      cb(src);
    } else {
      Context cpu_ctx, gpu_ctx = src.ctx();
      cpu_ctx.dev_type = Context::kCPU;
      cpu_ctx.dev_id = 0;
      NDArray on_cpu(src.shape(), cpu_ctx);
      on_cpu.CheckAndAlloc();
      TBlob tmp1 = on_cpu.data();
      mxnet::ndarray::Copy<gpu, cpu>(src.data(), &tmp1, cpu_ctx, gpu_ctx, run_ctx);
      cb(on_cpu);
      TBlob tmp2 = src.data();
      mxnet::ndarray::Copy<cpu, gpu>(on_cpu.data(), &tmp2, gpu_ctx, cpu_ctx, run_ctx);
    }
#else
    cb(src);
#endif
  }

  /*!
   * \brief Parse additional arguments into NodeAttrs structure
   * \param op Pointer to operator object
   * \param args vector of string pairs representing argument key/value pairs
   * \return Constructed NodeAttrs structure
   */
  static nnvm::NodeAttrs ParseAttrs(const nnvm::Op *op, const kwargs_t& args) {
    const size_t count = args.size();
    std::vector<const char *> keys, values;
    keys.reserve(count);
    values.reserve(count);
    for (kwargs_t::const_iterator i_iter = args.begin(), e_iter = args.end();
         i_iter != e_iter; ++i_iter) {
      keys.emplace_back(i_iter->first.c_str());
      values.emplace_back(i_iter->second.c_str());
    }
    return imperative::ParseAttrs(op, op->num_inputs, count, &keys[0], &values[0]);
  }

  /*!
   * \brief Return vector of data blobs associated with anm array of NDArray objects
   * \param src vector of NDArrays
   * \param dest Vector to store pointers to the NDArrays' data blobs
   * \return Reference to the supplied vector of TBlob results
   */
  static inline std::vector<TBlob>& CollectBlobs(const std::vector<NDArray>& src,
                                                 std::vector<TBlob> *dest) {
    dest->reserve(dest->size() + src.size());
    for (size_t i = 0, n = src.size(); i < n; ++i) {
      dest->emplace_back(src[i].data());
    }
    return *dest;
  }

  /*!
   * \brief Create NDArray of random data
   * \param shape Shape of the tensor to be created
   * \param ctx Context to use when creating the array/tensor
   * \return The created NDArray
   */
  NDArray CreateRandArray(const TShape& shape, const Context& ctx) const {
    CHECK_GT(shape.Size(), 0);  // Check it's a valid shape
    NDArray array(shape, ctx, true, mshadow::DataType<DType>::kFlag);
    array.CheckAndAlloc();
    AccessAsCPU(array, ctx_.run_ctx, [this](const NDArray &arr) {
      test::op::OperatorDataInitializer<DType>::FillRandom(arr.data());
    });
    return array;
  }

  /*!
   * \brief Create NDArray of zeros
   * \param shape Shape of the tensor to be created
   * \param ctx Context to use when creating the array/tensor
   * \return The created NDArray
   */
  NDArray CreateZeroArray(const TShape& shape, const Context& ctx) const {
    CHECK_GT(shape.Size(), 0);  // Check it's a valid shape
    NDArray array(shape, ctx, true, mshadow::DataType<DType>::kFlag);
    array.CheckAndAlloc();
    AccessAsCPU(array, ctx_.run_ctx, [this](const NDArray &arr) {
      test::op::OperatorDataInitializer<DType>::FillZero(arr.data());
    });
    return array;
  }

  nnvm::NodePtr MakeNode() const {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs = attrs_;
    return node;
  }

  /*!
   * \brief Get backward op executors
   * \return Vector of backward executors
   */
  std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> GetBackward() {
    std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> res;
    static auto gradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");
    nnvm::FGradient grad_fun = gradient.get(op_, nullptr);
    if (grad_fun) {
      std::vector<nnvm::NodeEntry> out_grads;
      std::vector<nnvm::NodeEntry> entries = grad_fun(MakeNode(), out_grads);
      CHECK_GE(entries.size(), 1U);
      res.reserve(entries.size());
      for (const nnvm::NodeEntry& node_entry : entries) {
        CHECK_NOTNULL(node_entry.node.get());
        CHECK_NOTNULL(node_entry.node->op());
        CHECK_GT(node_entry.node->op()->name.size(), 0);
        if (verbose_) {
          std::cout << node_entry.node->op()->name << std::endl;
        }
        std::shared_ptr<CoreOpExecutor> pOp = std::make_shared<CoreOpExecutor>(
          ctx().run_ctx.ctx.dev_type == Context::kGPU, ShapesOf(outputs()));
        res.push_back({ pOp, node_entry.node->op()->name });
      }
    }
    return res;
  }

  /*!
   * \brief Attach any temp or random resources required to perform the op's compute operation
   * \param ctx Operator context object
   * \param attrs NodeAttrs structure (node attributes)
   * \param op Pointer to nnvm Operator object
   */
  void AttachResources(OpContext *ctx, const nnvm::NodeAttrs& attrs, const nnvm::Op *op) {
    static auto& fresource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
    if (fresource.count(op) != 0) {
      std::vector<Resource>& requested = ctx->requested;
      auto reqs = fresource[op](attrs);
      // Get the resource of temporal space.
      for (const ResourceRequest& req : reqs) {
        if (req.type == ResourceRequest::kTempSpace) {
          Resource r = ResourceManager::Get()->Request(ctx->run_ctx.ctx, req);
          requested.emplace_back(r);
        } else if (req.type == ResourceRequest::kRandom) {
          requested.emplace_back(ResourceManager::Get()->Request(ctx->run_ctx.ctx, req));
        } else {
          LOG(FATAL) << "resource type not yet supported";
        }
      }
    }
  }

 public:
  typedef DType   DataType;

  /*! \brief Add 'fwd_op_name' to kwargs and return the new kwargs */
  static kwargs_t ArgsWithOpName(const kwargs_t& args,
                                 const std::string& fwd_op_name,
                                 const std::string& bwd_op_name = "") {
    CHECK(!fwd_op_name.empty());
    kwargs_t new_args;
    new_args.reserve(args.size() + 1);
    for (const auto& a : args) {
      if (a.first != COREOP_FWD_OP_NAME_KEY && a.first != COREOP_BWD_OP_NAME_KEY) {
        new_args.emplace_back(a);
      }
    }
    new_args.push_back({ COREOP_FWD_OP_NAME_KEY, fwd_op_name});
    if (!bwd_op_name.empty()) {
      new_args.push_back({ COREOP_BWD_OP_NAME_KEY, bwd_op_name});
    }
    return new_args;
  }

  /*! \brief Remove 'fwd_op_name' from kwargs and return the new kwargs */
  static kwargs_t ArgsSansOpName(const kwargs_t& args,
                                 std::string* fwd_op_name_ptr,
                                 std::string* bwd_op_name_ptr = nullptr) {
    CHECK_NOTNULL(fwd_op_name_ptr);
    CHECK_NOTNULL(bwd_op_name_ptr);
    bwd_op_name_ptr->resize(0);
    kwargs_t new_args;
    new_args.reserve(args.size());
    for (const auto& a : args) {
      if (a.first == COREOP_FWD_OP_NAME_KEY) {
        *fwd_op_name_ptr = a.second;
      } else if (a.first == COREOP_BWD_OP_NAME_KEY) {
        *bwd_op_name_ptr = a.second;
      } else {
        new_args.emplace_back(a);
      }
    }
    return new_args;
  }

  /*!
   * \brief Constructor
   * \param isGPU Is this going to be on the GPU?
   * \param shapes Array of input shapes
   */
  CoreOpExecutor(const bool isGPU, const std::vector<TShape>& shapes)
    : input_shapes_(shapes)
      , op_(nullptr)  {
    ctx_.is_train = true;
    ctx_.run_ctx.ctx.dev_id = 0;
    ctx_.run_ctx.stream = nullptr;
    ctx_.run_ctx.ctx.dev_type = Context::kCPU;
#if MXNET_USE_CUDA
    if (isGPU) {
      ctx_.run_ctx.ctx.dev_type = Context::kGPU;
      allocGPUStream_.reset(new GPUStreamScope(&ctx_));
    } else {
      ctx_.run_ctx.ctx.dev_type = Context::kCPU;
    }
#else
    CHECK(!isGPU);
    ctx_.run_ctx.ctx.dev_type = Context::kCPU;
#endif
  }

  /*!
   * \brief Initialize the execution objects and execution data (only occurs once)
   * \param args Parameter arguments
   * \param inputs Optional input data (otherwise, random data will be used as input)
   */
  void Init(const kwargs_t& in_args,
            const std::vector<NDArray>& inputs = {},
            const std::vector<NDArray>& outputs = {},
            const CoreOpExecutor *backward_for_op = nullptr
  ) {
    if (!initialized_) {
      initialized_ = true;

      std::string op_name, bwd_op_name;
      kwargs_t args = ArgsSansOpName(in_args, &op_name, &bwd_op_name);
      CHECK(op_name.empty() == false);

      CHECK(!backward_for_op || bwd_op_name.empty())
        << "Backward op should not be supplied another backward operator";

      if (verbose_ && backward_for_op) {
        std::cout << "Backward op: " << op_name;
      }

      op_ = nnvm::Op::Get(op_name);
      CHECK_NOTNULL(op_);

      // Set up forward
      attrs_ = ParseAttrs(op_, args);

      const int num_inputs = op_->num_inputs;

      if (!inputs.empty()) {
        CHECK_EQ(inputs.size(), static_cast<size_t>(num_inputs));
      }

      int inferred_num_outputs, num_visible_outputs;

      imperative::SetNumOutputs(op_, attrs_, num_inputs, &inferred_num_outputs,
                                &num_visible_outputs);

      // Generic, all shapes the same. Probably this will need to be adjusted for more complex
      // operators such as dot
      std::vector<TShape> shapes;
      for (size_t i = 0, n = std::max(num_visible_outputs, num_inputs); i < n; ++i) {
        shapes.emplace_back(i < input_shapes_.size() ? input_shapes_[i]
                                                  : input_shapes_[input_shapes_.size() - 1]);
      }
      std::vector<NDArray *> inputs_p, outputs_p;

      if (!outputs.empty()) {
        CHECK_EQ(outputs.size(), static_cast<size_t>(num_visible_outputs));
      }

      inputs_.reserve(num_inputs);
      inputs_p.reserve(num_inputs);
      outputs_.reserve(num_visible_outputs);
      outputs_p.reserve(num_visible_outputs);

      for (size_t i = 0; i < static_cast<size_t>(num_inputs); ++i) {
        CHECK_LT(i, static_cast<int>(shapes.size()));
        inputs_.emplace_back(i < inputs.size() ? inputs[i] : CreateRandArray(shapes[i],
                                                                          ctx_.run_ctx.ctx));
        inputs_p.emplace_back(&*inputs_.rbegin());
      }

      for (size_t i = 0; i < static_cast<size_t>(num_visible_outputs); ++i) {
        // If supplied and valid, pass from the supplied outputs vector
        // Otherwise use empty for forward pass, or zero-filled for backward pass
        outputs_.emplace_back(i < outputs.size()
                              ? outputs[i]
                              : (backward_for_op ? CreateZeroArray(shapes[i], ctx_.run_ctx.ctx)
                                                 : NDArray()));
        outputs_p.emplace_back(&*outputs_.rbegin());
      }

      if (!backward_for_op) {
        DispatchMode dispatch_mode = DispatchMode::kUndefined;
        imperative::SetShapeType(ctx_.run_ctx.ctx, attrs_, inputs_p, outputs_p, &dispatch_mode);
      } else {
        // Backward op, so set based upon inputs
        CHECK_EQ(static_cast<size_t>(num_visible_outputs), backward_for_op->inputs().size());
        for (int i = 0; i < num_visible_outputs; ++i) {
          CHECK_LT(static_cast<size_t>(i), shapes.size());
          // backward outputs should look like forward inputs
          // TODO(cjolivier01): This check fails for dot product...
          // Need better inference of backward shapes
          // CHECK_EQ(backward_for_op->inputs()[i].shape(), outputs_[i].shape());
        }
      }

      std::vector<OpReqType> req;
      imperative::SetWriteInplaceReq(inputs_p, outputs_p, &req_);

      CollectBlobs(inputs_, &blob_inputs_);
      CollectBlobs(outputs_, &blob_outputs_);

      function_ = common::GetFCompute<FCompute>(op_, "FCompute", ctx_.run_ctx.ctx);
      functionex_ = common::GetFCompute<FComputeEx>(op_, "FComputeEx", ctx_.run_ctx.ctx);

      AttachResources(&ctx_, attrs_, op_);

      if (!backward_for_op) {
        bool no_backward = false;
        // Set up backward
        std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> bwd;
        if (!bwd_op_name.empty()) {
          if (bwd_op_name != COREOP_BWD_OP_NAME_VALUE_NONE) {
            // Backward op was specified
            std::shared_ptr<CoreOpExecutor> pOp = std::make_shared<CoreOpExecutor>(
              ctx().run_ctx.ctx.dev_type == Context::kGPU, ShapesOf(this->outputs()));
            bwd.push_back({pOp, bwd_op_name});
          } else {
            no_backward = true;
          }
        } else {
          // Try to figure out backward op
          bwd = GetBackward();
        }
        if (!no_backward) {
          CHECK_GE(bwd.size(), 1U)
            << "Can't automatically determine backward op name. Please specify";
          for (std::pair<std::shared_ptr<CoreOpExecutor>, std::string> &bw_item : bwd) {
            bw_item.first->set_verbose(verbose_);
            backward_.emplace_back(bw_item.first);
            bw_item.first->Init(ArgsWithOpName(args, bw_item.second), {}, {}, this);
          }
        }
      }
    }
  }

  template<typename OpProp>
  inline bool initForward(const OpProp &opProp, std::vector<int> *in_type) {
    Init(opProp.GetArgs());
    return true;
  }

  template<typename OpProp>
  inline bool initBackward(const OpProp &opProp, std::vector<int> *in_type) { return true; }

  inline void forward(const size_t count) {
    perf::TimingItem timeF(&OperatorExecutorTiming::GetTiming(), kForward, "Forward", count);
    VTuneResume profile;
    for (size_t i = 0; i < count; ++i) {
      Execute();
    }
  }

  inline void backward(const size_t count) {
    CHECK(HasBackward());
    perf::TimingItem timeF(&OperatorExecutorTiming::GetTiming(), kBackward, "Backward", count);
    VTuneResume profile;
    for (size_t i = 0; i < count; ++i) {
      ExecuteBackward();
    }
  }

  /*!
   * \brief Execute the operator for a dense tensor
   */
  void Execute() {
    CHECK_EQ(initialized_, true);
    CHECK_NOTNULL(function_);
    function_(attrs_, ctx_, blob_inputs_, req_, blob_outputs_);
  }

  /*!
   * \brief Execute the operator for a sparse tensor
   */
  void ExecuteEx() {
    CHECK_EQ(initialized_, true);
    CHECK_NOTNULL(functionex_);
    functionex_(attrs_, ctx_, inputs_, req_, outputs_);
  }

  bool HasBackward() const {
    return !backward_.empty();
  }

  /*!
   * \brief Execute backward pass on operator
   */
  bool ExecuteBackward() {
    CHECK_EQ(initialized_, true);
    CHECK(HasBackward());
    if (!backward_.empty()) {
      // Avoid locked ref count here
      for (std::shared_ptr<CoreOpExecutor> &p : backward_) {
        p->Execute();
      }
      return true;
    }
    return false;
  }

  /*!
   * \brief Execute backward pass on operator
   */
  bool ExecuteBackwardEx() {
    CHECK_EQ(initialized_, true);
    CHECK(HasBackward());
    if (!backward_.empty()) {
      // Avoid locked ref count here
      for (std::shared_ptr<CoreOpExecutor> &p : backward_) {
        p->ExecuteEx();
      }
      return true;
    }
    return false;
  }

  /*!
   * \brief Get the operator context
   * \return Reference to this operator's context object
   */
  const OpContext& ctx() const {
    return ctx_;
  }

  /*!
   * \brief Access input NDArray vector
   * \return reference to NDArray vector of forward inputs
   */
  std::vector<NDArray>& inputs() { return inputs_; }
  const std::vector<NDArray>& inputs() const { return inputs_; }

  /*!
   * \brief Access input NDArray vector
   * \return reference to NDArray vector of forward outputs
   */
  std::vector<NDArray>& outputs() { return outputs_; }
  const std::vector<NDArray>& outputs() const { return outputs_; }

  /*!
   * \brief Backward inputs (i.e. output grad)
   * \return reference to NDArray vector of backward inputs
   */
  std::vector<NDArray>& bwd_inputs() {
    CHECK_EQ(backward_.size(), 1U);
    return backward_[0]->inputs();
  }

  /*!
   * \brief Backward outputs (i.e. input grad)
   * \return reference to NDArray vector of backward outputs
   */
  std::vector<NDArray>& bwd_outputs() {
    CHECK_EQ(backward_.size(), 1U);
    return backward_[0]->outputs();
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

 private:
  /*!
   * \brief Has the execution been initialized?
   */
  bool initialized_ = false;
  /*!
   * \brief Whether to print debug trace output
   */
  bool verbose_ = false;
  /*!
   * \brief This operator's context object
   */
  OpContext ctx_;

#if MXNET_USE_CUDA
  /*! \brief
   * Scoped GPU stream
   */
  std::unique_ptr<GPUStreamScope> allocGPUStream_;
#endif

  /*!
   * \brief Input data shape
   */
  std::vector<TShape> input_shapes_;
  /*
   * \brief Pointer to the operator object
   */
  const nnvm::Op *op_;
  /*!
   * \brief Operator attributes
   */
  nnvm::NodeAttrs attrs_;
  /*!
   * \brief Input and output NDArray vectors
   */
  std::vector<NDArray> inputs_, outputs_;
  /*!
   * \brief Vectors of the TBlob objects associated with the NDArrays in inputs_ and outputs_
   */
  std::vector<TBlob> blob_inputs_, blob_outputs_;
  /*!
   * \brief Operator request type vector
   */
  std::vector<OpReqType> req_;
  /*!
   * \brief Operator's FCompute function (for dense tensors)
   */
  FCompute function_;
  /*!
   * \brief Operator's FCompute function (for sparse tensors)
   */
  FComputeEx functionex_;

  /*!
   * \brief Backward executors (if any)
   */
  std::vector<std::shared_ptr<CoreOpExecutor>> backward_;
};

class CoreOpProp {
 public:
  void Init(const kwargs_t& kwargs) { kwargs_ = kwargs; }
  const kwargs_t& GetArgs() const { return kwargs_; }
 private:
  kwargs_t          kwargs_;
};

template<typename DType>
using CoreOperatorRunner = test::OperatorRunner<CoreOpProp, CoreOpExecutor<DType>>;


/*!
 * \brief Rune a core op forward and backward
 * \tparam DType Data type
 * \param isGPU true if operation is to be run on the GPU
 * \param op_kwargs Operator parameters
 * \param op_name Operator name as registered with nnvm
 * \param backward_op_name Backwards operator name as registered with nnvm
 *        If blank, the runner will attempt to determine the backwards operator. If it fails,
 *        an exception will be thrown.
 *        If the string is [none], then no backward operator will be created or executed
 */
template<typename DType = float>
inline void BasicRunCoreOpBidirectional(const bool isGPU,
                                        bool verbose,
                                        const kwargs_t& op_kwargs,
                                        const std::vector<TShape>& shapes,
                                        const char *op_name,
                                        const char *backward_op_name = "") {
  test::op::CoreOpExecutor<DType> op(isGPU, shapes);
  op.set_verbose(false);

  op.Init(op.ArgsWithOpName(op_kwargs, op_name, backward_op_name));

  if (verbose) {
    PRINT_NDARRAYS(op.ctx().run_ctx, op.inputs());
    PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  }
  op.Execute();
  if (verbose) {
    PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  }
  if (op.HasBackward()) {
    if (verbose) {
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_inputs());
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
    }
    op.ExecuteBackward();
    if (verbose) {
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
    }
  }
}

}  // namespace op
}  // namespace test
}  // namespace mxnet

#endif  // TEST_CORE_OP_H_
