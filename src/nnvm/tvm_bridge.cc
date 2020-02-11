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
 * \file tvm_bridge.cc
 * \brief Bridge to run TVM's PackedFunc in MXNet's async engine.
 *
 *  This bridge is mainly used to expose MXNet's async engine push to
 *  TVM. It only uses TVM runtime in aheader only mode, which means
 *  there is no link dependencies.
 *
 *  Support for TVM is optional even when this code
 *  is always compiled and built with the project.
 *  We choose this strategy because we do not yet want
 *  llvm as dependency(which TVM uses). So instead we expose hook
 *  to TVM and let user use this feature when they have TVM installed.
 *
 *  We do require TVM and MXNet to be built with same C++ ABI of std::function
 */
#define TVM_RUNTIME_HEADER_ONLY 1
#include <tvm/runtime/packed_func.h>
#include <mxnet/c_api.h>
#include <mxnet/ndarray.h>
#include <mxnet/engine.h>

#include <memory>

namespace mxnet {

using tvm::runtime::PackedFunc;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

/*!
 * \brief Async functor object
 *  calling argument of the function.
 */
class TVMFunctor {
 public:
  // constructor
  explicit TVMFunctor(PackedFunc func, PackedFunc fset_stream)
      : func_(func), fset_stream_(fset_stream) {}

  void Init(const TVMArgs& args,
            const std::vector<int>& const_loc,
            std::vector<Engine::VarHandle>* const_vars,
            std::vector<Engine::VarHandle>* mutate_vars) {
    values_.clear();
    type_codes_.clear();
    values_.insert(values_.end(), args.values, args.values + args.size());
    type_codes_.insert(
        type_codes_.end(), args.type_codes, args.type_codes + args.size());

    size_t const_loc_ptr = 0;
    for (int i = 0; i < args.size(); ++i) {
      if (args.type_codes[i] == kTVMNDArrayTypeCode) {
        const NDArray& nd =
            static_cast<NDArray*>(args.values[i].v_handle)[0];
        // We cannot set the value until
        type_codes_[i] = kTVMDLTensorHandle;
        array_data_.push_back(nd);
        array_loc_.push_back(i);
        // check if there is read or mutate
        // by default assume we mutate the array.
        if (const_loc_ptr < const_loc.size() &&
            i == const_loc[const_loc_ptr]) {
          const_vars->push_back(nd.var());
          ++const_loc_ptr;
        } else {
          mutate_vars->push_back(nd.var());
        }
      } else {
        CHECK_LT(args.type_codes[i], kTVMDataType)
            << "Only allow POD type in mxnet async call";
      }
    }
  }

  Context ctx() {
    return array_data_[0].ctx();
  }

  void Run(const RunContext& rctx) {
    // setup DLTensor
    for (size_t i = 0; i < array_loc_.size(); ++i) {
      values_[array_loc_[i]].v_handle =
          const_cast<DLTensor*>(&(array_data_[i].data().dltensor()));
    }
    // run the packed function
    TVMRetValue rv;
    TVMArgs args(&values_[0], &type_codes_[0], values_.size());
    if (ctx().dev_type == Context::kGPU) {
#if MXNET_USE_CUDA
      // pass stream via last argument.
      void* strm = static_cast<void*>(rctx.get_stream<gpu>()->stream_);
      int dev_type = kDLGPU;
      fset_stream_(dev_type, rctx.ctx.dev_id, strm);
      func_.CallPacked(args, &rv);
      fset_stream_(dev_type, rctx.ctx.dev_id, nullptr);
#else
      LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif
    } else {
      func_.CallPacked(args, &rv);
    }
  }

 private:
  /*! \brief The function */
  PackedFunc func_;
  /*! \brief Set stream */
  PackedFunc fset_stream_;
  /*! \brief Values field */
  std::vector<TVMValue> values_;
  /*! \brief type code field */
  std::vector<int> type_codes_;
  /*! \brief arrays field */
  std::vector<NDArray> array_data_;
  /*! \brief position of array in arguments */
  std::vector<int> array_loc_;
};


// Wrap a TVM function to a function that invokes MXNet's Engine
// It does two things: call the engine properly
// set up the NDArray to DLTensor during invocation.
void WrapAsyncCall(TVMArgs wrap_args, TVMRetValue* wrap_rv) {
  PackedFunc f = wrap_args[0];
  PackedFunc fset_stream =  wrap_args[1];
  int num_const = wrap_args[2];

  // sorted position of constant arguments
  std::vector<int> const_loc;
  for (int i = 0; i < num_const; ++i) {
    const_loc.push_back(wrap_args[i + 3].operator int());
  }
  std::sort(const_loc.begin(), const_loc.end());
  // wrapped function
  // This is the function that called by the user.
  auto wrapped = [f, fset_stream, const_loc](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<TVMFunctor> func =
      std::make_shared<TVMFunctor>(f, fset_stream);
    std::vector<Engine::VarHandle> const_vars, mutate_vars;
    func->Init(args, const_loc, &const_vars, &mutate_vars);
    Engine *engine = Engine::Get();
    engine->DeduplicateVarHandle(&const_vars, &mutate_vars);
    engine->PushSync([func](RunContext ctx) {
        func->Run(ctx);
      }, func->ctx(), const_vars, mutate_vars);
  };
  *wrap_rv = PackedFunc(wrapped);
}

}  // namespace mxnet

// C callback that can be used by TVM to extract
// the WrapAsyncCall function.
extern "C" MXNET_DLL int MXTVMBridge(TVMFunctionHandle pregister) {
  using tvm::runtime::PackedFunc;
  const PackedFunc& fregister =
      *static_cast<PackedFunc*>(pregister);
  fregister("WrapAsyncCall", PackedFunc(mxnet::WrapAsyncCall));
  return 0;
}
