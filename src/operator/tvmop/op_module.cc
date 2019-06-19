#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "op_module.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {

void TVMOpModule::Load(const std::string &filepath) {
  static const PackedFunc *f_load = Registry::Get("module._LoadFromFile");
  std::lock_guard<std::mutex> lock(mutex_);
  Module module = (*f_load)(filepath, "");
  module_ptr_ = std::make_shared<Module>();
  *module_ptr_ = module;
}

void TVMOpModule::Call(const std::string &func_name,
                       const mxnet::OpContext& ctx,
                       const std::vector<mxnet::TBlob> &args) {
  std::vector<int> type_codes;
  std::vector<TVMValue> values;

  type_codes.resize(args.size());
  values.resize(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    type_codes[i] = kArrayHandle;
    values[i].v_handle = const_cast<DLTensor *>(&(args[i].dltensor()));
  }

  TVMArgs tvm_args(&values[0], &type_codes[0], args.size());
  TVMRetValue rv;

  // TODO: cache
  PackedFunc func = module_ptr_->GetFunction(func_name, false);

  int dev_type = (ctx.run_ctx.ctx.dev_type == mxnet::Context::DeviceType::kGPU) ? kDLGPU : kDLCPU;
  int dev_id = ctx.run_ctx.ctx.dev_id;
#if MXNET_USE_CUDA
  if (dev_type == kDLGPU) {
    void *stream = static_cast<void *>(ctx.run_ctx.get_stream<mxnet::gpu>()->stream_);
    TVMSetStream(dev_type, dev_id, stream);
  }
#endif
  func.CallPacked(tvm_args, &rv);
#if MXNET_USE_CUDA
  if (dev_type == kDLGPU) {
    TVMSetStream(dev_type, dev_id, nullptr);
  }
#endif
}

}
}
