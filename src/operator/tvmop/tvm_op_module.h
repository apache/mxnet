#ifndef MXNET_TVM_OP_MODULE_H
#define MXNET_TVM_OP_MODULE_H

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mutex>

namespace tvm {
namespace runtime {

class Module;
class TVMOpModule {
 public:
  void Load(const std::string& filepath);

  void Call(const std::string& func_name,
            const mxnet::OpContext& ctx,
            const std::vector<mxnet::TBlob>& args);

  static TVMOpModule *Get() {
    static TVMOpModule inst;
    return &inst;
  }

 private:
  std::mutex mutex_;
  std::shared_ptr<Module> module_ptr_;
};

}  // namespace op
}  // namespace mxnet

#endif // MXNET_TVM_OP_MODULE_H
