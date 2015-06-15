#include <dmlc/logging.h>
#include <mxnet/narray.h>
#include <mshadow/tensor.h>

namespace mxnet {
NArray operator+(const NArray &lhs, const NArray &rhs) {
  // assume CPU for now
  // need to add context conversion and other things later
  CHECK(lhs.shape() == rhs.shape());
  // TODO: defer memory allocation until execution
  NArray ret(lhs.shape(), lhs.dev_mask(), 0);  
  // redirect everything to mshadow operations
  DAGEngine::Get()->Push([ret, lhs, rhs]() {
      ret.ptr_->data.FlatTo2D<cpu, real_t>()
          = lhs.ptr_->data.FlatTo2D<cpu, real_t>() + rhs.ptr_->data.FlatTo2D<cpu, real_t>();
    }, {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
  return ret;
}
}  // namespace mxnet
