#include <dmlc/logging.h>
#include <mxnet/narray.h>
#include <mshadow/tensor.h>
#include "./narray_op.h"

namespace mxnet {
/*!
 * \brief run a binary operation, returning a new dynamically allocated NArray
 * \param lhs left operand
 * \param rhs right operand
 * \param binary_op the real 
 */
template<typename OP>
inline NArray BinaryEWise(const NArray &lhs, const NArray &rhs) {
  CHECK(lhs.ctx() == rhs.ctx()) << "operands context mismatch";
  // defer memory allocation until execution
  NArray ret(OP::GetShape(lhs.shape(), rhs.shape()), lhs.ctx(), true);
  // redirect everything to mshadow operations
  DAGEngine::Get()->Push([ret, lhs, rhs](RunContext ctx) {
      ret.ptr_->Alloc();
      switch (lhs.ctx().dev_mask) {
        case cpu::kDevMask:
          narray::Eval<cpu, OP>(lhs.ptr_->data, rhs.ptr_->data, ret.ptr_->data, ctx);
          break;
#if MXNET_USE_CUDA
        case gpu::kDevMask:
          narray::Eval<gpu, OP>(lhs.ptr_->data, rhs.ptr_->data, ret.ptr_->data, ctx);
          break;
#endif
        default: LOG(FATAL) << "GPU is not enabled";              
      }
    }, lhs.ctx(), {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
  return ret;
}

NArray operator+(const NArray &lhs, const NArray &rhs) {
  return BinaryEWise<narray::Plus>(lhs, rhs);
}
NArray operator-(const NArray &lhs, const NArray &rhs) {
  return BinaryEWise<narray::Minus>(lhs, rhs);
}
NArray operator*(const NArray &lhs, const NArray &rhs) {
  return BinaryEWise<narray::Mul>(lhs, rhs);
}
NArray operator/(const NArray &lhs, const NArray &rhs) {
  return BinaryEWise<narray::Div>(lhs, rhs);
}
}  // namespace mxnet
