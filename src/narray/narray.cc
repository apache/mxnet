#include <dmlc/logging.h>
#include <mxnet/narray.h>
#include <mxnet/api_registry.h>
#include <mshadow/tensor.h>
#include "./narray_op.h"

namespace mxnet {
/*!
 * \brief run a binary operation, returning a new dynamically allocated NArray
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output narray
 * \param binary_op the real 
 */
template<typename OP>
inline void BinaryEWise(const NArray &lhs,
                        const NArray &rhs,
                        NArray *out) {
  CHECK(lhs.ctx() == rhs.ctx()) << "operands context mismatch";
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NArray(OP::GetShape(lhs.shape(), rhs.shape()), lhs.ctx(), true);
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == OP::GetShape(lhs.shape(), rhs.shape()))
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NArray ret = *out;
  // redirect everything to mshadow operations
  DAGEngine::Get()->Push([lhs, rhs, ret](RunContext ctx) {
      ret.ptr_->CheckAndAlloc();
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
}

template<typename OP>
inline NArray BinaryEWiseRet(const NArray &lhs,
                             const NArray &rhs) {
  NArray ret;
  BinaryEWise<OP>(lhs, rhs, &ret);
  return ret;
}

template<typename OP>
inline NArray &BinaryEWiseApply(NArray *dst,
                                const NArray &src) {                               
  BinaryEWise<OP>(*dst, src, dst);
  return *dst;
}

NArray operator+(const NArray &lhs, const NArray &rhs) {
  return BinaryEWiseRet<narray::Plus>(lhs, rhs);
}
NArray operator-(const NArray &lhs, const NArray &rhs) {
  return BinaryEWiseRet<narray::Minus>(lhs, rhs);
}
NArray operator*(const NArray &lhs, const NArray &rhs) {
  return BinaryEWiseRet<narray::Mul>(lhs, rhs);
}
NArray operator/(const NArray &lhs, const NArray &rhs) {
  return BinaryEWiseRet<narray::Div>(lhs, rhs);
}

NArray &NArray::operator+=(const NArray &src) {
  return BinaryEWiseApply<narray::Plus>(this, src);
}
NArray &NArray::operator-=(const NArray &src) {
  return BinaryEWiseApply<narray::Minus>(this, src);
}
NArray &NArray::operator*=(const NArray &src) {
  return BinaryEWiseApply<narray::Mul>(this, src);
}
NArray &NArray::operator/=(const NArray &src) {
  return BinaryEWiseApply<narray::Div>(this, src);
}

// register API function
REGISTER_NARRAY_FUN(plus).set_function(BinaryEWise<narray::Plus>);
REGISTER_NARRAY_FUN(minus).set_function(BinaryEWise<narray::Minus>);
REGISTER_NARRAY_FUN(mul).set_function(BinaryEWise<narray::Mul>);
REGISTER_NARRAY_FUN(div).set_function(BinaryEWise<narray::Div>);
}  // namespace mxnet
