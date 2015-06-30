#include <dmlc/logging.h>
#include <mxnet/narray.h>
#include <mxnet/api_registry.h>
#include <mshadow/tensor.h>
#include "./narray_op.h"

namespace mxnet {
/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output narray
 * \param binary_op the real 
 */
template<typename OP>
inline void BinaryOp(const NArray &lhs,
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
  switch (lhs.ctx().dev_mask) {
    case cpu::kDevMask:
      DAGEngine::Get()->Push([lhs, rhs, ret](RunContext ctx) {
          ret.ptr_->CheckAndAlloc();
          narray::Eval<cpu, OP>(lhs.ptr_->data, rhs.ptr_->data, &ret.ptr_->data, ctx);
        }, lhs.ctx(), {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
      break;
#if MXNET_USE_CUDA
    case gpu::kDevMask:
      DAGEngine::Get()->Push([lhs, rhs, ret](RunContext ctx) {
          ret.ptr_->CheckAndAlloc();
          narray::Eval<gpu, OP>(lhs.ptr_->data, rhs.ptr_->data, &ret.ptr_->data, ctx);
        }, lhs.ctx(), {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
      break;
#endif
    default: LOG(FATAL) << "GPU is not enabled";
  }
}

void CopyFromTo(const NArray &from, NArray *to) {
  CHECK(from.shape() == to->shape())
      << "operands shape mismatch";
  CHECK(from.shape().ndim() != 0)
      << "source operands have zero dimension shape";
  // important: callback must always capture by value
  NArray ret = *to;
  int a = from.ctx().dev_mask;
  int b = to->ctx().dev_mask;
  if (a == cpu::kDevMask && b == cpu::kDevMask) {
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        narray::Copy<cpu, cpu>(from.ptr_->data, &ret.ptr_->data,
                               from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), {from.ptr_->var}, {ret.ptr_->var});
  } else if (a == cpu::kDevMask && b == gpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        narray::Copy<cpu, gpu>(from.ptr_->data, &ret.ptr_->data,
                               from.ctx(), ret.ctx(), ctx);
      }, ret.ctx(), {from.ptr_->var}, {ret.ptr_->var});
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        narray::Copy<gpu, cpu>(from.ptr_->data, &ret.ptr_->data,
                               from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), {from.ptr_->var}, {ret.ptr_->var});
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        narray::Copy<gpu, gpu>(from.ptr_->data, &ret.ptr_->data,
                               from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), {from.ptr_->var}, {ret.ptr_->var});
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  } else {
    LOG(FATAL) << "unknown device mask";
  }
}

template<typename OP>
inline NArray BinaryOpRet(const NArray &lhs,
                          const NArray &rhs) {
  NArray ret;
  BinaryOp<OP>(lhs, rhs, &ret);
  return ret;
}

template<typename OP>
inline NArray &BinaryOpApply(NArray *dst,
                             const NArray &src) {                               
  BinaryOp<OP>(*dst, src, dst);
  return *dst;
}

NArray operator+(const NArray &lhs, const NArray &rhs) {
  return BinaryOpRet<narray::Plus>(lhs, rhs);
}
NArray operator-(const NArray &lhs, const NArray &rhs) {
  return BinaryOpRet<narray::Minus>(lhs, rhs);
}
NArray operator*(const NArray &lhs, const NArray &rhs) {
  return BinaryOpRet<narray::Mul>(lhs, rhs);
}
NArray operator/(const NArray &lhs, const NArray &rhs) {
  return BinaryOpRet<narray::Div>(lhs, rhs);
}

NArray &NArray::operator+=(const NArray &src) {
  return BinaryOpApply<narray::Plus>(this, src);
}
NArray &NArray::operator-=(const NArray &src) {
  return BinaryOpApply<narray::Minus>(this, src);
}
NArray &NArray::operator*=(const NArray &src) {
  return BinaryOpApply<narray::Mul>(this, src);
}
NArray &NArray::operator/=(const NArray &src) {
  return BinaryOpApply<narray::Div>(this, src);
}

// register API function
REGISTER_NARRAY_FUN(plus).set_function(BinaryOp<narray::Plus>);
REGISTER_NARRAY_FUN(minus).set_function(BinaryOp<narray::Minus>);
REGISTER_NARRAY_FUN(mul).set_function(BinaryOp<narray::Mul>);
REGISTER_NARRAY_FUN(div).set_function(BinaryOp<narray::Div>);

// copy function is special
//that we need to remove kAcceptEmptyMutateTarget from it
REGISTER_NARRAY_FUN(copy)
.set_function(CopyFromTo)
.set_type_mask(kNArrayArgBeforeScalar);

}  // namespace mxnet
