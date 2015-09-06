/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray.cc
 * \brief narry module of mxnet
 */
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/narray.h>
#include <mshadow/tensor.h>
#include "./narray_function.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NArrayFunctionReg);
}  // namespace dmlc

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
    case cpu::kDevMask: {
      auto func = [lhs, rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<cpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
      };
      if (lhs.ptr_->var == ret.ptr_->var && rhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {}, {ret.ptr_->var});
      } else if (lhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {rhs.ptr_->var}, {ret.ptr_->var});
      } else if (rhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var}, {ret.ptr_->var});
      } else {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
      }
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      auto func = [lhs, rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<gpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
      };
      if (lhs.ptr_->var == ret.ptr_->var && rhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {}, {ret.ptr_->var});
      } else if (lhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {rhs.ptr_->var}, {ret.ptr_->var});
      } else if (rhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var}, {ret.ptr_->var});
      } else {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var, rhs.ptr_->var}, {ret.ptr_->var});
      }
      break;
    }
#endif
    default: LOG(FATAL) << "GPU is not enabled";
  }
}

inline void SetValueOp(const NArray &lhs, const real_t &rhs, NArray *out) {
  if (out->is_none()) {
    *out = NArray(lhs.shape(), lhs.ctx(), true);
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == lhs.shape())
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NArray ret = *out;
  switch (ret.ctx().dev_mask) {
    case cpu::kDevMask: {
      auto func = [rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<cpu>(rhs, &tmp, ctx);
      };
      DAGEngine::Get()->Push(func, ret.ctx(), {}, {ret.ptr_->var});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      auto func = [rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<gpu>(rhs, &tmp, ctx);
      };
      DAGEngine::Get()->Push(func, ret.ctx(), {}, {ret.ptr_->var});
      break;
    }
#endif
    default: LOG(FATAL) << "GPU is not enabled";
  }
}
/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output narray
 * \param binary_op the real
 */
template<typename OP, bool reverse>
inline void ScalarOp(const NArray &lhs,
                     const real_t &rhs,
                     NArray *out) {
  if (out->is_none()) {
    *out = NArray(OP::GetShape(lhs.shape(), lhs.shape()), lhs.ctx(), true);
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == OP::GetShape(lhs.shape(), lhs.shape()))
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NArray ret = *out;
  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask) {
    case cpu::kDevMask: {
      auto func = [lhs, rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<cpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
      };
      if (lhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {}, {ret.ptr_->var});
      } else {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var}, {ret.ptr_->var});
      }
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      auto func = [lhs, rhs, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Eval<gpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
      };
      if (lhs.ptr_->var == ret.ptr_->var) {
        DAGEngine::Get()->Push(func, lhs.ctx(), {}, {ret.ptr_->var});
      } else {
        DAGEngine::Get()->Push(func, lhs.ctx(), {lhs.ptr_->var}, {ret.ptr_->var});
      }
      break;
    }
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
        TBlob tmp = ret.data();
        narray::Copy<cpu, cpu>(from.data(), &tmp,
                               from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), {from.ptr_->var}, {ret.ptr_->var});
  } else if (a == cpu::kDevMask && b == gpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Copy<cpu, gpu>(from.data(), &tmp,
                               from.ctx(), ret.ctx(), ctx);
      }, ret.ctx(), {from.ptr_->var}, {ret.ptr_->var});
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Copy<gpu, cpu>(from.data(), &tmp,
                               from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), {from.ptr_->var}, {ret.ptr_->var});
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
#if MXNET_USE_CUDA
    DAGEngine::Get()->Push([from, ret](RunContext ctx) {
        ret.ptr_->CheckAndAlloc();
        TBlob tmp = ret.data();
        narray::Copy<gpu, gpu>(from.data(), &tmp,
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

template<typename OP, bool reverse>
inline NArray ScalarOpRet(const NArray &lhs,
                          const real_t &rhs) {
  NArray ret;
  ScalarOp<OP, reverse>(lhs, rhs, &ret);
  return ret;
}

template<typename OP>
inline NArray &BinaryOpApply(NArray *dst,
                             const NArray &src) {
  BinaryOp<OP>(*dst, src, dst);
  return *dst;
}

template<typename OP>
inline NArray &ScalarOpApply(NArray *dst,
                             const real_t &src) {
  ScalarOp<OP, false>(*dst, src, dst);
  return *dst;
}

// Binary
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
// Scalar
NArray operator+(const NArray &lhs, const real_t &rhs) {
  return ScalarOpRet<narray::Plus, false>(lhs, rhs);
}
NArray operator-(const NArray &lhs, const real_t &rhs) {
  return ScalarOpRet<narray::Minus, false>(lhs, rhs);
}
NArray operator*(const NArray &lhs, const real_t &rhs) {
  return ScalarOpRet<narray::Mul, false>(lhs, rhs);
}
NArray operator/(const NArray &lhs, const real_t &rhs) {
  return ScalarOpRet<narray::Div, false>(lhs, rhs);
}
// Binary
NArray &NArray::operator=(real_t scalar) {
  SetValueOp(*this, scalar, this);
  return *this; 
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
// Scalar
NArray &NArray::operator+=(const real_t &src) {
  return ScalarOpApply<narray::Plus>(this, src);
}
NArray &NArray::operator-=(const real_t &src) {
  return ScalarOpApply<narray::Minus>(this, src);
}
NArray &NArray::operator*=(const real_t &src) {
  return ScalarOpApply<narray::Mul>(this, src);
}
NArray &NArray::operator/=(const real_t &src) {
  return ScalarOpApply<narray::Div>(this, src);
}

void NArray::Save(dmlc::Stream *strm) const {
  // save shape
  shape_.Save(strm);
  if (is_none()) return;
  // save context
  Context ctx = this->ctx();
  ctx.Save(strm);
  TBlob save_data;
  NArray temp;
  if (ctx.dev_mask != cpu::kDevMask) {
    temp = this->Copy(Context(cpu::kDevMask, 0));
    temp.Wait();
    save_data = temp.data();
  } else {
    this->Wait();
    save_data = this->data();
  }
  // save type flag
  int32_t type_flag = save_data.type_flag_;
  CHECK(type_flag == mshadow::DataType<real_t>::kFlag)
      << "Only support float NArray so far";
  strm->Write(&type_flag, sizeof(type_flag));
  CHECK(save_data.CheckContiguous());
  // save data: need to change this after more type mask is supported
  size_t type_size = sizeof(real_t);
  strm->Write(save_data.dptr_, type_size * shape_.Size());
}

bool NArray::Load(dmlc::Stream *strm) {
  // load shape
  TShape shape;
  if (!shape.Load(strm)) return false;
  if (shape.ndim() == 0) {
    *this = NArray(); return true;
  }
  // load context
  Context ctx;
  if (!ctx.Load(strm)) return false;
  // load type flag
  int32_t type_flag;
  if (strm->Read(&type_flag, sizeof(type_flag)) != sizeof(type_flag)) return false;
  CHECK(type_flag == mshadow::DataType<real_t>::kFlag)
      << "Only support float NArray so far";
  // load data into CPUbu
  NArray temp(shape, Context(cpu::kDevMask, ctx.dev_id));
  TBlob load_data = temp.data();
  size_t type_size = sizeof(real_t);
  size_t nread = type_size * shape.Size();

  if (strm->Read(load_data.dptr_, nread) != nread) return false;
  if (ctx.dev_mask == cpu::kDevMask) {
    *this = std::move(temp); return true;
  } else {
    *this = temp.Copy(ctx); return true;
  }
}

NArray NArray::Copy(Context ctx) const {
  NArray ret(shape(), ctx, true);
  CopyFromTo(*this, &ret);
  return ret;
}

// register API function
// those with underscore will be registered at NArray
MXNET_REGISTER_NARRAY_FUN(_set_value).set_function(SetValueOp);

MXNET_REGISTER_NARRAY_FUN(_plus).set_function(BinaryOp<narray::Plus>);
MXNET_REGISTER_NARRAY_FUN(_minus).set_function(BinaryOp<narray::Minus>);
MXNET_REGISTER_NARRAY_FUN(_mul).set_function(BinaryOp<narray::Mul>);
MXNET_REGISTER_NARRAY_FUN(_div).set_function(BinaryOp<narray::Div>);

// register API function
// those with underscore will be registered at NArray
// scalar
MXNET_REGISTER_NARRAY_FUN(_plus_scalar).set_function(ScalarOp<narray::Plus, false>);
MXNET_REGISTER_NARRAY_FUN(_minus_scalar).set_function(ScalarOp<narray::Minus, false>);
MXNET_REGISTER_NARRAY_FUN(_mul_scalar).set_function(ScalarOp<narray::Mul, false>);
MXNET_REGISTER_NARRAY_FUN(_div_scalar).set_function(ScalarOp<narray::Div, false>);

// register API function
// those with underscore will be registered at NArray
// scalar
// reverse scalar
MXNET_REGISTER_NARRAY_FUN(_rminus_scalar).set_function(ScalarOp<narray::Minus, true>);
MXNET_REGISTER_NARRAY_FUN(_rdiv_scalar).set_function(ScalarOp<narray::Div, true>);

// copy function is special
// that we need to remove kAcceptEmptyMutateTarget from it
MXNET_REGISTER_NARRAY_FUN(_copyto)
.set_function(CopyFromTo)
.set_type_mask(kNArrayArgBeforeScalar);

}  // namespace mxnet
