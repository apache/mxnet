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
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief ndarry module of mxnet
 */
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/resource.h>
#include <mxnet/imperative.h>
#include <mshadow/tensor.h>
#include "./ndarray_function.h"
#include "../common/utils.h"
#include "../operator/tensor/matrix_op-inl.h"
#include "../operator/tensor/init_op.h"

#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NDArrayFunctionReg);
}  // namespace dmlc

namespace mxnet {

NDArray NDArray::grad() const {
  if (Imperative::AGInfo::IsNone(*this)) return NDArray();
  Imperative::AGInfo& info = Imperative::AGInfo::Get(entry_.node);
  if (info.out_grads.size()) {
    CHECK_EQ(info.out_grads.size(), 1);
    return info.out_grads[0];
  }
  return NDArray();
}

nnvm::Symbol NDArray::get_autograd_symbol() const {
  CHECK(!Imperative::AGInfo::IsNone(*this))
    << "NDArray is not part of a computation graph. Did you forget to turn on recording?";
  nnvm::Symbol ret;
  ret.outputs.emplace_back(entry_);
  return ret;
}

NDArray NDArray::Reshape(const TShape &shape) const {
  CHECK(!is_none()) << "NDArray is not initialized";
  auto stype = storage_type();
  // reshape is not supported for non-default ndarray with dismatching shapes
  CHECK((shape_ == shape) || stype == kDefaultStorage)
    << "Reshape for storage type " << stype << " is not implemented yet";
  CHECK_GE(shape_.Size(), shape.Size())
    << "NDArray.Reshape: target shape size is larger current shape";
  NDArray ret = this->Detach();
  ret.shape_ = shape;
  return ret;
}

NDArray NDArray::ReshapeWithRecord(const TShape &shape) {
  NDArray ret = this->Reshape(shape);
  if (!Imperative::Get()->is_recording()) return ret;

  CHECK_EQ(shape_.Size(), shape.Size())
    << "NDArray.Reshape: target shape must have the same size as "
    << "current shape when recording with autograd.";
  nnvm::NodeAttrs attrs;
  attrs.op = nnvm::Op::Get("Reshape");;
  std::ostringstream os;
  os << shape;
  attrs.dict.insert({"shape", os.str()});
  attrs.op->attr_parser(&attrs);
  std::vector<NDArray*> inputs(1, this), outputs(1, &ret);
  Imperative::Get()->RecordOp(std::move(attrs), inputs, outputs);
  return ret;
}


NDArray NDArray::Slice(index_t begin, index_t end) const {
  CHECK(!is_none()) << "NDArray is empty";
  CHECK_LE(begin, end)
      << "Invalid slicing range [" << begin << ", " << end << ")";
  CHECK_GE(shape_[0], end) << "Slice end index out of range";
  CHECK_EQ(storage_type(), kDefaultStorage);
  NDArray ret = this->Detach();
  size_t length = shape_.ProdShape(1, shape_.ndim());
  MSHADOW_TYPE_SWITCH(ret.dtype(), DType, {
    ret.byte_offset_ += begin * length * sizeof(DType);
  });
  ret.shape_[0] = end - begin;
  return ret;
}

NDArray NDArray::SliceWithRecord(index_t begin, index_t end) {
  NDArray ret = this->Slice(begin, end);
  if (!Imperative::Get()->is_recording()) return ret;
  // fake a slice_axis op
  nnvm::NodeAttrs attrs;
  attrs.op = nnvm::Op::Get("slice_axis");
  attrs.dict.insert({"axis", "0"});
  attrs.dict.insert({"begin", std::to_string(begin)});
  attrs.dict.insert({"end", std::to_string(end)});
  attrs.op->attr_parser(&attrs);
  std::vector<NDArray*> inputs(1, this), outputs(1, &ret);
  Imperative::Get()->RecordOp(std::move(attrs), inputs, outputs);
  return ret;
}

NDArray NDArray::At(index_t idx) const {
  CHECK(storage_type() == kDefaultStorage) << "Storage type "
                                           << storage_type() << " doesn't support At()";
  NDArray ret = this->Slice(idx, idx+1);
  if (shape_.ndim() > 1) {
    return ret.Reshape(TShape(shape_.data()+1, shape_.data()+shape_.ndim()));
  } else {
    return ret;
  }
}

NDArray NDArray::AtWithRecord(index_t idx) {
  CHECK(storage_type() == kDefaultStorage)
      << "Storage type " << storage_type() << " doesn't support At()";
  NDArray ret = this->SliceWithRecord(idx, idx+1);
  if (shape_.ndim() > 1) {
    return ret.ReshapeWithRecord(TShape(shape_.data()+1, shape_.data()+shape_.ndim()));
  } else {
    return ret;
  }
}

/*!
 * \brief Return deep copy of the current ndarry's aux_data(i)
 * as an NDArray of default storage type. This function blocks.
 */
NDArray NDArray::aux_ndarray(size_t i) const {
  CHECK_NE(storage_type(), kDefaultStorage);
  CHECK(i < ptr_->aux_shapes.size());
  // create a delay_alloc default ndarray as output
  NDArray ret(TShape(), ctx(), true, aux_type(i));
  ret.SyncCopyFromNDArray(*this, i);
  return ret;
}

NDArray NDArray::data_ndarray() const {
  NDArray ret(TShape(), ctx(), true, dtype_);
  ret.SyncCopyFromNDArray(*this);
  return ret;
}

bool NDArray::fresh_out_grad() const {
  if (Imperative::AGInfo::IsNone(*this)) return false;
  Imperative::AGInfo& info = Imperative::AGInfo::Get(entry_.node);
  return info.fresh_out_grad;
}


void NDArray::set_fresh_out_grad(bool state) const {
  CHECK(!Imperative::AGInfo::IsNone(*this))
    << "NDArray has not been marked as a variable and does not have gradient state";
  Imperative::AGInfo& info = Imperative::AGInfo::Get(entry_.node);
  info.fresh_out_grad = state;
}


/*!
* \brief run a ternary operation
* \param lhs left operand
* \param mhs middle operand
* \param rhs right operand
* \param out the output ndarray
*/
template<typename OP>
void TernaryOp(const NDArray &lhs,
  const NDArray &mhs,
  const NDArray &rhs,
  NDArray *out) {
  // no check if all of them are on cpu
  if (lhs.ctx().dev_mask() != cpu::kDevMask || mhs.ctx().dev_mask() != cpu::kDevMask
                                            || rhs.ctx().dev_mask() != cpu::kDevMask) {
    CHECK((lhs.ctx() == mhs.ctx()) && (mhs.ctx() == rhs.ctx())) << "operands context mismatch";
  }
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NDArray(OP::GetShape(lhs.shape(), mhs.shape(), rhs.shape()), lhs.ctx(), true);
  } else {
    // no check if both of them are on cpu
    if (lhs.ctx().dev_mask() != cpu::kDevMask ||
      out->ctx().dev_mask() != cpu::kDevMask) {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    }
    CHECK(out->shape() == OP::GetShape(lhs.shape(), mhs.shape(), rhs.shape()))
      << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
  if (mhs.var() != ret.var()) const_vars.push_back(mhs.var());
  if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
  case cpu::kDevMask: {
    Engine::Get()->PushSync([lhs, mhs, rhs, ret](RunContext ctx) {
      TBlob tmp = ret.data();
      ndarray::Eval<cpu, OP>(lhs.data(), mhs.data(), rhs.data(), &tmp, ctx);
    }, lhs.ctx(), const_vars, { ret.var() },
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    break;
  }
#if MXNET_USE_CUDA
  case gpu::kDevMask: {
    Engine::Get()->PushSync([lhs, mhs, rhs, ret](RunContext ctx) {
      TBlob tmp = ret.data();
      ndarray::Eval<gpu, OP>(lhs.data(), mhs.data(), rhs.data(), &tmp, ctx);
      // Wait GPU kernel to complete
      ctx.get_stream<gpu>()->Wait();
    }, lhs.ctx(), const_vars, { ret.var() },
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
    break;
  }
#endif
  default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP>
void BinaryOp(const NDArray &lhs,
              const NDArray &rhs,
              NDArray *out) {
  // no check if both of them are on cpu
  if (lhs.ctx().dev_mask() != cpu::kDevMask || rhs.ctx().dev_mask() != cpu::kDevMask) {
    CHECK(lhs.ctx() == rhs.ctx()) << "operands context mismatch";
  }
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NDArray(OP::GetShape(lhs.shape(), rhs.shape()), lhs.ctx(), true, lhs.dtype());
  } else {
    // no check if both of them are on cpu
    if (lhs.ctx().dev_mask() != cpu::kDevMask ||
        out->ctx().dev_mask() != cpu::kDevMask) {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    }
    CHECK(out->shape() == OP::GetShape(lhs.shape(), rhs.shape()))
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
  if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
        Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
            TBlob tmp = ret.data();
            ndarray::Eval<cpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
          }, lhs.ctx(), const_vars, {ret.var()},
          FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void SetValueOp(const real_t &rhs, NDArray *out) {
  CHECK_NE(out->is_none(), true) << "Set value target must not be empty";
  // important: callback must always capture by value
  NDArray ret = *out;
  const NDArrayStorageType stype = ret.storage_type();
  Engine::Get()->PushSync([rhs, ret, stype](RunContext ctx) {
      TBlob tmp = ret.data();
      switch (ret.ctx().dev_mask()) {
        case cpu::kDevMask: {
          if (stype == kDefaultStorage) {
            ndarray::Eval<cpu>(rhs, &tmp, ctx);
          } else {
            ndarray::Eval(ctx.get_stream<cpu>(), rhs, ret);
          }
          break;
        }
#if MXNET_USE_CUDA
        case gpu::kDevMask: {
          if (stype == kDefaultStorage) {
            ndarray::Eval<gpu>(rhs, &tmp, ctx);
          } else {
            ndarray::Eval(ctx.get_stream<gpu>(), rhs, ret);
          }
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
          break;
        }
#endif
        default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      }
    }, ret.ctx(), {}, {ret.var()},
  FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
}

/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP, bool reverse>
void ScalarOp(const NDArray &lhs,
              const real_t &rhs,
              NDArray *out) {
  if (out->is_none()) {
    *out = NDArray(lhs.shape(), lhs.ctx(), true, lhs.dtype());
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == lhs.shape()) << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::Eval<cpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
        }, lhs.ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

size_t num_aux_data(NDArrayStorageType stype) {
  size_t num = 0;
  switch (stype) {
    case kDefaultStorage: num = 0; break;
    case kCSRStorage: num = 2; break;
    case kRowSparseStorage: num = 1; break;
     default: LOG(FATAL) << "Unknown storage type" << stype; break;
  }
  return num;
}

// Make a copy of a CSR NDArray
template<typename from_xpu, typename to_xpu>
inline void CopyFromToCsrImpl(const NDArray& from, const NDArray& to, RunContext ctx) {
  using namespace mshadow;
  CHECK_EQ(from.storage_type(), to.storage_type()) << "Copying with different storage type";
  // if source storage is not initialized, fill destination with zeros
  auto s = ctx.get_stream<to_xpu>();
  if (!from.storage_initialized()) {
    op::FillZerosCsrImpl(s, to);
    return;
  }
  // Allocate storage
  to.CheckAndAllocAuxData(csr::kIndPtr, from.aux_shape(csr::kIndPtr));
  to.CheckAndAllocAuxData(csr::kIdx, from.aux_shape(csr::kIdx));
  to.CheckAndAllocData(from.aux_shape(csr::kIdx));
  TBlob val = to.data();
  TBlob indptr = to.aux_data(csr::kIndPtr);
  TBlob idx = to.aux_data(csr::kIdx);
  ndarray::Copy<from_xpu, to_xpu>(from.data(), &val,
                                  from.ctx(), to.ctx(), ctx);
  ndarray::Copy<from_xpu, to_xpu>(from.aux_data(csr::kIndPtr), &indptr,
                                  from.ctx(), to.ctx(), ctx);
  ndarray::Copy<from_xpu, to_xpu>(from.aux_data(csr::kIdx), &idx,
                                  from.ctx(), to.ctx(), ctx);
}

// Make a copy of a row-sparse NDArray
template<typename from_xpu, typename to_xpu>
inline void CopyFromToRspImpl(const NDArray& from, const NDArray& to, RunContext ctx) {
  using namespace mshadow;
  CHECK_EQ(from.storage_type(), to.storage_type()) << "Copying with different storage type";
  // if source is zeros, fill destination with zeros, too
  auto s = ctx.get_stream<to_xpu>();
  if (!from.storage_initialized()) {
    op::FillZerosRspImpl(s, to);
    return;
  }
  auto aux_shape = from.aux_shape(rowsparse::kIdx);
  to.CheckAndAlloc({aux_shape});
  TBlob val = to.data();
  TBlob idx = to.aux_data(rowsparse::kIdx);
  ndarray::Copy<from_xpu, to_xpu>(from.data(), &val,
                                  from.ctx(), to.ctx(), ctx);
  ndarray::Copy<from_xpu, to_xpu>(from.aux_data(rowsparse::kIdx), &idx,
                                  from.ctx(), to.ctx(), ctx);
}

// Make a copy of a dense NDArray
template<typename from_xpu, typename to_xpu>
inline void CopyFromToDnsImpl(const NDArray& from, const NDArray& to, RunContext ctx) {
  using namespace mshadow;
  CHECK_EQ(from.storage_type(), to.storage_type()) << "Copying with different storage type";
  TBlob tmp = to.data();
  ndarray::Copy<from_xpu, to_xpu>(from.data(), &tmp,
                                  from.ctx(), to.ctx(), ctx);
}

// Make a copy of an NDArray based on storage type
template<typename from_xpu, typename to_xpu>
void CopyFromToImpl(const NDArray& from, const NDArray& to,
                    RunContext rctx, const std::vector<Resource>& requested) {
  using namespace std;
  using namespace mshadow;
  // if storage type doesn't match, cast the storage first
  const NDArrayStorageType from_stype = from.storage_type();
  const NDArrayStorageType to_stype = to.storage_type();
  CHECK(from_stype == kDefaultStorage
      || to_stype == kDefaultStorage
      || from_stype == to_stype)
    << "Copying ndarray of stype = " << from_stype
    << " to stype = " << to_stype << " is not supported";
  const Context from_ctx = from.ctx();
  const Context to_ctx = to.ctx();
  bool is_train = Imperative::Get()->is_training();

  OpContext opctx{is_train,
                  rctx,
                  engine::CallbackOnComplete(),
                  requested};
  if (from_ctx == to_ctx && from_stype != to_stype) {
    // same ctx, different stypes, use cast op directly without copying
    common::CastStorageDispatch<from_xpu>(opctx, from, to);
  } else {
    NDArray casted_nd;  // an intermediate result before copying from to to
    if (from_stype == to_stype) {
      casted_nd = from;  // same stype, no need to cast from
    } else {  // different stypes on different ctx needs an temporary casted_nd
      TShape shape = from.shape();
      if (to_stype == kDefaultStorage) {
        casted_nd = NDArray(shape, from_ctx);
      } else {
        casted_nd = NDArray(to_stype, shape, from_ctx);
      }
      // convert from_nd to the same stype as to_nd
      common::CastStorageDispatch<from_xpu>(opctx, from, casted_nd);
    }

    if (to_stype == kDefaultStorage) {
      CopyFromToDnsImpl<from_xpu, to_xpu>(casted_nd, to, rctx);
    } else if (to_stype == kRowSparseStorage) {
      CopyFromToRspImpl<from_xpu, to_xpu>(casted_nd, to, rctx);
    } else if (to_stype == kCSRStorage) {
      CopyFromToCsrImpl<from_xpu, to_xpu>(casted_nd, to, rctx);
    } else {
      LOG(FATAL) << "unknown storage type" << to_stype;
    }
  }
}

void CopyFromTo(const NDArray& from, const NDArray& to, int priority) {
  if (from.var() == to.var()) {
    // skip to copy to itself
    return;
  }
  CHECK(from.shape() == to.shape())
      << "operands shape mismatch"
      << "from.shape = " << from.shape() << " to.shape=" << to.shape();
  CHECK(from.shape().ndim() != 0)
      << "source operands have zero dimension shape";
  // important: callback must always capture by value
  const Context from_ctx = from.ctx();
  const int a = from_ctx.dev_mask();
  const int b = to.ctx().dev_mask();
  std::vector<Engine::VarHandle> const_vars;
  if (from.var() != to.var()) const_vars.push_back(from.var());

  const NDArrayStorageType from_stype = from.storage_type();
  const NDArrayStorageType to_stype = to.storage_type();

  std::vector<Engine::VarHandle> mutable_vars(1, to.var());

  std::vector<Resource> requested;
  if (from_stype != to_stype) {
    using namespace common;
    static bool log = dmlc::GetEnv("MXNET_STORAGE_FALLBACK_LOG_VERBOSE", true);
    if (log) {
      std::ostringstream os;
      os << "\nStorage fallback detected:\n"
         << "Copy from " << stype_string(from_stype) << " storage type on " << dev_type_string(a)
         << " to " << stype_string(to_stype) << " storage type on " << dev_type_string(b)
         << ".\nA temporary ndarray with " << stype_string(to_stype)
         << " storage type will be generated in order to perform the copy. "
         << "You can set environment variable "
         << "MXNET_STORAGE_FALLBACK_LOG_VERBOSE to 0 to suppress this warning.";
      LogOnce(os.str());
    }

    // request temp resource if cast_storage performs on GPU
    if (a == gpu::kDevMask) {
      Resource rsc = ResourceManager::Get()->Request(from_ctx,
          ResourceRequest(ResourceRequest::kTempSpace));
      requested.push_back(rsc);
      mutable_vars.push_back(rsc.var);
    }
  }

  if (a == cpu::kDevMask && b == cpu::kDevMask) {
    Engine::Get()->PushAsync(
      [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
        CopyFromToImpl<cpu, cpu>(from, to, ctx, requested);
        on_complete();
      }, from.ctx(), const_vars, mutable_vars,
      FnProperty::kNormal, priority, PROFILER_MESSAGE("CopyCPU2CPU"));
  } else {
#if MXNET_USE_CUDA
    if (a == cpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<cpu, gpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, to.ctx(), const_vars, mutable_vars,
        FnProperty::kCopyToGPU, priority, PROFILER_MESSAGE("CopyCPU2GPU"));
    } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<gpu, cpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, from.ctx(), const_vars, mutable_vars,
        FnProperty::kCopyFromGPU, priority, PROFILER_MESSAGE("CopyGPU2CPU"));
    } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<gpu, gpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, from.ctx(), const_vars, mutable_vars,
        from.dtype() != to.dtype() ? FnProperty::kNormal : FnProperty::kCopyFromGPU,
        priority, PROFILER_MESSAGE("CopyGPU2GPU"));
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}


void CopyFromTo(const NDArray& from, const NDArray *to, int priority) {
  CopyFromTo(from, *to, priority);
}

void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority) {
  std::vector<Engine::VarHandle> const_vars;
  const_vars.reserve(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    if (source[i].var() != out->var()) {
      const_vars.push_back(source[i].var());
    }
    CHECK_EQ(source[i].shape() , out->shape())
        << "operands shape mismatch";
    if (out->ctx().dev_mask() == Context::kCPU) {
      CHECK_EQ(source[i].ctx().dev_mask(), Context::kCPU)
          << "operands context mismatch";
    } else {
      CHECK(source[i].ctx() == out->ctx())
          << "operands context mismatch";
    }
  }
  // important: callback must always capture by value
  NDArray ret = *out;

  const NDArrayStorageType stype = ret.storage_type();

  if (stype == kDefaultStorage) {
    switch (out->ctx().dev_mask()) {
      case cpu::kDevMask: {
        Engine::Get()->PushSync([source, ret](RunContext ctx) {
            std::vector<TBlob> source_tblob(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
              source_tblob[i] = source[i].data();
            }
            TBlob tmp = ret.data();
            ndarray::ElementwiseSum<cpu>(source_tblob, &tmp, ctx);
          }, out->ctx(), const_vars, {ret.var()},
          FnProperty::kNormal, priority, PROFILER_MESSAGE_FUNCNAME);
        break;
      }
#if MXNET_USE_CUDA
      case gpu::kDevMask: {
        Engine::Get()->PushSync([source, ret](RunContext ctx) {
            std::vector<TBlob> source_tblob(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
              source_tblob[i] = source[i].data();
            }
            TBlob tmp = ret.data();
            ndarray::ElementwiseSum<gpu>(source_tblob, &tmp, ctx);
            // Wait GPU kernel to complete
            ctx.get_stream<gpu>()->Wait();
          }, out->ctx(), const_vars, {ret.var()},
          FnProperty::kNormal, priority, PROFILER_MESSAGE("DenseElementwiseSum"));
        break;
      }
#endif
      default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
    }
  } else if (stype == kRowSparseStorage) {
    Resource rsc = ResourceManager::Get()->Request(ret.ctx(),
      ResourceRequest(ResourceRequest::kTempSpace));

    Engine::Get()->PushSync(
      [source, ret, rsc](RunContext rctx) {
        NDArray result = ret;
        switch (ret.ctx().dev_mask()) {
          case cpu::kDevMask: {
            mxnet::ndarray::ElementwiseSum(rctx.get_stream<cpu>(), rsc, source, &result);
            break;
          }
#if MXNET_USE_CUDA
          case gpu::kDevMask: {
            mxnet::ndarray::ElementwiseSum(rctx.get_stream<gpu>(), rsc, source, &result);
            // wait for GPU operations to complete
            rctx.get_stream<gpu>()->Wait();
            break;
          }
#endif
          default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        }
      }, ret.ctx(), const_vars, {ret.var(), rsc.var},
    FnProperty::kNormal, priority, PROFILER_MESSAGE("RowSparseElementwiseSum"));
  } else {
    LOG(FATAL) << "Not implemented for storage_type " << common::stype_string(stype);
  }
}

void ClipOp(const NDArray &src,
            const real_t &a_min, const real_t &a_max,
            NDArray *out) {
  if (out->is_none()) {
    *out = NDArray(src.shape(), src.ctx(), true, src.dtype());
  } else {
    CHECK(out->ctx() == src.ctx()) << "target context mismatch";
    CHECK(out->shape() == src.shape()) << "target shape mismatch";
  }
  NDArray ret = *out;
  std::vector<Engine::VarHandle> const_vars;
  if (src.var() != ret.var()) const_vars.push_back(src.var());
  switch (src.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([src, a_min, a_max, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::EvalClip<cpu>(src.data(), a_min, a_max, &tmp, ctx);
        }, src.ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
    #if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([src, a_min, a_max, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::EvalClip<gpu>(src.data(), a_min, a_max, &tmp, ctx);
        }, src.ctx(), const_vars, {ret.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
    #endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

template<typename Distribution>
void SampleOP(const real_t &a,
              const real_t &b,
              NDArray *out) {
  CHECK(!out->is_none());
  Resource resource = ResourceManager::Get()->Request(
      out->ctx(), ResourceRequest::kRandom);
  // important: callback must always capture by value
  NDArray ret = *out;
  // redirect everything to mshadow operations
  switch (out->ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([a, b, resource, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::EvalRandom<cpu, Distribution>(a, b, resource, &tmp, ctx);
        }, out->ctx(), {}, {ret.var(), resource.var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([a, b, resource, ret](RunContext ctx) {
          TBlob tmp = ret.data();
          ndarray::EvalRandom<gpu, Distribution>(a, b, resource, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, out->ctx(), {}, {ret.var(), resource.var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void SampleUniform(real_t begin, real_t end, NDArray *out) {
  SampleOP<ndarray::UniformDistribution>(begin, end, out);
}

void SampleGaussian(real_t mu, real_t sigma, NDArray *out) {
  SampleOP<ndarray::GaussianDistribution>(mu, sigma, out);
}

void SampleExponential(real_t lambda, NDArray *out) {
  if ( out->ctx().dev_mask() != cpu::kDevMask ) {
    LOG(FATAL) <<"exponential sampling only valid on cpu";
  }
  real_t dummy;
  SampleOP<ndarray::ExponentialDistribution>(lambda, dummy, out);
}

void SamplePoisson(real_t lambda, NDArray *out) {
  if ( out->ctx().dev_mask() != cpu::kDevMask ) {
    LOG(FATAL) <<"poisson sampling only valid on cpu";
  }
  real_t dummy;
  SampleOP<ndarray::PoissonDistribution>(lambda, dummy, out);
}

void SampleNegBinomial(int32_t k, real_t p, NDArray *out) {
  if ( out->ctx().dev_mask() != cpu::kDevMask ) {
    LOG(FATAL) <<"negative binomial sampling only valid on cpu";
  }
  SampleOP<ndarray::NegBinomialDistribution>(k, p, out);
}

void SampleGenNegBinomial(real_t mu, real_t alpha, NDArray *out) {
  if ( out->ctx().dev_mask() != cpu::kDevMask ) {
    LOG(FATAL) <<"negative binomial sampling only valid on cpu";
  }
  SampleOP<ndarray::GenNegBinomialDistribution>(mu, alpha, out);
}

void RandomSeed(uint32_t seed) {
  ResourceManager::Get()->SeedRandom(seed);
}

template<typename OP>
inline NDArray BinaryOpRet(const NDArray &lhs,
                           const NDArray &rhs) {
  NDArray ret;
  BinaryOp<OP>(lhs, rhs, &ret);
  return ret;
}

template<typename OP, bool reverse>
inline NDArray ScalarOpRet(const NDArray &lhs,
                           const real_t &rhs) {
  NDArray ret;
  ScalarOp<OP, reverse>(lhs, rhs, &ret);
  return ret;
}

template<typename OP>
inline NDArray &BinaryOpApply(NDArray *dst,
                              const NDArray &src) {
  BinaryOp<OP>(*dst, src, dst);
  return *dst;
}

template<typename OP>
inline NDArray &ScalarOpApply(NDArray *dst,
                             const real_t &src) {
  ScalarOp<OP, false>(*dst, src, dst);
  return *dst;
}

// Binary
NDArray operator+(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Plus>(lhs, rhs);
}
NDArray operator-(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Minus>(lhs, rhs);
}
NDArray operator*(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Mul>(lhs, rhs);
}
NDArray operator/(const NDArray &lhs, const NDArray &rhs) {
  return BinaryOpRet<ndarray::Div>(lhs, rhs);
}
// Scalar
NDArray operator+(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Plus, false>(lhs, rhs);
}
NDArray operator-(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Minus, false>(lhs, rhs);
}
NDArray operator*(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Mul, false>(lhs, rhs);
}
NDArray operator/(const NDArray &lhs, const real_t &rhs) {
  return ScalarOpRet<ndarray::Div, false>(lhs, rhs);
}

// Binary
NDArray &NDArray::operator=(real_t scalar) {
  SetValueOp(scalar, this);
  return *this;
}

NDArray &NDArray::operator+=(const NDArray &src) {
  return BinaryOpApply<ndarray::Plus>(this, src);
}
NDArray &NDArray::operator-=(const NDArray &src) {
  return BinaryOpApply<ndarray::Minus>(this, src);
}
NDArray &NDArray::operator*=(const NDArray &src) {
  return BinaryOpApply<ndarray::Mul>(this, src);
}
NDArray &NDArray::operator/=(const NDArray &src) {
  return BinaryOpApply<ndarray::Div>(this, src);
}
// Scalar
NDArray &NDArray::operator+=(const real_t &src) {
  return ScalarOpApply<ndarray::Plus>(this, src);
}
NDArray &NDArray::operator-=(const real_t &src) {
  return ScalarOpApply<ndarray::Minus>(this, src);
}
NDArray &NDArray::operator*=(const real_t &src) {
  return ScalarOpApply<ndarray::Mul>(this, src);
}
NDArray &NDArray::operator/=(const real_t &src) {
  return ScalarOpApply<ndarray::Div>(this, src);
}

/* magic number for ndarray version 1, with int64_t TShape */
static const uint32_t NDARRAY_V1_MAGIC = 0xF993fac8;

/* magic number for ndarray version 2, with storage type */
static const uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;

void NDArray::Save(dmlc::Stream *strm) const {
  // write magic number to mark this version
  // for storage type
  strm->Write(NDARRAY_V2_MAGIC);

  // save storage type
  int32_t stype = storage_type();
  strm->Write(&stype, sizeof(stype));

  const int32_t nad = num_aux_data(storage_type());
  // save storage shape if ndarray is sparse
  if (nad > 0) {
    storage_shape().Save(strm);
  }

  // save shape
  shape_.Save(strm);
  if (is_none()) return;

  // save context
  Context ctx = this->ctx();
  ctx.Save(strm);
  TBlob save_data;
  NDArray nd_cpu;  // a copy of *this on cpu
  if (ctx.dev_mask() != cpu::kDevMask) {
    nd_cpu = this->Copy(Context::CPU());
    nd_cpu.WaitToRead();
    save_data = nd_cpu.data();
  } else {
    this->WaitToRead();
    save_data = this->data();
    nd_cpu = *this;
  }

  // save type flag
  int32_t type_flag = save_data.type_flag_;
  strm->Write(&type_flag, sizeof(type_flag));

  // save aux_types and aux_shapes
  if (nad > 0) {
    for (int i = 0; i < nad; ++i) {
      int32_t aux_type_flag = aux_type(i);
      strm->Write(&aux_type_flag, sizeof(aux_type_flag));
      aux_shape(i).Save(strm);
    }
  }

  // save data
  CHECK(save_data.CheckContiguous());
  size_t type_size = mshadow::mshadow_sizeof(type_flag);
  // save data could be values of sparse tensors
  // must use save_data.shape_ instead of this->shape_
  strm->Write(save_data.dptr_, type_size * save_data.shape_.Size());

  // save aux data
  if (nad > 0) {
    for (int i = 0; i < nad; ++i) {
      TBlob save_data = nd_cpu.aux_data(i);
      // save aux_data
      CHECK(save_data.CheckContiguous());
      size_t aux_type_size = mshadow::mshadow_sizeof(aux_type(i));
      strm->Write(save_data.dptr_, aux_type_size * save_data.Size());
    }
  }
}

bool LegacyTShapeLoad(dmlc::Stream *strm, TShape *shape, const uint32_t magic) {
  switch (magic) {
    case NDARRAY_V1_MAGIC:
      return shape->Load(strm);
    default:
      // meet legacy TShape, magic is ndim here
      uint32_t ndim = magic;
      *shape = TShape(ndim);
      std::vector<uint32_t> buffer(ndim);
      size_t nread = ndim * sizeof(uint32_t);
      if (strm->Read(buffer.data(), nread) != nread) return false;
      nnvm::ShapeTypeCast(buffer.begin(), buffer.end(), shape->begin());
      return true;
  }
}

bool NDArray::LegacyLoad(dmlc::Stream *strm, const uint32_t magic) {
  // load shape
  TShape shape;
  if (!LegacyTShapeLoad(strm, &shape, magic)) return false;
  if (shape.ndim() == 0) {
    *this = NDArray(); return true;
  }
  // load context
  Context ctx;
  if (!ctx.Load(strm)) return false;
  // load type flag
  int32_t type_flag;
  if (strm->Read(&type_flag, sizeof(type_flag)) != sizeof(type_flag)) return false;
  // load data into CPU
  NDArray temp(shape, Context::CPU(), false, type_flag);
  TBlob load_data = temp.data();
  size_t type_size = mshadow::mshadow_sizeof(type_flag);
  size_t nread = type_size * shape.Size();

  if (strm->Read(load_data.dptr_, nread) != nread) return false;
  if (ctx.dev_mask() == cpu::kDevMask) {
    *this = std::move(temp); return true;
  } else {
#if MXNET_USE_CUDA
    *this = temp.Copy(ctx); return true;
#else
    *this = std::move(temp); return true;
#endif
  }
}

bool NDArray::Load(dmlc::Stream *strm) {
  uint32_t magic;
  if (strm->Read(&magic, sizeof(uint32_t)) != sizeof(uint32_t)) return false;
  if (magic != NDARRAY_V2_MAGIC) {
    return LegacyLoad(strm, magic);
  }

  // load storage type
  int32_t stype;
  if (strm->Read(&stype, sizeof(stype)) != sizeof(stype)) return false;
  const int32_t nad = num_aux_data(static_cast<NDArrayStorageType>(stype));

  // load storage shape
  TShape sshape;
  if (nad > 0) {
    if (!sshape.Load(strm)) return false;
  }

  // load shape
  TShape shape;
  if (!shape.Load(strm)) return false;
  if (shape.ndim() == 0) {
    *this = NDArray(); return true;
  }

  // load context
  Context ctx;
  if (!ctx.Load(strm)) return false;

  // load type flag
  int32_t type_flag;
  if (strm->Read(&type_flag, sizeof(type_flag)) != sizeof(type_flag)) return false;

  // load aux_types and aux_shapes
  std::vector<int32_t> aux_types;
  std::vector<TShape> aux_shapes;
  if (nad > 0) {
    aux_types.resize(nad);
    aux_shapes.resize(nad);
    for (int i = 0; i < nad; ++i) {
      // load aux_type(i)
      if (strm->Read(&aux_types[i], sizeof(aux_types[i])) != sizeof(aux_types[i])) return false;
      // load aux_shapes(i)
      if (!aux_shapes[i].Load(strm)) return false;
    }
  }

  // load data into CPU
  NDArray temp;
  if (0 == nad) {
    temp = NDArray(shape, Context::CPU(), false, type_flag);
  } else {
    temp = NDArray(static_cast<NDArrayStorageType>(stype), shape,
                   Context::CPU(), false, type_flag,
                   aux_types, aux_shapes, sshape);
  }
  // load data
  TBlob load_data = temp.data();
  size_t type_size = mshadow::mshadow_sizeof(type_flag);
  size_t nread = type_size * load_data.Size();
  if (strm->Read(load_data.dptr_, nread) != nread) return false;

  // load aux_data
  if (nad > 0) {
    for (int i = 0; i < nad; ++i) {
      load_data = temp.aux_data(i);
      type_size = mshadow::mshadow_sizeof(load_data.type_flag_);
      nread = type_size * load_data.Size();
      if (strm->Read(load_data.dptr_, nread) != nread) return false;
    }
  }

  if (ctx.dev_mask() == cpu::kDevMask) {
    *this = std::move(temp); return true;
  } else {
#if MXNET_USE_CUDA
    *this = temp.Copy(ctx); return true;
#else
    *this = std::move(temp); return true;
#endif
  }
}

const uint64_t kMXAPINDArrayListMagic = 0x112;

void NDArray::Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names) {
  uint64_t header = kMXAPINDArrayListMagic, reserved = 0;
  fo->Write(&header, sizeof(header));
  fo->Write(&reserved, sizeof(reserved));
  fo->Write(data);
  fo->Write(names);
}

void NDArray::Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys) {
  uint64_t header, reserved;
  CHECK(fi->Read(&header))
      << "Invalid NDArray file format";
  CHECK(fi->Read(&reserved))
      << "Invalid NDArray file format";
  CHECK(header == kMXAPINDArrayListMagic)
      << "Invalid NDArray file format";
  CHECK(fi->Read(data))
      << "Invalid NDArray file format";
  CHECK(fi->Read(keys))
      << "Invalid NDArray file format";
  CHECK(keys->size() == 0 || keys->size() == data->size())
      << "Invalid NDArray file format";
}

NDArray NDArray::Copy(Context ctx) const {
  NDArray ret;
  if (kDefaultStorage == storage_type()) {
    ret = NDArray(shape(), ctx, true, dtype_);
  } else if (kUndefinedStorage != storage_type()) {
    ret = NDArray(storage_type(), shape(), ctx, true, dtype_,
                  ptr_->aux_types, ptr_->aux_shapes, storage_shape());
  } else {
    LOG(FATAL) << "NDArray::Copy cannot copy undefined storage-type ndarray to ctx.dev_type="
               << ctx.dev_type << ", ctx.dev_id=" << ctx.dev_id;
  }
  CopyFromTo(*this, ret);
  return ret;
}

void NDArray::SyncCopyFromCPU(const void *data, size_t size) const {
  TShape dshape = this->shape();
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  TBlob src((void*)data, dshape, cpu::kDevMask, this->dtype_, 0); // NOLINT(*)

  if (this->ctx().dev_mask() == cpu::kDevMask) {
    this->WaitToWrite();
    RunContext rctx{this->ctx(), nullptr};
    TBlob dst = this->data();
    ndarray::Copy<cpu, cpu>(src, &dst, Context::CPU(), Context::CPU(), rctx);
  } else {
#if MXNET_USE_CUDA
    Engine::Get()->PushAsync(
      [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        TBlob dst = this->data();
        ndarray::Copy<cpu, gpu>(src, &dst,
                                Context::CPU(), this->ctx(), rctx);
        // Wait GPU kernel to complete
        rctx.get_stream<gpu>()->Wait();
        on_complete();
      }, this->ctx(), {}, {this->var()},
      FnProperty::kCopyToGPU, 0, PROFILER_MESSAGE("SyncCopyCPU2GPU"));
    this->WaitToRead();
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
}

/*!
 * \brief Copy src.data()/aux_data(i) to dst->data()/aux_data(j).
 */
void NDArray::SyncCopyFromNDArray(const NDArray& src, int i, int j) {
  if (i >= 0) {
    CHECK_NE(src.storage_type(), kDefaultStorage);
  } else {
    CHECK(!src.is_none()) << "src dense ndarray must have been initialized";
  }
  if (j >= 0) {
    CHECK_NE(storage_type(), kDefaultStorage);
  } else {
    CHECK(!this->is_none()) << "dst dense ndarray must have been initialized";
  }

  if (src.var() == var()) {
    // skip to copy to itself
    LOG(WARNING) << "SyncCopyFromNDArray does not support copying to self";
    return;
  }
  const int src_dev_mask = src.ctx().dev_mask();
  const int dst_dev_mask = ctx().dev_mask();
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(src.var());

  // get or create a dst tblob for copying src to it
  // if dst is a dense format and has not been allocated, allocate memory for it
  // else if dst is not initialized, allocate corresponding data blob for it
  auto get_dst_data = [&](const TShape& src_shape) {
    if (this->storage_type() == kDefaultStorage) {
      this->ReshapeAndAlloc(src_shape);
    } else if (!this->storage_initialized()) {
      if (j < 0) {
        this->CheckAndAllocData(src_shape);
      } else {
        this->CheckAndAllocAuxData(j, src_shape);
      }
    }
    TBlob dst_data = (j >= 0? this->aux_data(j) : this->data());
    CHECK_LE(src_shape.Size(), dst_data.shape_.Size());
    return dst_data;
  };

  if (src_dev_mask == cpu::kDevMask && dst_dev_mask == cpu::kDevMask) {
    Engine::Get()->PushSync([&](RunContext rctx) {
        const TBlob src_data = (i >= 0? src.aux_data(i) : src.data());
        TBlob dst_data = get_dst_data(src_data.shape_);
        ndarray::Copy<cpu, cpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
      }, this->ctx(), const_vars, {this->var()},
      FnProperty::kNormal, 0, PROFILER_MESSAGE("SyncCopyFromNDArrayCPU2CPU"));
  } else {
#if MXNET_USE_CUDA
    if (src_dev_mask == cpu::kDevMask && dst_dev_mask == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<cpu, gpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, this->ctx(), const_vars, {this->var()},
        FnProperty::kCopyToGPU, 0, PROFILER_MESSAGE("SyncCopyFromNDArrayCPU2GPU"));
    } else if (src_dev_mask == gpu::kDevMask && dst_dev_mask == cpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<gpu, cpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, this->ctx(), const_vars, {this->var()},
        FnProperty::kCopyFromGPU, 0, PROFILER_MESSAGE("SyncCopyFromNDArrayGPU2CPU"));
    } else if (src_dev_mask == gpu::kDevMask && dst_dev_mask == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<gpu, gpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, this->ctx(), const_vars, {this->var()},
        src.dtype() != this->dtype() ? FnProperty::kNormal : FnProperty::kCopyFromGPU,
        0, PROFILER_MESSAGE("SyncCopyFromNDArrayGPU2GPU"));
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
  // The copy operation was pushed to engine to execute.
  // Need to wait here for it being completed.
  // The reason for pushing the copy operation to engine
  // is because when copying data from a sparse tensor
  // to the current one, that sparse ndarray's storage_shape/aux_shape
  // may not be ready or changed and we need to ensure
  // thread safty for reading the correct shape info to allocate
  // memory for the current ndarray.
  WaitToRead();
}

void NDArray::SyncCopyToCPU(void *data, size_t size) const {
  TShape dshape = this->shape();
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  TBlob dst(data, dshape, cpu::kDevMask, this->dtype_, 0); // NOLINT(*)

  if (this->ctx().dev_mask() == cpu::kDevMask) {
    this->WaitToRead();
    RunContext rctx{this->ctx(), nullptr};
    ndarray::Copy<cpu, cpu>(this->data(), &dst,
                            Context::CPU(), Context::CPU(), rctx);
  } else {
#if MXNET_USE_CUDA
    Engine::Get()->PushAsync(
      [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        ndarray::Copy<gpu, cpu>(this->data(), &dst,
                                this->ctx(), Context::CPU(), rctx);
        // Wait GPU kernel to complete
        rctx.get_stream<gpu>()->Wait();
        on_complete();
      }, this->ctx(), {this->var()}, {},
      FnProperty::kCopyFromGPU, 0, PROFILER_MESSAGE("SyncCopyGPU2CPU"));
    this->WaitToWrite();
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
}

void NDArray::SyncCheckFormat(const bool full_check) const {
  int32_t err = kNormalErr;
  TBlob err_cpu(&err, mshadow::Shape1(1), cpu::kDevMask, 0);
  if (this->ctx().dev_mask() == cpu::kDevMask) {
    Engine::Get()->PushSync([&](RunContext rctx) {
        common::CheckFormatWrapper<cpu>(rctx, *this, err_cpu, full_check);
      }, this->ctx(), {this->var()}, {},
      FnProperty::kNormal, 0, PROFILER_MESSAGE("CheckFormat"));
  } else {
#if MXNET_USE_CUDA
    Engine::Get()->PushSync([&](RunContext rctx) {
        common::CheckFormatWrapper<gpu>(rctx, *this, err_cpu, full_check);
        rctx.get_stream<gpu>()->Wait();
      }, this->ctx(), {this->var()}, {},
      FnProperty::kNormal, 0, PROFILER_MESSAGE("CheckFormat"));
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
  this->WaitToWrite();
  CHECK_NE(err, kCSRShapeErr) << "Shape mismatch of this csr NDArray";
  CHECK_NE(err, kCSRIndPtrErr)
           << "IndPtr of csr NDArray should be non-negative, in non-decreasing order, "
           << "start with 0, and end with value equal with size of indices.";
  CHECK_NE(err, kCSRIdxErr)
           << "Indices of csr NDArray should be non-negative, in ascending order per row "
           << " and less than the number of columns.";
  CHECK_NE(err, kRSPShapeErr) << "Shape mismatch of this row_sparse NDArray";
  CHECK_NE(err, kRSPIdxErr)
          << "Indices of row_sparse NDArray should be non-negative, "
          << "less than the size of first dimension and in ascending order";
  CHECK_EQ(err, kNormalErr) << "Check the validity of this sparse NDArray";
}

#if MXNET_PREDICT_ONLY == 0
// register API function
// those with underscore will be registered at NDArray
MXNET_REGISTER_NDARRAY_FUN(_set_value)
.set_function(SetValueOp);


MXNET_REGISTER_NDARRAY_FUN(_onehot_encode)
.set_function(BinaryOp<ndarray::OneHotEncode>);

MXNET_REGISTER_NDARRAY_FUN(choose_element_0index)
.set_function(BinaryOp<ndarray::MatChooseRowElem>)
.describe("Choose one element from each line(row for python, column for R/Julia)"
          " in lhs according to index indicated by rhs."
          " This function assume rhs uses 0-based index.");

MXNET_REGISTER_NDARRAY_FUN(fill_element_0index)
.set_function(TernaryOp<ndarray::MatFillRowElem>)
.describe("Fill one element of each line(row for python, column for R/Julia)"
" in lhs according to index indicated by rhs and values indicated by mhs."
" This function assume rhs uses 0-based index.");

// register API function
// those with underscore will be registered at NDArray

void CopyFromToSimple(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  CopyFromTo(inputs[0], outputs[0], 0);
}

// copy function is special
// that we need to remove kAcceptEmptyMutateTarget from it
NNVM_REGISTER_OP(_copyto)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", op::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType",
  [](const NodeAttrs& attrs, std::vector<int> *in_type, std::vector<int> *out_type) {
    return !op::type_is_none((*in_type)[0]) && !op::type_is_none((*out_type)[0]);
  })
.set_attr<FInferStorageType>("FInferStorageType",
  [](const NodeAttrs& attrs,
     const int dev_mask,
     DispatchMode* dispatch_mode,
     std::vector<int>* in_attrs,
     std::vector<int>* out_attrs) {
    op::dispatch_mode_assign(dispatch_mode, DispatchMode::kFComputeEx);
    if (op::storage_type_is_none((*out_attrs)[0])) {
      (*out_attrs)[0] = (*in_attrs)[0];
    }
    return true;
  })
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
    return ExecType::kCrossDeviceCopy;
  })
.set_attr<nnvm::FGradient>("FGradient", op::ElemwiseGradUseNone{"_copyto"})
.set_attr<bool>("TIsBackward", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", CopyFromToSimple)
.set_attr<FComputeEx>("FComputeEx<gpu>", CopyFromToSimple)
.add_argument("data", "NDArray", "input data");


void Imdecode(NDArray *ret, NDArray mean, size_t index,
              size_t x0, size_t y0, size_t x1, size_t y1, size_t n_channels,
              size_t size, char *str_img) {
#if MXNET_USE_OPENCV
  cv::Mat buf(1, size, CV_8U, str_img);
  cv::Mat res = cv::imdecode(buf, n_channels == 1 ? 0 : -1);
  CHECK(res.data != NULL) << "OpenCV Failed to decode image";
  CHECK_LE(n_channels, static_cast<size_t>(res.channels()));
  if (y1 - y0 == 0) {
    x0 = 0;
    x1 = res.cols;
    y0 = 0;
    y1 = res.rows;
  }
  CHECK(x1 <= static_cast<size_t>(res.cols) &&
        y1 <= static_cast<size_t>(res.rows));

  if (ret->is_none()) {
    *ret = NDArray(mshadow::Shape3(n_channels, y1-y0, x1-x0),
                   Context::CPU(), false,
                   mean.is_none() ? mshadow::default_type_flag : mean.dtype());
  }
  NDArray buff;
  if (ret->shape().ndim() == 3) {
    buff = ret->Reshape(mshadow::Shape4(1, ret->shape()[0], ret->shape()[1], ret->shape()[2]));
  } else {
    CHECK_EQ(ret->shape().ndim(), 4U);
    buff = ret->Slice(index, index+1);
  }
  CHECK_EQ(buff.ctx().dev_mask(), Context::kCPU);
  CHECK_EQ(n_channels, buff.shape()[1]);
  CHECK_EQ(y1-y0, buff.shape()[2]);
  CHECK_EQ(x1-x0, buff.shape()[3]);
  buff.WaitToWrite();
  if (mean.is_none()) {
    MSHADOW_TYPE_SWITCH(buff.dtype(), DType, {
      mshadow::Tensor<cpu, 4, DType> tensor = buff.data().get<cpu, 4, DType>();
      for (index_t i = 0; i < y1-y0; i++) {
        uchar* im_data = res.ptr<uchar>(y0+i) + res.channels()*x0;
        for (index_t j = 0; j < x1-x0; j++) {
          for (index_t k = 0; k < n_channels; k++) {
            tensor[0][k][i][j] = DType(im_data[k]);  // NOLINT(*)
          }
          im_data += res.channels();
        }
      }
    })
  } else {
    CHECK_EQ(mean.dtype(), buff.dtype());
    CHECK_EQ(mean.ctx().dev_mask(), Context::kCPU);
    CHECK_EQ(mean.shape()[0], buff.shape()[1]);
    CHECK_EQ(mean.shape()[1], buff.shape()[2]);
    CHECK_EQ(mean.shape()[2], buff.shape()[3]);
    mean.WaitToRead();
    MSHADOW_TYPE_SWITCH(buff.dtype(), DType, {
      mshadow::Tensor<cpu, 4, DType> tensor = buff.data().get<cpu, 4, DType>();
      mshadow::Tensor<cpu, 3, DType> tmean = mean.data().get<cpu, 3, DType>();
      for (index_t i = 0; i < y1-y0; i++) {
        uchar* im_data = res.ptr<uchar>(y0+i) + res.channels()*x0;
        for (index_t j = 0; j < x1-x0; j++) {
          for (index_t k = 0; k < n_channels; k++) {
            tensor[0][k][i][j] = DType(im_data[k]) - tmean[k][i][j];  // NOLINT(*)
          }
          im_data += res.channels();
        }
      }
    })
  }
#else
  LOG(FATAL) << "Compile with OpenCV for image decoding.";
#endif  // MXNET_USE_OPENCV
}

MXNET_REGISTER_NDARRAY_FUN(_imdecode)
.set_type_mask(kAcceptEmptyMutateTarget | kNDArrayArgBeforeScalar)
.set_body([](NDArray **u, real_t *s, NDArray **out,
             int num_params, char **param_keys, char **param_vals) {
    CHECK_EQ(num_params, 1);
    Imdecode(out[0], *u[0],
             static_cast<size_t>(s[0]),
             static_cast<size_t>(s[1]),
             static_cast<size_t>(s[2]),
             static_cast<size_t>(s[3]),
             static_cast<size_t>(s[4]),
             static_cast<size_t>(s[5]),
             static_cast<size_t>(s[6]),
             param_vals[0]);
  })
.set_num_use_vars(1)
.set_num_scalars(7)
.set_num_mutate_vars(1)
.describe("Decode an image, clip to (x0, y0, x1, y1), subtract mean, and write to buffer")
.add_argument("mean", "NDArray-or-Symbol", "image mean")
.add_argument("index", "int", "buffer position for output")
.add_argument("x0", "int", "x0")
.add_argument("y0", "int", "y0")
.add_argument("x1", "int", "x1")
.add_argument("y1", "int", "y1")
.add_argument("c", "int", "channel")
.add_argument("size", "int", "length of str_img");
#endif
}  // namespace mxnet
