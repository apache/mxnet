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
#include "../operator/nn/mkldnn/mkldnn_base-inl.h"
#include "../profiler/storage_profiler.h"

#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NDArrayFunctionReg);
}  // namespace dmlc

namespace mxnet {

void NDArray::ReInit(const NDArrayStorageType stype, const mxnet::TShape &shape,
                     Context ctx, int dtype, bool delay_alloc, const std::vector<int> *pAux_types,
                     const mxnet::ShapeVector *pAux_shapes, const mxnet::TShape *pStorage_shapes) {
  Init(stype, shape, dtype);
  if (stype != kDefaultStorage) {
    const auto sparseStorage = stype == kRowSparseStorage;
    if (!sparseStorage && stype != kCSRStorage)
      LOG(FATAL) << "Unknown storage type " << stype;

    const auto &aux_types = (pAux_types && pAux_types->size())? *pAux_types :
                             std::vector<int>(sparseStorage? 1 : 2, mshadow::kInt64);

    const auto &aux_shapes = (pAux_shapes && pAux_shapes->size()) ? *pAux_shapes :
                             ShapeVector(sparseStorage? 1 : 2, TShape(mshadow::Shape1(0)));

    mxnet::TShape storage_shape;
    if (!pStorage_shapes || !pStorage_shapes->Size()) {
      if (sparseStorage) {
        storage_shape = shape;
        storage_shape[0] = aux_shapes[rowsparse::kIdx][0];
      } else {
        storage_shape = aux_shapes[csr::kIdx];
      }
    } else {
      storage_shape = *pStorage_shapes;
    }
    ptr_ = std::make_shared<Chunk>(stype, storage_shape, ctx, delay_alloc,
                                   dtype, aux_types, aux_shapes);
  } else {
    ptr_ = std::make_shared<Chunk>(shape, ctx, delay_alloc, dtype);
  }
}

void NDArray::AssignStorageInfo(const std::string& profiler_scope,
                                const std::string& name) {
  if (is_none()) {
    return;
  }
  ptr_->shandle.profiler_scope = profiler_scope;
  ptr_->shandle.name = name;
#if MXNET_USE_CUDA
  profiler::GpuDeviceStorageProfiler::Get()->UpdateStorageInfo(ptr_->shandle);
#endif  // MXNET_USE_CUDA
  for (Storage::Handle& aux_handle : ptr_->aux_handles) {
    aux_handle.profiler_scope = profiler_scope;
    aux_handle.name = name + "_aux_data";
#if MXNET_USE_CUDA
    profiler::GpuDeviceStorageProfiler::Get()->UpdateStorageInfo(aux_handle);
#endif  // MXNET_USE_CUDA
  }
}

void NDArray::SetShapeFromChunk() const {
  if (Imperative::Get()->is_np_shape() ||
      !(ptr_->storage_shape.ndim() == 1 && ptr_->storage_shape[0] == 0)) {
    shape_ = ptr_->storage_shape;
  }
}

struct ChunkMem {
  Storage::Handle h;
  std::vector<Storage::Handle> aux_h;
#if MXNET_USE_MKLDNN == 1
  std::shared_ptr<MKLDNNMemory> mem;
#endif
};

NDArray::Chunk::~Chunk() {
  bool skip_free = static_data || delay_alloc;
  ChunkMem mem;
  mem.h = this->shandle;
  mem.aux_h = this->aux_handles;
#if MXNET_USE_MKLDNN == 1
  // We want to delete mkldnn memory after deleting the variable.
  mem.mem = this->mkl_mem_;
#endif
  if (auto engine = engine_ref_.lock()) {
    engine->DeleteVariable([mem, skip_free](RunContext s) {
      if (skip_free == false) {
#if MXNET_USE_MKLDNN == 1
        if (mem.mem) {
          CHECK_LE(mem.mem->GetSize(), mem.h.size);
          CHECK_EQ(mem.mem->GetDataHandle(), mem.h.dptr);
        }
#endif
        Storage::Get()->Free(mem.h);
        for (const auto &aux : mem.aux_h) {
          Storage::Get()->Free(aux);
        }
      }
    }, shandle.ctx, var);
  }
}

void NDArray::Chunk::CheckAndAllocData(const mxnet::TShape &shape, int dtype) {
  CHECK_NE(aux_shapes.size(), 0)
      << "data is expected to be allocated after aux_data";
  auto dbytes = shape.Size() * mshadow::mshadow_sizeof(dtype);
  if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
    CHECK_LT(shape.Size(), (int64_t{1} << 31) - 1) <<
              "[CheckAndAllocData] Size of tensor you are trying to allocate is larger than "
              "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
  }
  if (shandle.size < dbytes) {
    // free storage
    Storage::Get()->Free(shandle);
    // init storage
    shandle.size = dbytes;
    Storage::Get()->Alloc(&shandle);
#if MXNET_USE_MKLDNN == 1
    mkl_mem_ = nullptr;
#endif
  }
  // init shape
  storage_shape = shape;
  // delay_alloc is only set when data storage handle is present
  delay_alloc = false;
}

NDArray NDArray::grad() const {
  if (Imperative::AGInfo::IsNone(*this)) return NDArray();
  Imperative::AGInfo& info = Imperative::AGInfo::Get(autograd_entry_.node);
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
  ret.outputs.emplace_back(autograd_entry_);
  return ret;
}

#if MXNET_USE_MKLDNN == 1

NDArray::NDArray(const mkldnn::memory::desc &md)
    : storage_type_(kDefaultStorage), autograd_entry_(nullptr) {
  shape_ = mxnet::TShape(md.data.dims, md.data.dims + md.data.ndims);
  dtype_ = get_mxnet_type(md.data.data_type);
  ptr_ = std::make_shared<Chunk>(shape_, Context::CPU(), true, dtype_);
  ptr_->CheckAndAlloc(md.get_size());
  ptr_->mkl_mem_ = std::make_shared<MKLDNNMemory>(md, ptr_->shandle.dptr);
}

NDArray::NDArray(const std::shared_ptr<mkldnn::memory> &mkldnn_mem)
    : storage_type_(kDefaultStorage), autograd_entry_(nullptr) {
  auto mem_desc = mkldnn_mem->get_desc();
  shape_ = mxnet::TShape(mem_desc.data.dims, mem_desc.data.dims + mem_desc.data.ndims);
  dtype_ = get_mxnet_type(mem_desc.data.data_type);
  ptr_ = std::make_shared<Chunk>(shape_, Context::CPU(), true, dtype_);
  ptr_->shandle.dptr = mkldnn_mem->get_data_handle();
  ptr_->shandle.size = mem_desc.get_size();
  ptr_->delay_alloc = false;
  ptr_->mkl_mem_ = std::make_shared<MKLDNNMemory>(mkldnn_mem);
  ptr_->static_data = true;
}

NDArray NDArray::MKLDNNDataReshape(const mxnet::TShape &shape) const {
  CHECK(!is_none()) << "NDArray is not initialized";
  CHECK_GE(shape_.Size(), shape.Size())
    << "NDArray.Reshape: target shape size is larger current shape";
  CHECK_EQ(storage_type(), kDefaultStorage);
  if (!IsMKLDNNData()) {
    NDArray ret = this->Detach();
    ret.shape_ = shape;
    return ret;
  } else {
    NDArray ret(shape, ctx(), true, dtype());
    // We shouldn't submit the reorder primitive here because submit will
    // be called in operators.
    mkldnn_format_tag_t format = ptr_->mkl_mem_->GetDefaultFormat();
    CHECK(ptr_->IsMKLDNN());
    mkldnn::memory::desc def_desc = ptr_->mkl_mem_->GetDesc(format);
    mkldnn::memory *def_mem = TmpMemMgr::Get()->Alloc(def_desc);
    MKLDNNStream *stream = MKLDNNStream::Get();
    std::shared_ptr<mkldnn::memory> curr_mem = ptr_->mkl_mem_->GetMem();
    stream->RegisterMem(curr_mem);
    std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *curr_mem},
                                                 {MKLDNN_ARG_TO, *def_mem}});
    stream->RegisterPrimArgs(mkldnn::reorder(*curr_mem, *def_mem), args);
    // def_mem points to a memory region in the temp space. It's only valid
    // inside an operator. As such, the returned NDArray can only be valid
    // inside an operator and the shared point doesn't need to do anything
    // when it's destroyed.
    auto tmp = std::shared_ptr<mkldnn::memory>(def_mem, [](mkldnn::memory *mem) {});
    ret.ptr_->mkl_mem_.reset(new MKLDNNMemory(tmp));
    ret.ptr_->shandle.dptr = def_mem->get_data_handle();
    ret.ptr_->shandle.size = def_mem->get_desc().get_size();
    ret.ptr_->delay_alloc = false;
    ret.ptr_->static_data = true;
    ret.byte_offset_ = byte_offset_;
    ret.reuse_ = false;
    return ret;
  }
}
#endif

NDArray NDArray::Reshape(const mxnet::TShape &shape) const {
  CHECK(!is_none()) << "NDArray is not initialized";
  if (Imperative::Get()->is_np_shape()) {
    CHECK_EQ(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape must have the same size as "
        << "current shape.";
  } else {
    CHECK_GE(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape size is larger than the current shape";
  }
  NDArray ret = this->Detach();
  // If the shape doesn't change, we can just return it now.
  if (ret.shape_ == shape)
    return ret;
  // Otherwise, reshape only works on the default layout.
  CHECK_EQ(storage_type(), kDefaultStorage);
  ret.shape_ = shape;
  ret.reuse_ = false;
  return ret;
}

NDArray NDArray::ReshapeWithRecord(const mxnet::TShape &shape) {
  bool is_recording = Imperative::Get()->is_recording();
  bool is_deferred_compute = Imperative::Get()->is_deferred_compute();
  NDArray ret;
  if (!is_deferred_compute) {
    // The new array shares memory with this array, thus make sure this array
    // has been computed already computed. (noop if this array is not deferred)
    Imperative::DCInfo::Compute(*this);
    ret = this->Reshape(shape);
    if (!is_recording) {
      return ret;
    }
  } else {
    if (shape_is_known(this->shape())) {
      // Imperative reshape only works if shape is already known.
      ret = this->Reshape(shape);
    } else {
      // Reshape called on after dynamic shape operator.
      ret = this->Detach();
    }
  }

  if (!is_deferred_compute || shape_is_known(this->shape())) {
    CHECK_EQ(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape must have the same size as "
        << "current shape when recording with autograd "
        << "or in deferred compute mode.";
  }

  nnvm::NodeAttrs attrs;
  std::ostringstream os;
  os << shape;
  if (!Imperative::Get()->is_np_shape()) {
      attrs.op = nnvm::Op::Get("Reshape");;
      attrs.dict.insert({"shape", os.str()});
  } else {
      attrs.op = nnvm::Op::Get("_np_reshape");;
      attrs.dict.insert({"newshape", os.str()});
  }
  attrs.op->attr_parser(&attrs);
  std::vector<NDArray*> inputs(1, this), outputs(1, &ret);

  if (is_recording) {
    Imperative::Get()->RecordOp(std::move(attrs), inputs, outputs);
  } else if (is_deferred_compute) {
    Imperative::Get()->RecordDeferredCompute(std::move(attrs), inputs, outputs);
  }
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
  MSHADOW_TYPE_SWITCH_WITH_BOOL(ret.dtype(), DType, {
    ret.byte_offset_ += begin * length * sizeof(DType);
  });
  ret.reuse_ = false;
  ret.shape_[0] = end - begin;
  return ret;
}

NDArray NDArray::SliceWithRecord(index_t begin, index_t end) {
  bool is_recording = Imperative::Get()->is_recording();
  bool is_deferred_compute = Imperative::Get()->is_deferred_compute();
  NDArray ret;
  if (!is_deferred_compute) {
    // The new array shares memory with this array, thus make sure this array
    // has been computed already computed. (noop if this array is not deferred)
    Imperative::DCInfo::Compute(*this);
    ret = this->Slice(begin, end);
    if (!is_recording) {
      return ret;
    }
  } else {
    if (shape_is_known(this->shape())) {
      // Imperative slice only works if shape is already known.
      ret = this->Slice(begin, end);
    } else {
      // Slice called on after dynamic shape operator.
      ret = this->Detach();
    }
  }

  // fake a slice op
  nnvm::NodeAttrs attrs;
  attrs.op = nnvm::Op::Get("slice");
  attrs.dict.insert({"begin", std::to_string(begin)});
  attrs.dict.insert({"end", std::to_string(end)});
  attrs.op->attr_parser(&attrs);
  std::vector<NDArray*> inputs(1, this), outputs(1, &ret);

  if (is_recording) {
    Imperative::Get()->RecordOp(std::move(attrs), inputs, outputs);
  } else if (is_deferred_compute) {
    Imperative::Get()->RecordDeferredCompute(std::move(attrs), inputs, outputs);
  }

  return ret;
}

NDArray NDArray::At(index_t idx) const {
  CHECK(storage_type() == kDefaultStorage)
      << "Storage type " << storage_type() << " doesn't support At()";
  NDArray ret = this->Slice(idx, idx+1);
  if (shape_.ndim() > 1) {
    return ret.Reshape(mxnet::TShape(shape_.data()+1, shape_.data()+shape_.ndim()));
  } else {
    return ret;
  }
}

NDArray NDArray::AtWithRecord(index_t idx) {
  CHECK(storage_type() == kDefaultStorage)
      << "Storage type " << storage_type() << " doesn't support At()";
  NDArray sliced = this->SliceWithRecord(idx, idx+1);
  if (shape_.ndim() > 1 || Imperative::Get()->is_np_shape()) {
    // Imperative reshape with concrete shape
    NDArray reshaped = sliced.Reshape(mxnet::TShape(shape_.data()+1, shape_.data()+shape_.ndim()));

    // Record reshape with magic numbers
    nnvm::NodeAttrs attrs;
    std::ostringstream os;
    if (!Imperative::Get()->is_np_shape()) {
        os << mxnet::TShape({-3, -2});  // See ndarray.py reshape for definition of magic numbers
        attrs.op = nnvm::Op::Get("Reshape");;
        attrs.dict.insert({"shape", os.str()});
    } else {
        // See NumpyXReshapeInferShape for definition of magic numbers
        os << mxnet::TShape({-3, -4});
        attrs.op = nnvm::Op::Get("_npx_reshape");;
        attrs.dict.insert({"newshape", os.str()});
    }
    attrs.op->attr_parser(&attrs);
    std::vector<NDArray*> inputs(1, &sliced), outputs(1, &reshaped);

    bool is_recording = Imperative::Get()->is_recording();
    bool is_deferred_compute = Imperative::Get()->is_deferred_compute();
    if (is_recording) {
        Imperative::Get()->RecordOp(std::move(attrs), inputs, outputs);
    } else if (is_deferred_compute) {
        Imperative::Get()->RecordDeferredCompute(std::move(attrs), inputs, outputs);
    }

    return reshaped;
  } else {
    return sliced;
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
  NDArray ret(mxnet::TShape(), ctx(), true, aux_type(i));
  ret.SyncCopyFromNDArray(*this, i);
  return ret;
}

NDArray NDArray::data_ndarray() const {
  NDArray ret(mxnet::TShape(), ctx(), true, dtype_);
  ret.SyncCopyFromNDArray(*this);
  return ret;
}

struct NDArrayDLManager {
  NDArray handle;  // ref NDArray
  DLManagedTensor tensor;
};

DLManagedTensor* NDArray::ToDLPack() const {
  CHECK(!is_none()) << "NDArray is not initialized";
  NDArrayDLManager* dlmanager(new NDArrayDLManager);
  dlmanager->handle = *this;
  dlmanager->tensor.dl_tensor = dlmanager->handle.data().dltensor();
  dlmanager->tensor.manager_ctx = dlmanager;
  dlmanager->tensor.deleter = [](DLManagedTensor* dlmanager){
    delete static_cast<NDArrayDLManager*>(dlmanager->manager_ctx);
  };
  return &(dlmanager->tensor);
}

NDArray NDArray::FromDLPack(const DLManagedTensor* tensor, bool transient_handle) {
  DLManagedTensor *tensor_copy = transient_handle
                               ? new DLManagedTensor(*tensor)
                               : const_cast<DLManagedTensor*>(tensor);
  auto deleter = [tensor_copy, transient_handle](){
    if (tensor_copy->deleter != nullptr) {
      tensor_copy->deleter(tensor_copy);
    }
    if (transient_handle) {
      delete tensor_copy;
    }
  };
  return NDArray(TBlob(tensor_copy->dl_tensor), tensor_copy->dl_tensor.ctx.device_id, deleter);
}

bool NDArray::fresh_out_grad() const {
  if (Imperative::AGInfo::IsNone(*this)) return false;
  Imperative::AGInfo& info = Imperative::AGInfo::Get(autograd_entry_.node);
  return info.fresh_out_grad;
}


void NDArray::set_fresh_out_grad(bool state) const {
  CHECK(!Imperative::AGInfo::IsNone(*this))
    << "NDArray has not been marked as a variable and does not have gradient state";
  Imperative::AGInfo& info = Imperative::AGInfo::Get(autograd_entry_.node);
  info.fresh_out_grad = state;
}

#if MXNET_USE_MKLDNN == 1

bool NDArray::Chunk::IsMKLDNN() const {
  if (storage_type != kDefaultStorage)
    return false;
  if (mkl_mem_ == nullptr)
    return false;
  return mkl_mem_->IsMKLDNN();
}

bool NDArray::Chunk::IsDefault() const {
  if (storage_type != kDefaultStorage)
    return false;
  // If we don't have mkldnn memory yet, we just assume it's not the default
  // format.
  if (mkl_mem_ == nullptr)
    return true;
  return !mkl_mem_->IsMKLDNN();
}

void NDArray::Chunk::Reorder2Default() {
  if (mkl_mem_ == nullptr)
    return;

  if (IsDefault())
    return;

  mkldnn_format_tag_t format = mkl_mem_->GetDefaultFormat();
  mkldnn::memory::desc def_desc = mkl_mem_->GetDesc(format);
  mkldnn_mem_ptr def_mem(new mkldnn::memory(def_desc, CpuEngine::Get()->get_engine()));
  mkl_mem_->ReorderTo(def_mem.get());

  CHECK(shandle.size >= def_desc.get_size());
  CheckAndAlloc(def_desc.get_size());
  // TODO(zhengda) We need to avoid memory copy here.
  memcpy(shandle.dptr, def_mem->get_data_handle(), def_desc.get_size());
  mkl_mem_ = nullptr;
}

void NDArray::Chunk::MKLDNNDataReorder(const mkldnn::memory::desc &md) {
  // If the memory already uses the specified layout, don't do anything.
  if (mkl_mem_ != nullptr && mkl_mem_->SameFormat(md))
    return;

  // If the memory is default, don't do anything.
  if (!mxnet::IsMKLDNN(md) && IsDefault())
    return;
  if (!mxnet::IsMKLDNN(md)) {
    // If the specified layout is default, we should use Reorder2Default.
    Reorder2Default();
    return;
  }
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::stream s(engine);

  std::shared_ptr<mkldnn::memory> new_mem(new mkldnn::memory(md, engine));
  std::shared_ptr<mkldnn::memory> old_mem;
  if (IsDefault()) {
    mkldnn_format_tag_t def_format = GetDefaultFormat(md);
    mkldnn::memory::desc def_desc = GetDesc(md, def_format);
    old_mem.reset(new mkldnn::memory(def_desc, engine, shandle.dptr));
  } else {
    old_mem = this->mkl_mem_->GetMem();
  }
  CHECK(old_mem->get_desc().data.ndims == md.data.ndims);

  // This may be called in MKLDNN operators. We can't use MKLDNNStream here.
  mkldnn::reorder(*old_mem, *new_mem).execute(s, *old_mem, *new_mem);

  CHECK(shandle.size >= md.get_size());
  CheckAndAlloc(md.get_size());
  // TODO(zhengda) We need to avoid memory copy here.
  memcpy(shandle.dptr, new_mem->get_data_handle(), md.get_size());
  mkl_mem_.reset(new MKLDNNMemory(md, shandle.dptr));
}

void NDArray::Chunk::SetMKLMem(const mxnet::TShape &shape, int dtype) {
  // The shape of the array and the one of the MKL memory may mismatch.
  // For example, if the array stores parameters, the MKL memory may store data
  // in 5 dimensions while the NDArray stores data in 4 dimensions.
  if (mkl_mem_ && mkl_mem_->GetDataHandle() == shandle.dptr
      && mkl_mem_->SameFormat(shape, dtype)) {
    return;
  }

  mkldnn::memory::dims dims;
  // These are shapes supprted by MKLDNN.
  if (shape.ndim() >= 1 && shape.ndim() <= 6) {
    dims.resize(shape.ndim());
    for (size_t i = 0; i < dims.size(); i++)
      dims[i] = shape[i];
  } else {
    LOG(FATAL) << "MKLDNN doesn't support " << shape.ndim() << " dimensions";
  }
  mkldnn::memory::format_tag layout = mkldnn::memory::format_tag::undef;
  switch (dims.size()) {
    case 1: layout = mkldnn::memory::format_tag::a; break;
    case 2: layout = mkldnn::memory::format_tag::ab; break;
    case 3: layout = mkldnn::memory::format_tag::abc; break;
    case 4: layout = mkldnn::memory::format_tag::abcd; break;
    case 5: layout = mkldnn::memory::format_tag::abcde; break;
    case 6: layout = mkldnn::memory::format_tag::abcdef; break;
    default:
      LOG(FATAL) << "Not implemented dimension (" << dims.size() << ") for MKLDNN";
  }
  mkldnn::memory::desc data_md{dims, get_mkldnn_type(dtype), layout};
  if (shandle.dptr == nullptr) {
    CHECK(delay_alloc);
    CheckAndAlloc();
  }
  CHECK(shandle.size >= data_md.get_size());
  mkl_mem_.reset(new MKLDNNMemory(data_md, shandle.dptr));
}

const mkldnn::memory *NDArray::GetMKLDNNData(const mkldnn::memory::desc &desc) const {
  if (desc.get_size() != shape().Size() * GetTypeSize(dtype_)) {
    LOG(FATAL) << "The size of NDArray doesn't match the requested MKLDNN memory desc";
    return nullptr;
  }
  const mkldnn::memory *mem = GetMKLDNNData();
  mkldnn::memory::desc desc1 = mem->get_desc();
  // The MKL memory has the same format and shape as required,
  // or both use the default format, we can return the MKL memory.
  if (desc1 == desc || ((!mxnet::IsMKLDNN(desc1)) && (!mxnet::IsMKLDNN(desc)))) {
    return GetMKLDNNExact(mem, desc);
  } else {
    return nullptr;
  }
}

const mkldnn::memory *NDArray::GetMKLDNNDataReorder(
    const mkldnn::memory::desc &new_desc) const {
  CHECK(storage_type() == kDefaultStorage);

  const mkldnn::memory *mem = GetMKLDNNData();
  // If the memory descriptor matches, it's easy.
  MKLDNNStream *stream = MKLDNNStream::Get();
  if (mem->get_desc() == new_desc) {
    return GetMKLDNNExact(mem, new_desc);
  }

  mkldnn::memory::desc old_desc = mem->get_desc();
  // Now we need to determine if we should reorder the memory.
  // If both use the default formats, we think we don't need to reorder.
  if ((!mxnet::IsMKLDNN(old_desc)) && (!mxnet::IsMKLDNN(new_desc))) {
    mkldnn_mem_ptr ret(new mkldnn::memory(new_desc,
        CpuEngine::Get()->get_engine(), mem->get_data_handle()));
    stream->RegisterMem(ret);
    return ret.get();
  } else if (same_shape(old_desc, new_desc)) {
    // If they have the same shape, we can reorder data directly.
    mkldnn::memory *ret = TmpMemMgr::Get()->Alloc(new_desc);
    std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *mem}, {MKLDNN_ARG_TO, *ret}});
    stream->RegisterPrimArgs(mkldnn::reorder(*mem, *ret), args);
    return ret;
  } else {
    // If they have different shapes, we need to reshape the array first.
    // Since this method will only be used inside an operator, we can call
    // MKLDNNDataReshape to reshape an array.
    mxnet::TShape required_shape(new_desc.data.ndims, -1);
    for (int i = 0; i < new_desc.data.ndims; i++)
      required_shape[i] = new_desc.data.dims[i];
    NDArray reshaped = MKLDNNDataReshape(required_shape);
    const mkldnn::memory *ret = reshaped.GetMKLDNNData();
    if (ret->get_desc() == new_desc) {
      return GetMKLDNNExact(ret, new_desc);
    } else {
      mkldnn::memory *ret2 = TmpMemMgr::Get()->Alloc(new_desc);
      std::unordered_map<int, mkldnn::memory> args({{MKLDNN_ARG_FROM, *ret},
                                                   {MKLDNN_ARG_TO, *ret2}});
      stream->RegisterPrimArgs(mkldnn::reorder(*ret, *ret2), args);
      return ret2;
    }
  }
}

NDArray NDArray::Reorder2Default() const {
  CHECK(storage_type() == kDefaultStorage);

  if (ptr_->mkl_mem_ == nullptr)
    return *this;
  if (!ptr_->mkl_mem_->IsMKLDNN())
    return *this;

  // create new ndarray from  mkldnn layout
  mkldnn::memory::desc from_desc = ptr_->mkl_mem_->GetDesc();
  mxnet::TShape tshape(from_desc.data.ndims, -1);
  for (int i = 0; i < from_desc.data.ndims; i++) tshape[i] = from_desc.data.dims[i];
  NDArray ret(tshape, ctx(), false, dtype());
  mkldnn_format_tag_t format = ptr_->mkl_mem_->GetDefaultFormat();
  mkldnn::memory::desc def_desc = ptr_->mkl_mem_->GetDesc(format);
  CHECK(ret.ptr_->shandle.size >= def_desc.get_size());
  mkldnn::memory def_mem(def_desc, CpuEngine::Get()->get_engine(), ret.ptr_->shandle.dptr);
  ptr_->mkl_mem_->ReorderTo(&def_mem);
  // reshape as needed
  ret.shape_ = shape_;
  ret.byte_offset_ = byte_offset_;
  ret.reuse_ = false;
  return ret;
}

void NDArray::SelfReorder2Default() {
  if (!IsMKLDNNData())
    return;

  CHECK(storage_type() == kDefaultStorage);

  const auto mkl_mem = ptr_->mkl_mem_;
  if (mkl_mem == nullptr || !mkl_mem->IsMKLDNN())
    return;

  // create new ndarray from  mkldnn layout
  mkldnn::memory::desc from_desc = mkl_mem->GetDesc();
  mxnet::TShape tshape(from_desc.data.ndims, -1);
  for (int i = 0; i < from_desc.data.ndims; i++)
    tshape[i] = from_desc.data.dims[i];

  const auto saved_shape = shape_;
  const auto saved_byte_offset = byte_offset_;
  this->ReInit(kDefaultStorage, tshape,  ctx(), dtype(), false);

  mkldnn_format_tag_t format = mkl_mem->GetDefaultFormat();
  mkldnn::memory::desc def_desc = mkl_mem->GetDesc(format);
  CHECK(ptr_->shandle.size >= def_desc.get_size());
  mkldnn::memory def_mem(def_desc, CpuEngine::Get()->get_engine(), ptr_->shandle.dptr);
  mkl_mem->ReorderTo(&def_mem);
  // reshape as needed
  shape_ = saved_shape;
  byte_offset_ = saved_byte_offset;
  reuse_ = false;
}

void NDArray::Reorder2DefaultAsync() const {
  std::vector<Engine::VarHandle> const_vars;
  std::vector<Engine::VarHandle> mutable_vars(1, this->var());
  NDArray tmp = *this;
  Engine::Get()->PushAsync(
    [tmp](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      tmp.ptr_->Reorder2Default();
      on_complete();
    }, ctx(), const_vars, mutable_vars,
    FnProperty::kNormal, 0, "Reorder2Default");
}

// now just support bf16->fp32
NDArray NDArray::Reorder2DefaultFloatFormat() const {
  CHECK(storage_type() == kDefaultStorage && IsView() == false);
  if (dtype() !=  mshadow::kBfloat16) {
    return Reorder2Default();
  }
  NDArray ret(shape(), ctx(), false, mshadow::DataType<float>::kFlag);
  auto src_mem = GetMKLDNNData();
  auto dst_mem = ret.GetMKLDNNData();
  ReorderTo(src_mem, dst_mem);

  return ret;
}

void NDArray::MKLDNNDataReorderAsync(const mkldnn::memory::desc &desc) const {
  std::vector<Engine::VarHandle> const_vars;
  std::vector<Engine::VarHandle> mutable_vars(1, this->var());
  NDArray tmp = *this;
  const auto version = this->version();
  Engine::Get()->PushAsync(
    [tmp, version, desc](RunContext ctx, Engine::CallbackOnComplete on_complete) {
      // MXNet will try to reuse NDArray from memory planning, so we need to ensure
      // the NDArray is still holding the original trunk data.
      if (tmp.version() == version) {
        tmp.ptr_->MKLDNNDataReorder(desc);
      }
      on_complete();
    }, ctx(), const_vars, mutable_vars,
    FnProperty::kNormal, 0, "Reorder");
}

const mkldnn::memory *NDArray::GetMKLDNNData() const {
  CHECK(storage_type() == kDefaultStorage);
  const auto is_view = IsView();
  if (IsMKLDNNData()) {
    // If this array uses MKLDNN layout, we have to make sure it's not a view.
    // Otherwise, we'll have to change the layout inside the array.
    CHECK(!is_view);
    MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
    // If this array uses MKLDNN format, we should return now. Otherwise,
    // SetMKLMem may mess up mkl_mem_.
    return ptr_->mkl_mem_->GetRaw();
  }

  CheckAndAlloc();
  if (is_view) {
    // If this is a view, we can't create a MKLDNN memory for the chunk
    // because we don't have the complete data type and shape information for
    // the chunk.
    void *off_addr = static_cast<char *>(ptr_->shandle.dptr) + byte_offset_;
    // Create the primitive desc for the new mkldnn memory.
    mkldnn::memory::dims dims(shape().ndim());
    for (size_t i = 0; i < dims.size(); i++)
      dims[i] = shape()[i];

    const auto cpp_format = static_cast<mkldnn::memory::format_tag>(
        GetDefaultFormat(shape().ndim()));
    mkldnn::memory::desc data_md(dims, get_mkldnn_type(dtype_), cpp_format);
    std::shared_ptr<mkldnn::memory> ret(
        new mkldnn::memory(data_md, CpuEngine::Get()->get_engine(), off_addr));
    MKLDNNStream::Get()->RegisterMem(ret);
    return ret.get();
  }

  // If this isn't a view, we can create a MKLDNN memory and store it in the chunk
  ptr_->SetMKLMem(shape_, dtype_);
  MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
  return ptr_->mkl_mem_->GetRaw();
}

void NDArray::InvalidateMKLDNNData() {
  // Removing mkl_mem_ means the NDArray will store data in the default format.
  if (ptr_->mkl_mem_ && ptr_->mkl_mem_->IsMKLDNN())
    ptr_->mkl_mem_ = nullptr;
}

void NDArray::CopyFrom(const mkldnn::memory &mem) {
  CHECK(ptr_ != nullptr) << "The NDArray hasn't been initialized";
  if (ptr_->mkl_mem_ && ptr_->mkl_mem_->GetRaw() == &mem)
    return;

  CHECK(mem.get_desc().get_size() == shape().Size() * GetTypeSize(dtype_))
      << "The size of NDArray doesn't match the requested MKLDNN memory desc";
  // If this array uses MKLDNN layout, we have to make sure it's not a view.
  // Otherwise, we'll have to change the layout inside the array.

  if (IsMKLDNNData() && IsView())
    ptr_->Reorder2Default();

  const mkldnn::memory *this_mem = GetMKLDNNData();
  MKLDNNMemoryCopy(mem, this_mem);
}

mkldnn::memory *NDArray::CreateMKLDNNData(const mkldnn::memory::desc &desc) {
  if (desc.get_size() != shape().Size() * GetTypeSize(dtype_)) {
    LOG(FATAL) << "The size of NDArray doesn't match the requested MKLDNN memory desc. "
        << "MKLDNN memory requests for " << desc.get_size() << " bytes, but got "
        << shape().Size() * GetTypeSize(dtype_) << " bytes from NDArray";
    return nullptr;
  }
  bool isDefaultFormat = IsDefaultFormat(desc);
  if (isDefaultFormat && !IsView()) {
    ptr_->SetMKLMem(shape_, dtype_);
    MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
    return GetMKLDNNExact(ptr_->mkl_mem_->GetRaw(), desc);
  } else if (isDefaultFormat) {
    ptr_->CheckAndAlloc();
    CHECK(ptr_->shandle.dptr);
    // When this is a view and a user wants the default layout, we can simply
    // create a new mkldnn memory that points to the right memory.
    std::shared_ptr<mkldnn::memory> mem(new mkldnn::memory(desc,
        CpuEngine::Get()->get_engine(), static_cast<char *>(ptr_->shandle.dptr) + byte_offset_));
    MKLDNNStream::Get()->RegisterMem(mem);
    return mem.get();
  } else if (IsView()) {
    // If this is a view and a user wants to write data to it with special
    // a MKLDNN format, we should reorder the data in the array and return NULL.
    // In this way, the user will create a new NDArray for the special format
    // and copy data back.
    ptr_->Reorder2Default();
    return nullptr;
  }

  if (ptr_->mkl_mem_)
    CHECK(ptr_->mkl_mem_->GetDataHandle() == ptr_->shandle.dptr);
  if (ptr_->mkl_mem_ && ptr_->mkl_mem_->GetDesc() == desc) {
    MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
    return GetMKLDNNExact(ptr_->mkl_mem_->GetRaw(), desc);
  }

  CHECK(ptr_->shandle.size >= desc.get_size());
  ptr_->CheckAndAlloc(desc.get_size());
  ptr_->mkl_mem_.reset(new MKLDNNMemory(desc, ptr_->shandle.dptr));
  MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
  return ptr_->mkl_mem_->GetRaw();
}

void NDArray::UpdateMKLDNNMemDesc(const mkldnn::memory::desc &desc) {
  auto new_desc = desc;
  auto this_dtype = get_mkldnn_type(dtype());
  new_desc.data.data_type = static_cast<mkldnn_data_type_t>(this_dtype);
  ptr_->mkl_mem_.reset(new MKLDNNMemory(new_desc, ptr_->shandle.dptr));
  MKLDNNStream::Get()->RegisterMem(ptr_->mkl_mem_->GetMem());
}

#endif

void NDArray::SetTBlob() const {
  CHECK(ptr_ != nullptr);
  mxnet::TShape shape = shape_;
  char *dptr = static_cast<char*>(ptr_->shandle.dptr);
  auto stype = storage_type();
  if (stype == kDefaultStorage) {
#if MXNET_USE_MKLDNN == 1
    CHECK(!IsMKLDNNData()) << "We can't generate TBlob for MKLDNN data. "
        << "Please use Reorder2Default() to generate a new NDArray first";
#endif
    dptr += byte_offset_;
  } else if (stype == kCSRStorage || stype == kRowSparseStorage) {
    CHECK_EQ(byte_offset_, 0);
    shape = storage_shape();
  } else {
    LOG(FATAL) << "unknown storage type " << stype;
  }
  tblob_.dptr_ = dptr;
  tblob_.shape_ = shape;
  tblob_.type_flag_ = dtype_;
  tblob_.SetDLTensor(ptr_->shandle.ctx.dev_mask(), ptr_->shandle.ctx.dev_id);
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
* \brief Performs some preparation required to apply binary operators.
* Checks context and shape of ndarrays, allocates space for output
* and prepares const variables for engine
* \param lhs left operand
* \param rhs right operand
* \param out the output ndarray
* \param binary_op the real operation
*/
template<typename OP>
std::vector<Engine::VarHandle> BinaryOpPrepare(const NDArray &lhs,
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
  std::vector<Engine::VarHandle> const_vars;
  // prepare const variables for engine
  if (lhs.var() != out->var()) const_vars.push_back(lhs.var());
  if (rhs.var() != out->var()) const_vars.push_back(rhs.var());
  return const_vars;
}

/*!
* \brief run a binary operation using the kernel launch method
* \param lhs left operand
* \param rhs right operand
* \param out the output ndarray
* \param binary_op the real operation
*/
template<typename OP>
void BinaryOpKernel(const NDArray &lhs,
                    const NDArray &rhs,
                    NDArray *out) {
  std::vector<Engine::VarHandle> const_vars = BinaryOpPrepare<OP>(lhs, rhs, out);
  // important: callback must always capture by value
  NDArray ret = *out;
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
        TBlob tmp = ret.data();
        mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
        ndarray::BinaryOpKernelImpl<OP>(s, lhs.data(), rhs.data(), &tmp);
      },
      lhs.ctx(), const_vars, {ret.var()},
      FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
        TBlob tmp = ret.data();
        mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
        ndarray::BinaryOpKernelImpl<OP>(s, lhs.data(), rhs.data(), &tmp);
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

/*!
 * \brief run a binary operation using mshadow operations
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real operation
 */
template<typename OP>
void BinaryOp(const NDArray &lhs,
              const NDArray &rhs,
              NDArray *out) {
  std::vector<Engine::VarHandle> const_vars = BinaryOpPrepare<OP>(lhs, rhs, out);
  // important: callback must always capture by value
  NDArray ret = *out;
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
  const auto& aux_shape = from.aux_shape(rowsparse::kIdx);
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
#if MXNET_USE_MKLDNN == 1
  // If neither is MKLDNN, we can copy data normally.
  if (!from.IsMKLDNNData() && !to.IsMKLDNNData()) {
#endif
    using namespace mshadow;
    CHECK_EQ(from.storage_type(), to.storage_type()) << "Copying with different storage type";
    TBlob tmp = to.data();
    ndarray::Copy<from_xpu, to_xpu>(from.data(), &tmp,
                                    from.ctx(), to.ctx(), ctx);
#if MXNET_USE_MKLDNN == 1
  } else if (SupportMKLDNN(from.dtype(), from.shape())
             && SupportMKLDNN(to.dtype(), to.shape())
             && from.ctx().dev_mask() == cpu::kDevMask
             && to.ctx().dev_mask() == cpu::kDevMask) {
    // If we copy data directly, we need to make sure both NDArrays are supported
    // by MKLDNN.
    auto from_mem = from.GetMKLDNNData();
    auto to_mem = to.GetMKLDNNData();
    if (from_mem->get_desc() == to_mem->get_desc()) {
      size_t size = std::min(from_mem->get_desc().get_size(),
                             to_mem->get_desc().get_size());
      memcpy(to_mem->get_data_handle(), from_mem->get_data_handle(), size);
    } else {
      const_cast<NDArray &>(to).CopyFrom(*from_mem);
      MKLDNNStream::Get()->Submit();
    }
  } else {
    // In this case, one of the NDArray isn't supported by MKLDNN, we need
    // to convert the MKLDNN array to the default format first and copy data
    // with Copy().
    NDArray tmp_from = from;
    if (tmp_from.IsMKLDNNData()) {
      // TODO(zhengda) tmp_from should be cached.
      tmp_from = NDArray(from.shape(), from.ctx(), false, from.dtype());
      auto tmp_mem = from.GetMKLDNNData();
      tmp_from.CopyFrom(*tmp_mem);
      MKLDNNStream::Get()->Submit();
    }
    CHECK(tmp_from.IsDefaultData());
    CHECK(to.IsDefaultData());
    TBlob tmp = to.data();
    ndarray::Copy<from_xpu, to_xpu>(tmp_from.data(), &tmp,
                                    from.ctx(), to.ctx(), ctx);
  }
#endif
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

  OpContext opctx{Imperative::Get()->is_recording(),
                  is_train,
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
      const mxnet::TShape& shape = from.shape();
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

void CopyFromTo(const NDArray& from, const NDArray& to, int priority, bool is_opr) {
  if (from.var() == to.var() && from.byte_offset() == to.byte_offset()) {
    // skip to copy to itself
    return;
  }
  CHECK(from.shape() == to.shape())
      << "operands shape mismatch "
      << "from.shape = " << from.shape() << " to.shape=" << to.shape();
  CHECK(!mxnet::op::shape_is_none(from.shape()))
      << "source operands have undefined shape";
  // zero-size array, no need to copy
  if (from.shape().Size() == 0U) {
    return;
  }
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
            "This does not affect the correctness of the programme. "
            "You can set environment variable "
            "MXNET_STORAGE_FALLBACK_LOG_VERBOSE to 0 to suppress this warning.";
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
      FnProperty::kNormal, priority, "CopyCPU2CPU");
  } else {
#if MXNET_USE_CUDA
    if (a == cpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<cpu, gpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, to.ctx(), const_vars, mutable_vars,
        FnProperty::kCopyToGPU, priority, "CopyCPU2GPU");
    } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<gpu, cpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, from.ctx(), const_vars, mutable_vars,
        FnProperty::kCopyFromGPU, priority, "CopyGPU2CPU");
    } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [from, to, requested](RunContext ctx, Engine::CallbackOnComplete on_complete) {
          CopyFromToImpl<gpu, gpu>(from, to, ctx, requested);
          ctx.get_stream<gpu>()->Wait();
          on_complete();
        }, from.ctx(), const_vars, mutable_vars,
        from.dtype() != to.dtype() ? FnProperty::kNormal : FnProperty::kCopyFromGPU,
        priority, is_opr ? "_copyto_GPU2GPU" : "CopyGPU2GPU");
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
  for (const auto& source_array : source) {
    if (source_array.var() != out->var()) {
      const_vars.push_back(source_array.var());
    }
    CHECK_EQ(source_array.shape() , out->shape())
        << "operands shape mismatch";
    if (out->ctx().dev_mask() == Context::kCPU) {
      CHECK_EQ(source_array.ctx().dev_mask(), Context::kCPU)
          << "operands context mismatch";
    } else {
      CHECK_EQ(source_array.ctx(), out->ctx())
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
          FnProperty::kNormal, priority, "DenseElementwiseSum");
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
    FnProperty::kNormal, priority, "RowSparseElementwiseSum");
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

void RandomSeed(Context ctx, uint32_t seed) {
  ResourceManager::Get()->SeedRandom(ctx, seed);
}

template<typename OP>
inline NDArray BinaryOpRet(const NDArray &lhs,
                           const NDArray &rhs) {
  NDArray ret;
  BinaryOpKernel<OP>(lhs, rhs, &ret);
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
  BinaryOpKernel<OP>(*dst, src, dst);
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

/* magic number for ndarray version 1, with int64_t mxnet::TShape */
static const uint32_t NDARRAY_V1_MAGIC = 0xF993fac8;

/* magic number for ndarray version 2, with storage type */
static const uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;

// magic number for ndarray version 3, with np shape semantics.
// The ndarray must be saved and loaded within np shape semantics.
static const uint32_t NDARRAY_V3_MAGIC = 0xF993faca;

void NDArray::Save(dmlc::Stream *strm) const {
  if (Imperative::Get()->is_np_shape()) {
    CHECK_EQ(storage_type(), kDefaultStorage)
        << "only allow serializing ndarray of default storage type in np shape semantics";
    strm->Write(NDARRAY_V3_MAGIC);
  } else {
    // write magic number to mark this version
    // for storage type
    strm->Write(NDARRAY_V2_MAGIC);
  }

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
    nd_cpu = *this;
#if MXNET_USE_MKLDNN == 1
    if (nd_cpu.IsMKLDNNData())
      nd_cpu = nd_cpu.Reorder2Default();
#endif
    save_data = nd_cpu.data();
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

bool LegacyTShapeLoad(dmlc::Stream *strm, mxnet::TShape *shape, const uint32_t magic) {
  switch (magic) {
    case NDARRAY_V1_MAGIC:
      return shape->Load(strm);
    default:
      // meet legacy mxnet::TShape, magic is ndim here
      uint32_t ndim = magic;
      *shape = mxnet::TShape(ndim, -1);
      std::vector<uint32_t> buffer(ndim);
      size_t nread = ndim * sizeof(uint32_t);
      if (strm->Read(buffer.data(), nread) != nread) return false;
      nnvm::ShapeTypeCast(buffer.begin(), buffer.end(), shape->begin());
      return true;
  }
}

bool NDArray::LegacyLoad(dmlc::Stream *strm, const uint32_t magic) {
  // load shape
  mxnet::TShape shape;
  if (!LegacyTShapeLoad(strm, &shape, magic)) return false;
  if (mxnet::op::shape_is_none(shape)) {
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
  if (magic == NDARRAY_V3_MAGIC) {
    CHECK(Imperative::Get()->is_np_shape())
        << "ndarray was saved in np shape semantics, must be loaded in the same semantics."
           " Please turn on np shape semantics in Python using `with np_shape(True)`"
           " or decorator `use_np_shape` to scope the code of loading the ndarray.";
  } else {
    // when the flag is global on, skip the check since it would be always global on.
    CHECK(Imperative::Get()->is_np_shape() == GlobalOn || !Imperative::Get()->is_np_shape())
        << "ndarray was not saved in np shape semantics, but being loaded in np shape semantics."
           " Please turn off np shape semantics in Python using `with np_shape(False)`"
           " to scope the code of loading the ndarray.";
  }
  if (magic != NDARRAY_V2_MAGIC && magic != NDARRAY_V3_MAGIC) {
    return LegacyLoad(strm, magic);
  }

  // load storage type
  int32_t stype;
  if (strm->Read(&stype, sizeof(stype)) != sizeof(stype)) return false;
  if (Imperative::Get()->is_np_shape()) {
    CHECK_EQ(stype, kDefaultStorage)
        << "only allow deserializing ndarray of default storage type in np shape semantics";
  }
  const int32_t nad = num_aux_data(static_cast<NDArrayStorageType>(stype));

  // load storage shape
  mxnet::TShape sshape;
  if (nad > 0) {
    if (!sshape.Load(strm)) return false;
  }

  // load shape
  mxnet::TShape shape;
  if (!shape.Load(strm)) return false;
  if (Imperative::Get()->is_np_shape()) {
    if (!shape_is_known(shape)) {
      *this = NDArray();
      return true;
    }
  } else if (shape.ndim() == 0) {
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
  mxnet::ShapeVector aux_shapes;
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
    int device_count = -1;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
      *this = temp.Copy(ctx); return true;
    } else {
      *this = std::move(temp); return true;
    }
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
    ret = NDArray(shape(), ctx, false, dtype_);
  } else if (kUndefinedStorage != storage_type()) {
    ret = NDArray(storage_type(), shape(), ctx, false, dtype_,
                  ptr_->aux_types, ptr_->aux_shapes, storage_shape());
  } else {
    LOG(FATAL) << "NDArray::Copy cannot copy undefined storage-type ndarray to ctx.dev_type="
               << ctx.dev_type << ", ctx.dev_id=" << ctx.dev_id;
  }
  CopyFromTo(*this, ret);
  return ret;
}

void NDArray::SyncCopyFromCPU(const void *data, size_t size) const {
  mxnet::TShape dshape = this->shape();
  if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
    CHECK_LT(size, (int64_t{1} << 31) - 1) <<
              "[SyncCopyFromCPU] Size of tensor you are trying to allocate is larger than "
              "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
  }
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  // zero-size array, no need to copy
  if (size == 0U) {
    return;
  }
  TBlob src((void*)data, dshape, cpu::kDevMask, this->dtype_, 0); // NOLINT(*)

  if (this->ctx().dev_mask() == cpu::kDevMask) {
    this->WaitToWrite();
    RunContext rctx{this->ctx(), nullptr, nullptr, false};
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
      FnProperty::kCopyToGPU, 0, "SyncCopyCPU2GPU");
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
  auto get_dst_data = [&](const mxnet::TShape& src_shape) {
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
      FnProperty::kNormal, 0, "SyncCopyFromNDArrayCPU2CPU");
  } else {
#if MXNET_USE_CUDA
    if (src_dev_mask == cpu::kDevMask && dst_dev_mask == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0 ? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<cpu, gpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, this->ctx(), const_vars, {this->var()},
        FnProperty::kCopyToGPU, 0, "SyncCopyFromNDArrayCPU2GPU");
    } else if (src_dev_mask == gpu::kDevMask && dst_dev_mask == cpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0 ? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<gpu, cpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, src.ctx(), const_vars, {this->var()},
        FnProperty::kCopyFromGPU, 0, "SyncCopyFromNDArrayGPU2CPU");
    } else if (src_dev_mask == gpu::kDevMask && dst_dev_mask == gpu::kDevMask) {
      Engine::Get()->PushAsync(
        [&](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob src_data = (i >= 0 ? src.aux_data(i) : src.data());
          TBlob dst_data = get_dst_data(src_data.shape_);
          ndarray::Copy<gpu, gpu>(src_data, &dst_data, src.ctx(), this->ctx(), rctx);
          rctx.get_stream<gpu>()->Wait();
          on_complete();
        }, this->ctx(), const_vars, {this->var()},
        src.dtype() != this->dtype() ? FnProperty::kNormal : FnProperty::kCopyFromGPU,
        0, "SyncCopyFromNDArrayGPU2GPU");
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
  mxnet::TShape dshape = this->shape();
  if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
    CHECK_LT(size, (int64_t{1} << 31) - 1) <<
              "[SyncCopyToCPU] Size of tensor you are trying to allocate is larger than "
              "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
  }
  CHECK_EQ(dshape.Size(), size)
      << "Memory size do not match";
  // zero-size array, no need to copy
  if (size == 0U) {
    return;
  }
  TBlob dst(data, dshape, cpu::kDevMask, this->dtype_, 0); // NOLINT(*)

  this->WaitToRead();

  if (this->ctx().dev_mask() == cpu::kDevMask) {
    RunContext rctx{this->ctx(), nullptr, nullptr, false};
    NDArray src = *this;
#if MXNET_USE_MKLDNN == 1
    if (src.IsMKLDNNData())
      src = this->Reorder2Default();
#endif
    ndarray::Copy<cpu, cpu>(src.data(), &dst,
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
      FnProperty::kCopyFromGPU, 0, "SyncCopyGPU2CPU");
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
      FnProperty::kNormal, 0, "CheckFormat");
  } else {
#if MXNET_USE_CUDA
    Engine::Get()->PushSync([&](RunContext rctx) {
        common::CheckFormatWrapper<gpu>(rctx, *this, err_cpu, full_check);
        rctx.get_stream<gpu>()->Wait();
      }, this->ctx(), {this->var()}, {},
      FnProperty::kNormal, 0, "CheckFormat");
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

void NDArray::WaitToRead() const {
  if (is_none()) return;
  Imperative::DCInfo::Compute(*this);
  Engine::Get()->WaitForVar(ptr_->var);
}

void NDArray::WaitToWrite() const {
  if (is_none()) return;
  Imperative::DCInfo::Compute(*this);
  // Push an empty mutable function to flush all preceding reads to the variable.
  Engine::Get()->PushAsync(
      [](RunContext, Engine::CallbackOnComplete on_complete) { on_complete(); },
      Context{}, {}, {ptr_->var});
  Engine::Get()->WaitForVar(ptr_->var);
}

#if MXNET_PREDICT_ONLY == 0
// register API function
// those with underscore will be registered at NDArray
MXNET_REGISTER_NDARRAY_FUN(_set_value)
.set_function(SetValueOp);


MXNET_REGISTER_NDARRAY_FUN(_onehot_encode)
.set_function(BinaryOp<ndarray::OneHotEncode>);


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
  CopyFromTo(inputs[0], outputs[0], 0, true);
}

bool CopyToType(const nnvm::NodeAttrs &attrs, std::vector<int> *in_attrs,
                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  int in_type = in_attrs->at(0);
  if (out_attrs->at(0) == -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_type);
  }
  return out_attrs->at(0) != -1;
}

// copy function is special
// that we need to remove kAcceptEmptyMutateTarget from it
NNVM_REGISTER_OP(_copyto)
.add_alias("_npi_copyto")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", op::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", CopyToType)
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
  CHECK(res.data != nullptr) << "OpenCV Failed to decode image";
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
      for (size_t i = 0; i < y1-y0; i++) {
        uchar* im_data = res.ptr<uchar>(y0+i) + res.channels()*x0;
        for (size_t j = 0; j < x1-x0; j++) {
          for (size_t k = 0; k < n_channels; k++) {
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
      for (size_t i = 0; i < y1-y0; i++) {
        uchar* im_data = res.ptr<uchar>(y0+i) + res.channels()*x0;
        for (size_t j = 0; j < x1-x0; j++) {
          for (size_t k = 0; k < n_channels; k++) {
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
