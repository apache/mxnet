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
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <nnvm/node.h>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <memory>
#include <algorithm>
#if MXNET_USE_MKLDNN == 1
#include <mkldnn.hpp>
#endif
#include "./base.h"
#include "./storage.h"
#include "./engine.h"
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {
// enum for storage types
namespace csr {
enum CSRAuxType {kIndPtr, kIdx};
}

namespace rowsparse {
enum RowSparseAuxType {kIdx};
}

enum NDArrayStorageType {
  kUndefinedStorage = -1,  // undefined storage
  kDefaultStorage,         // dense
  kRowSparseStorage,       // row sparse
  kCSRStorage,             // csr
};

enum NDArrayFormatErr {
  kNormalErr,     // normal
  kCSRShapeErr,   // shape mismatch for csr
  kCSRIndPtrErr,  // indptr error for csr
  kCSRIdxErr,     // idx error for csr
  kRSPShapeErr,   // shape mismatch for row sparse
  kRSPIdxErr,     // indices error for row sparse
};

class MKLDNNMemory;

/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default constructor */
  NDArray()
    : entry_(nullptr) {
  }
  /*!
   * \brief constructs a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const mxnet::TShape &shape, Context ctx,
          bool delay_alloc = false, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(shape, ctx, delay_alloc, dtype)),
        shape_(shape),
        dtype_(dtype),
        storage_type_(kDefaultStorage),
        entry_(nullptr) {
  }
  /*! \brief constructor for NDArray with storage type
   */
  NDArray(const NDArrayStorageType stype, const mxnet::TShape &shape, Context ctx,
          bool delay_alloc = true, int dtype = mshadow::default_type_flag,
          std::vector<int> aux_types = {}, mxnet::ShapeVector aux_shapes = {},
          mxnet::TShape storage_shape = mxnet::TShape(mshadow::Shape1(0)));
  /*!
   * \brief constructs a new dynamic NDArray whose shape is unknown,
   *        hence the NDArray is inherently lazily created
   * \param ctx context of NDArray
   * \param dtype data type of this ndarray
   */
  explicit NDArray(Context ctx, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(mxnet::TShape(mshadow::Shape1(0)), ctx, true, dtype)),
        shape_(),
        dtype_(dtype),
        storage_type_(kDefaultStorage),
        entry_(nullptr) {
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)),
        shape_(data.shape_),
        dtype_(data.type_flag_),
        storage_type_(kDefaultStorage),
        entry_(nullptr) {
  }

  /*!
   * \brief constructing a static NDArray that shares data with TBlob which is with deleter
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   * \param deleter the function pointer of custom deleter
   */
  NDArray(const TBlob &data, int dev_id, const std::function<void()>& deleter)
      : ptr_(new Chunk(data, dev_id), [deleter](Chunk *p) {
            deleter();    // call custom deleter
            delete p;     // delete Chunk object
        }),
        shape_(data.shape_),
        dtype_(data.type_flag_), storage_type_(kDefaultStorage),
        entry_(nullptr) {
  }

  /*! \brief create ndarray from shared memory */
  NDArray(int shared_pid, int shared_id, const mxnet::TShape& shape, int dtype)
      : ptr_(std::make_shared<Chunk>(shared_pid, shared_id, shape, dtype)),
        shape_(shape),
        dtype_(dtype),
        storage_type_(kDefaultStorage),
        entry_(nullptr) {
  }

  /*!
   * \brief constructing a static NDArray of non-default storage that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param stype the storage type of NDArray
   * \param shape the shape of NDArray
   * \param data the memory content of static data
   * \param aux_data the memory content of static aux data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const NDArrayStorageType stype, const mxnet::TShape &shape,
          const TBlob &data, const std::vector<TBlob> &aux_data, int dev_id)
      : ptr_(std::make_shared<Chunk>(stype, data, aux_data, dev_id)),
        shape_(shape),
        dtype_(data.type_flag_),
        storage_type_(stype),
        entry_(nullptr) {
  }
  /*!
   * \brief initialize the NDArray, assuming it is not assigned a meaningful shape before
   * \param shape the shape of the NDArray
   */
  void Init(const mxnet::TShape &shape) {
    ptr_->Init(shape, this->dtype_);
    this->shape_ = shape;
  }
  /*!
   * \brief set the correct shape of NDArray directly from the storage_shape of its own chunk.
   */
  void SetShapeFromChunk();
  /*
   * This indicates whether an array is a view of another array (created by
   * reshape or slice). If an array is a view and the data is stored in
   * MKLDNN format, we need to convert the data to the default format when
   * data in the view is accessed.
   */
  inline bool IsView() const {
    // View only works on the default storage
    if (storage_type() != kDefaultStorage)
      return false;
    // If the array reuses memory, its shape may be different from the storage
    // shape. However, we shouldn't consider it as a view.
    if (reuse_)
      return false;
    return byte_offset_ > 0 || shape() != ptr_->storage_shape;
  }

  /* \brief Check whether the two arrays are the same array */
  inline bool IsSame(const NDArray& other) const {
    return ptr_ == other.ptr_ &&
        shape_ == other.shape_ &&
        byte_offset_ == other.byte_offset_ &&
        dtype_ == other.dtype_;
  }

  /*!
   * \return the shape of current NDArray.
   */
  inline const mxnet::TShape& shape() const {
    return shape_;
  }
  /*!
   * \return the shape of underlying chunk which stores the NDArray data/value.
   *  It is only intended for non-default storage. For row-sparse storage, it is the shape of
   *  the tensor which stores the non-zero values.
   */
  inline const mxnet::TShape &storage_shape() const {
    CHECK(ptr_ != nullptr);
    CHECK_NE(storage_type(), kDefaultStorage)
             << "storage_shape() is not intended for kDefaultStorage.";
    return ptr_->storage_shape;
  }

  /*!
   * \brief get the shape of aux_data(index)
   * \param index the index of the aux data
   * \return the shape of aux data at given index
   */
  inline const mxnet::TShape& aux_shape(size_t index) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "aux_shape() is not intended for kDefaultStorage.";
    return ptr_->aux_shapes[index];
  }

  /* \return the shapes of all aux data */
  const mxnet::ShapeVector& aux_shapes() const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "aux_shapes() is not intended for kDefaultStorage.";
    return ptr_->aux_shapes;
  }

  /*! returns the dtypes of all aux data */
  const std::vector<int>& aux_types() const {
    CHECK_NE(storage_type(), kDefaultStorage)
      << "aux_types() is not intended for kDefaultStorage.";
    return ptr_->aux_types;
  }

  /*!
   * \brief For a sparse operation on a csr matrix for example,
   * the size of the column index array
   * is an estimated value in the beginning for allocating enough capacity
   * for the final result. After the operation is done, the exact size of
   * the shape is known and need to be reset using this function.
   */
  inline void set_aux_shape(size_t index, const mxnet::TShape& shape) const {
    CHECK_NE(storage_type(), kDefaultStorage)
      << "set_aux_shape() is not intended for kDefaultStorage.";
    ptr_->set_aux_shape(index, shape);
  }

  /*!
   * \return the data TBlob
   */
  inline const TBlob& data() const {
    if (storage_type() == kDefaultStorage) CheckAndAlloc();
    SetTBlob();
    return tblob_;
  }
  /*!
   * \return the gradient ndarray.
   */
  NDArray grad() const;

  /*!
   * \return the aux TBlob
   */
  inline TBlob aux_data(size_t i) const {
    auto stype = storage_type();
    TBlob res;
    auto shape = aux_shape(i);
    auto type = aux_type(i);
    MSHADOW_TYPE_SWITCH(type, DType, {
      auto dptr = static_cast<DType*>(ptr_->aux_handles[i].dptr);
      CHECK(stype == kRowSparseStorage || stype == kCSRStorage)
            << "Unexpected storage type: " << stype;
      res = TBlob(dptr, shape, ptr_->aux_handles[i].ctx.dev_mask(), type);
    });
    return res;
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    CHECK(!is_none());
    return ptr_->shandle.ctx;
  }
  /*!
   * \return the data type of NDArray, this function is only valid when the NDArray is not empty
   */
  inline int dtype() const {
    return dtype_;
  }
  inline int aux_type(size_t i) const {
    CHECK(!is_none());
    return ptr_->aux_types[i];
  }

  inline NDArrayStorageType storage_type() const {
    return storage_type_;
  }
  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*! \return updated grad state in entry_ */
  bool fresh_out_grad() const;
  /*! \return updated grad state in entry_ */
  void set_fresh_out_grad(bool state) const;
  /*! \brief Returns true if a sparse ndarray's aux_data and storage are initialized
   * Throws an exception if the indices array shape is inconsistent
   * Returns false if the indices array is empty(nnz = 0) for csr/row_sparse
   */
  inline bool storage_initialized() const {
    if (is_none()) return false;
    auto stype = storage_type();
    CHECK_NE(stype, kDefaultStorage)
             << "storage_initialized() is not intended for kDefaultStorage.";
    if (stype == kRowSparseStorage) {
      CHECK_EQ(aux_shape(rowsparse::kIdx)[0], storage_shape()[0])
               << "inconsistent storage shape " << storage_shape()
               << " vs. aux shape " << aux_shape(rowsparse::kIdx);
      return aux_shape(rowsparse::kIdx).Size() != 0;
    } else if (stype == kCSRStorage) {
      CHECK_EQ(aux_shape(csr::kIdx)[0], storage_shape()[0])
               << "inconsistent storage shape " << storage_shape()
               << " vs. aux shape " << aux_shape(csr::kIdx);
      return aux_shape(csr::kIdx).Size() != 0;
    } else {
      LOG(FATAL) << "Unknown storage type";
    }
    return true;
  }
  /*! \brief get storage handle */
  inline Storage::Handle storage_handle() const {
    CHECK(!is_none());
    CHECK_EQ(storage_type(), kDefaultStorage);
    CheckAndAlloc();
    return ptr_->shandle;
  }
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  inline void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  inline void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushAsync(
      [](RunContext, Engine::CallbackOnComplete on_complete) {
        on_complete();
      }, Context{}, {}, {ptr_->var});
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the ndarray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*! \return byte offset in chunk of the ndarray*/
  inline size_t byte_offset() const {
    return byte_offset_;
  }
  /*! \brief return var version of the NDArray*/
  inline size_t version() const {
    return var()->version();
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load ndarrays before supporting sparse ndarrays
   * \param strm the output stream
   * \param magic the magic number used for version control
   */
  bool LegacyLoad(dmlc::Stream *strm, const uint32_t magic);
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);
  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the size of the source array, in sizeof(DType) not raw btyes.
   */
  void SyncCopyFromCPU(const void *data, size_t size) const;

  /*!
   * \brief Copy from src.data()/aux_data(i) to this->data()/aux_data(j)
   */
  void SyncCopyFromNDArray(const NDArray &src, int i = -1, int j = -1);

  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into, in sizeof(DType) not raw btyes.
   */
  void SyncCopyToCPU(void *data, size_t size) const;
  /*!
  * \brief check whether the NDArray format is valid
  * \param full_check if `True`, rigorous check, O(N) operations
  *    Otherwise basic check, O(1) operations
  */
  void SyncCheckFormat(const bool full_check) const;
  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim (inclusive)
   * \param end end index in first dim (exclusive)
   * \return sliced NDArray
   */
  NDArray Slice(index_t begin, index_t end) const;
  /*!
   * \brief Slice a NDArray. Supports recording with autograd
   * \param begin begin index in first dim (inclusive)
   * \param end end index in first dim (exclusive)
   * \return sliced NDArray
   */
  NDArray SliceWithRecord(index_t begin, index_t end);
  /*!
   * \brief Index a NDArray
   * \param idx the index
   * \return idx-th sub array NDArray
   */
  NDArray At(index_t idx) const;
  /*!
   * \brief Index a NDArray
   * \param idx the index
   * \return idx-th sub array NDArray
   */
  NDArray AtWithRecord(index_t idx);
  /*!
   * \brief Generate a deep copy of aux_data(i) returned as
   * a default storage type NDArray
   */
  NDArray aux_ndarray(size_t i) const;

  /*!
   * \brief Generate a deep copy of data() returned as a
   * default storage type NDArray
   */
  NDArray data_ndarray() const;

  /*!
   * \brief Create a NDArray that shares memory with current one
   *  The new array must have smaller memory size than the current array.
   * \param shape new shape
   * \param dtype The data type.
   * \return NDArray in new shape and type.
   */
  inline NDArray AsArray(const mxnet::TShape &shape, int dtype) const {
    CHECK_EQ(storage_type(), kDefaultStorage)
             << "AsArray is intended only for kDefaultStorage.";
    CHECK_GE(ptr_->shandle.size,
             shape.Size() * mshadow::mshadow_sizeof(dtype))
        << "NDArray.AsArray: target memory size is bigger";
    // We can't reuse memory in a view.
    CHECK(!IsView());
    NDArray ret = *this;
    ret.shape_ = shape;
    ret.dtype_ = dtype;
    ret.reuse_ = true;
    return ret;
  }

  /*!
   * \brief Create a reference view of NDArray that
   *  represents as DLManagedTensor.
   * \return A DLManagedTensor
   */
  DLManagedTensor* ToDLPack() const;

  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   *
   * \return The created NDArray view.
   */
  static NDArray FromDLPack(const DLManagedTensor* tensor, bool transient_handle);

  /*!
   * \brief Update ndarray chunk storage handles using existing ndarray storage handles
   * Also update the aux_handle, aux_shapes and aux_types.
   * This is specifically used for custom op to update the inputs and outputs from
   * the temporary ndarray which stores intermediate custom op results.
   * Should be used with caution elsewhere. Supports only CSR and RSP formats.
   */
  inline void SparseUpdateChunk(const NDArray &arr) const {
    CHECK(shape_ == arr.shape_) << "ndarray shape is different from the target";
    CHECK(dtype_ == arr.dtype_) << "ndarray dtype is different from the target";
    auto stype = arr.storage_type();
    CHECK(stype == kCSRStorage || stype == kRowSparseStorage)
        << "Only to be used with CSR and RSP storage types";
    // swap shandles between src and dst
    Storage::Handle shandle_dst = arr.ptr_->shandle;
    arr.ptr_->shandle = ptr_->shandle;
    ptr_->shandle = shandle_dst;

    ptr_->storage_shape = arr.ptr_->storage_shape;
    ptr_->storage_type = arr.ptr_->storage_type;
    ptr_->ctx = arr.ptr_->ctx;

    // swap aux_handles between src and dst
    size_t aux_idx = 0;
    CHECK(ptr_->aux_handles.size() == arr.ptr_->aux_handles.size())
        << "ndarray number of aux_handles is different from target";
    for (auto &aux_handle : arr.ptr_->aux_handles) {
      Storage::Handle aux_dst = ptr_->aux_handles[aux_idx];
      ptr_->aux_handles[aux_idx] = aux_handle;
      aux_handle = aux_dst;
      aux_idx++;
    }
    ptr_->aux_types = arr.ptr_->aux_types;
    ptr_->aux_shapes = arr.ptr_->aux_shapes;
  }

  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  NDArray Reshape(const mxnet::TShape &shape) const;
  /*!
   * \brief Get an reshaped NDArray. Supports autograd recording
   * \param shape new shape
   * \return NDArray in new shape
   */
  NDArray ReshapeWithRecord(const mxnet::TShape &shape);
  /*!
   * \brief Return a copy of this NDArray without autograd history
   */
  NDArray Detach() const {
    NDArray ret(*this);
    ret.entry_ = nnvm::NodeEntry(nullptr);
    return ret;
  }

  nnvm::Symbol get_autograd_symbol() const;
  /*!
   * \brief Allocate the space if it is delayed allocated.
   * This is an internal function used by system that normal user should not use
   */
  inline void CheckAndAlloc() const {
    CHECK_EQ(storage_type(), kDefaultStorage);
    ptr_->CheckAndAlloc();
  }

  /*!
   * \brief Allocate the space if the allocation has been delayed
   * or the requested size is bigger than the available one.
   * This function can only be called by ndarray of default
   * storage type and effectively changes the ndarray's shape_.
   * Note: This function is named as this to avoid overload conflict
   * with CheckAndAlloc(const mxnet::ShapeVector &aux_shapes), since
   * mxnet::TShape tmp = some_shape is equivalent to mxnet::TShape tmp = {some_shape}.
   */
  void ReshapeAndAlloc(const mxnet::TShape& shape) {
    CHECK_EQ(storage_type(), kDefaultStorage);
    CHECK(!is_none());
    shape_ = shape;
    ptr_->CheckAndAlloc(shape.Size() * mshadow::mshadow_sizeof(dtype_));
  }

  /* !
   * \brief Alloc memory for non-default storage
   * aux_shape is only known at run time
   */
  inline void CheckAndAlloc(const mxnet::ShapeVector &aux_shapes) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAlloc(aux_shapes) is not intended for kDefaultStorage";
    ptr_->CheckAndAlloc(shape_, aux_shapes, dtype_);
  }
  inline void CheckAndAllocData(const mxnet::TShape &storage_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAllocData is not intended for kDefaultStorage";
    ptr_->CheckAndAllocData(storage_shape, dtype_);
  }
  inline void CheckAndAllocAuxData(size_t i, const mxnet::TShape &aux_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAllocAuxData is not intended for kDefaultStorage";
    ptr_->CheckAndAllocAuxData(i, aux_shape);
  }

#if MXNET_USE_MKLDNN == 1
  /*
   * Create NDArray from mkldnn memory.
   * mkldnn_mem The mkldnn memory to be managed.
   */
  explicit NDArray(const std::shared_ptr<mkldnn::memory> &mkldnn_mem);
  /*
   * Create NDArray from mkldnn memory descriptor.
   * mem_pd The mkldnn memory descriptor to be created.
   */
  explicit NDArray(const mkldnn::memory::desc &md);
  /*
   * Test if the data is stored in one of special MKLDNN format.
   */
  bool IsMKLDNNData() const {
    return ptr_->IsMKLDNN();
  }
  /*
   * Test if the data is stored in one of default MXNet formats.
   */
  bool IsDefaultData() const {
    return ptr_->IsDefault();
  }
  /*
   * All functions below return a raw pointer to mkldnn memory. Actually there
   * is a shared pointer that hold the memory either in NDArray or in MKLDNN
   * stream. As long as we call these functions inside an operator, the return
   * memory is always valid.
   */

  /*
   * This function returns mkldnn::memory with the default primitive_desc.
   */
  const mkldnn::memory *GetMKLDNNData() const;
  /*
   * This function returns mkldnn::memory with the given primitive_desc
   * as long as the array size meets the required size in the given primitive_desc.
   */
  const mkldnn::memory *GetMKLDNNData(const mkldnn::memory::desc &md) const;
  /*
   * This function returns mkldnn::memory with the given primitive_desc.
   * The returned mkldnn::memory will have the same physical layout as
   * the given primitive_desc.
   */
  const mkldnn::memory *GetMKLDNNDataReorder(
      const mkldnn::memory::desc &md) const;

  /*
   * This function copies data from mkldnn memory.
   */
  void CopyFrom(const mkldnn::memory &mem);
  /*
   * This function allocates memory for array and creates mkldnn memory
   * with the specified format.
   */
  mkldnn::memory *CreateMKLDNNData(const mkldnn::memory::desc &md);

  /*
   * These are the async version of the methods above.
   * It changes the layout of this NDArray, but it happens after all accesses to
   * the array are complete.
   */
  void Reorder2DefaultAsync() const;
  void MKLDNNDataReorderAsync(const mkldnn::memory::desc &md) const;

  /*
   * This creates a new NDArray with the reordered data.
   * It doesn't affect the data of the original NDArray.
   */
  NDArray Reorder2Default() const;

  void InvalidateMKLDNNData();

  /*
   * This function is used inside operators to reshape an array.
   * It doesn't change the layout of the original array and allocate memory from
   * the temporary buffer. The returned array is only valid inside the current
   * invocation of this operator.
   * This is different from Reshape. Reshape will cause data in the array to be
   * converted to the default layout and allocate memory from malloc directly,
   * which can be expensive.
   * It's used by FullyConnected right now.
   */
  NDArray MKLDNNDataReshape(const mxnet::TShape &shape) const;

   /*!
   * \ Fix mkldnn memory descriptor mismatch from NDArray.
   */
  void UpdateMKLDNNMemDesc(const mkldnn::memory::desc &desc);
#endif

  /*!
   * \brief Save list of ndarray into the Stream.x
   * \param fo The stream of output.
   * \param data the NDArrays to be saved.
   * \param names the name of the NDArray, optional, can be zero length.
   */
  static void Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names);
  /*!
   * \brief Load list of ndarray into from the stream.
   * \param fi The stream of the input file.
   * \param data the NDArrays to be loaded
   * \param keys the name of the NDArray, if saved in the file.
   */
  static void Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys);

 private:
  friend class Imperative;
  /*! \brief the real data chunk that backs NDArray */
  // shandle is used to store the actual values in the NDArray
  // aux_handles store the aux data(such as indices) if it's needed by non-default storage.
  struct Chunk {
    /*! \brief storage handle from storage engine.
               for non-default storage, shandle stores the data(value) array.
     */
    Storage::Handle shandle;
    /*! \brief storage handles for aux data (e.g index)
               for row_sparse, aux_handles[0] = indices
               for csr, aux_handles[0] = indptr, aux_handles[1] = indices
    */
    std::vector<Storage::Handle> aux_handles;

#if MXNET_USE_MKLDNN == 1
    /*! This is created when data is stored in MKLDNN format.
     */
    std::shared_ptr<MKLDNNMemory> mkl_mem_;
#endif
    /*! \brief variable from engine */
    Engine::VarHandle var;
    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    /*! \brief construct from static data */
    bool static_data;
    /*! \brief whether data allocation is delayed. This doesn't indicate whether aux data
               allocation is delayed. */
    bool delay_alloc;
    // the type of the storage. The storage_type is never kUndefinedStorage once the chunk
    // is constructed.
    NDArrayStorageType storage_type = kDefaultStorage;
    /*! \brief type of aux */
    std::vector<int> aux_types;
    // context of data
    Context ctx;
    // The shape of the chunk data.
    // This might not be the same shape as the NDArray, since the storage may be sparse.
    // The default value for storage_shape is {0} when an empty non-default NDArray is created.
    mxnet::TShape storage_shape;
    // The shape of aux data. The default value for the shape depends on the type of storage.
    // If aux_shapes[i].Size() is zero, aux data i is empty.
    mxnet::ShapeVector aux_shapes;
    /*! \brief Reference to the storage to ensure proper destruct order */
    std::shared_ptr<Storage> storage_ref_;
    /*! \brief Reference to the engine to ensure we cleanup without calling a destructed engine */
    std::weak_ptr<Engine> engine_ref_;


    /*! \brief default constructor */
    Chunk() : static_data(true), delay_alloc(false),
              storage_ref_(Storage::_GetSharedRef()),
              engine_ref_(Engine::_GetSharedRef()) {}

    /*! \brief construct a new chunk */
    Chunk(mxnet::TShape shape, Context ctx_, bool delay_alloc_, int dtype)
        : static_data(false), delay_alloc(true), ctx(ctx_),
          storage_ref_(Storage::_GetSharedRef()),
          engine_ref_(Engine::_GetSharedRef()) {
      storage_shape = shape;
      if (shape_is_known(storage_shape)) {
        shandle.size = shape.Size() * mshadow::mshadow_sizeof(dtype);
      }
      var = Engine::Get()->NewVariable();
      shandle.ctx = ctx_;
      if (!delay_alloc_) {
        this->CheckAndAlloc();
      }
    }

    Chunk(const TBlob &data, int dev_id)
        : static_data(true), delay_alloc(false),
          storage_ref_(Storage::_GetSharedRef()),
          engine_ref_(Engine::_GetSharedRef()) {
      CHECK(storage_type == kDefaultStorage);
      var = Engine::Get()->NewVariable();
      if (data.dev_mask() == cpu::kDevMask) {
        ctx = Context::CPU();
      } else {
        CHECK_EQ(data.dev_mask(), gpu::kDevMask);
        ctx = Context::GPU(dev_id);
      }
      // init shandle
      shandle.ctx = ctx;
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * mshadow::mshadow_sizeof(data.type_flag_);
      storage_shape = data.shape_;
    }

    Chunk(int shared_pid, int shared_id, const mxnet::TShape& shape, int dtype)
        : static_data(false), delay_alloc(false),
          storage_ref_(Storage::_GetSharedRef()),
          engine_ref_(Engine::_GetSharedRef()) {
      var = Engine::Get()->NewVariable();
      ctx = Context::CPUShared(0);
      shandle.size = shape.Size() * mshadow::mshadow_sizeof(dtype);
      shandle.ctx = ctx;
      shandle.shared_pid = shared_pid;
      shandle.shared_id = shared_id;
      Storage::Get()->Alloc(&shandle);
      storage_shape = shape;
    }
    // Constructor for a non-default storage chunk
    Chunk(NDArrayStorageType storage_type_, const mxnet::TShape &storage_shape_, Context ctx_,
          bool delay_alloc_, int dtype, const std::vector<int> &aux_types_,
          const mxnet::ShapeVector &aux_shapes_)
        : static_data(false), delay_alloc(delay_alloc_), storage_type(storage_type_),
          aux_types(aux_types_), ctx(ctx_), storage_shape(storage_shape_),
          aux_shapes(aux_shapes_), storage_ref_(Storage::_GetSharedRef()),
          engine_ref_(Engine::_GetSharedRef()) {
      shandle.ctx = ctx;
      var = Engine::Get()->NewVariable();
      // aux_handles always reflect the correct number of aux data
      for (size_t i = 0; i < aux_shapes.size(); i++) {
        CheckAndAllocAuxData(i, aux_shapes[i]);
        // this line is needed in case when aux_shapes[i].Size() = 0
        // aux_handles[i] will not be updated and take only default value.
        aux_handles[i].ctx = ctx;
      }
      if (!delay_alloc) {
        CheckAndAllocData(storage_shape, dtype);
      }
    }

    Chunk(const NDArrayStorageType storage_type_, const TBlob &data,
          const std::vector<TBlob> &aux_data, int dev_id)
        : static_data(true), delay_alloc(false), storage_type(storage_type_),
          storage_ref_(Storage::_GetSharedRef()), engine_ref_(Engine::_GetSharedRef()) {
      using namespace mshadow;
      CHECK_NE(storage_type, kDefaultStorage);
      // init var
      var = Engine::Get()->NewVariable();
      // init ctx
      if (data.dev_mask() == cpu::kDevMask) {
        ctx = Context::CPU();
      } else {
        CHECK_EQ(data.dev_mask(), gpu::kDevMask);
        ctx = Context::GPU(dev_id);
      }
      // init shandle
      shandle.ctx = ctx;
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * mshadow_sizeof(data.type_flag_);
      storage_shape = data.shape_;
      // init aux handles
      for (const auto &aux : aux_data) {
        Storage::Handle aux_handle;
        aux_handle.ctx = ctx;
        aux_handle.dptr = aux.dptr_;
        aux_handle.size = aux.shape_.Size() * mshadow_sizeof(aux.type_flag_);
        aux_handles.push_back(aux_handle);
        aux_types.emplace_back(aux.type_flag_);
        aux_shapes.emplace_back(aux.shape_);
      }
    }

    /*! \brief set the shape for ith aux data, and update storage shape if necessary */
    inline void set_aux_shape(const size_t i, const mxnet::TShape& shape) {
      aux_shapes[i] = shape;
      if (storage_shape.ndim() >= 0) {
        if (storage_type == kRowSparseStorage && i == rowsparse::kIdx) {
          storage_shape[0] = shape[0];
        } else if (storage_type == kCSRStorage && i == csr::kIdx) {
          storage_shape[0] = shape[0];
        }
      }
    }

    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(shandle.size, shandle.ctx);
#if MXNET_USE_MKLDNN == 1
        mkl_mem_ = nullptr;
#endif
        delay_alloc = false;
      }
    }

    /*! \brief Check and alloc memory for a dense ndarray */
    // size is the number of bytes
    void CheckAndAlloc(uint64_t dbytes) {
      CHECK_EQ(kDefaultStorage, storage_type)
          << "CheckAndAlloc(dbytes) is only intended for kDefaultStorage";
      dbytes = std::max(dbytes, static_cast<uint64_t>(shandle.size));
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(dbytes, shandle.ctx);
#if MXNET_USE_MKLDNN == 1
        mkl_mem_ = nullptr;
#endif
        delay_alloc = false;
      } else if (shandle.size < dbytes) {
        // free storage
        Storage::Get()->Free(shandle);
        // init storage
        shandle = Storage::Get()->Alloc(dbytes, shandle.ctx);
#if MXNET_USE_MKLDNN == 1
        mkl_mem_ = nullptr;
#endif
      }
    }
    /*! \brief initialize the shape and dtype, assuming it is not initialized before. */
    void Init(const mxnet::TShape &shape, int dtype) {
      auto size = shape.Size();
      storage_shape = shape;
      shandle.size = size * mshadow::mshadow_sizeof(dtype);
      this->CheckAndAlloc();
    }
    inline void CheckAndAlloc(const mxnet::TShape &shape, const mxnet::ShapeVector &aux_shapes,
                              int dtype) {
      // calculate size, perform allocation
      if (kRowSparseStorage == storage_type) {
        // For row sparse, aux_shape indicates the number of rows to allocate
        auto aux_shape = aux_shapes[rowsparse::kIdx];
        CheckAndAllocAuxData(rowsparse::kIdx, aux_shape);
        mxnet::TShape storage_shape(shape);
        storage_shape[0] = aux_shape[0];
        CheckAndAllocData(storage_shape, dtype);
      } else if (kCSRStorage == storage_type) {
        CheckAndAllocAuxData(csr::kIndPtr, aux_shapes[csr::kIndPtr]);
        CheckAndAllocAuxData(csr::kIdx, aux_shapes[csr::kIdx]);
        CheckAndAllocData(aux_shapes[csr::kIdx], dtype);
      } else {
        LOG(FATAL) << "Storage type " << storage_type << " not implemented for CheckAndAlloc";
      }
    }
    // create storage handle for data based on shape and dtype, assuming ctx is set
    // storage shape is also updated
    // if data is already allocated, try reuse the storage. Otherwise, free the current one
    // and allocate new storage
    void CheckAndAllocData(const mxnet::TShape &shape, int dtype);

#if MXNET_USE_MKLDNN == 1
    // Have MKL memory reference to the data in the default storage
    // or create memory for MKLDNN.
    void SetMKLMem(const mxnet::TShape &shape, int dtype);
    // If the data is stored in MKLDNN layout, we reorder data in mkl_mem_ and
    // save the result in shandle.
    void Reorder2Default();
    // Reroder data to a specified layout.
    void MKLDNNDataReorder(const mkldnn::memory::desc &md);
    bool IsMKLDNN() const;
    bool IsDefault() const;
#endif

    // create storage handle for aux data based on shape
    // this function assumes ctx, aux shapes and aux types are set
    // aux shape is also updated
    // if aux data is already allocated, try reuse the storage. Otherwise, free the current one
    // and allocate new storage
    inline void CheckAndAllocAuxData(size_t i, const mxnet::TShape &shape) {
      CHECK_EQ(shape.ndim(), 1) << "shape must be 1D in CheckAndAllocAuxData";
      CHECK_NE(storage_type, kUndefinedStorage)
        << "storage type cannot be kUndefinedStorage in CheckAndAllocAuxData";
      CHECK_NE(storage_type, kDefaultStorage)
        << "storage type cannot be kDefaultStorage in CheckAndAllocAuxData";
      if (aux_handles.size() <= i) {
        aux_handles.resize(i + 1);
      }
      size_t aux_bytes = shape.Size() * mshadow::mshadow_sizeof(aux_types[i]);
      if (aux_handles[i].size < aux_bytes) {
        // free storage
        Storage::Get()->Free(aux_handles[i]);
        // init aux storage
        aux_handles[i] = Storage::Get()->Alloc(aux_bytes, ctx);
      }
      // init shape
      set_aux_shape(i, shape);
    }
    /*! \brief destructor */
    ~Chunk();
  };  // struct Chunk

  void SetTBlob() const;

  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_{nullptr};
  /*! \brief shape of current NDArray */
  mxnet::TShape shape_;
  /*! \brief byte offset in chunk */
  size_t byte_offset_ = 0;
  /*! \brief type of data */
  int dtype_ = -1;
  /*! \brief whether the NDArray uses memory of another NDArray. */
  bool reuse_ = false;
  /*! \brief storage type of data */
  NDArrayStorageType storage_type_ = kUndefinedStorage;
  /*! \brief node entry for autograd */
  nnvm::NodeEntry entry_;
  /*!
   * \brief internal TBlob
   * \note When user access tblob_ by some const methods like
   *     NDArray::data(), the dptr in tblob_ still need to be updated
   *     in case that allocation happens. So we make it mutable for
   *     this situation.
   */
  mutable TBlob tblob_;
};  // class NDArray

/*!
 * \return the number of aux data used for given storage type
 */
size_t num_aux_data(NDArrayStorageType stype);

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, const NDArray *to, int priority = 0);

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \param is_opr whether it is invoked by an operator. For example, false if invoked from
       KVStore, true if invoked from `_copyto` operator.
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, const NDArray& to, int priority = 0, bool is_opr = false);

/*!
 * \brief Perform elementwise sum over each data from source, store result into out.
 * \param source the ndarray we want to sum
 * \param out the target ndarray
 * \param priority Priority of the action.
 */
void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority = 0);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const NDArray &rhs); \
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const real_t &rhs);

/*!
 * \brief Seed all random number generator in mxnet.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
/*!
 * \brief Seed the random number generator of the device.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(Context ctx, uint32_t seed);
/*!
 * \brief Sample uniform distribution for each elements of out.
 * \param begin lower bound of distribution.
 * \param end upper bound of distribution.
 * \param out output NDArray.
 */
void SampleUniform(real_t begin, real_t end, NDArray *out);
/*!
 * \brief Sample gaussian distribution for each elements of out.
 * \param mu mean of gaussian distribution.
 * \param sigma standard deviation of gaussian distribution.
 * \param out output NDArray.
 */
void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
/*!
 * \brief Sample gamma distribution for each elements of out.
 * \param alpha parameter (shape) of the gamma distribution
 * \param beta parameter (scale) of the gamma distribution
 * \param out output NDArray.
 */
void SampleGamma(real_t alpha, real_t beta, NDArray *out);
/*!
 * \brief Sample exponential distribution for each elements of out.
 * \param lambda parameter (rate) of the exponential distribution
 * \param out output NDArray.
 */
void SampleExponential(real_t lambda, NDArray *out);
/*!
 * \brief Sample Poisson distribution for each elements of out.
 * \param lambda parameter (rate) of the Poisson distribution
 * \param out output NDArray.
 */
void SamplePoisson(real_t lambda, NDArray *out);
/*!
 * \brief Sample negative binomial distribution for each elements of out.
 * \param k failure limit
 * \param p success probability
 * \param out output NDArray.
 */
void SampleNegBinomial(int32_t k, real_t p, NDArray *out);
/*!
 * \brief Sample generalized negative binomial distribution for each elements of out.
 * \param mu parameter (mean) of the distribution
 * \param alpha parameter (over dispersion) of the distribution
 * \param out output NDArray.
 */
void SampleGenNegBinomial(real_t mu, real_t alpha, NDArray *out);


//--------------------------------------------------------------
// The following part are API Registration of NDArray functions.
//--------------------------------------------------------------

/*! \brief definition of NDArray function */
typedef std::function<void (NDArray **used_vars,
                            real_t *scalars,
                            NDArray **mutate_vars,
                            int num_params,
                            char **param_keys,
                            char **param_vals)> NDArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NDArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNDArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNDArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NDArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NDArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NDArrayFunction */
struct NDArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NDArrayFunctionReg,
                                        NDArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NDArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NDArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fsetvalue)(const real_t &rhs,
                                                            NDArray *out)) {
    body = [fsetvalue] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                        int num_params, char **param_keys, char **param_vals) {
      (*fsetvalue)(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    this->add_argument("src", "real_t", "Source input to the function.");
    return *this;
  }
  /*!
  * \brief set the function body to a ternary NDArray function
  *  this will also auto set the parameters correctly
  * \param fternary function body to set
  * \return ref to the registered entry, used to set properties
  */
  inline NDArrayFunctionReg &set_function(void(*fternary)(const NDArray &lhs,
                                                          const NDArray &mhs,
                                                          const NDArray &rhs,
                                                                NDArray *out)) {
    body = [fternary](NDArray **used_vars,
      real_t *s, NDArray **mutate_vars,
      int num_params, char **param_keys, char **param_vals) {
      (*fternary)(*used_vars[0], *used_vars[1], *used_vars[2], mutate_vars[0]);
    };
    num_use_vars = 3; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("mhs", "NDArray", "Middle operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fbinary)(const NDArray &lhs,
                                                          const NDArray &rhs,
                                                          NDArray *out)) {
    body = [fbinary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fbinary)(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fscalar)(const NDArray &lhs,
                                                          const real_t &rhs,
                                                          NDArray *out)) {
    body = [fscalar] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fscalar)(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*funary)(const NDArray &src,
                                                         NDArray *out)) {
    body = [funary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                     int num_params, char **param_keys, char **param_vals) {
      (*funary)(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NDArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param fgeneric function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(
    void (*fgeneric)(NDArray **used_vars,
                     real_t *s,
                     NDArray **mutate_vars,
                     const std::map<std::string, std::string>& param)) {
    body = [fgeneric] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                       int num_params, char **param_keys, char **param_vals) {
      std::map<std::string, std::string> param;
      for (int i = 0; i < num_params; ++i) {
        param[param_keys[i]] = param_vals[i];
      }
      fgeneric(used_vars, s, mutate_vars, param);
    };
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NDArrayFunctionReg

/*!
 * \brief Macro to register NDArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NDARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NDARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NDArrayFunctionReg, NDArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NDArray, true);
}  // namespace dmlc
#endif  // MXNET_NDARRAY_H_
