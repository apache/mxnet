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
#include "./base.h"
#include "./storage.h"
#include "./engine.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#endif
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


/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default constructor */
  NDArray() {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = MKLMemHolder::create();
#endif
  }
  /*!
   * \brief constructs a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const TShape &shape, Context ctx,
          bool delay_alloc = false, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(shape, ctx, delay_alloc, dtype)),
        shape_(shape), dtype_(dtype), storage_type_(kDefaultStorage),
        entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }
  /*! \brief constructor for NDArray with storage type
   */
  NDArray(const NDArrayStorageType stype, const TShape &shape, Context ctx,
          bool delay_alloc = true, int dtype = mshadow::default_type_flag,
          std::vector<int> aux_types = {}, std::vector<TShape> aux_shapes = {},
          TShape storage_shape = TShape(mshadow::Shape1(0)))
      : shape_(shape), dtype_(dtype), storage_type_(stype),
        entry_({nullptr, 0, 0}) {
      // Assign default aux types if not given
      if (aux_types.size() == 0) {
        if (stype == kRowSparseStorage) {
          aux_types = {mshadow::kInt64};
        } else if (stype == kCSRStorage) {
          aux_types = {mshadow::kInt64, mshadow::kInt64};
        } else {
          LOG(FATAL) << "Unknown storage type " << stype;
        }
      }
      // Assign default shapes if not given
      // unknown shapes are intialized as {0} such that Size() would return 0
      if (aux_shapes.size() == 0) {
        if (stype == kRowSparseStorage) {
          aux_shapes = {TShape(mshadow::Shape1(0))};
        } else if (stype == kCSRStorage) {
          // aux shapes for indptr and indices
          aux_shapes = {TShape(mshadow::Shape1(0)), TShape(mshadow::Shape1(0))};
        } else {
          LOG(FATAL) << "Unknown storage type " << stype;
        }
      }
      if (storage_shape.Size() == 0) {
        if (stype == kRowSparseStorage) {
          storage_shape = shape;
          storage_shape[0] = aux_shapes[rowsparse::kIdx][0];
        } else if (stype == kCSRStorage) {
          storage_shape = aux_shapes[csr::kIdx];
        } else {
          LOG(FATAL) << "Unknown storage type " << stype;
        }
      }
      ptr_ = std::make_shared<Chunk>(stype, storage_shape, ctx, delay_alloc,
                                     dtype, aux_types, aux_shapes);
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_),
        dtype_(data.type_flag_), storage_type_(kDefaultStorage),
        entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }
  /*! \brief Create from shared memory
   * @param key Key to shared memory
   * @param shape The expected shape
   * @param dtype The expected type
   * */
  NDArray(const char* key, const TShape& shape, int dtype)
      : ptr_(std::make_shared<Chunk>(key, shape, dtype)), shape_(shape),
        dtype_(dtype), storage_type_(kDefaultStorage), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
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
  NDArray(const NDArrayStorageType stype, const TShape &shape,
          const TBlob &data, const std::vector<TBlob> &aux_data, int dev_id)
      : ptr_(std::make_shared<Chunk>(stype, data, aux_data, dev_id)), shape_(shape),
        dtype_(data.type_flag_), storage_type_(stype), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }

  /*!
   * \return The shape of current NDArray.
   */
  const TShape& shape() const {
    return shape_;
  }

  /*!
   * \return the shape of underlying chunk which stores the NDArray data/value.
   *  It is only intended for non-default storage. For row-sparse storage, it is the shape of
   *  the tensor which stores the non-zero values.
   */
  const TShape& storage_shape() const {
    CHECK(ptr_ != nullptr);
    CHECK_NE(storage_type(), kDefaultStorage)
      << "storage_shape() is not intended for kDefaultStorage.";
    return ptr_->storage.shape;
  }

  /*!
   * \brief get the shape of aux_data(index)
   * \param index the index of the aux data
   * \return the shape of aux data at given index
   */
  inline const TShape& aux_shape(size_t index) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "aux_shape() is not intended for kDefaultStorage.";
    return ptr_->aux_shapes[index];
  }

  /* \return the shapes of all aux data */
  const std::vector<TShape>& aux_shapes() const {
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
  inline void set_aux_shape(size_t index, const TShape& shape) const {
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
      auto dptr = static_cast<DType*>(ptr_->aux_handles[i]->dptr);
      CHECK(stype == kRowSparseStorage || stype == kCSRStorage)
            << "Unexpected storage type: " << stype;
      res = TBlob(dptr, shape, ptr_->aux_handles[i]->ctx.dev_mask(), type);
    });
#if MKL_EXPERIMENTAL == 1
    res.Mkl_mem_ = Mkl_mem_;
#endif
    return res;
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    CHECK(!is_none());
    return ptr_->storage.context;
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
      return aux_shape(0).Size() != 0;
    } else if (stype == kCSRStorage) {
      CHECK_EQ(aux_shape(csr::kIdx)[0], storage_shape()[0])
               << "inconsistent storage shape " << storage_shape()
               << " vs. aux shape " << aux_shape(csr::kIdx);
      return aux_shape(0).Size() != 0;
    } else {
      LOG(FATAL) << "Unknown storage type";
    }
    return true;
  }

  /*! \brief get storage handle */
  std::shared_ptr<storage::Handle> storage_handle() const {
    CHECK(!is_none());
    CHECK_EQ(storage_type(), kDefaultStorage);
    CheckAndAlloc();
    return ptr_->storage.handle;
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
  inline NDArray AsArray(const TShape &shape, int dtype) const {
    CHECK_EQ(storage_type(), kDefaultStorage)
             << "AsArray is intended only for kDefaultStorage.";
    CHECK_GE(ptr_->storage.size,
             shape.Size() * mshadow::mshadow_sizeof(dtype))
        << "NDArray.AsArray: target memory size is bigger";
#if MKL_EXPERIMENTAL == 1
    if (Mkl_mem_ != nullptr) {
      // convert prv to cpu
      Mkl_mem_->check_and_prv_to_cpu(ptr_->storage.handle->dptr);
    }
#endif
    NDArray ret = *this;
    ret.shape_ = shape;
    ret.dtype_ = dtype;
    return ret;
  }
  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  NDArray Reshape(const TShape &shape) const;
  /*!
   * \brief Get an reshaped NDArray. Supports autograd recording
   * \param shape new shape
   * \return NDArray in new shape
   */
  NDArray ReshapeWithRecord(const TShape &shape);
  /*!
   * \brief Return a copy of this NDArray without autograd history
   */
  NDArray Detach() const {
    NDArray ret(*this);
    ret.entry_ = nnvm::NodeEntry{nullptr, 0, 0};
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
   * with CheckAndAlloc(const std::vector<TShape> &aux_shapes), since
   * TShape tmp = some_shape is equivalent to TShape tmp = {some_shape}.
   */
  void ReshapeAndAlloc(const TShape& shape) {
    CHECK_EQ(storage_type(), kDefaultStorage);
    CHECK(!is_none());
    shape_ = shape;
    ptr_->CheckAndAlloc(shape.Size() * mshadow::mshadow_sizeof(dtype_));
  }

  /* !
   * \brief Alloc memory for non-default storage
   * aux_shape is only known at run time
   */
  inline void CheckAndAlloc(const std::vector<TShape> &aux_shapes) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAlloc(aux_shapes) is not intended for kDefaultStorage";
    ptr_->CheckAndAlloc(shape_, aux_shapes, dtype_);
  }
  inline void CheckAndAllocData(const TShape &storage_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAllocData is not intended for kDefaultStorage";
    ptr_->CheckAndAllocData(storage_shape, dtype_);
  }
  inline void CheckAndAllocAuxData(size_t i, const TShape &aux_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage)
             << "CheckAndAllocAuxData is not intended for kDefaultStorage";
    ptr_->CheckAndAllocAuxData(i, aux_shape);
  }
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

  /*! \brief The real data chunk that backs NDArray
   *
   * storage is used to store the actual values in the NDArray.
   * aux_handles store the aux data(such as indices) if it's needed by non-default storage.
   */
  struct Chunk {
    /*! \brief Storage info.
     *
     *  The storage handle creation can be deferred.
     */
    struct {
      /*! \brief Storage size in bytes. */
      std::size_t size { 0 };

      /*! \brief Context of data for storage allocation. */
      Context context;

      /*! \brief The shape of the data chunk.
       *
       * This might not be the same shape as the NDArray, since the storage may be sparse.
       * The default value {0} when an empty non-default NDArray is created.
       */
      TShape shape;

      /*! \brief Type of the storage.
       * Is never kUndefinedStorage.
       * */
      NDArrayStorageType type { kDefaultStorage };

      /*! \brief Handle of the storage holding the data. */
      std::shared_ptr<storage::Handle> handle;
    } storage;

    /*! \brief Storage handles for auxiliary data (e.g index)
     * For row_sparse, aux[0] = indices
     * For csr, aux[0] = indptr, aux[1] = indices */
    std::vector<std::shared_ptr<storage::Handle>> aux_handles;

    /*! \brief type of aux */
    std::vector<int> aux_types;

    /*! \brief The shapes of aux data.
     * The default shape depends on the type of storage.
     * If aux_shapes[i].size() is zero, auxiliary data i is empty. */
    std::vector<TShape> aux_shapes;

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

    /*! \brief Default construct.
     *
     * Does not allocate memory for data at construction and does not free it upon destruction.
     */
    Chunk();

    /*! \brief Construct with parameters.
     *
     * @param shape The shape of the data
     * @param context The context for data allocation
     * @param delay_allocation Option for delaying allocation
     * @param dtype Type of the data
     */
    Chunk(TShape shape, Context context, bool delay_allocation, int dtype);

    /*! \brief Construct from TBlob.
     *
     * Does not allocate memory for data at construction and does not free it upon destruction.
     *
     * @param data The blob data
     * @param dev_id Device id for GPU context
     */
    Chunk(const TBlob& data, int dev_id);

    /*! \brief Construct from shared memory.
     *
     * Does not allocate memory for data at construction and does not free it upon destruction.
     * Assumes that CPUSharedStorageManager is used.
     *
     * @param key The key to reference the shared memory
     * @param shape The shape of the data
     * @param dtype The type of the data
     */
    Chunk(const char* key, const TShape& shape, int dtype);

    /*! \brief Construct with non-default storage and with auxiliary data.
     *
     * @param storage_type The type of the storage
     * @param shape The shape of the data
     * @param context The context for data allocation
     * @param delay_allocation Option for delaying allocation
     * @param dtype Type of the auxiliary data
     * @param aux_types The auxiliary types
     * @param aux_shapes The auxiliary shapes
     */
    Chunk(NDArrayStorageType storage_type,
          const TShape& shape,
          Context context,
          bool delay_allocation,
          int dtype,
          const std::vector<int>& aux_types,
          const std::vector<TShape>& aux_shapes);

    /*! \brief Construct from TBlob, with non-default storage and with auxiliary data.
     *
     * Does not allocate memory for data at construction and does not free it upon destruction.
     *
     * @param type The type of the storage
     * @param data The blob data
     * @param aux_data The auxiliary data
     * @param dev_id Device id for GPU context
     */
    Chunk(NDArrayStorageType type,
          const TBlob& data,
          const std::vector<TBlob>& aux_data,
          int dev_id);

    /*! \brief set the shape for ith aux data, and update storage shape if necessary */
    void set_aux_shape(std::size_t index, const TShape& shape);

    /*! \brief Check if delay_alloc is on and do allocation
     *
     * Performs a check whether the storage was already allocated. Assumes that
     * the size, the context, the shape and the storage type were already set.
     */
    void CheckAndAlloc();

    /*! \brief Check and alloc memory for a dense ndarray
     *
     * Assumes that the size, the context, the shape and the storage type were already set.
     *
     * @param size The size in bytes
     */
    void CheckAndAlloc(std::size_t size);

    /*! \brief Check and alloc memory for auxiliary shapes given their shape and type
     *
     * Assumes that the size, the context, the shape and the storage type were already set.

     * @param shape The shape
     * @param aux_shapes The auxiliary shapes
     * @param dtype Type of the data
     */
    void CheckAndAlloc(const TShape& shape,
                       const std::vector<TShape>& aux_shapes,
                       int dtype);

    /*! \brief Check and allocate memory given shape and type
     *
     * Assumes that the size, the context, the storage type and the auxiliry data were already set.
     * Tries to reuse the storage if requested size is smaller or equal the already allocated size.
     *
     * @param shape The shape
     * @param dtype Type of the data
     */
    void CheckAndAllocData(const TShape& shape, int dtype);

    /*! \brief Check and allocate memory for auxiliary data given size and shape
     *
     * Assumes that the context, the storage type and the auxiliry types were already set.
     * Tries to reuse the storage if data was already allocated.
     *
     * @param shape The shape
     * @param dtype Type of the data
     */
    void CheckAndAllocAuxData(std::size_t index, const TShape& shape);

    ~Chunk();
  };  // struct Chunk

  void SetTBlob() const {
    CHECK(ptr_ != nullptr);
    TShape shape = shape_;
    char *dptr = static_cast<char*>(ptr_->storage.handle->dptr);
    auto stype = storage_type();
    if (stype == kDefaultStorage) {
      dptr += byte_offset_;
    } else if (stype == kCSRStorage || stype == kRowSparseStorage) {
      shape = storage_shape();
    } else {
      LOG(FATAL) << "unknown storage type " << stype;
    }
    tblob_.dptr_ = dptr;
    tblob_.shape_ = shape;
    tblob_.type_flag_ = dtype_;
    tblob_.SetDLTensor(ptr_->storage.handle->ctx.dev_mask(), ptr_->storage.handle->ctx.dev_id);
#if MKL_EXPERIMENTAL == 1
    tblob_.Mkl_mem_ = Mkl_mem_;
#endif
  }

#if MKL_EXPERIMENTAL == 1
  std::shared_ptr<MKLMemHolder> Mkl_mem_;
#endif
  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_{nullptr};
  /*! \brief shape of current NDArray */
  TShape shape_;
  /*! \brief byte offset in chunk */
  size_t byte_offset_ = 0;
  /*! \brief type of data */
  int dtype_ = -1;
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
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, const NDArray& to, int priority = 0);

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
 * \brief Seed the random number generator.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
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
