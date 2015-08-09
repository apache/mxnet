/*!
 * Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief the memory allocator that manages the memory across multiple devices
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {

/*! \brief memory allocator of storage */
class StorageManager {
 public:
  /*!
   * \brief storage handle the represents storage information
   */
  struct Handle {
    /*! \brief pointer to the data */
    void* dptr;
    /*! \brief context information about device and deviceID */
    Context ctx;
  };
  /*!
   * \brief allocate a new contiguous memory for a given size
   * \param size the total size of memory in bytes
   * \param ctx context information about the device and deviceID
   * \return Handle struct
   */
  Handle Alloc(size_t size, Context ctx);
  /*!
   * \brief free the space represened the handle
   * \param handle the handle to memory to be freed
   */
  void Free(Handle handle);
  /*! \return storage manager singleton */
  static StorageManager* Get();

 private:
  /*!
   * \brief disabled constructors
   */
  StorageManager() {}
  DISALLOW_COPY_AND_ASSIGN(StorageManager);
};  // class StorageManager

}  // namespace mxnet

#endif  // MXNET_STORAGE_H_
