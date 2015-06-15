/*!
 *  Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief the memory allocator that manages the memory across multiple devices
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_
#include "./base.h"

namespace mxnet {
/*! \brief memory allocator of storage */
class StorageManager {
public:
  /*!
   * \brief storage handle the represents storage information
   */
  struct Handle {
    /*! \brief pointer to the data */
    void *dptr;
    /*! \brief device mask of the memory */
    int dev_mask;
    /*! \brief device id of the memory */
    int dev_id;
    /*!
     * \brief internal handle reserved for manager,
     *   user should not change or use this
     */
    void *handle_;        
  };  
  /*!
   * \brief allocate a new contiguous memory for a given size
   * \param size the total size of memory in bytes
   * \param dev_mask the device mask of the target device, can be cpu::kDevMask or gpu::kDevMask
   * \param dev_id the device ID of the target device
   */
  virtual Handle Alloc(size_t size, int dev_mask, int dev_id);
  /*!
   * \brief free the space represened the handle
   * \param handle the handle to memory to be freed
   */
  virtual void Free(Handle handle);

 protected:
  
};
}
#endif  // MXNET_STORAGE_H_
