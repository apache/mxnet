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
 * Copyright (c) 2015 by Contributors
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_

#include <memory>
#include <string>
#include "./base.h"

namespace mxnet {

namespace {
/// \brief Given a path, extract the filename.
inline std::string __extract_fname(const std::string& path) {
  std::size_t last_dir_pos = path.find_last_of("/\\");
  if (last_dir_pos == std::string::npos) {
    return path;
  }
  return path.substr(last_dir_pos + 1);
}
}  // anonymous namespace

#if __GNUG__  // if compiled with GCC
#define MXNET_STORAGE_DEFAULT_NAME_FARG(tag) \
    std::string(tag) \
    + "_" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__) \
    + "_" + __extract_fname(__builtin_FILE()) \
    + "+" +  std::to_string(__builtin_LINE())
#else  // !__GNUG__
#define MXNET_STORAGE_DEFAULT_NAME_FARG(tag) \
    std::string(tag) \
    + "_" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__)
#endif  // __GNUG__
#define MXNET_STORAGE_DEFAULT_PROFILER_SCOPE_CSTR  "<unk>:"

/*!
 * \brief Storage manager across multiple devices.
 */
class Storage {
 public:
  enum class DataStruct {
      kDataEntry,        ///< Data Entries (!Important)
      kTempSpace,        ///< Temporary Workspace
      kParameter,        ///< Weight Parameter
      kParameterGrad,    ///< Weight Parameter Gradient
      kOptimizerState,   ///< Optimizer State (e.g., Adam Mean & Var)
      kAuxState,         ///< Auxiliary State
      kEphemeral,        ///< Ephemeral Allocations (i.e., Allocations that are
                         ///  short-lived and expected to be deallocated soon)
      kUnknown};
  /*!
   * \brief Storage handle.
   */
  struct Handle {
    /*!
     * \brief Pointer to the data.
     */
    void* dptr{nullptr};
    /*!
     * \brief Size of the storage.
     */
    size_t size{0};
    /*!
     * \brief Context information about device and ID.
     */
    Context ctx;
    /*!
     * \brief Id for IPC shared memory
     */
    int shared_pid{-1};
    int shared_id {-1};
    /*!
     * \brief Attribute Name & Scope for tracking storage allocations.
     */
    std::string profiler_scope{"<unk>:"};
    std::string name{"unknown"};
    /*!
     * \brief Data Structure categorizes storage allocations
     *          based on their functionality.
     *        It is also used for tracking storage allocations.
     */
    DataStruct data_struct;
  };
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param size Total size of memory in bytes.
   * \param ctx Context information about the device and ID.
   * \param profiler scope, name, data_struct 
   * \return Handle struct.
   */
  Handle Alloc(size_t size, Context ctx,
      const std::string& profiler_scope =
        MXNET_STORAGE_DEFAULT_PROFILER_SCOPE_CSTR,
      const std::string& name =
        MXNET_STORAGE_DEFAULT_NAME_FARG("unknown"),
      const DataStruct data_struct = DataStruct::kUnknown) {
    Handle hd;
    hd.size = size;
    hd.ctx = ctx;
    hd.profiler_scope = profiler_scope;
    hd.name = name;
    hd.data_struct = data_struct;
    this->Alloc(&hd);
    return hd;
  }
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param handle handle initialized with size and ctx
   */
  virtual void Alloc(Handle* handle) = 0;
  /*!
   * \brief Increase ref counter on shared memory.
   * \param handle handle to shared memory.
   */
  virtual void SharedIncrementRefCount(Handle handle) = 0;
  /*!
   * \brief Free storage.
   * \param handle Handle struct.
   */
  virtual void Free(Handle handle) = 0;
  /*!
   * \brief Free storage directly, without putting it into memory pool.
   *  This can synchronization of all previous runned device functions.
   *
   *  This function is suitable for conatiner structure with requirement on upsizing
   *  in the beginning phase of the iteration.
   *
   * \param handle Handle struct.
   */
  virtual void DirectFree(Handle handle) = 0;
  /*!
  * \brief Release all memory from device if using a pooled storage manager
  *
  * This release all memory from pool storage managers such as
  * GPUPooledStorageManager and GPUPooledRoundedStorageManager.
  * For non-pool memory managers this has no effect.
  */
  virtual void ReleaseAll(Context ctx) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~Storage() {}
  /*!
   * \brief Returns mutex used by storage manager
   */
  std::mutex& GetMutex(Context::DeviceType dev) {
    if (dev == Context::kCPU) {
      return cpu_mutex_;
    } else {
      return gpu_mutex_;
    }
  }
  /*!
   * \return Storage singleton.
   */
  static Storage* Get();
  /*!
   * \brief Get shared pointer reference to storage singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  Storage to be destructed after X.
   *
   * \return A shared pointer to Storage singleton.
   */
  static std::shared_ptr<Storage> _GetSharedRef();

 private:
  std::mutex cpu_mutex_;
  std::mutex gpu_mutex_;
};  // class Storage
}  // namespace mxnet
#endif  // MXNET_STORAGE_H_
