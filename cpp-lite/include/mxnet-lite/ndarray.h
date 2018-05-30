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

#ifndef MXNET_LITE_NDARRAY_H_
#define MXNET_LITE_NDARRAY_H_

#include <map>
#include <string>
#include <vector>

#include <dmlc/logging.h>
#include <mxnet/c_api.h>

#include <mxnet-lite/context.h>

namespace mxnet {
namespace lite {
/*!
* \brief NDArray interface
*/
class NDArray {
 public:
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and read/write can be performed.
  */
  static void WaitAll() {
    CHECK_EQ(MXNDArrayWaitAll(), 0);
  }
  /*!
  * \brief Load NDArrays from binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays returned, do not fill the list if
  * nullptr is given.
  * \param array_map a map from names to NDArrays returned, do not fill the map
  * if nullptr is given or no names is stored in binary file.
  */
  static std::vector<NDArray> LoadList(const std::string &file_name) {
    mx_uint out_size, out_name_size;
    NDArrayHandle *out_arr;
    const char **out_names;
    CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr,
                           &out_name_size, &out_names), 0);
    std::vector<NDArray> array_list;
    array_list.reserve(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      array_list.push_back(NDArray(out_arr[i]));
    }
    return array_list;
  }
  /*!
  * \brief Load map of NDArrays from binary file.
  * \param file_name name of the binary file.
  * \return a list of NDArrays.
  */
  static std::map<std::string, NDArray> LoadDict(const std::string &file_name) {
    mx_uint out_size, out_name_size;
    NDArrayHandle *out_arr;
    const char **out_names;
    CHECK_EQ(MXNDArrayLoad(file_name.c_str(), &out_size, &out_arr,
                          &out_name_size, &out_names), 0);
    CHECK_EQ(out_name_size, out_size)
        << "Cannot load to dict because file was saved as list.";
    std::map<std::string, NDArray> array_map;
    for (mx_uint i = 0; i < out_size; ++i) {
      array_map[out_names[i]] = NDArray(out_arr[i]);
    }
    return array_map;
  }
  /*!
  * \brief save a map of string->NDArray to binary file.
  * \param file_name name of the binary file.
  * \param array_map a map from names to NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::map<std::string, NDArray> &array_map) {
    std::vector<NDArrayHandle> args;
    std::vector<const char *> keys;
    for (const auto &t : array_map) {
      args.push_back(t.second.GetHandle());
      keys.push_back(t.first.c_str());
    }
    CHECK_EQ(MXNDArraySave(file_name.c_str(), args.size(), args.data(),
                           keys.data()), 0);
  }
  /*!
  * \brief save a list of NDArrays to binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::vector<NDArray> &array_list) {
    std::vector<NDArrayHandle> args;
    for (const auto &t : array_list) {
      args.push_back(t.GetHandle());
    }
    CHECK_EQ(MXNDArraySave(file_name.c_str(), args.size(), args.data(),
                           nullptr), 0);
  }
  /*!
  * \brief construct with a none handle
  */
  NDArray() {
    NDArrayHandle handle;
    CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
    blob_ptr_ = std::make_shared<NDBlob>(handle);
  }
  /*!
  * \brief construct with a NDArrayHandle
  */
  explicit NDArray(const NDArrayHandle &handle) {
    blob_ptr_ = std::make_shared<NDBlob>(handle);
  }
  /*!
  * \brief construct a new dynamic NDArray
  * \param shape the shape of array
  * \param constext context of NDArray
  * \param delay_alloc whether delay the allocation
  */
  NDArray(const std::vector<mx_uint> &shape,
          const Context &context,
          bool delay_alloc = true) {
    NDArrayHandle handle;
    CHECK_EQ(MXNDArrayCreate(shape.data(), shape.size(), context.GetDeviceType(),
                             context.GetDeviceId(), delay_alloc, &handle), 0);
    blob_ptr_ = std::make_shared<NDBlob>(handle);
  }
  NDArray(const float *data, size_t size) {
    NDArrayHandle handle;
    CHECK_EQ(MXNDArrayCreateNone(&handle), 0);
    MXNDArraySyncCopyFromCPU(handle, data, size);
    blob_ptr_ = std::make_shared<NDBlob>(handle);
  }
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  */
  NDArray(const float *data,
          const std::vector<mx_uint> &shape,
          const Context &context) {
    NDArrayHandle handle;
    CHECK_EQ(MXNDArrayCreate(shape.data(), shape.size(), context.GetDeviceType(),
                             context.GetDeviceId(), false, &handle), 0);
    size_t size = shape.size() ? 1 : 0;
    for (auto& i : shape) size *= i;
    MXNDArraySyncCopyFromCPU(handle, data, size);
    blob_ptr_ = std::make_shared<NDBlob>(handle);
  }

  void CopyFrom(const float *data, size_t size) {
    CHECK_EQ(MXNDArraySyncCopyFromCPU(blob_ptr_->handle, data, size), 0);
  }

  void CopyTo(float *data, size_t size = 0) {
     CHECK_EQ(MXNDArraySyncCopyToCPU(blob_ptr_->handle, data,
                                     size > 0 ? size : GetSize()), 0);
  }
  /*!
  * \brief return a new copy to this NDArray
  * \param Context the new context of this NDArray
  * \return the new copy
  */
  NDArray Copy(const Context &) const {
    return NDArray();
  }
  /*!
  * \brief Block until all the pending write operations with respect
  *    to current NDArray are finished, and read can be performed.
  */
  void WaitToRead() const {
    CHECK_EQ(MXNDArrayWaitToRead(blob_ptr_->handle), 0);
  }
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and write can be performed.
  */
  void WaitToWrite() const {
    CHECK_EQ(MXNDArrayWaitToWrite(blob_ptr_->handle), 0);
  }
  /*!
  * \return the size of current NDArray, a.k.a. the production of all shape dims
  */
  size_t GetSize() const {
    size_t ret = 1;
    for (auto &i : GetShape()) ret *= i;
    return ret;
  }
  /*!
  * \return the shape of current NDArray, in the form of mx_uint vector
  */
  std::vector<mx_uint> GetShape() const {
    const mx_uint *out_pdata;
    mx_uint out_dim;
    CHECK_EQ(MXNDArrayGetShape(blob_ptr_->handle, &out_dim, &out_pdata), 0);
    std::vector<mx_uint> ret;
    for (mx_uint i = 0; i < out_dim; ++i) {
      ret.push_back(out_pdata[i]);
    }
    return ret;
  }
  /*!
  * \return the data type of current NDArray
  */
  int GetDType() const {
    int ret;
    CHECK_EQ(MXNDArrayGetDType(blob_ptr_->handle, &ret), 0);
    return ret;
  }
  /*!
  * \brief Get the pointer to data (IMPORTANT: The ndarray should not be in GPU)
  * \return the data pointer to the current NDArray
  */
  const float *GetData() const {
    if (GetDType() != 0) {
      return NULL;
    }
    void *ret;
    CHECK_EQ(MXNDArrayGetData(blob_ptr_->handle, &ret), 0);
    return static_cast<float*>(ret);
  }
  /*!
  * \return the context of NDArray
  */
  Context GetContext() const {
    int out_dev_type;
    int out_dev_id;
    CHECK_EQ(MXNDArrayGetContext(blob_ptr_->handle, &out_dev_type,
                                 &out_dev_id), 0);
    return Context(static_cast<DeviceType>(out_dev_type), out_dev_id);
  }
  /*!
  * \return the NDArrayHandle of the current NDArray
  */
  NDArrayHandle GetHandle() const {
    return blob_ptr_->handle;
  }

 private:
  /*!
  * \brief struct to store NDArrayHandle
  */
  struct NDBlob {
    /*!
    * \brief default constructor
    */
    NDBlob() : handle(nullptr) {}
    /*!
    * \brief construct with a NDArrayHandle
    * \param handle NDArrayHandle to store
    */
    explicit NDBlob(NDArrayHandle handle_) : handle(handle_) {}
    /*!
    * \brief destructor, free the NDArrayHandle
    */
    ~NDBlob() { MXNDArrayFree(handle); }
    /*!
    * \brief the NDArrayHandle
    */
    NDArrayHandle handle;
  };
  std::shared_ptr<NDBlob> blob_ptr_;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_LITE_NDARRAY_H_
