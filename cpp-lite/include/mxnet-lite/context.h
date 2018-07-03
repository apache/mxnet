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

#ifndef MXNET_LITE_CONTEXT_H_
#define MXNET_LITE_CONTEXT_H_

namespace mxnet {
namespace lite {

enum DeviceType {
  kCPU = 1,
  kGPU = 2,
  kCPUPinned = 3
};

/*!
* \brief Context interface
*/
class Context {
 public:
  /*!
   * \brief Return a GPU context
   * \param device_id id of the device
   * \return the corresponding GPU context
   */
  static Context GPU(int device_id = 0) {
    return Context(DeviceType::kGPU, device_id);
  }
  /*!
   * \brief Return a CPU context
   * \param device_id id of the device. this is not needed by CPU
   * \return the corresponding CPU context
   */
  static Context CPU(int device_id = 0) {
    return Context(DeviceType::kCPU, device_id);
  }
  /*!
  * \brief Context constructor
  * \param type type of the device
  * \param id id of the device
  */
  Context(const DeviceType &type, int id) : type_(type), id_(id) {}
  /*!
  * \return the type of the device
  */
  DeviceType GetDeviceType() const { return type_; }
  /*!
  * \return the id of the device
  */
  int GetDeviceId() const { return id_; }
  bool operator==(const Context& other) {
    return type_ == other.type_ && id_ == other.id_;
  }
  bool operator!=(const Context& other) {
    return !(*this == other);
  }

 private:
  DeviceType type_;
  int id_;
};

}  // namespace lite
}  // namespace mxnet

#endif  // MXNET_LITE_CONTEXT_H_
