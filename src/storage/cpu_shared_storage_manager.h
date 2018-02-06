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

#ifndef MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_

#include <unordered_map>
#include <mutex>
#include <string>
#include <sstream>

#include "storage_manager.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager for cpu shared memory
 */
class CPUSharedStorageManager : public AbstractManager {
public:

  CPUSharedStorageManager() = default;

  ~CPUSharedStorageManager();

  std::shared_ptr<storage::Handle> Alloc(std::size_t size, Context context) override;

  std::shared_ptr<storage::Handle> GetByID(int shared_pid, int shared_id, std::size_t size);

protected:
  void Free(storage::Handle& handle) override;

private:
  static std::string SharedHandleToString(int shared_pid, int shared_id) {
    std::stringstream name;
    name << "/mx_" << std::hex << shared_pid << "_" << std::hex << shared_id;
    return name.str();
  }

  static int Random() {

    static auto seed_once = []() -> std::mt19937 {
      /* seed once */
      std::random_device random_device;
      return std::mt19937(random_device());
    };

    static thread_local std::mt19937 generator = seed_once();
    static thread_local std::uniform_int_distribution<> distribution;

    return distribution(generator);
  }

  DISALLOW_COPY_AND_ASSIGN(CPUSharedStorageManager);
};  // class CPUSharedStorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_CPU_SHARED_STORAGE_MANAGER_H_
