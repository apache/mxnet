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
 * \file storage_manager.h
 * \brief Storage manager interface.
 */

#ifndef MXNET_STORAGE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_STORAGE_MANAGER_H_

#include <type_traits>
#include <mxnet/storage.h>

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager interface.
 */
class AbstractManager
  : public virtual AbstractStorage, public virtual std::enable_shared_from_this<AbstractManager> {
 protected:
  /*!
   * \brief The protected default constructor.
   *
   * Since this class uses std::enable_shared_from_this instances of derived classes should be
   * constructed only with std::make_shared.
   */
  AbstractManager() = default;

 public:
  /*!
   * \brief Create a shared pointer to an object of a derived from AbstractManager class.
   *
   * \see AbstractManager()
   */
  template<class T>
  static
  std::shared_ptr<
    typename std::enable_if<std::is_base_of<AbstractManager, T>::value, T>::type>
  make() {
    return std::make_shared<T>();
  }

  /*!
   * \brief The default custom deleter for shared_ptr
   *
   * Will call Free and delete the object.
   */
  std::function<void(Handle*)> DefaultDeleter();

  void DirectFree(std::shared_ptr<Handle>* handle) override;

  /*!
   * \brief Free storage
   * \param handle The storage handle
   */
  virtual void Free(Handle* handle) = 0;

  /*!
   * \brief Direct free storage
   *
   * The default implementation calls Free(Handle& handle).
   *
   * \param handle The storage handle
   *
   * \see DirectFree(std::shared_ptr<Handle>&)
   */
  virtual void DirectFree(Handle* handle);

  virtual ~AbstractManager() = default;

}; // class AbstractManager

} // namespace storage
} // namespace mxnet

#endif  // MXNET_STORAGE_STORAGE_MANAGER_H_
