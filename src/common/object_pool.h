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
 */
#ifndef MXNET_COMMON_OBJECT_POOL_H_
#define MXNET_COMMON_OBJECT_POOL_H_
#include <dmlc/logging.h>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace mxnet {
namespace common {
/*!
 * \brief Object pool for fast allocation and deallocation.
 */
template <typename T>
class ObjectPool {
 public:
  /*!
   * \brief Destructor.
   */
  ~ObjectPool();
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
  template <typename... Args>
  T* New(Args&&... args);
  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  void Delete(T* ptr);

  /*!
   * \brief Get singleton instance of pool.
   * \return Object Pool.
   */
  static ObjectPool* Get();

  /*!
   * \brief Get a shared ptr of the singleton instance of pool.
   * \return Shared pointer to the Object Pool.
   */
  static std::shared_ptr<ObjectPool> _GetSharedRef();

 private:
  /*!
   * \brief Internal structure to hold pointers.
   */
  struct LinkedList {
#if defined(_MSC_VER)
    T t;
    LinkedList* next{nullptr};
#else
    union {
      T t;
      LinkedList* next{nullptr};
    };
#endif
  };
  /*!
   * \brief Page size of allocation.
   *
   * Currently defined to be 4KB.
   */
  constexpr static std::size_t kPageSize = 1 << 12;
  /*! \brief internal mutex */
  std::mutex m_;
  /*!
   * \brief Head of free list.
   */
  LinkedList* head_{nullptr};
  /*!
   * \brief Pages allocated.
   */
  std::vector<void*> allocated_;
  /*!
   * \brief Private constructor.
   */
  ObjectPool();
  /*!
   * \brief Allocate a page of raw objects.
   *
   * This function is not protected and must be called with caution.
   */
  void AllocateChunk();
  DISALLOW_COPY_AND_ASSIGN(ObjectPool);
};  // class ObjectPool

/*!
 * \brief Helper trait class for easy allocation and deallocation.
 */
template <typename T>
struct ObjectPoolAllocatable {
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
  template <typename... Args>
  static T* New(Args&&... args);
  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  static void Delete(T* ptr);
};  // struct ObjectPoolAllocatable

template <typename T>
ObjectPool<T>::~ObjectPool() {
  // TODO(hotpxl): mind destruction order
  // for (auto i : allocated_) {
  //   free(i);
  // }
}

template <typename T>
template <typename... Args>
T* ObjectPool<T>::New(Args&&... args) {
  LinkedList* ret;
  {
    std::lock_guard<std::mutex> lock{m_};
    if (head_->next == nullptr) {
      AllocateChunk();
    }
    ret = head_;
    head_ = head_->next;
  }
  return new (static_cast<void*>(ret)) T(std::forward<Args>(args)...);
}

template <typename T>
void ObjectPool<T>::Delete(T* ptr) {
  ptr->~T();
  auto linked_list_ptr = reinterpret_cast<LinkedList*>(ptr);
  {
    std::lock_guard<std::mutex> lock{m_};
    linked_list_ptr->next = head_;
    head_ = linked_list_ptr;
  }
}

template <typename T>
ObjectPool<T>* ObjectPool<T>::Get() {
  return _GetSharedRef().get();
}

template <typename T>
std::shared_ptr<ObjectPool<T> > ObjectPool<T>::_GetSharedRef() {
  static std::shared_ptr<ObjectPool<T> > inst_ptr(new ObjectPool<T>());
  return inst_ptr;
}

template <typename T>
ObjectPool<T>::ObjectPool() {
  AllocateChunk();
}

template <typename T>
void ObjectPool<T>::AllocateChunk() {
  static_assert(sizeof(LinkedList) <= kPageSize, "Object too big.");
  static_assert(sizeof(LinkedList) % alignof(LinkedList) == 0, "ObjectPooll Invariant");
  static_assert(alignof(LinkedList) % alignof(T) == 0, "ObjectPooll Invariant");
  static_assert(kPageSize % alignof(LinkedList) == 0, "ObjectPooll Invariant");
  void* new_chunk_ptr;
#ifdef _MSC_VER
  new_chunk_ptr = _aligned_malloc(kPageSize, kPageSize);
  CHECK(new_chunk_ptr != NULL) << "Allocation failed";
#else
  int ret = posix_memalign(&new_chunk_ptr, kPageSize, kPageSize);
  CHECK_EQ(ret, 0) << "Allocation failed";
#endif
  allocated_.emplace_back(new_chunk_ptr);
  auto new_chunk = static_cast<LinkedList*>(new_chunk_ptr);
  auto size = kPageSize / sizeof(LinkedList);
  for (std::size_t i = 0; i < size - 1; ++i) {
    new_chunk[i].next = &new_chunk[i + 1];
  }
  new_chunk[size - 1].next = head_;
  head_ = new_chunk;
}

template <typename T>
template <typename... Args>
T* ObjectPoolAllocatable<T>::New(Args&&... args) {
  return ObjectPool<T>::Get()->New(std::forward<Args>(args)...);
}

template <typename T>
void ObjectPoolAllocatable<T>::Delete(T* ptr) {
  ObjectPool<T>::Get()->Delete(ptr);
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_OBJECT_POOL_H_
