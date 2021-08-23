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
 * \file lazy_alloc_array.h
 * \brief An array that lazily allocate elements as
 *   First time the cell get visited.
 */
#ifndef MXNET_COMMON_LAZY_ALLOC_ARRAY_H_
#define MXNET_COMMON_LAZY_ALLOC_ARRAY_H_

#include <dmlc/logging.h>
#include <memory>
#include <mutex>
#include <array>
#include <vector>
#include <atomic>

namespace mxnet {
namespace common {

template<typename TElem>
class LazyAllocArray {
 public:
  LazyAllocArray();
  /*!
   * \brief Get element of corresponding index,
   *  if it is not created create by creator
   * \param index the array index position
   * \param creator a lambda function to create new element when needed.
   */
  template<typename FCreate>
  inline std::shared_ptr<TElem> Get(int index, FCreate creator);
  /*!
   * \brief for each not null element of the array, call fvisit
   * \param fvisit a function of (size_t, TElem*)
   */
  template<typename FVisit>
  inline void ForEach(FVisit fvisit);
  /*! \brief clear all the allocated elements in array */
  inline void Clear();

 private:
  template<typename SyncObject>
  class unique_unlock {
   public:
    explicit unique_unlock(std::unique_lock<SyncObject> *lock)
      : lock_(lock) {
      if (lock_) {
        lock_->unlock();
      }
    }
    ~unique_unlock() {
      if (lock_) {
        lock_->lock();
      }
    }
   private:
    std::unique_lock<SyncObject> *lock_;
  };

  /*! \brief the initial size of the array */
  static constexpr std::size_t kInitSize = 16;
  /*! \brief mutex used during creation */
  std::mutex create_mutex_;
  /*! \brief internal data fir initial size */
  std::array<std::shared_ptr<TElem>, kInitSize> head_;
  /*! \brief overflow array of more elements */
  std::vector<std::shared_ptr<TElem> > more_;
  /*! \brief Signal shutdown of array */
  std::atomic<bool> is_clearing_;
};

template<typename TElem>
inline LazyAllocArray<TElem>::LazyAllocArray()
  : is_clearing_(false) {
}

// implementations
template<typename TElem>
template<typename FCreate>
inline std::shared_ptr<TElem> LazyAllocArray<TElem>::Get(int index, FCreate creator) {
  CHECK_GE(index, 0);
  size_t idx = static_cast<size_t>(index);
  if (idx < kInitSize) {
    std::shared_ptr<TElem> ptr = head_[idx];
    if (ptr) {
      return ptr;
    } else {
      std::lock_guard<std::mutex> lock(create_mutex_);
      if (!is_clearing_.load()) {
        std::shared_ptr<TElem> ptr = head_[idx];
        if (ptr) {
          return ptr;
        }
        ptr = head_[idx] = std::shared_ptr<TElem>(creator());
        return ptr;
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(create_mutex_);
    if (!is_clearing_.load()) {
      idx -= kInitSize;
      if (more_.size() <= idx) {
        more_.reserve(idx + 1);
        while (more_.size() <= idx) {
          more_.push_back(std::shared_ptr<TElem>(nullptr));
        }
      }
      std::shared_ptr<TElem> ptr = more_[idx];
      if (ptr) {
        return ptr;
      }
      ptr = more_[idx] = std::shared_ptr<TElem>(creator());
      return ptr;
    }
  }
  return nullptr;
}

template<typename TElem>
inline void LazyAllocArray<TElem>::Clear() {
  std::unique_lock<std::mutex> lock(create_mutex_);
  is_clearing_.store(true);
  // Currently, head_ and more_ never get smaller, so it's safe to
  // iterate them outside of the lock.  The loops should catch
  // any growth which might happen when create_mutex_ is unlocked
  for (size_t i = 0; i < head_.size(); ++i) {
    std::shared_ptr<TElem> p = head_[i];
    head_[i] = std::shared_ptr<TElem>(nullptr);
    unique_unlock<std::mutex> unlocker(&lock);
    p = std::shared_ptr<TElem>(nullptr);
  }
  for (size_t i = 0; i < more_.size(); ++i) {
    std::shared_ptr<TElem> p = more_[i];
    more_[i] = std::shared_ptr<TElem>(nullptr);
    unique_unlock<std::mutex> unlocker(&lock);
    p = std::shared_ptr<TElem>(nullptr);
  }
  more_.clear();
  is_clearing_.store(false);
}

template<typename TElem>
template<typename FVisit>
inline void LazyAllocArray<TElem>::ForEach(FVisit fvisit) {
  std::lock_guard<std::mutex> lock(create_mutex_);
  for (size_t i = 0; i < head_.size(); ++i) {
    if (head_[i].get() != nullptr) {
      fvisit(i, head_[i].get());
    }
  }
  for (size_t i = 0; i < more_.size(); ++i) {
    if (more_[i].get() != nullptr) {
      fvisit(i + kInitSize, more_[i].get());
    }
  }
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_LAZY_ALLOC_ARRAY_H_
