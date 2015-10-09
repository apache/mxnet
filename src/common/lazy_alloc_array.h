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

namespace mxnet {
namespace common {

template<typename TElem>
class LazyAllocArray {
 public:
  /*!
   * \brief Get element of corresponding index,
   *  if it is not created create by creator
   * \param index the array index position
   * \param creator a lambda function to create new element when needed.
   */
  template<typename FCreate>
  inline TElem* Get(int index, FCreate creator);
  /*!
   * \brief for each not null element of the array, call fvisit
   * \param fvisit a function of (size_t, TElem*)
   */
  template<typename FVisit>
  inline void ForEach(FVisit fvisit);
  /*! \brief clear all the allocated elements in array */
  inline void Clear();

 private:
  /*! \brief the initial size of the array */
  static constexpr std::size_t kInitSize = 16;
  /*! \brief mutex used during creation */
  std::mutex create_mutex_;
  /*! \brief internal data fir initial size */
  std::array<std::unique_ptr<TElem>, kInitSize> head_;
  /*! \brief overflow array of more elements */
  std::vector<std::unique_ptr<TElem> > more_;
};

// implementations
template<typename TElem>
template<typename FCreate>
inline TElem* LazyAllocArray<TElem>::Get(int index, FCreate creator) {
  CHECK_GE(index, 0);
  size_t idx = static_cast<size_t>(index);
  if (idx < kInitSize) {
    TElem *ptr = head_[idx].get();
    if (ptr != nullptr) {
      return ptr;
    } else {
      std::lock_guard<std::mutex> lock(create_mutex_);
      TElem *ptr = head_[idx].get();
      if (ptr != nullptr) return ptr;
      head_[idx].reset(ptr = creator());
      return ptr;
    }
  } else {
    std::lock_guard<std::mutex> lock(create_mutex_);
    idx -= kInitSize;
    if (more_.size() <= idx) more_.resize(idx + 1);
    TElem *ptr = more_[idx].get();
    if (ptr != nullptr) return ptr;
    more_[idx].reset(ptr = creator());
    return ptr;
  }
}

template<typename TElem>
inline void LazyAllocArray<TElem>::Clear() {
  std::lock_guard<std::mutex> lock(create_mutex_);
  for (size_t i = 0; i < head_.size(); ++i) {
    head_[i].reset(nullptr);
  }
  for (size_t i = 0; i < more_.size(); ++i) {
    more_[i].reset(nullptr);
  }
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
