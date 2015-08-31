/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_OBJECT_POOL_H_
#define MXNET_DAG_ENGINE_OBJECT_POOL_H_
#include <cstdlib>
#include <mutex>
#include "common.h"

template <typename T>
class SmallObjectPool {
 public:
  struct LinkedList {
    union {
      LinkedList* next{nullptr};
      T t;
    };
  };
  ~SmallObjectPool() = default;
  T* New();
  void Delete(T* ptr);

  static SmallObjectPool* Get();

 private:
  constexpr static std::size_t kPageSize = 1 << 12;
  std::recursive_mutex m_;
  LinkedList* head_{nullptr};
  SmallObjectPool();
  void AllocateChunk();

  SmallObjectPool(SmallObjectPool const&) = delete;
  SmallObjectPool(SmallObjectPool&&) = delete;
  SmallObjectPool& operator=(SmallObjectPool const&) = delete;
  SmallObjectPool& operator=(SmallObjectPool&&) = delete;
};

template <typename T>
T* SmallObjectPool<T>::New() {
  LinkedList* ret;
  {
    std::lock_guard<std::recursive_mutex> lock{m_};
    if (head_->next == nullptr) {
      AllocateChunk();
    }
    ret = head_;
    head_ = head_->next;
  }
  return new(static_cast<void*>(ret)) T{};
}

template <typename T>
void SmallObjectPool<T>::Delete(T* ptr) {
  ptr->~T();
  auto linked_list_ptr = reinterpret_cast<LinkedList*>(ptr);
  {
    std::lock_guard<std::recursive_mutex> lock{m_};
    linked_list_ptr->next = head_;
    head_ = linked_list_ptr;
  }
}

template <typename T>
SmallObjectPool<T>* SmallObjectPool<T>::Get() {
  static SmallObjectPool<T> inst;
  return &inst;
}

template <typename T>
SmallObjectPool<T>::SmallObjectPool() {
  AllocateChunk();
}

template <typename T>
void SmallObjectPool<T>::AllocateChunk() {
  std::lock_guard<std::recursive_mutex> lock{m_};
  static_assert(kPageSize % sizeof(LinkedList) == 0,
                "Could not align to page size.");
  auto&& new_chunk = static_cast<LinkedList*>(malloc(kPageSize));
  auto size = kPageSize / sizeof(LinkedList);
  for (std::size_t i = 0 ; i < size - 1; ++i) {
    new_chunk[i].next = &new_chunk[i + 1];
  }
  new_chunk[size - 1].next = head_;
  head_ = new_chunk;
}


struct A {
  A() {
    LOG("constructing");
  }
  ~A() {
    LOG("destructing");
  }
};

int main() {
  auto&& pool = SmallObjectPool<A>::Get();
  auto a = pool->New();
  auto b = pool->New();
  LOG("addresses %p %p", a, b);
  pool->Delete(a);
  a = pool->New();
  LOG("address again %p", a);
  return 0;
}
#endif  // MXNET_DAG_ENGINE_OBJECT_POOL_H_
