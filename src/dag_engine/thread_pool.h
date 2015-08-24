/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_DAG_ENGINE_THREAD_POOL_H_
#define MXNET_DAG_ENGINE_THREAD_POOL_H_

#include <dmlc/base.h>
#include <cstddef>
#include <array>
#include <thread>
#include <utility>
#include "mxnet/base.h"

namespace mxnet {

namespace engine {

template <std::size_t kSize>
class ThreadPool {
 public:
  template <typename Function, typename... Args>
  explicit ThreadPool(Function&& func, Args&&... args);
  ~ThreadPool();

 private:
  std::array<std::thread, kSize> worker_threads_;
  ThreadPool();
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

template <std::size_t kSize>
template <typename Function, typename... Args>
ThreadPool<kSize>::ThreadPool(Function&& func, Args&&... args) {
  for (auto&& i : worker_threads_) {
    i = std::thread{std::forward<Function>(func), std::forward<Args>(args)...};
  }
}

template <std::size_t kSize>
ThreadPool<kSize>::~ThreadPool() {
  for (auto&& i : worker_threads_) {
    i.join();
  }
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_DAG_ENGINE_THREAD_POOL_H_
