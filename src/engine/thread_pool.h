/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_THREAD_POOL_H_
#define MXNET_ENGINE_THREAD_POOL_H_

#include <dmlc/base.h>
#include <cstddef>
#include <array>
#include <thread>
#include <utility>
#include "mxnet/base.h"

namespace mxnet {

namespace engine {

/*!
 * \brief Thread pool.
 */
template <std::size_t kSize>
class ThreadPool {
 public:
  /*!
   * \brief Constructor takes function to run and its arguments.
   */
  template <typename Function, typename... Args>
  explicit ThreadPool(Function&& func, Args&&... args);
  /*!
   * \brief Destructor.
   */
  ~ThreadPool() noexcept(false);

 private:
  /*!
   * \brief Worker threads.
   */
  std::array<std::thread, kSize> worker_threads_;
  /*!
   * \brief Disallow default construction.
   */
  ThreadPool() = delete;
  /*!
   * \brief Disallow copy construction and assignment.
   */
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
ThreadPool<kSize>::~ThreadPool() noexcept(false) {
  for (auto&& i : worker_threads_) {
    i.join();
  }
}

}  // namespace engine
}  // namespace mxnet

#endif  // MXNET_ENGINE_THREAD_POOL_H_
