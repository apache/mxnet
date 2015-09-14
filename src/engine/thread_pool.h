/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_THREAD_POOL_H_
#define MXNET_ENGINE_THREAD_POOL_H_

#include <dmlc/base.h>
#include <cstddef>
#include <vector>
#include <thread>
#include <utility>
#include "mxnet/base.h"

namespace mxnet {
namespace engine {

/*!
 * \brief Thread pool.
 */
class ThreadPool {
 public:
  /*!
   * \brief Constructor takes function to run.
   * \param size size of the thread pool.
   * \param func the function to run on the thread pool.
   */
  explicit ThreadPool(size_t size, std::function<void()> func)
      : worker_threads_(size) {
    for (auto& i : worker_threads_) {
      i = std::thread(func);
    }
  }
  ~ThreadPool() noexcept(false) {
    for (auto&& i : worker_threads_) {
      i.join();
    }
  }

 private:
  /*!
   * \brief Worker threads.
   */
  std::vector<std::thread> worker_threads_;
  /*!
   * \brief Disallow default construction.
   */
  ThreadPool() = delete;
  /*!
   * \brief Disallow copy construction and assignment.
   */
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};
}  // namespace engine
}  // namespace mxnet
#endif  // MXNET_ENGINE_THREAD_POOL_H_
