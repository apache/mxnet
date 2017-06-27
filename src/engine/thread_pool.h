/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_ENGINE_THREAD_POOL_H_
#define MXNET_ENGINE_THREAD_POOL_H_

#include <dmlc/base.h>
#include <cstddef>
#include <vector>
#include <list>
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
  /*! \brief Simple manually-signalled event gate which remains open */
  class SimpleEvent {
   public:
    SimpleEvent()
      : signaled_(false) {}
    void wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!signaled_) {
        condition_variable_.wait(lock);
      }
    }
    void signal() {
      signaled_ = true;
      std::unique_lock<std::mutex> lk(mutex_);
      condition_variable_.notify_all();
    }

    /*! \brief Signal event upon destruction, even for exceptions (RAII) */
    struct SetReadyOnDestroy {
      explicit inline SetReadyOnDestroy(std::shared_ptr<SimpleEvent> event)
        : event_(event) {
      }
      inline ~SetReadyOnDestroy() {
        if (event_) {
          event_->signal();
        }
      }
      std::shared_ptr<SimpleEvent>  event_;
    };

   private:
    std::mutex              mutex_;
    std::condition_variable condition_variable_;
    std::atomic<bool>       signaled_;
  };

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
  explicit ThreadPool(size_t size,
                      std::function<void(std::shared_ptr<SimpleEvent> ready)> func,
                      const bool wait)
      : worker_threads_(size) {
    for (auto& i : worker_threads_) {
      std::shared_ptr<SimpleEvent> ptr = std::make_shared<SimpleEvent>();
      ready_events_.emplace_back(ptr);
      i = std::thread(func, ptr);
    }
    if (wait) {
      WaitForReady();
    }
  }
  ~ThreadPool() noexcept(false) {
    for (auto&& i : worker_threads_) {
      i.join();
    }
  }

 private:
  /*!
   * \brief Wait for all started threads to signal that they're ready
   */
  void WaitForReady() {
    for (std::shared_ptr<SimpleEvent> ptr : ready_events_) {
      ptr->wait();
    }
  }

  /*!
   * \brief Worker threads.
   */
  std::vector<std::thread> worker_threads_;
  /*!
   * \brief Startup synchronization objects
   */
  std::list<std::shared_ptr<SimpleEvent>> ready_events_;
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
