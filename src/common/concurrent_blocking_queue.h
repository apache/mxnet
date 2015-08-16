/*!
 *  Copyright (c) 2015 by Contributors
 * \file concurrent_blocking_queue.h
 * \brief A simple lock-based consumer-producer queue.
 */
#ifndef MXNET_COMMON_CONCURRENT_BLOCKING_QUEUE_H_
#define MXNET_COMMON_CONCURRENT_BLOCKING_QUEUE_H_

#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <cstdio>

/*!
 * \brief Common components.
 */
namespace common {

/*!
 * \brief A simple lock-based consumer-producer queue.
 */
template<typename T> class ConcurrentBlockingQueue {
  static const int kBusyLoop = 1000;

 public:
  ConcurrentBlockingQueue() : has_elmt_(false), exit_now_(false) {
  }
  /*!
   * \brief Push object into the queue. Notify anyone who is waiting.
   * \param e the object
   */
  void Push(const T& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    has_elmt_ = true;
    queue_.push_back(e);
    if (queue_.size() == 1) {
      cv_.notify_all();
    }
  }
  /*!
   * \brief Pop object out of the queue. If the queue is empty, the caller thread will sleep until
   *       (1) Producer pushed some product into the queue and the caller thread wins it.
   *       (2) A kill signal is passed to the queue.
   * \param rv the pointer point to the return object
   * \return whether an object is returned
   */
  bool Pop(T* rv) {
    for (int i = 0; i < kBusyLoop; i++) {
      if (has_elmt_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_elmt_) {
          assert(queue_.empty());
          continue;
        }
        *rv = queue_.front();
        queue_.pop_front();
        if (queue_.empty())
          has_elmt_ = false;
        return false;
      }
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.empty() && !exit_now_) {
        cv_.wait(lock);
      }
      if (!exit_now_) {
        *rv = queue_.front();
        queue_.pop_front();
        if (queue_.empty())
          has_elmt_ = false;
        return false;
      } else {
        return true;
      }
    }
  }
  /*!
   * \brief pop all objects in the queue.
   * \return a list containing all objects in the queue.
   */
  std::list<T> PopAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::list<T> rv;
    rv.swap(queue_);
    return rv;
  }
  /*!
   * \brief tell the queue to release all waiting consumers
   */
  void SignalForKill() {
    std::unique_lock<std::mutex> lock(mutex_);
    exit_now_ = true;
    cv_.notify_all();
  }
  /*!
   * \brief return the current queue size
   * \return queue size
   */
  size_t QueueSize() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::atomic<bool> has_elmt_;
  std::list<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> exit_now_;

  ConcurrentBlockingQueue(const ConcurrentBlockingQueue&) = delete;
  ConcurrentBlockingQueue& operator=(const ConcurrentBlockingQueue&) = delete;
};

}  // namespace common

#endif  // MXNET_COMMON_CONCURRENT_BLOCKING_QUEUE_H_
