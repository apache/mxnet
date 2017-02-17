/*!
 * Copyright (c) 2015 by Contributors
 * \file concurrency.h
 * \brief thread-safe data structures.
 * \author Yutian Li
 */
#ifndef MXNET_ENGINE_CONCURRENCY_EXT_H_
#define MXNET_ENGINE_CONCURRENCY_EXT_H_
// this code depends on c++11
#if DMLC_USE_CXX11
#include <atomic>
#include <queue>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "dmlc/base.h"

#ifdef MXNET_USE_MOODYCAMEL
#include "moodycamel/blockingconcurrentqueue.inc"
#endif

namespace dmlc {

/*!
 * \brief Cocurrent blocking queue with custom lock primitive.
 */
template <typename T,
          typename L,
          ConcurrentQueueType type = ConcurrentQueueType::kFIFO>
class ConcurrentBlockingQueueWithLock {
 public:
  ConcurrentBlockingQueueWithLock();
  ~ConcurrentBlockingQueueWithLock() = default;
  /*!
   * \brief Push element into the queue.
   * \param e Element to push into.
   * \param priority the priority of the element, only used for priority queue.
   *            The higher the priority is, the better.
   * \tparam E the element type
   *
   * It will copy or move the element into the queue, depending on the type of
   * the parameter.
   */
  template <typename E>
  void Push(E&& e, int priority = 0);
  /*!
   * \brief Pop element from the queue.
   * \param rv Element popped.
   * \return On false, the queue is exiting.
   *
   * The element will be copied or moved into the object passed in.
   */
  bool Pop(T* rv);

    bool Last(T *rv) {
        std::unique_lock<L> lock{mutex_};
        if (type == ConcurrentQueueType::kFIFO) {
            if (fifo_queue_.empty()) return false;
            *rv = fifo_queue_.back();
            return true;
        }
    }

  /*!
   * \brief Signal the queue for destruction.
   *
   * After calling this method, all blocking pop call to the queue will return
   * false.
   */
  void SignalForKill();
  /*!
   * \brief Get the size of the queue.
   * \return The size of the queue.
   */
  size_t Size();

  /*!
   * \brief Get an approximate size of the queue.
   * \return The approximate size of the queue.
   */
  size_t ApproxSize() {
    return Size();
  }

 private:
  struct Entry {
    T data;
    int priority;
    inline bool operator<(const Entry &b) const {
      return priority < b.priority;
    }
  };

  L mutex_;

  std::condition_variable_any cv_;
  std::atomic<bool> exit_now_;
  int nwait_consumer_;
  // a priority queue
  std::vector<Entry> priority_queue_;
  // a FIFO queue
  std::queue<T> fifo_queue_;
  /*!
   * \brief Disable copy and move.
   */
  DISALLOW_COPY_AND_ASSIGN(ConcurrentBlockingQueueWithLock);
};

template <typename T, typename L, ConcurrentQueueType type>
ConcurrentBlockingQueueWithLock<T, L, type>::ConcurrentBlockingQueueWithLock()
    : exit_now_{false}, nwait_consumer_{0} {}

template <typename T, typename L, ConcurrentQueueType type>
template <typename E>
void ConcurrentBlockingQueueWithLock<T, L, type>::Push(E&& e, int priority) {
  static_assert(std::is_same<typename std::remove_cv<
                                 typename std::remove_reference<E>::type>::type,
                             T>::value,
                "Types must match.");
  bool notify;
  {
    std::lock_guard<L> lock{mutex_};
    if (type == ConcurrentQueueType::kFIFO) {
      fifo_queue_.emplace(std::forward<E>(e));
      notify = nwait_consumer_ != 0;
    } else {
      Entry entry;
      entry.data = std::move(e);
      entry.priority = priority;
      priority_queue_.push_back(std::move(entry));
      std::push_heap(priority_queue_.begin(), priority_queue_.end());
      notify = nwait_consumer_ != 0;
    }
  }
  if (notify) cv_.notify_one();
}

template <typename T, typename L, ConcurrentQueueType type>
bool ConcurrentBlockingQueueWithLock<T, L, type>::Pop(T* rv) {
  std::unique_lock<L> lock{mutex_};
  if (type == ConcurrentQueueType::kFIFO) {
    ++nwait_consumer_;
    cv_.wait(lock, [this] {
        return !fifo_queue_.empty() || exit_now_.load();
      });
    --nwait_consumer_;
    if (!exit_now_.load()) {
      *rv = std::move(fifo_queue_.front());
      fifo_queue_.pop();
      return true;
    } else {
      return false;
    }
  } else {
    ++nwait_consumer_;
    cv_.wait(lock, [this] {
        return !priority_queue_.empty() || exit_now_.load();
      });
    --nwait_consumer_;
    if (!exit_now_.load()) {
      std::pop_heap(priority_queue_.begin(), priority_queue_.end());
      *rv = std::move(priority_queue_.back().data);
      priority_queue_.pop_back();
      return true;
    } else {
      return false;
    }
  }
}

template <typename T, typename L, ConcurrentQueueType type>
void ConcurrentBlockingQueueWithLock<T, L, type>::SignalForKill() {
  {
    std::lock_guard<L> lock{mutex_};
    exit_now_.store(true);
  }
  cv_.notify_all();
}

template <typename T, typename L, ConcurrentQueueType type>
size_t ConcurrentBlockingQueueWithLock<T, L, type>::Size() {
  std::lock_guard<L> lock{mutex_};
  if (type == ConcurrentQueueType::kFIFO) {
    return fifo_queue_.size();
  } else {
    return priority_queue_.size();
  }
}

#ifdef MXNET_USE_MOODYCAMEL
/*!
 * \brief Cocurrent blocking queue with custom lock primitive.
 */
template <typename T>
class LocklessConcurrentBlockingQueue {
 public:
  LocklessConcurrentBlockingQueue() { }
  ~LocklessConcurrentBlockingQueue() = default;

  /*!
   * \brief Push element into the queue.
   * \param e Element to push into.
   * \param priority the priority of the element, only used for priority queue.
   *            The higher the priority is, the better.
   * \tparam E the element type
   *
   * It will copy or move the element into the queue, depending on the type of
   * the parameter.
   */
  template <typename E>
  void Push(E&& e) {
      if (!queue_.enqueue(e)) throw std::runtime_error("Out of memory");
  }

  /*!
   * \brief Pop element from the queue.
   * \param rv Element popped.
   * \return On false, the queue is exiting.
   *
   * The element will be copied or moved into the object passed in.
   */
  bool Pop(T* rv) {
    if (exiting_) return false;
    queue_.wait_dequeue(*rv);
    if (exiting_) return false;
    return true;
  }

  /*!
   * \brief Signal the queue for destruction.
   *
   * After calling this method, all blocking pop call to the queue will return
   * false.
   */
  void SignalForKill() {
    exiting_ = true;
    queue_.enqueue(nullptr);
  }

  /*!
   * \brief Get the size of the queue.
   * \return The size of the queue.
   */
  size_t ApproxSize() {
    return queue_.size_approx();
  }

 private:
  moodycamel::BlockingConcurrentQueue<T> queue_;
  std::atomic<bool> exiting_ { false };

  /*!
   * \brief Disable copy and move.
   */
  DISALLOW_COPY_AND_ASSIGN(LocklessConcurrentBlockingQueue);
};
#endif  // MXNET_USE_MOODYCAMEL

}  // namespace dmlc
#endif  // DMLC_USE_CXX11
#endif  // MXNET_ENGINE_CONCURRENCY_EXT_H_
