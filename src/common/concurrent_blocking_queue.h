#pragma once
#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <cstdio>

template<typename T> class ConcurrentBlockingQueue {
  const static int BUSY_LOOP = 1000;
 public:
  ConcurrentBlockingQueue() : has_elmt_(false), exit_now_(false) {
  }
  void Push(const T& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    has_elmt_ = true;
    queue_.push_back(e);
    if (queue_.size() == 1) {
      cv_.notify_all();
    }
  }
  bool Pop(T& rv) {
    for (int i = 0; i < BUSY_LOOP; i++) {
      if (has_elmt_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_elmt_) {
          assert(queue_.empty());
          continue;
        }
        rv = queue_.front();
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
        rv = queue_.front();
        queue_.pop_front();
        if (queue_.empty())
          has_elmt_ = false;
        return false;
      } else {
        return true;        
      }
    }
  }
  std::list<T> PopAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::list<T> rv;
    rv.swap(queue_);
    return rv;
  }
  // Call `SignalForKill` before destruction
  void SignalForKill() {
    std::unique_lock<std::mutex> lock(mutex_);
    exit_now_ = true;
    cv_.notify_all();
  }
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
