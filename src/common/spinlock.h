#ifndef MXNET_COMMON_SPINLOCK_H_
#define MXNET_COMMON_SPINLOCK_H_

#include <atomic>

namespace mxnet {
namespace common {

/*!
 * \brief Simple userspace spinlock implementation.
 */
class Spinlock {
 public:
  Spinlock() = default;
  /*!
   * \brief Disable copy and move.
   */
  Spinlock(Spinlock const&) = delete;
  Spinlock(Spinlock&&) = delete;
  Spinlock& operator=(Spinlock const&) = delete;
  Spinlock& operator=(Spinlock&&) = delete;
  ~Spinlock() = default;
  /*!
   * \brief Acquire lock.
   */
  void lock() noexcept {
    while (lock_.test_and_set(std::memory_order_acquire));
  }
  /*!
   * \brief Release lock.
   */
  void unlock() noexcept {
    lock_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};

}  // namespace common
}  // namespace mxnet

#endif  // MXNET_COMMON_SPINLOCK_H_

