/*  Copyright (c) 2015 by Contributors
 * Spin lock using xchg.
 * Copied from http://locklessinc.com/articles/locks/
 */

#ifndef MXNET_COMMON_SPIN_LOCK_H_
#define MXNET_COMMON_SPIN_LOCK_H_

/* Compile read-write barrier */
#define barrier() asm volatile("": : :"memory")

/* Pause instruction to prevent excess processor bus usage */
#define cpu_relax() asm volatile("pause\n": : :"memory")

static inline unsigned short xchg_8(void *ptr, unsigned char x) { // NOLINT(*)
  __asm__ __volatile__("xchgb %0,%1"
      :"=r" (x)
      :"m" (*(volatile unsigned char *)ptr), "0" (x)
      :"memory");

  return x;
}

#define BUSY 1
typedef unsigned char spinlock;

/*!
 * \brief use this value to initialize lock object
 */
#define SPINLOCK_INITIALIZER 0

/*!
 * \brief lock
 * \param lock the pointer to lock object
 */
static inline void spin_lock(spinlock *lock) {
  while (1) {
    if (!xchg_8(lock, BUSY)) return;

    while (*lock) cpu_relax();
  }
}

/*!
 * \brief unlock
 * \param lock the pointer to lock object
 */
static inline void spin_unlock(spinlock *lock) {
  barrier();
  *lock = 0;
}

/*!
 * \brief try lock
 * \param lock the pointer to lock object
 * \return whether the lock is grabbed or not
 */
static inline int spin_trylock(spinlock *lock) {
  return xchg_8(lock, BUSY);
}

#endif  // MXNET_COMMON_SPIN_LOCK_H_
