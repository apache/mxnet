/*!
 * Copyright by Contributors
 * \file thread.h
 * \brief this header include the minimum necessary resource
 * for multi-threading that can be compiled in windows, linux, mac
 * \author Tianqi Chen
 */
#ifndef MSHADOW_PS_THREAD_H_ // NOLINT(*)
#define MSHADOW_PS_THREAD_H_ // NOLINT(*)

#ifdef _MSC_VER
#include <windows.h>
#include <process.h>
#include "../mshadow/logging.h"
namespace mshadow {
namespace utils {
/*! \brief simple semaphore used for synchronization */
class Semaphore {
 public :
  inline void Init(int init_val) {
    sem = CreateSemaphore(NULL, init_val, 10, NULL);
    CHECK_NE(sem, NULL) << "create Semaphore error";
  }
  inline void Destroy(void) {
    CloseHandle(sem);
  }
  inline void Wait(void) {
    CHECK_EQ(WaitForSingleObject(sem, INFINITE), WAIT_OBJECT_0)
      << "WaitForSingleObject error";
  }
  inline void Post(void) {
    CHECK_NE(ReleaseSemaphore(sem, 1, NULL), 0) << "ReleaseSemaphore error";
  }

 private:
  HANDLE sem;
};

/*! \brief mutex under windows */
class Mutex {
 public:
  inline void Init(void) {
    CHECK_NE(InitializeCriticalSectionAndSpinCount(&mutex, 0x00000400), 0)
      << "Mutex::Init fail";
  }
  inline void Lock(void) {
    EnterCriticalSection(&mutex);
  }
  inline void Unlock(void) {
    LeaveCriticalSection(&mutex);
  }
  inline void Destroy(void) {
    DeleteCriticalSection(&mutex);
  }

 private:
  friend class ConditionVariable;
  CRITICAL_SECTION mutex;
};

// conditional variable that uses pthread
class ConditionVariable {
 public:
  // initialize conditional variable
  inline void Init(void) {
    InitializeConditionVariable(&cond);
  }
  // destroy the thread
  inline void Destroy(void) {
    // DeleteConditionVariable(&cond);
  }
  // wait on the conditional variable
  inline void Wait(Mutex *mutex) {
    CHECK_NE(SleepConditionVariableCS(&cond, &(mutex->mutex), INFINITE), 0)
      << "ConditionVariable:Wait fail";
  }
  inline void Broadcast(void) {
    WakeAllConditionVariable(&cond);
  }
  inline void Signal(void) {
    WakeConditionVariable(&cond);
  }

 private:
  CONDITION_VARIABLE cond;
};

/*! \brief simple thread that wraps windows thread */
class Thread {
 private:
  HANDLE    thread_handle;
  unsigned  thread_id;
 public:
  inline void Start(unsigned int __stdcall entry(void*p), void *param) {
    thread_handle = (HANDLE)_beginthreadex(NULL, 0, entry, param, 0, &thread_id);
  }
  inline int Join(void) {
    WaitForSingleObject(thread_handle, INFINITE);
    return 0;
  }
};
/*! \brief exit function called from thread */
inline void ThreadExit(void *status) {
  _endthreadex(0);
}
#define MSHADOW_THREAD_PREFIX unsigned int __stdcall
}  // namespace utils
}  // namespace mshadow
#else
// thread interface using g++
#include <semaphore.h>
#include <pthread.h>
#include <errno.h>
namespace mshadow {
namespace utils {
/*!\brief semaphore class */
class Semaphore {
  #ifdef __APPLE__

 private:
  sem_t* semPtr;
  char sema_name[20];

 private:
  inline void GenRandomString(char *s, const int len) {
    static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (int i = 0; i < len; ++i) {
      s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
  }

 public:
  inline void Init(int init_val) {
    sema_name[0] = '/';
    sema_name[1] = 's';
    sema_name[2] = 'e';
    sema_name[3] = '/';
    GenRandomString(&sema_name[4], 16);
    if ((semPtr = sem_open(sema_name, O_CREAT, 0644, init_val)) == SEM_FAILED) {
      perror("sem_open");
      exit(1);
    }
    CHECK_NE(semPtr, NULL) << "create Semaphore error";
  }
  inline void Destroy(void) {
    if (sem_close(semPtr) == -1) {
      perror("sem_close");
      exit(EXIT_FAILURE);
    }
    if (sem_unlink(sema_name) == -1) {
      perror("sem_unlink");
      exit(EXIT_FAILURE);
    }
  }
  inline void Wait(void) {
    sem_wait(semPtr);
  }
  inline void Post(void) {
    sem_post(semPtr);
  }
  #else

 private:
  sem_t sem;

 public:
  inline void Init(int init_val) {
    if (sem_init(&sem, 0, init_val) != 0) {
      LOG(FATAL) << "Semaphore.Init: " << strerror(errno);
    }
  }
  inline void Destroy(void) {
    if (sem_destroy(&sem) != 0) {
      LOG(FATAL) << "Semaphore.Destroy: " << strerror(errno);
    }
  }
  inline void Wait(void) {
    if (sem_wait(&sem) != 0) {
      LOG(FATAL) << "Semaphore.Wait: " << strerror(errno);
    }
  }
  inline void Post(void) {
    if (sem_post(&sem) != 0) {
      LOG(FATAL) << "Semaphore.Post: " << strerror(errno);
    }
  }
  #endif
};

// mutex that works with pthread
class Mutex {
 public:
  inline void Init(void) {
    pthread_mutex_init(&mutex, NULL);
  }
  inline void Lock(void) {
    pthread_mutex_lock(&mutex);
  }
  inline void Unlock(void) {
    pthread_mutex_unlock(&mutex);
  }
  inline void Destroy(void) {
    pthread_mutex_destroy(&mutex);
  }

 private:
  friend class ConditionVariable;
  pthread_mutex_t mutex;
};

// conditional variable that uses pthread
class ConditionVariable {
 public:
  // initialize conditional variable
  inline void Init(void) {
    pthread_cond_init(&cond, NULL);
  }
  // destroy the thread
  inline void Destroy(void) {
    pthread_cond_destroy(&cond);
  }
  // wait on the conditional variable
  inline void Wait(Mutex *mutex) {
    pthread_cond_wait(&cond, &(mutex->mutex));
  }
  inline void Broadcast(void) {
    pthread_cond_broadcast(&cond);
  }
  inline void Signal(void) {
    pthread_cond_signal(&cond);
  }

 private:
  pthread_cond_t cond;
};

/*!\brief simple thread class */
class Thread {
 private:
  pthread_t thread;
 public :
  inline void Start(void * entry(void*), void *param) { // NOLINT(*)
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&thread, &attr, entry, param);
  }
  inline int Join(void) {
    void *status;
    return pthread_join(thread, &status);
  }
};
inline void ThreadExit(void *status) {
  pthread_exit(status);
}
}  // namespace utils
}  // namespace mshadow
#define MSHADOW_THREAD_PREFIX void *
#endif  // Linux
#endif  // MSHADOW_PS_THREAD_H_  NOLINT(*)
