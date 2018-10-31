/*!
 * Copyright (c) 2017 by Contributors
 * \file thread_group.h
 * \brief Thread and synchronization primitives and lifecycle management
 */
#ifndef DMLC_THREAD_GROUP_H_
#define DMLC_THREAD_GROUP_H_

#include <dmlc/concurrentqueue.h>
#include <dmlc/blockingconcurrentqueue.h>
#include <dmlc/logging.h>
#include <string>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#if defined(DMLC_USE_CXX14) || __cplusplus > 201103L  /* C++14 */
#include <shared_mutex>
#endif
#include <condition_variable>
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#endif

namespace dmlc {

/*!
 * \brief Simple manual-reset event gate which remains open after signalled
 */
class ManualEvent {
 public:
  ManualEvent() : signaled_(false) {}

  /*!
   * \brief Wait for the object to become signaled.  If the object
   * is already in the signaled state and reset() has not been called, then no wait will occur
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!signaled_) {
      condition_variable_.wait(lock);
    }
  }

  /*!
   * \brief Set this object's state to signaled (wait() will release or pass through)
   */
  void signal() {
    signaled_ = true;
    std::unique_lock<std::mutex> lk(mutex_);
    condition_variable_.notify_all();
  }

  /*!
   * \brief Manually reset this object's state to unsignaled (wait() will block)
   */
  void reset() {
    std::unique_lock<std::mutex> lk(mutex_);
    signaled_ = false;
  }

 private:
  /*! \brief Internal mutex to protect condition variable and signaled_ variable */
  std::mutex mutex_;
  /*! \brief Internal condition variable */
  std::condition_variable condition_variable_;
  /*! \brief lockfree signal state check */
  std::atomic<bool> signaled_;
};

#if defined(DMLC_USE_CXX14) || __cplusplus > 201103L  /* C++14 */
/*! \brief Mutex which can be read-locked and write-locked */
using SharedMutex = std::shared_timed_mutex;
/*! \brief Write lock, disallows both reads and writes */
using WriteLock = std::unique_lock<SharedMutex>;
/*! \brief Read lock, allows concurrent data reads */
using ReadLock = std::shared_lock<SharedMutex>;
#else
/*! \brief Standard mutex for C++ < 14 */
using SharedMutex = std::recursive_mutex;
/*! \brief Standard unique lock for C++ < 14 */
using WriteLock = std::unique_lock<SharedMutex>;
/*! \brief Standard unique lock for C++ < 14 */
using ReadLock = std::unique_lock<SharedMutex>;
#endif

/*!
 * \brief Thread lifecycle management group
 * \note See gtest unit tests Syc.* for a usage examples
 */
class ThreadGroup {
 public:
  /*!
   * \brief Lifecycle-managed thread (used by ThreadGroup)
   * \note See gtest unit tests Syc.* for a usage examples
   */
  class Thread {
   public:
    /*! \brief Shared pointer type for readability */
    using SharedPtr = std::shared_ptr<Thread>;

    /*!
     * \brief Constructor
     * \param threadName User-defined name of the thread. must be unique per ThreadGroup
     * \param owner The ThreadGroup object managing the lifecycle of this thread
     * \param thrd Optionally-assigned std::thread object associated with this Thread class
     */
    Thread(std::string threadName, ThreadGroup *owner, std::thread *thrd = nullptr)
      : name_(std::move(threadName))
        , thread_(thrd)
        , ready_event_(std::make_shared<ManualEvent>())
        , start_event_(std::make_shared<ManualEvent>())
        , owner_(owner)
        , shutdown_requested_(false)
        , auto_remove_(false) {
      CHECK_NOTNULL(owner);
    }

    /*!
     * \brief Destructor with cleanup
     */
    virtual ~Thread() {
      const bool self_delete = is_current_thread();
      if (!self_delete) {
        request_shutdown();
        internal_join(true);
      }
      WriteLock guard(thread_mutex_);
      if (thread_.load()) {
        std::thread *thrd = thread_.load();
        thread_ = nullptr;
        if (self_delete) {
          thrd->detach();
        }
        delete thrd;
      }
    }

    /*!
     * \brief Name of the thread
     * \return Pointer to the thread name's string
     * \note This shoul ndly be used as immediate for the sacope of the
     *       shared pointer pointing to this object
     */
    const char *name() const {
      return name_.c_str();
    }

    /*!
     * \brief Launch the given Thread object
     * \tparam StartFunction Function type for the thread 'main' function
     * \tparam Args Arguments to pass to the thread 'main' function
     * \param pThis Shared pointer for the managed thread to launch
     * \param autoRemove if true, automatically remove this Thread object from the
     *                   ThreadGroup owner upon exit
     * \param start_function The Thread's 'main' function
     * \param args Arguments to pass to the Thread's 'main' function
     * \return true if the thread was successfully created and added to the ThreadGroup
     *              If false is returned, the thread may have already been started, but if something
     *              went wrong (ie duplicte thread name for the ThreadGroup), then request_shutdown()
     *              will have been been called on the running thread
     */
    template<typename StartFunction, typename ...Args>
    static bool launch(std::shared_ptr<Thread> pThis,
                       bool autoRemove,
                       StartFunction start_function,
                       Args ...args);

    /*!
     * \brief Check if this class represents the currently running thread (self)
     * \return true if the current running thread belongs to this class
     */
    bool is_current_thread() const {
      ReadLock guard(thread_mutex_);
      return thread_.load() ? (thread_.load()->get_id() == std::this_thread::get_id()) : false;
    }

    /*!
     * \brief Signal to this thread that a thread shutdown/exit is requested.
     * \note This is a candidate for overrise in a derived class which may trigger shutdown
     *       by means other than a boolean (ie condition variable, SimpleManualkEvent, etc).
     */
    virtual void request_shutdown() {
      shutdown_requested_ = true;
    }

    /*!
     * \brief Check whether shutdown has been requested (request_shutdown() was called)
     * \return true if shutdown was requested.
     * \note This may be overriden to match an overriden to match an overriden 'request_shutdown()',
     *       for instance.
     */
    virtual bool is_shutdown_requested() const {
      return shutdown_requested_.load();
    }

    /*!
     * \brief Check whether the thread is set to auto-remove itself from the ThreadGroup owner
     *        when exiting
     * \return true if the thread will auto-remove itself from the ThreadGroup owner
     *        when exiting
     */
    bool is_auto_remove() const {
      return auto_remove_;
    }

    /*!
     * \brief Make the thread joinable (by removing the auto_remove flag)
     * \warning Care should be taken not to cause a race condition between this call
     *          and parallel execution of this thread auto-removing itself
     */
    void make_joinable() {
      auto_remove_ = false;
    }

    /*!
     * \brief Check whether the thread is joinable
     * \return true if the thread is joinable
     */
    bool joinable() const {
      ReadLock guard(thread_mutex_);
      if (thread_.load()) {
        CHECK_EQ(auto_remove_, false);
        // be checked by searching the group or exit event.
        return thread_.load()->joinable();
      }
      return false;
    }

    /*!
     * \brief Thread join
     * \note join() may not be called on auto-remove threads
     */
    void join() {
      internal_join(false);
    }

    /*!
     * \brief Get this thread's id
     * \return this thread's id
     */
    std::thread::id get_id() const {
      ReadLock guard(thread_mutex_);
      return thread_.load()->get_id();
    }

   private:
    /*!
     * \brief Internal join function
     * \param auto_remove_ok Whether to allow join on an auto-remove thread
     */
    void internal_join(bool auto_remove_ok) {
      ReadLock guard(thread_mutex_);
      // should be careful calling (or any function externally) this when in
      // auto-remove mode
      if (thread_.load() && thread_.load()->get_id() != std::thread::id()) {
        std::thread::id someId;
        if (!auto_remove_ok) {
          CHECK_EQ(auto_remove_, false);
        }
        CHECK_NOTNULL(thread_.load());
        if (thread_.load()->joinable()) {
          thread_.load()->join();
        } else {
          LOG(WARNING) << "Thread " << name_ << " ( "
                       << thread_.load()->get_id() << " ) not joinable";
        }
      }
    }

    /*!
     * \brief Thread bootstrapping and teardown wrapper
     * \tparam StartFunction Thread's "main" function
     * \tparam Args Argument types to be passed to the start_function
     * \param pThis Shared pointer to the Thread object to operate upon
     * \param start_function Thread's "main" function (i.e. passed to launch())
     * \param args Arguments to be passed to the start_function
     * \return The thread's return code
     */
    template <typename StartFunction, typename ...Args>
    static int entry_and_exit_f(std::shared_ptr<Thread> pThis,
                                StartFunction start_function,
                                Args... args);
    /*! \brief Thread name */
    std::string name_;
    /*! \brief Shared mutex for some thread operations */
    mutable SharedMutex thread_mutex_;
    /*! \brief Pointer to the stl thread object */
    std::atomic<std::thread *> thread_;
    /*! \brief Signaled when the thread is started and ready to execute user code */
    std::shared_ptr<ManualEvent> ready_event_;
    /*! \brief Thread will block after setting ready_event_ until start_event_ is signaled */
    std::shared_ptr<ManualEvent> start_event_;
    /*! \brief The ThreadGroup ownber managing this thread's lifecycle */
    ThreadGroup *owner_;
    /*! \brief Flag to determine if shutdown was requested. */
    std::atomic<bool> shutdown_requested_;
    /*!
     * \brief Whether to automatically remove this thread's object from the ThreadGroup when the
     *        thread exists (perform its own cleanup)
     */
    volatile bool auto_remove_;
  };

  /*!
   * \brief Constructor
   */
  inline ThreadGroup()
    : evEmpty_(std::make_shared<ManualEvent>()) {
    evEmpty_->signal();  // Starts out empty
  }

  /*!
   * \brief Destructor, perform cleanup. All child threads will be exited when this
   *        destructor completes
   */
  virtual ~ThreadGroup() {
    request_shutdown_all();
    join_all();
  }

  /*!
   * \brief Check if the current thread a member if this ThreadGroup
   * \return true if the current thread is a member of this thread group
   * \note This lookup involved a linear search, so for a large number of threads,
   *       is it not advised to call this function in a performance-sensitive area
   */
  inline bool is_this_thread_in() const {
    std::thread::id id = std::this_thread::get_id();
    ReadLock guard(m_);
    for (auto it = threads_.begin(), end = threads_.end(); it != end; ++it) {
      std::shared_ptr<Thread> thrd = *it;
      if (thrd->get_id() == id)
        return true;
    }
    return false;
  }

  /*!
   * \brief Check if the current thread is a member of this ThreadGroup
   * \param thrd The thread to search for
   * \return true if the given thread is a member of this ThreadGroup
   */
  inline bool is_thread_in(std::shared_ptr<Thread> thrd) const {
    if (thrd) {
      std::thread::id id = thrd->get_id();
      ReadLock guard(m_);
      for (auto it = threads_.begin(), end = threads_.end(); it != end; ++it) {
        std::shared_ptr<Thread> thrd = *it;
        if (thrd->get_id() == id)
          return true;
      }
      return false;
    } else {
      return false;
    }
  }

  /*!
   * \brief Add a Thread object to this thread group
   * \param thrd The thread to add to this ThreadGroup object
   * \return true if the given thread was added to this ThreadGroup
   */
  inline bool add_thread(std::shared_ptr<Thread> thrd) {
    if (thrd) {
      WriteLock guard(m_);
      auto iter = name_to_thread_.find(thrd->name());
      if (iter == name_to_thread_.end()) {
        name_to_thread_.emplace(std::make_pair(thrd->name(), thrd));
        CHECK_EQ(threads_.insert(thrd).second, true);
        evEmpty_->reset();
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Remove a Thread object from this thread group
   * \param thrd The thread to remove from this ThreadGroup object
   * \return true if the given thread was removed from this ThreadGroup
   */
  inline bool remove_thread(std::shared_ptr<Thread> thrd) {
    if (thrd) {
      WriteLock guard(m_);
      auto iter = threads_.find(thrd);
      if (iter != threads_.end()) {
        name_to_thread_.erase(thrd->name());
        threads_.erase(iter);
        if (threads_.empty()) {
          evEmpty_->signal();
        }
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Join all threads in this ThreadGroup
   * \note While it is not valid to call 'join' on an auto-remove thread, this function will
   *       wait for auto-remove threads to exit (waits for the ThreadGroup to become empty)
   */
  inline void join_all() {
    CHECK_EQ(!is_this_thread_in(), true);
    do {
      std::unique_lock<std::mutex> lk(join_all_mtx_);
      std::unordered_set<std::shared_ptr<Thread>> working_set;
      {
        ReadLock guard(m_);
        for (auto iter = threads_.begin(), e_iter = threads_.end(); iter != e_iter; ++iter) {
          if (!(*iter)->is_auto_remove()) {
            working_set.emplace(*iter);
          }
        }
      }
      // Where possible, prefer to do a proper join rather than simply waiting for empty
      // (easier to troubleshoot)
      while (!working_set.empty()) {
        std::shared_ptr<Thread> thrd;
        thrd = *working_set.begin();
        if (thrd->joinable()) {
          thrd->join();
        }
        remove_thread(thrd);
        working_set.erase(working_set.begin());
        thrd.reset();
      }
      // Wait for auto-remove threads (if any) to complete
    } while (0);
    evEmpty_->wait();
    CHECK_EQ(threads_.size(), 0);
  }

  /*!
   * \brief Call request_shutdown() on all threads in this ThreadGroup
   * \param make_all_joinable If true, remove all auto_remove flags from child threads
   */
  inline void request_shutdown_all(const bool make_all_joinable = true) {
    std::unique_lock<std::mutex> lk(join_all_mtx_);
    ReadLock guard(m_);
    for (auto &thread : threads_) {
      if (make_all_joinable) {
        thread->make_joinable();
      }
      thread->request_shutdown();
    }
  }

  /*!
   * \brief Return the number of threads in this thread group
   * \return Number of threads in this thread group
   */
  inline size_t size() const {
    ReadLock guard(m_);
    return threads_.size();
  }

  /*!
   * \brief Check if the ThreadGroup is empty
   * \return true if the ThreadGroup is empty
   */
  inline bool empty() const {
    ReadLock guard(m_);
    return threads_.size() == 0;
  }

  /*!
   * \brief Create and launch a new Thread object which will be owned by this ThreadGroup
   * \tparam StartFunction Function type for the thread 'main' function
   * \tparam ThreadType managedThreadclass type (in case it's derived, for instance)
   * \tparam Args Arguments to pass to the thread 'main' function
   * \param threadName Name if the thread. Must be unique for a ThreadGroup object
   * \param auto_remove If true, automatically remove this Thread object from the
   *                    ThreadGroup owner upon exit
   * \param start_function The Thread's 'main' function
   * \param args Arguments to pass to the Thread's 'main' function
   * \return true if the thread was successfully created and added to the ThreadGroup
   *              If false is returned, the thread may have already been started, but if something
   *              went wrong (ie duplicte thread name for the ThreadGroup), then request_shutdown()
   *              will have been been called on the running thread
   */
  template<typename StartFunction, typename ThreadType = Thread, typename ...Args>
  inline bool create(const std::string &threadName,
                     bool auto_remove,
                     StartFunction start_function,
                     Args... args) {
    typename ThreadType::SharedPtr newThread(new ThreadType(threadName, this));
    return Thread::launch(newThread, auto_remove, start_function, args...);
  }

  /*!
   * \brief Lookup Thread object by name
   * \param name Name of the thread to look up
   * \return A shared pointer to the Thread object
   */
  inline std::shared_ptr<Thread> thread_by_name(const std::string& name) {
    ReadLock guard(m_);
    auto iter = name_to_thread_.find(name);
    if (iter != name_to_thread_.end()) {
      return iter->second;
    }
    return nullptr;
  }

 private:
  /*! \brief ThreadGroup synchronization mutex */
  mutable SharedMutex m_;
  /*! \brief join_all/auto_remove synchronization mutex */
  mutable std::mutex join_all_mtx_;
  /*! \brief Set of threads owned and managed by this ThreadGroup object */
  std::unordered_set<std::shared_ptr<Thread>> threads_;
  /*! \brief Manual event which is signaled when the thread group is empty */
  std::shared_ptr<ManualEvent> evEmpty_;
  /*! \brief name->thread mapping */
  std::unordered_map<std::string, std::shared_ptr<Thread>> name_to_thread_;
};

/*!
 * \brief Blocking queue thread class
 * \tparam ObjectType Object type to queue
 * \tparam quit_item Object value to signify queue shutdown (ie nullptr for pointer type is common)
 * \note See gtest unit test Syc.ManagedThreadLaunchQueueThread for a usage example
 */
template<typename ObjectType, ObjectType quit_item>
class BlockingQueueThread : public ThreadGroup::Thread {
  using BQT = BlockingQueueThread<ObjectType, quit_item>;

 public:
  /*!
   * \brief Constructor
   * \param name Name for the blockin g queue thread. Must be unique for a specific ThreadGroup
   * \param owner ThreadGroup lifecycle manafger/owner
   * \param thrd Optionally attach an existing stl thread object
   */
  BlockingQueueThread(const std::string& name,
                      dmlc::ThreadGroup *owner,
                      std::thread *thrd = nullptr)
    : ThreadGroup::Thread(std::move(name), owner, thrd)
      , shutdown_in_progress_(false) {
  }


  /*!
   * \brief Destructor
   */
  ~BlockingQueueThread() override {
    // Call to parent first because we don't want to wait for the queue to empty
    ThreadGroup::Thread::request_shutdown();
    request_shutdown();
  }

  /*!
   * \brief Signal the thread that a shutdown is desired
   * \note Since consumer doesn't necessarily get items in order, we must wait for
   *       the queue to empty.
   *       This is generally a shutdown procedure and should not be called from
   *       a performance-sensitive area
   */
  void request_shutdown() override {
    shutdown_in_progress_ = true;
    while (queue_->size_approx() > 0 && !ThreadGroup::Thread::is_shutdown_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ThreadGroup::Thread::request_shutdown();
    queue_->enqueue(quit_item);
  }

  /*!
   * \brief Enqueue and item
   * \param item The item to enqueue
   */
  void enqueue(const ObjectType& item) {
    if (!shutdown_in_progress_) {
      queue_->enqueue(item);
    }
  }

  /*!
   * \brief Get the approximate size of the queue
   * \return The approximate size of the queue
   */
  size_t size_approx() const { return queue_->size_approx(); }

  /*!
   * \brief Launch to the 'run' function which will, in turn, call the class'
   *        'run' function, passing it the given 'secondary_function'
   *        for it to call as needed
   * \tparam SecondaryFunction Type of the secondary function for 'run' override
   *         to call as needed
   * \param pThis Pointer to the managed thread to launch
   * \param secondary_function secondary function for 'run' override to call as needed
   * \return true if thread is launched successfully and added to the ThreadGroup
   */
  template<typename SecondaryFunction>
  static bool launch_run(std::shared_ptr<BQT> pThis,
                         SecondaryFunction secondary_function) {
    return ThreadGroup::Thread::launch(pThis, true, [](std::shared_ptr<BQT> pThis,
                                                       SecondaryFunction secondary_function) {
                                         return pThis->run(secondary_function);
                                       },
                                       pThis, secondary_function);
  }

  /*!
   * \brief Thread's main queue processing function
   * \tparam OnItemFunction Function type to call when an item is dequeued
   * \param on_item_function Function to call when an item is dequeued
   * \return 0 if completed through a `quit_item`, nonzero if on_item_function requested an exit
   */
  template<typename OnItemFunction>
  inline int run(OnItemFunction on_item_function) {
    int rc = 0;
    do {
      ObjectType item;
      queue_->wait_dequeue(item);
      if (item == quit_item) {
        break;
      }
      rc = on_item_function(item);
      if (rc) {
        break;
      }
    } while (true);
    return rc;
  }

 private:
  /*! \brief The blocking queue associated with this thread */
  std::shared_ptr<dmlc::moodycamel::BlockingConcurrentQueue<ObjectType>> queue_ =
    std::make_shared<dmlc::moodycamel::BlockingConcurrentQueue<ObjectType>>();
  /*! \brief Whether shutdown request is in progress */
  std::atomic<bool> shutdown_in_progress_;
};

/*!
 * \brief Managed timer thread
 * \tparam Duration Duration type (ie seconds, microseconds, etc)
 */
template<typename Duration>
class TimerThread : public ThreadGroup::Thread {
  using ThreadGroup::Thread::is_shutdown_requested;

 public:
  /*!
   * \brief Constructor
   * \param name Name of the timer thread
   * \param owner ThreadGroup owner if the timer thread
   */
  TimerThread(const std::string& name, ThreadGroup *owner)
    : Thread(name, owner) {
  }

  /*!
   * \brief Destructor
   */
  ~TimerThread() override {
    request_shutdown();
  }

  /*!
   * \brief Launch to the 'run' function which will, in turn, call the class'
   *        'run' function, passing it the given 'secondary_function'
   *        for it to call as needed
   * \tparam SecondaryFunction Type of the secondary function for 'run' override
   *         to call as needed
   * \param pThis Pointer to the managed thread to launch
   * \param secondary_function secondary function for 'run' override to call as needed
   * \return true if thread is launched successfully and added to the ThreadGroup
   */
  template<typename SecondaryFunction>
  static bool launch_run(std::shared_ptr<TimerThread<Duration>> pThis,
                         SecondaryFunction secondary_function) {
    return ThreadGroup::Thread::launch(pThis, true, [](std::shared_ptr<TimerThread<Duration>> pThis,
                                                       SecondaryFunction secondary_function) {
                                         return pThis->run(secondary_function);
                                       },
                                       pThis, secondary_function);
  }

  /*!
   * \brief Start a given timer thread
   * \tparam Function Type of the timer function
   * \param timer_thread Thread object to perform the timer events
   * \param duration Duration between the end end of the timer function and the next timer event
   * \param function Function to call when the timer expires
   * \note Calling shutdown_requested() will cause the thread to exit the next time that the timer
   *       expires.
   */
  template<typename Function>
  static void start(std::shared_ptr<TimerThread> timer_thread,
                    Duration duration,
                    Function function) {
    timer_thread->duration_ = duration;
    launch_run(timer_thread, function);
  }

  /*!
   * \brief Internal timer execution function
   * \tparam OnTimerFunction Type of function to call each time the timer expires
   * \param on_timer_function Function to call each time the timer expires
   * \return Exit code of the thread
   */
  template<typename OnTimerFunction>
  inline int run(OnTimerFunction on_timer_function) {
    int rc = 0;
    while (!is_shutdown_requested()) {
      std::this_thread::sleep_for(duration_);
      if (!is_shutdown_requested()) {
        rc = on_timer_function();
      }
    }
    return rc;
  }

 private:
  Duration duration_;
};

/*
 * Inline functions - see declarations for usage
 */
template <typename StartFunction, typename ...Args>
inline int ThreadGroup::Thread::entry_and_exit_f(std::shared_ptr<Thread> pThis,
                                                 StartFunction start_function,
                                                 Args... args) {
  int rc;
  if (pThis) {
    // Signal launcher that we're up and running
    pThis->ready_event_->signal();
    // Wait for launcher to be ready for us to start
    pThis->start_event_->wait();
    // Reset start_event_ for possible reuse
    pThis->start_event_->reset();  // Reset in case it needs to be reused
    // If we haven't been requested to shut down prematurely, then run the desired function
    if (!pThis->is_shutdown_requested()) {
      rc = start_function(args...);
    } else {
      rc = -1;
    }
    // If we're set up as auto-remove, then remove this thread from the thread group
    if (pThis->is_auto_remove()) {
      pThis->owner_->remove_thread(pThis);
    }
    // Release this thread shared pinter. May or may not be the last reference.
    pThis.reset();
  } else {
    LOG(ERROR) << "Null pThis thread pointer";
    rc = EINVAL;
  }
  return rc;
}

template<typename StartFunction, typename ...Args>
inline bool ThreadGroup::Thread::launch(std::shared_ptr<Thread> pThis,
                                        bool autoRemove,
                                        StartFunction start_function,
                                        Args ...args) {
  WriteLock guard(pThis->thread_mutex_);
  CHECK_EQ(!pThis->thread_.load(), true);
  CHECK_NOTNULL(pThis->owner_);
  // Set auto remove
  pThis->auto_remove_ = autoRemove;
  // Create the actual stl thread object
  pThis->thread_ = new std::thread(Thread::template entry_and_exit_f<
                                     StartFunction, Args...>,
                                   pThis,
                                   start_function,
                                   args...);
  // Attempt to add the thread to the thread group (after started, since in case
  // something goes wrong, there's not a zombie thread in the thread group)
  if (!pThis->owner_->add_thread(pThis)) {
    pThis->request_shutdown();
    LOG(ERROR) << "Duplicate thread name within the same thread group is not allowed";
  }
  // Wait for the thread to spin up
  pThis->ready_event_->wait();
  // Signal the thgread to continue (it will check its shutdown status)
  pThis->start_event_->signal();
  // Return if successful
  return pThis->thread_.load() != nullptr;
}

/*!
 * \brief Utility function to easily create a timer
 * \tparam Duration Duration type (i.e. std::chrono::milliseconds)
 * \tparam TimerFunction Function to call each time the timer expires
 * \param timer_name Name of the timer. Must be unique per ThreadGroup object
 * \param duration Duration of the timer between calls to timer_function
 * \param owner ThreadGroup owner of the timer
 * \param timer_function Function to call each time the timer expires
 * \return true if the timer was successfully created
 */
template<typename Duration, typename TimerFunction>
inline bool CreateTimer(const std::string& timer_name,
                        const Duration& duration,
                        ThreadGroup *owner,
                        TimerFunction timer_function) {
  std::shared_ptr<dmlc::TimerThread<Duration>> timer_thread =
    std::make_shared<dmlc::TimerThread<Duration>>(timer_name, owner);
  dmlc::TimerThread<Duration>::start(timer_thread, duration, timer_function);
  return timer_thread != nullptr;
}
}  // namespace dmlc

#endif  // DMLC_THREAD_GROUP_H_
