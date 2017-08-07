/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine.h
 * \brief Implements base class of threaded engine
 *    that tracks the dependency and pushes actions to execute.
 * \author Yutian Li
 */
#ifndef MXNET_ENGINE_THREADED_ENGINE_H_
#define MXNET_ENGINE_THREADED_ENGINE_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <vector>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include "./engine_impl.h"
#include "./profiler.h"
#include "../common/object_pool.h"

namespace mxnet {
namespace engine {

// Define helper macros for debug information.
#if ENGINE_DEBUG
#define DEFINE_ENGINE_DEBUG_INFO(Type)                          \
  static std::atomic<std::size_t> counter;                      \
  Type() { LOG(INFO) << __func__ << " " << ++counter; }         \
  ~Type() { LOG(INFO) << __func__ << " " << --counter; }
#else
#define DEFINE_ENGINE_DEBUG_INFO(Type)
#endif

// Forward declarations
struct ThreadedOpr;

/*!
 * \brief Operation block in the scheduler.
 *  Each OprBlock corresponds to an operation pushed to the engine.
 */
struct OprBlock : public common::ObjectPoolAllocatable<OprBlock> {
  /*!
   * \brief wait number of pending tasks this OprBlock is waiting for.
   */
  std::atomic<int> wait{0};
  /*! \brief Pointer to information on performing real operation */
  ThreadedOpr* opr{nullptr};
  /*! \brief The context this operator */
  Context ctx;
  /*! \brief priority of the function */
  int priority;
  /*! \brief indicate whether to profile this operator */
  bool profiling{false};
  /*! \brief operator execution statistics */
  OprExecStat *opr_stat;
  // define possible debug information
  DEFINE_ENGINE_DEBUG_INFO(OprBlock);
  /*!
   * \brief call this function to decrease the wait counter.
   * \return the wait counter after the decreasement.
   */
  inline int decr_wait() {
    // chack invariant, avoid over trigger
    int ret = --wait;
    CHECK_GE(ret, 0);
    return ret;
  }
};  // struct OprBlock

/*!
 * \brief VersionedVarBlock that corresponding to a variable version.
 *  This is a basic unit of LinkedList in the ThreadedVar.
 */
struct VersionedVarBlock
    : public common::ObjectPoolAllocatable<VersionedVarBlock> {
  /*! \brief next block in the LinkedList */
  VersionedVarBlock* next{nullptr};
  /*! \brief the operation this block triggers */
  OprBlock* trigger{nullptr};
  /*! \brief whether this operation is a write(mutate) operation. */
  bool write{false};
  /*! \brief define possible debug information */
  DEFINE_ENGINE_DEBUG_INFO(VersionedVarBlock);
};  // struct VersionedVarBlock

/*!
 * \brief Variable implementation.
 *  Each ThreadedVar is a linked list(queue) of operations to be performed.
 */
class ThreadedVar final : public Var,
                          public common::ObjectPoolAllocatable<ThreadedVar> {
 public:
  /*!
   * \brief constructor
   * \param head head block of the LinkedList,
   *             need to be initialized with next==nullptr and trigger=nullptr.
   */
  explicit ThreadedVar(VersionedVarBlock* head);
  /*!
   * \brief Schedule a read operation on this variable.
   *  If the opr_block can be runed right away,
   *  the wait counter of opr_block will be decreased.
   *  Otherwise, the opr_block will be added to waiting queue.
   * \param opr_block The operation to be scheduled.
   */
  inline void AppendReadDependency(OprBlock* opr_block);
  /*!
   * \brief Schedule a write operation on this variable.
   *  If the opr_block can be runed right away,
   *  the wait counter of opr_block will be decreased.
   *  Otherwise, the opr_block will be added to waiting queue.
   * \param opr_block The operation to be scheduled.
   */
  inline void AppendWriteDependency(OprBlock* opr_block);
  /*!
   * \brief A read operation is completed on this variable.
   *  This function may trigger subsequent waiting operations on this variable.
   *
   * \param dispatcher the function called to trigger the operation,
   *            when all of its dependencies are satiesfied.
   * \tparam Dispatcher the function called to trigger an operation.
   */
  template <typename Dispatcher>
  inline void CompleteReadDependency(Dispatcher dispatcher);
  /*!
   * \brief A write operation is completed on this variable.
   *  This function may trigger subsequent waiting operations on this variable.
   *
   * \param dispatcher the function called to trigger the operation,
   *            when all of its dependencies are satiesfied.
   * \tparam Dispatcher the function called to trigger an operation.
   * \return to_delete, whether this Variable can be deleted after this functin.
   */
  template <typename Dispatcher>
  inline bool CompleteWriteDependency(Dispatcher dispatcher);
  /*! \brief Mark this variable to be deleted. */
  inline void SetToDelete();
  /*! \return whether this variable is ready to read. */
  inline bool ready_to_read();
  /*!
   * \brief Cast a Var pointer to ThreadedVar pointer
   * \param ptr pointer from base.
   * \return a casted pointer.
   */
  inline static ThreadedVar* CastFromBase(Var* ptr) {
    return ptr->Cast<ThreadedVar>();
  }
  // code for debug.
#if ENGINE_DEBUG
  static std::atomic<std::size_t> counter;
  ~ThreadedVar() { LOG(INFO) << __func__ << " " << --counter; }
#endif  // ENGINE_DEBUG

 private:
  // TODO(hotpxl) change this to spinlock for faster runtime
  // TODO(hotpxl) consider rename head
  /*! \brief inetrnal mutex of the ThreadedVar */
  std::mutex m_;
  /*!
   * \brief number of pending reads operation in the variable.
   *  will be marked as -1 when there is a already triggered pending write.
   */
  int num_pending_reads_{0};
  /*!
   * \brief Points to the last VersionedVarBlock in the queue.
   *  head_ always points to a empty VersionedVarBlock.
   *  So when we want to append an operation to the queue:
   *    1) update head_->trigger to be new op
   *    2) update head_->next to be a new VersionedVarBlock
   *    3) move head to head->next.
   */
  VersionedVarBlock* head_{nullptr};
  /*!
   * \brief The pointer to next write to perform.
   *  This pointer will only be updated when the write completes.
   *  This is actually the head(oldest operation) in the queue.
   */
  VersionedVarBlock* pending_write_{nullptr};
  /*!
   * \brief If true, delete after operation completes.
   */
  bool to_delete_{false};
  /*! \brief special const on num_pending_reads_ to mark write being triggered */
  static constexpr int kWriteTriggered = -1;
  /*!
   * \brief derived invariant of ready to ready, without lock.
   * \return whether the current variable is ready to read.
   */
  inline bool is_ready_to_read() const {
    return pending_write_ == nullptr;
  }
};  // struct ThreadedVar

/*!
 * \brief Operator used in ThreadedEngine.
 */
struct ThreadedOpr final : public Opr,
                           public common::ObjectPoolAllocatable<ThreadedOpr> {
  /*! \brief The function to be invoked each time. */
  Engine::AsyncFn fn;
  /*! \brief The variable this operation will read from. */
  std::vector<ThreadedVar*> const_vars;
  /*! \brief The variable this operation will mutate. */
  std::vector<ThreadedVar*> mutable_vars;
  /*! \brief The property of the operator */
  FnProperty prop;
  /*! \brief The name of the operator */
  const char* opr_name{nullptr};
  /*!
   * \brief Whether this is an temporary operator
   *        that can be deleted right after the operation completed.
   */
  bool temporary{false};
  /*!
   * \brief Cast a Opr pointer to ThreadedOpr pointer
   * \param ptr pointer from base.
   * \return a casted pointer.
   */
  inline static ThreadedOpr* CastFromBase(Opr* ptr) {
    return ptr->Cast<ThreadedOpr>();
  }
  // define possible debug information
  DEFINE_ENGINE_DEBUG_INFO(ThreadedOpr);
};  // struct ThreadedOpr

/*!
 * \brief Base class of all ThreadedEngine.
 *  This class implements a thread safe version of engine.
 *  The engine tracks the dependencies, and will call PushToExecute
 *  to execute a specific task.
 *
 *  Subclass can implement PushToExecute to design specific
 *  execution policy for the tasks.
 */
class ThreadedEngine : public Engine {
 public:
  // implementing all the functions from Engine.
  ThreadedVar* NewVariable() override;
  ThreadedOpr* NewOperator(AsyncFn fn,
                           std::vector<VarHandle> const& const_vars,
                           std::vector<VarHandle> const& mutable_vars,
                           FnProperty prop = FnProperty::kNormal,
                           const char* opr_name = nullptr) override;
  void DeleteOperator(OprHandle op) override;
  void Push(OprHandle op, Context exec_ctx, int priority = 0, bool profiling = false) override;
  void PushAsync(AsyncFn exec_fun, Context exec_ctx,
                 std::vector<VarHandle> const& const_vars,
                 std::vector<VarHandle> const& mutable_vars,
                 FnProperty prop = FnProperty::kNormal,
                 int priority = 0,
                 const char* opr_name = nullptr) override;
  void DeleteVariable(SyncFn delete_fn, Context exec_ctx, VarHandle var) override;
  void WaitForVar(VarHandle var) override;
  void WaitForAll() override;
  void NotifyShutdown() override {
    shutdown_phase_.store(true);
  }

  ThreadedEngine() {
    engine_info_ = dmlc::GetEnv("MXNET_ENGINE_INFO", false);

    objpool_opr_ref_    = common::ObjectPool<ThreadedOpr>::_GetSharedRef();
    objpool_blk_ref_    = common::ObjectPool<OprBlock>::_GetSharedRef();
    objpool_varblk_ref_ = common::ObjectPool<VersionedVarBlock>::_GetSharedRef();
    objpool_var_ref_    = common::ObjectPool<ThreadedVar>::_GetSharedRef();
  }
  ~ThreadedEngine() {
    {
      std::unique_lock<std::mutex> lock{finished_m_};
      kill_.store(true);
    }
    finished_cv_.notify_all();
  }

 protected:
  /*!
   * \brief Push the opr block to execution queue to be executed.
   *  This function is implemented by the corresponding subclass
   *  for specific policy.
   *
   * \param opr_block The operator block.
   * \param pusher_thread whether the caller is the thread that calls push
   */
  virtual void PushToExecute(OprBlock* opr_block, bool pusher_thread) = 0;
  /*!
   * \brief Call this function to actually execute an opr_block
   *  This function also deletes the opr_block after execution.
   * \param run_ctx runtime context used to execute the function.
   * \param opr_block the opr_block to be executed and deleted.
   */
  void ExecuteOprBlock(RunContext run_ctx, OprBlock *opr_block) {
    ThreadedOpr* threaded_opr = opr_block->opr;
#if MXNET_USE_PROFILER
    if (opr_block->profiling && threaded_opr->opr_name) {
      const Context& ctx = opr_block->ctx;
      opr_block->opr_stat = Profiler::Get()->AddOprStat(ctx.dev_type, ctx.dev_id);
      uint64_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
      opr_block->opr_stat->thread_id = id;
      strncpy(opr_block->opr_stat->opr_name,
        threaded_opr->opr_name,
        sizeof(opr_block->opr_stat->opr_name) - 1);
      // record operator start timestamp
      SetOprStart(opr_block->opr_stat);
    }
#endif
    CallbackOnComplete callback = this->CreateCallback(
        ThreadedEngine::OnCompleteStatic, opr_block);
    bool debug_info = (engine_info_ && debug_push_opr_ == opr_block);
    if (debug_info) {
      LOG(INFO) << "ExecuteOprBlock " << opr_block
                << "shutdown_phase=" << shutdown_phase_;
    }
    if (!shutdown_phase_) {
      try {
        if (debug_info) {
          LOG(INFO) << "ExecuteOprFn ";
        }
        threaded_opr->fn(run_ctx, callback);
        if (debug_info) {
          LOG(INFO) << "Fin ExecuteOprFn ";
        }
      } catch(dmlc::Error &e) {
        std::string what = e.what();
        if (what.find("driver shutting down") == std::string::npos &&
            !shutdown_phase_) {
          LOG(FATAL) << e.what() << "\n" <<
            "An fatal error occurred in asynchronous engine operation. "
            "If you do not know what caused this error, "
            "you can try set environment variable MXNET_ENGINE_TYPE "
            "to NaiveEngine and run with debugger (i.e. gdb). "
            "This will force all operations to be synchronous and "
            "backtrace will give you the series of calls that lead "
            "to this error. Remember to set MXNET_ENGINE_TYPE back to "
            "empty after debugging.";
        }
      }
    } else {
      callback();
    }
  }

 private:
  /*!
   * \brief check if thee is duplication in const_vars and mutable_vars.
   * \param const_vars the variables to read from.
   * \param mutable_vars the variables to mutate.
   */
  void CheckDuplicate(std::vector<VarHandle> const& const_vars,
                      std::vector<VarHandle> const& mutable_vars);
  /*!
   * \brief Callback on operation completion.
   *
   * On operation completion, this will trigger subsequent operations.
   */
  inline void OnComplete(ThreadedOpr* threaded_opr);
  // callback to the threaded engine
  static void OnCompleteStatic(Engine *engine, void *threaded_opr);
  /*!
   * \brief Number of pending operations.
   */
  std::atomic<int> pending_{0};
  /*! \brief whether we want to kill the waiters */
  std::atomic<bool> kill_{false};
  /*! \brief whether it is during shutdown phase*/
  std::atomic<bool> shutdown_phase_{false};
  /*!\brief show more information from engine actions */
  bool engine_info_{false};
  /*! \brief debug information about wait for var. */
  std::atomic<ThreadedVar*> debug_wait_var_{nullptr};
  /*! \brief debug information about wait for var. */
  std::atomic<OprBlock*> debug_push_opr_{nullptr};
  /*!
   * \brief Mutex and condition_variable,
   *  used to Notify waits for single or all variables.
   */
  std::mutex finished_m_;
  std::condition_variable finished_cv_;

  /*!
   * \brief Holding a shared_ptr to the object pool to prevent it from being destructed too early
   * See also #309 (https://github.com/dmlc/mxnet/issues/309)
   */
  std::shared_ptr<common::ObjectPool<ThreadedOpr> >       objpool_opr_ref_;
  std::shared_ptr<common::ObjectPool<OprBlock> >          objpool_blk_ref_;
  std::shared_ptr<common::ObjectPool<VersionedVarBlock> > objpool_varblk_ref_;
  std::shared_ptr<common::ObjectPool<ThreadedVar> >       objpool_var_ref_;
  /*!
   * \brief Disallow copy construction and assignment.
   */
  DISALLOW_COPY_AND_ASSIGN(ThreadedEngine);
};  // class ThreadedEngine

}  // namespace engine
}  // namespace mxnet
#endif  // MXNET_ENGINE_THREADED_ENGINE_H_
