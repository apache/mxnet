/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine.cc
 * \brief implements base threaded engine.
 * \author Yutian Li
 */
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <utility>
#include "./threaded_engine.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace engine {

#if ENGINE_DEBUG
std::atomic<std::size_t> OprBlock::counter{0};
std::atomic<std::size_t> VersionedVarBlock::counter{0};
std::atomic<std::size_t> ThreadedVar::counter{0};
std::atomic<std::size_t> ThreadedOpr::counter{0};
#endif  // ENGINE_DEBUG

ThreadedVar::ThreadedVar(VersionedVarBlock* head) : head_{head} {
#if ENGINE_DEBUG
  LOG(INFO) << __func__ << " " << ++counter;
#endif  // ENGINE_DEBUG
}

inline void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{m_};
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    CHECK_GE(num_pending_reads_, 0);
    // STATE CHANGE
    ++num_pending_reads_;
    // decrease wait counter
    opr_block->decr_wait();
  } else {
    auto&& new_var_block = VersionedVarBlock::New();
    assert(head_->next == nullptr);
    assert(head_->trigger == nullptr);
    assert(head_->write == false);
    // append things to next.
    head_->next = new_var_block;
    head_->trigger = opr_block;
    head_ = new_var_block;
  }
}

inline void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
  auto&& new_var_block = VersionedVarBlock::New();
  std::lock_guard<std::mutex> lock{m_};
  // invariant.
  assert(head_->next == nullptr);
  assert(head_->trigger == nullptr);
  assert(head_->write == false);
  // attach to head.
  head_->next = new_var_block;
  head_->trigger = opr_block;
  head_->write = true;

  // check if it is ready to write
  if (pending_write_ == nullptr) {
    // invariant: is_ready_to_read()
    pending_write_ = head_;
    CHECK_GE(num_pending_reads_, 0);
    if (num_pending_reads_ == 0) {
      // STATE CHANGE
      opr_block->decr_wait();
      num_pending_reads_ = kWriteTriggered;
    }
  } else {
    CHECK_NE(num_pending_reads_, 0);
  }
  head_ = new_var_block;
}

template <typename Dispatcher>
inline void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
  OprBlock *trigger = nullptr;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{m_};
    CHECK_GT(num_pending_reads_, 0);

    if (--num_pending_reads_ == 0) {
      if (pending_write_ != nullptr) {
        // STATE CHANGE
        trigger = pending_write_->trigger;
        num_pending_reads_ = kWriteTriggered;
      }
    }
  }
  if (trigger != nullptr && trigger->decr_wait() == 0) {
    dispatcher(trigger);
  }
}

template <typename Dispatcher>
inline bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  // this is lock scope
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  OprBlock* trigger_write = nullptr;
  {
    std::lock_guard<std::mutex> lock{m_};
    // invariants
    assert(head_->next == nullptr);
    assert(pending_write_ != nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // really delete
    if (to_delete_) {
      VersionedVarBlock *head = pending_write_->next;
      VersionedVarBlock::Delete(pending_write_);
      assert(head_ == head);
      VersionedVarBlock::Delete(head);
      return true;
    }
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    // reset to 0 pending reads
    num_pending_reads_ = 0;
    while (end_of_read_chain != head_ &&
           end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    if (end_of_read_chain == head_) {
      pending_write_ = nullptr;
    } else {
      // check if there is pending reads, if not trigger write
      assert(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        // mark write as already actived in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not reply ont the two variables.
  // The linked list \in [old_pending_write, end_of_read_chain)
  // is already detached from this Var.
  // So it is safe to modify these
  VersionedVarBlock *cur_head = old_pending_write->next;
  VersionedVarBlock::Delete(old_pending_write);
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (cur_head->trigger->decr_wait() == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head = cur_head->next;
    assert(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  if (trigger_write != nullptr && trigger_write->decr_wait() == 0) {
    dispatcher(trigger_write);
  }
  return false;
}

inline void ThreadedVar::SetToDelete() {
  std::lock_guard<std::mutex> lock{m_};
  to_delete_ = true;
}

inline bool ThreadedVar::ready_to_read() {
  std::lock_guard<std::mutex> lock{m_};
  return this->is_ready_to_read();
}

// implementation of threaded engine
ThreadedVar* ThreadedEngine::NewVariable() {
  return ThreadedVar::New(VersionedVarBlock::New());
}

ThreadedOpr* ThreadedEngine::NewOperator(
    ThreadedEngine::AsyncFn fn,
    std::vector<VarHandle> const& const_vars,
    std::vector<VarHandle> const& mutable_vars,
    FnProperty prop) {
  auto ret = ThreadedOpr::New();
  ret->fn = fn;
  ret->prop = prop;
  ret->const_vars.resize(const_vars.size());
  ret->mutable_vars.resize(mutable_vars.size());
  std::transform(const_vars.begin(), const_vars.end(),
                 ret->const_vars.begin(), ThreadedVar::CastFromBase);
  std::transform(mutable_vars.begin(), mutable_vars.end(),
                 ret->mutable_vars.begin(), ThreadedVar::CastFromBase);
  if (ENGINE_DEBUG != 0) {
    CheckDuplicate(const_vars, mutable_vars);
  }
  return ret;
}

void ThreadedEngine::CheckDuplicate(std::vector<VarHandle> const& const_vars,
                                    std::vector<VarHandle> const& mutable_vars) {
  // Check for duplicates.
  auto use = const_vars;
  auto mutate = mutable_vars;
  auto use_size = use.size();
  auto mutate_size = mutate.size();
  std::sort(use.begin(), use.end());
  std::sort(mutate.begin(), mutate.end());
  for (std::size_t i = 0; i < use_size; ++i) {
    if (i != 0 && use.at(i) == use.at(i - 1)) {
      LOG(FATAL) << "duplicate items found in `const_vars`";
    }
  }
  for (std::size_t i = 0; i < mutate_size; ++i) {
    if (i != 0 && mutate.at(i) == mutate.at(i - 1)) {
      LOG(FATAL) << "duplicate items found in `mutable_vars`";
    }
  }
  std::size_t j = 0;
  for (std::size_t i = 0; i < use_size; ++i) {
    while (j < mutate_size && mutate.at(j) < use.at(i)) {
      ++j;
    }
    if (j == mutate_size) {
      break;
    }
    if (mutate.at(j) == use.at(i)) {
      LOG(FATAL)
          << "duplicate items found between `const_vars` and `mutable_vars`";
    }
  }
}

void ThreadedEngine::DeleteOperator(OprHandle op) {
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  std::vector<VarHandle> deps;
  deps.reserve(threaded_opr->const_vars.size() +
               threaded_opr->mutable_vars.size());
  deps.insert(deps.end(),
              threaded_opr->const_vars.begin(),
              threaded_opr->const_vars.end());
  deps.insert(deps.end(),
              threaded_opr->mutable_vars.begin(),
              threaded_opr->mutable_vars.end());
  this->PushSync([threaded_opr](RunContext) {
      ThreadedOpr::Delete(threaded_opr);
    }, Context::CPU(), {}, deps, FnProperty::kAsync);
}

void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority) {
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  OprBlock* opr_block = OprBlock::New();
  opr_block->opr = threaded_opr;

  opr_block->wait.store(static_cast<int>(
      threaded_opr->const_vars.size() +
      threaded_opr->mutable_vars.size() + 1));
  opr_block->ctx = exec_ctx;
  opr_block->priority = priority;
  ++pending_;
  // Add read dependencies.
  for (auto&& i : threaded_opr->const_vars) {
    i->AppendReadDependency(opr_block);
  }
  // Add write dependencies.
  for (auto&& i : threaded_opr->mutable_vars) {
    i->AppendWriteDependency(opr_block);
  }
  if (opr_block->decr_wait() == 0) {
    this->PushToExecute(opr_block, true);
  }
}

void ThreadedEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                               std::vector<VarHandle> const& const_vars,
                               std::vector<VarHandle> const& mutable_vars,
                               FnProperty prop, int priority) {
  ThreadedOpr *opr = NewOperator(fn, const_vars, mutable_vars, prop);
  opr->temporary = true;
  Push(opr, exec_ctx, priority);
}

void ThreadedEngine::DeleteVariable(SyncFn delete_fn,
                                    Context exec_ctx,
                                    VarHandle var) {
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  this->PushSync([delete_fn, threaded_var](RunContext ctx) {
      // Mark variable as orphan,
      // so during `ThreadedEngine::OnComplete` it could be recycled.
      threaded_var->SetToDelete();
      delete_fn(ctx);
    }, exec_ctx, {}, {var}, FnProperty::kAsync);
}

void ThreadedEngine::WaitForVar(VarHandle var) {
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  if (threaded_var->ready_to_read()) return;
  if (engine_info_) {
    LOG(INFO) << "Wait for " << threaded_var;
    debug_wait_var_ = threaded_var;
  }
  std::atomic<bool> done{false};
  this->PushSync([this, &done](RunContext) {
      if (engine_info_) {
        LOG(INFO) << "Sync is executed";
      }
      {
        std::unique_lock<std::mutex> lock{finished_m_};
        done.store(true);
      }
      finished_cv_.notify_all();
      if (engine_info_) {
        LOG(INFO) << "Sync is notified";
      }
    }, Context::CPU(), {var}, {}, FnProperty::kNormal);
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    finished_cv_.wait(lock, [this, &done]() {
        return done.load() || kill_.load();
      });
  }
}

void ThreadedEngine::WaitForAll() {
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() {
      return pending_.load() == 0 || kill_.load();
    });
}

inline void ThreadedEngine::OnComplete(ThreadedOpr* threaded_opr) {
  // Mark complete for read variables
  for (auto&& i : threaded_opr->const_vars) {
    i->CompleteReadDependency([this](OprBlock* opr) {
        this->PushToExecute(opr, false);
      });
  }
  // Mark complete for write variables.
  for (auto&& i : threaded_opr->mutable_vars) {
    bool debug_info = (engine_info_ && debug_wait_var_ == i);
    if (debug_info) {
      LOG(INFO) << "Complete write dep for " << i;
    }
    bool to_delete = i->CompleteWriteDependency(
        [this, debug_info](OprBlock* opr) {
          if (debug_info) {
            LOG(INFO) << "PushToExecute " << opr;
            debug_push_opr_ = opr;
          }
          this->PushToExecute(opr, false);
          if (debug_info) {
            LOG(INFO) << "Fin PushToExecute " << opr;
          }
        });
    if (to_delete) {
      ThreadedVar::Delete(i);
    }
  }
  int npending;
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    npending = --pending_;
  }
  CHECK_GE(npending, 0);
  if (npending == 0) {
    // no need to grab lock when notify.
    finished_cv_.notify_all();
  }

  // delte operator if it is temperory
  if (threaded_opr->temporary) {
    ThreadedOpr::Delete(threaded_opr);
  }
}

void ThreadedEngine::OnCompleteStatic(
    Engine *engine, void *threaded_opr) {
  static_cast<ThreadedEngine*>(engine)->OnComplete(
      static_cast<ThreadedOpr*>(threaded_opr));
}

}  // namespace engine
}  // namespace mxnet
