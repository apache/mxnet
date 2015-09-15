/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine.cc
 * \brief implements base threaded engine.
 * \author Yutian Li
 */
#include "threaded_engine.h"
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <utility>
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

void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{m_};
  if (ready_to_read_) {
    assert(pending_write_ == nullptr);
    ++num_pending_reads_;
    --opr_block->wait;
  } else {
    auto&& new_var_block = VersionedVarBlock::New();
    assert(head_->next == nullptr);
    assert(head_->trigger == nullptr);
    assert(head_->write == false);
    head_->next = new_var_block;
    head_->trigger = opr_block;
    head_ = new_var_block;
  }
}

void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
  std::lock_guard<std::mutex> lock{m_};
  auto&& new_var_block = VersionedVarBlock::New();
  head_->next = new_var_block;
  head_->trigger = opr_block;
  head_->write = true;
  if (ready_to_read_) {
    // Raise `num_pending_reads_` temporarily to avoid premature triggering.
    ++num_pending_reads_;
    pending_write_ = head_;
    if (--num_pending_reads_ == 0) {
      --opr_block->wait;
    }
    ready_to_read_ = false;
  }
  head_ = new_var_block;
}

template <typename Dispatcher>
void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
  bool trigger = false;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{m_};
    if (--num_pending_reads_ == 0) {
      if (pending_write_ != nullptr && --pending_write_->trigger->wait == 0) {
        trigger = true;
      }
    }
  }
  if (trigger) {
    dispatcher(pending_write_->trigger);
  }
}

template <typename Dispatcher>
bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  bool trigger_write = false;
  {
    // this is lock scope
    std::lock_guard<std::mutex> lock{m_};
    assert(ready_to_read_ == false);
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    assert(num_pending_reads_ == 0);
    while (end_of_read_chain->next != nullptr &&
           end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    // check the states
    if (end_of_read_chain->next == nullptr) {
      ready_to_read_ = true;
      pending_write_ = nullptr;
    } else {
      assert(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        trigger_write = true;
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
  if (to_delete_) {
    assert(cur_head->next == nullptr);
    VersionedVarBlock::Delete(cur_head);
    return true;
  }
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (--cur_head->trigger->wait == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head = cur_head->next;
    assert(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  // Be careful, do not use pending_write_  or num_pending_reads_ here.
  // As they can change, use end_of_read_chain
  if (trigger_write) {
    if (--end_of_read_chain->trigger->wait == 0) {
      dispatcher(end_of_read_chain->trigger);
    }
  }
  return false;
}

void ThreadedVar::SetToDelete() {
  std::lock_guard<std::mutex> lock{m_};
  to_delete_ = true;
}

bool ThreadedVar::ready_to_read() {
  std::lock_guard<std::mutex> lock{m_};
  return ready_to_read_;
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
    }, Context(), {}, deps, FnProperty::kAsync);
}

void ThreadedEngine::Push(OprHandle op, Context exec_ctx) {
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  OprBlock* opr_block = OprBlock::New();
  opr_block->opr = threaded_opr;
  opr_block->wait.store(threaded_opr->const_vars.size() +
                        threaded_opr->mutable_vars.size() + 1);
  opr_block->ctx = exec_ctx;
  ++pending_;
  // Add read dependencies.
  for (auto&& i : threaded_opr->const_vars) {
    i->AppendReadDependency(opr_block);
  }
  // Add write dependencies.
  for (auto&& i : threaded_opr->mutable_vars) {
    i->AppendWriteDependency(opr_block);
  }
  if (--opr_block->wait == 0) {
    this->PushToExecute(opr_block, true);
  }
}

void ThreadedEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                               std::vector<VarHandle> const& const_vars,
                               std::vector<VarHandle> const& mutable_vars,
                               FnProperty prop) {
  ThreadedOpr *opr = NewOperator(fn, const_vars, mutable_vars, prop);
  opr->temporary = true;
  Push(opr, exec_ctx);
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
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    std::atomic<bool> done{false};
    this->PushSync([this, &done](RunContext) {
        std::unique_lock<std::mutex> lock{finished_m_};
        done.store(true);
        finished_cv_.notify_all();
      }, Context{}, {var}, {}, FnProperty::kNormal);
    finished_cv_.wait(lock, [&done]() { return done.load(); });
  }
}

void ThreadedEngine::WaitForAll() {
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() { return pending_.load() == 0; });
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
    bool to_delete = i->CompleteWriteDependency(
        [this](OprBlock* opr) {
          this->PushToExecute(opr, false);
        });
    if (to_delete) {
      ThreadedVar::Delete(i);
    }
  }
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    if (--pending_ == 0) {
      finished_cv_.notify_all();
    }
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
