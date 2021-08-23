/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
#include "../common/cuda/utils.h"

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
  std::lock_guard<std::mutex> lock{mutex_};
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
  std::lock_guard<std::mutex> lock{mutex_};
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
    std::lock_guard<std::mutex> lock{mutex_};
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
    std::lock_guard<std::mutex> lock{mutex_};
    // invariants
    assert(head_->next == nullptr);
    assert(pending_write_ != nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // increment version number
    ++version_;

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
        // mark write as already activated in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not rely on these two variables.
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
  std::lock_guard<std::mutex> lock{mutex_};
  to_delete_ = true;
}

inline bool ThreadedVar::ready_to_read() {
  std::lock_guard<std::mutex> lock{mutex_};
  return this->is_ready_to_read();
}

inline size_t ThreadedVar::version() {
  std::lock_guard<std::mutex> lock{mutex_};
  return this->version_;
}

// implementation of threaded engine
ThreadedVar* ThreadedEngine::NewVariable() {
  return ThreadedVar::New(VersionedVarBlock::New());
}

ThreadedOpr* ThreadedEngine::NewOperator(
    ThreadedEngine::AsyncFn fn,
    std::vector<VarHandle> const& const_vars,
    std::vector<VarHandle> const& mutable_vars,
    FnProperty prop,
    const char* opr_name,
    bool wait) {
  auto ret = ThreadedOpr::New();
  ret->opr_name = opr_name ? std::string(opr_name) : std::string();
  ret->fn = std::move(fn);
  ret->prop = prop;
  ret->const_vars.resize(const_vars.size());
  ret->mutable_vars.resize(mutable_vars.size());
  ret->wait = wait;
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
  const size_t use_size = use.size();
  const size_t mutate_size = mutate.size();
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
  this->PushAsync([threaded_opr](RunContext, CallbackOnComplete on_complete) {
      ThreadedOpr::Delete(threaded_opr);
      on_complete();
    }, Context::CPU(), {}, deps, FnProperty::kDeleteVar, 0,
    "DeleteOperator");
}

void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority, bool profiling) {
  BulkFlush();
  ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
  if (profiling) {
    threaded_opr->opr_name =
        profiler::CustomOpProfiler::Get()->GenerateDisplayName(threaded_opr->opr_name.c_str());
  }
  OprBlock* opr_block = OprBlock::New();
  opr_block->opr = threaded_opr;

  opr_block->wait.store(static_cast<int>(
      threaded_opr->const_vars.size() +
      threaded_opr->mutable_vars.size() + 1));
  opr_block->ctx = exec_ctx;
  opr_block->priority = priority;
  opr_block->profiling = profiling;
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
                               FnProperty prop,
                               int priority,
                               const char* opr_name,
                               bool wait) {
#if MXNET_USE_CUDA
  if (exec_ctx.dev_mask() == gpu::kDevMask) {
    if (device_count_ < 0) {
      int tmp = -1;
      cudaGetDeviceCount(&tmp);
      device_count_ = tmp;
      CHECK_GT(device_count_, 0) << "GPU usage requires at least 1 GPU";
    }
    CHECK_LT(exec_ctx.dev_id, device_count_)
        << "Invalid GPU Id: " << exec_ctx.dev_id
        << ", Valid device id should be less than device_count: "
        << device_count_;
  }
#endif
  const bool profiling = profiler_->IsProfiling(profiler::Profiler::kImperative);
  ThreadedOpr *opr = NewOperator(std::move(fn), const_vars, mutable_vars,
                                 prop, opr_name, wait);
  opr->temporary = true;
  Push(opr, exec_ctx, priority, profiling);
}

void ThreadedEngine::PushSync(SyncFn exec_fn, Context exec_ctx,
                              std::vector<VarHandle> const& const_vars,
                              std::vector<VarHandle> const& mutable_vars,
                              FnProperty prop,
                              int priority,
                              const char* opr_name) {
  if (!bulk_size() || prop != FnProperty::kNormal || priority) {
    this->PushAsync([exec_fn](RunContext ctx, CallbackOnComplete on_complete) {
        exec_fn(ctx);
        on_complete();
      }, exec_ctx, const_vars, mutable_vars, prop, priority, opr_name);
    return;
  }

  const BulkStatus& bulk_status = *BulkStatusStore::Get();
  if (bulk_status.count && exec_ctx != bulk_status.ctx) BulkFlush();
  BulkAppend(exec_fn, exec_ctx, const_vars, mutable_vars);
}

void ThreadedEngine::DeleteVariable(SyncFn delete_fn,
                                    Context exec_ctx,
                                    VarHandle var) {
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  this->PushAsync([delete_fn, threaded_var](RunContext ctx, CallbackOnComplete on_complete) {
      // Mark variable as orphan,
      // so during `ThreadedEngine::OnComplete` it could be recycled.
      threaded_var->SetToDelete();
      delete_fn(ctx);
      on_complete();
    }, exec_ctx, {}, {var}, FnProperty::kDeleteVar, 0,
    "DeleteVariable");
}

void ThreadedEngine::WaitForVar(VarHandle var) {
  BulkFlush();
  ThreadedVar* threaded_var = ThreadedVar::CastFromBase(var);
  if (threaded_var->ready_to_read()) {
    ThrowException(threaded_var);
    return;
  }
  if (engine_info_) {
    LOG(INFO) << "Wait for " << threaded_var;
    debug_wait_var_ = threaded_var;
  }
  std::atomic<bool> done{false};
  this->PushAsync([this, &done](RunContext, CallbackOnComplete on_complete) {
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
      on_complete();
    }, Context::CPU(), {var}, {}, FnProperty::kNormal, 0,
    "WaitForVar", true);
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    finished_cv_.wait(lock, [this, &done]() {
        return done.load() || kill_.load();
    });
  }

  ThrowException(threaded_var);
}

void ThreadedEngine::WaitForAll() {
  BulkFlush();
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() {
      return pending_.load() == 0 || kill_.load();
    });
  std::exception_ptr exception_to_rethrow = nullptr;
  if (!global_exception_refs_.empty()) {
    // iterate through all exception refs
    for (const auto& global_exception_ref : global_exception_refs_) {
      // the first exception will be saved to be rethrown later
      if (*global_exception_ref != nullptr && exception_to_rethrow == nullptr) {
        exception_to_rethrow = *global_exception_ref;
      }
      // clear exceptions, WaitToRead following WaitForAll shouldn't throw
      *global_exception_ref = nullptr;
    }
    // A waitall following a waitall shouldn't throw any exceptions
    global_exception_refs_.clear();
    if (exception_to_rethrow != nullptr) {
      std::rethrow_exception(exception_to_rethrow);
    }
  }
}

inline void ThreadedEngine::OnComplete(ThreadedOpr* threaded_opr) {
  bool is_temporary_opr = threaded_opr->temporary;
  // Mark complete for read variables
  for (auto&& i : threaded_opr->const_vars) {
    i->CompleteReadDependency(
        [this](OprBlock* opr) { this->PushToExecute(opr, false); });
  }
  // Mark complete for write variables.
  for (auto&& i : threaded_opr->mutable_vars) {
    if (threaded_opr->opr_exception && *threaded_opr->opr_exception) {
      i->var_exception = threaded_opr->opr_exception;
      // add current operator exceptions to global exceptions if not already
      // added
      AddToGlobalExceptions(threaded_opr->opr_exception);
    }
    const bool debug_info = (engine_info_ && debug_wait_var_ == i);
    if (debug_info) {
      LOG(INFO) << "Complete write dep for " << i;
    }
    const bool to_delete =
        i->CompleteWriteDependency([this, debug_info](OprBlock* opr) {
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
  // The function been pushed from `ThreadedEngine::DeleteOperator`
  // could execute right after we mark all vars as complete, so if
  // threaded_opr is not temporary, its value is not reliable
  // anymore start from here.
  int npending = 0;
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    npending = --pending_;
  }
  CHECK_GE(npending, 0);
  if (npending == 0) {
    // no need to grab lock when notify.
    finished_cv_.notify_all();
  }

  // delete operator if it is temperory
  if (is_temporary_opr) {
    ThreadedOpr::Delete(threaded_opr);
  }
}

inline void ThreadedEngine::ThrowException(ThreadedVar* threaded_var) {
  if (threaded_var->var_exception && *threaded_var->var_exception) {
    std::exception_ptr tmp = *threaded_var->var_exception;
    *threaded_var->var_exception = nullptr;
    std::rethrow_exception(tmp);
  }
  return;
}

void ThreadedEngine::Throw(VarHandle var) {
  ThreadedVar *threaded_var = ThreadedVar::CastFromBase(var);
  ThrowException(threaded_var);
}

void ThreadedEngine::OnCompleteStatic(Engine *engine, void *opr_block_,
                                      const dmlc::Error* error) {
  OprBlock *opr_block = static_cast<OprBlock*>(opr_block_);
  ThreadedOpr *threaded_opr = opr_block->opr;
  if (error != nullptr) {
    auto ex_p = std::make_exception_ptr(*error);
    threaded_opr->opr_exception = std::make_shared<std::exception_ptr>(ex_p);
  }
  if (opr_block->profiling && threaded_opr->opr_name.size()) {
    // record operator end timestamp
    opr_block->opr_profile->stop();
  }
  static_cast<ThreadedEngine*>(engine)->OnComplete(threaded_opr);
  OprBlock::Delete(opr_block);
}

}  // namespace engine
}  // namespace mxnet
