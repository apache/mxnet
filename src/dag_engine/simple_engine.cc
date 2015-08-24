/*!
 * Copyright (c) 2015 by Contributors
 */
#include "simple_engine.h"
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <utility>
#include <condition_variable>
#include <mutex>
#include "../common/cuda_utils.h"

namespace mxnet {

namespace engine {

SimpleVar* SimpleVar::CastFromBase(Var* v) { return v->Cast<SimpleVar>(); }

SimpleOpr* SimpleOpr::CastFromBase(Opr* o) { return o->Cast<SimpleOpr>(); }

SimpleEngine::SimpleEngine()
    : pending_{0}, thread_pool_{[this]() { ThreadWorker(); }} {}

SimpleEngine::~SimpleEngine() noexcept(false) { task_queue_.SignalForKill(); }

SimpleEngine::Variable SimpleEngine::NewVar() {
  auto ret = new SimpleVar{};
  ret->var = new VersionedVarBlock{};
  return ret;
}

SimpleEngine::Operator SimpleEngine::NewOperator(
    SimpleEngine::AsyncFn fn, std::vector<Variable> const& use_vars,
    std::vector<Variable> const& mutate_vars) {
  auto ret = new SimpleOpr{};
  ret->fn = fn;
  ret->use_vars.resize(use_vars.size());
  ret->mutate_vars.resize(mutate_vars.size());
  std::transform(use_vars.begin(), use_vars.end(), ret->use_vars.begin(),
                 SimpleVar::CastFromBase);
  std::transform(mutate_vars.begin(), mutate_vars.end(),
                 ret->mutate_vars.begin(), SimpleVar::CastFromBase);
  return ret;
}

void SimpleEngine::DeleteOperator(Operator op) { delete op; }

void SimpleEngine::Push(Operator op, Context exec_ctx) {
  auto opr = SimpleOpr::CastFromBase(op);
  auto opr_block = new OprBlock{};
  ++pending_;
  opr_block->wait.store(opr->use_vars.size() + opr->mutate_vars.size() + 1);
  // Add reading dependencies.
  auto add_dependency = [&opr_block](SimpleVar* i) {
    std::lock_guard<dmlc::Spinlock> lock{i->var->lock};
    if (!i->var->waiting) {
      assert(i->var->next == nullptr);
      assert(i->var->join == nullptr);
      assert(i->var->trigger == nullptr);
      --opr_block->wait;
    } else {
      auto new_var = new VersionedVarBlock{};
      new_var->waiting = true;
      assert(i->var->next == nullptr);
      i->var->next = new_var;
      i->var->trigger = opr_block;
      i->var = new_var;
    }
  };
  std::for_each(opr->use_vars.begin(), opr->use_vars.end(), add_dependency);
  std::for_each(opr->mutate_vars.begin(), opr->mutate_vars.end(),
                add_dependency);
  // Add mutation dependencies.
  VersionedVarBlock* first = nullptr;
  VersionedVarBlock* previous = nullptr;
  for (auto i : opr->mutate_vars) {
    i->var->lock.lock();
    if (!i->var->waiting) {
      assert(i->var->next == nullptr);
      assert(i->var->join == nullptr);
      assert(i->var->trigger == nullptr);
      i->var->waiting = true;
    } else {
      auto new_var = new VersionedVarBlock{};
      new_var->waiting = true;
      // Moving out from old block, set flag to false.
      i->var->waiting = false;
      new_var->lock.lock();
      i->var->lock.unlock();
      i->var = new_var;
    }
    if (first == nullptr) {
      first = i->var;
    } else {
      previous->join = i->var;
      previous->lock.unlock();
    }
    previous = i->var;
  }
  if (previous != nullptr) {
    previous->lock.unlock();
  }
  auto callback = [this, first]() { OnComplete(first); };
  RunContext rctx{};
  rctx.stream = nullptr;
  opr_block->fn = [exec_ctx, opr, rctx, callback]() {
    if (exec_ctx.dev_mask == gpu::kDevMask) {
#if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(exec_ctx.dev_id));
#else  // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
    }
    opr->fn(rctx, callback);
  };
  if (--opr_block->wait == 0) {
    task_queue_.Push(opr_block);
  }
}

void SimpleEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                             std::vector<Variable> const& use_vars,
                             std::vector<Variable> const& mutate_vars) {
  auto&& opr = NewOperator(fn, use_vars, mutate_vars);
  Push(opr, exec_ctx);
  DeleteOperator(opr);
}

void SimpleEngine::PushDelete(Fn delete_fn, Context exec_ctx, Variable var) {
  auto&& callback = [delete_fn, var](RunContext ctx) {
    // If you used `var` after `PushDelete`, then the following will be
    // undefined
    delete SimpleVar::CastFromBase(var)->var;
    delete var;
    delete_fn(ctx);
  };
  Push(callback, exec_ctx, {}, {var});
}

void SimpleEngine::WaitForVar(Variable var) {
  std::condition_variable cv;
  std::mutex m;
  std::unique_lock<std::mutex> lock{m};
  std::atomic<bool> done{false};
  auto&& callback = [&cv, &done](RunContext) {
    done.store(true);
    cv.notify_all();
  };
  cv.wait(lock, [&done]() { return done.load(); });
  Push(callback, Context{}, {var}, {});
}

void SimpleEngine::WaitForAll() {
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() { return pending_.load() == 0; });
}

void SimpleEngine::OnComplete(VersionedVarBlock* trigger) {
  auto head = trigger;
  while (head != nullptr) {
    auto cur = head;
    head = trigger->join;
    VersionedVarBlock* previous = nullptr;
    // Food for thought. This could also be `while true`.
    while (cur != nullptr) {
      std::lock_guard<dmlc::Spinlock> lock{cur->lock};
      assert((cur->next == nullptr) || cur->waiting);
      assert((previous == nullptr) || (cur->join == nullptr));
      assert((cur->trigger == nullptr) == (cur->next == nullptr));
      if (cur->trigger != nullptr && --cur->trigger->wait == 0) {
        task_queue_.Push(cur->trigger);
      }
      if (previous != nullptr) {
        delete previous;
      }
      previous = cur;
      if (cur->next == nullptr) {
        if (!cur->waiting) {
          // No `SimpleOpr` is on this block. Safe to delete.
          delete cur;
        } else {
          cur->waiting = false;
        }
        break;
      }
      cur = cur->next;
    }
  }
}

void SimpleEngine::ThreadWorker() {
  OprBlock* opr_block;
  while (task_queue_.Pop(&opr_block)) {
    assert(opr_block->wait.load() == 0);
    opr_block->fn();
    delete opr_block;
    if (--pending_ == 0) {
      finished_cv_.notify_all();
    }
  }
}

}  // namespace engine

}  // namespace mxnet
