/*!
 * Copyright (c) 2015 by Contributors
 */
#include "threaded_engine.h"
#include <dmlc/logging.h>
#include <cassert>
#include <algorithm>
#include <utility>
#include <condition_variable>
#include <mutex>
#include "../common/cuda_utils.h"

namespace mxnet {

namespace engine {

#if DAG_ENGINE_DEBUG
std::atomic<std::size_t> OprBlock::counter{0};
std::atomic<std::size_t> VersionedVarBlock::counter{0};
std::atomic<std::size_t> ThreadedVar::counter{0};
std::atomic<std::size_t> ThreadedOpr::counter{0};
#endif  // DAG_ENGINE_DEBUG

ThreadedVar* ThreadedVar::CastFromBase(Var* v) {
  return v->Cast<ThreadedVar>();
}

ThreadedOpr* ThreadedOpr::CastFromBase(Opr* o) {
  return o->Cast<ThreadedOpr>();
}

ThreadedEngine::ThreadedEngine()
    : pending_{0}, thread_pool_{[this]() { ThreadWorker(); }} {}

ThreadedEngine::~ThreadedEngine() noexcept(false) {
  task_queue_.SignalForKill();
}

ThreadedVar* ThreadedEngine::NewVar() {
  auto ret = ThreadedVar::New();
  ret->head = VersionedVarBlock::New();
  return ret;
}

ThreadedOpr* ThreadedEngine::NewOperator(
    ThreadedEngine::AsyncFn fn, std::vector<Variable> const& use_vars,
    std::vector<Variable> const& mutate_vars) {
  auto ret = ThreadedOpr::New();
  ret->fn = fn;
  ret->use_vars.resize(use_vars.size());
  ret->mutate_vars.resize(mutate_vars.size());
  std::transform(use_vars.begin(), use_vars.end(), ret->use_vars.begin(),
                 ThreadedVar::CastFromBase);
  std::transform(mutate_vars.begin(), mutate_vars.end(),
                 ret->mutate_vars.begin(), ThreadedVar::CastFromBase);
#if DAG_ENGINE_DEBUG
  // Check for duplicates.
  auto use = use_vars;
  auto mutate = mutate_vars;
  auto use_size = use.size();
  auto mutate_size = mutate.size();
  std::sort(use.begin(), use.end());
  std::sort(mutate.begin(), mutate.end());
  for (std::size_t i = 0; i < use_size; ++i) {
    if (i != 0 && use.at(i) == use.at(i - 1)) {
      LOG(FATAL) << "duplicate items found in `use_vars`";
    }
  }
  for (std::size_t i = 0; i < mutate_size; ++i) {
    if (i != 0 && mutate.at(i) == mutate.at(i - 1)) {
      LOG(FATAL) << "duplicate items found in `mutate_vars`";
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
          << "duplicate items found between `use_vars` and `mutate_vars`";
    }
  }
#endif  // DAG_ENGINE_DEBUG
  return ret;
}

void ThreadedEngine::DeleteOperator(OprHandle op) {
  auto&& threaded_opr = ThreadedOpr::CastFromBase(op);
  std::vector<Variable> deps{};
  deps.reserve(threaded_opr->use_vars.size() +
               threaded_opr->mutate_vars.size());
  deps.insert(deps.end(), threaded_opr->use_vars.begin(),
              threaded_opr->use_vars.end());
  deps.insert(deps.end(), threaded_opr->mutate_vars.begin(),
              threaded_opr->mutate_vars.end());
  auto&& func =
      [threaded_opr](RunContext) { ThreadedOpr::Delete(threaded_opr); };
  Push(func, Context{}, {}, deps);
}

void ThreadedEngine::Push(Fn exec_fun, Context exec_ctx,
                          std::vector<Variable> const& use_vars,
                          std::vector<Variable> const& mutate_vars) {
  auto f = [exec_fun](RunContext ctx, Callback on_complete) {
    exec_fun(ctx);
    on_complete();
  };
  PushAsync(f, exec_ctx, use_vars, mutate_vars);
}

void ThreadedEngine::Push(OprHandle op, Context exec_ctx) {
  auto&& threaded_opr = ThreadedOpr::CastFromBase(op);
  auto&& opr_block = OprBlock::New();
  opr_block->opr = threaded_opr;
  opr_block->wait.store(threaded_opr->use_vars.size() +
                        threaded_opr->mutate_vars.size() + 1);
  opr_block->ctx = exec_ctx;
  opr_block->rctx = RunContext{nullptr};
  ++pending_;
  // Add read dependencies.
  for (auto&& i : threaded_opr->use_vars) {
    std::lock_guard<std::mutex> lock{i->m};
    if (i->ready_to_read) {
      assert(i->pending_write == nullptr);
      ++i->num_pending_reads;
      --opr_block->wait;
    } else {
      auto&& new_var_block = VersionedVarBlock::New();
      assert(i->head->next == nullptr);
      assert(i->head->trigger == nullptr);
      assert(i->head->write == false);
      i->head->next = new_var_block;
      i->head->trigger = opr_block;
      i->head = new_var_block;
    }
  }
  // Add write dependencies.
  for (auto&& i : threaded_opr->mutate_vars) {
    std::lock_guard<std::mutex> lock{i->m};
    auto&& new_var_block = VersionedVarBlock::New();
    i->head->next = new_var_block;
    i->head->trigger = opr_block;
    i->head->write = true;
    if (i->ready_to_read) {
      /*!
       * Raise `num_pending_reads` temporarily to avoid premature triggering.
       */
      ++i->num_pending_reads;
      i->pending_write = i->head;
      if (--i->num_pending_reads == 0) {
        --opr_block->wait;
      }
      i->ready_to_read = false;
    }
    i->head = new_var_block;
  }
  if (--opr_block->wait == 0) {
    task_queue_.Push(opr_block);
  }
}

void ThreadedEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                               std::vector<Variable> const& use_vars,
                               std::vector<Variable> const& mutate_vars) {
  auto&& opr = NewOperator(fn, use_vars, mutate_vars);
  opr->temporary = true;
  Push(opr, exec_ctx);
}

void ThreadedEngine::PushDelete(Fn delete_fn, Context exec_ctx, Variable var) {
  auto&& threaded_var = ThreadedVar::CastFromBase(var);
  auto&& func = [delete_fn, threaded_var](RunContext ctx) {
    /*!
     * Mark variable as orphan, so during `ThreadedEngine::OnComplete` it could
     * be
     * recycled.
     */
    threaded_var->to_delete = true;
    delete_fn(ctx);
  };
  Push(func, exec_ctx, {}, {var});
}

void ThreadedEngine::WaitForVar(Variable var) {
  std::unique_lock<std::mutex> lock{finished_m_};
  std::atomic<bool> done{false};
  auto&& callback = [this, &done](RunContext) {
    std::unique_lock<std::mutex> lock{finished_m_};
    done.store(true);
    finished_cv_.notify_all();
  };
  Push(callback, Context{}, {var}, {});
  finished_cv_.wait(lock, [&done]() { return done.load(); });
}

void ThreadedEngine::WaitForAll() {
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() { return pending_.load() == 0; });
}

void ThreadedEngine::OnComplete(ThreadedOpr* threaded_opr) {
  /*!
   * Mark complete for read variables.
   */
  for (auto&& i : threaded_opr->use_vars) {
    std::lock_guard<std::mutex> lock{i->m};
    if (--i->num_pending_reads == 0) {
      if (i->pending_write != nullptr &&
          --i->pending_write->trigger->wait == 0) {
        task_queue_.Push(i->pending_write->trigger);
      }
    }
  }
  /*!
   * Mark complete for write variables.
   */
  for (auto&& i : threaded_opr->mutate_vars) {
    bool to_delete = false;
    {
      std::lock_guard<std::mutex> lock{i->m};
      assert(i->ready_to_read == false);
      auto head = i->pending_write->next;
      VersionedVarBlock::Delete(i->pending_write);
      i->pending_write = nullptr;
      if (i->to_delete) {
        assert(head->next == nullptr);
        VersionedVarBlock::Delete(head);
        to_delete = true;
      } else {
        while (true) {
          if (head->write == true) {
            ++i->num_pending_reads;
            i->pending_write = head;
            if (--i->num_pending_reads == 0) {
              if (--head->trigger->wait == 0) {
                task_queue_.Push(head->trigger);
              }
            }
            break;
          } else if (head->next == nullptr) {
            i->ready_to_read = true;
            break;
          } else {
            ++i->num_pending_reads;
            if (--head->trigger->wait == 0) {
              task_queue_.Push(head->trigger);
            }
            auto prev = head;
            head = head->next;
            VersionedVarBlock::Delete(prev);
          }
        }
      }
    }
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
}

void ThreadedEngine::ThreadWorker() {
  OprBlock* opr_block;
  while (task_queue_.Pop(&opr_block)) {
    assert(opr_block->wait.load() == 0);
    auto threaded_opr = opr_block->opr;
    auto callback = [this, threaded_opr]() {
      OnComplete(threaded_opr);
      if (threaded_opr->temporary) {
        ThreadedOpr::Delete(threaded_opr);
      }
    };
    if (opr_block->ctx.dev_mask == gpu::kDevMask) {
#if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(opr_block->ctx.dev_id));
#else  // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
    }
    threaded_opr->fn(opr_block->rctx, callback);
    OprBlock::Delete(opr_block);
  }
}

}  // namespace engine

}  // namespace mxnet
