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

#ifdef DAG_ENGINE_DEBUG
std::atomic<std::size_t> OprBlock::counter{0};
std::atomic<std::size_t> VersionedVarBlock::counter{0};
std::atomic<std::size_t> SimpleVar::counter{0};
std::atomic<std::size_t> SimpleOpr::counter{0};
#endif  // DAG_ENGINE_DEBUG

SimpleVar* SimpleVar::CastFromBase(Var* v) { return v->Cast<SimpleVar>(); }

SimpleOpr* SimpleOpr::CastFromBase(Opr* o) { return o->Cast<SimpleOpr>(); }

SimpleEngine::SimpleEngine()
    : pending_{0}, thread_pool_{[this]() { ThreadWorker(); }} {}

SimpleEngine::~SimpleEngine() noexcept(false) { task_queue_.SignalForKill(); }

SimpleVar* SimpleEngine::NewVar() {
  auto ret = new SimpleVar{};
  ret->head = new VersionedVarBlock{};
  return ret;
}

SimpleOpr* SimpleEngine::NewOperator(SimpleEngine::AsyncFn fn,
                                     std::vector<Variable> const& use_vars,
                                     std::vector<Variable> const& mutate_vars) {
  auto ret = new SimpleOpr{};
  ret->fn = fn;
  ret->use_vars.resize(use_vars.size());
  ret->mutate_vars.resize(mutate_vars.size());
  std::transform(use_vars.begin(), use_vars.end(), ret->use_vars.begin(),
                 SimpleVar::CastFromBase);
  std::transform(mutate_vars.begin(), mutate_vars.end(),
                 ret->mutate_vars.begin(), SimpleVar::CastFromBase);
#ifdef DAG_ENGINE_DEBUG
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

void SimpleEngine::DeleteOperator(OprHandle op) {
  auto&& simple_opr = SimpleOpr::CastFromBase(op);
  std::vector<Variable> deps{};
  deps.reserve(simple_opr->use_vars.size() + simple_opr->mutate_vars.size());
  deps.insert(deps.end(), simple_opr->use_vars.begin(),
              simple_opr->use_vars.end());
  deps.insert(deps.end(), simple_opr->mutate_vars.begin(),
              simple_opr->mutate_vars.end());
  auto&& func = [simple_opr](RunContext) { delete simple_opr; };
  Push(func, Context{}, {}, deps);
}

void SimpleEngine::Push(OprHandle op, Context exec_ctx) {
  auto&& simple_opr = SimpleOpr::CastFromBase(op);
  auto&& opr_block = new OprBlock{};
  opr_block->opr = simple_opr;
  opr_block->wait.store(simple_opr->use_vars.size() +
                        simple_opr->mutate_vars.size() + 1);
  opr_block->ctx = exec_ctx;
  opr_block->rctx = RunContext{nullptr};
  ++pending_;
  // Add read dependencies.
  for (auto&& i : simple_opr->use_vars) {
    std::lock_guard<std::mutex> lock{i->m};
    if (i->ready_to_read) {
      assert(i->pending_write == nullptr);
      ++i->num_pending_reads;
      --opr_block->wait;
    } else {
      auto&& new_var_block = new VersionedVarBlock{};
      assert(i->head->next == nullptr);
      assert(i->head->trigger == nullptr);
      assert(i->head->write == false);
      i->head->next = new_var_block;
      i->head->trigger = opr_block;
      i->head = new_var_block;
    }
  }
  // Add write dependencies.
  for (auto&& i : simple_opr->mutate_vars) {
    std::lock_guard<std::mutex> lock{i->m};
    auto&& new_var_block = new VersionedVarBlock{};
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

void SimpleEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                             std::vector<Variable> const& use_vars,
                             std::vector<Variable> const& mutate_vars) {
  auto&& opr = NewOperator(fn, use_vars, mutate_vars);
  opr->temporary = true;
  Push(opr, exec_ctx);
}

void SimpleEngine::PushDelete(Fn delete_fn, Context exec_ctx, Variable var) {
  auto&& simple_var = SimpleVar::CastFromBase(var);
  auto&& func = [delete_fn, simple_var](RunContext ctx) {
    /*!
     * Mark variable as orphan, so during `SimpleEngine::OnComplete` it could be
     * recycled.
     */
    simple_var->to_delete = true;
    delete_fn(ctx);
  };
  Push(func, exec_ctx, {}, {var});
}

void SimpleEngine::WaitForVar(Variable var) {
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

void SimpleEngine::WaitForAll() {
  std::unique_lock<std::mutex> lock{finished_m_};
  finished_cv_.wait(lock, [this]() { return pending_.load() == 0; });
}

void SimpleEngine::OnComplete(SimpleOpr* simple_opr) {
  /*!
   * Mark complete for read variables.
   */
  for (auto&& i : simple_opr->use_vars) {
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
  for (auto&& i : simple_opr->mutate_vars) {
    bool to_delete = false;
    {
      std::lock_guard<std::mutex> lock{i->m};
      assert(i->ready_to_read == false);
      auto head = i->pending_write->next;
      delete i->pending_write;
      i->pending_write = nullptr;
      if (i->to_delete) {
        assert(head->next == nullptr);
        delete head;
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
            delete prev;
          }
        }
      }
    }
    if (to_delete) {
      delete i;
    }
  }
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    if (--pending_ == 0) {
      finished_cv_.notify_all();
    }
  }
}

void SimpleEngine::ThreadWorker() {
  OprBlock* opr_block;
  while (task_queue_.Pop(&opr_block)) {
    assert(opr_block->wait.load() == 0);
    auto simple_opr = opr_block->opr;
    auto callback = [this, simple_opr]() {
      OnComplete(simple_opr);
      if (simple_opr->temporary) {
        delete simple_opr;
      }
    };
    if (opr_block->ctx.dev_mask == gpu::kDevMask) {
#if MXNET_USE_CUDA
      CUDA_CALL(cudaSetDevice(opr_block->ctx.dev_id));
#else  // MXNET_USE_CUDA
      LOG(FATAL) << "Please compile with CUDA enabled";
#endif  // MXNET_USE_CUDA
    }
    simple_opr->fn(opr_block->rctx, callback);
    delete opr_block;
  }
}

}  // namespace engine

}  // namespace mxnet
