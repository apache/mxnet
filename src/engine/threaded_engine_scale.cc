/*!
* Copyright (c) 2017 by Contributors
* \file threaded_engine_perdevice.cc
* \brief ThreadedEngine that uses fix amount of thread for each device.
*/
#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/concurrency.h>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/lazy_alloc_array.h"
#include "../common/utils.h"

namespace mxnet {
  namespace engine {
    /*!
    * \brief ThreadedEngine uses per device threads.
    * The policy of this Engine:
    *  - Execute Async operation immediately if pushed from Pusher.
    *  - Use fixed amount of threads for each device.
    *  - Each stream is allocated and bound to each of the thread.
    */
    class ThreadedEngineScale : public ThreadedEngine {
    public:
 
      ThreadedEngineScale() noexcept(false) : abort(false) {
        int threads = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 16);
        for (int i = 0; i < threads; i++) {
          std::thread *thread = new std::thread([this] { this->CPUWorker(); });
          auto id = thread->get_id();
          IndexMap[id] = i;
          Threads.push_back(thread);
          Queues.push_back(new std::priority_queue<OprBlock*, std::deque<OprBlock*>, PriorityComparer>());
          QueueMutex.push_back(new std::mutex());
        }
      }
      ~ThreadedEngineScale() noexcept(false) {
        abort = true;
        for (int i = 0; i < Threads.size(); i++) {
          Threads[i]->join();
          delete Threads[i];
          delete QueueMutex[i];
          delete Queues[i];
        }
      }

    protected:
      void PushToExecute(OprBlock *opr_block, bool pusher_thread) override {
        const Context& ctx = opr_block->ctx;
        if (opr_block->opr->prop == FnProperty::kAsync && pusher_thread) {
          RunContext run_ctx;
          run_ctx.stream = nullptr;
          this->ExecuteOprBlock(run_ctx, opr_block);
        }
        else {
          int index = IndexFromCurrentThread();
          if (index == -1) {
            // push to worker[0]
            std::lock_guard<std::mutex> lock(*QueueMutex[0]);
            Queues[0]->push(opr_block);
            {
              // notify if there is anyone is waiting
              std::unique_lock<std::mutex> lk(WaitMutex);
              WaitCV.wait(lk);
              lk.unlock();
            }
          }
          else {
            std::lock_guard<std::mutex> lock(*QueueMutex[index]);
            Queues[index]->push(opr_block);
          }
        }
      }

      int IndexFromCurrentThread() {
        std::thread::id id = std::this_thread::get_id();
        auto iter = IndexMap.find(id);
        if (iter != IndexMap.end())
          return iter->second;
        else
          return -1;
      }

      void CPUWorker() {
        int index = IndexFromCurrentThread();
        auto myQueue = Queues[index];
        RunContext run_ctx;
        run_ctx.stream = nullptr;
        OprBlock* opr_block;

        for (;!this->abort;)
        {
          opr_block = nullptr;
          {
            // check my queue
            std::lock_guard<std::mutex> lock(*QueueMutex[index]);

            if (!myQueue->empty()) {
              opr_block = myQueue->top();
              myQueue->pop();

              if (!myQueue->empty()) {
                // since we have more workitem in queues, try to wake up other threads
                std::lock_guard<std::mutex> lock(WaitMutex);
                WaitCV.notify_one();
              }
            }

            if (opr_block == NULL) {
              // check if we get one work item
              for (int i = 0; i < Queues.size(); i++) {
                if (i == index) continue; // skip self
                std::lock_guard<std::mutex> lock(*QueueMutex[i]);
                auto otherQueue = Queues[i];
                if (!otherQueue->empty()) {
                  opr_block = otherQueue->top();
                  otherQueue->pop();
                  break;
                }
              }
            }

            if (opr_block != NULL) {
              // excute work item otherwise wait
              this->ExecuteOprBlock(run_ctx, opr_block);
            }
            else {
              std::unique_lock<std::mutex> lock(WaitMutex);
              WaitCV.wait(lock);
              lock.unlock();
            }
          }
        }
      }

      class PriorityComparer
      {
      public:
        const bool operator()(const OprBlock* _Left, const OprBlock* _Right) {
          return _Left->priority < _Right->priority;
        }
      };

    private:
      bool abort;
      std::vector<std::mutex*> QueueMutex;
      std::vector<std::priority_queue<OprBlock*, std::deque<OprBlock*>, PriorityComparer>* > Queues;
      std::vector<std::thread*> Threads;
      std::map<std::thread::id, int> IndexMap;

      std::mutex WaitMutex;
      std::condition_variable WaitCV;
    };

    Engine *CreateThreadedEngineScale() {
      return new ThreadedEngineScale();
    }
  }  // namespace engine
}  // namespace mxnet
