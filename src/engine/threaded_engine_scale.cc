/*!
* Copyright (c) 2017 by Contributors
* \file threaded_engine_perdevice.cc
* \brief ThreadedEngine that uses fix amount of thread for each device.
*/
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include "./threaded_engine.h"
#include "../common/utils.h"
#include <vector>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <queue>
#include <atomic>

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
 
      ThreadedEngineScale() noexcept(false) : abort(false), NextQueue(0) {
        int threads = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 16);
        for (int i = 0; i < threads; i++) {
          Queues.push_back(new std::priority_queue<OprBlock*, std::deque<OprBlock*>, PriorityComparer>());
          QueueMutex.push_back(new std::mutex());
        }
        for (int i = 0; i < threads; i++) {
          std::thread *thread = new std::thread([this] { this->CPUWorker(); });
          auto id = thread->get_id();
          IndexMap[id] = i;
          Threads.push_back(thread);
        }
      }
      ~ThreadedEngineScale() noexcept(false) {
        abort = true;
        {
        std::unique_lock<std::mutex> lk(WaitMutex);
        WaitCV.notify_all();
        lk.unlock();
        }
        for (size_t i = 0; i < Threads.size(); i++) {
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
            {
            // push to worker,
            // roundrobin the queue
            size_t next = NextQueue++;
            next = next % QueueMutex.size();
            std::lock_guard<std::mutex> lock(*QueueMutex[next]);
            Queues[next]->push(opr_block);
            }
            {
              // notify if there is anyone is waiting
              std::unique_lock<std::mutex> lk(WaitMutex);
              WaitCV.notify_one();
              lk.unlock();
            }
          }
          else {
            std::lock_guard<std::mutex> lock(*QueueMutex[index]);
            Queues[index]->push(opr_block);
          }
        }
      }

      size_t IndexFromCurrentThread() {
        std::thread::id id = std::this_thread::get_id();
        auto iter = IndexMap.find(id);
        if (iter != IndexMap.end())
          return iter->second;
        else
          return -1;
      }

      void CPUWorker() {
        size_t index;
        do {
           index = IndexFromCurrentThread();
        } while(index == (size_t)-1);

        auto myQueue = Queues[index];
        RunContext run_ctx;
        run_ctx.stream = nullptr;
        OprBlock* opr_block;
        bool hasMoreWork;
#if 1
        pthread_t thread;
        cpu_set_t cpuset;

        thread = pthread_self();

        CPU_ZERO(&cpuset);
        CPU_SET(index, &cpuset);

        pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
        for (;!this->abort;)
        {
          opr_block = nullptr;
          {
            // check my queue
            std::lock_guard<std::mutex> lock(*QueueMutex[index]);

            if (!myQueue->empty()) {
              opr_block = myQueue->top();
              myQueue->pop();

              if (myQueue->size() > 1) {
                // since we have more workitem in queues, try to wake up other threads
                hasMoreWork = true;
              }
            }
          }

          if (hasMoreWork) {
              hasMoreWork = false;
              std::lock_guard<std::mutex> lock(WaitMutex);
              WaitCV.notify_one();
          }
          if (opr_block == NULL) {
            // check if we get one work item
            for (size_t i = 1; i < Queues.size(); i++) {
              size_t other = (index + i ) % Queues.size();
              std::lock_guard<std::mutex> lock(*QueueMutex[other]);
              auto otherQueue = Queues[other];
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
      std::atomic<size_t> NextQueue;
    };

    Engine *CreateThreadedEngineScale() {
      return new ThreadedEngineScale();
    }
  }  // namespace engine
}  // namespace mxnet
