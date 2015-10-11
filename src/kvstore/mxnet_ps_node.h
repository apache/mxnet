/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_MXNET_PS_NODE_H_
#define MXNET_KVSTORE_MXNET_PS_NODE_H_
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <vector>
#include "ps.h"
#include "mxnet/kvstore.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief encode/decode a command id
 */
struct CommandID {
  /**
   * \brief commmand id for stoping
   */
  static const int kStop = -1;
  /**
   * \brief returns the commmand id given a barrier count
   */
  static int SetBarrier(int count) {
    return - count - 10;
  }
  /**
   * \brief returns true if it is a barrier command
   */
  static bool GetBarrier(int cmd_id, int* count) {
    if (cmd_id <= 10) {
      *count = - cmd_id - 10;
      return true;
    }
    return false;
  }
};


/**
 * \brief executor runs a function using it's own thread
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p.set_value();
      } else {
        blk.p.set_value(); break;
      }

      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p.get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
    explicit Block(const Func& func) : f(func) { }
    Func f;
    std::promise<void> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

#define APP_ID 10
/**
 * \brief a server node on ps
 */
class MXNetServer : public ps::App {
 public:
  MXNetServer() : App(APP_ID) { }
  virtual ~MXNetServer() { }

  void set_executor(Executor* exec) {
    executor_ = exec;
  }

  void set_controller(const KVStore::Controller& ctrl) {
    controller_ = ctrl;
  }

  void ProcessRequest(ps::Message* request) override {
    // wait for one second if controller_ is not inited
    for (int i = 0; i < 100; ++i) {
      if (!controller_) usleep(10000);
    }
    CHECK(controller_);

    if (request->task.cmd() == CommandID::kStop) {
      executor_->Stop();
    } else {
      // let the main thread to execute updater_, which is necessary for python
      executor_->Exec([request, this]() {
          controller_(request->task.cmd(), request->task.msg());
        });
    }
  }

 private:
  KVStore::Controller controller_;
  Executor* executor_;
};

/**
 * \brief a worker node on ps
 */
class MXNetWorker : public ps::App {
 public:
  MXNetWorker() : App(APP_ID) { }
  virtual ~MXNetWorker() { }
};

/**
 * \brief a scheduler node on ps
 */
class MXNetScheduler : public ps::App {
 public:
  MXNetScheduler() : App(APP_ID) { }
  virtual ~MXNetScheduler() { }

  void ProcessRequest(ps::Message* request) override {
    int count;
    if (CommandID::GetBarrier(request->task.cmd(), &count)) {
      if (barrier_.msgs.empty()) {
        barrier_.barrier_count = count;
      } else {
        CHECK_EQ(barrier_.barrier_count, count);
      }
      barrier_.msgs.push_back(*request);
      // disable automatical reply so the worker node will be sleeped on node->Wait()
      request->replied = true;

      if (++barrier_.num_nodes == ps::NodeInfo::NumWorkers()) {
        for (auto& m : barrier_.msgs) {
          Reply(&m);
        }
        barrier_.clear();
      }
    }
  }

 private:
  // a simple barrier. probably move to ps-lite later
  struct Barrier {
    Barrier() { clear(); }
    void clear() {
      barrier_count = -1;
      num_nodes = 0;
      msgs.clear();
    }
    int num_nodes;
    int barrier_count;
    std::vector<ps::Message> msgs;
  };

  Barrier barrier_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_MXNET_PS_NODE_H_
