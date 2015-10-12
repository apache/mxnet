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
   * \brief command id to set the server to the sync mode
   */
  static const int kSyncMode = -2;

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


/** \brief to match worker/server's app id */
#define PS_KV_ID 9

/** \brief to match worker/server's app id */
#define PS_APP_ID 10

/**
 * \brief a server node on ps
 */
class MXNetServer : public ps::App {
 public:
  MXNetServer() : App(PS_APP_ID) { }
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

    int cmd = request->task.cmd();
    if (cmd == CommandID::kStop) {
      executor_->Stop();
    } else if (cmd == CommandID::kSyncMode) {

    } else {
      // let the main thread to execute updater_, which is necessary for python
      executor_->Exec([cmd, request, this]() {
          controller_(cmd, request->task.msg());
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
  MXNetWorker() : App(PS_APP_ID) { }
  virtual ~MXNetWorker() { }
};

/**
 * \brief a scheduler node on ps
 */
class MXNetScheduler : public ps::App {
 public:
  MXNetScheduler() : App(PS_APP_ID) { }
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


/**
 * \brief distributed kvstore for servers
 */
class KVStoreDistServer {
 public:

  explicit KVStoreDistServer(const KVStore::Controller& ctrl) {

  }

  void set_updater(const KVStore::Updater& updater)  {
  }

  void Run() {

  }
 private:
  /**
   * \brief let the main thread execute python codes
   */
  Executor exec_;
};



  // /**
  //  * \brief convert from a key in ps
  //  */
  // inline int DecodeKey(ps::Key key) {
  //   return static_cast<int>((key << kIndexBits) >> kIndexBits);
  // }

  // /**
  //  * \brief value type stored at server
  //  */
  // struct ServerVal {
  //   NDArray array;
  //   inline void Load(dmlc::Stream *fi) { array.Load(fi); }
  //   inline void Save(dmlc::Stream *fo) const { array.Save(fo); }
  //   inline bool Empty() const { return array.is_none(); }
  // };

  // /**
  //  * \brief server handle
  //  */
  // class ServerHandle {
  //  public:
  //   explicit ServerHandle(KVStoreDist* kvstore) {
  //     kvstore_ = kvstore;
  //     ps_ = CHECK_NOTNULL(CHECK_NOTNULL(kvstore_->store_)->server());
  //     curr_timestamp_ = -1;
  //   }

  //   /**
  //    * \brief it is called before any push and pull.
  //    *
  //    * we manage the data consistency among workers here.
  //    */
  //   inline void Start(bool push, int timestamp, int cmd_id, void* msg) {
  //     curr_timestamp_ = timestamp;
  //     if (!kvstore_->updater_) {
  //       // BSP

  //       // use the shared pointer version of the message to prevent it being
  //       // deleted by the system
  //       auto msg = ps_->LastRequest();
  //       // should only has a single key
  //       CHECK_EQ(msg->key.size(), sizeof(int));

  //       pending_push_[curr_timestamp_].push_back(msg);
  //       // prevent system to reply this request, so we can hang the worker for a
  //       // while
  //       msg->replied = true;
  //     }
  //   }

  //   inline void Finish() { }
  //   inline void Load(dmlc::Stream *fi) { }
  //   inline void Save(dmlc::Stream *fo) const { }

  //   inline void Push(ps::Key recv_key,
  //                    ps::Blob<const real_t> recv_val,
  //                    ServerVal& my_val) {  // NOLINT(*)
  //     // construct NDArray without data copy
  //     size_t ds[] = {recv_val.size};
  //     TShape dshape(ds, ds + 1);
  //     TBlob recv_blob((real_t*)recv_val.data,  // NOLINT(*)
  //                     dshape, cpu::kDevMask);
  //     NDArray recv_array(recv_blob, 0);
  //     bool reply = false;
  //     bool reduce = !kvstore_->updater_;

  //     if (my_val.Empty()) {
  //       // initialization
  //       my_val.array = NDArray(dshape, Context());
  //       CopyFromTo(recv_array, &my_val.array);
  //       if (reduce) reply = true;
  //     } else {
  //       if (reduce) {
  //         // runs eventual consistency model. so update immediately

  //         // let the main thread to execute updater_, which is necessary for
  //         // python
  //         int key = kvstore_->DecodeKey(recv_key);
  //         kvstore_->exec_.Exec([this, key, &recv_array, &my_val](){
  //             kvstore_->updater_(key, recv_array, &my_val.array);
  //           });
  //       } else {
  //         // just aggregate data from workers
  //         int num_recv = pending_push_[curr_timestamp_].size();
  //         CHECK_NE(num_recv, 0);

  //         if (num_recv == 1) {
  //           my_val.array = 0;
  //         }

  //         my_val.array += recv_array;

  //         if (num_recv == ps::NodeInfo::NumWorkers()) {
  //           reply = true;
  //         }
  //       }
  //     }

  //     if (reply) {
  //       for (auto& m : pending_push_[curr_timestamp_]) {
  //         ps_->Reply(m.get());
  //       }
  //       pending_push_.erase(curr_timestamp_);
  //     }
  //     // place waittoread here rather than the beginning of pull.
  //     my_val.array.WaitToRead();
  //   }

  //   inline void Pull(ps::Key recv_key,
  //                    const ServerVal& my_val,
  //                    ps::Blob<real_t>& send_val) {  // NOLINT(*)
  //     CHECK(!my_val.Empty())
  //         << kvstore_->DecodeKey(recv_key) << " is not inited";

  //     send_val.data = static_cast<real_t*>(my_val.array.data().dptr_);
  //     send_val.size = my_val.array.shape()[0];
  //   }

  //  private:

  //   KVStoreDist* kvstore_;

  //   ps::Customer* ps_;
  //   /**
  //    * \brief for BSP model
  //    */
  //   std::unordered_map<int, std::vector<
  //                             std::shared_ptr<ps::Message>>> pending_push_;
  //   /**
  //    * \brief the current timestamp
  //    */
  //   int curr_timestamp_;
  // };

  // /**
  //  * \brief kv store at server node
  //  */
  // ps::OnlineServer<real_t, ServerVal, ServerHandle>* store_;





    // if (IsServerNode()) {
    //   ServerHandle handle(this);
    //   store_ = new ps::OnlineServer<real_t, ServerVal, ServerHandle>(handle);
    // } else



    // if (store_) store_->server()->Clear();
    // delete store_;















}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_MXNET_PS_NODE_H_

    // need to explicit clear the NDArray before Engine is deleted

      // server_ = new KVStoreDistServer(controller);
      // FIXME
      // auto node = CHECK_NOTNULL(ps::NodeInfo::MyApp());
      // auto server = static_cast<MXNetServer*>(node);
      // server->set_executor(&exec_);
      // server->set_controller(controller);
      // exec_.Start();
    // }
