/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_MXNET_PS_NODE_H_
#define MXNET_KVSTORE_MXNET_PS_NODE_H_
#include <queue>
#include <string>
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

  /**
   * \brief number of bits used to encode the key in mxnet
   */
  static const int kIndexBits = 32;
};

/**
 * \brief a simple aggregator over time.
 */
class Aggregator {
 public:
  /**
   * \param num number of nodes for aggregation
   */
  Aggregator(int num, ps::Customer* obj) {
    num_ = num;
    obj_ = obj;
  }

  using Message = std::shared_ptr<ps::Message>;

  bool Has(int time) {
    return msgs_.find(time) != msgs_.end();
  }

  void Add(int time, const Message& msg) {
    msgs_[time].push_back(msg);
    msg->replied = true;
  }

  size_t Size() {
    return msgs_.size();
  }

  size_t Count(int time) {
    return msgs_[time].size();
  }

  bool Done(int time) {
    return Count(time) == (size_t)num_;
  }

  void Remove(int time) {
    for (auto& m : msgs_[time]) {
      CHECK_NOTNULL(obj_)->Reply(m.get());
    }
    msgs_.erase(time);
  }

 private:
  std::unordered_map<int, std::vector<Message>> msgs_;
  int num_;
  ps::Customer* obj_;
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

  void set_controller(const KVStore::Controller& ctrl) {
    controller_ = ctrl;
  }

  void ProcessRequest(ps::Message* request) override {
    // wait for one second if controller_ is not inited
    for (int i = 0; i < 100; ++i) {
      if (!controller_) usleep(10000);
    }
    CHECK(controller_);
    controller_(request->task.cmd(), request->task.msg());
  }

 private:
  KVStore::Controller controller_;
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
  MXNetScheduler()
      : App(PS_APP_ID),
        barrier_(ps::NodeInfo::NumWorkers(), this) {
  }
  virtual ~MXNetScheduler() { }

  void ProcessRequest(ps::Message* request) override {
    int count;
    if (CommandID::GetBarrier(request->task.cmd(), &count)) {
      barrier_.Add(count, LastRequest());
      CHECK_EQ(barrier_.Size(), 1);

      if (barrier_.Done(count)) {
        barrier_.Remove(count);
      }
    }
  }

 private:
  Aggregator barrier_;
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

/**
 * \brief distributed kvstore for servers
 */
class KVStoreDistServer {
 public:
  explicit KVStoreDistServer(const KVStore::Controller& user_ctrl)
      // set updater
      : store_(ServerHandle(this), 1, 1, PS_KV_ID) {
    // set controller
    sync_mode_ = false;
    auto controller
        = [user_ctrl, this](int cmd_id, const std::string& cmd_body) {
      if (cmd_id == CommandID::kStop) {
        exec_.Stop();
      } else if (cmd_id == CommandID::kSyncMode) {
        sync_mode_ = true;
      } else {
        // let the main thread to execute ctrl, which is necessary for python
        exec_.Exec([user_ctrl, cmd_id, cmd_body]() {
            CHECK(user_ctrl);
            user_ctrl(cmd_id, cmd_body);
        });
      }
    };
    auto node = CHECK_NOTNULL(ps::NodeInfo::MyApp());
    static_cast<MXNetServer*>(node)->set_controller(controller);
  }

  // ~KVStoreDistServer() {
  //   // clear all ndarrays before Engine is shutting down.
  //   store_.server()->Clear();
  // }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  void Run() {
    exec_.Start();
  }

 private:
  /**
   * \brief value type stored at server
   */
  struct ServerVal {
    std::vector<real_t> data;
    inline void Load(dmlc::Stream *fi) { fi->Read(&data); }
    inline void Save(dmlc::Stream *fo) const { fo->Write(data); }
    inline bool Empty() const { return data.empty(); }
  };

  /**
   * \brief server handle
   */
  class ServerHandle {
   public:
    explicit ServerHandle(KVStoreDistServer* kvstore)
        : kvstore_(kvstore),
          ps_obj_(nullptr),
          aggregator_(nullptr) {
    }

    ~ServerHandle() {
      delete aggregator_;
    }

    /**
     * \brief get a cpu ndarray from a c-array without data copy
     */
    inline NDArray GetNDArray(real_t* data, size_t size) {
      size_t ds[] = {size};
      TShape dshape(ds, ds + 1);
      TBlob data_blob(data, dshape, cpu::kDevMask);
      return NDArray(data_blob, 0);
    }

    inline void Start(bool push, int timestamp, int cmd_id, void* msg) { }
    inline void Finish() { }
    inline void Load(dmlc::Stream *fi) { }
    inline void Save(dmlc::Stream *fo) const { }

    inline void Push(ps::Key recv_key,
                     ps::Blob<const real_t> recv_val,
                     ServerVal& my_val) {  // NOLINT(*)
      // initialization
      if (my_val.Empty()) {
        my_val.data.resize(recv_val.size);
        memcpy(my_val.data.data(), recv_val.data,
               recv_val.size * sizeof(real_t));
        return;
      }

      int key = DecodeKey(recv_key);
      NDArray recv_array = GetNDArray((real_t*)recv_val.data,  // NOLINT(*)
                                      recv_val.size);
      NDArray my_array = GetNDArray(my_val.data.data(), my_val.data.size());

      if (kvstore_->sync_mode_) {
        // create aggregator
        if (aggregator_ == nullptr) {
          ps_obj_ = CHECK_NOTNULL(kvstore_)->store_.server();
          aggregator_ = new Aggregator(
              ps::NodeInfo::NumWorkers(), ps_obj_);
        }

        // init merge buf
        std::vector<real_t>& buf = merge_buf_[key];
        if (!aggregator_->Has(key)) {
          if (buf.empty()) {
            buf.resize(recv_val.size);
          }
          memset(buf.data(), 0, buf.size() * sizeof(real_t));
        }

        // add recved data into merge
        NDArray merge = GetNDArray(buf.data(), buf.size());
        merge += recv_array;

        // update if aggregation is done
        aggregator_->Add(key, ps_obj_->LastRequest());
        if (aggregator_->Done(key)) {
          // let the main thread to execute updater_, which is necessary for
          // python
          merge.WaitToRead();
          kvstore_->exec_.Exec([this, key, &merge, &my_array](){
              CHECK(kvstore_->updater_);
              kvstore_->updater_(key, merge, &my_array);
            });
          aggregator_->Remove(key);
        }
      } else {
        // runs eventual consistency model. so update immediately

        // let the main thread to execute updater_, which is necessary for
        // python
        kvstore_->exec_.Exec([this, key, &recv_array, &my_array](){
            CHECK(kvstore_->updater_);
            kvstore_->updater_(key, recv_array, &my_array);
          });
      }
      // place waittoread here rather than the beginning of pull.
      my_array.WaitToRead();
    }

    inline void Pull(ps::Key recv_key,
                     const ServerVal& my_val,
                     ps::Blob<real_t>& send_val) {  // NOLINT(*)
      CHECK(!my_val.Empty())
          << DecodeKey(recv_key) << " is not inited";

      send_val.data = (real_t*) my_val.data.data();  // NOLINT(*)
      send_val.size = my_val.data.size();
    }

   private:
    /**
     * \brief convert from a key in ps
     */
    inline int DecodeKey(ps::Key key) {
      return static_cast<int>(
          (key << CommandID::kIndexBits) >> CommandID::kIndexBits);
    }
    /**
     * \brief for BSP model
     */
    std::unordered_map<int, std::vector<real_t>> merge_buf_;
    /**
     * \brief the current timestamp
     */
    // int curr_timestamp_;

    KVStoreDistServer* kvstore_;

    ps::Customer* ps_obj_;
    Aggregator* aggregator_;
  };


  /**
   * \brief let the main thread execute python codes
   */
  Executor exec_;

  bool sync_mode_;

  KVStore::Updater updater_;

  ps::OnlineServer<real_t, ServerVal, ServerHandle> store_;
};


}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_MXNET_PS_NODE_H_
