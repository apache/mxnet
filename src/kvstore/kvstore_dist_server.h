/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "ps/ps.h"
#include "mxnet/kvstore.h"

namespace mxnet {
namespace kvstore {

static const int kRowSparsePushPull = 1;
static const int kDefaultPushPull = 0;
static const int kStopServer = -1;
static const int kSyncMode = -2;

/**
 * \brief executor runs a function using the thread called \ref Start
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
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    sync_mode_ = false;
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopServer) {
      exec_.Stop();
    } else if (recved.head == kSyncMode) {
      sync_mode_ = true;
    } else {
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<real_t>& req_data,
                    ps::KVServer<real_t>* server) {
    if (req_meta.cmd == kRowSparsePushPull) {
      DataHandleRowSparse(req_meta, req_data, server);
    } else {
      DataHandleDefault(req_meta, req_data, server);
    }
    return;
  }

  inline void MergeUpdates(const NDArray& recved, int key,
                           std::unordered_set<int> *change_set) {
    auto& merged = merge_buf_[key];
    if (merged.is_none()) {
      merged = NDArray(recved.shape(), Context());
    }
    if (change_set->find(key) == change_set->end()) {
      CopyFromTo(recved, &merged, 0);
    } else {
      // TODO(haibin) handle row sparse gradient NDArray with `ReduceSumCPUExParallel`
      merged += recved;
    }
    change_set->insert(key);
  }

  void DataHandleRowSparse(const ps::KVMeta& req_meta,
                       const ps::KVPairs<real_t>& req_data,
                       ps::KVServer<real_t>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    if (req_meta.push) {
      CHECK_EQ(req_data.lens[0], 0);
      CHECK_GT(req_data.lens.size(), 0);
      auto unit_len = req_data.lens[1];
      CHECK_GT(unit_len, 0);
      real_t* data = req_data.vals.data();
      auto& stored = store_[master_key];
      if (stored.is_none()) {
        // LOG(INFO) << "initial push: " << master_key << " size = " << num_rows * unit_len;
        // initialization
        size_t ds[] = {num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        CHECK_EQ(req_data.vals.size(), num_rows * unit_len);
        TBlob recv_blob(data, dshape, cpu::kDevMask);  // NOLINT(*)
        NDArray recved = NDArray(recv_blob, 0);
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        stored.WaitToRead();
        server->Response(req_meta);
        return;
      }
      // synced push
      if (sync_mode_) {
        // LOG(INFO) << "sync push: " << master_key;
        size_t offset = 0;
        auto& stored = store_[master_key];
        // merge updates
        auto& request_buf = request_buf_[master_key];
        for (size_t i = 1; i <= num_rows; i++) {
          // TODO(haibin) decode once and cache result
          int key = DecodeKey(req_data.keys[i]);
          auto len = req_data.lens[i];
          size_t ds[] = {(size_t)len};
          TShape dshape(ds, ds + 1);
          TBlob recv_blob(data, // NOLINT(*)
                          dshape, cpu::kDevMask);
          NDArray recved = NDArray(recv_blob, 0);
          MergeUpdates(recved, key, &request_buf.change_set);
          offset += len;
        }
        // perform updates
        request_buf.requests.push_back(req_meta);
        if (request_buf.requests.size() == (size_t) ps::NumWorkers()) {
          // let the main thread to execute updater_, which is necessary for python
          for (auto key : request_buf.change_set) {
            // slice a row
            auto row_id = key - master_key;
            NDArray slice = stored.At(row_id);
            NDArray update = merge_buf_[key];
            if (updater_) {
              exec_.Exec([this, key, &update, &slice](){
                  CHECK(updater_);
                  updater_(key, update, &slice);
                });
            } else {
              // if no updater, just copy
              CopyFromTo(update, &slice);
            }
            slice.WaitToRead();
          }
          request_buf.change_set.clear();
          // LOG(INFO) << "RESPONSE SYNC to " << request_buf.requests.size() << " clients";
          for (const auto& req : request_buf.requests) {
            server->Response(req);
          }
          request_buf.requests.clear();
        } else {
          for (size_t i = 1; i <= num_rows; i++) {
            int key = DecodeKey(req_data.keys[i]);
            merge_buf_[key].WaitToRead();
          }
        }
      } else {
        // async push
        auto& stored = store_[master_key];
        for (size_t i = 1; i <= num_rows; i++) {
          int key = DecodeKey(req_data.keys[i]);
          auto row_id = key - master_key;
          auto len = req_data.lens[i];
          size_t ds[] = {(size_t)len};
          TShape dshape(ds, ds + 1);
          TBlob recv_blob(data, // NOLINT(*)
                          dshape, cpu::kDevMask);
          NDArray recved = NDArray(recv_blob, 0);
          NDArray slice = stored.At(row_id);
          exec_.Exec([this, key, &recved, &slice](){
              CHECK(updater_);
              updater_(key, recved, &slice);
            });
        }
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      ps::KVPairs<real_t> response;
      auto& stored = store_[master_key];
      CHECK(!stored.is_none()) << "init " << master_key << " first";
      auto shape = stored.shape();
      auto unit_len = shape.ProdShape(1, shape.ndim());
      const float* data = stored.data().dptr<float>();
      auto len = unit_len * num_rows;
      // LOG(INFO) << "received pull: " << len;
      // concat response values
      response.vals.resize(len);
      for (size_t i = 1; i <= num_rows; i++) {
        int key = DecodeKey(req_data.keys[i]);
        const auto src = data + key * unit_len;
        auto begin = (i - 1) * unit_len;
        auto end = i * unit_len;
        response.vals.segment(begin, end).CopyFrom(src, unit_len);
      }
      // setup response
      response.keys = req_data.keys;
      std::vector<int> lens(req_data.keys.size(), unit_len);
      lens[0] = 0;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
    }
  }

  void DataHandleDefault(const ps::KVMeta& req_meta,
                         const ps::KVPairs<real_t> &req_data,
                         ps::KVServer<real_t>* server) {
    CHECK_EQ(req_meta.cmd, kDefaultPushPull);
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        auto& request_buf = request_buf_[key];
        MergeUpdates(recved, key, &request_buf.change_set);
        request_buf.requests.push_back(req_meta);
        if (request_buf.requests.size() == (size_t) ps::NumWorkers()) {
          CHECK_EQ(request_buf.change_set.size(), 1);
          // let the main thread to execute updater_, which is necessary for python
          if (updater_) {
            exec_.Exec([this, key, &merged, &stored](){
                CHECK(updater_);
                updater_(key, merged, &stored);
              });
          } else {
            // if no updater, just copy
            CopyFromTo(merged, &stored);
          }
          request_buf.change_set.clear();
          for (const auto& req : request_buf.requests) {
            server->Response(req);
          }
          request_buf.requests.clear();
          stored.WaitToRead();
        } else {
          merged.WaitToRead();
        }
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      ps::KVPairs<real_t> response;
      CHECK(!stored.is_none()) << "init " << key << " first";
      auto len = stored.shape().Size();
      response.keys = req_data.keys;
      response.lens = {len};
      // TODO(mli) try to remove this CopyFrom
      response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
      server->Response(req_meta, response);
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  std::unordered_map<int, NDArray> store_;

  struct RequestBuf {
    std::vector<ps::KVMeta> requests;
    std::unordered_set<int> change_set;
    };

  std::unordered_map<int, NDArray> merge_buf_;
  std::unordered_map<int, RequestBuf> request_buf_;


  Executor exec_;

  ps::KVServer<float>* ps_server_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
