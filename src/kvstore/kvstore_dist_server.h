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
#include "../operator/tensor/elemwise_binary_op.h"
#include "../operator/tensor/init_op.h"

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
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
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
  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
  };

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

  inline void ApplyUpdates(const int key, MergeBuf *merged, NDArray *stored,
                           ps::KVServer<real_t>* server) {
    if (merged->request.size() == (size_t) ps::NumWorkers()) {
      // let the main thread to execute updater_, which is necessary for python
      if (updater_) {
        exec_.Exec([this, key, merged, stored](){
            CHECK(updater_);
            updater_(key, merged->array, stored);
          });
      } else {
        // if no updater, just copy
        CopyFromTo(merged->array, stored);
      }
      if (log_verbose_)  {
        LOG(INFO) << "sync response to " << merged->request.size() << " workers";
      }
      for (const auto& req : merged->request) {
        server->Response(req);
      }
      merged->request.clear();
      stored->WaitToRead();
    } else {
      merged->array.WaitToRead();
    }
  }

  void DecodeRowIds(const ps::SArray<ps::Key> &keys, int64_t *indices,
                    const int64_t master_key, const int64_t num_rows) {
    indices[0] = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(keys[i]);
      auto row_id = key - master_key;
      indices[i - 1] = row_id;
    }
  }

  void DataHandleRowSparse(const ps::KVMeta& req_meta,
                       const ps::KVPairs<real_t>& req_data,
                       ps::KVServer<real_t>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key];
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      real_t* data = req_data.vals.data();
      if (stored.is_none()) {
        if (log_verbose_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        size_t ds[] = {num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        CHECK_EQ(req_data.vals.size(), num_rows * unit_len);
        TBlob recv_blob(data, dshape, cpu::kDevMask);  // NOLINT(*)
        NDArray recved = NDArray(recv_blob, 0);
        stored = NDArray(kRowSparseStorage, dshape, Context());
        Engine::Get()->PushSync([recved, stored](RunContext ctx) {
            NDArray rsp = stored;
            stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
            mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
            op::PopulateFullIdxRspImpl(s, &rsp);
            mshadow::Copy(rsp.data().FlatTo1D<cpu, float>(),
                          recved.data().FlatTo1D<cpu, float>(), s);
          }, recved.ctx(), {recved.var()}, {stored.var()},
          FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
        stored.WaitToRead();
        server->Response(req_meta);
        return;
      }
      // synced push
      if (sync_mode_) {
        if (log_verbose_) LOG(INFO) << "sync push: " << master_key << " " << req_data.keys;
        auto& merged = merge_buf_[master_key];
        if (merged.array.is_none()) {
          merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
        }
        if (num_rows == 0) {
          // reset to zeros
          if (merged.request.size() == 0) {
            merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
          } else {
            // nothing to aggregate
          }
          merged.request.push_back(req_meta);
          ApplyUpdates(master_key, &merged,  &stored, server);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        // data
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        // row_sparse NDArray
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          NDArray out(kRowSparseStorage, stored.shape(), Context());
          std::vector<Engine::VarHandle> const_vars;
          const_vars.push_back(recved.var());
          const_vars.push_back(merged.array.var());
          // accumulate row_sparse gradients
          // TODO(haibin) override + operator for row_sparse NDArray
          // instead of calling BinaryComputeRspRsp directly
          using namespace mshadow;
          Engine::Get()->PushSync([recved, merged, out](RunContext ctx) {
              std::vector<NDArray> inputs, outputs;
              inputs.push_back(recved);
              inputs.push_back(merged.array);
              outputs.push_back(out);
              op::BinaryComputeRspRspImpl<cpu, cpu>({}, {}, inputs, {kWriteTo}, outputs);
            }, recved.ctx(), const_vars, {out.var()},
            FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
          CopyFromTo(out, &merged.array, 0);
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(master_key, &merged,  &stored, server);
      } else {
        // async push
        if (log_verbose_) LOG(INFO) << "async push: " << master_key;
        if (num_rows == 0) {
          server->Response(req_meta);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);
        exec_.Exec([this, master_key, &recved, &stored](){
            CHECK(updater_);
            updater_(master_key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      if (log_verbose_) LOG(INFO) << "pull: " << master_key;
      ps::KVPairs<real_t> response;
      if (num_rows == 0) {
        std::vector<int> lens(req_data.keys.size(), 0);
        response.keys = req_data.keys;
        response.lens.CopyFrom(lens.begin(), lens.end());
        server->Response(req_meta, response);
        return;
      }
      CHECK(!stored.is_none()) << "init " << master_key << " first";
      auto shape = stored.shape();
      auto unit_len = shape.ProdShape(1, shape.ndim());
      const float* data = stored.data().dptr<float>();
      auto len = unit_len * num_rows;
      // concat values
      response.vals.resize(len);
      for (size_t i = 1; i <= num_rows; i++) {
        int key = DecodeKey(req_data.keys[i]);
        int64_t row_id = key - master_key;
        const auto src = data + row_id * unit_len;
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
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          merged.array += recved;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, &merged, &stored, server);
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
  std::unordered_map<int, MergeBuf> merge_buf_;

  Executor exec_;
  ps::KVServer<float>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
