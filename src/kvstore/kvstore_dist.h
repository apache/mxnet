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

/**
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * for a worker node, it always guarantees that all push and pull issued from
 * this worker on the same key are serialized. namely push(3) and then pull(3),
 * then the data pulled is always containing the modification from the push(3).
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier();
        if (get_rank() == 0) {
          // stop the executor at servers
          SendCommandToServers(kStopServer, "");
        }
      }
      ps::Finalize(barrier_before_exit_);
      delete ps_worker_;
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0) {
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const auto& v : values) {
        v.WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    Push_(keys, values, priority, true);
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
               << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                           true, grouped_vals[i][0]->dtype());
      }
      auto pull_from_servers = [this, key, recv_buf](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = recv_buf.shape().Size();
        PSKV& pskv = EncodeKey(key, size);
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(recv_buf.data());
#endif
        real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);
        // issue pull
        CHECK_NOTNULL(ps_worker_)->ZPull(
          pskv.keys, vals, &pskv.lens, kDefaultPushPull, [vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal,
          priority,
          PROFILER_MESSAGE("KVStoreDistDefaultPull"));

      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }

  void PullRowSparse(const std::vector<int>& keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     const int priority = 0) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<std::pair<NDArray*, NDArray>>> grouped_val_rowids;
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      auto& grouped_val_rowid = grouped_val_rowids[i];
      const auto storage_type = grouped_val_rowid[0].first->storage_type();
      CHECK_EQ(storage_type, kRowSparseStorage)
               << "expected kRowSparseStorage, but got " << storage_type;
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(storage_type, grouped_val_rowid[0].first->shape(),
                           pinned_ctx_, true, grouped_val_rowid[0].first->dtype());
      }
      auto &target_val_rowids = grouped_val_rowids[i];
      const size_t num_vals = target_val_rowids.size();
      size_t num_rows = 0;
      // TODO(haibin) refactor this for loop
      for (size_t i = 0; i < num_vals; i++) {
        auto &row_id = target_val_rowids[i].second;
        NDArray indices = row_id.Copy(pinned_ctx_);
        Unique(&indices, priority);
        target_val_rowids[i].second = indices;
        num_rows += indices.shape().Size();
      }
      if (num_vals > 1) {
        // TODO(haibin) aggregate over all unique indices
        LOG(FATAL) << "RowSparsePull with multiple values is not implemented yet";
      } else {
        auto& indices = target_val_rowids[0].second;
        PullRowSparse_(key, &recv_buf, indices, priority);
        comm_->BroadcastRowSparse(key, recv_buf, grouped_val_rowid, num_vals == 1, priority);
      }
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }


  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync("mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
    if (server_) server_->Run();
    ps::Finalize();
    if (server_) {
      delete server_;
    }
    server_ = nullptr;
  }

 private:
  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge)  {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

      auto& send_buf = comm_buf_[key];
      const auto storage_type = merged.storage_type();
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        // make sure the previous push/pull is completed
        send_buf.WaitToWrite();
        send_buf = merged;  // avoid memory copy
      } else {
        if (send_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            send_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
          } else {
            send_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
          }
        }
        CopyFromTo(merged, &send_buf);
      }

      // push to servers
      if (storage_type == kDefaultStorage) {
      auto push_to_servers =
          [this, key, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
          // convert to ps keys
          size_t size = send_buf.shape().Size();
          PSKV& pskv = EncodeKey(key, size);

#if MKL_EXPERIMENTAL == 1
          mkl_set_tblob_eager_mode(send_buf.data());
#endif
          real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
          // do push. false means no delete
          ps::SArray<real_t> vals(data, size, false);
          CHECK_NOTNULL(ps_worker_)->ZPush(
              pskv.keys, vals, pskv.lens, 0, [cb]() { cb(); });
        };
        Engine::Get()->PushAsync(
            push_to_servers,
            pinned_ctx_,
            {send_buf.var()},
            {},
            FnProperty::kNormal,
            priority,
            PROFILER_MESSAGE("KVStoreDistDefaultPush"));
      } else if (storage_type == kRowSparseStorage) {
        PushRowSparse(key, send_buf, priority);
      } else {
        LOG(FATAL) << "unknown storage type";
      }
    }
  }

  // pull row sparse weight into `recv_buf` based on indices given by `indices`
  void PullRowSparse_(int key, NDArray *recv_buf, const NDArray& indices, int priority) {
    using namespace rowsparse;
    auto pull_from_servers = [this, key, recv_buf, indices]
                             (RunContext rctx, Engine::CallbackOnComplete cb) {
      // allocate memory for the buffer
      size_t num_rows = indices.shape().Size();
      recv_buf->CheckAndAlloc({mshadow::Shape1(num_rows)});
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(recv_buf->data());
#endif
      real_t* data = static_cast<real_t*>(recv_buf->data().dptr_);
      auto indices_data = indices.data();
      const auto offsets = indices_data.dptr<int64_t>();
      const auto unit_len = recv_buf->shape().ProdShape(1, recv_buf->shape().ndim());
      const int64_t size = num_rows * unit_len;
       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, recv_buf->shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << get_rank() << " pull lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      auto vals = new ps::SArray<real_t>(data, size, false);
      CHECK_NOTNULL(ps_worker_)->ZPull(pskv.keys, vals, &pskv.lens, kRowSparsePushPull,
        [vals, cb]() { delete vals; cb(); });
      // copy indices to recv_buf
      mshadow::Copy(recv_buf->aux_data(kIdx).FlatTo1D<cpu, int64_t>(),
                    indices_data.FlatTo1D<cpu, int64_t>());
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
        pull_from_servers,
        pinned_ctx_,
        {indices.var()},
        {recv_buf->var()},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePull"));
  }

  // push row sparse gradient
  void PushRowSparse(int key, const NDArray &send_buf, int priority) {
    using namespace rowsparse;
    auto push_to_servers = [this, key, &send_buf]
                           (RunContext rctx, Engine::CallbackOnComplete cb) {
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(send_buf.data());
#endif
      real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
      bool init = send_buf.storage_initialized();
      const int64_t num_rows = init ? send_buf.aux_shape(kIdx)[0] : 0;
      const auto offsets = init ? send_buf.aux_data(kIdx).dptr<int64_t>() : nullptr;
      const auto unit_len = send_buf.shape().ProdShape(1, send_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;

       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, send_buf.shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << get_rank() << " push lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      ps::SArray<real_t> vals(data, size, false);
      CHECK_NOTNULL(ps_worker_)->ZPush(pskv.keys, vals, pskv.lens, kRowSparsePushPull, [cb]() {
        cb();
      });
    };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePush"));
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

  /**
   * \brief cache all key partitions
   */
  std::unordered_map<int, PSKV> ps_kv_;

  /**
   * \brief serizelize EncodeRowSparseKey and EncodeKey
   */
  std::mutex mu_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeKey(int key, size_t size) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
      } else {
        // parition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
              static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
              static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  // TODO(haibin) this encoding method for row sparse keys doesn't allow cross-layer batching
  inline PSKV& EncodeRowSparseKey(const int key, const int64_t size, const int64_t num_rows,
                                  const int64_t *offsets, const size_t unit_len,
                                  const int64_t total_num_rows) {
    using namespace common;
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    pskv.keys.clear();
    pskv.lens.clear();
    // TODO(haibin) cache this information
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    int num_servers = krs.size();
    CHECK_GT(num_servers, 0);

    if (total_num_rows * unit_len >= bigarray_bound_) {
      pskv.size = 0;
      int64_t start_row = 0;
      // parition it to all servers
      for (int i = 0; i < num_servers; ++i) {
        // calculate partition ranges
        int64_t part_num_rows =
            llround(static_cast<double>(total_num_rows) / num_servers * (i + 1)) -
            llround(static_cast<double>(total_num_rows) / num_servers * i);
        auto end_row = start_row + part_num_rows;
        auto lb = std::lower_bound(offsets, offsets + num_rows, start_row);
        auto ub = std::upper_bound(offsets, offsets + num_rows, end_row - 1);
        ps::Key master_key = krs[i].begin() + key;
        pskv.keys.push_back(master_key);
        pskv.lens.push_back(0);
        for (auto offset = lb; offset < ub; offset++) {
          ps::Key ps_key = krs[i].begin() + key + (*offset - start_row);
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(unit_len);
          pskv.size += unit_len;
        }
        start_row = end_row;
      }
      CHECK_EQ(static_cast<size_t>(pskv.size), size);
    } else {
      // send it to a single random picked server
      int server = (key * 9973) % num_servers;
      ps::Key master_key = krs[server].begin() + key;
      pskv.keys.push_back(master_key);
      pskv.lens.push_back(0);
      for (int64_t i = 0; i < num_rows; i++) {
        ps::Key ps_key = krs[server].begin() + key + offsets[i];
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(unit_len);
      }
      pskv.size = size;
    }
    return pskv;
  }


  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /// \brief send & recver buffer
  std::unordered_map<int, NDArray> comm_buf_;
  bool log_verbose_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
