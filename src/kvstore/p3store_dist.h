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
 * Copyright (c) 2015 by Contributors
 * @file   p3store_dist.h
 * @brief  priority-based kvstore implementation
 */
#ifndef MXNET_KVSTORE_P3STORE_DIST_H_
#define MXNET_KVSTORE_P3STORE_DIST_H_
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "./kvstore_dist.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed p3store
 */
class P3StoreDist : public KVStoreDist {
 public:
  explicit P3StoreDist(bool use_device_comm)
      : KVStoreDist(use_device_comm) {
    slice_threshold_ = dmlc::GetEnv("MXNET_KVSTORE_SLICE_THRESHOLD", 40 * 1000);
  }

  void PullRowSparse(const std::vector<int>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority) final {
    LOG(FATAL) << "NotImplementedError: PullRowSparse not supported in P3StoreDist.";
  }

  void PullRowSparse(const std::vector<std::string>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority) final {
    LOG(FATAL) << "NotImplementedError: PullRowSparse not supported in P3StoreDist.";
  }

  void set_updater(const Updater& updater) final {
      LOG(FATAL) << "NotImplementedError: Update on P3StoreDist is not supported. "
          << "Please set MXNET_UPDATE_ON_KVSTORE to false.";
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string>>
                              & kwargs) final {
    LOG(FATAL) << "NotImplementedError: Gradient compression not supported in P3StoreDist.";
  }

 private:
  inline void InitKV(const int key, const NDArray& value) override {
    CHECK_EQ(value.storage_type(), kDefaultStorage)
        << "Default storage type for values expected in P3StoreDist";
    comm_->Init(key, value.storage_type(), value.shape(), value.dtype());
    /* Fix the key encoding */
    EncodeDefaultKey(key, value.shape().Size(), mshadow::mshadow_sizeof(value.dtype()));
  }

  void PushCompressed(int key, const NDArray& comm_buf, const PSKV& pskv,
                      int priority) final {
    LOG(FATAL) << "NotImplementedError: PushCompressed not implemented in P3StoreDist.";
  }

  void PushDefault(int key, const NDArray &send_buf, const PSKV& pskv,
                   int priority) override {
    auto push_to_servers = [this, key, pskv, send_buf, priority]
      (RunContext rctx, Engine::CallbackOnComplete cb) {
        const int dtype = send_buf.dtype();
        // convert to ps keys
        const size_t size = send_buf.shape().Size() * mshadow::mshadow_sizeof(dtype);
        char* data = static_cast<char *>(send_buf.data().dptr_);
        // do push. false means no delete
        ps::SArray<char> vals(data, size, false);
        int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);

        size_t off = 0;
        auto counter = new std::atomic<int>(pskv.keys.size());
        for (size_t idx = 0; idx < pskv.keys.size(); idx++) {
          auto ks = pskv.keys.segment(idx, idx+1);
          auto ls = pskv.lens.segment(idx, idx+1);
          auto vs = vals.segment(off, off + pskv.lens[idx]);
          CHECK_NOTNULL(ps_worker_)->ZPush(
            ks, vs, ls, cmd, [counter, cb]() {
                if (--(*counter) == 0) {
                  delete counter;
                  cb();
                }
              }, priority);
          off += pskv.lens[idx];
        }
      };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        "P3StoreDistDefaultPush");
  }

  void PushRowSparse(int key, const NDArray &send_buf, int priority) override {
    LOG(FATAL) << "NotImplementedError: PushRowSparse not implemented in P3StoreDist.";
  }

  void PullDefault(int key, const NDArray &recv_buf, int priority) override {
    CHECK(gradient_compression_->get_type() == CompressionType::kNone)
       << "Gradient compression not supported in P3StoreDist.";
    auto pull_from_servers = [this, key, recv_buf, priority](
        RunContext rctx, Engine::CallbackOnComplete cb) {
      // convert to ps keys
      size_t size = recv_buf.shape().Size();
      const int dtype = recv_buf.dtype();
      const int num_bytes = mshadow::mshadow_sizeof(dtype);
      PSKV& pskv = EncodeDefaultKey(key, size, num_bytes);
      char* data = static_cast<char*> (recv_buf.data().dptr_);
      // issue pull
      const int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
      size_t off = 0;
      auto counter = new std::atomic<int>(pskv.keys.size());
      for (size_t idx = 0; idx < pskv.keys.size(); idx++) {
        auto ks = pskv.keys.segment(idx, idx+1);
        auto ls = new ps::SArray<int>(1, pskv.lens[idx]);
        auto vs = new ps::SArray<char>(data + off, pskv.lens[idx], false);
        CHECK_NOTNULL(ps_worker_)->ZPull(
          ks, vs, ls, cmd, [vs, ls, counter, cb]() {
              delete vs;
              delete ls;
              if (--(*counter) == 0) {
                delete counter;
                cb();
              }
            }, priority);
        off += pskv.lens[idx];
      }
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
        pull_from_servers,
        pinned_ctx_,
        {},
        {recv_buf.var()},
        FnProperty::kNormal,
        priority,
        "P3StoreDistDefaultStoragePull");
  }

  void PullRowSparse_(const int key, const NDArray& recv_buf,
                      const NDArray& indices, int priority) override {
    LOG(FATAL) << "NotImplementedError: PullRowSparse not implemented in P3StoreDist.";
  }

  void PushPullDefault(int key, const NDArray &comm_buf, int priority) override {
    CHECK(gradient_compression_->get_type() == CompressionType::kNone)
             << "Compression not supported in P3StoreDist";
    auto pushpull = [this, key, comm_buf, priority](
        RunContext rctx, Engine::CallbackOnComplete cb) {
      size_t size = comm_buf.shape().Size();
      const int dtype = comm_buf.dtype();
      const int num_bytes = mshadow::mshadow_sizeof(dtype);
      const int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);

      PSKV& pskv = EncodeDefaultKey(key, size, num_bytes);
      char* data = static_cast<char*>(comm_buf.data().dptr_);

      size_t off = 0;
      auto counter = new std::atomic<int>(pskv.keys.size());
      for (size_t idx = 0; idx < pskv.keys.size(); idx++) {
        auto ks = pskv.keys.segment(idx, idx+1);
        auto ls = new ps::SArray<int>(1, pskv.lens[idx]);
        auto vs = new ps::SArray<char>(data + off, pskv.lens[idx], false);
        CHECK_NOTNULL(ps_worker_)->ZPushPull(
          ks, *vs, vs, ls, cmd, [vs, ls, counter, cb]() {
              delete vs;
              delete ls;
              if (--(*counter) == 0) {
                delete counter;
                cb();
              }
            }, priority);
        off += pskv.lens[idx];
      }
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
        pushpull,
        pinned_ctx_,
        {},
        {comm_buf.var()},
        FnProperty::kNormal,
        priority,
        "P3StoreDistDefaultStoragePushPull");
  }

  inline PSKV& EncodeDefaultKey(const int key, const size_t num_arr_elems,
                                const int num_bytes) override {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    size_t pskv_size = num_arr_elems * num_bytes;
    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), pskv_size)
        << "The value size cannot be changed " << pskv_size << ". Key is " << key;
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      const int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      int64_t num_params = num_arr_elems * num_bytes;
      int64_t slice_bound = slice_threshold_ * num_bytes;
      static size_t server = 0;
      while (num_params > 0) {
        ps::Key ps_key = krs[server%num_servers].begin()
                         + (ps::Key)(key + server/num_servers);
        CHECK_LT(ps_key, krs[server%num_servers].end());
        pskv.keys.push_back(ps_key);
        const size_t part_size = static_cast<size_t>((num_params > slice_bound)
                ? slice_bound : num_params);
        pskv.lens.push_back(part_size);
        pskv.size += part_size;

        num_params -= part_size;
        server++;
      }
    }
    return pskv;
  }

  /**
   * \brief threshold for the parameter slice size
   */
  size_t slice_threshold_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_P3STORE_DIST_H_
