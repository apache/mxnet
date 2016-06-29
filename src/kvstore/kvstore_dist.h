/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include "./kvstore_device.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"

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
class KVStoreDist : public KVStoreDevice {
 public:
  explicit KVStoreDist(bool device_mode)
      : KVStoreDevice(device_mode),
        ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::Start("mxnet\0");
    }
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
      if (get_rank() == 0) {
        // stop the executor at servers
        SendCommandToServers(kStopServer, "");
      }
      ps::Finalize();
      delete ps_worker_;
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    if (get_rank() == 0) {
      Push(keys, values, 0);
      // wait until the push is finished
      Wait(keys);
    } else {
      // do nothing
    }
    Barrier();
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
              int priority) override {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];
      const NDArray& merged = MergePushValue(key, grouped_vals[i], priority);

      // push to servers
      auto push_to_servers =
          [this, key, merged](RunContext rctx, Engine::CallbackOnComplete cb) {
         // convert to ps keys
        size_t size = merged.shape().Size();
        PSKV& pskv = EncodeKey(key, size);

        // do push
        real_t* data = static_cast<real_t*>(merged.data().dptr_);
        // false means no delete
        ps::SArray<real_t> vals(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPush(
        pskv.keys, vals, pskv.lens, 0, [cb]() { cb(); });
      };
      Engine::Get()->PushAsync(
          push_to_servers,
          pinned_ctx_,
          {merged.var()},
          {},
          FnProperty::kNormal, priority);
    }
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];

      // first pull to a buffer. we reuse the merge buf so that all pushes and
      // pulls on the same key on the local machine are always sequentials
      auto& buf = merge_buf_[key].merged;
      if (buf.is_none()) {
        buf = NDArray(vals[0]->shape(), pinned_ctx_);
      }

      auto pull_from_servers = [this, key, buf] (
          RunContext rctx, Engine::CallbackOnComplete cb) {
        real_t* data = static_cast<real_t*>(buf.data().dptr_);
        size_t size = buf.shape().Size();
        // convert to ps keys
        PSKV& pskv = EncodeKey(key, size);

        // issue pull, false means no delete
        auto vals = new ps::SArray<real_t>(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPull(
        pskv.keys, vals, &pskv.lens, 0, [vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {buf.var()},
          FnProperty::kNormal, priority);

      ScatterPullValue(key, buf, vals, priority);
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

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::Start("mxnet_server\0");
    if (server_) server_->Run();
    ps::Finalize();
    delete server_; server_ = nullptr;
  }

 private:
  /**
   * \brief Wait until all pushes and pulls issued on each key have been
   * finished
   *
   * \param keys a list of keys
   */
  void Wait(const std::vector<int>& keys) {
    for (int key : keys) {
      auto it = merge_buf_.find(key);
      CHECK(it != merge_buf_.end())
          << "there is no push/pull on key " << key << " before";
      CHECK(!it->second.merged.is_none())
          << "there is no push/pull on key " << key << " before";
      it->second.merged.WaitToWrite();
    }
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
   * \brief serizelize EncodeKey
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
              static_cast<size_t>(static_cast<double>(size)/num_servers*(i+1)) -
              static_cast<size_t>(static_cast<double>(size)/num_servers*i);
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

  // whether use device distributed local sync.
  bool device_mode_;
  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;

  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
