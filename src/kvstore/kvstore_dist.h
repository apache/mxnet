/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_

#include "./kvstore_local.h"
#include "./mxnet_node.h"
#include "mxnet/engine.h"
#include "ps.h"
#include "base/range.h"

namespace mxnet {
namespace kvstore {

class KVStoreDist : public KVStoreLocal {
 public:
  KVStoreDist()
      : store_(NULL),
        cache_(NULL),
        barrier_count_(0) {
    if (IsServerNode()) {
      ServerHandle handle(this);
      store_ = new ps::OnlineServer<real_t, ServerVal, ServerHandle>(handle);
    } else if (IsWorkerNode()) {
      cache_ = new ps::KVCache<ps::Key, real_t>(ps::NextID());
      StartPS();
    }
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    // need to explicit clear the NDArray before Engine is deleted
    if (store_) store_->server()->Clear();
    delete store_;
    delete cache_;

    if (IsWorkerNode()) {
      ps::StopSystem();
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    Push(keys, values, 0);
    // wait until the push is finished
    Wait(keys);
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
        ps::SArray<real_t> vals(data, size, ps::EmptyDel<real_t>());
        ps::SyncOpts opts;
        opts.callback = [cb]() { cb(); };
        CHECK_NOTNULL(cache_)->Push(
            opts.GetTask(),
            pskv.keys,
            vals,
            pskv.vals_size,
            opts.callback);
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
      real_t* data = static_cast<real_t*>(buf.data().dptr_);
      size_t size = buf.shape().Size();

      auto pull_from_servers = [this, key, data, size](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        PSKV& pskv = EncodeKey(key, size);

        // pull opts
        ps::SyncOpts opts;
        opts.callback = [cb]() { cb(); };

        // issue pull
        CHECK_NOTNULL(cache_)->Pull(
            opts.GetTask(),
            pskv.keys,
            opts.callback,
            data,
            size,
            pskv.vals_size.data());
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {buf.var()},
          FnProperty::kNormal, priority);

      // copy data from buffer to vals
      for (auto v : vals) {
        CopyFromTo(buf, v);
      }
    }
  }

  void Barrier() override {
    KVStoreCommand cmd;
    cmd.set_barrier(barrier_count_++);
    ps::Task task;
    task.set_cmd(cmd.cmd);
    auto node = CHECK_NOTNULL(ps::NodeInfo::MyApp());
    node->Wait(node->Submit(task, ps::NodeInfo::SchedulerID()));
  }

  void Wait(const std::vector<int>& keys) override {
    for (int key : keys) {
      auto it = merge_buf_.find(key);
      CHECK(it != merge_buf_.end())
          << "there is no push/pull on key " << key << " before";
      CHECK(!it->second.merged.is_none())
          << "there is no push/pull on key " << key << " before";
      it->second.merged.WaitToWrite();
    }
  }

  void WaitAll() override {
    for (auto& buf : merge_buf_) {
      if (!buf.second.merged.is_none()) {
        buf.second.merged.WaitToWrite();
      }
    }
  }

  void SendCommandToServers(int head, const char* body) override {
    CHECK_GE(head, 0) << "negative head is preserved for system usage";
    ps::Task task;
    task.set_cmd(head);
    task.set_msg(body);
    auto node = CHECK_NOTNULL(ps::NodeInfo::MyApp());
    node->Wait(node->Submit(task, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NodeInfo::RankSize(); }

  int get_rank() const override { return ps::NodeInfo::MyRank(); }

  bool IsDistributed() const override { return true; }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    StartPS();
    if (IsServerNode()) {
      auto node = CHECK_NOTNULL(ps::NodeInfo::MyApp());
      static_cast<MXNetServer*>(node)->set_controller(controller);
    }
    ps::StopSystem();
  }

 private:

  /**
   * \brief start the network threads in ps-lite
   */
  void StartPS() {
    // hack argc argv
    int argc = 1;
    char** argv = new char*[1];
    char name[] = "mxnet";
    argv[0] = new char[strlen(name)+1];
    memcpy(argv[0], name, strlen(name));
    argv[0][strlen(name)] = '\0';
    ps::StartSystem(&argc, &argv);
  }

  /**
   * \brief struct for ps keys and vals_size
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> vals_size;  // the length of the i-th value
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
   * \brief key partition of server nodes in ps
   */
  std::vector<ps::Key> server_key_partition_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeKey(int key, size_t size) {
    CHECK_EQ(sizeof(ps::Key), 8) << "Do not use USE_KEY32=1 to compile ps-lite";
    int num_servers = ps::NodeInfo::NumServers();
    CHECK_GT(num_servers, 0);

    mu_.lock();
    // init key parititon
    if (server_key_partition_.empty()) {
      auto all = ps::Range<ps::Key>::All();
      for (int i = 0; i < num_servers; ++i) {
        ps::Key key = all.EvenDivide(num_servers, i).begin();
        server_key_partition_.push_back(((key >> kIndexBits)+1) << kIndexBits);
      }
    }

    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    if (!pskv.keys.empty()) {
      CHECK_EQ(pskv.size, size) << "The value size cannot be changed";
    } else {
      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        pskv.keys.push_back(server_key_partition_[server] | key);
        pskv.vals_size.push_back(size);
      } else {
        // divide it to all servers
        auto all = ps::Range<size_t>(0, size);
        for (int i = 0; i < num_servers; ++i) {
          pskv.keys.push_back(server_key_partition_[i] | key);
          pskv.vals_size.push_back(all.EvenDivide(num_servers, i).size());
        }
      }
      pskv.size = size;
    }
    return pskv;
  }

  /**
   * \brief convert from a key in ps
   */
  inline int DecodeKey(ps::Key key) {
    return static_cast<int>((key << kIndexBits) >> kIndexBits);
  }

  /**
   * \brief value type stored at server
   */
  struct ServerVal {
    NDArray array;
    inline void Load(dmlc::Stream *fi) { array.Load(fi); }
    inline void Save(dmlc::Stream *fo) const { array.Save(fo); }
    inline bool Empty() const { return array.is_none(); }
  };

  /**
   * \brief server handle
   */
  class ServerHandle {
   public:
    ServerHandle(KVStoreDist* kvstore) {
      kvstore_ = kvstore;
    }

    inline void Start(bool push, int timestamp, int cmd, void* msg) { }
    inline void Finish() { }
    inline void Load(dmlc::Stream *fi) { }
    inline void Save(dmlc::Stream *fo) const { }

    inline void Push(
        ps::Key recv_key, ps::Blob<const real_t> recv_val, ServerVal& my_val) {
      // construct NDArray without data copy
      size_t ds[] = {recv_val.size};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)recv_val.data, dshape, cpu::kDevMask);
      NDArray recv_array(recv_blob, 0);

      if (my_val.Empty()) {
        // initialization
        my_val.array = NDArray(dshape, Context());
        // my_val.array.SyncCopyFromCPU(recv_val.data, recv_val.size);
        CopyFromTo(recv_array, &my_val.array);
        // LOG(ERROR) << "init";
      } else {
        // call updater
        int key = kvstore_->DecodeKey(recv_key);
        if (kvstore_->updater_) {
          kvstore_->updater_(key, recv_array, &my_val.array);
        } else {
          CopyFromTo(recv_array, &my_val.array);
        }
      }
    }

    inline void Pull(
        ps::Key recv_key, const ServerVal& my_val, ps::Blob<real_t>& send_val) {
      CHECK(!my_val.Empty())
          << kvstore_->DecodeKey(recv_key) << " is not inited";

      my_val.array.WaitToRead();
      send_val.data = static_cast<real_t*>(my_val.array.data().dptr_);
      send_val.size = my_val.array.shape()[0];
      LOG(ERROR) << send_val.data[0] << " " << send_val.data[send_val.size-1];
    }

   private:
    KVStoreDist* kvstore_;
  };

  /**
   * \brief kv store at server node
   */
  ps::OnlineServer<real_t, ServerVal, ServerHandle>* store_;

  /**
   * \brief for worker to push and pull data
   * use KVCache rather than KVWorker for the c-style pull
   */
  ps::KVCache<ps::Key, real_t>* cache_;

  /**
   * \brief number of bits used to encode the key in mxnet
   */
  static const int kIndexBits = 32;

  /**
   * \brief the count for barrier
   */
  int barrier_count_;

};

}  // namespace kvstore
}  // namespace mxnet


#endif /* MXNET_KVSTORE_KVSTORE_DIST_H_ */
