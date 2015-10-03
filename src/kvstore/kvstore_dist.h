/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_

#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps.h"

namespace mxnet {
namespace kvstore {

class KVStoreDist : public KVStoreLocal {
 public:
  KVStoreDist() {
    engine_ = Engine::Get();
    cache_ = CHECK_NOTNULL((new ps::KVCache<ps::Key, real_t>(ps::NextID())));
  }

  virtual ~KVStoreDist() {

  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& merged = MergePushValue(key, grouped_vals[i], priority);

      // push to servers
      auto push_to_servers =
          [this, key, merged](RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = merged.shape().Size();
        ps::SArray<ps::Key> keys;
        ps::SArray<int> vals_size;
        EncodeKey(key, size, &keys, &vals_size);

        // do push
        real_t* data = static_cast<real_t*>(merged.data().dptr_);
        ps::SArray<real_t> vals(data, size, ps::EmptyDel<real_t>());
        ps::SyncOpts opts;
        // TODO(mli) add filters to reduce the bandwidth
        opts.callback = [cb]() { cb(); };
        last_push_[key] =
          cache_->Push(opts.GetTask(), keys, vals, vals_size, opts.callback);
      };
      engine_->PushAsync(push_to_servers, pinned_ctx_, {merged.var()}, {},
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

      // first pull to vals[0], and copy to others
      auto pull_from_servers = [this, key, vals, priority](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = vals[0]->shape().Size();
        ps::SArray<ps::Key> keys;
        ps::SArray<int> vals_size;
        EncodeKey(key, size, &keys, &vals_size);

        // do pull
        ps::SyncOpts opts;
        auto it = last_push_.find(key);
        if (it != last_push_.end()) {
          // make sure pull happens after the previous push on this key
          opts.deps.push_back(it->second);
        }
        // TODO(mli) add filters to reduce the bandwidth
        opts.callback = [cb, vals, priority]() {
          for (size_t i = 1; i < vals.size(); ++i) {
            CopyFromTo(*vals[0], vals[i], priority);
          }
          cb();
        };
        real_t* data = static_cast<real_t*>(vals[0]->data().dptr_);
        cache_->Pull(opts.GetTask(), keys, vals_size, opts.callback, data);
      };

      std::vector<Engine::VarHandle> mut_vars;
      mut_vars.reserve(vals.size());
      for (auto& v : vals) mut_vars.push_back(v->var());
      engine_->PushAsync(pull_from_servers, pinned_ctx_, {}, mut_vars,
                         FnProperty::kNormal, priority);
    }
  }

  int get_group_size() const override {
    return ps::NodeInfo::RankSize();
  }
  int get_rank() const override {
    return ps::NodeInfo::MyRank();
  }

  bool is_distributed() const override {
    return true;
  }

 private:
  /**
   * \brief convert to a key in ps
   */
  inline void EncodeKey(int key, size_t size,
                        ps::SArray<ps::Key>* keys,
                        ps::SArray<int>* vals_size) {

  }

  /**
   * \brief convert from a key in ps
   */
  inline int DecodeKey(ps::Key key) {
    return 0;
  }

  /**
   * \brief for push and pull
   * use KVCache rather than KVWorker for more advanced push and pull
   */
  ps::KVCache<ps::Key, real_t>* cache_;

  Engine* engine_;

  /**
   * \brief the timestamps for the last push
   */
  std::unordered_map<int, int> last_push_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif /* MXNET_KVSTORE_KVSTORE_DIST_H_ */
