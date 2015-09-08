/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_base.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_BASE_H_
#define MXNET_KVSTORE_BASE_H_
#include <unordered_map>
#include <bitset>
#include "mxnet/narray.h"
#include "mxnet/dag_engine.h"

namespace mxnet {

/**
 * \brief store data in local machine
 */
class KVStoreBase {
 public:
  typedef int Key;
  KVStoreBase() : inited_(false), engine_(DAGEngine::Get()), aggregate_(true) { }
  virtual ~KVStoreBase() { }
  virtual void InitDevices(const std::vector<Context>& devices) {
    CHECK(!inited_) << "double intializatino";
    num_devs_ = 0;
    for (auto d : devices) devs_[d.UID()] = num_devs_ ++;
    inited_ = true;
  }

  virtual void Push(Key key, const NArray& value, bool init) {
    CHECK(inited_) << "call InitDevices first";
    auto it = local_.find(key);
    if (init) {
      CHECK(it == local_.end()) << "duplicate init of key = " << key;
      Value val(num_devs_, value.Copy(local_ctx_));
      local_.insert({key, val}).first;
      return;
    }
    CHECK(it != local_.end()) << "key " << key << " has not been inited";
    auto& local_val = it->second;
    CHECK_EQ(local_val.arr.shape(), value.shape())
        << "shape mismatch: " << local_val.arr.shape() << ", " << value.shape();
    if (aggregate_) {
      int dix = GetDevIdx(value.ctx());
      CHECK(!local_val.pending_push[dix])
          << "duplicate push on key " << key << "from " << value.ctx().Name();
      local_val.pending_push[dix] = true;
      local_val.pending_push_arr.push_back(value);
      if (local_val.pending_push_arr.size() == num_devs_) {
        // do copy for the clossure
        std::vector<NArray> read;
        std::swap(read, local_val.pending_push_arr);
        std::vector<DAGEngine::Variable> read_val;
        for (const auto& r : read) read_val.push_back(r.var());
        NArray write = local_val.arr;

        // issue push to engine
        engine_->Push([this, read, write](RunContext rctx) mutable {
            for (const auto& r : read) write += r;
          }, local_ctx_, read_val, {write.var()});

        // issue pull if necessary
        for (auto& w : local_val.pending_pull_arr) {
          CopyFromTo(local_val.arr, &w);
        }

        // clean
        local_val.pending_push.flip();
        local_val.pending_pull_arr.clear();
      }
    } else {
      LOG(FATAL) << "TODO";
    }
  }

  virtual void Pull(Key key, NArray* value) {
    CHECK(inited_) << "call InitDevices first";

    auto it = local_.find(key);
    CHECK(it != local_.end()) << "key " << key << " has not been inited";
    auto& local_val = it->second;
    CHECK_EQ(local_val.arr.shape(), value->shape())
        << "shape mismatch: " << local_val.arr.shape() << ", " << value->shape();

    if (aggregate_) {
      int dix = GetDevIdx(value->ctx());
      if (local_val.pending_push[dix]) {
        local_val.pending_pull_arr.push_back(*value);
        return;
      }
      CopyFromTo(local_val.arr, value);
    }
  }

  virtual int GetRank() { return 0; }
  virtual int GetGroupSize() { return 1; }

 protected:
  /// get the continous device index starting from 0
  inline int GetDevIdx(const Context& ctx) {
    auto it = devs_.find(ctx.UID());
    CHECK(it != devs_.end())
        << "unknow device " << ctx.Name();
    return it->second;
  }
  bool inited_;
  DAGEngine* engine_;
  bool aggregate_;

  /// map a device into an index
  size_t num_devs_;
  std::unordered_map<uint64_t, int> devs_;

  /// internal storage of a value
  struct Value {
    Value() {}
    Value(int num_devs, NArray data)
        : pending_push(num_devs, false), pending_pull(num_devs, false) {
      arr = data;
    }
    std::vector<bool> pending_push, pending_pull;
    std::vector<NArray> pending_push_arr, pending_pull_arr;
    NArray arr;
  };
  Context local_ctx_;
  std::unordered_map<Key, Value> local_;
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_BASE_H_
