/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore_nccl.h
 * \brief Implementation of KVStore based on nvida/nccl
 */
#ifndef MXNET_KVSTORE_KVSTORE_NCCL_H_
#define MXNET_KVSTORE_KVSTORE_NCCL_H_

#include <nccl.h>
#include <mxnet/kvstore.h>
#include <algorithm>
#include <vector>

namespace mxnet {
namespace kvstore {

#define CUDACHECK(cmd) do {                              \
    cudaError_t e = cmd;                                 \
    CHECK(e == cudaSuccess) << "CUDA failure "           \
                            << cudaGetErrorString(e);    \
  } while(false)

#define NCCLCHECK(cmd) do {                             \
    ncclResult_t e = cmd;                               \
    CHECK(e == ncclSuccess) << "NCLL failure "          \
                            << ncclGetErrorString(e);   \
  } while(false)

class KVStoreNCCL : public KVStore {
 public:
  KVStoreNCCL() {
    inited_nccl_ = false;
  }

  virtual ~KVStoreNCCL() {
    for (size_t i = 0; i < devs_.size(); ++i) {
      CUDACHECK(cudaSetDevice(devs_[i]));
      CUDACHECK(cudaStreamDestroy(streams_[i]));
    }
    for (auto c : comms_) {
      ncclCommDestroy(c);
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
  }


  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(FATAL) << "sorry, not implemented yet";
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    LOG(FATAL) << "sorry, not implemented yet";
  }


  void PushPull(const std::vector<int>& keys,
                const std::vector<NDArray>& push_values,
                const std::vector<NDArray*>& pull_values,
                int priority) override {
    // group values on the same key
    CHECK_EQ(push_values.size(), pull_values.size());
    std::vector<int> uniq_push_keys, uniq_pull_keys;
    std::vector<std::vector<NDArray> > grouped_push_vals;
    std::vector<std::vector<NDArray*> > grouped_pull_vals;
    GroupKVPairs(keys, push_values, &uniq_push_keys, &grouped_push_vals);
    GroupKVPairs(keys, pull_values, &uniq_pull_keys, &grouped_pull_vals);

    // do allreduce one by one
    CHECK_EQ(uniq_push_keys.size(), uniq_pull_keys.size());
    for (size_t i = 0; i < uniq_push_keys.size(); ++i) {
      CHECK_EQ(uniq_push_keys[i], uniq_pull_keys[i]);
      const auto& push_vals = grouped_push_vals[i];
      const auto& pull_vals = grouped_pull_vals[i];
      CHECK_EQ(push_vals.size(), pull_vals.size());
      size_t n = push_vals.size();

      // init
      if (!inited_nccl_) {
        devs_.resize(n);
        for (size_t j = 0; j < n; ++j) {
          CHECK_EQ(push_vals[j].ctx().dev_type, Context::kGPU);
          devs_[j] = push_vals[j].ctx().dev_id;
        }
        InitNCCL();
      } else {
        CHECK_EQ(devs_.size(), n);
      }

      // prepare data
      size_t len = push_vals[0].shape().Size();
      std::vector<engine::VarHandle> cvars, mvars(n);
      std::vector<void*> sendbuff(n), recvbuff(n);
      for (size_t j = 0; j < n; ++j) {
        CHECK_EQ(push_vals[j].ctx().dev_type, Context::kGPU);
        CHECK_EQ(pull_vals[j]->ctx().dev_type, Context::kGPU);
        CHECK_EQ(push_vals[j].ctx().dev_id,
                 pull_vals[j]->ctx().dev_id);
        CHECK_EQ(push_vals[j].shape().Size(), len);
        CHECK_EQ(pull_vals[j]->shape().Size(), len);

        mvars[j] = pull_vals[j]->var();
        if (pull_vals[j]->var() != push_vals[j].var()) {
          cvars.push_back(push_vals[j].var());
        }
        sendbuff[j] = push_vals[j].data().dptr_;
        recvbuff[j] = pull_vals[j]->data().dptr_;
      }

      // the callback which does the work
      auto allreduce = [this, sendbuff, recvbuff, len] (
          RunContext rctx, Engine::CallbackOnComplete cb) {
        AllReduce(sendbuff, recvbuff, len);
        cb();
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          allreduce,
          Context::CPUPinned(0),
          cvars,
          mvars,
          FnProperty::kNormal, priority);
    }
  }

 private:
  /**
   * \brief group values on keys
   */
  template <typename V>
  void GroupKVPairs(const std::vector<int>& keys,
                    const std::vector<V>& values,
                    std::vector<int>* uniq_keys,
                    std::vector<std::vector<V> >* grouped_vals) {
    CHECK_EQ(keys.size(), values.size());
    // TODO(mli) check if already sorted as an optimization
    using Idx = std::pair<int, int>;
    std::vector<Idx> idx(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      idx[i].first = keys[i]; idx[i].second = i;
    }
    std::sort(idx.begin(), idx.end(), [](const Idx& a, const Idx& b) {
        return a.first < b.first;
      });

    int pre_key = idx[0].first - 1;
    for (auto i : idx) {
      if (i.first != pre_key) {
        uniq_keys->push_back(i.first);
        grouped_vals->push_back({values[i.second]});
        pre_key = i.first;;
      } else {
        grouped_vals->back().push_back(values[i.second]);
      }
    }
  }

  void InitNCCL() {
    size_t n = devs_.size();
    comms_.resize(n);
    streams_.resize(n);
    NCCLCHECK(ncclCommInitAll(comms_.data(), n, devs_.data()));
    for (size_t i = 0; i < n; ++i) {
      CUDACHECK(cudaSetDevice(devs_[i]));
      CUDACHECK(cudaStreamCreate(&streams_[i]));
    }
    inited_nccl_ = true;
  }

  void AllReduce(const std::vector<void*>& sendbuff,
                 const std::vector<void*>& recvbuff,
                 size_t len) {
    // do allreduce
    for (size_t i = 0; i < devs_.size(); ++i) {
      CUDACHECK(cudaSetDevice(devs_[i]));
      NCCLCHECK(ncclAllReduce(
          sendbuff[i], recvbuff[i],
          len, ncclFloat, ncclSum, comms_[i], streams_[i]));
    }
    // wait until finished
    for (size_t i = 0; i < devs_.size(); ++i) {
      CUDACHECK(cudaSetDevice(devs_[i]));
      CUDACHECK(cudaStreamSynchronize(streams_[i]));
    }
  }

  bool inited_nccl_;
  std::vector<int> devs_;
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
};

} // namespace kvstore
} // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_NCCL_H_
