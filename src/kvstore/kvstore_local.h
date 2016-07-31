/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_KVSTORE_LOCAL_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <utility>
#include <algorithm>

namespace mxnet {
namespace kvstore {
/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  KVStoreLocal() {
    pinned_ctx_ = (MXNET_USE_CUDA != 0) ?
        Context::CPUPinned(0) : Context::CPU();
    // the server perameters
    nthread_reduction_ = dmlc::GetEnv("MXNET_KVSTORE_REDUCTION_NTHREADS", 4);
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
    }
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& merged = MergePushValue(key, grouped_vals[i], priority);
      if (updater_ != nullptr) {
        auto it = local_.find(key);
        CHECK(it != local_.end()) << "key " << key << " has not been inited";
        updater_(key, merged,  &(it->second));
      }
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
      auto it = merge_buf_.find(key);
      if (updater_ != nullptr || it == merge_buf_.end()) {
        auto it = local_.find(key);
        CHECK(it != local_.end()) << "key " << key << " has not been inited";
        ScatterPullValue(
            key, it->second, grouped_vals[i], priority);
      } else {
        ScatterPullValue(
            key, it->second.merged, grouped_vals[i], priority);
      }
    }
  }

 protected:
  /// \brief temperal space for pushing and pull
  struct BufferEntry {
    // Context of merged
    Context ctx;
    // the merged value
    NDArray merged;
    // the merged value on device
    NDArray merged_device;
    /// \brief the cpu buffer for gpu data
    std::vector<NDArray> copy_buf;
    // allocate copy buffer, if it has not been allocated
    inline NDArray *AllocCopyBuf(size_t index, Context ctx, const TShape& shape) {
      if (index >= copy_buf.size()) copy_buf.resize(index + 1);
      if (copy_buf[index].is_none()) {
        copy_buf[index] = NDArray(shape, ctx);
      }
      return &copy_buf[index];
    }
  };
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
  /*!
   * \brief returns the aggregated push value
   */
  virtual const NDArray& MergePushValue(
      int key, const std::vector<NDArray>& val, int priority) {
    auto& buf = merge_buf_[key];
    // copy buffer
    std::vector<Engine::VarHandle> const_vars(val.size() - 1);
    std::vector<NDArray> reduce(val.size());

    if (buf.merged.is_none()) {
      buf.ctx = Context::CPUPinned(val[0].ctx().dev_id);
      if (MXNET_USE_CUDA == 0) buf.ctx = Context::CPU();
      buf.merged = NDArray(val[0].shape(), buf.ctx);
    }

    CopyFromTo(val[0], &(buf.merged), priority);
    reduce[0] = buf.merged;

    for (size_t i = 1; i < val.size(); ++i) {
      const NDArray& v = val[i];
      Context ctx = v.ctx();
      if (ctx.dev_mask() == cpu::kDevMask) {
        reduce[i] = val[i];
      } else {
        NDArray *copy_buf = buf.AllocCopyBuf(
            i, Context::CPUPinned(ctx.dev_id), val[0].shape());
        CopyFromTo(val[i], copy_buf, priority);
        reduce[i] = *copy_buf;
      }
      const_vars[i - 1] = reduce[i].var();
    }

    Engine::Get()->PushSync([reduce, this](RunContext rctx) {
        ReduceSumCPU(reduce);
      }, Context::CPU(), const_vars, {reduce[0].var()},
      FnProperty::kCPUPrioritized, priority);
    return buf.merged;
  }

  virtual void ScatterPullValue(
      int key,
      const NDArray& src,
      const std::vector<NDArray*>& vals,
      int priority) {
    for (auto* vptr : vals) {
      CopyFromTo(src, vptr, priority);
    }
  }

  /// \brief buffer for merging push value
  std::unordered_map<int, BufferEntry> merge_buf_;
  // pinned context
  Context pinned_ctx_;
  // the lower bound of a big array
  size_t bigarray_bound_;

 private:
  inline static void ReduceSumCPU(const std::vector<real_t*> &dptr,
                                  size_t offset, index_t size) {
    using namespace mshadow;  // NOLINT(*)
    Tensor<cpu, 1> in_0(dptr[0] + offset, Shape1(size));
    switch (dptr.size()) {
      case 2: {
        Tensor<cpu, 1> in_1(dptr[1] + offset, Shape1(size));
        in_0 += in_1;
        break;
      }
      case 3: {
        Tensor<cpu, 1> in_1(dptr[1] + offset, Shape1(size));
        Tensor<cpu, 1> in_2(dptr[2] + offset, Shape1(size));
        in_0 += in_1 + in_2;
        break;
      }
      case 4: {
        Tensor<cpu, 1> in_1(dptr[1] + offset, Shape1(size));
        Tensor<cpu, 1> in_2(dptr[2] + offset, Shape1(size));
        Tensor<cpu, 1> in_3(dptr[3] + offset, Shape1(size));
        in_0 += in_1 + in_2 + in_3;
        break;
      }
      default: {
        for (size_t i = 1; i < dptr.size(); ++i) {
          Tensor<cpu, 1> in_k(dptr[i] + offset, Shape1(size));
          in_0 += in_k;
        }
      }
    }
  }
  // reduce sum into val[0]
  // this is performance critical
  inline void ReduceSumCPU(const std::vector<NDArray> &in_data) {
    const size_t step = std::min(bigarray_bound_, static_cast<size_t>(4 << 10));
    // ge ptr out
    std::vector<real_t*> dptr(in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      TBlob data = in_data[i].data();
      CHECK(data.CheckContiguous());
      dptr[i] = data.FlatTo2D<cpu, real_t>().dptr_;
    }
    size_t total = in_data[0].shape().Size();
    long ntask = (total + step - 1) / step; // NOLINT(*)
    if (total < bigarray_bound_ || nthread_reduction_ <= 1) {
      ReduceSumCPU(dptr, 0, total);
    } else {
      #pragma omp parallel for schedule(static) num_threads(nthread_reduction_)
      for (long j = 0; j < ntask; ++j) { // NOLINT(*)
        size_t k = static_cast<size_t>(j);
        size_t begin = std::min(k * step, total);
        size_t end = std::min((k + 1) * step, total);
        if (j == ntask - 1) CHECK_EQ(end, total);
        ReduceSumCPU(dptr, begin, static_cast<index_t>(end - begin));
      }
    }
  }

  /// \brief buffer for storing local values
  std::unordered_map<int, NDArray> local_;

  // number of threads to do reduction
  int nthread_reduction_;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
