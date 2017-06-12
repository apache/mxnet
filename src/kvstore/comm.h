/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_KVSTORE_COMM_H_
#define MXNET_KVSTORE_COMM_H_
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include "mxnet/ndarray.h"
namespace mxnet {
namespace kvstore {
/**
 * \brief multiple device commmunication
 */
class Comm {
 public:
  Comm() {
    pinned_ctx_ = Context::CPUPinned(0);
  }
  virtual ~Comm() { }
  /**
   * \brief init key with the data shape
   */
  virtual void Init(int key, const TShape& shape, int dtype = mshadow::kFloat32) = 0;
  /**
   * \brief returns src[0] + .. + src[src.size()-1]
   */
  virtual const NDArray& Reduce(
      int key, const std::vector<NDArray>& src, int priority) = 0;
  /**
   * \brief copy from src to dst[i] for every i
   */
  virtual void Broadcast(
      int key, const NDArray& src,
      const std::vector<NDArray*> dst, int priority) = 0;

  /**
   * \brief return a pinned contex
   */
  Context pinned_ctx() const {
    return pinned_ctx_;
  }

 protected:
  Context pinned_ctx_;
};

/**
 * \brief an implemention of Comm that first copy data to CPU memeory, and then
 * reduce there
 */
class CommCPU : public Comm {
 public:
  CommCPU() {
    nthread_reduction_ = dmlc::GetEnv("MXNET_KVSTORE_REDUCTION_NTHREADS", 4);
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
  }
  virtual ~CommCPU() { }

  void Init(int key, const TShape& shape, int type = mshadow::kFloat32) override {
    merge_buf_[key].merged = NDArray(shape, pinned_ctx_, false, type);
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      return src[0];
    }
    std::vector<Engine::VarHandle> const_vars(src.size() - 1);
    std::vector<NDArray> reduce(src.size());
    auto& buf = merge_buf_[key];
    CopyFromTo(src[0], &buf.merged, priority);
    reduce[0] = buf.merged;

    if (buf.copy_buf.empty()) {
      buf.copy_buf.resize(src.size()-1);
      for (size_t j = 0; j < src.size() - 1; ++j) {
        buf.copy_buf[j] = NDArray(
          src[0].shape(), pinned_ctx_, false, src[0].dtype());
      }
    }
    for (size_t i = 1; i < src.size(); ++i) {
      CopyFromTo(src[i], &(buf.copy_buf[i-1]), priority);
      reduce[i] = buf.copy_buf[i-1];
      const_vars[i-1] = reduce[i].var();
    }

    Engine::Get()->PushSync([reduce, this](RunContext rctx) {
        ReduceSumCPU(reduce);
      }, Context::CPU(), const_vars, {reduce[0].var()},
      FnProperty::kCPUPrioritized, priority, PROFILER_MESSAGE("KVStoreReduce"));

    return buf.merged;
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    int mask = src.ctx().dev_mask();
    if (mask == Context::kCPU) {
      for (auto d : dst) CopyFromTo(src, d, priority);
    } else {
      // first copy data to cpu, then broadcast
      auto& buf = merge_buf_[key];
      CopyFromTo(src, &buf.merged, priority);
      for (auto d : dst) CopyFromTo(buf.merged, d, priority);
    }
  }

 private:
  // reduce sum into val[0]
  inline void ReduceSumCPU(const std::vector<NDArray> &in_data) {
    MSHADOW_TYPE_SWITCH(in_data[0].dtype(), DType, {
      std::vector<DType*> dptr(in_data.size());
      for (size_t i = 0; i < in_data.size(); ++i) {
        TBlob data = in_data[i].data();
        CHECK(data.CheckContiguous());
        dptr[i] = data.FlatTo2D<cpu, DType>().dptr_;
      }
      size_t total = in_data[0].shape().Size();
      ReduceSumCPUImpl(dptr, total);
    });
  }

  template<typename DType>
  inline static void ReduceSumCPU(
      const std::vector<DType*> &dptr, size_t offset, index_t size) {
    using namespace mshadow;  // NOLINT(*)
    Tensor<cpu, 1, DType> in_0(dptr[0] + offset, Shape1(size));
    for (size_t i = 1; i < dptr.size(); i+=4) {
      switch (dptr.size() - i) {
        case 1: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          in_0 += in_1;
          break;
        }
        case 2: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          in_0 += in_1 + in_2;
          break;
        }
        case 3: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_3(dptr[i+2] + offset, Shape1(size));
          in_0 += in_1 + in_2 + in_3;
          break;
        }
        default: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_3(dptr[i+2] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_4(dptr[i+3] + offset, Shape1(size));
          in_0 += in_1 + in_2 + in_3 + in_4;
          break;
        }
      }
    }
  }

  template<typename DType>
  inline void ReduceSumCPUImpl(std::vector<DType*> dptr, size_t total) {
    const size_t step = std::min(bigarray_bound_, static_cast<size_t>(4 << 10));
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

  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the merged value
    NDArray merged;
    /// \brief the cpu buffer for gpu data
    std::vector<NDArray> copy_buf;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;
  size_t bigarray_bound_;
  int nthread_reduction_;
};

/**
 * \brief an implementation of Comm that performs reduction on device
 * directly.
 *
 * It is faster if the total device-to-device bandwidths is larger than
 * device-to-cpu, which is often true for 4 or 8 GPUs. But it uses more device
 * memory.
 */
class CommDevice : public Comm {
 public:
  CommDevice() {
    inited_ = false;
  }

  virtual ~CommDevice() { }

  void Init(int key, const TShape& shape, int dtype = mshadow::kFloat32) override {
    sorted_key_attrs_.push_back(std::make_tuple(key, shape, dtype));
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      return src[0];
    }

    if (!inited_) {
      std::vector<Context> devs;
      for (const auto& a : src) {
        devs.push_back(a.ctx());
      }
      InitMergeBuffer(devs);
      if (dmlc::GetEnv("MXNET_ENABLE_GPU_P2P", 1)) {
        EnableP2P(devs);
      }
    }

    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(src.size());
    CopyFromTo(src[0], &(buf.merged), priority);
    reduce[0] = buf.merged;

    if (buf.copy_buf.empty()) {
      // TODO(mli) this results in large device memory usage for huge ndarray,
      // such as the largest fullc in VGG. consider to do segment reduce with
      // NDArray.Slice or gpu direct memory access. for the latter, we need to
      // remove some ctx check, and also it reduces 20% perf
      buf.copy_buf.resize(src.size()-1);
      for (size_t i = 0; i < src.size()-1; ++i) {
        buf.copy_buf[i] = NDArray(
          buf.merged.shape(), buf.merged.ctx(), false, buf.merged.dtype());
      }
    }
    for (size_t i = 0; i < src.size()-1; ++i) {
      CopyFromTo(src[i+1], &(buf.copy_buf[i]), priority);
      reduce[i+1] = buf.copy_buf[i];
    }

    ElementwiseSum(reduce, &buf.merged);

    return buf.merged;
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    if (!inited_) {
      // copy to a random device first
      int dev_id = key % dst.size();
      CopyFromTo(src, dst[dev_id], priority);
      for (size_t i = 0; i < dst.size(); ++i) {
        if (i != static_cast<size_t>(dev_id)) {
          CopyFromTo(*dst[dev_id], dst[i], priority);
        }
      }
    } else {
      auto& buf = merge_buf_[key];
      CopyFromTo(src, &buf.merged, priority);
      for (auto d : dst) {
        CopyFromTo(buf.merged, d, priority);
      }
    }
  }

 private:
  void EnableP2P(const std::vector<Context>& devs) {
#if MXNET_USE_CUDA
    std::vector<int> gpus;
    for (const auto& d : devs) {
      if (d.dev_mask() == gpu::kDevMask) {
        gpus.push_back(d.dev_id);
      }
    }
    int n = static_cast<int>(gpus.size());
    int enabled = 0;
    std::vector<int> p2p(n*n);
    for (int i = 0; i < n; ++i) {
      cudaSetDevice(gpus[i]);
      for (int j = 0; j < n; j++) {
        int access;
        cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
        if (access) {
          cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
          if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            ++enabled;
            p2p[i*n+j] = 1;
          }
        }
      }
    }
    if (enabled != n*(n-1)) {
      // print warning info if not fully enabled
      LOG(WARNING) << "only " << enabled <<  " out of "
                   << n*(n-1) << " GPU pairs are enabled direct access. "
                   << "It may affect the performance. "
                   << "You can set MXNET_ENABLE_GPU_P2P=0 to turn it off";
      std::string access(n, '.');
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          access[j] = p2p[i*n+j] ? 'v' : '.';
        }
        LOG(WARNING) << access;
      }
    }
#endif
  }

  using KeyAttrs = std::tuple<int, TShape, int>;
  // try to allocate buff on device evenly
  void InitMergeBuffer(const std::vector<Context>& devs) {
    std::sort(sorted_key_attrs_.begin(), sorted_key_attrs_.end(), [](
              const KeyAttrs& a, const KeyAttrs& b) {
      return std::get<1>(a).Size() > std::get<1>(b).Size();
    });

    std::unordered_map<int, std::pair<Context, size_t>> ctx_info;
    for (auto d : devs) {
      ctx_info[d.dev_id] = std::make_pair(d, 0);
    }
    for (size_t i = 0; i < sorted_key_attrs_.size(); ++i) {
      int key  = std::get<0>(sorted_key_attrs_[i]);
      TShape s = std::get<1>(sorted_key_attrs_[i]);
      int type = std::get<2>(sorted_key_attrs_[i]);
      auto& buf = merge_buf_[key];
      Context ctx;
      size_t min_size = std::numeric_limits<size_t>::max();
      for (auto it = ctx_info.begin(); it != ctx_info.end(); ++it) {
        size_t size = it->second.second;
        if (size <= min_size) {
          ctx = it->second.first;
          min_size = size;
        }
      }
      buf.merged = NDArray(s, ctx, false, type);
      ctx_info[ctx.dev_id].second += s.Size();
    }
    inited_ = true;
  }

  std::vector<KeyAttrs> sorted_key_attrs_;
  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the merged value
    NDArray merged;
    /// \brief the gpu buffer
    std::vector<NDArray> copy_buf;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;
  bool inited_;
};

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_COMM_H_
