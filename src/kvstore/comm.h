/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_KVSTORE_COMM_H_
#define MXNET_KVSTORE_COMM_H_
#include <dmlc/omp.h>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include <thread>
#include "mxnet/ndarray.h"
#include "../common/utils.h"
namespace mxnet {
namespace kvstore {
/**
 * \brief multiple device commmunication
 */
class Comm {
 public:
  Comm() {
#if MXNET_USE_CUDA
    int gpu_num;
    int ret = cudaGetDeviceCount(&gpu_num);
    pinned_ctx_ = (ret == 0 && gpu_num > 0) ?
                  Context::CPUPinned(0) : Context::CPU();
#else
    pinned_ctx_ = Context::CPU();
#endif
  }
  virtual ~Comm() { }
  /**
   * \brief init key with the data shape and storage shape
   */
  virtual void Init(int key, const NDArrayStorageType stype,
                    const TShape& shape, int dtype = mshadow::kFloat32) = 0;
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
    // TODO(junwu) delete the following data member, now for benchmark only
    is_serial_push_ = dmlc::GetEnv("MXNET_KVSTORE_SERIAL_PUSH", 0);
  }
  virtual ~CommCPU() { }

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int type = mshadow::kFloat32) override {
    if (stype == kDefaultStorage) {
      merge_buf_[key].merged = NDArray(shape, pinned_ctx_, false, type);
    } else {
      merge_buf_[key].merged = NDArray(stype, shape, pinned_ctx_, true, type);
    }
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    auto& buf = merge_buf_[key];
    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      if (src[0].storage_type() == buf.merged.storage_type()) {
        return src[0];
      } else {
        CopyFromTo(src[0], &buf.merged, priority);
        return buf.merged;
      }
    }

    if (buf.merged.storage_type() == kDefaultStorage) {
      std::vector<Engine::VarHandle> const_vars(src.size() - 1);
      std::vector<NDArray> reduce(src.size());
      CopyFromTo(src[0], &buf.merged, priority);
      reduce[0] = buf.merged;

      if (buf.copy_buf.empty()) {
        buf.copy_buf.resize(src.size()-1);
        for (size_t j = 0; j < src.size() - 1; ++j) {
          // allocate NDArray basd on storage type
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

    } else {
      // buf.merged is a sparse ndarray.
      std::vector<Engine::VarHandle> const_vars(src.size());
      std::vector<NDArray> reduce(src.size());

      if (buf.copy_buf.empty()) {
        buf.copy_buf.resize(src.size());
        for (size_t j = 0; j < src.size(); ++j) {
          buf.copy_buf[j] = NDArray(
            src[0].storage_type(), src[0].shape(), pinned_ctx_, true, src[0].dtype());
        }
      }
      for (size_t i = 0; i < src.size(); ++i) {
        CopyFromTo(src[i], &(buf.copy_buf[i]), priority);
        reduce[i] = buf.copy_buf[i];
        const_vars[i] = reduce[i].var();
      }
      auto result = buf.merged;
      Engine::Get()->PushSync([reduce, result, this](RunContext rctx) {
          NDArray out = result;
          is_serial_push_?
            ReduceSumCPUExSerial(reduce, &out) : ReduceSumCPUExParallel(reduce, &out);
        }, Context::CPU(), const_vars, {result.var()},
        FnProperty::kCPUPrioritized, priority, PROFILER_MESSAGE("KVStoreReduce"));
    }

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

  // serial implementation of reduce sum for row sparse NDArray.
  inline void ReduceSumCPUExSerial(const std::vector<NDArray> &in, NDArray *out) {
    using namespace rowsparse;
    using namespace mshadow;
    auto stype = out->storage_type();
    CHECK_EQ(stype, kRowSparseStorage) << "Unexpected storage type " << stype;
    size_t total_num_rows = 0;
    size_t num_in = in.size();
    // skip the ones with empty indices and values
    std::vector<bool> skip(num_in, false);
    // the values tensor of the inputs
    MSHADOW_TYPE_SWITCH(out->dtype(), DType, {
      MSHADOW_INT_TYPE_SWITCH(out->aux_type(kIdx), IType, {
        std::vector<Tensor<cpu, 2, DType>> in_vals(num_in);
        std::vector<Tensor<cpu, 1, IType>> in_indices(num_in);
        // offset to the values tensor of all inputs
        std::vector<size_t> offsets(num_in, 0);
        std::vector<size_t> num_rows(num_in, 0);
        for (size_t i = 0; i < num_in; i++) {
          if (!in[i].storage_initialized()) {
            skip[i] = true;
            continue;
          }
          auto size = in[i].aux_shape(kIdx).Size();
          num_rows[i] = size;
          total_num_rows += size;
          in_vals[i] = in[i].data().FlatTo2D<cpu, DType>();
          in_indices[i] = in[i].aux_data(kIdx).FlatTo1D<cpu, IType>();
        }
        std::vector<IType> indices;
        indices.reserve(total_num_rows);
        // gather indices from all inputs
        for (size_t i = 0; i < num_in; i++) {
          for (size_t j = 0; j < num_rows[i]; j++) {
            indices.emplace_back(in_indices[i][j]);
          }
        }
        CHECK_EQ(indices.size(), total_num_rows);
        // dedup indices
        std::sort(indices.begin(), indices.end());
        indices.resize(std::unique(indices.begin(), indices.end()) - indices.begin());
        // the one left are unique non-zero rows
        size_t nnr = indices.size();
        // allocate memory for output
        out->CheckAndAlloc({Shape1(nnr)});
        auto idx_data = out->aux_data(kIdx).FlatTo1D<cpu, IType>();
        auto val_data = out->data().FlatTo2D<cpu, DType>();

        for (size_t i = 0; i < nnr; i++) {
          // copy indices back
          idx_data[i] = indices[i];
          bool zeros = true;
          for (size_t j = 0; j < num_in; j++) {
            if (skip[j]) continue;
            size_t offset = offsets[j];
            if (offset < num_rows[j]) {
              if (indices[i] == in_indices[j][offset]) {
                if (zeros) {
                  Copy(val_data[i], in_vals[j][offset], nullptr);
                  zeros = false;
                } else {
                  val_data[i] += in_vals[j][offset];
                }
                offsets[j] += 1;
              }
            }
          }
        }
      });
    });
  }

  template<typename DType, typename IType>
  void ReduceSumCPUExImpl(const std::vector<NDArray>& nds,
                          const std::vector<IType>& uniq_row_idx,
                          NDArray* out) {
#pragma omp parallel num_threads(nthread_reduction_)
    {
      const size_t nnr = uniq_row_idx.size();
      const int num_threads = omp_get_num_threads();
      size_t row_block_len = (nnr + num_threads  - 1) / num_threads;
      const size_t row_block_start = omp_get_thread_num() * row_block_len;
      if (row_block_start < nnr) {
        const size_t row_block_end = std::min(row_block_start+row_block_len, nnr);

        auto out_values = out->data().FlatTo2D<cpu, DType>();
        auto out_indices = out->aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>();
        for (size_t i = row_block_start; i < row_block_end; ++i) {
          out_indices[i] = uniq_row_idx[i];
        }
        for (const auto& nd : nds) {
          if (nd.storage_initialized()) {
            const auto nd_indices = nd.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>();
            const auto nd_values = nd.data().FlatTo2D<cpu, DType>();
            const auto nd_num_rows = nd.aux_shape(rowsparse::kIdx).Size();
            const IType* nd_indices_start = &nd_indices[0];
            const IType* nd_indices_end = nd_indices_start + nd_num_rows;
            const IType* row_idx_ptr = std::lower_bound(nd_indices_start, nd_indices_end,
                                                        out_indices[row_block_start]);
            // skip this nd if all of its row indices are smaller than out_indices[row_block_start]
            // or current row block is not covered by [*row_idx_ptr, nd_indices_end).
            if (nd_indices_end == row_idx_ptr || *row_idx_ptr > out_indices[row_block_end-1]) {
              continue;
            }
            for (size_t irow = row_block_start;
                 irow < row_block_end && row_idx_ptr != nd_indices_end;) {
              if (out_indices[irow] == *row_idx_ptr) {
                auto out_value_cur_row = out_values[irow];
                const auto offset = row_idx_ptr - nd_indices_start;
                auto nd_value_cur_row = nd_values[offset];
                for (size_t j = 0; j < nd_value_cur_row.shape_[0]; ++j) {
                  out_value_cur_row[j] += nd_value_cur_row[j];
                }
                ++irow;
                ++row_idx_ptr;
              } else if (out_indices[irow] < *row_idx_ptr) {
                ++irow;
              } else {
                ++row_idx_ptr;
              }
            }
          }
        }
      }
    }
  }

  /*!
   * \brief Given a vector of ndarrays, generate a index vector containing
   * all the unique row indices of the ndarrays.
   */
  template<typename IType>
  void GetUniqueRspRowIdx(const std::vector<NDArray>& nds,
                          std::vector<IType>* uniq_row_idx) {
    using namespace rowsparse;
    size_t total_num_rows = 0;
    for (const auto& nd : nds) {
      CHECK_EQ(nd.storage_type(), kRowSparseStorage);
      if (nd.storage_initialized()) {
        total_num_rows += nd.aux_shape(kIdx).Size();
      }
    }

    uniq_row_idx->resize(total_num_rows);
    int nthreads = omp_get_max_threads();
    int offset = 0;
    for (const auto& nd : nds) {
      if (nd.storage_initialized()) {
        const IType* nd_row_idx = nd.aux_data(kIdx).dptr<IType>();
        const int num_rows = nd.aux_shape(kIdx).Size();
#pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < num_rows; ++i) {
          (*uniq_row_idx)[offset+i] = nd_row_idx[i];
        }
        offset += num_rows;
      }
    }

    common::ParallelSort(uniq_row_idx->begin(), uniq_row_idx->end(), nthreads);
    auto it = std::unique(uniq_row_idx->begin(), uniq_row_idx->end());
    uniq_row_idx->resize(it - uniq_row_idx->begin());
  }

  void ReduceSumCPUExParallel(const std::vector<NDArray>& nds, NDArray* out) {
    if (nds.empty()) return;
    using namespace rowsparse;
    CHECK_EQ(out->storage_type(), kRowSparseStorage)
      << "Expected row sparse storage type ("
      << out->storage_type() << " given)";

    MSHADOW_TYPE_SWITCH(out->dtype(), DType, {
      MSHADOW_INT_TYPE_SWITCH(out->aux_type(kIdx), IType, {
        std::vector<IType> uniq_row_idx;
        GetUniqueRspRowIdx(nds, &uniq_row_idx);
        out->CheckAndAlloc({mshadow::Shape1(uniq_row_idx.size())});
        out->data().FlatTo2D<cpu, DType>() = static_cast<DType>(0);
        ReduceSumCPUExImpl<DType, IType>(nds, uniq_row_idx, out);
      });
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
  bool is_serial_push_;
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

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int dtype = mshadow::kFloat32) override {
    if (stype == kDefaultStorage) {
      sorted_key_attrs_.push_back(std::make_tuple(key, shape, dtype));
    } else {
      LOG(FATAL) << "storage type " << stype << " not implemented for device yet";
    }
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
