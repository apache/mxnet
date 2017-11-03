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
 */
#ifndef MXNET_KVSTORE_COMM_H_
#define MXNET_KVSTORE_COMM_H_
#include <dmlc/omp.h>
#if MXNET_USE_NCCL
#include <nccl.h>
#endif  // MXNET_USE_NCCL
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include <thread>
#include "mxnet/ndarray.h"
#include "../ndarray/ndarray_function.h"
#include "../operator/tensor/sparse_retain-inl.h"

#if MXNET_USE_NCCL
#include "../common/cuda_utils.h"

#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#endif

#if NCCL_MAJOR == 1
#define ncclGroupStart()
#define ncclGroupEnd()
#define ncclNumTypes nccl_NUM_TYPES
#endif  // NCCL_MAJOR == 1
#endif  // MXNET_USE_NCCL

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

  virtual void CommSync(const std::vector<const NDArray*>& dst, int priority) { }
  virtual void CommSync(const std::vector<NDArray>& dst, int priority) { }

  /**
   * \brief broadcast src to dst[i] with target row_ids for every i
   * \param dst a list of destination row_sparse NDArray and its target row_ids to broadcast,
            where the row_ids are expected to be unique and sorted
   * \param use_copy if set to true, directly copy src to dst[i] without looking up the
            provided row_ids
   */
  virtual void BroadcastRowSparse(int key, const NDArray& src,
                                  const std::vector<std::pair<NDArray*, NDArray>>& dst,
                                  const bool use_copy,
                                  const int priority) = 0;

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
      if (src[0].storage_type() == kDefaultStorage) {
        return src[0];
      } else {  // if sparse and only one GPU, always update weight on CPU
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
          // allocate NDArray based on storage type
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
          Resource rsc = ResourceManager::Get()->Request(rctx.ctx,
              ResourceRequest(ResourceRequest::kTempSpace));
          is_serial_push_?
            ReduceSumCPUExSerial(reduce, &out)
            : mxnet::ndarray::ElementwiseSum(rctx.get_stream<cpu>(), rsc, reduce, &out);
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

  void BroadcastRowSparse(int key, const NDArray& src,
                          const std::vector<std::pair<NDArray*, NDArray>>& dst,
                          const bool use_copy,
                          const int priority) override {
    using namespace mshadow;
    CHECK_EQ(src.storage_type(), kRowSparseStorage)
      << "BroadcastRowSparse expects row-sparse src NDArray";
    CHECK_EQ(src.ctx().dev_mask(), Context::kCPU)
      << "BroadcastRowSparse with src on gpu context not supported";
    for (size_t i = 0; i < dst.size(); ++i) {
      NDArray* out = dst[i].first;
      NDArray row_id = dst[i].second;
      if (use_copy) {
        CopyFromTo(src, out, priority);
      } else {
        CHECK_EQ(out->storage_type(), kRowSparseStorage)
                 << "BroadcastRowSparse expects row_sparse dst NDArray";
        CHECK_EQ(row_id.ctx().dev_mask(), Context::kCPU)
                 << "BroadcastRowSparse with row_indices on gpu context not supported";
        // retain according to unique indices
        const bool use_sparse_retain = (src.shape()[0] != src.storage_shape()[0])
          || (row_id.dtype() != out->aux_type(rowsparse::kIdx))
          || (out->ctx().dev_mask() != Context::kGPU);
        if (use_sparse_retain) {  // use sparse_retain op
          const bool is_to_gpu = out->ctx().dev_mask() == Context::kGPU;
          NDArray out_cpu = is_to_gpu? NDArray(kRowSparseStorage, src.shape(),
              src.ctx(), true, src.dtype(), src.aux_types()) : *out;
          Engine::Get()->PushSync([=](RunContext rctx) {
              const TBlob& indices = row_id.data();
              NDArray temp = out_cpu;  // get rid of const qualifier
              op::SparseRetainOpForwardRspImpl<cpu>(rctx.get_stream<cpu>(),
                                                    src, indices, kWriteTo,
                                                    &temp);
            }, Context::CPU(), {src.var(), row_id.var()}, {out_cpu.var()},
            FnProperty::kNormal, priority, PROFILER_MESSAGE("KVStoreSparseRetain"));
          if (is_to_gpu) {
            CopyFromTo(out_cpu, out, priority);
          }
        } else {  // direct copy rows
          Engine::Get()->PushSync([=](RunContext rctx) {
              CopyRetainedRowsToGPU(rctx.get_stream<cpu>(), rctx.get_stream<gpu>(),
                                    src, row_id, out);
            }, out->ctx(), {src.var(), row_id.var()}, {out->var()},
            FnProperty::kCopyToGPU, priority, PROFILER_MESSAGE("KVStoreCopyRetainedRowsToGPU"));
        }
      }
    }
  }

 private:
  /*!
   * \brief When src is a rsp with full rows,
   * simply copy retained rows directly from cpu to gpu
   * without invoking sparse_retain op.
   */
  void CopyRetainedRowsToGPU(mshadow::Stream<cpu>* cpu_stream,
                             mshadow::Stream<gpu>* gpu_stream,
                             const NDArray& src,
                             const NDArray& indices,
                             NDArray* dst) {
#if MXNET_USE_CUDA == 1
    CHECK_EQ(src.storage_type(), kRowSparseStorage)
      << "CopyRetainedRowsToGPU expects row-sparse src NDArray";
    CHECK_EQ(src.ctx().dev_mask(), Context::kCPU)
      << "CopyRetainedRowsToGPU with src on gpu context not supported";
    CHECK_EQ(src.storage_shape()[0], src.shape()[0])
      << "CopyRetainedRowsToGPU only supports src rsp with full rows";
    CHECK_EQ(indices.storage_type(), kDefaultStorage);
    CHECK_EQ(indices.ctx().dev_mask(), Context::kCPU);
    CHECK_EQ(dst->storage_type(), kRowSparseStorage);
    CHECK_EQ(dst->ctx().dev_mask(), Context::kGPU);
    CHECK_EQ(indices.dtype(), dst->aux_type(rowsparse::kIdx))
      << "CopyRetainedRowsToGPU only supports same data type for idx array and dst aux_data(0)";
    if (!src.storage_initialized() || indices.data().Size() == 0U) {
      op::FillZerosRspImpl(gpu_stream, *dst);
      return;
    }
    using namespace mshadow;

    const TBlob& src_data = src.data();
    const TBlob& idx_data = indices.data();
    const size_t row_length = src.shape().ProdShape(1, src.shape().ndim());
    const size_t num_rows_retained = idx_data.Size();
    dst->CheckAndAlloc({Shape1(num_rows_retained)});
    TBlob dst_data = dst->data();
    TBlob dst_idx_data = dst->aux_data(rowsparse::kIdx);
    MSHADOW_TYPE_SWITCH(src.dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(indices.dtype(), IType, {
        // copy idx array
        Tensor<gpu, 1, IType> dst_idx_tensor = dst_idx_data.FlatTo1D<gpu, IType>(gpu_stream);
        const Tensor<cpu, 1, IType> idx_tensor = idx_data.FlatTo1D<cpu, IType>(cpu_stream);
        Copy(dst_idx_tensor, idx_tensor, gpu_stream);
        // copy src data
        const Tensor<cpu, 2, DType> src_data_tensor = src_data.get_with_shape<cpu, 2, DType>(
            Shape2(src_data.shape_[0], row_length), cpu_stream);
        Tensor<gpu, 2, DType> dst_data_tensor = dst_data.get_with_shape<gpu, 2, DType>(
            Shape2(dst_data.shape_[0], row_length), gpu_stream);
        for (size_t i = 0; i < num_rows_retained; ++i) {
          Copy(dst_data_tensor[i], src_data_tensor[idx_tensor[i]], gpu_stream);
        }
      })
    })
#else
    LOG(FATAL) << "GPU not enabled";
#endif
  }

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
      MSHADOW_IDX_TYPE_SWITCH(out->aux_type(kIdx), IType, {
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

  void BroadcastRowSparse(int key, const NDArray& src,
                          const std::vector<std::pair<NDArray*, NDArray>>& dst,
                          const bool use_copy,
                          const int priority) override {
    LOG(FATAL) << "Not implemented yet";
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

#if MXNET_USE_NCCL
class CommNCCL : public Comm {
 public:
  CommNCCL() {
    inited_ = false;
    pinned_ctx_ = Context::CPUPinned(0);
  }

  virtual ~CommNCCL() {
    for (auto e : nccl_data_) {
      cudaStreamDestroy(e.second.stream);
      ncclCommDestroy(e.second.comm);
    }
  }

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int dtype = mshadow::kFloat32) override {
    if (stype == kDefaultStorage) {
      sorted_key_attrs_.push_back(std::make_tuple(key, shape, dtype));
    } else {
      LOG(FATAL) << "NCCL KVStore does not support sparse storage type";
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
      InitNCCL(devs);
      InitMergeBuffer(devs);
    }

    std::vector<int> dev_ids;
    for (auto e : src) {
      dev_ids.push_back(e.ctx().dev_id);
    }
    std::sort(dev_ids.begin(), dev_ids.end());
    CHECK(device_ids_ == dev_ids) << "NCCL KVStore supports only single set of devices";

    auto& buf = merge_buf_[key];
    int root = buf.merged.ctx().dev_id;
    size_t root_id = -1;
    for (size_t i = 0; i < src.size(); ++i) {
      if (src[i].ctx().dev_id == root) {
        root_id = i;
        break;
      }
    }

    auto& reduce = buf.merged;

    std::vector<Engine::VarHandle> const_vars;
    for (size_t i = 0; i < src.size(); ++i) {
      const_vars.push_back(src[i].var());
    }
    Engine::Get()->PushSync([src, reduce, root_id, this](RunContext rctx) {
          {
            std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
            int root = nccl_data_[src[root_id].ctx().dev_id].rank;
            ncclGroupStart();
            for (size_t i = 0; i < src.size(); ++i) {
              NCCLEntry cur = nccl_data_[src[i].ctx().dev_id];
              if (i == root_id) {
              MSHADOW_TYPE_SWITCH(src[i].dtype(), DType,
              ncclReduce(src[i].data().dptr<DType>(),
                                reduce.data().dptr<DType>(),
                                src[i].shape().Size(),
                                GetNCCLType(src[i].dtype()),
                                ncclSum,
                                root,
                                cur.comm,
                                cur.stream););
              } else {
              MSHADOW_TYPE_SWITCH(src[i].dtype(), DType,
              ncclReduce(src[i].data().dptr<DType>(),
                                NULL,
                                src[i].shape().Size(),
                                GetNCCLType(src[i].dtype()),
                                ncclSum,
                                root,
                                cur.comm,
                                cur.stream););
              }
            }
            ncclGroupEnd();
          }
        },
        Context::CPU(),
        const_vars,
        {reduce.var()},
        FnProperty::kCPUPrioritized,
        priority,
        PROFILER_MESSAGE("KVStoreReduce"));

    return buf.merged;
  }

  void CommSync(const std::vector<const NDArray*>& dst,
                int priority) override {
    std::vector<Engine::VarHandle> const_vars;
    std::vector<Engine::VarHandle> mutate_vars;
    for (size_t i = 0; i < dst.size(); ++i) {
        mutate_vars.push_back(dst[i]->var());
    }
    Engine::Get()->PushSync([this](RunContext rctx) {
          for (auto cur : nccl_data_) {
            CUDA_CALL(cudaSetDevice(cur.second.dev_id));
            CUDA_CALL(cudaStreamSynchronize(cur.second.stream));
          }
        },
        Context::CPU(),
        const_vars,
        mutate_vars,
        FnProperty::kCPUPrioritized,
        priority,
        PROFILER_MESSAGE("KVStoreStreamSync"));
  }

  void CommSync(const std::vector<NDArray>& dst,
                int priority) override {
    std::vector<Engine::VarHandle> const_vars;
    std::vector<Engine::VarHandle> mutate_vars;
    for (size_t i = 0; i < dst.size(); ++i) {
        mutate_vars.push_back(dst[i].var());
    }
    Engine::Get()->PushSync([this](RunContext rctx) {
          for (auto cur : nccl_data_) {
            CUDA_CALL(cudaSetDevice(cur.second.dev_id));
            CUDA_CALL(cudaStreamSynchronize(cur.second.stream));
          }
        },
        Context::CPU(),
        const_vars,
        mutate_vars,
        FnProperty::kCPUPrioritized,
        priority,
        PROFILER_MESSAGE("KVStoreStreamSync"));
  }

  void BroadcastRowSparse(int key, const NDArray& src,
                          const std::vector<std::pair<NDArray*, NDArray>>& dst,
                          const bool use_copy,
                          const int priority) override {
    LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    if (!inited_) {
      // copy to a random device first
      int dev_id = key % dst.size();
      CopyFromTo(src, dst[dev_id], priority);
      for (size_t i = 0; i < dst.size(); ++i) {
        if (i != static_cast<size_t>(dev_id)) {
          CopyFromTo(*(dst[dev_id]), dst[i], priority);
        }
      }
    } else {
      auto& buf = merge_buf_[key];
      int root = src.ctx().dev_id;
      assert(root == buf.ctx().dev_id);
      size_t root_id = -1;
      for (size_t i = 0; i < dst.size(); ++i) {
        if (dst[i]->ctx().dev_id == root) {
          root_id = i;
          break;
        }
      }
      std::vector<int> dev_ids;
      for (size_t i = 0; i < dst.size(); ++i) {
        auto& bcast = (i == root_id) ? src : *(dst[i]);
        dev_ids.push_back(bcast.ctx().dev_id);
      }
      std::sort(dev_ids.begin(), dev_ids.end());
      CHECK(device_ids_ == dev_ids) << "NCCL KVStore supports only single set of devices";
      CopyFromTo(src, dst[root_id], priority);
      if (dst.size() == 1) return;
      std::vector<Engine::VarHandle> mutable_vars;
      for (size_t i = 0; i < dst.size(); ++i) {
        if ( i != root_id)
          mutable_vars.push_back(dst[i]->var());
      }
      std::vector<NDArray> broadcast(dst.size());
      for(size_t i = 0; i < dst.size(); ++dst) {
        broadcast[i] = *(dst[i]);
      }
      Engine::Get()->PushSync([src, broadcast, root_id, this](RunContext rctx) {
          {
            std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
            int root = nccl_data_[src.ctx().dev_id].rank;
            ncclGroupStart();
            for (size_t i = 0; i < broadcast.size(); ++i) {
              auto& bcast = (i == root_id) ? src : broadcast[i];
              NCCLEntry cur = nccl_data_[bcast.ctx().dev_id];
              MSHADOW_TYPE_SWITCH(bcast.dtype(), DType,
                  ncclBcast(bcast.data().dptr<DType>(),
                    bcast.shape().Size(),
                    GetNCCLType(bcast.dtype()),
                    root,
                    cur.comm,
                    cur.stream););
            }
            ncclGroupEnd();
          }
      },
      Context::CPU(),
      {src.var()},
      mutable_vars,
      FnProperty::kCPUPrioritized,
      priority,
      PROFILER_MESSAGE("KVStoreBCast"));
    }
  }

 private:
  ncclDataType_t GetNCCLType(int dtype) {
    switch (dtype) {
      case mshadow::kFloat32:
        return ncclFloat;
      case mshadow::kFloat16:
        return ncclHalf;
      case mshadow::kFloat64:
        return ncclDouble;
      case mshadow::kUint8:
        return ncclChar;
      case mshadow::kInt32:
        return ncclInt;
      case mshadow::kInt64:
        return ncclInt64;
      default:
        LOG(FATAL) << "Unknown type passed to NCCL KVStore";
    }
    return ncclNumTypes;
  }

  void InitNCCL(const std::vector<Context>& devs) {
    for (size_t i = 0; i < devs.size(); ++i) {
      device_ids_.push_back(devs[i].dev_id);
    }
    std::sort(device_ids_.begin(), device_ids_.end());
    std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
    std::vector<ncclComm_t> comms(devs.size());
    ncclCommInitAll(&(comms[0]), devs.size(), &(device_ids_[0]));
    for (size_t i = 0; i < devs.size(); ++i) {
      NCCLEntry e;
      e.dev_id = device_ids_[i];
      e.comm = comms[i];
      e.rank = i;
      cudaSetDevice(e.dev_id);
      cudaStreamCreate(&(e.stream));
      nccl_data_[device_ids_[i]] = e;
    }
  }

  using KeyAttrs = std::tuple<int, TShape, int>;
  // try to allocate buff on device evenly
  void InitMergeBuffer(const std::vector<Context>& devs) {
    for (size_t i = 0; i < sorted_key_attrs_.size(); ++i) {
      int key  = std::get<0>(sorted_key_attrs_[i]);
      TShape s = std::get<1>(sorted_key_attrs_[i]);
      int type = std::get<2>(sorted_key_attrs_[i]);
      auto& buf = merge_buf_[key];
      Context ctx;
      // use devs[0] as root
      ctx = devs[0];
      buf.merged = NDArray(s, ctx, false, type);
    }
    inited_ = true;
  }

  std::vector<KeyAttrs> sorted_key_attrs_;
  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the merged value
    NDArray merged;
  };
  struct NCCLEntry {
    /// \brief device ID
    int dev_id;
    /// \brief NCCL commmunicator
    ncclComm_t comm;
    /// \brief NCCL rank
    int rank;
    /// \brief GPU stream to use with NCCL
    cudaStream_t stream;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;
  std::unordered_map<int, NCCLEntry> nccl_data_;
  bool inited_;
  // \brief devices used with this KVStore
  std::vector<int> device_ids_;
};
#endif  // MXNET_USE_NCCL
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_COMM_H_
