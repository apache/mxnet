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
 * Copyright (c) 2018 by Contributors
 */
#ifndef MXNET_KVSTORE_COMM_TREE_H_
#define MXNET_KVSTORE_COMM_TREE_H_
#include <dmlc/omp.h>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include <thread>
#include <map>
#include "mxnet/ndarray.h"
#include "gradient_compression.h"
#include "../ndarray/ndarray_function.h"
#include "../operator/tensor/sparse_retain-inl.h"
#include "./kvstore_utils.h"
#include "./gpu_topology.h"
namespace mxnet {
namespace kvstore {
/**
 * \brief an implementation of Comm that performs reduction on device
 * directly using tree.
 *
 * It is faster if the total device-to-device bandwidths is larger than
 * device-to-cpu, which is often true for 4 or 8 GPUs. But it uses more device
 * memory.
 */
class CommDeviceTree : public CommDevice {
 public:
  CommDeviceTree() {
    inited_ = false;
    gpuarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_TREE_ARRAY_BOUND", 10000000);
    backtrack_ = dmlc::GetEnv("MXNET_KVSTORE_TREE_BACKTRACK", 0);
    link_usage_penalty_ = dmlc::GetEnv("MXNET_KVSTORE_TREE_LINK_USAGE_PENALTY", 0.7);
  }

  virtual ~CommDeviceTree() { }

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int dtype = mshadow::kFloat32) override {
    tree_sorted_key_attrs_.emplace_back(key, shape, dtype);
    sorted_key_attrs_.emplace_back(key, shape, dtype);
  }

  void InitBuffersAndComm(const std::vector<NDArray>& src) {
    if (!inited_) {
      for (const auto& a : src) {
        devs_.push_back(a.ctx());
      }
      QueryTopology();
      // Note: delayed allocation set to true, because we do not want to allocate
      // both in TreeBufferEntry and BufferEntry, so we use a size_t to keep
      // track of each key's shape within BufferEntry
      // -this information is required for inherited Reduce- and
      //  BroadcastRowSparse
      InitMergeBuffer(devs_);
      InitMergeBufferTree();
    }
  }

  /**
   * \brief Reduce src to tree_merge_buf_
   * \param key is the id of the gradient we are doing Reduce on
   * \param src is the array of values located on different GPUs
   * \param root is the id of the GPU we want to send result of reduce to
   * \param merged_row is the id of the slice we are taking
   * \param priority the priority of the operation
   */
  const NDArray& ReduceInner(int key, const std::vector<NDArray>& src, int root,
                             int merged_row, int priority) {
    std::vector<std::vector<NDArray>> reduce(devs_.size());

    TreeBufferEntry& random_buf = tree_merge_buf_[0][key];
    const NDArrayStorageType stype = random_buf.merged[0].storage_type();
    std::vector<size_t>& topology = topology_[root];
    NDArray buf_slice;

    if (stype == kDefaultStorage) {
      // Copy everything into buf.merged for each gpu
      for (const auto& src_gpu_value : src) {
        int start = scan_[root][depth_];
        int end = scan_[root][depth_+1];

        for (int j = start; j < end; ++j) {
          int topo_id = topology[j];
          TreeBufferEntry& buf = tree_merge_buf_[topo_id][key];

          if (devs_[topo_id] == src_gpu_value.ctx()) {
            CopyFromTo(src_gpu_value, &(buf.merged[merged_row]), priority);
          }
        }
      }

      for (int level = depth_; level > 0; --level) {
        int start = scan_[root][level  ];
        int end = scan_[root][level+1];

        unsigned is_dest = 0;
        int dest_id = 0;
        for (int j = start; j < end; ++j) {
          int topo_id = topology[j];
          dest_id = (is_dest == 0) ? topo_id : dest_id;

          TreeBufferEntry& buf_dest = tree_merge_buf_[dest_id][key];
          TreeBufferEntry& buf_from = tree_merge_buf_[topo_id][key];

          if (!is_dest) {
            if (reduce[dest_id].size() == 0) {
              reduce[dest_id].push_back(buf_dest.merged[merged_row]);
            }
          } else {
            if (dest_id != topo_id) {
              CopyFromTo(buf_from.merged[merged_row],
                         &(buf_dest.copy_buf[merged_row][is_dest-1]),
                         priority);
              reduce[dest_id].push_back(
                  buf_dest.copy_buf[merged_row][is_dest-1]);
            }
          }

          is_dest = (is_dest == static_cast<unsigned>(kBranch)-1) ?
              0 : is_dest+1;
        }

        start = scan_[root][level-1];
        end = scan_[root][level];
        int source = end;
        for (int i = start; i < end; ++i) {
          int gpu_id = topology[i];

          // source keeps track of 2 leaf nodes, while start keeps track of parent
          int dest_id = topology[source];
          int from_id = topology[source+1];
          source += 2;

          // conditional to detect whether operation must be done
          if (reduce[gpu_id].size() > 1 && dest_id != from_id) {
            TreeBufferEntry& buf = tree_merge_buf_[gpu_id][key];
            ElementwiseSum(reduce[gpu_id], &(buf.merged[merged_row]), priority);
          }
        }

        // reset
        for (unsigned i = 0; i < devs_.size(); ++i) {
          reduce[i].clear();
        }
      }
    } else {
      LOG(FATAL) << "Only dense input supported for now";
    }

    int topo_id = topology[0];
    TreeBufferEntry& buf = tree_merge_buf_[topo_id][key];
    return buf.merged[merged_row];
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    // when this reduce is called from kvstore_dist, gc is not set
    // we don't do compression twice in dist_sync_device
    if ((gc_ != nullptr) && (gc_->get_type() != CompressionType::kNone)) {
      return ReduceCompressed(key, src, priority);
    }

    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      return src[0];
    }

    InitBuffersAndComm(src);
    std::vector<std::vector<NDArray>>  slice(devs_.size());
    std::vector<std::vector<NDArray*>> broadcast_slice(devs_.size());
    std::vector<int>                   slice_scan(devs_.size()+1);

    int total_size = src[0].shape().Size();
    unsigned first_size = src[0].shape()[0];

    const NDArrayStorageType stype = src[0].storage_type();
    // normal dense reduce
    if (stype == kDefaultStorage) {
      if (total_size > gpuarray_bound_ && first_size >= 2*devs_.size()) {
        // Find slice bounds
        slice_scan[0] = 0;
        int slice_size = first_size/devs_.size();
        for (unsigned i = 1; i < devs_.size(); ++i) {
          slice_scan[i] = slice_scan[i-1] + slice_size;
        }
        slice_scan[devs_.size()] = src[0].shape()[0];

        // row: which slice
        // col: which gpu
        for (unsigned row = 0; row < devs_.size(); ++row) {
          for (unsigned col = 0; col < devs_.size(); ++col) {
            TreeBufferEntry& buf = tree_merge_buf_[col][key];
            NDArray curr_slice = src[col].Slice(slice_scan[row],
                slice_scan[row+1]);
            slice[row].push_back(curr_slice);
            broadcast_slice[row].push_back(&(buf.merged[row]));
          }
        }

        // Do reduce-scatter (multiroot reduce)
        // input:  slice (src)
        // output: buf.merge_buf
        for (unsigned i = 0; i < devs_.size(); ++i) {
          ReduceInner(key, slice[i], i, i, priority);
        }

        for (unsigned i = 0; i < devs_.size(); ++i) {
          BroadcastInner(key, *(broadcast_slice[i][i]), broadcast_slice[i], i, i, priority);
        }
      } else {
        int root = 0;
        ReduceInner(key, src, root, 0, priority);

        TreeBufferEntry& buf = tree_merge_buf_[root][key];
        return buf.merged[0];
      }

      // Copy from list of small NDArrays to one big NDArray, which is returned
      int gpu_id = 0;
      return src[gpu_id];
    } else {
      // sparse reduce
      return ReduceRowSparse(key, src, priority);
    }
  }

  void BroadcastInner(int key, const NDArray& src,
                      const std::vector<NDArray*>& dst, int root,
                      int merged_row, int priority) {
    // copy to root of tree
    std::vector<size_t>& topology = topology_[root];
    std::vector<NDArray> temp(devs_.size());
    int gpu_id = topology[0];
    if (merged_row == -1)
      CopyFromTo(src, dst[gpu_id], priority);
    temp[gpu_id] = *dst[gpu_id];

    for (int level = 1; level <= depth_; ++level) {
      int start = scan_[root][level];
      int end = scan_[root][level+1];

      unsigned is_src = 0;
      int src_id = 0;
      for (int j = start; j < end; ++j) {
        int topo_id = topology[j];
        src_id = (is_src == 0) ? topo_id : src_id;

        if (is_src && src_id != topo_id) {
          CopyFromTo(temp[src_id], dst[topo_id], priority);
          temp[topo_id] = *dst[topo_id];
        }

        is_src = (is_src == static_cast<unsigned>(kBranch)-1) ? 0 : is_src+1;
      }
    }
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
      int total_size = src.shape().Size();
      unsigned first_size = src.shape()[0];
      const NDArrayStorageType stype = src.storage_type();
      // normal dense reduce
      if (stype == kDefaultStorage) {
      if (total_size > gpuarray_bound_ && first_size >= 2*devs_.size()) {
        std::vector<int> slice_scan(devs_.size()+1);
        slice_scan[0] = 0;
        int slice_size = (dst[0]->shape()[0])/devs_.size();
        for (unsigned i = 1; i < devs_.size(); ++i) {
          slice_scan[i] = slice_scan[i-1] + slice_size;
        }
        slice_scan[devs_.size()] = dst[0]->shape()[0];

        for (unsigned gpu_id = 0; gpu_id < dst.size(); ++gpu_id) {
          TreeBufferEntry& buf = tree_merge_buf_[gpu_id][key];
          for (unsigned i = 0; i < devs_.size(); ++i) {
            if (devs_[gpu_id] == dst[gpu_id]->ctx()) {
              NDArray curr_slice = dst[gpu_id]->Slice(slice_scan[i], slice_scan[i+1]);
              CopyFromTo(buf.merged[i], &curr_slice, priority);
            }
          }
        }
      } else {
        int root = 0;
        BroadcastInner(key, src, dst, root, -1, priority);
      }} else {
        LOG(FATAL) << "Only dense input supported for now";
      }
    }
  }

 private:
  void EnableP2P(std::vector<int>* p2p) {
#if MXNET_USE_CUDA
    std::vector<int> gpus;
    for (const auto& d : devs_) {
      if (d.dev_mask() == gpu::kDevMask) {
        gpus.push_back(d.dev_id);
      }
    }
    int n = static_cast<int>(gpus.size());
    int enabled = 0;
    p2p->clear();
    p2p->resize(n*n, 0);
    for (int i = 0; i < n; ++i) {
      mxnet::common::cuda::DeviceStore device_store(gpus[i]);
      for (int j = 0; j < n; j++) {
        int access;
        cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
        if (access) {
          cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
          if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            ++enabled;
            (*p2p)[i*n+j] = 1;
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
          access[j] = (*p2p)[i*n+j] ? 'v' : '.';
        }
        LOG(WARNING) << access;
      }
    }
#endif
  }

  void QueryTopology() {
#if MXNET_USE_CUDA
    std::vector<float> link_matrix(devs_.size()*devs_.size());
    std::vector<int> p2p_matrix(devs_.size()*devs_.size());
    EnableP2P(&p2p_matrix);
    GetP2PWeight(devs_, p2p_matrix, &link_matrix);
    if (backtrack_)
      LOG(INFO) << "Using Backtracking to generate trees";
    else
      LOG(INFO) << "Using Kernighan-Lin to generate trees";
    ComputeTrees(link_matrix, devs_.size(), link_usage_penalty_, backtrack_,
        &topology_, &scan_);

    depth_ = ComputeDepth(devs_.size());
#endif
  }

  using KeyAttrs = std::tuple<int, TShape, int>;
  // try to allocate buff on device evenly
  void InitMergeBufferTree() {
    LOG(INFO) << "Using Tree";

    // same as all-reduce, except:
    // 1) Allocate copy_buf here instead of in Reduce()
    // 2) Force copy_buf to be of kRecvBufferSize
    // 3) Do not use greedy assignment; all keys are assigned to each GPU
    for (unsigned i = 0; i < devs_.size(); ++i)
      tree_merge_buf_.emplace_back();

    bool delay_alloc = true;
    std::map<int, int> key_dist;

    for (auto& tree_sorted_key_attr : tree_sorted_key_attrs_) {
      const int key  = std::get<0>(tree_sorted_key_attr);
      const TShape& shape = std::get<1>(tree_sorted_key_attr);
      const int type = std::get<2>(tree_sorted_key_attr);

      if (key_dist.find(shape.Size()) == key_dist.end())
        key_dist[shape.Size()] = 1;
      else
        key_dist[shape.Size()]++;

      int start = scan_[0][depth_];
      int end = scan_[0][depth_+1];

      // In order to generalize to any number of GPUs in arbitrary order, we use
      // strategy of having found the mapping from 0, 1, ..., n_gpus to dev_id.
      // For example, if the user wants to use --gpus 4,2,3,1,7,5,0, they can do      // so:
      //
      //   idx:    0 1 2 3 4 5 6
      //   dev_id: 4 2 3 1 7 5 0
      //
      // From this, we:
      // 1) generate a link topology matrix with dimensions n_gpus x n_gpus
      //    (link_matrix)
      //
      // 2) the reduction trees are saved as indices from 0, 1, ..., n_gpus
      //    in a vector of vectors (topology_):
      //
      //    index  | topology_[index]
      //    -------------------------
      //    0      | [Tree 0]
      //    1      | [Tree 1]
      //           .
      //           .
      //           .
      //    n_gpus | [Tree n_gpus]
      //
      // 3) We use the mapping (devs_) to retrieve dev_id and device context
      for (int j = start; j < end; ++j) {
        int topo_id = topology_[0][j];
        auto& buf = tree_merge_buf_[topo_id][key];
        Context ctx = devs_[topo_id];

        // buf.merged enforces that we only visit each GPU once
        if (buf.merged.empty()) {
          TShape shape_copy = shape;
          int total_size = shape.Size();
          unsigned first_size = shape[0];
          if (total_size > gpuarray_bound_ && first_size >= 2*devs_.size()) {
            // Find slice bounds
            int slice_size = first_size/devs_.size();
            int last_slice = first_size-(devs_.size()-1)*slice_size;
            shape_copy[0]   = slice_size;
            buf.merged.resize(devs_.size());
            for (unsigned row = 0; row < devs_.size(); ++row) {
              if (row == devs_.size()-1)
                shape_copy[0] = last_slice;
              buf.merged[row] = NDArray(shape_copy, ctx, delay_alloc, type);
              buf.copy_buf.emplace_back();
              if (buf.copy_buf[row].empty()) {
                buf.copy_buf[row].resize(kBranch-1);
                for (size_t col = 0; col < buf.copy_buf[0].size(); ++col) {
                  buf.copy_buf[row][col] = NDArray(buf.merged[row].shape(),
                                                   buf.merged[row].ctx(),
                                                   delay_alloc,
                                                   buf.merged[row].dtype());
                }
              }
            }
          } else {
            buf.merged.emplace_back(shape, ctx, false, type);
            if (buf.copy_buf.empty()) {
              buf.copy_buf.emplace_back();
              buf.copy_buf[0].resize(kBranch-1);
              for (size_t col = 0; col < buf.copy_buf[0].size(); ++col) {
                buf.copy_buf[0][col] = NDArray(buf.merged[0].shape(),
                                               buf.merged[0].ctx(), delay_alloc,
                                               buf.merged[0].dtype());
              }
            }
          }
        }
      }
    }

    for (auto& kv : key_dist) {
      LOG(INFO) << "Size " << kv.first << " occurs " << kv.second << " times";
    }
    inited_ = true;
  }

  std::vector<KeyAttrs> tree_sorted_key_attrs_;
  /// \brief temporal space for pushing and pulling
  struct TreeBufferEntry {
    /// \brief the dense merged value for reduce and broadcast operations
    std::vector<NDArray> merged;
    /// \brief the gpu buffer for copy during reduce operation
    std::vector<std::vector<NDArray>> copy_buf;
    /// \brief the residual buffer for gradient compression
    std::vector<NDArray> residual;
    /// \brief the small buffer for compressed data in sender
    std::vector<NDArray> compressed_send_buf;
    /// \brief the small buffer for compressed data in receiver
    std::vector<NDArray> compressed_recv_buf;

   private:
    /// \brief the sparse merged value for reduce and rowsparse broadcast operations
    NDArray sparse_merged;
  };
  /// \brief intent of tree_merge_buf_ in old comm.h: store key->gpu mapping
  ///        new intent: for every gpu: store key->memory mapping
  std::vector<std::unordered_map<int, TreeBufferEntry>> tree_merge_buf_;

  /// \brief NVLink-connected topology in full binary tree format
  std::vector<std::vector<size_t>> topology_;
  std::vector<std::vector<size_t>> scan_;
  std::vector<Context> devs_;

  int depth_;
  int gpuarray_bound_;
  bool backtrack_;
  float link_usage_penalty_;

  /// \brief constant for maximum size of recv buffer per GPU
  ///        2: only receive from 1 other GPU
  const int kBranch = 2;
};

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_COMM_TREE_H_
