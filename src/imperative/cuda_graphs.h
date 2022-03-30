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

/*!
 * Copyright (c) 2020 by Contributors
 * \file cuda_graphs.h
 * \brief Wrappers for use of CUDA Graphs API
 */
#ifndef MXNET_IMPERATIVE_CUDA_GRAPHS_H_
#define MXNET_IMPERATIVE_CUDA_GRAPHS_H_

#include <mxnet/base.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <sstream>

#include "./exec_pass.h"
#include "../common/cuda/utils.h"

#if MXNET_USE_CUDA
#define CUDA_GRAPHS_AVAILABLE (CUDA_VERSION >= 10020)
#else
#define CUDA_GRAPHS_AVAILABLE (0)
#endif

#if CUDA_GRAPHS_AVAILABLE

namespace mxnet {
namespace cuda_graphs {

inline std::string CudaDim3ToString(const dim3& dims) {
  std::stringstream ss;
  if (dims.z != 1)
    ss << "(" << dims.x << "," << dims.y << "," << dims.z << ")";
  else if (dims.y != 1)
    ss << "(" << dims.x << "," << dims.y << ")";
  else
    ss << "(" << dims.x << ")";
  return ss.str();
}

// Return the list of CUDA Graph nodes from a graph
inline std::vector<cudaGraphNode_t> GetCudaGraphNodes(cudaGraph_t cuda_graph) {
  size_t numNodes;
  CUDA_CALL(cudaGraphGetNodes(cuda_graph, static_cast<cudaGraphNode_t*>(nullptr), &numNodes));
  if (numNodes == 0)
    return std::vector<cudaGraphNode_t>();
  std::vector<cudaGraphNode_t> graphNodes(numNodes);
  CUDA_CALL(cudaGraphGetNodes(cuda_graph, graphNodes.data(), &numNodes));
  return graphNodes;
}

// Create a description of a CUDA Graph node
inline std::string CudaGraphNodeToString(const cudaGraphNode_t node) {
  std::stringstream ss;

  // The following introspection calls are made through the driver API in order to bypass
  // problems that would arise if multiple statically-linked copies of the runtime exist.

  CUgraphNode cu_node = node;
  CUgraphNodeType t;
  CUDA_DRIVER_CALL(cuGraphNodeGetType(cu_node, &t));
  switch (t) {
    case CU_GRAPH_NODE_TYPE_KERNEL: {
      CUDA_KERNEL_NODE_PARAMS kparams;
      auto err = cuGraphKernelNodeGetParams(cu_node, &kparams);
      if (err == CUDA_SUCCESS) {
        ss << "GPUKernel@" << kparams.func;
        dim3 gridDim(kparams.gridDimX, kparams.gridDimY, kparams.gridDimZ);
        dim3 blockDim(kparams.blockDimX, kparams.blockDimY, kparams.blockDimZ);
        ss << "<<<gridDim=" << CudaDim3ToString(gridDim)
           << ", blkDim=" << CudaDim3ToString(blockDim) << ">>>";
        ss << "(...";
        if (kparams.sharedMemBytes != 0)
          ss << ", dynSharedMemBytes=" << kparams.sharedMemBytes;
        ss << ")";
      } else {
        ss << "GPU Kernel: cuGraphKernelNodeGetParams() fails with " << err;
      }
    } break;
    case CU_GRAPH_NODE_TYPE_MEMCPY: {
      cudaMemcpy3DParms mparams = {};
      CUDA_CALL(cudaGraphMemcpyNodeGetParams(node, &mparams));
      // If memcpy is seen, return without setting up runnable executor
      switch (mparams.kind) {
        case cudaMemcpyHostToHost:
          ss << "Host->Host ";
          break;
        case cudaMemcpyHostToDevice:
          ss << "Host->Device ";
          break;
        case cudaMemcpyDeviceToHost:
          ss << "Device->Host ";
          break;
        case cudaMemcpyDeviceToDevice:
          ss << "Device->Device ";
          break;
        default:
          break;
      }
      ss << "Memcpy";
    } break;
    case CU_GRAPH_NODE_TYPE_MEMSET: {
      cudaMemsetParams mparams = {};
      CUDA_CALL(cudaGraphMemsetNodeGetParams(node, &mparams));
      if (mparams.height == 1 && mparams.elementSize == 1) {
        ss << "cudaMemset(devPtr=" << mparams.dst << ", value=" << mparams.value
           << ", count=" << mparams.width << ")";
      } else {
        if (mparams.elementSize == 1)
          ss << "cudaMemset2D";
        else
          ss << "MemSet<elemBytes=" << mparams.elementSize << ">";
        ss << "(devPtr=" << mparams.dst << ", pitch=" << mparams.pitch
           << ", value=" << mparams.value << ", width=" << mparams.width
           << ", height=" << mparams.height << ")";
      }
    } break;
    case CU_GRAPH_NODE_TYPE_HOST:
      ss << "Host (executable) node";
      break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
      ss << "Node which executes an embedded graph";
      break;
    case CU_GRAPH_NODE_TYPE_EMPTY:
      ss << "Empty (no-op) node";
      break;
    default:
      ss << "Unknown/Invalid node type " << t;
  }
  return ss.str();
}

// CUDA Graphs are managed in RAII fashion by smart pointers below.
// Function objects (preferred for readability) provide the deleter function.
class CudaGraphDeleter {
 public:
  void operator()(cudaGraph_t graph) {
    if (graph != nullptr)
      CUDA_CALL(cudaGraphDestroy(graph));
  }
};

// CUDA Graphs Executors are managed in RAII fashion by smart pointers below.
// Function objects (preferred for readability) provide the deleter function.
class CudaGraphExecDeleter {
 public:
  void operator()(cudaGraphExec_t graph_exec) {
    if (graph_exec != nullptr)
      CUDA_CALL(cudaGraphExecDestroy(graph_exec));
  }
};

// A CUDA Graphs executor for a portion of an Operator Segment (i.e. a 'SubSegment'),
// characterized by a starting index in the OpExecutor list and a number of ops.
class CudaGraphsSubSegExec {
 public:
  CudaGraphsSubSegExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                       const RunContext& rctx,
                       bool is_gpu,
                       bool verbose,
                       int from_op_idx,
                       int num_ops,
                       bool ops_are_cuda_graph_compatible = true)
      : from_op_idx_(from_op_idx),
        num_ops_(num_ops),
        graph_(nullptr),
        graph_exec_(nullptr),
        graph_exec_id_(0) {
    if (ops_are_cuda_graph_compatible) {
      MakeGraph(exec_list, rctx, is_gpu, verbose, from_op_idx, num_ops);
      MakeGraphExec(exec_list, rctx);
    }
  }

  void Update(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
              const RunContext& rctx,
              bool is_gpu,
              bool verbose) {
    // Current executor should be Runnable with the same parameters
    CHECK(IsRunnable());
    MakeGraph(exec_list, rctx, is_gpu, verbose, from_op_idx_, num_ops_);

    cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
    cudaGraphNode_t error_node;
    cudaError_t err =
        cudaGraphExecUpdate(graph_exec_.get(), graph_.get(), &error_node, &update_result);
    switch (err) {
      case cudaErrorGraphExecUpdateFailure:
        MakeGraphExec(exec_list, rctx);
        break;
      case cudaSuccess:
        CHECK_EQ(update_result, cudaGraphExecUpdateSuccess);
        break;
      default:
        // Respond normally to unusual cudaGraphExecUpdate() ret vals
        CUDA_CALL(err);
    }
  }

  void RunSubSeg(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 const RunContext& rctx,
                 bool is_gpu) {
    if (IsRunnable()) {
      auto s                  = rctx.get_stream<gpu>();
      const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
      CUDA_CALL(cudaGraphLaunch(graph_exec_.get(), cu_s));
    } else {
      // No CUDA Graph could be made for this portion of the OpSegment.  Run conventionally.
      for (int i = 0; i != num_ops_; ++i)
        exec_list[from_op_idx_ + i]->Run(rctx, is_gpu);
    }
  }

  bool IsRunnable() {
    return graph_exec_ != nullptr;
  }

  int NumGraphNodes() {
    size_t numNodes;
    CUDA_CALL(cudaGraphGetNodes(graph_.get(), static_cast<cudaGraphNode_t*>(nullptr), &numNodes));
    return numNodes;
  }

 private:
  void MakeGraph(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 const RunContext& rctx,
                 bool is_gpu,
                 bool verbose,
                 int from_op_idx,
                 int num_ops) {
    auto s                  = rctx.get_stream<gpu>();
    const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
    // Create CUDA Graph
    // Use of cudaStreamCaptureModeThreadLocal allows other threads like GPU Copy workers
    // to sync their streams without disturbing this capture.
    CUDA_CALL(cudaStreamBeginCapture(cu_s, cudaStreamCaptureModeThreadLocal));
    // Run those oprs in the sub segment while capturing- no actual GPU work is launched.
    for (int i = 0; i != num_ops; ++i)
      exec_list[from_op_idx + i]->Run(rctx, is_gpu);
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CALL(cudaStreamEndCapture(cu_s, &cuda_graph));
    graph_.reset(cuda_graph, CudaGraphDeleter());

    if (verbose) {
      std::vector<cudaGraphNode_t> graph_nodes = GetCudaGraphNodes(cuda_graph);
      size_t num_nodes                         = graph_nodes.size();
      LOG(INFO) << "  Graph has " << num_nodes << " nodes:";
      for (size_t i = 0; i != num_nodes; ++i) {
        LOG(INFO) << "    node " << i << " = " << CudaGraphNodeToString(graph_nodes[i]);
      }
    }
  }

  void MakeGraphExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                     const RunContext& rctx) {
    // Note that this routine is not invoked when a graph executor is merely updated.
    cudaGraphExec_t cuda_graph_exec;
    cudaGraphNode_t error_node;
    char log_buffer[1000];

    CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec, graph_.get(), &error_node, log_buffer, 1000));
    graph_exec_.reset(cuda_graph_exec, CudaGraphExecDeleter());

    // At this point we have a CUDA Graph executor
    static int num_graph_creations = 0;
    graph_exec_id_                 = num_graph_creations++;

    static size_t max_log_entries = dmlc::GetEnv("MXNET_CUDA_GRAPHS_MAX_LOG_ENTRIES", 0);
    if (graph_exec_id_ < max_log_entries) {
      LOG(INFO) << "Created CUDA graph " << graph_exec_id_;
      if (num_graph_creations == max_log_entries)
        LOG(INFO) << "Further CUDA graph creation log messages are suppressed.";
    }
    // Create a .dot file for graph visualization if requested
    static std::string dotfile_base = dmlc::GetEnv("MXNET_CUDA_GRAPHS_DBG_FILE", std::string());
    if (dotfile_base.size() > 0) {
#if CUDA_VERSION >= 11030
      static int dotfile_flags = dmlc::GetEnv("MXNET_CUDA_GRAPHS_DBG_FILE_FLAGS",
                                              static_cast<int>(cudaGraphDebugDotFlagsVerbose));
      std::ostringstream filename;
      const bool is_train = exec_list.size() > 0 && exec_list[0]->op_ctx.is_train;
      int dev_id          = rctx.ctx.dev_id;
      filename << dotfile_base << "-"
               << "dev" << dev_id << "-" << (is_train ? "trn" : "inf") << "-" << graph_exec_id_
               << ".dot";
      CUDA_CALL(cudaGraphDebugDotPrint(graph_.get(), filename.str().c_str(), dotfile_flags));
#else
      [[maybe_unused]] static bool dot_file_unsupported = []() {  // NOLINT
        LOG(INFO) << "MXNET_CUDA_GRAPHS_DBG_FILE setting ignored- requires CUDA version >= 11.3";
        return true;
      }();
#endif  // CUDA_VERSION >= 11030
    }
  }

  int from_op_idx_;
  int num_ops_;
  using cudaGraphStruct_t     = typename std::remove_pointer<cudaGraph_t>::type;
  using cudaGraphExecStruct_t = typename std::remove_pointer<cudaGraphExec_t>::type;
  std::shared_ptr<cudaGraphStruct_t> graph_;
  std::shared_ptr<cudaGraphExecStruct_t> graph_exec_;
  size_t graph_exec_id_;
};

// The CudaGraph executor and associated Tempspace ptrs for which it is valid.
struct CudaGraphInfo {
  std::vector<CudaGraphsSubSegExec> cuda_graph_subseg_execs;
  bool has_been_run_conventionally = false;
  std::vector<void*> tempspace_dptrs;
};
// A CUDA graph is maintained for every combination of cudaStream_t (i.e. GPU Worker) and
// the state of the is_train flag of the OpContext.  If the tempspace_dptrs change, we
// don't expect to ever see the old tempspace_dptrs config again, so we discard the CUDA graph.
struct CudaGraphCacheKey {
  cudaStream_t cu_s;
  bool is_train;
  // overload '<' so CudaGraphCacheKey can be used as a std::map key
  bool operator<(const CudaGraphCacheKey& other) const {
    return cu_s < other.cu_s || (cu_s == other.cu_s && is_train < other.is_train);
  }
};
using CudaGraphCache = std::map<CudaGraphCacheKey, CudaGraphInfo>;

class CudaGraphsExec {
 public:
  CudaGraphsExec(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                 bool is_gpu,
                 const char* opr_names)
      : verbose_(false), is_enabled_(false) {
    opr_names_ = opr_names ? std::string(opr_names) : std::string();
    if (is_gpu) {
      is_enabled_ = dmlc::GetEnv("MXNET_ENABLE_CUDA_GRAPHS", false);
      verbose_    = dmlc::GetEnv("MXNET_CUDA_GRAPHS_VERBOSE", false);
      SetTempSpaces(exec_list);
    }
  }

  void RunAll(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
              const RunContext& rctx,
              bool is_gpu) {
    // If this a CPU op or CUDA Graphs use isn't possible, run normally and return
    if (!is_gpu || !is_enabled_) {
      // Run all opr in the sub-graph
      exec::OpExecutor::RunAll(exec_list, rctx, is_gpu);
      return;
    }

    // Also if we're in a warm-up period where tempspace pointers are likely
    // to change, run normally and return
    auto s                  = rctx.get_stream<gpu>();
    const cudaStream_t cu_s = mshadow::Stream<gpu>::GetStream(s);
    // All the ops in the bulked segment will have the same setting of is_train as the first op
    const bool is_train         = exec_list.size() > 0 && exec_list[0]->op_ctx.is_train;
    const CudaGraphCacheKey key = {cu_s, is_train};
    // Look-up the CUDA Graph info for this combo of stream and is_train setting
    // This may create a default-initialized new entry.
    auto& cuda_graph_info = cache_[key];
    if (!cuda_graph_info.has_been_run_conventionally) {
      // Run all opr in the sub-graph
      exec::OpExecutor::RunAll(exec_list, rctx, is_gpu);
      cuda_graph_info.has_been_run_conventionally = true;
      return;
    }

    // At this point we will launch one or more CUDA Graphs through CUDA Graphs 'executors'
    //     (there might be more than one executor if some ops in the segment are not capturable)
    auto before_exec_tempspace_ptrs = GetGPUTempspacePtrs(s);

    // Executors exist, but the tempspace pts have changed, so update them in-place via 'recapture'.
    if (cuda_graph_info.cuda_graph_subseg_execs.size() > 0 &&
        cuda_graph_info.tempspace_dptrs != before_exec_tempspace_ptrs) {
      // Update all runnable executors.  Non-runnable executors launch their ops conventionally.
      for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs) {
        if (subseg_exec.IsRunnable())
          subseg_exec.Update(exec_list, rctx, is_gpu, verbose_);
      }
    } else if (cuda_graph_info.cuda_graph_subseg_execs.size() == 0) {
      // No executors exist yet, so create them.
      if (verbose_)
        LOG(INFO) << "Capturing CUDA graph of op segment " << opr_names_;
      // Make one or more CUDA Graphs, avoiding ops that are not compatible.
      for (size_t first_op_idx = 0; first_op_idx != exec_list.size();) {
        int num_good_ops = 0;
        for (size_t last_op_idx = first_op_idx; last_op_idx != exec_list.size(); ++last_op_idx) {
          if (OpOK(exec_list[last_op_idx]))
            num_good_ops++;
          else
            break;
        }
        if (num_good_ops > 0) {
          CreateSubExecOverRegion(exec_list,
                                  rctx,
                                  is_gpu,
                                  first_op_idx,
                                  first_op_idx + num_good_ops,
                                  &cuda_graph_info.cuda_graph_subseg_execs);
          first_op_idx += num_good_ops;
        }
        if (first_op_idx != exec_list.size()) {
          // We had to have hit an op that was not OK.
          if (verbose_) {
            LOG(INFO) << "Bypassing notOK op segment[" << first_op_idx << "," << first_op_idx << "]"
                      << " of op segment " << opr_names_;
          }
          CudaGraphsSubSegExec notOK_opseg(exec_list, rctx, is_gpu, false, first_op_idx, 1, false);
          cuda_graph_info.cuda_graph_subseg_execs.push_back(notOK_opseg);
          first_op_idx++;
        }
      }
      // During graph capture, the ops may be asking for the tempworkspace.  This should
      // not alter the base pointers, since this op seg has been executed before on this
      // stream (i.e. on this gpu worker).  Safest to double-check this though.
      auto after_capture_tempspace_ptrs = GetGPUTempspacePtrs(s);
      if (before_exec_tempspace_ptrs != after_capture_tempspace_ptrs)
        LOG(FATAL) << "Internal error: saw change in TempSpace ptrs during CUDA graph use.";
      cuda_graph_info.tempspace_dptrs = before_exec_tempspace_ptrs;
    }
    // Now execute the CUDA Graph that we either just created or looked-up in the cache.
    if (verbose_) {
      int runnable_execs = 0;
      int bypassed_ops   = 0;
      for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs) {
        if (subseg_exec.IsRunnable()) {
          LOG(INFO) << "Launching captured graph with " << subseg_exec.NumGraphNodes() << " nodes.";
          runnable_execs++;
        } else {
          bypassed_ops++;
        }
      }
      if (bypassed_ops > 0)
        LOG(INFO) << "    (bypassing " << bypassed_ops << " un-capturable ops)";
    }
    for (auto& subseg_exec : cuda_graph_info.cuda_graph_subseg_execs)
      subseg_exec.RunSubSeg(exec_list, rctx, is_gpu);
  }

 private:
  // Make a CUDA Graph of the region of ops [from_op_idx, upto_op_idx).  If such a graph
  // is not runnable, e.g. if it includes memcpys from unpinned cpu memory, then make a
  // number of smaller graphs that avoid those ops with the memcpys.
  void CreateSubExecOverRegion(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list,
                               const RunContext& rctx,
                               bool is_gpu,
                               size_t from_op_idx,
                               size_t upto_op_idx,
                               std::vector<CudaGraphsSubSegExec>* cuda_graph_subseg_execs) {
    // Optimistically try to create a CUDA Graph of the entire op segment region

    int num_ops = upto_op_idx - from_op_idx;
    CudaGraphsSubSegExec full_opseg(exec_list, rctx, is_gpu, verbose_, from_op_idx, num_ops);
    if (full_opseg.IsRunnable()) {
      cuda_graph_subseg_execs->push_back(full_opseg);
    } else {
      if (verbose_)
        LOG(INFO) << "  Graph was not runnable- creating op sub-segments...";
      // Enter fall-back approach to making many sub-execs
      for (size_t first_op_idx = from_op_idx; first_op_idx != upto_op_idx;) {
        int num_good_ops = 0;
        for (size_t last_op_idx = first_op_idx; last_op_idx != upto_op_idx; ++last_op_idx) {
          CudaGraphsSubSegExec single_opseg(exec_list, rctx, is_gpu, false, last_op_idx, 1);
          if (single_opseg.IsRunnable())
            num_good_ops++;
          // Is it time to create a subseg exec from accumulated good ops?
          if (num_good_ops > 0 && (last_op_idx == upto_op_idx - 1 || !single_opseg.IsRunnable())) {
            if (verbose_)
              LOG(INFO) << "Capturing CUDA graph of op sub segment[" << first_op_idx << ":"
                        << (first_op_idx + num_good_ops - 1) << "]"
                        << " of op segment " << opr_names_;
            CudaGraphsSubSegExec good_opseg(
                exec_list, rctx, is_gpu, verbose_, first_op_idx, num_good_ops);
            CHECK(good_opseg.IsRunnable()) << "Unexpected issue with CUDA Graphs creation";
            cuda_graph_subseg_execs->push_back(good_opseg);
            first_op_idx += num_good_ops;
          }
          // If the last single op was not runnable, use the exec to handle that op conventionally
          if (!single_opseg.IsRunnable()) {
            if (verbose_) {
              LOG(INFO) << "Bypassing op sub segment[" << last_op_idx << "," << last_op_idx << "]"
                        << " of op segment " << opr_names_;
              // Generate throw-away exec in order to produce a diagnostic listing of graph nodes
              CudaGraphsSubSegExec dummy(exec_list, rctx, is_gpu, verbose_, last_op_idx, 1);
            }
            cuda_graph_subseg_execs->push_back(single_opseg);
            first_op_idx++;
            break;
          }
        }
      }
    }
  }

  // Is the Op OK to make part of a CUDA Graph?
  bool OpOK(const std::shared_ptr<exec::OpExecutor>& exec) {
    static auto& fgraphcompatible = Op::GetAttr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible");
    static auto& fcompute_ex      = Op::GetAttr<FComputeEx>("FComputeEx<gpu>");
    static auto& fstatefulcompute = Op::GetAttr<FStatefulCompute>("FStatefulCompute<gpu>");
    static auto& fstatefulcompute_ex = Op::GetAttr<FStatefulComputeEx>("FStatefulComputeEx<gpu>");
    const auto& attrs                = exec->attrs;
    if (attrs.op != nullptr) {
      const auto f = fgraphcompatible.get(attrs.op, nullptr);
      if (f != nullptr) {
        return f(attrs, exec->op_ctx.is_train);
      }
      if (fstatefulcompute.get(attrs.op, nullptr) != nullptr ||
          fstatefulcompute_ex.get(attrs.op, nullptr) != nullptr) {
        if (verbose_) {
          LOG(INFO) << "Omitting stateful operator " << attrs.op->name << " from CUDA graph.";
        }
        return false;
      }
      if ((fcompute_ex.get(attrs.op, nullptr) != nullptr &&
           exec->dispatch_mode == DispatchMode::kFComputeEx) ||
          exec->dispatch_mode == DispatchMode::kFComputeFallback) {
        if (verbose_) {
          LOG(INFO) << "Omitting operator " << attrs.op->name
                    << " from CUDA graph due to dispatch mode "
                    << static_cast<int>(exec->dispatch_mode);
        }
        return false;
      }
    }
    for (auto& resource : exec->op_ctx.requested) {
      if (!(resource.req.type == ResourceRequest::kTempSpace)) {
        if (verbose_) {
          LOG(INFO) << "Omitting operator " << attrs.op->name
                    << " from CUDA graph due to using the resource type "
                    << static_cast<int>(resource.req.type);
        }
        return false;
      }
    }
    return true;
  }

  // Determine Tempspaces used by ops.  Other resource uses disable CUDA Graphs.
  void SetTempSpaces(const std::vector<std::shared_ptr<exec::OpExecutor>>& exec_list) {
    // Gather info about the ops use of TempSpace.
    if (is_enabled_) {
      std::set<Resource*> tempspaces_set;
      for (auto& exec : exec_list) {
        for (auto& resource : exec->op_ctx.requested) {
          if (resource.req.type == ResourceRequest::kTempSpace) {
            tempspaces_set.insert(&resource);
          }
        }
      }
      tempspaces_.assign(tempspaces_set.begin(), tempspaces_set.end());
    }
  }

  // Return the addresses of the gpu TempSpace areas
  std::vector<void*> GetGPUTempspacePtrs(mshadow::Stream<gpu>* s) {
    std::vector<void*> ret;
    for (const auto& resource : tempspaces_) {
      // Ask for minimal allocation to get base pointer without increasing the size
      auto* base_ptr = resource->get_space_typed<gpu, 1, char>(mshadow::Shape1(1), s).dptr_;
      ret.push_back(static_cast<void*>(base_ptr));
    }
    return ret;
  }

  CudaGraphCache cache_;
  std::vector<Resource*> tempspaces_;
  std::string opr_names_;
  bool verbose_;
  bool is_enabled_;
};

}  // namespace cuda_graphs
}  // namespace mxnet

#endif  // CUDA_GRAPHS_AVAILABLE

#endif  // MXNET_IMPERATIVE_CUDA_GRAPHS_H_
