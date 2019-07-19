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
 * \file exec_utils.h
 * \brief Common utility functions for executors.
 */
#ifndef MXNET_COMMON_EXEC_UTILS_H_
#define MXNET_COMMON_EXEC_UTILS_H_

#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../common/utils.h"
#include "../executor/exec_pass.h"

namespace mxnet {
namespace common {

/*
 * \brief setup default-storage tblobs from source NDArrays. If any source NDArray has non-default
 *        storage, it creates a temp NDArray with default storage and uses the temp tblob. The
 *        function also records the indices of non-default source NDArrays and the indices of
 *        their corresponding temporary NDArrays in the temp array.
 * \param src list of source NDArray
 * \param blobs list of tblobs to return
 * \param temp_src list of source NDArrays which requires temporary default storage representation
 * \param temp_dst list of temporary destination NDArrays for default storage representation
 * \param idx_map mapping from indices in source NDArrays to indices in temp_dst. When not set,
          indices are not recorded
 * \return true if any source NDArray need to cast storage
 */
bool SetupDefaultBlobsIn(const std::vector<NDArray>& src,
                                const std::vector<NDArray> *bufs,
                                std::vector<TBlob> *blobs,
                                std::vector<NDArray> *temp_src,
                                std::vector<NDArray> *temp_dst,
                                std::unordered_map<uint32_t, uint32_t> *idx_map);

bool SetupDefaultBlobsOut(const std::vector<NDArray>& src,
                                 const std::vector<NDArray> *bufs,
                                 std::vector<OpReqType> *req,
                                 std::vector<TBlob> *blobs,
                                 std::vector<NDArray> *temp_src,
                                 std::vector<NDArray> *temp_dst);

/*
 * \brief setup default-storage tblobs for input and output NDArrays.
 *        If any NDArray has non-default storage,
 *        it creates a temp NDArray with default storage and uses the temp tblob. The
 *        function also records the indices of non-default source NDArrays and the indices of
 *        their corresponding temporary NDArrays in the temp array.
 */
void SetupDefaultBlobsInOut(const std::vector<NDArray> &ndinputs,
                                   const std::vector<NDArray> &ndoutputs,
                                   const std::vector<NDArray> *in_bufs,
                                   const std::vector<NDArray> *out_bufs,
                                   std::vector<OpReqType> *req,
                                   std::vector<TBlob> *input_blobs,
                                   std::vector<TBlob> *output_blobs,
                                   std::vector<NDArray> *pre_temp_src,
                                   std::vector<NDArray> *pre_temp_dst,
                                   std::vector<NDArray> *post_temp_src,
                                   std::vector<NDArray> *post_temp_dst,
                                   std::unordered_map<uint32_t, uint32_t> *in_temp_idx_map,
                                   const std::vector<uint32_t> &mutate_idx);

/*
 * \brief cast the NDArrays in `src` and store the result in NDArrays in `dst`.
 *        This is only used for storage fallback in executor.
 * \param src list of source NDArray to cast
 * \param dst list of destionation NDArray which hold the result of cast_storage operation
 * \param ctx operator context for cast_storage operation
 */
void CastNonDefaultStorage(const std::vector<NDArray>& src,
                                  const std::vector<NDArray>& dst,
                                  const OpContext& ctx,
                                  const bool is_gpu);

/*! \brief The default type inference function, which assigns all undefined
 *         types to the same type of one of the inputs or outputs.
 */
bool SameType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr);


/*! \brief The default storage type inference function, which assigns all undefined
 *         storage types to kDefaultStorage. If all of input and output storage types
 *         are kDefaultStorage, DispatchMode::kFCompute is assigned to dispatch_mode. Otherwise,
 *         DispatchMode::kFComputeFallback is assigned to dispatch_mode.
 */
bool DefaultStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *iattr,
                               std::vector<int> *oattr);

// string representation of storage id
std::string storage_str(int storage_id);

/* log the static memory plan of the graph. Example:
   node 0 var
   node 1 _copy
            input 0: [80,3,224,224] (47040 KB) -> var storage (-1)
            output 1: [80,3,224,224] (47040 KB) -> group 0
   node 2 var
   node 3 var
   node 4 var
   node 5 var
   node 6 BatchNorm
            input 1: [80,3,224,224] (47040 KB) -> group 0
            input 2: [3] (0 KB) -> var storage (-1)
            input 3: [3] (0 KB) -> var storage (-1)
            input 4: [3] (0 KB) -> var storage (-1)
            input 5: [3] (0 KB) -> var storage (-1)
            output 6: [80,3,224,224] (47040 KB) -> group 1
            output 7: [3] (0 KB) -> group 3
            output 8: [3] (0 KB) -> group 2
   ...
 */
void LogMemoryPlan(const nnvm::Graph& g);

/* log the static memory plan of the graph. Example:
    node 0 var
    node 1 _copy: fcompute
                input 0: default
                output 1: default
    node 2 var
    node 3 Convolution: fcompute
                input 1: default
                input 2: default
                output 3: default
    node 4 var
    node 5 var
    node 6 var
    node 7 var
    node 8 BatchNorm: fcompute
                input 3: default
                input 4: default
                input 5: default
                input 6: default
                input 7: default
                output 8: default
                output 9: default
                output 10: default
    ...
 */
void LogInferStorage(const nnvm::Graph& g);

// prints a helpful message after shape inference errors in executor.
void HandleInferShapeError(const size_t num_forward_inputs,
                           const nnvm::IndexedGraph& idx,
                           const mxnet::ShapeVector& inferred_shapes);

// prints a helpful message after type inference errors in executor.
void HandleInferTypeError(const size_t num_forward_inputs,
                          const nnvm::IndexedGraph& idx,
                          const nnvm::DTypeVector& inferred_dtypes);

// prints a helpful message after storage type checking errors in executor.
void HandleInferStorageTypeError(const size_t num_forward_inputs,
                                 const nnvm::IndexedGraph& idx,
                                 const StorageTypeVector& inferred_stypes);

/*!
 * \brief If the requested ndarray's shape size is less than
 * the corresponding shared_data_array's shape size and the
 * storage type is shareable, reuse the memory allocation
 * in shared_buffer; otherwise, create a zero ndarray.
 * Shareable storages include both default storage and row_sparse storage
 * if enable_row_sparse_sharing is `True`, otherwise default storage only.
 */
NDArray ReshapeOrCreate(const std::string& name,
                        const mxnet::TShape& dest_arg_shape,
                        const int dest_arg_dtype,
                        const NDArrayStorageType dest_arg_stype,
                        const Context& ctx,
                        std::unordered_map<std::string, NDArray>* shared_buffer,
                        bool enable_row_sparse_sharing);

/*!
 * \brief Assign context to the graph.
 * This is triggered by both simple_bind and bind flows.
 */
nnvm::Graph AssignContext(nnvm::Graph g,
                          const Context& default_ctx,
                          const std::map<std::string, Context>& ctx_map,
                          const std::vector<Context>& in_arg_ctxes,
                          const std::vector<Context>& arg_grad_ctxes,
                          const std::vector<Context>& aux_state_ctxes,
                          const std::vector<OpReqType>& grad_req_types,
                          size_t num_forward_inputs,
                          size_t num_forward_outputs);

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_EXEC_UTILS_H_

