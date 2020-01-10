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

#ifndef MXNET_OPERATOR_FUSION_FUSED_OP_H_
#define MXNET_OPERATOR_FUSION_FUSED_OP_H_

#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>
#include <utility>
#include <mutex>
#include <tuple>

#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

namespace mxnet {

namespace fusion {
  enum KernelVariants {kGeneral, kShapeOptimized,
    kNumKernelVariants  // Not a variant- leave this at the end
  };
}

struct FusedOpConfig : public dmlc::Parameter<FusedOpConfig> {
  int num_inputs;
  int num_outputs;
  DMLC_DECLARE_PARAMETER(FusedOpConfig) {
    DMLC_DECLARE_FIELD(num_inputs)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("Number of outputs.");
  }
};

struct FusedOpEntry {
  FusedOpEntry() : dtype(-1), ndim(-1) {}
  int dtype;
  int ndim;
};

class FusedOp {
 public:
  static const int NTHREADS = 512;
  static const int CACHESIZE_WARN_THRESHOLD = 10000;

  explicit FusedOp(const nnvm::NodeAttrs* attrs, const FusedOpConfig& config);
  ~FusedOp() {}
  uint32_t num_inputs() const {
    return inputs_.size();
  }
  uint32_t num_outputs() const {
    return outputs_.size();
  }

  template <typename xpu>
  void Forward(const nnvm::NodeAttrs& attrs,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs);

  bool InferShape(const nnvm::NodeAttrs &attrs,
                  std::vector<mxnet::TShape> *in_attrs,
                  std::vector<mxnet::TShape> *out_attrs);

  bool InferType(const nnvm::NodeAttrs &attrs,
                 std::vector<int> *in_attrs,
                 std::vector<int> *out_attrs);

  template <typename Attr>
  std::tuple<const nnvm::NodePtr,
             std::vector<Attr>,
             std::vector<Attr>>
    GetAttrs(const std::string& attr_name,
             const uint32_t node_id);

  void ProvideShape(const std::vector<nnvm::NodePtr>& nodes,
                    const std::vector<std::vector<mxnet::TShape>> &in_attrs,
                    const std::vector<std::vector<mxnet::TShape>> &out_attrs) {
    aux_nodes_ = nodes;
    aux_in_shapes_ = in_attrs;
    aux_out_shapes_ = out_attrs;
  }

  void ProvideType(const std::vector<nnvm::NodePtr>& nodes,
                   const std::vector<std::vector<int>> &in_attrs,
                   const std::vector<std::vector<int>> &out_attrs) {
    aux_nodes_ = nodes;
    aux_in_types_ = in_attrs;
    aux_out_types_ = out_attrs;
  }

  std::tuple<const nnvm::NodePtr,
             std::vector<mxnet::TShape>,
             std::vector<mxnet::TShape>>
    GetAuxShape(const int node_id) const {
    return std::make_tuple(aux_nodes_[node_id],
                           aux_in_shapes_[node_id],
                           aux_out_shapes_[node_id]);
  }

  std::tuple<const nnvm::NodePtr,
             std::vector<int>,
             std::vector<int>>
    GetAuxType(const int node_id) const {
    return std::make_tuple(aux_nodes_[node_id],
                           aux_in_types_[node_id],
                           aux_out_types_[node_id]);
  }

 private:
  std::string GenerateCode(const std::vector<OpReqType> &req,
                           const std::vector<int> &in_dtypes,
                           const std::vector<int> &out_dtypes,
                           const std::vector<int> &in_ndims,
                           const std::vector<int> &out_ndims,
                           const mxnet::ShapeVector &node_shapes,
                           const std::vector<int> &node_dtypes,
                           const int nvec,
                           const std::string& kernel_name,
                           std::vector<uint32_t> *check_shapes);

  CUfunction CompileCode(const std::string &code,
                         const std::string &kernel_name, int dev_id);

  void CheckShapesAndTypes(const std::vector<TBlob> &inputs,
                           const std::vector<TBlob> &outputs,
                           std::vector<int> *in_dtypes,
                           std::vector<int> *in_ndims,
                           std::vector<int> *out_dtypes,
                           std::vector<int> *out_ndims,
                           int *nvec);

  std::vector<FusedOpEntry> inputs_;
  std::vector<FusedOpEntry> outputs_;

  nnvm::Graph subgraph_;

  template <typename T>
  struct IntermediateAttr {
    std::vector<T> input_attr;
    std::vector<T> output_attr;
    std::vector<T> internal_attr;
  };

  // Shapes and types inside the subgraph
  // copied here, because a subsequent call
  // to InferShape/InferType can overwrite the
  // original information stored in subgraph_
  // attributes while the previous iterations
  // still need them.
  std::vector<IntermediateAttr<mxnet::TShape> > intermediate_shapes_;
  std::vector<IntermediateAttr<int> > intermediate_dtypes_;

  std::vector<nnvm::NodePtr> aux_nodes_;
  std::vector<std::vector<mxnet::TShape>> aux_in_shapes_;
  std::vector<std::vector<mxnet::TShape>> aux_out_shapes_;
  std::vector<std::vector<int>> aux_in_types_;
  std::vector<std::vector<int>> aux_out_types_;
  std::vector<OpReqType> saved_reqs_;
  std::vector<uint32_t> extra_shape_args_;
  std::vector<uint32_t> check_shape_args_;

  CUfunction kernel_functions_[fusion::kNumKernelVariants];
  bool initialized_;
  int kernel_function_dev_id_;

  static std::mutex mutex_;
  std::mutex my_mutex_;
};

using FusedOpPtr = std::shared_ptr<FusedOp>;

struct FusedOpHelperParam {
  FusedOpPtr op;
  uint32_t node_id;

  FusedOpHelperParam(FusedOpPtr op, uint32_t node_id) :
    op(op),
    node_id(node_id) {}
};

using FusedOpHelperParamPtr = std::shared_ptr<FusedOpHelperParam>;

}  // namespace mxnet

#endif  // MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_H_
