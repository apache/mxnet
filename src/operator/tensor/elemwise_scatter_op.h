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
 * \file elementwise_scatter_op.h
 * \brief Function definition of elementwise scatter operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_

#include <vector>
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"
#include "sparse_retain-inl.h"
#include "cast_storage-inl.h"

namespace mxnet {
namespace op {

/*!
 * \brief Shared helper functions for scatter ops
 */
class ScatterOpBase {
  /*! \brief Protected in order to prevent widespread use. Scatter ops is a special case */
 protected:
  /*!
   * \brief For some situations, we need to do the computation as dense and then use
   * sparse-retain to strip out the portions we aren't interested in.
   * \note If your operastor uses this function, it must request kTempStorage
   * \tparam xpu gpu or cpu
   * \tparam Function Function to call with dense inputs and outputs
   * \param attrs Operator attributes
   * \param ctx Operator context
   * \param inputs Input NDArrays
   * \param req Operation request
   * \param outputs Output NDArrays
   * \param function
   */
  template<typename xpu, typename Function>
  static void ComputeAsDense(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs,
                             Function function) {
    std::vector<bool> output_converted;
    std::vector<TBlob>   input_data, output_data;
    std::vector<NDArray> other_inputs, other_outputs;
    other_inputs.reserve(inputs.size());
    input_data.reserve(inputs.size());
    output_data.reserve(outputs.size());
    other_outputs.reserve(outputs.size());
    output_converted.reserve(outputs.size());
    // Inputs...
    for (const NDArray& nd : inputs) {
      if (nd.storage_type() != kDefaultStorage) {
        NDArray in(nd.shape(), ctx.run_ctx.get_ctx());
        CastStorageComputeEx<xpu>(attrs, ctx, { nd }, req, { in });
        other_inputs.push_back(in);
        input_data.push_back(in.data());
      } else {
        input_data.push_back(nd.data());
      }
    }

    // Outputs...
    for (const NDArray& nd : outputs) {
      if (nd.storage_type() != kDefaultStorage) {
        NDArray out(nd.shape(), ctx.run_ctx.get_ctx());
        CastStorageComputeEx<xpu>(attrs, ctx, { nd }, req, { out });
        other_outputs.push_back(out);
        output_data.push_back(out.data());
        output_converted.push_back(true);
      } else {
        other_outputs.push_back(nd);
        output_data.push_back(nd.data());
        output_converted.push_back(false);
      }
    }

    // Call the function
    function(attrs, ctx, input_data, req, output_data);

    // Convert output(s) back if necessary
    for (size_t i = 0, n = outputs.size(); i < n; ++i) {
      if (output_converted[i]) {
        CastStorageComputeEx<xpu>(attrs,
                                  ctx,
                                  { other_outputs[i] },
                                  req,
                                  { outputs[i] });
      }
    }
  }

  /*!
   * \brief Execute the supplied function/operation, followed by a sparse retain operation
   * of the lhs argument's rows only (row indices)
   * \tparam xpu gpu or cpu
   * \tparam Function Function type call to wrap and return sparse-retained output
   * \param attrs Operator attributes
   * \param ctx Operator context
   * \param inputs Input NDArrays
   * \param req Operation request
   * \param outputs Output NDArrays
   * \param pre_retain Whether to call SparseRetain before calling the given function
   * \param function Function call to wrap and return sparse-retained output
   */
  template <typename xpu, typename Function>
  static void ScatterWrap(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const std::vector<NDArray> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<NDArray> &outputs,
                          bool pre_retain,
                          Function function) {
    CHECK_EQ(outputs.size(), 1U);
    if (inputs[0].storage_type() == kRowSparseStorage
        && outputs[0].storage_type() == kRowSparseStorage) {
      if (pre_retain && inputs[1].storage_type() == kRowSparseStorage) {
        // Retain only rhs rows which have same row as lhs input
        NDArray retained_input(outputs[0].storage_type(), outputs[0].shape(), outputs[0].ctx());
        SparseRetainOpForwardEx<xpu>(attrs, ctx,
                                     { inputs[1], inputs[0].aux_ndarray(rowsparse::kIdx) },
                                     req,
                                     {retained_input});
        CHECK(retained_input.storage_initialized());
        // Perform the operation
        function(attrs, ctx, {inputs[0], retained_input}, req, outputs);
        // Sanity check
        DCHECK_LE(outputs[0].aux_shape(rowsparse::kIdx).Size(),
                  inputs[0].aux_shape(rowsparse::kIdx).Size());
      } else {
        // Perform the operation as usual
        NDArray temp_out(outputs[0].storage_type(), outputs[0].shape(), outputs[0].ctx());
        function(attrs, ctx, inputs, req, { temp_out });
        CHECK(temp_out.storage_initialized());
        CHECK_EQ(temp_out.storage_type(), kRowSparseStorage);
        // Sparse-retain the output based upon lhs-input sparsity
        const NDArray indices(inputs[0].aux_data(rowsparse::kIdx), inputs[0].ctx().dev_id);
        SparseRetainOpForwardEx<xpu>(attrs, ctx, { temp_out, indices },
                                     req, outputs);
        DCHECK_LE(outputs[0].aux_shape(rowsparse::kIdx).Size(),
                  inputs[0].aux_shape(rowsparse::kIdx).Size());
      }
    } else {
      function(attrs, ctx, inputs, req, outputs);
    }
  }
};

/*! \brief Scatter elemwise binary op handlers */
class ElemwiseScatterBinaryOp : public ElemwiseBinaryOp,
                                public ScatterOpBase {
  /*! \brief  CPU version, RspRsp knows how to do an efficient scatter,
   * otherwise retain rhs + normal op */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<cpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    // row_sparse-op-row_sparse or row_sparse-op-default can call RspRsp
    const NDArrayStorageType input0_stype = inputs[0].storage_type();
    const NDArrayStorageType input1_stype = inputs[1].storage_type();
    if (input0_stype == kRowSparseStorage
        && (input1_stype == kRowSparseStorage || input1_stype == kDefaultStorage)
        && outputs[0].storage_type() == kRowSparseStorage) {
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          RspRspOp<DType, IType, OP>(s, attrs, ctx, inputs[0], inputs[1], req[0], outputs[0],
                                     false, true, false, true);
        });
      });
      CHECK_EQ(inputs[0].aux_shape(rowsparse::kIdx).Size(),
               outputs[0].aux_shape(rowsparse::kIdx).Size());
    } else {
      ScatterWrap<cpu>(attrs, ctx, inputs, req,
                       outputs, true, [input0_stype, input1_stype](const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<NDArray> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &outputs) {
          if ((input0_stype == kCSRStorage || input1_stype == kCSRStorage)
              && input0_stype != input1_stype) {
            // Fallback to dense + retain
            ComputeAsDense<cpu>(attrs, ctx, inputs, req,
                                outputs, ElemwiseBinaryOp::Compute<cpu, OP>);
          } else {
            // Normal operation + retain
            ElemwiseBinaryOp::ComputeEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
          }
        });
    }
  }

#ifdef __CUDACC__
  /*! \brief GPU version, fallback op + retain */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<gpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    ScatterWrap<gpu>(attrs, ctx, inputs, req,
                     outputs, false, [](const nnvm::NodeAttrs &attrs,
                                        const OpContext &ctx,
                                        const std::vector<NDArray> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &outputs) {
        ComputeAsDense<gpu>(attrs, ctx, inputs, req, outputs, ElemwiseBinaryOp::Compute<gpu, OP>);
      });
  }
#endif  // #ifdef __CUDACC__

 public:
  /*! \brief General compute for operations which include sparse tensors */
  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
    DCHECK_EQ(inputs.size(), 2U);
    DCHECK_EQ(outputs.size(), 1U);
    ComputeEx_<OP>(ctx.get_stream<xpu>(), attrs, ctx, inputs, req, outputs);
  }
};

/*! \brief Scatter elemwise binary scalar op handlers */
class ElemwiseScatterBinaryScalarOp : public BinaryScalarOp,
                                      public ScatterOpBase {
  /*! \brief  CPU version, retain rhs + normal op */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<cpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    ScatterWrap<cpu>(attrs, ctx, inputs, req,
                     outputs, true, [](const nnvm::NodeAttrs &attrs,
                                        const OpContext &ctx,
                                        const std::vector<NDArray> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &outputs) {
        // Normal operation + retain
        BinaryScalarOp::ComputeEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
    });
  }

#ifdef __CUDACC__
  /*! \brief GPU version, fallback op + retain */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<gpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    if (outputs[0].storage_type() == inputs[0].storage_type()) {
      BinaryScalarOp::ComputeEx<gpu, OP>(attrs, ctx, inputs, req, outputs);
    } else {
      ScatterWrap<cpu>(attrs, ctx, inputs, req,
                       outputs, false, [](const nnvm::NodeAttrs &attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) {
          // Fallback to dense + retain
          ComputeAsDense<gpu>(attrs, ctx, inputs, req, outputs, BinaryScalarOp::Compute<gpu, OP>);
      });
    }
  }
#endif  // __CUDACC__

 public:
  using BinaryScalarOp::Compute;

  /*! \brief General compute for operations which include sparse tensors */
  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
    DCHECK_EQ(inputs.size(), 1U);
    DCHECK_EQ(outputs.size(), 1U);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    if (inputs[0].storage_type() == kRowSparseStorage
        && outputs[0].storage_type() == kRowSparseStorage) {
      UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    } else {
      ComputeEx_<OP>(ctx.get_stream<xpu>(), attrs, ctx, inputs, req, outputs);
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_
