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
 * \file elemwise_binary_scalar_op.h
 * \brief Function definition of elementwise binary scalar operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_

#include <mxnet/operator_util.h>
#include <limits>
#include <vector>
#include <utility>
#include <string>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../../common/alm.h"
#include "elemwise_unary_op.h"

namespace mxnet {
namespace op {

struct NumpyBinaryScalarParam : public dmlc::Parameter<NumpyBinaryScalarParam> {
  double scalar;
  bool is_int;
  DMLC_DECLARE_PARAMETER(NumpyBinaryScalarParam) {
    DMLC_DECLARE_FIELD(scalar).set_default(1).describe("Scalar input value");
    DMLC_DECLARE_FIELD(is_int).set_default(true).describe(
        "Indicate whether scalar input is int type");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream scalar_s, is_int_s;
    scalar_s << std::setprecision(std::numeric_limits<double>::max_digits10) << scalar;
    is_int_s << is_int;
    (*dict)["scalar"] = scalar_s.str();
    (*dict)["is_int"] = is_int_s.str();
  }
};

inline bool NumpyBinaryScalarType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_attrs,
                                  std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
  bool scalar_is_int                  = param.is_int;
  if (common::is_int(in_attrs->at(0)) && !scalar_is_int) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  } else if (in_attrs->at(0) == mshadow::kBool) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, scalar_is_int ? mshadow::kInt64 : mshadow::kFloat64);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }
  return out_attrs->at(0) != -1;
}

class BinaryScalarOp : public UnaryOp {
  /*! \brief Tensor operation against a scalar with a dense result */
  template <typename OP, typename DType, typename IType>
  static void ComputeExDenseResultRsp(mshadow::Stream<cpu>* stream,
                                      const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const NDArray& input,
                                      const OpReqType req,
                                      const NDArray& output) {
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    const double alpha                  = param.scalar;
    CHECK_EQ(output.shape(), input.shape());
    const int64_t row_count                    = output.shape()[0];
    const int64_t items_per_row                = output.shape().Size() / row_count;
    const DType result_for_zero                = OP::Map(DType(0), DType(alpha));
    mshadow::Tensor<cpu, 1, DType> input_data  = input.data().FlatTo1D<cpu, DType>(stream);
    mshadow::Tensor<cpu, 1, DType> output_data = output.data().FlatTo1D<cpu, DType>(stream);
    const int64_t sparse_row_count             = input.aux_shape(rowsparse::kIdx).Size();
    if (sparse_row_count != row_count) {
      mshadow::Tensor<cpu, 1, IType> row_indexes =
          input.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(stream);
      int64_t input_iter   = 0;
      int64_t output_row   = 0;
      IType next_input_row = 0;
      while (output_row < row_count) {
        next_input_row =
            input_iter < sparse_row_count ? int64_t(row_indexes[input_iter]) : row_count;
        // Split up into blocks of contiguous data and do those together

        // Do contiguous dense blocks
        const int64_t dense_block_count = next_input_row - output_row;
        if (dense_block_count > 0) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, cpu>::Launch(
                stream,
                items_per_row * dense_block_count,
                output_data.dptr_ + items_per_row * output_row,
                result_for_zero);
          });
          output_row += dense_block_count;
          continue;
        }

        // Do contiguous sparse blocks
        int64_t next_non_contiguous_sparse = input_iter;
        while (next_non_contiguous_sparse < sparse_row_count - 1) {
          if (row_indexes[next_non_contiguous_sparse + 1] !=
              row_indexes[next_non_contiguous_sparse] + 1) {
            break;
          }
          ++next_non_contiguous_sparse;
        }
        const int64_t sparse_block_count = next_non_contiguous_sparse - input_iter + 1;
        if (sparse_block_count > 0) {
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
                stream,
                items_per_row * sparse_block_count,
                &output_data.dptr_[items_per_row * output_row],
                &input_data.dptr_[items_per_row * input_iter],
                DType(alpha));
          });
          output_row += sparse_block_count;
          input_iter += sparse_block_count;
          continue;
        }
      }
    } else {
      // All rows exist (eventually we don't have to do complex
      // things to call GPU kernels because we don't need to access row indices)
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
            stream, items_per_row * row_count, output_data.dptr_, input_data.dptr_, DType(alpha));
      });
    }
  }

  /*! \brief Tensor operation against a scalar with a dense result */
  template <typename OP, typename DType, typename IType>
  static void ComputeExDenseResultRsp(mshadow::Stream<gpu>* stream,
                                      const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const NDArray& input,
                                      const OpReqType req,
                                      const NDArray& output) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  }

  /*! \brief Tensor operation against a scalar with a dense result */
  template <typename OP, typename DType, typename IType, typename CType>
  static void ComputeExDenseResultCsr(mshadow::Stream<cpu>* stream,
                                      const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const NDArray& input,
                                      const OpReqType req,
                                      const NDArray& output) {
    CHECK_EQ(output.shape(), input.shape());

    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    const double alpha                  = param.scalar;
    const DType dense_fill_val          = OP::Map(DType(0), DType(alpha));
    const TBlob column_indexes          = input.aux_data(csr::kIdx);
    const size_t item_count             = column_indexes.Size();

    // Pre-fill dense with 0-input/output value
    FillDense<DType>(
        stream, output.shape().Size(), dense_fill_val, req, output.data().dptr<DType>());

    mshadow::Tensor<cpu, 2, DType> out = AsRowise2D<DType>(stream, output.data());
    if (item_count) {
      const DType* in                 = input.data().dptr<DType>();
      const IType* column_indexes_ptr = column_indexes.dptr<IType>();

      const auto row_count        = static_cast<size_t>(input.shape()[0]);
      const TBlob row_starts      = input.aux_data(csr::kIndPtr);
      const CType* row_starts_ptr = row_starts.dptr<CType>();

#pragma omp parallel for
      for (int i = 0; i < static_cast<int>(row_count); ++i) {
        const bool last_row = i == static_cast<int>(row_count) - 1;
        // Split up into blocks of contiguous data and do those together
        const size_t row_item_start_iter = row_starts_ptr[i];
        const size_t input_items_this_row =
            !last_row ? static_cast<size_t>(row_starts_ptr[i + 1]) - row_item_start_iter :
                        item_count - row_item_start_iter;
        if (input_items_this_row) {
          const IType* this_row_column_indexes = column_indexes_ptr + row_item_start_iter;
          const DType* row_data_start          = in + row_item_start_iter;
          DType* output_this_row               = out[i].dptr_;
          // More overhead to use OMP for small loops, so don't
          if (input_items_this_row > 1000) {
#pragma omp parallel for
            for (CType j = 0; j < static_cast<CType>(input_items_this_row); ++j) {
              const IType col      = this_row_column_indexes[j];
              const DType val      = row_data_start[j];
              output_this_row[col] = OP::Map(val, DType(alpha));
            }
          } else {
            for (CType j = 0; j < static_cast<CType>(input_items_this_row); ++j) {
              const IType col      = this_row_column_indexes[j];
              const DType val      = row_data_start[j];
              output_this_row[col] = OP::Map(val, DType(alpha));
            }
          }
        }
      }
    }
  }

  /*! \brief Tensor operation against a scalar with a dense result */
  template <typename OP, typename DType, typename IType, typename CType>
  static void ComputeExDenseResultCsr(mshadow::Stream<gpu>* stream,
                                      const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const NDArray& input,
                                      const OpReqType req,
                                      const NDArray& output) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  }

  template <typename xpu, typename OP, typename DType, typename IType>
  static void ComputeExDenseResult(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const NDArray& input,
                                   const OpReqType req,
                                   const NDArray output) {
    mshadow::Stream<xpu>* stream = ctx.get_stream<xpu>();
    CHECK_EQ(output.storage_type(), kDefaultStorage);
    switch (input.storage_type()) {
      case kRowSparseStorage: {
        ComputeExDenseResultRsp<OP, DType, IType>(stream, attrs, ctx, input, req, output);
        break;
      }
      case kCSRStorage: {
        MSHADOW_IDX_TYPE_SWITCH(input.aux_data(csr::kIndPtr).type_flag_, CType, {
          ComputeExDenseResultCsr<OP, DType, IType, CType>(stream, attrs, ctx, input, req, output);
        });
        break;
      }
      default:
        CHECK(false) << "Unsupported sparse storage type";
        break;
    }
  }

 public:
  template <typename OP>
  static void Compute_(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       mshadow::Stream<cpu>* s,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    TBlob temp_tblob;
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    bool scalar_is_int                  = param.is_int;
    const double alpha                  = param.scalar;
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      if ((common::is_int(inputs[0].type_flag_) && !scalar_is_int) ||
          (inputs[0].type_flag_ == kBool)) {
        Tensor<cpu, 1, DType> temp_tensor =
            ctx.requested[0].get_space_typed<cpu, 1, DType>(Shape1(inputs[0].Size()), s);
        temp_tblob = TBlob(temp_tensor);
        CastCompute<cpu>(attrs, ctx, {inputs[0]}, {kWriteTo}, {temp_tblob});
      } else {
        temp_tblob = inputs[0];
      }
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<DType>(), temp_tblob.dptr<DType>(), DType(alpha));
      });
    });
  }

  template <typename xpu, typename OP>
  static void Compute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
    Compute_<OP>(attrs, ctx, s, inputs, req, outputs);
  }

  template <typename xpu, typename OP>
  static void ComputeInt(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu>* s                      = ctx.get_stream<xpu>();
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    const double alpha                  = param.scalar;
    MXNET_INT_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), DType(alpha));
      });
    });
  }

  template <typename xpu, typename OP>
  static void ComputeLogic(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu>* s                      = ctx.get_stream<xpu>();
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    bool scalar_is_int                  = param.is_int;
    const double alpha                  = param.scalar;
    TBlob temp_tblob;
    if (common::is_int(inputs[0].type_flag_) && !scalar_is_int) {
      Tensor<xpu, 1, double> temp_tensor =
          ctx.requested[0].get_space_typed<xpu, 1, double>(Shape1(inputs[0].Size()), s);
      temp_tblob = TBlob(temp_tensor);
      CastCompute<xpu>(attrs, ctx, {inputs[0]}, {kWriteTo}, {temp_tblob});
    } else {
      temp_tblob = inputs[0];
    }
    MSHADOW_TYPE_SWITCH_EXT_WITH_BOOL(temp_tblob.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<bool>(), temp_tblob.dptr<DType>(), DType(alpha));
      });
    });
  }

  template <typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    const auto in_stype  = inputs[0].storage_type();
    const auto out_stype = outputs[0].storage_type();
    if (req[0] == kNullOp) {
      return;
    }
    if ((in_stype == kRowSparseStorage && out_stype == kRowSparseStorage) ||
        (in_stype == kCSRStorage && out_stype == kCSRStorage)) {
      // csr -> csr, or rsp -> rsp
      UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    } else if (out_stype == kDefaultStorage &&
               (in_stype == kRowSparseStorage || in_stype == kCSRStorage)) {
      MSHADOW_TYPE_SWITCH(outputs[0].data().type_flag_, DType, {
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          ComputeExDenseResult<xpu, OP, DType, IType>(attrs, ctx, inputs[0], req[0], outputs[0]);
        });
      });
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }

  template <typename xpu, typename OP>
  static void LogicComputeEx(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    const auto in_stype  = inputs[0].storage_type();
    const auto out_stype = outputs[0].storage_type();
    if (req[0] == kNullOp) {
      return;
    }
    if ((in_stype == kRowSparseStorage && out_stype == kRowSparseStorage) ||
        (in_stype == kCSRStorage && out_stype == kCSRStorage)) {
      // csr -> csr, or rsp -> rsp
      UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }

  template <typename OP>
  static void Backward_(const nnvm::NodeAttrs& attrs,
                        mshadow::Stream<cpu>* s,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    const double alpha                  = param.scalar;
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet::op::mxnet_op::Kernel<
            mxnet::op::mxnet_op::op_with_req<mxnet::op::mxnet_op::backward_grad_tuned<OP>, Req>,
            cpu>::Launch(s,
                         inputs[0].Size(),
                         outputs[0].dptr<DType>(),
                         inputs[0].dptr<DType>(),
                         inputs[1].dptr<DType>(),
                         DType(alpha));
      });
    });
  }

  template <typename xpu, typename OP>
  static void Backward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu>* s = ctx.get_stream<xpu>();
    Backward_<OP>(attrs, s, inputs, req, outputs);
  }
};

#if MXNET_USE_ONEDNN == 1
bool PowerStorageType(const nnvm::NodeAttrs& attrs,
                      const int dev_mask,
                      DispatchMode* dispatch_mode,
                      std::vector<int>* inputs,
                      std::vector<int>* outputs);

void PowerComputeExCPU(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<mxnet::NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<mxnet::NDArray>& outputs);
#endif

#define MXNET_OPERATOR_REGISTER_BINARY_SCALAR(name)                                       \
  NNVM_REGISTER_OP(name)                                                                  \
      .set_num_inputs(1)                                                                  \
      .set_num_outputs(1)                                                                 \
      .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)                               \
      .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                   \
      .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarType)                    \
      .set_attr<mxnet::alm::FChangeLayout>("FChangeLayout", ElemwiseChangeLayout)         \
      .set_attr<nnvm::FInplaceOption>("FInplaceOption",                                   \
                                      [](const NodeAttrs& attrs) {                        \
                                        return std::vector<std::pair<int, int> >{{0, 0}}; \
                                      })                                                  \
      .set_attr<FResourceRequest>(                                                        \
          "FResourceRequest",                                                             \
          [](const NodeAttrs& attrs) {                                                    \
            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};             \
          })                                                                              \
      .add_argument("data", "NDArray-or-Symbol", "source input")                          \
      .add_arguments(NumpyBinaryScalarParam::__FIELDS__())

#if MXNET_USE_CUDA

struct BinaryScalarRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);
};

struct BinaryScalarRTCBackward {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_
