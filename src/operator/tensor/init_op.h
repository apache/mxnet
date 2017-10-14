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
 * \file init_op.h
 * \brief Function definition of initialization op
 */
#ifndef MXNET_OPERATOR_TENSOR_INIT_OP_H_
#define MXNET_OPERATOR_TENSOR_INIT_OP_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/parameter.h>
#include <dmlc/optional.h>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"


namespace mxnet {
namespace op {

struct EyeParam : public dmlc::Parameter<EyeParam> {
  nnvm::dim_t N;
  nnvm::dim_t M;
  nnvm::dim_t k;
  std::string ctx;
  int dtype;

  DMLC_DECLARE_PARAMETER(EyeParam) {
    DMLC_DECLARE_FIELD(N)
    .describe("Number of rows in the output.");
    DMLC_DECLARE_FIELD(M)
    .set_default(0)
    .describe("Number of columns in the output. If 0, defaults to N");
    DMLC_DECLARE_FIELD(k)
    .set_default(0)
    .describe("Index of the diagonal. 0 (the default) refers to the main diagonal."
              "A positive value refers to an upper diagonal."
              "A negative value to a lower diagonal.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int8", mshadow::kInt8)
    .add_enum("int64", mshadow::kInt64)
    .describe("Target data type.");
  }
};

template<typename ParamType>
inline bool InitEyeShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(param.N, param.M > 0 ? param.M : param.N));
  return true;
}

template<int req>
struct eye_dns_fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const nnvm::dim_t num_cols,
                                  const nnvm::dim_t k) {
    if ((i % num_cols) == ((i / num_cols) + k)) {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(1));
    } else {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(0));
    }
  }
};

template<typename xpu>
void EyeFill(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const EyeParam& param = nnvm::get<EyeParam>(attrs.parsed);
  const TBlob& out_data = outputs[0];
  const nnvm::dim_t num_cols = param.M > 0 ? param.M : param.N;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<eye_dns_fill<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(),
          num_cols, param.k);
    });
  });
}

struct eye_csr_indptr_fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const nnvm::dim_t nnz,
                                  const nnvm::dim_t init_row) {
    const nnvm::dim_t data = (i - init_row) > 0 ? (i - init_row) : 0;
    out_data[i] = static_cast<DType>(data <= nnz ? data : nnz);
  }
};

struct eye_csr_idx_fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const nnvm::dim_t init_col) {
    out_data[i] = static_cast<DType>(i + init_col);
  }
};

struct eye_csr_data_fill {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data) {
    out_data[i] = static_cast<DType>(1);
  }
};

template<typename xpu>
void EyeFillCsrImpl(mshadow::Stream<xpu> *s, const NDArray& out,
                    const nnvm::dim_t N, const nnvm::dim_t M, const nnvm::dim_t k) {
  const nnvm::dim_t num_cols = M > 0 ? M : N;
  const nnvm::dim_t cnnz = std::max(num_cols - std::abs(k), (nnvm::dim_t)0);
  const nnvm::dim_t rnnz = std::max(N - std::abs(k), (nnvm::dim_t)0);
  const nnvm::dim_t nnz = k > 0 ? std::min(cnnz, N) :
                                  std::min(rnnz, num_cols);
  using namespace mxnet_op;
  out.CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(N + 1));
  out.CheckAndAllocAuxData(csr::kIdx, mshadow::Shape1(nnz));
  out.CheckAndAllocData(mshadow::Shape1(nnz));

  MSHADOW_IDX_TYPE_SWITCH(out.aux_type(csr::kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(out.aux_type(csr::kIdx), IType, {
      MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
        Kernel<eye_csr_indptr_fill, xpu>::Launch(
          s, N + 1, out.aux_data(csr::kIndPtr).dptr<RType>(),
          nnz, std::abs(std::min((nnvm::dim_t)0, k)));
        Kernel<eye_csr_idx_fill, xpu>::Launch(
          s, nnz, out.aux_data(csr::kIdx).dptr<IType>(),
          std::max(std::min(k, num_cols-1), (nnvm::dim_t)0));
        Kernel<eye_csr_data_fill, xpu>::Launch(
          s, nnz, out.data().dptr<DType>());
      });
    });
  });
}

template<typename xpu>
void EyeFillEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs.size(), 1);
  auto stype = outputs[0].storage_type();
  const EyeParam& param = nnvm::get<EyeParam>(attrs.parsed);
  if (req[0] == kNullOp) return;
  CHECK_EQ(req[0], kWriteTo) << "kWriteTo is expected for EyeFillEx";
  if (stype == kCSRStorage) {
    EyeFillCsrImpl<xpu>(s, outputs[0], param.N, param.M, param.k);
  } else {
    LOG(FATAL) << "Not implemented: " << operator_string(attrs, ctx, inputs, req, outputs);
  }
}

struct InitOpParam : public dmlc::Parameter<InitOpParam> {
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(InitOpParam) {
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Target data type.");
  }
};

struct RangeParam : public dmlc::Parameter<RangeParam> {
  real_t start;
  dmlc::optional<real_t> stop;
  real_t step;
  int repeat;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(RangeParam) {
    DMLC_DECLARE_FIELD(start)
    .describe("Start of interval. The interval includes this value. The default start value is 0.");
    DMLC_DECLARE_FIELD(stop)
    .set_default(dmlc::optional<real_t>())
    .describe("End of interval. The interval does not include this value,"
              " except in some cases where step is not an integer and"
              " floating point round-off affects the length of out.");
    DMLC_DECLARE_FIELD(step)
    .set_default(1)
    .describe("Spacing between values.");
    DMLC_DECLARE_FIELD(repeat)
    .set_default(1)
    .describe("The repeating time of all elements."
              " E.g repeat=3, the element a will be repeated three times --> a, a, a.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Target data type.");
  }
};

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
inline void RangeParamParser(nnvm::NodeAttrs* attrs) {
  RangeParam param;
  param.Init(attrs->dict);
  if (!static_cast<bool>(param.stop)) {
    param.stop = param.start;
    param.start = 0;
  }
  attrs->parsed = std::move(param);
}

template<typename ParamType>
inline bool InitShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*out_attrs)[0].ndim() != 0 && param.shape.ndim() == 0) return true;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, param.shape);
  return true;
}

template<typename ParamType>
inline bool InitType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return true;
}

template<typename ParamType, bool rsp, bool csr>
inline bool InitStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto &out_stype = out_attrs->at(0);
  bool dispatched = false;
  type_assign(&out_stype, kDefaultStorage);
  if (!dispatched && out_stype == kDefaultStorage) {
    // default
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && rsp && out_stype == kRowSparseStorage) {
    // rsp
    dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched && csr && out_stype == kCSRStorage) {
    // csr
    dispatched = storage_type_assign(out_attrs, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatch_fallback(out_attrs, dispatch_mode);
    LogStorageFallback(attrs, dev_mask, in_attrs, out_attrs);
  }
  return true;
}

template<typename xpu, int value>
void FillCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], scalar<DType>(value));
  });
}

// Fill in the indices and values of a RowSparse NDArray to represent a zeros NDArray,
// instead of the usual compact representation.
template<typename xpu>
inline void FillDnsZerosRspImpl(mshadow::Stream<xpu> *s, NDArray *dst) {
  using namespace rowsparse;
  using namespace mshadow::expr;
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(dst->storage_type(), kRowSparseStorage);
  MSHADOW_REAL_TYPE_SWITCH(dst->dtype(), DType, {
    MSHADOW_IDX_TYPE_SWITCH(dst->aux_type(kIdx), IType, {
      auto num_rows = dst->shape()[0];
      dst->CheckAndAlloc({Shape1(num_rows)});
      auto idx = dst->aux_data(kIdx).FlatTo1D<xpu, IType>(s);
      auto val = dst->data();
      Kernel<set_zero, xpu>::Launch(s, val.Size(), val.dptr<DType>());
      ASSIGN_DISPATCH(idx, kWriteTo, range<IType>(0, num_rows, 1, 1));
    });
  });
}

struct PopulateFullIdxRspKernel {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* out) {
    KERNEL_ASSIGN(out[i], kWriteTo, i);
  }
};

// Fill full indices NDArray with zeros by updating the aux shape.
template<typename xpu>
void PopulateFullIdxRspImpl(mshadow::Stream<xpu> *s, NDArray *dst) {
  using namespace rowsparse;
  CHECK_EQ(dst->storage_type(), kRowSparseStorage);
  nnvm::dim_t nnr = dst->shape()[0];
  dst->CheckAndAllocAuxData(kIdx, mshadow::Shape1(nnr));
  MSHADOW_IDX_TYPE_SWITCH(dst->aux_type(kIdx), IType, {
    IType* idx = dst->aux_data(kIdx).dptr<IType>();
    mxnet_op::Kernel<PopulateFullIdxRspKernel, xpu>::Launch(s, nnr, idx);
  });
}

/*!
 * \brief Fill a rsp NDArray with zeros by updating the aux shape.
 * \tparam xpu - cpu or gpu
 * \param s - The device stream
 * \param dst - NDArray which is to be set to "all zeroes"
 */
template<typename xpu>
void FillZerosRspImpl(mshadow::Stream<xpu> *, const NDArray& dst) {
  if (dst.storage_initialized()) {
    // reset the shapes if it's not zeros (set_aux_shape() will set storage_shape to zero as well)
    dst.set_aux_shape(rowsparse::kIdx, TShape(mshadow::Shape1(0)));
  }
}

/*!
 * \brief Fill a CSR NDArray with zeros by updating the aux shape
 * \param s - The device stream
 * \param dst - NDArray which is to be set to "all zeroes"
 */
inline void FillZerosCsrImpl(mshadow::Stream<mshadow::cpu> *s, const NDArray& dst) {
  dst.set_aux_shape(csr::kIdx, mshadow::Shape1(0));
  dst.CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(dst.shape()[0] + 1));
  TBlob indptr_data = dst.aux_data(csr::kIndPtr);
  MSHADOW_IDX_TYPE_SWITCH(dst.aux_type(csr::kIndPtr), IType, {
    mxnet_op::Kernel<mxnet_op::set_zero, mshadow::cpu>::Launch(
      s, indptr_data.Size(), indptr_data.dptr<IType>());
  });
}
void FillZerosCsrImpl(mshadow::Stream<mshadow::gpu> *s, const NDArray& dst);

/*!
 * \brief Fill an NDArray with zeros
 * \tparam xpu - cpu or gpu
 * \param attrs  - node attributes (unused)
 * \param ctx - Device context
 * \param inputs - NDArray inputs (unused)
 * \param req - Request type (i.e. kWrite, kNullOp, etc.)
 * \param outputs - Array which contains at position zero (0) the array to be set to zeros
 */
template<typename xpu>
void FillComputeZerosEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs.size(), 1);
  auto stype = outputs[0].storage_type();
  if (req[0] == kNullOp) return;
  CHECK_EQ(req[0], kWriteTo) << "kWriteTo is expected for FillComputeZerosEx";
  if (stype == kRowSparseStorage) {
    FillZerosRspImpl(s, outputs[0]);
  } else if (stype == kCSRStorage) {
    FillZerosCsrImpl(s, outputs[0]);
  } else {
    LOG(FATAL) << "Not implemented: " << operator_string(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void RangeCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const RangeParam& param = nnvm::get<RangeParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], range<DType>(param.start,
                                              param.stop.value(),
                                              param.step,
                                              param.repeat));
  });
}


inline bool RangeShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const RangeParam& param = nnvm::get<RangeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE(param.step, 0U)
    << "Range does not support step=0, received " << param.step;
  CHECK(param.repeat > 0)
    << "Range only supports repeat > 0, received " << param.repeat;
  if (param.step > 0) {
    CHECK(param.start < param.stop.value())
      << "Range does not support (start, stop, step) = "
      << "(" << param.start << "," << param.stop.value() << "," << param.step << ")";
  } else {
    CHECK(param.start > param.stop.value())
      << "Range does not support (start, stop, step)= "
      << "(" << param.start << "," << param.stop.value() << "," << param.step << ")";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     mshadow::Shape1(mshadow::expr::RangeOutSize(param.start,
                                                                 param.stop.value(),
                                                                 param.step,
                                                                 param.repeat)));
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INIT_OP_H_
