/*!
 *  Copyright (c) 2015 by Contributors
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
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"


namespace mxnet {
namespace op {

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

// Fill a rsp NDArray with zeros by updating the aux shape.
template<typename xpu>
void FillZerosRspImpl(mshadow::Stream<xpu> *s, NDArray *dst) {
  if (!dst->storage_initialized()) return;
  // reset the shapes if it's not zeros
  auto storage_shape = dst->storage_shape();
  storage_shape[0] = 0;
  dst->SetAuxShape(rowsparse::kIdx, TShape(mshadow::Shape1(0)));
  dst->SetStorageShape(storage_shape);
}

// Fill a CSR NDArray with zeros by updating the aux shape.
template<typename xpu>
void FillZerosCsrImpl(mshadow::Stream<xpu> *s, NDArray *dst) {
  if (!dst->storage_initialized()) return;
  // reset the shapes if it's not zeros
  TShape new_shape(mshadow::Shape1(0));
  dst->SetAuxShape(csr::kIndPtr, new_shape);
  dst->SetAuxShape(csr::kIdx, new_shape);
  dst->SetStorageShape(new_shape);
}

// This operator never needs to fall back, since there's no input NDArray
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
  CHECK_EQ(inputs.size(), 0);
  auto stype = outputs[0].storage_type();
  if (stype == kRowSparseStorage) {
    NDArray nd(outputs[0]);
    FillZerosRspImpl<xpu>(s, &nd);
  } else if (stype == kCSRStorage) {
    NDArray nd(outputs[0]);
    FillZerosCsrImpl<xpu>(s, &nd);
  } else {
    LOG(FATAL) << "storage type not implemented.";
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
                     mshadow::Shape1(param.repeat *
                                     ceil((param.stop.value() -
                                           param.start) / param.step)));
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INIT_OP_H_
