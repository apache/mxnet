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
 * \file np_init_op.h
 * \brief Function definition of numpy init op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_

#include <algorithm>
#include <vector>
#include <string>
#include "../../api/operator/op_utils.h"
#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"
#include "../../api/operator/op_utils.h"

namespace mxnet {
namespace op {

struct NumpyEyeParam : public dmlc::Parameter<NumpyEyeParam> {
  nnvm::dim_t N;
  dmlc::optional<nnvm::dim_t> M;
  nnvm::dim_t k;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(NumpyEyeParam) {
    DMLC_DECLARE_FIELD(N).describe("Number of rows in the output.");
    DMLC_DECLARE_FIELD(M)
        .set_default(dmlc::optional<nnvm::dim_t>())
        .describe("Number of columns in the output. If None, defaults to N.");
    DMLC_DECLARE_FIELD(k).set_default(0).describe(
        "Index of the diagonal. 0 (the default) refers to the main diagonal,"
        "a positive value refers to an upper diagonal."
        "and a negative value to a lower diagonal.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(-1).add_enum("None", -1)
        MXNET_ADD_ALL_TYPES_EXT_WITH_BOOL.describe("Data-type of the returned array.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream N_s, M_s, k_s, dtype_s;
    N_s << N;
    M_s << M;
    k_s << k;
    dtype_s << dtype;
    (*dict)["N"]     = N_s.str();
    (*dict)["M"]     = M_s.str();
    (*dict)["k"]     = k_s.str();
    (*dict)["dtype"] = MXNetTypeWithBool2String(dtype);
  }
};

struct IndicesOpParam : public dmlc::Parameter<IndicesOpParam> {
  mxnet::TShape dimensions;
  int dtype;
  std::string ctx;
  DMLC_DECLARE_PARAMETER(IndicesOpParam) {
    DMLC_DECLARE_FIELD(dimensions).describe("The shape of the grid.");
    DMLC_DECLARE_FIELD(dtype).set_default(-1).add_enum("None", -1)
        MXNET_ADD_ALL_TYPES.describe("Target data type.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        "Only used for imperative calls.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream dimensions_s, dtype_s;
    dimensions_s << dimensions;
    dtype_s << dtype;
    (*dict)["dimensions"] = dimensions_s.str();
    (*dict)["dtype"]      = MXNetTypeWithBool2String(dtype);
  }
};

/*! \brief Initialize and fill output with an arbitrary value */
struct NumpyInitOpWithScalarParam : public dmlc::Parameter<NumpyInitOpWithScalarParam> {
  mxnet::TShape shape;
  std::string ctx;
  int dtype;
  double double_value;
  int64_t int_value;
  uint64_t uint_value;
  int value_type;
  DMLC_DECLARE_PARAMETER(NumpyInitOpWithScalarParam) {
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(-1).add_enum("None", -1)
        MXNET_ADD_ALL_TYPES_EXT_WITH_BOOL.describe("Target data type.");
    DMLC_DECLARE_FIELD(double_value)
        .describe("Float value with which to fill newly created tensor");
    DMLC_DECLARE_FIELD(int_value).describe("Integer value with which to fill newly created tensor");
    DMLC_DECLARE_FIELD(uint_value)
        .describe("Unsigned integer value with which to fill newly created tensor");
    DMLC_DECLARE_FIELD(value_type).describe("Choose the value type");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream shape_s, dtype_s, double_value_s, int_value_s, value_type_s, uint_value_s;
    shape_s << shape;
    dtype_s << dtype;
    double_value_s << double_value;
    int_value_s << int_value;
    uint_value_s << uint_value;
    value_type_s << value_type;
    (*dict)["shape"]        = shape_s.str();
    (*dict)["dtype"]        = MXNetTypeWithBool2String(dtype);
    (*dict)["int_value"]    = int_value_s.str();
    (*dict)["uint_value"]   = uint_value_s.str();
    (*dict)["double_value"] = double_value_s.str();
    (*dict)["value_type"]   = value_type_s.str();
    // We do not set ctx, because ctx has been set in dict instead of InitOpParam.
    // Setting ctx here results in an error.
  }
};

/*! \brief Fill output with an arbitrary value */
template <typename xpu>
void NumpyInitFillWithScalarCompute(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 0);
  CHECK_EQ(outputs.size(), 1U);
  const auto& param = nnvm::get<NumpyInitOpWithScalarParam>(attrs.parsed);
  if (param.value_type == 0) {
    Fill<false>(ctx.get_stream<xpu>(), outputs[0], req[0], param.int_value);
  } else if (param.value_type == 1) {
    Fill<false>(ctx.get_stream<xpu>(), outputs[0], req[0], param.uint_value);
  } else {
    Fill<false>(ctx.get_stream<xpu>(), outputs[0], req[0], param.double_value);
  }
}

struct NumpyLinspaceParam : public dmlc::Parameter<NumpyLinspaceParam> {
  double start_double = 0.0;
  double stop_double  = 0.0;
  int64_t start_int   = 0;
  int64_t stop_int    = 0;
  uint64_t start_uint = 0;
  uint64_t stop_uint  = 0;
  index_t num;
  bool endpoint;
  std::string ctx;
  int dtype;
  int value_type;
  DMLC_DECLARE_PARAMETER(NumpyLinspaceParam) {
    DMLC_DECLARE_FIELD(start_double).describe("The double type starting value of the sequence.");
    DMLC_DECLARE_FIELD(stop_double).describe("The double type ending value of the sequence");
    DMLC_DECLARE_FIELD(start_int).describe("The int type starting value of the sequence.");
    DMLC_DECLARE_FIELD(stop_int).describe("The int type ending value of the sequence");
    DMLC_DECLARE_FIELD(start_uint)
        .describe("The unsigned int type starting value of the sequence.");
    DMLC_DECLARE_FIELD(stop_uint).describe("The unsigned int type ending value of the sequence");
    DMLC_DECLARE_FIELD(num).describe("Number of samples to generate. Must be non-negative.");
    DMLC_DECLARE_FIELD(endpoint).set_default(true).describe(
        "If True, stop is the last sample. Otherwise, it is not included.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(-1).add_enum("None", -1)
        MXNET_ADD_ALL_TYPES_EXT_WITH_BOOL.describe("Target data type.");
    DMLC_DECLARE_FIELD(value_type).set_default(0).describe("Data type for start and stop value");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream start_double_s, stop_double_s, num_s, endpoint_s, dtype_s, start_int_s,
        stop_int_s, start_uint_s, stop_uint_s, value_type_s;
    start_double_s << start_double;
    stop_double_s << stop_double;
    start_int_s << start_int;
    stop_int_s << stop_int;
    start_uint_s << start_uint;
    stop_uint_s << stop_uint;
    value_type_s << value_type;
    num_s << num;
    endpoint_s << endpoint;
    dtype_s << dtype;
    (*dict)["start_double"] = start_double_s.str();
    (*dict)["stop_double"]  = stop_double_s.str();
    (*dict)["start_int"]    = start_int_s.str();
    (*dict)["stop_int"]     = stop_int_s.str();
    (*dict)["start_uint"]   = start_uint_s.str();
    (*dict)["stop_uint"]    = stop_uint_s.str();
    (*dict)["value_type"]   = value_type_s.str();
    (*dict)["num"]          = num_s.str();
    (*dict)["endpoint"]     = endpoint_s.str();
    (*dict)["dtype"]        = MXNetTypeWithBool2String(dtype);
  }
};

inline bool NumpyLinspaceShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_attrs,
                               mxnet::ShapeVector* out_attrs) {
  const NumpyLinspaceParam& param = nnvm::get<NumpyLinspaceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_GE(param.num, 0) << "Number of sequence should be non-negative, received " << param.num;
  mxnet::TShape shape = mxnet::TShape({static_cast<nnvm::dim_t>(param.num)});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  return true;
}

struct numpy_linspace_fwd {
  template <typename DType, typename ValueType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  index_t size,
                                  ValueType start,
                                  ValueType stop,
                                  bool endpoint,
                                  double step,
                                  int req,
                                  DType* out) {
    if (i == 0) {
      // Special cases : start = 9007199254740993
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(start));
    } else {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(start + step * i));
    }
    if (endpoint && i != 0 && i == size - 1) {
      KERNEL_ASSIGN(out[i], req, static_cast<DType>(stop));
    }
  }
};

template <typename xpu>
void NumpyLinspaceCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s                  = ctx.get_stream<xpu>();
  const NumpyLinspaceParam& param = nnvm::get<NumpyLinspaceParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH_EXT_WITH_BOOL(outputs[0].type_flag_, DType, {
    index_t step_num = param.endpoint ? param.num - 1 : param.num;
    if (param.value_type == 0) {
      int64_t start = param.start_int;
      int64_t stop  = param.stop_int;
      double step =
          step_num > 0 ? (static_cast<double>(stop) - static_cast<double>(start)) / step_num : 0.0f;
      Kernel<numpy_linspace_fwd, xpu>::Launch(s,
                                              outputs[0].Size(),
                                              outputs[0].Size(),
                                              start,
                                              stop,
                                              param.endpoint,
                                              step,
                                              req[0],
                                              outputs[0].dptr<DType>());
    } else if (param.value_type == 1) {
      uint64_t start = param.start_uint;
      uint64_t stop  = param.stop_uint;
      double step =
          step_num > 0 ? (static_cast<double>(stop) - static_cast<double>(start)) / step_num : 0.0f;
      Kernel<numpy_linspace_fwd, xpu>::Launch(s,
                                              outputs[0].Size(),
                                              outputs[0].Size(),
                                              start,
                                              stop,
                                              param.endpoint,
                                              step,
                                              req[0],
                                              outputs[0].dptr<DType>());
    } else {
      double start = param.start_double;
      double stop  = param.stop_double;
      double step  = step_num > 0 ? (stop - start) / step_num : 0.0f;
      Kernel<numpy_linspace_fwd, xpu>::Launch(s,
                                              outputs[0].Size(),
                                              outputs[0].Size(),
                                              start,
                                              stop,
                                              param.endpoint,
                                              step,
                                              req[0],
                                              outputs[0].dptr<DType>());
    }
  });
}

inline bool NumpyRangeShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_shapes,
                            mxnet::ShapeVector* out_shapes) {
  const RangeParam& param = nnvm::get<RangeParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK_NE(param.step, 0) << "_npi_arange does not support step=0";
  CHECK_EQ(param.repeat, 1) << "_npi_arange only supports repeat=1, received " << param.repeat;
  CHECK(param.stop.has_value()) << "_npi_arange requires stop to have a value";
  double out_size = std::ceil((param.stop.value() - param.start) / param.step);
  if (out_size < 0) {
    out_size = 0;
  }
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, mxnet::TShape({static_cast<nnvm::dim_t>(out_size)}));
  return true;
}

inline bool NumpyEyeShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  const NumpyEyeParam& param = nnvm::get<NumpyEyeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);
  nnvm::dim_t M = param.M.has_value() ? param.M.value() : param.N;
  CHECK(param.N >= 0) << "negative dimensions are not allowed. N is " << param.N;
  CHECK(M >= 0) << "negative dimensions are not allowed. M is " << M;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(param.N, M));

  return out_attrs->at(0).ndim() != 0U;
}
template <typename xpu>
void NumpyEyeFill(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0)
    return;  // zero-size tensor
  const NumpyEyeParam& param = nnvm::get<NumpyEyeParam>(attrs.parsed);
  const nnvm::dim_t num_cols = param.M.has_value() ? param.M.value() : param.N;
  EyeFillImpl<xpu>(outputs[0], ctx, req, num_cols, param.N, param.k);
}

template <int req>
struct indices_fwd {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out,
                                  const nnvm::dim_t value,
                                  const nnvm::dim_t N,
                                  const nnvm::dim_t dim_i,
                                  const nnvm::dim_t j,
                                  const nnvm::dim_t k,
                                  const nnvm::dim_t t) {
    KERNEL_ASSIGN(out[dim_i * N + N / (t * value) * j + i + k * N / t], req, static_cast<DType>(j));
  }
};

template <int req>
struct identity {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const int n) {
    using namespace mxnet_op;

    const index_t row_id = i / n;
    const index_t col_id = i % n;
    if (row_id == col_id) {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(1));
    } else {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(0));
    }
  }
};

template <typename xpu>
void IndicesCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const IndicesOpParam& param = nnvm::get<IndicesOpParam>(attrs.parsed);
  const TBlob& out_data       = outputs[0];
  mshadow::Stream<xpu>* s     = ctx.get_stream<xpu>();
  dim_t indim                 = param.dimensions.ndim();
  dim_t t                     = 1;
  dim_t N                     = out_data.Size() / indim;
  dim_t value                 = 0;
  if (out_data.Size() == 0)
    return;
  if (req[0] != kNullOp) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        for (int i = 0; i < indim; ++i) {
          value = param.dimensions[i];
          for (int k = 0; k < t; ++k) {
            for (int j = 0; j < param.dimensions[i]; ++j) {
              Kernel<indices_fwd<req_type>, xpu>::Launch(
                  s, N / (param.dimensions[i] * t), out_data.dptr<DType>(), value, N, i, j, k, t);
            }
          }
          t = t * param.dimensions[i];
        }
      });
    });
  }
}

template <typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu>* s        = ctx.get_stream<xpu>();
  const TBlob& out_data = outputs[0];
  int n                 = out_data.shape_[0];
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<identity<req_type>, xpu>::Launch(s, out_data.Size(), out_data.dptr<DType>(), n);
    });
  });
}

struct LogspaceParam : public dmlc::Parameter<LogspaceParam> {
  double start;
  double stop;
  index_t num;
  bool endpoint;
  double base;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(LogspaceParam) {
    DMLC_DECLARE_FIELD(start).describe("The starting value of the sequence.");
    DMLC_DECLARE_FIELD(stop).describe("The ending value of the sequence");
    DMLC_DECLARE_FIELD(num).describe("Number of samples to generate. Must be non-negative.");
    DMLC_DECLARE_FIELD(endpoint).set_default(true).describe(
        "If True, stop is the last sample. Otherwise, it is not included.");
    DMLC_DECLARE_FIELD(base).set_default(10.0).describe(
        "The base of the log space. The step size between the elements in "
        "ln(samples) / ln(base) (or log_base(samples)) is uniform.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype).set_default(-1).add_enum("None", -1)
        MXNET_ADD_ALL_TYPES.describe("Target data type.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream start_s, stop_s, num_s, endpoint_s, base_s, dtype_s;
    start_s << start;
    stop_s << stop;
    num_s << num;
    endpoint_s << endpoint;
    base_s << base;
    (*dict)["start"]    = start_s.str();
    (*dict)["stop"]     = stop_s.str();
    (*dict)["num"]      = num_s.str();
    (*dict)["endpoint"] = endpoint_s.str();
    (*dict)["base"]     = base_s.str();
    (*dict)["dtype"]    = MXNetTypeWithBool2String(dtype);
  }
};

struct logspace_fwd {
  template <typename DType>
  MSHADOW_XINLINE static void
  Map(index_t i, double start, double stop, double base, double step, int req, DType* out) {
    KERNEL_ASSIGN(
        out[i], req, static_cast<DType>(math::pow(base, static_cast<double>(start + step * i))));
  }
};

template <typename xpu>
void LogspaceCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s             = ctx.get_stream<xpu>();
  const LogspaceParam& param = nnvm::get<LogspaceParam>(attrs.parsed);
  if (param.num == 0)
    return;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    index_t step_num = param.endpoint ? param.num - 1 : param.num;
    double step      = step_num > 0 ? (param.stop - param.start) / step_num : 0.0f;
    Kernel<logspace_fwd, xpu>::Launch(s,
                                      outputs[0].Size(),
                                      param.start,
                                      param.stop,
                                      param.base,
                                      step,
                                      req[0],
                                      outputs[0].dptr<DType>());
  });
}

struct AtleastNDParam : dmlc::Parameter<AtleastNDParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(AtleastNDParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1).describe("Number of input arrays.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream num_args_s;
    num_args_s << num_args;
    (*dict)["num_args"] = num_args_s.str();
  }
};

template <typename xpu>
void AtleastNDCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  auto& param = nnvm::get<AtleastNDParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), param.num_args);
  CHECK_EQ(outputs.size(), param.num_args);
  for (int i = 0; i < param.num_args; ++i) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, {inputs[i]}, {req[i]}, {outputs[i]});
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INIT_OP_H_
