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
 * \file matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../channel_op_common.h"
#include "../mxnet_op.h"
#include "broadcast_reduce_op.h"

#if MXNET_USE_CUDA
#include <thrust/device_vector.h>
#endif

namespace mxnet {
namespace op {

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape target_shape;
  bool keep_highest;
  nnvm::Tuple<int> shape;
  bool reverse;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(shape)
    .set_default(nnvm::Tuple<int>())
    .describe("The target shape");
    DMLC_DECLARE_FIELD(reverse)
    .set_default(false)
    .describe("If true then the special values are inferred from right to left");
    DMLC_DECLARE_FIELD(target_shape)
    .set_default(TShape())
    .describe("(Deprecated! Use ``shape`` instead.) "
              "Target new shape. One and only one dim can be 0, "
              "in which case it will be inferred from the rest of dims");
    DMLC_DECLARE_FIELD(keep_highest).set_default(false)
    .describe("(Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged."
              "If set to true, then the first dim in target_shape is ignored,"
              "and always fixed as input");
  }
};

inline bool ReshapeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const ReshapeParam& param_ = nnvm::get<ReshapeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  if (param_.shape.ndim() != 0) {
    std::vector<int> dshape_vec;
    std::vector<int> param_shape_vec(param_.shape.begin(), param_.shape.end());
    for (index_t i = 0; i < dshape.ndim(); ++i) {
      dshape_vec.push_back(dshape[i]);
    }
    std::vector<int> tmp;
    size_t src_idx = 0;
    int inf_idx = -1;
    if (param_.reverse) {
      std::reverse(dshape_vec.begin(), dshape_vec.end());
      std::reverse(param_shape_vec.begin(), param_shape_vec.end());
    }
    auto dshape_len = dshape_vec.size();
    auto params_len = param_shape_vec.size();
    for (index_t i = 0; i < params_len; ++i) {
      int proposed_dim = param_shape_vec[i];
      if (proposed_dim == 0) {
        // keep same
        CHECK_LT(src_idx, dshape_len);
        tmp.push_back(dshape_vec[src_idx++]);
      } else if (proposed_dim == -1) {
        // infer
        CHECK_LT(inf_idx, 0) << "One and only one dim can be inferred";
        inf_idx = i;
        tmp.push_back(1);
        src_idx++;
      } else if (proposed_dim == -2) {
        // copy all remaining dims from source
        while (src_idx < dshape_len) {
          size_t dn = dshape_vec[src_idx++];
          tmp.push_back(dn);
        }
      } else if (proposed_dim == -3) {
        // merge two dims from source
        CHECK_LT(src_idx, dshape_len-1);
        size_t d1 = dshape_vec[src_idx++];
        size_t d2 = dshape_vec[src_idx++];
        size_t dn = d1 * d2;
        tmp.push_back(dn);
      } else if (proposed_dim == -4) {
        // split the source dim s into two dims
        // read the left dim and then the right dim (either can be -1)
        CHECK_LT(i + 2, params_len);
        CHECK_LT(src_idx, dshape_len);
        size_t d0 = dshape_vec[src_idx++];
        int d1 = param_shape_vec[++i];
        int d2 = param_shape_vec[++i];
        CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
        if (d1 == -1) d1 = d0 / d2;
        if (d2 == -1) d2 = d0 / d1;
        CHECK_EQ(d1 * d2, static_cast<int>(d0)) <<
          "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
        tmp.push_back(d1);
        tmp.push_back(d2);
      } else {
        // greater than 0, new shape
        tmp.push_back(proposed_dim);
        src_idx++;
      }
    }

    if (inf_idx >= 0) {
      if (dshape.Size() > 0) {
        int new_size = 1;
        for (int x : tmp) new_size *= x;
        tmp[inf_idx] = dshape.Size() / new_size;
      } else {
        tmp[inf_idx] = 0;
      }
    }
    if (param_.reverse) {
      std::reverse(param_shape_vec.begin(), param_shape_vec.end());
      std::reverse(dshape_vec.begin(), dshape_vec.end());
      std::reverse(tmp.begin(), tmp.end());
    }
    TShape oshape(tmp.begin(), tmp.end());
    CHECK_EQ(oshape.Size(), dshape.Size())
      << "Target shape size is different to source. "
      << "Target: " << oshape
      << "\nSource: " << dshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  } else if (param_.target_shape.ndim()) {
    LOG(INFO) << "Using target_shape will be deprecated.";
    TShape oshape = param_.target_shape;
    int neg_count = 0;
    index_t inf_idx = 0;
    index_t start_idx = param_.keep_highest ? 1 : 0;
    if (param_.keep_highest) {
      oshape[0] = dshape[0];
    }
    for (index_t i = start_idx; i < oshape.ndim(); ++i) {
      if (oshape[i] == 0) {
        neg_count++;
        inf_idx = i;
      }
    }
    if (neg_count == 1) {
      oshape[inf_idx] = 1;
      oshape[inf_idx] = dshape.Size() / oshape.Size();
    }

    CHECK(oshape.Size() == dshape.Size())
        << "Target shape size is different to source. "
        << "Target: " << param_.target_shape.Size()
        << "\nSource: " << dshape.Size();
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  } else {
    return (*out_attrs)[0].ndim();
  }
  return true;
}

inline bool FlattenShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(dshape[0], target_dim));
  return true;
}

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;
  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

template<typename xpu>
void TransposeImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const TShape& axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src.type_flag_, ret.type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
    switch (axes.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret.get<xpu, 1, DType>(s);
      Copy(out, in, s);
      break;
     }
     case 2: {
      mshadow::Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> out = ret.FlatTo2D<xpu, DType>(s);
      if (axes[0] == 1 && axes[1] == 0) {
        out = in.T();
      } else {
        Copy(out, in, s);
      }
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret.get<xpu, 3, DType>(s);
      out = transpose(in, axes.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret.get<xpu, 4, DType>(s);
      out = transpose(in, axes.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret.get<xpu, 5, DType>(s);
      out = transpose(in, axes.get<5>());
      break;
     }
     case 6: {
      Tensor<xpu, 6, DType> in = src.get<xpu, 6, DType>(s);
      Tensor<xpu, 6, DType> out = ret.get<xpu, 6, DType>(s);
      out = transpose(in, axes.get<6>());
      break;
     }
     default:
      LOG(FATAL) << "Transpose support at most 6 dimensions";
      break;
    }
  });
}

// matrix transpose
template<typename xpu>
void Transpose(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (param.axes.ndim() == 0) {
    TShape axes = TShape(inputs[0].ndim());
    for (index_t i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  } else {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  }
}

inline bool TransposeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 6U) << "Transpose support at most 6 dimensions";
  TShape ret(shp.ndim());
  if (param.axes.ndim() == 0) {
    for (index_t i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  } else {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (size_t i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[param.axes[i]];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}


struct ExpandDimParam : public dmlc::Parameter<ExpandDimParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(ExpandDimParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("Position where new axis is to be inserted. Suppose that "
              "the input `NDArray`'s dimension is `ndim`, the range of "
              "the inserted axis is `[-ndim, ndim]`");
  }
};


inline bool ExpandDimShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  const ExpandDimParam& param = nnvm::get<ExpandDimParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (in_attrs->at(0).ndim() == 0U && out_attrs->at(0).ndim() == 0U) {
    return false;
  }

  TShape& ishape = (*in_attrs)[0];
  TShape& oshape = (*out_attrs)[0];
  int indim = ishape.ndim();
  bool unknown_ishape = false;
  if (0 == indim) {
    indim = oshape.ndim() - 1;
    unknown_ishape = true;
  }

  int axis = param.axis;
  if (axis < 0) {
    axis += indim;
  }
  CHECK(axis >= 0 && axis <= indim)
      << "axis must be in the range [" << -indim << ", " << indim << "] ("
      << param.axis << " provided)";
  TShape ret(indim + 1);
  for (int i = 0; i < axis; ++i) {
    ret[i] = (unknown_ishape? 0 : ishape[i]);
  }
  ret[axis] = 1;
  for (int i = axis+1; i < indim+1; ++i) {
    ret[i] = (unknown_ishape? 0 : ishape[i-1]);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);

  ret = TShape(indim);
  for (int i = 0; i < axis; ++i) ret[i] = oshape[i];
  for (int i = axis+1; i < indim+1; ++i) ret[i-1] = oshape[i];
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, ret);
  return true;
}

struct SliceParam : public dmlc::Parameter<SliceParam> {
  nnvm::Tuple<dmlc::optional<int> > begin, end;
  DMLC_DECLARE_PARAMETER(SliceParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
  }
};

inline TShape GetSliceShape(const SliceParam& param, const TShape& dshape) {
  CHECK_LE(param.begin.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_LE(param.end.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_EQ(param.begin.ndim(), param.end.ndim())
    << "begin and end must have the same length";

  TShape oshape = dshape;
  for (index_t i = 0; i < param.begin.ndim(); ++i) {
    int s = 0, e = dshape[i];
    if (e != 0) {
      if (param.begin[i]) {
        CHECK_LE(*param.begin[i], e)
          << "Slicing begin exceeds data dimensions "
          << param.begin << " vs " << dshape;
        s = *param.begin[i];
        if (s < 0) s += dshape[i];
      }
      if (param.end[i]) {
        CHECK_LE(*param.end[i], e)
          << "Slicing end exceeds data dimensions "
          << param.end << " vs " << dshape;
        e = *param.end[i];
        if (e < 0) e += dshape[i];
      }
      CHECK(s >= 0 && s < e && e <= static_cast<int>(dshape[i]))
        << "Invalid slicing begin " << param.begin << " and end "
        << param.end << " for data of shape " << dshape;
    }
    oshape[i] = e - s;
  }
  return oshape;
}

inline bool SliceShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const TShape& dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, GetSliceShape(param, dshape));
  return true;
}

// matrix crop for multi dimensional cropping: see also slice
template<typename xpu>
void Slice(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  index_t N = inputs[0].ndim();
  TShape begin(N), end(N);
  for (index_t i = 0; i < N; ++i) {
    int s = 0;
    if (param.begin[i]) {
      s = *param.begin[i];
      if (s < 0) s += inputs[0].size(i);
    }
    begin[i] = s;
    end[i] = s + outputs[0].size(i);
  }

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (inputs[0].ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      out = slice(in, begin.get<1>(), end.get<1>());
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = inputs[0].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
      out = slice(in, begin.get<2>(), end.get<2>());
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = inputs[0].get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
      out = slice(in, begin.get<3>(), end.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = inputs[0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
      out = slice(in, begin.get<4>(), end.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = inputs[0].get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
      out = slice(in, begin.get<5>(), end.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "slice supports at most 5 dimensions";
      break;
    }
  });
}

// slice the indptr of a csr
struct SliceCsrIndPtr {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* out, const IType* in, const IType* base) {
    KERNEL_ASSIGN(out[i], kWriteTo, in[i] - *base);
  }
};

/*
 * a wrapper to launch SliceCsrIndPtr kernel.
 * slice [src[begin] .. src[end]) and store in dst[0, end - begin)
 */
template<typename xpu, typename IType>
void SliceCsrIndPtrImpl(const int begin, const int end, RunContext ctx,
                        const IType* src, IType* dst) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int indptr_len = end - begin + 1;
  Kernel<SliceCsrIndPtr, xpu>::Launch(s, indptr_len, dst, src + begin, src + begin);
}

/*
 * Slice a CSR NDArray
 * Only implemented for CPU
 */
template<typename xpu>
void SliceCsrImpl(const SliceParam &param, const OpContext& ctx,
                  const NDArray &in, OpReqType req, const NDArray &out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  CHECK((std::is_same<xpu, cpu>::value)) << "Slice for CSR input only implemented for CPU";
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo for Slice on CSR input is not supported";
  CHECK_NE(req, kWriteInplace) << "kWriteInplace for Slice on CSR input is not supported";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int begin = *param.begin[0];
  int end = *param.end[0];
  int indptr_len = end - begin + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  if (!in.storage_initialized()) {
    out.set_aux_shape(kIndPtr, Shape1(0));
    return;
  }
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        auto in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        auto out_indptr = out.aux_data(kIndPtr).dptr<RType>();
        SliceCsrIndPtrImpl<cpu, RType>(begin, end, ctx.run_ctx, in_indptr, out_indptr);

        // retrieve nnz (CPU implementation)
        int nnz = out_indptr[indptr_len - 1];
        // copy indices and values
        out.CheckAndAllocAuxData(kIdx, Shape1(nnz));
        out.CheckAndAllocData(Shape1(nnz));
        auto in_idx = in.aux_data(kIdx).dptr<IType>();
        auto out_idx = out.aux_data(kIdx).dptr<IType>();
        auto in_data = in.data().dptr<DType>();
        auto out_data = out.data().dptr<DType>();
        int offset = in_indptr[begin];
        // this is also a CPU-only implementation
        memcpy(out_idx, in_idx + offset, nnz * sizeof(IType));
        memcpy(out_data, in_data + offset, nnz * sizeof(DType));
      });
    });
  });
}

template<typename xpu>
void SliceEx(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<NDArray>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  auto in_stype = inputs[0].storage_type();
  CHECK_NE(in_stype, kDefaultStorage)
           << "SliceEx is not expected to execute for input with default storage type";
  if (in_stype == kCSRStorage) {
    SliceCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
  } else {
    LOG(FATAL) << "Slice not implemented for storage type" << in_stype;
  }
}

inline bool SliceAssignShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const TShape& lshape = (*in_attrs)[0];
  if (lshape.ndim() == 0) return false;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, GetSliceShape(param, lshape));
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, lshape);
  return true;
}

template<typename xpu>
void SliceAssignImpl(mshadow::Stream<xpu> *s, const SliceParam& param,
                     const TBlob& dst, const TBlob& src) {
  using namespace mshadow;
  using namespace mshadow::expr;
  index_t N = dst.ndim();
  TShape begin(N), end(N);
  for (index_t i = 0; i < N; ++i) {
    int s = 0;
    if (param.begin[i]) {
      s = *param.begin[i];
      if (s < 0) s += dst.size(i);
    }
    begin[i] = s;
    end[i] = s + src.size(i);
  }

  MSHADOW_TYPE_SWITCH(dst.type_flag_, DType, {
    switch (dst.ndim()) {
      case 0:
        break;
      case 1: {
        Tensor<xpu, 1, DType> out = dst.get<xpu, 1, DType>(s);
        Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
        slice(out, begin.get<1>(), end.get<1>()) = in;
        break;
      }
      case 2: {
        Tensor<xpu, 2, DType> out = dst.get<xpu, 2, DType>(s);
        Tensor<xpu, 2, DType> in = src.get<xpu, 2, DType>(s);
        slice(out, begin.get<2>(), end.get<2>()) = in;
        break;
      }
      case 3: {
        Tensor<xpu, 3, DType> out = dst.get<xpu, 3, DType>(s);
        Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
        slice(out, begin.get<3>(), end.get<3>()) = in;
        break;
      }
      case 4: {
        Tensor<xpu, 4, DType> out = dst.get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
        slice(out, begin.get<4>(), end.get<4>()) = in;
        break;
      }
      case 5: {
        Tensor<xpu, 5, DType> out = dst.get<xpu, 5, DType>(s);
        Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
        slice(out, begin.get<5>(), end.get<5>()) = in;
        break;
      }
      default:
        LOG(FATAL) << "CropAssign supports at most 5 dimensions";
        break;
    }
  });
}

template<typename xpu>
void SliceAssign(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (req[0] == kNullOp) {
    return;
  } else if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "CropAssign only supports kWriteTo and kWriteInplace";
  }

  SliceAssignImpl<xpu>(s, param, outputs[0], inputs[1]);
}

template<typename xpu>
void SliceBackward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (req[0] == kNullOp) {
    return;
  } else if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      out = DType(0);
    });
  } else {
    LOG(FATAL) << "CropAssign only supports kWriteTo";
  }

  SliceAssignImpl<xpu>(s, param, outputs[0], inputs[0]);
}

struct SimpleCropAssignScalarParam : public dmlc::Parameter<SimpleCropAssignScalarParam> {
  real_t scalar;
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropAssignScalarParam) {
    DMLC_DECLARE_FIELD(scalar)
    .set_default(0)
    .describe("The scalar value for assignment.");
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

template<typename xpu>
void CropAssignScalar(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SimpleCropAssignScalarParam& param = nnvm::get<SimpleCropAssignScalarParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "CropAssignScalar only supports kWriteTo and kWriteInplace";
  }

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (outputs[0].shape_.ndim()) {
      case 0:
        break;
      case 1: {
        Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
        slice(out, param.begin.get<1>(), param.end.get<1>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 2: {
        Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
        slice(out, param.begin.get<2>(), param.end.get<2>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 3: {
        Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
        slice(out, param.begin.get<3>(), param.end.get<3>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 4: {
        Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
        slice(out, param.begin.get<4>(), param.end.get<4>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      case 5: {
        Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
        slice(out, param.begin.get<5>(), param.end.get<5>()) = \
            static_cast<DType>(param.scalar);
        break;
      }
      default:
        LOG(FATAL) << "CropAssign supports at most 5 dimensions";
        break;
    }
  });
}

inline bool CropAssignScalarShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_attrs,
                                  std::vector<TShape> *out_attrs) {
  const SimpleCropAssignScalarParam& param = nnvm::get<SimpleCropAssignScalarParam>(attrs.parsed);
  TShape& lshape = (*in_attrs)[0];
  CHECK_EQ(lshape.ndim(), param.begin.ndim());
  CHECK_EQ(lshape.ndim(), param.end.ndim());
  for (index_t i = 0; i < lshape.ndim(); ++i) {
    CHECK_LT(param.begin[i], param.end[i]);
    CHECK_LE(param.end[i], lshape[i]);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, lshape);
  return true;
}

struct SliceAxisParam : public dmlc::Parameter<SliceAxisParam> {
  int axis;
  int begin;
  dmlc::optional<int> end;
  DMLC_DECLARE_PARAMETER(SliceAxisParam) {
    DMLC_DECLARE_FIELD(axis)
      .describe("Axis along which to be sliced, supports negative indexes.");
    DMLC_DECLARE_FIELD(begin)
      .describe("The beginning index along the axis to be sliced, "
                " supports negative indexes.");
    DMLC_DECLARE_FIELD(end)
      .describe("The ending index along the axis to be sliced, "
                " supports negative indexes.");
  }
};

inline void GetSliceAxisParams(const SliceAxisParam& param, const TShape& ishape,
                           int* axis, int* begin, int* end) {
  *axis = param.axis;
  if (*axis < 0) {
    *axis += static_cast<int>(ishape.ndim());
  }
  CHECK(*axis < static_cast<int>(ishape.ndim()) && *axis >= 0) <<
    "Transformed axis must be smaller than the source ndim and larger than zero! Recieved axis=" <<
    param.axis << ", src_ndim=" << ishape.ndim() << ", transformed axis=" << *axis;
  int axis_size = static_cast<int>(ishape[*axis]);
  *begin = param.begin;
  *end = -1;
  if (*begin < 0) {
    *begin += axis_size;
  }
  if (!static_cast<bool>(param.end)) {
    *end = axis_size;
  } else {
    *end = param.end.value();
    if (*end < 0) {
      *end += axis_size;
    }
  }
  CHECK((*end <= axis_size) && (*end >= 0))
    << "Invalid begin, end, get begin=" << param.begin << ", end=" << param.end;
  CHECK((*begin < *end) && (*begin >= 0))
    << "Invalid begin, end, get begin=" << param.begin << ", end=" << param.end;
}

inline bool SliceAxisShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& ishape = (*in_attrs)[0];
  int axis, begin, end;
  GetSliceAxisParams(param, ishape, &axis, &begin, &end);
  TShape shape(ishape.ndim());
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (static_cast<int>(i) == axis) {
      shape[i] = static_cast<index_t>(end - begin);
    } else {
      shape[i] = ishape[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  return true;
}


template<typename xpu>
void SliceAxis(const nnvm::NodeAttrs& attrs,
           const OpContext& ctx,
           const std::vector<TBlob>& inputs,
           const std::vector<OpReqType>& req,
           const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int axis, begin, end;
  GetSliceAxisParams(param, inputs[0].shape_, &axis, &begin, &end);
  int ndim = static_cast<int>(outputs[0].ndim());

  if (axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> in =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> out =
            outputs[0].FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, begin, end));
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> in =
            inputs[0].FlatTo3D<xpu, DType>(axis, s);
        mshadow::Tensor<xpu, 3, DType> out =
            outputs[0].FlatTo3D<xpu, DType>(axis, s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, begin, end));
      });
  }
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void SliceAxisGrad_(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  using namespace mshadow::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int axis, begin, end;
  GetSliceAxisParams(param, outputs[0].shape_, &axis, &begin, &end);
  int ndim = static_cast<int>(outputs[0].shape_.ndim());

  if (axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> ograd =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> igrad =
            outputs[0].FlatTo2D<xpu, DType>(s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, begin, end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, begin, end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> ograd =
            inputs[0].FlatTo3D<xpu, DType>(axis, s);
        mshadow::Tensor<xpu, 3, DType> igrad =
            outputs[0].FlatTo3D<xpu, DType>(axis, s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, begin, end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, begin, end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  }
}

struct ClipParam : public dmlc::Parameter<ClipParam> {
  real_t a_min, a_max;
  DMLC_DECLARE_PARAMETER(ClipParam) {
    DMLC_DECLARE_FIELD(a_min)
    .describe("Minimum value");
    DMLC_DECLARE_FIELD(a_max)
    .describe("Maximum value");
  }
};


struct clip {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = a_max;
    } else if (data < a_min) {
      out[i] = a_min;
    } else {
      out[i] = data;
    }
  }
};


struct clip_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* grad, const DType* datas,
                                  DType a_min, DType a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = 0;
    } else if (data < a_min) {
      out[i] = 0;
    } else {
      out[i] = grad[i];
    }
  }
};


template<typename xpu>
void Clip(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const ClipParam& param = nnvm::get<ClipParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<clip, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
    inputs[0].dptr<DType>(), DType(param.a_min), DType(param.a_max));
  });
}

template<typename xpu>
void ClipGrad_(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const ClipParam& param = nnvm::get<ClipParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<clip_grad, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
    inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), DType(param.a_min), DType(param.a_max));
  });
}

/*!
 * \brief The parameters of the repeat operator include
 * the number of repeating time and axis (optional).
 * The parameters will be later used to deduce the
 * output ndarray shape in bool RepeatShape() function.
 */
struct RepeatParam : public dmlc::Parameter<RepeatParam> {
  int repeats = 1;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(RepeatParam) {
    DMLC_DECLARE_FIELD(repeats)
      .describe("The number of repetitions for each element.");
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("The axis along which to repeat values."
                " The negative numbers are interpreted counting from the backward."
                " By default, use the flattened input array,"
                " and return a flat output array.");
  }
};

/*!
 * \brief Helper function for getting user input params for the operator repeat.
 * Sanity check the user input values.
 */
inline void GetRepeatParams(const RepeatParam& param, const TShape& ishape,
                            int* repeats, dmlc::optional<int>* axisOpt) {
  *repeats = param.repeats;
  CHECK_GE(*repeats, 0) << "repeats cannot be a negative number";
  *axisOpt = param.axis;
  if (static_cast<bool>(*axisOpt)) {
    int ndims = static_cast<int>(ishape.ndim());
    int axis = axisOpt->value();
    if (axis < 0) {
      axis += ndims;
    }
    CHECK(axis >= 0 && axis < ndims) << "axis = " << axisOpt->value() << " out of bounds";
  }
}

inline bool RepeatOpShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& ishape = (*in_attrs)[0];
  int repeats = 0;
  dmlc::optional<int> axisOpt;
  GetRepeatParams(param, ishape, &repeats, &axisOpt);
  // If 0 repeats, return an empty 0 dim array
  if (0 == repeats) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape());
    return true;
  }

  // If repeats > 0, multiply the size of the corresponding axis by repeats
  if (static_cast<bool>(axisOpt)) {
    int ndims = static_cast<int>(ishape.ndim());
    int axis = axisOpt.value();
    if (axis < 0) {
      axis += ndims;
    }
    TShape shape(ishape.ndim());
    for (index_t i = 0; i < ishape.ndim(); ++i) {
      if (static_cast<int>(i) == axis) {
        shape[i] = static_cast<index_t>(repeats) * ishape[i];
      } else {
        shape[i] = ishape[i];
      }
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  } else {  // If axis is not input by user, return a flat 1D array of size = in.size*repeats
    TShape shape(1);
    shape[0] = ishape.Size() * static_cast<index_t>(repeats);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  }
  return true;
}

inline bool RepeatOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if ((*in_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  } else if ((*out_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  }
  return true;
}

/*!
 * \brief Reshape the input and output tensors for
 * using broadcast_to to achieve the funcitonality
 * of operator repeat.
 * \return a pair of TShape's, first is the reshaped
 * input shape, second is the reshaped output shape.
 */
inline std::pair<TShape, TShape> ReshapeInputOutputForRepeatOp(const TShape& ishape,
                                                               const dmlc::optional<int>& axisOpt,
                                                               const int repeats) {
  if (static_cast<bool>(axisOpt)) {
    int axis = axisOpt.value();
    int ndim = static_cast<int>(ishape.ndim());
    if (axis < 0)  {
      axis += ndim;
    }
    CHECK(axis >= 0 && axis < static_cast<int>(ishape.ndim())) << "Invalid input of axis";

    // reshape the input tensor by adding a dim at the (axis+1)-th dim
    TShape rshape(ishape.ndim()+1);
    // the shape we want to broadcast to
    TShape bshape(rshape.ndim());
    int i = 0;
    while (i <= axis) {
      rshape[i] = bshape[i] = ishape[i];
      ++i;
    }
    rshape[i] = 1;
    bshape[i] = repeats;
    while (i < static_cast<int>(ishape.ndim())) {
      rshape[i+1] = ishape[i];
      bshape[i+1] = ishape[i];
      ++i;
    }
    return std::make_pair(rshape, bshape);
  } else {
    // axis is not input by user
    // reshape the tensor into shape (ishape.Size(), 1)
    // then add one dim at axis = 1 and broadcast to
    // shape (ishape.Size(), repeats)
    TShape rshape(2);
    rshape[0] = ishape.Size();
    rshape[1] = 1;

    TShape bshape(2);
    bshape[0] = rshape[0];
    bshape[1] = repeats;
    return std::make_pair(rshape, bshape);
  }
}

template<typename xpu>
void RepeatOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  const TBlob& iTBlob = inputs[0];
  const TShape& ishape = iTBlob.shape_;
  if (ishape.ndim() == 0) return;

  int repeats = 0;
  dmlc::optional<int> axisOpt;
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  GetRepeatParams(param, ishape, &repeats, &axisOpt);
  if (0 == repeats) return;

  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  std::pair<TShape, TShape> rshapes = ReshapeInputOutputForRepeatOp(ishape, axisOpt, repeats);

  // reshaped input tblob
  TBlob iblob(inputs[0].dptr_, rshapes.first, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

  // reshaped output tblob
  TBlob oblob(outputs[0].dptr_, rshapes.second, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  BroadcastCompute<xpu>(attrs, ctx, newInputs, req, newOutputs);
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to the input of the operator.
 * Backpropagation is employed to implement the
 * chain rule.
 * \param inputs the gradient of the loss function
 * with respect to the outputs of the operator
 * \param outputs the gradient of the loss function
 * with respect to the inputs of the operator
 */
template<typename xpu>
void RepeatOpBackward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const TShape& oshape = outputs[0].shape_;
  if (oshape.ndim() == 0) return;

  int repeats = 0;
  dmlc::optional<int> axisOpt;
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  GetRepeatParams(param, oshape, &repeats, &axisOpt);
  if (0 == repeats) return;

  std::pair<TShape, TShape> rshapes =
    ReshapeInputOutputForRepeatOp(oshape, axisOpt, repeats);

  // reshaped output grad tblob
  TBlob oblob(outputs[0].dptr_, rshapes.first, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  // reshaped input grad tblob
  TBlob iblob(inputs[0].dptr_, rshapes.second, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

  ReduceAxesComputeImpl<xpu, mshadow::red::sum, false>(
      attrs, ctx, newInputs, req, newOutputs, rshapes.first);
}

struct TileParam : public dmlc::Parameter<TileParam> {
  TShape reps;
  DMLC_DECLARE_PARAMETER(TileParam) {
    DMLC_DECLARE_FIELD(reps)
      .describe("The number of times for repeating the tensor a."
                " If reps has length d, the result will have dimension of max(d, a.ndim);"
                " If a.ndim < d, a is promoted to be d-dimensional by prepending new axes."
                " If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.");
  }
};

inline bool TileOpShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TileParam& param = nnvm::get<TileParam>(attrs.parsed);
  const TShape& ishape = (*in_attrs)[0];
  const TShape& reps = param.reps;
  // If reps is empty, return a identical input array
  if (reps.ndim() == 0 || ishape.ndim() == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, ishape);
    return true;
  }
  TShape oshape(std::max(ishape.ndim(), reps.ndim()));
  int i1 = static_cast<int>(ishape.ndim()) - 1;
  int i2 = static_cast<int>(reps.ndim()) - 1;
  for (int i = static_cast<int>(oshape.ndim()) - 1; i >= 0; --i) {
    if (i1 >= 0 && i2 >= 0) {
      oshape[i] = ishape[i1--] * reps[i2--];
    } else if (i1 >= 0) {
      oshape[i] = ishape[i1--];
    } else if (i2 >= 0) {
      oshape[i] = reps[i2--];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return true;
}

inline bool TileOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if ((*in_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  } else if ((*out_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  }
  return true;
}

/*!
 * \brief Reshape the input and output tensors for
 * using broadcast_to to achieve the funcitonality
 * of operator tile.
 * \return a pair of TShape's, first is the reshaped
 * input shape, second is the reshaped output shape.
 */
inline std::pair<TShape, TShape> ReshapeInputOutputForTileOp(const TShape& ishape,
                                                             const TShape& reps) {
  if (ishape.ndim() == 0 || reps.ndim() == 0) {
    return std::make_pair(ishape, ishape);
  }

  // The shape we want to broadcast to
  TShape bshape(std::max(ishape.ndim(), reps.ndim()) * 2);

  // The shape of the input tensor after adding new axes before each dim
  TShape rshape(bshape.ndim());

  int i1 = static_cast<int>(ishape.ndim()) - 1;
  int i2 = static_cast<int>(reps.ndim()) - 1;
  for (int i = static_cast<int>(bshape.ndim()) - 1; i >= 0; --i) {
    if (0 == (i & 1)) {
      bshape[i] = (i2 >= 0? reps[i2--] : 1);
      rshape[i] = 1;
    } else {
      rshape[i] = bshape[i] = (i1 >= 0? ishape[i1--] : 1);
    }
  }

  return std::make_pair(rshape, bshape);
}

/*!
 * \brief Implementation of tiling the input tensor a based
 * on the user-input shape, reps.
 * If a.ndim < reps.ndim, new axes are pre-pended to a. For example,
 * the input tensor has shape (3,), and the reps is (2, 4); the input
 * tensor would be reshaped to (1, 3).
 * If a.ndim > reps.ndim, pre-pending 1's to reps. For example,
 * the input tensor has shape (2, 3, 4, 5), and reps is (2, 2);
 * the reps would be changed to (1, 1, 2, 2).
 * Suppose we have a.ndim = reps.ndim now. To achieve tiling,
 * we utilize the operator broadcast_to. For example, for a tensor
 * of shape (2, 3, 4, 5) and reps (2, 8, 9, 3), we first reshape
 * the tensor to the shape (1, 2, 1, 3, 1, 4, 1, 5) by adding
 * one axis before each dimension. Then, we want to broadcast
 * the new tensor to shape (2, 2, 8, 3, 9, 4, 3, 5). The final
 * output tensor would have shape (2*2, 8*3, 9*4, 3*5).
 */
template<typename xpu>
void TileOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[0].Size() == 0) return;
  const TShape& ishape = inputs[0].shape_;
  const TShape& reps = nnvm::get<TileParam>(attrs.parsed).reps;

  // If any one of the number in reps is zero, return immediately
  for (index_t i = 0; i < reps.ndim(); ++i) {
    if (0 == reps[i]) return;
  }

  std::pair<TShape, TShape> rshapes = ReshapeInputOutputForTileOp(ishape, reps);

  // reshaped input tblob
  TBlob iblob(inputs[0].dptr_, rshapes.first, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};
  // reshaped output tblob
  TBlob oblob(outputs[0].dptr_, rshapes.second, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  BroadcastCompute<xpu>(attrs, ctx, newInputs, req, newOutputs);
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to the input of the operator.
 * Backpropagation is employed to implement the
 * chain rule.
 * \param inputs the gradient of the loss function
 * with respect to the outputs of the operator
 * \param outputs the gradient of the loss function
 * with respect to the inputs of the operator
 */
template<typename xpu>
void TileOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[0].Size() == 0) return;
  const TShape& oshape = outputs[0].shape_;
  const TShape& reps = nnvm::get<TileParam>(attrs.parsed).reps;

  // If any one of the number in reps is zero, return immediately
  for (index_t i = 0; i < reps.ndim(); ++i) {
    if (0 == reps[i]) return;
  }

  std::pair<TShape, TShape> rshapes = ReshapeInputOutputForTileOp(oshape, reps);

  // reshaped output grad tblob
  TBlob oblob(outputs[0].dptr_, rshapes.first, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};
  // reshaped input grad tblob
  TBlob iblob(inputs[0].dptr_, rshapes.second, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

  ReduceAxesComputeImpl<xpu, mshadow::red::sum, false>(
      attrs, ctx, newInputs, req, newOutputs, rshapes.first);
}

struct ReverseParam : public dmlc::Parameter<ReverseParam> {
  nnvm::Tuple<int> axis;
  DMLC_DECLARE_PARAMETER(ReverseParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The axis which to reverse elements.");
  }
};


#define REVERSE_MAX_DIM 10U

struct reverse {
  MSHADOW_XINLINE static int ReverseIndex(index_t idx,
                                          index_t nreversedim,
                                          const index_t * stride_,
                                          const index_t * trailing_) {
    index_t outputIndex = idx;
    for (index_t i = 0; i < nreversedim; ++i) {
      const index_t low = outputIndex % trailing_[i];
      index_t high = outputIndex / trailing_[i];
      const index_t x = high%stride_[i];
      high /= stride_[i];
      outputIndex = (high*stride_[i] + stride_[i] - 1 - x)*trailing_[i] + low;
    }
    return outputIndex;
  }
#ifdef __CUDACC__
  template<typename DType>
  __device__  static void Map(int index, index_t nreversedim, const DType *src, DType *dst,
                              const index_t * stride_,
                              const index_t * trailing_) {
    __shared__ index_t stride_share[REVERSE_MAX_DIM];
    __shared__ index_t trailing_share[REVERSE_MAX_DIM];
    if (threadIdx.x < REVERSE_MAX_DIM) {
      stride_share[threadIdx.x] = stride_[threadIdx.x];
      trailing_share[threadIdx.x] = trailing_[threadIdx.x];
    }
    __syncthreads();
    index_t new_idx = ReverseIndex(index, nreversedim, stride_share, trailing_share);
    dst[new_idx] = src[index];
  }
#else
  template<typename DType>
  MSHADOW_XINLINE  static void Map(int index, index_t nreversedim, const DType *src, DType *dst,
                                   const index_t * stride_,
                                   const index_t * trailing_) {
    index_t new_idx = ReverseIndex(index, nreversedim, stride_, trailing_);
    dst[new_idx] = src[index];
  }
#endif
};


template<typename xpu>
void ReverseOpForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const ReverseParam& param = nnvm::get<ReverseParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  CHECK_LT(param.axis.ndim(), REVERSE_MAX_DIM);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TShape& ishape = inputs[0].shape_;

  std::vector<index_t> stride_(param.axis.ndim());
  std::vector<index_t>  trailing_(param.axis.ndim());
  index_t reverse_index = 0;
  for (auto axis_iter = param.axis.begin() ; axis_iter!= param.axis.end(); ++axis_iter) {
    CHECK_LT(*axis_iter, static_cast<int>(ishape.ndim()));
    stride_[reverse_index] = ishape[*axis_iter];
    trailing_[reverse_index] = 1;
    for (index_t i2 = *axis_iter + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[reverse_index] *= ishape[i2];
    }
    reverse_index++;
  }

#ifdef __CUDACC__
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(reverse_index * sizeof(index_t) * 2), s);

  auto stride_workspace = workspace.dptr_;
  auto trailing_workspace = workspace.dptr_ + reverse_index * sizeof(index_t);

  cudaMemcpyAsync(stride_workspace, thrust::raw_pointer_cast(stride_.data()),
                  stride_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));
  cudaMemcpyAsync(trailing_workspace, thrust::raw_pointer_cast(trailing_.data()),
                  trailing_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));

#endif

#ifdef __CUDACC__
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), reverse_index,
    inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
    reinterpret_cast<index_t*>(stride_workspace), reinterpret_cast<index_t*>(trailing_workspace));
  });
#else
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), reverse_index,
    inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
    stride_.data(), trailing_.data());
  });
#endif
}


struct StackParam : public dmlc::Parameter<StackParam> {
  int axis;
  int num_args;
  DMLC_DECLARE_PARAMETER(StackParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(0)
    .describe("The axis in the result array along which the input arrays are stacked.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be stacked.");
  }
};


inline bool StackOpShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);

  TShape dshape;
  for (const TShape& i : (*in_attrs)) {
    shape_assign(&dshape, i);
  }
  if (dshape.ndim() == 0) return false;

  TShape oshape(dshape.ndim() + 1);
  int axis = CheckAxis(param.axis, oshape.ndim());
  for (int i = 0; i < axis; ++i) {
    oshape[i] = dshape[i];
  }
  oshape[axis] = param.num_args;
  for (index_t i = axis + 1; i < oshape.ndim(); ++i) {
    oshape[i] = dshape[i-1];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return true;
}


template<typename xpu>
void StackOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, outputs[0].ndim());

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    std::vector<Tensor<xpu, 3, DType> > data(inputs.size());
    Tensor<xpu, 3, DType> out;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= outputs[0].shape_[i];
    }
    for (int i = axis + 1; i < outputs[0].ndim(); ++i) {
      trailing *= outputs[0].shape_[i];
    }
    size_t mid = outputs[0].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    out = outputs[0].get_with_shape<xpu, 3, DType>(oshape, s);

    for (index_t i = 0; i < inputs.size(); ++i) {
      Shape<3> dshape = Shape3(leading, 1, trailing);
      data[i] = inputs[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Concatenate(data, &out, 1, req[0]);
  })
}

template<typename xpu>
void StackOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    std::vector<Tensor<xpu, 3, DType> > grad_in(outputs.size());
    Tensor<xpu, 3, DType> grad;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= inputs[0].shape_[i];
    }
    for (int i = axis + 1; i < inputs[0].ndim(); ++i) {
      trailing *= inputs[0].shape_[i];
    }
    size_t mid = inputs[0].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    grad = inputs[0].get_with_shape<xpu, 3, DType>(oshape, s);

    for (index_t i = 0; i < outputs.size(); ++i) {
      Shape<3> dshape = Shape3(leading, 1, trailing);
      grad_in[i] = outputs[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Split(grad, &grad_in, 1, req);
  })
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
