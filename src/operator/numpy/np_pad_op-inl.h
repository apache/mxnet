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
 *  Copyright (c) 2019 by Contributors
 * \file np_pad_op-inl.h
 * \brief Function definition of matrix related operators
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_PAD_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_PAD_OP_INL_H_

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include "../tensor/matrix_op-inl.h"
#include "../nn/concat-inl.h"
#include "../../common/utils.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template <int ndim>
MSHADOW_XINLINE index_t rravel(const mshadow::Shape<ndim>& coord,
                               const index_t* shape) {
  index_t ret = 0;
  int nndim = ndim;
  #pragma unroll
  for (int i = 0; i < nndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}

/* Compute coordinates from flattened index given shape */
template<int ndim>
MSHADOW_XINLINE mshadow::Shape<ndim> uunravel(const int idx,
                                              const index_t* shape) {
  mshadow::Shape<ndim> ret;
  #pragma unroll
  for (int i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}

namespace pad_enum {
enum PadOpType { kConstant, kEdge, kReflect, kSymmetric, kMinimum, kMaximum };
}

struct NumpyPadParam : public dmlc::Parameter<NumpyPadParam> {
  mxnet::Tuple<mxnet::Tuple<int>> pad_width;
  int mode;
  double constant_value;
  std::string reflect_type;
  DMLC_DECLARE_PARAMETER(NumpyPadParam) {
    DMLC_DECLARE_FIELD(pad_width)
        .describe("Number of values padded to the edges of each axis. "
                  "((before_1, after_1), … (before_N,"
                  "after_N)) unique pad widths for each axis. ((before, after),) "
                  "yields same before and"
                  "after pad for each axis. "
                  "(pad,) or int is a shortcut for before = after = pad width for all"
                  "axes.");
    DMLC_DECLARE_FIELD(mode)
        .add_enum("constant", pad_enum::kConstant)
        .add_enum("edge", pad_enum::kEdge)
        .add_enum("reflect", pad_enum::kReflect)
        .add_enum("symmetric", pad_enum::kSymmetric)
        .add_enum("maximum", pad_enum::kMaximum)
        .add_enum("minimum", pad_enum::kMinimum)
        .set_default(pad_enum::kConstant)
        .describe(
            "Padding type to use."
            " \"constant\" pads with `constant_value`"
            " \"edge\" pads using the edge values of the input array"
            " \"reflect\" Pads with the reflection of the vector mirrored"
            "on the first and last values of the vector along each axis."
            " \"symmetric\" Pads with the reflection of the vector mirrored"
            "along the edge of the array."
            " \"maximum\" Pads with the maximum value of all or part of the"
            "vector along each axis."
            " \"minimum\" Pads with the minimum value of all or part of the"
            "vector along each axis.");
    DMLC_DECLARE_FIELD(constant_value)
        .set_default(0.0)
        .describe("Used in ‘constant’. The values to set the padded values for each axis."
                  "((before_1, after_1), ... (before_N, after_N)) unique pad constants for"
                  "each axis."
                  "((before, after),) yields same before and after constants for each axis."
                  "(constant,) or constant is a shortcut for before = after = constant for all"
                  "axes."
                  "Default is 0.");
    DMLC_DECLARE_FIELD(reflect_type)
        .set_default("even")
        .describe("Used in ‘reflect’, and ‘symmetric’. "
                  "The ‘even’ style is the default with an unaltered reflection around "
                  "the edge value. For the ‘odd’ style,"
                  "the extended part of the array is created by subtracting the "
                  "reflected values from two times the edge value.");
  }
};

inline mxnet::TShape NumpyPadShapeImpl(const mxnet::TShape& ishape,
                                       const mxnet::Tuple<Tuple<int>> pad_width) {
  if (ishape.ndim() == 1) {
    auto s = ishape[0] + pad_width[0][0] + pad_width[1][0];
    return mxnet::TShape({s});
  } else if (ishape.ndim() >= 2) {
    int i;
    mxnet::TShape oshape(ishape.ndim(), -1);
    for (i = ishape.ndim() - 1; i >=0; i--) {
      int base = ishape[i];
      base = base + pad_width[i][0] + pad_width[i][1];
      oshape[i] = base;
    }
  return oshape;
  }
  return mxnet::TShape({-1, -1});
}

template <typename xpu, int req, int ndim>
struct constant_pad {
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  double constant_value) {
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] && indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        KERNEL_ASSIGN(out[i], req, constant_value);
      }
    }
    if (origin) {
      for (m = 0; m < ndim; m++) {
        indexshape[m] = indexshape[m] - indexwidth[m * 2];
      }
      index_t l = rravel<ndim>(j, ishape);
      KERNEL_ASSIGN(out[i], req, a[l]);
    }
  }
};

template <typename xpu, int req, int ndim>
struct pad_copy {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    // if is origin
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] && indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
      for (m = 0; m < ndim; m++) {
        indexshape[m] = indexshape[m] - indexwidth[m * 2];
      }
      int l = rravel<ndim>(j, ishape);
      KERNEL_ASSIGN(out[i], req, a[l]);
    } else {
      return;
    }
  }
};

template <typename xpu, int req, int ndim>
struct symmetric_pad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  size_t index){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < index; m++) {
      if (indexshape[m] < indexwidth[m * 2] || indexshape[m] >= indexwidth[m * 2] + ishape[m]) {
        // we can not do this now
        return;
      }
    }

    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] && indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
      // this thread is in the origin position, then return
      return;
    }
    if (indexshape[index] < indexwidth[index * 2]) {
    // we need to do the assignment
      int distance = indexwidth[index * 2] - indexshape[index];
      int total = ishape[index];
      // the round of this element
      int round = (distance - 1) / total;
      int position = distance % total;
      if (position == 0) {
        position = ishape[index];
      }
      if (round % 2 == 0) {
        indexshape[index] = indexwidth[index * 2] + position - 1;
      } else {
        indexshape[index] = indexwidth[index * 2] + ishape[index] - 1 - (position - 1);
      }
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
    } else if (indexshape[index] >= (indexwidth[index * 2]+ishape[index])) {
      int distance = (indexshape[index]+1) - (indexwidth[index * 2]+ishape[index]);
      int total = ishape[index];
      int position = distance % total;
      int round = (distance - 1) / total;
      if (position == 0) {
        position = ishape[index];
      }
      if (round % 2 == 0) {
        indexshape[index] = indexwidth[index * 2] + ishape[index] - 1 - (position - 1);
      } else {
        indexshape[index] = indexwidth[index * 2] + position - 1;
      }
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
    }
  }
};

template <typename xpu, int req, int ndim>
struct edge_pad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  size_t index){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < index; m++) {
      if (indexshape[m] < indexwidth[m * 2] ||
          indexshape[m] >= indexwidth[m * 2] + ishape[m]) {
      // we can not do this now, since this is a former axis
        return;
      }
    }
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] &&
          indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
    // this thread is in the origin position, then return
      return;
    }
    if (indexshape[index] < indexwidth[index * 2]) {
    // we need to do the assignment
      indexshape[index] = indexwidth[index * 2];
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
    } else if (indexshape[index] >= (indexwidth[index * 2]+ishape[index])) {
      indexshape[index] = indexwidth[index * 2] + ishape[index] - 1;
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
    }
  }
};

template <typename xpu, int req, int ndim>
struct reflect_pad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  size_t index){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < index; m++) {
      if (indexshape[m] < indexwidth[m * 2] ||
          indexshape[m] >= indexwidth[m * 2] + ishape[m]) {
        // we can not do this now
        return;
      }
    }
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] &&
          indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
      // this thread is in the origin position, then return
      return;
    }
    if (indexshape[index] < indexwidth[index * 2]) {
      // we need to do the assignment
      int distance = indexwidth[index * 2] - indexshape[index];
      int total = ishape[index];
      if (total == 1) {
        indexshape[index] = indexwidth[index * 2];
        int l = rravel<ndim>(j, oshape);
        KERNEL_ASSIGN(out[i], req, out[l]);
        return;
      }
      int round = (distance - 1) / (total - 1);
      if (round % 2 == 0) {
        int position = (distance + round) % total;
        indexshape[index] = indexwidth[index * 2] + position;
      } else {
        int position = (distance + round) % total;
        indexshape[index] = indexwidth[index * 2] + ishape[index] - 1 - (position);
      }
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
    } else if (indexshape[index] >= (indexwidth[index * 2] + ishape[index])) {
      int distance = (indexshape[index]+1) - (indexwidth[index * 2] + ishape[index]);
      int total = ishape[index];
      if (total == 1) {
        indexshape[index] = indexwidth[index * 2];
        int l = rravel<ndim>(j, oshape);
        KERNEL_ASSIGN(out[i], req, out[l]);
        return;
      }
      int round = (distance - 1) / (total - 1);
      if (round % 2 == 0) {
        int position = (distance + round) % total;
        indexshape[index] = indexwidth[index * 2] + ishape[index] - 1 - (position);
      } else {
        int position = (distance + round) % total;
        indexshape[index] = indexwidth[index * 2] + position;
      }
      int l = rravel<ndim>(j, oshape);
      KERNEL_ASSIGN(out[i], req, out[l]);
  }
  }
};

template <typename xpu, int req, int ndim>
struct max_pad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  size_t index){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < index; m++) {
      if (indexshape[m] < indexwidth[m * 2] ||
          indexshape[m] >= indexwidth[m * 2] + ishape[m]) {
        // we can not do this now
        return;
      }
    }
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] &&
          indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
      // this thread is in the origin position, then return
      return;
    }

    if (indexshape[index] < indexwidth[index * 2] ||
        indexshape[index] >= indexwidth[index * 2] + ishape[index]) {
      indexshape[index] = indexwidth[index * 2];
      int l = rravel<ndim>(j, oshape);
      int max_count = 0;
      auto max_value = out[l];
      for (max_count = 0; max_count < ishape[index]; max_count++) {
        indexshape[index] = indexwidth[index * 2] + max_count;
        l = rravel<ndim>(j, oshape);
        if (out[l] > max_value) {
            max_value = out[l];
        }
      }
      KERNEL_ASSIGN(out[i], req, max_value);
    }
  }
};

template <typename xpu, int req, int ndim>
struct min_pad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a,
                                  const index_t* ishape,
                                  const index_t* oshape,
                                  mshadow::Shape<ndim*2> width,
                                  size_t index){
    using namespace mxnet_op;
    auto j = uunravel<ndim>(i, oshape);
    size_t m;
    bool origin = true;
    index_t* indexwidth = width.shape_;
    index_t* indexshape = j.shape_;
    for (m = 0; m < index; m++) {
      if (indexshape[m] < indexwidth[m * 2] ||
          indexshape[m] >= indexwidth[m * 2] + ishape[m]) {
        // we can not do this now
        return;
      }
    }
    for (m = 0; m < ndim; m++) {
      if (indexshape[m] >= indexwidth[m * 2] &&
          indexshape[m] < indexwidth[m * 2] + ishape[m]) {
        continue;
      } else {
        origin = false;
        break;
      }
    }
    if (origin) {
      // this thread is in the origin position, then return
      return;
    }
    if (indexshape[index] < indexwidth[index * 2] ||
        indexshape[index] >= (indexwidth[index * 2] + ishape[index])) {
      indexshape[index] = indexwidth[index * 2];
      int l = rravel<ndim>(j, oshape);
      int min_count = 0;
      auto min_value = out[l];
      for (min_count = 0; min_count < ishape[index]; min_count++) {
        indexshape[index] = indexwidth[index * 2] + min_count;
        l = rravel<ndim>(j, oshape);
        if (out[l] < min_value) {
            min_value = out[l];
        }
      }
      j = uunravel<ndim>(i, oshape);
      KERNEL_ASSIGN(out[i], req, min_value);
    } else {
      return;
    }
  }
};


template <typename xpu, int req>
struct pad_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *a){
    using namespace mxnet_op;
    KERNEL_ASSIGN(out[i], req, 1);
  }
};

template<typename xpu>
void NumpyPadOpImpl(const TBlob& in_data,
                    const TBlob& out_data,
                    index_t* ishape,
                    index_t* oshape,
                    index_t dsize,
                    const NumpyPadParam& param,
                    const std::vector<OpReqType>& req,
                    mxnet_op::Stream<xpu> *s) {
  using namespace mxnet_op;
  using namespace mshadow;
  int mode = param.mode;
  int ndim = in_data.ndim();
  MXNET_NDIM_SWITCH(ndim, NDim, {
    mshadow::Shape<NDim*2> width;
    int dimcounter = 0;
    index_t* odptr = reinterpret_cast<index_t*>(oshape);
    if (ndim == 1) {
      width[0] = param.pad_width[0][0];
      width[1] = param.pad_width[1][0];
    } else {
      for (dimcounter = 0; dimcounter < NDim; dimcounter++) {
        width[dimcounter*2] = param.pad_width[dimcounter][0];
        width[dimcounter*2 + 1] = param.pad_width[dimcounter][1];
      }
    }
    index_t* idptr = reinterpret_cast<index_t*>(ishape);
    switch (mode) {
      case pad_enum::kConstant:
        {
          // constant padding start
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<constant_pad<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width, param.constant_value);
            });
          });
          // constant padding end
          break;
        }
      case pad_enum::kSymmetric:
        {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<pad_copy<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width);
            });
          });
          index_t index;
          index_t dim = ndim;
          // symmetric padding start
          for (index = dim-1; index >= 0; index--) {
            MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
              MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                Kernel<symmetric_pad<xpu, req_type, NDim>, xpu>::Launch(
                  s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                  idptr, odptr, width, index);
              });
            });
          }
          // symmetric padding end
          break;
        }
      case pad_enum::kReflect:
        {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<pad_copy<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width);
            });
          });
          index_t index;
          index_t dim = ndim;
          // reflect padding start
          for (index = dim-1; index >= 0; index--) {
            MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
              MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                Kernel<reflect_pad<xpu, req_type, NDim>, xpu>::Launch(
                  s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                  idptr, odptr, width, index);
              });
            });
          }
          // reflect padding end
          break;
        }
      case pad_enum::kEdge:
        {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<pad_copy<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width);
            });
          });
          index_t index;
          index_t dim = ndim;
          // edge padding start
          for (index = dim-1; index >= 0; index--) {
            MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
              MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                Kernel<edge_pad<xpu, req_type, NDim>, xpu>::Launch(
                  s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                  idptr, odptr, width, index);
              });
            });
          }
          // edge padding end
          break;
        }
      case pad_enum::kMinimum:
        {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<pad_copy<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width);
            });
          });
          index_t index;
          index_t dim = ndim;
          // minimum padding start
          for (index = dim-1; index >= 0; index--) {
            MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
              MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                Kernel<min_pad<xpu, req_type, NDim>, xpu>::Launch(
                  s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                  idptr, odptr, width, index);
              });
            });
          }
          // minimum padding end
          break;
        }
      case pad_enum::kMaximum:
        {
          MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
            MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
              Kernel<pad_copy<xpu, req_type, NDim>, xpu>::Launch(
                s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                idptr, odptr, width);
            });
          });
          index_t index;
          index_t dim = ndim;
          // maximum padding start
          for (index = dim-1; index >= 0; index--) {
            MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
              MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                Kernel<max_pad<xpu, req_type, NDim>, xpu>::Launch(
                  s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>(),
                  idptr, odptr, width, index);
              });
            });
          }
          // maximum padding end
          break;
        }
      default:
        LOG(FATAL) << "Other modes are not supported. ";
    }
  })
}

template<typename xpu>
void NumpyPadOpBackImpl(const TBlob& in_data,
                        const TBlob& out_data,
                        index_t dsize,
                        const std::vector<OpReqType>& req,
                        mxnet_op::Stream<xpu> *s) {
  using namespace mxnet_op;
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<pad_grad<xpu, req_type>, xpu>::Launch(
        s, dsize, out_data.dptr<DType>(), in_data.dptr<DType>());
    });
  });
}


template<typename xpu>
void NumpyPadOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  MXNET_NDIM_SWITCH(inputs[0].ndim(), NDim, {
    using namespace mxnet_op;
    using namespace mshadow;
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[0], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    size_t ts = in_data.ndim();
    size_t count;
    mshadow::Shape<NDim> inshape;
    for (count = 0; count < ts; count++) {
      inshape[count] = static_cast<index_t>((in_data.shape_)[count]);
    }

    Tensor<xpu, 1, index_t> tsp = ctx.requested[0].
                                  get_space_typed<xpu, 1, index_t>(Shape1(2*ts), s);
    Tensor<cpu, 1, index_t> ta(reinterpret_cast<index_t*>(inshape.shape_),
                               Shape1(ts), ctx.get_stream<cpu>());
    Tensor<xpu, 1, index_t> ti(reinterpret_cast<index_t*>(tsp.dptr_),
                               Shape1(ts), ctx.get_stream<xpu>());
    mshadow::Copy(ti, ta, ctx.get_stream<xpu>());

    mshadow::Shape<NDim> outshape;
    for (count = 0; count < ts; count++) {
      outshape[count] = static_cast<index_t>((out_data.shape_)[count]);
    }
    index_t* wcp = tsp.dptr_;
    wcp += ts;
    Tensor<cpu, 1, index_t> tb(reinterpret_cast<index_t*>(outshape.shape_),
                               Shape1(ts), ctx.get_stream<cpu>());
    Tensor<xpu, 1, index_t> to(reinterpret_cast<index_t*>(wcp), Shape1(ts),
                               ctx.get_stream<xpu>());
    mshadow::Copy(to, tb, ctx.get_stream<xpu>());
    const NumpyPadParam& param = nnvm::get<NumpyPadParam>(attrs.parsed);

    index_t* wt = reinterpret_cast<index_t*>(to.dptr_);
    index_t* wi = reinterpret_cast<index_t*>(ti.dptr_);

    NumpyPadOpImpl<xpu>(in_data, out_data, wi,
                        wt, out_data.Size(), param, req, s);
  })
}

template<typename xpu>
void NumpyPadOpBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  NumpyPadOpBackImpl<xpu>(in_data, out_data,
                          out_data.Size(), req, s);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_PAD_OP_INL_H_
