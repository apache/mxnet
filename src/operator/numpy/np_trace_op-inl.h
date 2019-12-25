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
 * \file np_trace_op-inl.h
 * \brief Function definition of matrix numpy-compatible trace operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_TRACE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TRACE_OP_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyTraceParam: public dmlc::Parameter<NumpyTraceParam> {
  int offset, axis1, axis2;
  DMLC_DECLARE_PARAMETER(NumpyTraceParam) {
    DMLC_DECLARE_FIELD(offset)
    .set_default(0)
    .describe("Offset of the diagonal from the main diagonal. "
              "Can be both positive and negative. Defaults to 0.");
    DMLC_DECLARE_FIELD(axis1)
    .set_default(0)
    .describe("Axes to be used as the first axis of the 2-D sub-arrays "
              "from which the diagonals should be taken. Defaults to 0.");
    DMLC_DECLARE_FIELD(axis2)
    .set_default(1)
    .describe("Axes to be used as the second axis of the 2-D sub-arrays "
              "from which the diagonals should be taken. Defaults to 1.");
  }
};

template<int ndim, int req, bool back>
struct numpy_trace {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* a,
                                  mshadow::Shape<ndim> oshape,
                                  mshadow::Shape<ndim> ishape,
                                  index_t stride, index_t offset, int dlength) {
    using namespace mxnet_op;
    using namespace mshadow;
    index_t j = ravel(unravel(i, oshape), ishape) + offset;
    if (back) {
      for (index_t k = 0; k < dlength; ++k) {
        KERNEL_ASSIGN(out[j], req, a[i]);
        j += stride;
      }
    } else {
      if (req == kWriteTo) {
        out[i] = 0;
        for (index_t k = 0; k < dlength; ++k) {
          out[i] += a[j];
          j += stride;
        }
      } else if (req == kAddTo) {
        for (index_t k = 0; k < dlength; ++k) {
          out[i] += a[j];
          j += stride;
        }
      }
    }
  }
};

template<typename xpu, bool back>
void NumpyTraceOpProcess(const TBlob& in_data,
                         const TBlob& out_data,
                         const mxnet::TShape& ishape,
                         const mxnet::TShape& oshape,
                         index_t dsize,
                         const NumpyTraceParam& param,
                         mxnet_op::Stream<xpu> *s,
                         const std::vector<OpReqType>& req) {
  using namespace mxnet_op;
  using namespace mshadow;
  if (dsize == 0) {
    if (back) {
      if (out_data.Size() != 0) {
        MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
          MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
            if (req_type == kWriteTo) {
              out_data.FlatTo1D<xpu, DType>(s) = 0;
            }
          });
        });
      }
    }
    return;
  } else if (ishape.Size() == 0) {
    if (!back) {
      MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          if (req_type == kWriteTo) {
            out_data.FlatTo1D<xpu, DType>(s) = 0;
          }
        });
      });
    }
    return;
  }
  uint32_t x1 = CheckAxis(param.axis1, ishape.ndim());
  uint32_t x2 = CheckAxis(param.axis2, ishape.ndim());

  uint32_t idim = ishape.ndim();

  uint32_t minx = x1, maxx = x2;
  if (minx > maxx) {
    std::swap(minx, maxx);
  }

  // merges contiguous axes that are not separated
  // by axis1 or axis2 since they can be directly
  // mapped to the output and there is no need
  // to distinguish them
  // (After this the input will have no more than
  // three axes, hence improving the rave and
  // unravel efficiency)

  index_t oleading = 1,
          obody = 1,
          otrailing = 1;

  for (uint32_t i = 0; i < minx; ++i) {
    oleading *= ishape[i];
  }
  for (uint32_t i = minx + 1; i < maxx; ++i) {
    obody *= ishape[i];
  }
  for (uint32_t i = maxx + 1; i < idim; ++i) {
    otrailing *= ishape[i];
  }

  index_t ileading = oleading,
          ibody = obody * ishape[minx],
          itrailing = otrailing * ishape[maxx];

  index_t stride1 = itrailing * obody,
          stride2 = otrailing;
  // stride1 + stride2 is the stride for
  // iterating over the diagonal in question

  if (x1 == maxx) {
    std::swap(stride1, stride2);
  }

  // the extra index offset introduced by offset
  index_t offset;
  if (param.offset > 0) {
    offset = stride2 * param.offset;
  } else if (param.offset < 0) {
    offset = stride1 * -param.offset;
  } else {
    offset = 0;
  }

  // number of elements in the offset diagonal
  // may be negative
  int dlength;
  if (param.offset > 0) {
    dlength = std::min(ishape[x1], ishape[x2] - param.offset);
  } else if (param.offset < 0) {
    dlength = std::min(ishape[x1] - (-param.offset), ishape[x2]);
  } else {
    dlength = std::min(ishape[x1], ishape[x2]);
  }

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (back) {
        out_data.FlatTo1D<xpu, DType>(s) = 0;
      }
      Kernel<numpy_trace<3, req_type, back>, xpu>::Launch(s, dsize, out_data.dptr<DType>(),
                                                          in_data.dptr<DType>(),
                                                          Shape3(oleading, obody, otrailing),
                                                          Shape3(ileading, ibody, itrailing),
                                                          stride1 + stride2, offset, dlength);
    });
  });
}

template<typename xpu>
void NumpyTraceOpForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const NumpyTraceParam& param = nnvm::get<NumpyTraceParam>(attrs.parsed);

  NumpyTraceOpProcess<xpu, false>(in_data, out_data, ishape, oshape,
                                  out_data.Size(), param, s, req);
}

template<typename xpu>
void NumpyTraceOpBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const NumpyTraceParam& param = nnvm::get<NumpyTraceParam>(attrs.parsed);

  NumpyTraceOpProcess<xpu, true>(in_data, out_data, oshape, ishape,
                                 in_data.Size(), param, s, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TRACE_OP_INL_H_
