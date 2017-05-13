/*!
 * Copyright (c) 2017 by Contributors
 * \file sampling_op.h
 * \brief Function definitions of operators for sampling from multiple distributions
 */
#ifndef MXNET_OPERATOR_TENSOR_MULTISAMPLE_OP_H_
#define MXNET_OPERATOR_TENSOR_MULTISAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct MultiSampleParam : public dmlc::Parameter<MultiSampleParam> {
  TShape shape;
  int dtype;
  DMLC_DECLARE_PARAMETER(MultiSampleParam) {
    DMLC_DECLARE_FIELD(shape)
      .set_default(TShape())
      .describe("Shape to be sampled from each random distribution.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

inline bool MultiSampleOpShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_attrs,
                               std::vector<TShape>* out_attrs) {
  CHECK_GT(in_attrs->size(), 0)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_LT(in_attrs->size(), 3)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);
  // Get shape to be sampled for each parameter set.
  const MultiSampleParam& param = nnvm::get<MultiSampleParam>(attrs.parsed);
  TShape sshape = param.shape;
  for (size_t i = 0; i < sshape.ndim(); ++i) {
    CHECK_GT(sshape[i], 0) << "shape parameter must be non-zero within each dimension";
  }
  // Examine output shape whether it is already defined.
  TShape tshape((*out_attrs)[0]);
  // The illegal case of tshape.ndim() <= sshape.ndim() will
  // automatically crash when we back-propagate from inputs to outputs.
  if (tshape.ndim() > sshape.ndim()) {
    // Promote down by removing last dimensions which represent the samples.
    tshape = TShape(tshape.begin(), tshape.begin()+(tshape.ndim()-sshape.ndim()));
  }
  // Shape assignemnt/checking for inputs.
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    if ( !shape_assign(&tshape, (*in_attrs)[i])) return false;
  }
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    SHAPE_ASSIGN_CHECK(*in_attrs, i, tshape);
  }
  if (tshape.ndim() > 0) {
    // Shape assignment/check for propagation from inputs to output.
    std::vector<int> cshape(tshape.begin(), tshape.end());
    cshape.insert(cshape.end(), sshape.begin(), sshape.end());
    TShape oshape(cshape.begin(), cshape.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  return true;
}

inline bool MultiSampleOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_GT(in_attrs->size(), 0)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_LT(in_attrs->size(), 3)
    << "sampling operator takes 1 or 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);

  // All inputs must have same type.
  int dtype = -1;
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    if (!type_assign(&dtype, (*in_attrs)[i])) return false;
  }
  for (size_t i = 0; i < in_attrs->size(); ++i) {
    TYPE_ASSIGN_CHECK(*in_attrs, i, dtype);
  }
  if (-1 == dtype) return false;

  // The output may have a different type so we can't infer from inputs.
  const MultiSampleParam& param = nnvm::get<MultiSampleParam>(attrs.parsed);
  dtype = (*out_attrs)[0];
  if (dtype != -1) {
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype, param.dtype) << "Inferred output type does not match requested type: "
      << dtype << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred. Use type in args or default.
    dtype = (param.dtype == -1 ? mshadow::kFloat32 : param.dtype);
  }
  bool dtype_ok = (dtype == mshadow::kFloat16) || (dtype == mshadow::kFloat32) ||
    (dtype == mshadow::kFloat64);
  CHECK_EQ(dtype_ok, true) << "Output type must be float16, float32, or float64: dtype is "
    << dtype<< " vs " << mshadow::kFloat16 << " or " << mshadow::kFloat32 << " or "
    << mshadow::kFloat64;
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);
  return true;
}


template<typename xpu, typename generator>
void MultiSampleOpForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_GT(inputs.size(), 0);
  CHECK_LT(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  using namespace mxnet_op;
  const MultiSampleParam& param = nnvm::get<MultiSampleParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in0 = inputs[0];
  const TBlob& in1 = (inputs.size() == 1 ? inputs[0] : inputs[1]);
  const TBlob& out = outputs[0];
  if (out.Size() == 0) return;
  CHECK_EQ(in0.CheckContiguous(), true);
  CHECK_EQ(in1.CheckContiguous(), true);
  CHECK_GT(in0.Size(), 0);
  CHECK_EQ(out.CheckContiguous(), true);
  CHECK_EQ(out.Size() % in0.Size(), 0);
  const int N(in0.Size()), M(out.Size()/in0.Size());

  // Seed for the sampling process. In order to guarantee deterministic
  // behaviour for single threaded cpu, this is taken from mshadow random generator.
  const int seed(ctx.requested[0].get_random<xpu, float>(s)->GetRandInt());

  MSHADOW_TYPE_SWITCH(in0.type_flag_, IType, {
    MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, OType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        // Get the output as a 2D-tensor with dimensions NxM
        Tensor<xpu, 2, OType> samples = out.get_with_shape<xpu, 2, OType>(Shape2(N, M), s);
        const IType *iptr1 = in0.dptr<IType>(), *iptr2 = in1.dptr<IType>();

        // The seeds for the different generators are itself a random sequence. We don't
        // want to create the same samples in case that we have two samplers with same
        // input parameters.
        std::mt19937 seed_generator(seed);
        for (int i = 0; i < N; ++i) {
          // Generate seed for this sampler. Must be mutexed as calling
          // a random generator is not thread safe.
          int seed = seed_generator();
          typename generator::template Sampler<OType> sampler(iptr1[i], iptr2[i], seed);
          // Get the sub-tensor that will hold the results of this sampler.
          Tensor<xpu, 1, OType> slice = samples.Slice(i, i+1).FlatTo1D();
          for (int j = 0; j < M; ++j) {
            KERNEL_ASSIGN(slice[j], req_type, sampler());
          }
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_MULTISAMPLE_OP_H_
