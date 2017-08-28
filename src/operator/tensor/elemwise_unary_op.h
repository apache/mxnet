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
 * \file elementwise_unary_op-inl.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"
#include "../mxnet_op.h"
#include "cast_storage-inl.h"

namespace mxnet {
namespace op {

class OpBase {
 public:
  template<int req>
  struct SetToScalar {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType value) {
      KERNEL_ASSIGN(out[i], req, value);
    }
  };

  enum KernelComplexity {
    kComplexityLow  = 2 << 16
  };

  template<typename OP, typename xpu> class KernelEx;
  template<typename OP>
  class KernelEx<OP, cpu> {
   public:
    /*! \brief For relatively small number of iterations, don't use OMP, since it incurs
     * a significant amount of overhead relative to a low number of iterations
     * of trivial operations
     * Assumption is that the OP::Map() function is trivial
     *
     * @tparam CountForOMP Number of iterations before OMP will be used
     * @tparam Args Argument types to pass to Map function
     * @param s Stream
     * @param N Number of iterations
     * @param args Arguments to pass to Map function
     */
    template<KernelComplexity CountForOMP = kComplexityLow, typename ...Args>
    inline static void LaunchEx(mshadow::Stream<cpu> *s, int N, Args... args) {
      if (N < CountForOMP) {
        for (int i = 0; i < N; ++i) {
          OP::Map(i, args...);
        }
      } else {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
          OP::Map(i, args...);
        }
      }
    }
    template<typename ...Args>
    MSHADOW_CINLINE static void Launch(mshadow::Stream<cpu> *s, int N, Args... args) {
      mxnet_op::Kernel<OP, cpu>::Launch(s, N, args...);
    }
  };

  template<typename OP>
  class KernelEx<OP, gpu> {
   public:
    template<KernelComplexity CountForOMP = kComplexityLow, typename ...Args>
    MSHADOW_CINLINE static void LaunchTrivial(mshadow::Stream<gpu> *s, int N, Args... args) {
      mxnet_op::Kernel<OP, gpu>::Launch(s, N, args...);
    }
    template<typename ...Args>
    MSHADOW_CINLINE static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
      mxnet_op::Kernel<OP, gpu>::Launch(s, N, args...);
    }
  };

 protected:
  /*! \brief Copy blob data */
  template<typename xpu>
  static void inline CopyBlob(mshadow::Stream<xpu> *s,
                              const TBlob *dest_blob,
                              const OpReqType reqi,
                              const TBlob& src_blob) {
    CHECK_EQ(src_blob.type_flag_, dest_blob->type_flag_);
    CHECK_EQ(src_blob.shape_, dest_blob->shape_);
    MSHADOW_TYPE_SWITCH(src_blob.type_flag_, DType, {
      // Check if the pointers are the same (in-place operation needs no copy)
      if (reqi != kNullOp && src_blob.dptr<DType>() != dest_blob->dptr<DType>()) {
        mshadow::Copy(dest_blob->FlatTo1D<xpu, DType>(s), src_blob.FlatTo1D<xpu, DType>(s), s);
      }
    });
  }

  /*! \brief Allocate geometry-related blob data for sparse tensors
   * \param dest Destination sparse NDArray
   * \param clone_from sparse NDArray from which to clone storage attributes
   */
  static void AllocateGeometry(const NDArray *dest,
                               const OpReqType req,
                               const NDArray* clone_from = nullptr) {
    if (req != kNullOp) {
      if (clone_from) {
        const TShape ishape = clone_from->storage_shape();
        TShape sshape = dest->storage_shape();
        CHECK(shape_assign(&sshape, ishape));
        dest->CheckAndAllocData(sshape);
        CHECK_EQ(dest->storage_type(), clone_from->storage_type());
        for (size_t i = 0, n = clone_from->aux_shapes().size(); i < n; ++i) {
          TShape ashape = dest->aux_shape(i);
          CHECK(shape_assign(&ashape, clone_from->aux_shape(i)));
          dest->CheckAndAllocAuxData(i, ashape);
        }
        DCHECK_EQ(dest->aux_shapes().size(), clone_from->aux_shapes().size());
      } else {
        for (size_t i = 0, n = dest->aux_shapes().size(); i < n; ++i) {
          dest->CheckAndAllocAuxData(i, dest->aux_shape(i));
        }
        dest->CheckAndAllocData(dest->storage_shape());
      }
    }
  }

  /*! \brief Copy the geometry-related blobs (row sparse indexes, etc.) */
  template<typename xpu>
  static inline void CopyGeometryBlobs(mshadow::Stream<xpu> *s,
                                       const NDArray *dest,
                                       const OpReqType reqi,
                                       const NDArray &src) {
    CHECK_EQ(src.aux_shapes().size(), dest->aux_shapes().size());
    // My assumption is that the geometry blobs are not large enough to justify an omp loop here,
    // since the thread synchronization calls for each fork will take longer
    // than copying a few floats
    for (size_t i = 0, n = src.aux_shapes().size(); i < n; ++i) {
      const TBlob src_blob = src.aux_data(i);
      const TBlob dest_blob = dest->aux_data(i);
      CopyBlob<xpu>(s, &dest_blob, reqi, src_blob);
    }
  }

  /*! \brief Generic copy NDArray */
  template<typename xpu>
  static inline void CopyNDArray(mshadow::Stream<xpu> *s,
                                 const NDArray *dest,
                                 const OpReqType reqi,
                                 const NDArray& src) {
    DCHECK_NE(dest->storage_type(), kDefaultStorage);
    DCHECK_EQ(dest->storage_type(), src.storage_type());
    AllocateGeometry(dest, reqi, &src);
    CopyGeometryBlobs(s, dest, reqi, src);
    CopyBlob(s, &dest->data(), reqi, src.data());
  }

  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    std::vector<TBlob> in_blobs, out_blobs;
    in_blobs.reserve(inputs.size());
    out_blobs.reserve(outputs.size());
    for (size_t i = 0, n = inputs.size(); i < n; ++i) {
      in_blobs.emplace_back(std::move(inputs[i].data()));
    }
    for (size_t i = 0, n = outputs.size(); i < n; ++i) {
      out_blobs.emplace_back(std::move(outputs[i].data()));
    }
    computer(attrs, ctx, in_blobs, req, out_blobs);
  }

  /*! \brief Keep row shape[0] dimension and gather the remaining dimensions in location shape[1] */
  template<typename DType, typename xpu>
  static inline mshadow::Tensor<xpu, 2, DType> AsRowise2D(mshadow::Stream<xpu> *s,
                                                          const TBlob& blob) {
    const size_t dim = blob.shape_.ndim();
    if (dim) {
      TShape shape({blob.shape_[0], 1});
      for (size_t i = 1; i < dim; ++i) {
        shape[1] *= blob.shape_[i];
      }
      return mshadow::Tensor<xpu, 2, DType>(
        blob.dptr<DType>(), mshadow::Shape2(shape[0], shape[1]), s);
    }
    return mshadow::Tensor<xpu, 2, DType>();
  }

  /*! \brief Fill dense output block with a single scalar value */
  template<typename xpu, typename DType>
  static inline void FillDense(mshadow::Stream<xpu> *s,
                               const size_t size,
                               const DType val,
                               const OpReqType req,
                               DType *out) {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      KernelEx<SetToScalar<Req>, xpu>::LaunchEx(s, size, out, val);
    });
  }
};  // OpBase

/*! \brief Unary operator class */
class UnaryOp : public OpBase {
  /*! \brief Infer the output storage geometry
   * \return boolean signifying whether the proper storage geometry was initialized
   */
  template<int n_in, int n_out>
  static bool InitStorageGeometry(const nnvm::NodeAttrs& attrs,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), static_cast<size_t>(n_in))
      << " in operator " << attrs.name;
    CHECK_EQ(outputs.size(), static_cast<size_t>(n_out))
      << " in operator " << attrs.name;
    static_assert(n_in > 0 && n_out > 0, "Invalid input and/or output count values");
    const TShape& isshape = inputs[0].storage_shape();
    if (!shape_is_none(isshape)) {
      NDArray *output = nullptr;
      for (size_t i = 0, n = inputs.size(); i < n; ++i) {
        const NDArray &input = inputs[i];
        if (i < n_out) {
          output = const_cast<NDArray *>(&outputs[i]);
        }
        CHECK_EQ(output->shape(), inputs[i].shape());
        CHECK_EQ(output->storage_type(), input.storage_type());
        CHECK_EQ(output->aux_shapes().size(), input.aux_shapes().size());
        std::vector<TShape> aux_shapes;
        const size_t aux_shape_count = input.aux_shapes().size();
        aux_shapes.reserve(aux_shape_count);
        for (size_t j = 0; j < aux_shape_count; ++j) {
          aux_shapes.emplace_back(input.aux_shape(j));
        }
        output->CheckAndAlloc(aux_shapes);
        DCHECK_EQ(output->storage_shape(), input.storage_shape());
      }
      return true;
    } else if (isshape.ndim() > 0 && !isshape.Size()
      && inputs[0].storage_type() != kDefaultStorage) {
      return true;  // 0% density
    } else {
      CHECK(false);  // implement when necessary
    }
    return false;
  }

 protected:
  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    // Copy over geometry
    DCHECK_EQ(inputs.size(), 1U);
    DCHECK_EQ(outputs.size(), 1U);
    InitStorageGeometry<1, 1>(attrs, inputs, outputs);
    CHECK_EQ(inputs.size(), outputs.size());  // need to figure out what to do for binary type
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage);
    CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type());
    AllocateGeometry(&outputs[0], req[0], &inputs[0]);
    CopyGeometryBlobs<xpu>(ctx.get_stream<xpu>(), &outputs[0], req[0], inputs[0]);
    outputs[0].CheckAndAllocData(inputs[0].storage_shape());
    OpBase::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, computer);
  }

 public:
  template<typename xpu, typename OP>
  static void Compute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<mxnet_op::mshadow_to_kernel<OP, Req>, xpu>::Launch(
          s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
      });
    });
  }

  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    if (inputs[0].storage_shape().Size()) {
      MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    }
  }

  template<typename xpu, typename op>
  static void KernelCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    if (req[0] != kNullOp) {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<op, xpu>::Launch(s, outputs[0].Size(),
                                outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
      });
    }
  }

  template<typename xpu, typename op>
  static void ComputeWithHalf2(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      Kernel<op, xpu>::Launch(s, outputs[0].Size(),
                              outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
    });
  }

  template<typename xpu, typename OP>
  static void KernelComputeEx(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    if (inputs[0].storage_shape().Size()) {
      MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, KernelCompute<xpu, OP>);
    }
  }

  template<typename xpu>
  static void IdentityCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (req[0] == kNullOp) return;
    if (req[0] == kWriteInplace) {
      CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_); return;
    }
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req[0], F<mshadow_op::identity>(inputs[0].FlatTo1D<xpu, DType>(s)));
    });
  }

  template<typename xpu>
  static void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    if (inputs[0].storage_type() == outputs[0].storage_type()) {
      MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, IdentityCompute<xpu>);
    } else {
      FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                           IdentityCompute<xpu>, "IdentityComputeEx");
    }
  }

  template<typename xpu>
  static void IdentityComputeFirstItemsEx(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    OpBase::CopyNDArray(ctx.get_stream<xpu>(), &outputs[0], req[0], inputs[0]);
  }
};

template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a * GRAD_OP::Map(b);
  }
};

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Output data type.");
  }
};

inline bool CastType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void CastCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0], tcast<DstDType>(data));
    });
  });
}

namespace kernel_launch_op {
/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *in) {
    out[i] = DType(DType(1.0f) / (DType(1.0f) + expf(-in[i])));
  }
};
struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType out_grad, DType in) {
    return out_grad * DType(in * (DType(1.0f) - in));
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *in) {
    DType x = in[i];
    out[i] = x > DType(0.0f) ? x : DType(0.0f);
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType out_grad, DType in) {
    return out_grad * DType(in > DType(0.0f) ? DType(1.0f) : DType(0.0f));
  }
};
}  // namespace kernel_launch_op

#define MXNET_OPERATOR_REGISTER_UNARY(__name$)                      \
  NNVM_REGISTER_OP(__name$)                                         \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

#define MXNET_OPERATOR_REGISTER_UNARY_DR(__name$)                   \
  NNVM_REGISTER_OP(__name$)                                         \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

/*! \brief Register scalar op name as an alias */
#define MXNET_ADD_SPARSE_OP_ALIAS(__name$) \
  .add_alias("_sparse_" #__name$)

/*! \brief Unary compute */
#define MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE(__name$, __xpu$, __kernel$)              \
  MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                   \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)        \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeEx<__xpu$, __kernel$>)

/*! \brief Unary compute, dense result */
#define MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(__name$, __xpu$, __kernel$)           \
  MXNET_OPERATOR_REGISTER_UNARY_DR(__name$)                                                \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
