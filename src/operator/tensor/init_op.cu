/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cu
 * \brief GPU Implementation of init op
 */
#include <mshadow/tensor.h>
#include "./init_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief Fill a CSR NDArray with zeros by updating the aux shape
 * \param s - The device stream
 * \param dst - NDArray which is to be set to "all zeroes"
 */
void FillZerosCsrImpl(mshadow::Stream<mshadow::gpu> *s, const NDArray& dst) {
  dst.set_aux_shape(csr::kIdx, mshadow::Shape1(0));
  dst.CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(dst.shape()[0] + 1));
  TBlob indptr_data = dst.aux_data(csr::kIndPtr);
  MSHADOW_IDX_TYPE_SWITCH(dst.aux_type(csr::kIndPtr), IType, {
    mxnet_op::Kernel<mxnet_op::set_zero, mshadow::gpu>::Launch(
      s, indptr_data.Size(), indptr_data.dptr<IType>());
  });
}

NNVM_REGISTER_OP(_zeros)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>)
.set_attr<FComputeEx>("FComputeEx<gpu>", FillComputeZerosEx<gpu>);

NNVM_REGISTER_OP(_ones)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

NNVM_REGISTER_OP(_full)
.set_attr<FCompute>("FCompute<gpu>", InitFillWithScalarCompute<gpu>);

NNVM_REGISTER_OP(_arange)
.set_attr<FCompute>("FCompute<gpu>", RangeCompute<gpu>);

NNVM_REGISTER_OP(zeros_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>)
.set_attr<FComputeEx>("FComputeEx<gpu>", FillComputeZerosEx<gpu>);

NNVM_REGISTER_OP(ones_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

}  // namespace op
}  // namespace mxnet
