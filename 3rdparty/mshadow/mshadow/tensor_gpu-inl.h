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
 * \file tensor_gpu-inl.h
 * \brief implementation of GPU host code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_GPU_INL_H_
#define MSHADOW_TENSOR_GPU_INL_H_
#include "./base.h"
#include "./tensor.h"

namespace mshadow {
#if MSHADOW_USE_CUDA
template<>
inline void InitTensorEngine<gpu>(int dev_id) {
  cudaDeviceProp prop;
  int device_id = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  CHECK_GT(device_count, 0) << "Cannot find CUDA device. Please check CUDA-Configuration";
  if (dev_id < 0) {
    device_id = 0;
  } else {
    device_id = dev_id;
  }
  CHECK_LT(device_id, device_count) << "Incorrect Device ID";
  MSHADOW_CUDA_CALL(cudaSetDevice(device_id));
  MSHADOW_CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));
}
template<>
inline void ShutdownTensorEngine<gpu>(void) {
}
template<>
inline void SetDevice<gpu>(int devid) {
  MSHADOW_CUDA_CALL(cudaSetDevice(devid));
}
template<int dim, typename DType>
inline void AllocSpace(Tensor<gpu, dim, DType> *obj, bool pad) {
  size_t pitch;
  // common choice for cuda mem align unit is 32
  if (pad && obj->size(dim - 1) >= MSHADOW_MIN_PAD_RATIO * 32) {
    MSHADOW_CUDA_CALL(cudaMallocPitch(reinterpret_cast<void**>(&(obj->dptr_)), &pitch,
                                      obj->size(dim - 1) * sizeof(DType),
                                      obj->shape_.FlatTo2D()[0]));
    obj->stride_ = static_cast<index_t>(pitch / sizeof(DType));
  } else {
    obj->stride_ = obj->size(dim - 1);
    MSHADOW_CUDA_CALL(cudaMallocPitch(reinterpret_cast<void**>(&(obj->dptr_)), &pitch,
                                      obj->shape_.Size() * sizeof(DType), 1));
  }
}
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> *obj) {
  MSHADOW_CUDA_CALL(cudaFree(obj->dptr_));
  obj->dptr_ = NULL;
}
template<typename A, typename B, int dim, typename DType>
inline void Copy(Tensor<A, dim, DType> _dst,
                 Tensor<B, dim, DType> _src,
                 cudaMemcpyKind kind,
                 Stream<gpu> *stream) {
  CHECK_EQ(_dst.shape_, _src.shape_) << "Copy:shape mismatch";
  Tensor<A, 2, DType> dst = _dst.FlatTo2D();
  Tensor<B, 2, DType> src = _src.FlatTo2D();
  MSHADOW_CUDA_CALL(cudaMemcpy2DAsync(dst.dptr_, dst.stride_ * sizeof(DType),
                                      src.dptr_, src.stride_ * sizeof(DType),
                                      dst.size(1) * sizeof(DType),
                                      dst.size(0), kind,
                                      Stream<gpu>::GetStream(stream)));
  // use synchronize call behavior for zero stream
  if (stream == NULL) {
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(0));
  }
}
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<gpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
                 const Tensor<cpu, dim, DType> &src,
                 Stream<gpu> *stream) {
  Copy(dst, src, cudaMemcpyHostToDevice, stream);
}
#endif  // MSHADOW_USE_CUDA
}  // namespace mshadow

// the following part is included only if compiler is nvcc
#ifdef __CUDACC__
#include "./cuda/tensor_gpu-inl.cuh"

namespace mshadow {
template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, gpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, dim, DType, E>::kMapPass>
      ::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
  Shape<dim> eshape = expr::ShapeCheck<dim, E>::Check(exp.self());
  Shape<dim> dshape = expr::ShapeCheck<dim, R>::Check(dst->self());
  CHECK(eshape[0] == 0 || eshape == dshape)
    << "Assignment: Shape of Tensors are not consistent with target, "
    << "eshape: " << eshape << " dshape:" << dshape;
  cuda::MapPlan<Saver>(MakePlan(dst->self()),
                       MakePlan(exp.self()),
                       dshape.FlatTo2D(),
                       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}

template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, gpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, 1, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  Shape<2> eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self()).FlatTo2D();
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[1], dshape[0]) << "MapReduceKeepLowest::reduction dimension do not match";
  CHECK_NE(eshape[0], 0U) << "can not reduce over empty tensor";
  cuda::MapReduceKeepLowest<Saver, Reducer>
      (MakePlan(dst->self()), MakePlan(exp.self()), scale, eshape,
       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}

template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, gpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<gpu, dimkeep, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  typedef Shape<expr::ExpInfo<E>::kDim> EShape;
  EShape eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self());
    Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  CHECK_EQ(eshape[dimkeep], dshape[0]) << "MapReduceKeepHighDim::reduction dimension do not match";
  // use equvalent form
  Shape<4> pshape = Shape4(eshape.ProdShape(0, dimkeep),
                           eshape[dimkeep],
                           eshape.ProdShape(dimkeep + 1, EShape::kSubdim),
                           eshape[EShape::kSubdim]);
  // call equavalent map red dim 2
  cuda::MapReduceKeepDim1<Saver, Reducer>
      (MakePlan(dst->self()), MakePlan(exp.self()), scale, pshape,
       Stream<gpu>::GetStream(expr::StreamInfo<gpu, R>::Get(dst->self())));
}
template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> dst,
                    const Tensor<gpu, 2, DType>& src) {
  cuda::Softmax(dst, src);
}

template<typename DType>
inline void Softmax(Tensor<gpu, 3, DType> dst,
                    const Tensor<gpu, 3, DType>& src) {
  cuda::Softmax(dst, src);
}

template<typename DType>
inline void SoftmaxGrad(const Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label) {
  cuda::SoftmaxGrad(dst, src, label);
}

template<typename DType>
inline void SmoothSoftmaxGrad(const Tensor<gpu, 2, DType> &dst,
                              const Tensor<gpu, 2, DType> &src,
                              const Tensor<gpu, 1, DType> &label,
                              const float alpha) {
  cuda::SmoothSoftmaxGrad(dst, src, label, alpha);
}

template<typename DType>
inline void SoftmaxGrad(const Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const DType &ignore_label) {
  cuda::SoftmaxGrad(dst, src, label, ignore_label);
}

template<typename DType>
inline void SmoothSoftmaxGrad(const Tensor<gpu, 2, DType> &dst,
                              const Tensor<gpu, 2, DType> &src,
                              const Tensor<gpu, 1, DType> &label,
                              const DType &ignore_label,
                              const float alpha) {
  cuda::SmoothSoftmaxGrad(dst, src, label, ignore_label, alpha);
}

template<typename DType>
inline void SoftmaxGrad(const Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label) {
  cuda::SoftmaxGrad(dst, src, label);
}

template<typename DType>
inline void SoftmaxGrad(const Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const DType &ignore_label) {
  cuda::SoftmaxGrad(dst, src, label, ignore_label);
}

template<bool clip, typename IndexType, typename DType>
inline void AddTakeGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 1, IndexType>& index,
                        const Tensor<gpu, 2, DType> &src) {
  cuda::AddTakeGrad<clip, IndexType, DType>(dst, index, src);
}

template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(Tensor<gpu, 2, DType> dst,
                                  const Tensor<gpu, 1, IndexType>& sorted,
                                  const Tensor<gpu, 1, IndexType>& index,
                                  const Tensor<gpu, 2, DType> &src) {
  cuda::AddTakeGradLargeBatch(dst, sorted, index, src);
}

template<typename KDType, typename VDType>
inline void SortByKey(Tensor<gpu, 1, KDType> keys, Tensor<gpu, 1, VDType> values,
                      bool is_ascend) {
  cuda::SortByKey(keys, values, is_ascend);
}

template<typename IndexType, typename DType>
inline void IndexFill(Tensor<gpu, 2, DType> dst,
                      const Tensor<gpu, 1, IndexType>& index,
                      const Tensor<gpu, 2, DType> &src) {
  cuda::IndexFill(dst, index, src);
}
}  // namespace mshadow
#endif  // __CUDACC__
#endif  // MSHADOW_TENSOR_GPU_INL_H_
