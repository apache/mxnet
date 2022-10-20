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
 * \file stream_gpu-inl.h
 * \brief implementation of GPU code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_STREAM_GPU_INL_H_
#define MSHADOW_STREAM_GPU_INL_H_
#include <memory>
#include "./base.h"
#include "./tensor.h"
#include "dmlc/logging.h"

namespace mshadow {
#if MSHADOW_USE_CUDA == 1
// Stream alocation
// actual implementation of GPU stream in CUDA
template<>
struct Stream<gpu> {
  /*! \brief handle state */
  enum HandleState {
    NoHandle = 0,
    OwnHandle = 1,
  };
  /*! \brief cudaStream */
  cudaStream_t stream_;
  /*! \brief cublas handle */
  cublasHandle_t blas_handle_;
  /*! \brief cusolver handle */
  #if MSHADOW_USE_CUSOLVER == 1
  cusolverDnHandle_t solver_handle_;
  #endif
  /*! \brief cudnn handle */
  #if MSHADOW_USE_CUDNN == 1
  cudnnHandle_t dnn_handle_;
  #endif
  /*! \brief cutensor handle */
  #if MSHADOW_USE_CUTENSOR== 1
  cutensorHandle_t cutensor_handle_;
  #endif
  /*! \brief cublas handle ownership */
  HandleState blas_handle_ownership_;
  /*! \brief cusolver handle ownership */
  HandleState solver_handle_ownership_;
  /*! \brief cudnn handle ownership */
  HandleState dnn_handle_ownership_;
  /*! \brief cutensor handle ownership */
  HandleState cutensor_handle_ownership_;
  void* cutensor_cachelines_ = nullptr;
  /*! \brief cudaDeviceProp */
  cudaDeviceProp prop;
  /*! \brief dev id */
  int dev_id;

  Stream(void)
    : stream_(0)
      , blas_handle_(0)
#if MSHADOW_USE_CUDNN == 1
      , dnn_handle_(0)
#endif
      //, cutensor_handle_()
      , blas_handle_ownership_(NoHandle)
      , solver_handle_ownership_(NoHandle)
      , dnn_handle_ownership_(NoHandle)
      , cutensor_handle_ownership_(NoHandle)
      , cutensor_cachelines_(nullptr){}
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(stream_));
  }
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    LOG(FATAL) << cudaGetErrorString(err);
    return false;
  }
  /*!
   * \brief returns actual cudaStream_t given an input GPU stream pointer
   * \param stream pointer to GPU stream
   */
  inline static cudaStream_t GetStream(Stream<gpu> *stream) {
    if (stream == NULL) {
#if MSHADOW_FORCE_STREAM
      LOG(FATAL) << "Default GPU stream was used when MSHADOW_FORCE_STREAM was on";
#endif
      return 0;
    } else {
      return stream->stream_;
    }
  }
  /*!
   * \brief return actual cublasHandle
   * \param pointer to GPU stream
   */
  inline static cublasHandle_t GetBlasHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->blas_handle_ownership_, NoHandle)
        << "No handle exist in source stream";
      return stream->blas_handle_;
    }
  }
  /*! \brief Destory cublas handle if own it */
  inline void DestroyBlasHandle() {
    if (blas_handle_ownership_ == OwnHandle) {
      cublasStatus_t err = cublasDestroy(blas_handle_);
      blas_handle_ownership_ = NoHandle;
      CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Destory cublas handle failed";
    }
  }
  /*! \brief Destory original blas handle and create a new one */
  inline void CreateBlasHandle() {
    this->DestroyBlasHandle();
    cublasStatus_t err = cublasCreate(&blas_handle_);
    blas_handle_ownership_ = OwnHandle;
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Create cublas handle failed";
    err = cublasSetStream(blas_handle_, stream_);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Setting cublas stream failed";
  }
#if MSHADOW_USE_CUSOLVER == 1
  inline static cusolverDnHandle_t GetSolverHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->solver_handle_ownership_, NoHandle) << "No handle exist in source stream";
      return stream->solver_handle_;
    }
  }
#endif
  inline void DestroySolverHandle() {
#if MSHADOW_USE_CUSOLVER == 1
    if (solver_handle_ownership_ == OwnHandle) {
      cusolverStatus_t err = cusolverDnDestroy(solver_handle_);
      CHECK_EQ(err, CUSOLVER_STATUS_SUCCESS) << "Destory cusolver handle failed";
    }
#endif
  }
  inline void CreateSolverHandle() {
#if MSHADOW_USE_CUSOLVER == 1
    this->DestroySolverHandle();
    cusolverStatus_t err = cusolverDnCreate(&solver_handle_);
    CHECK_EQ(err, CUSOLVER_STATUS_SUCCESS) << "Create cusolver handle failed";
    err = cusolverDnSetStream(solver_handle_, stream_);
    CHECK_EQ(err, CUSOLVER_STATUS_SUCCESS) << "Setting cusolver stream failed";
    this->solver_handle_ownership_ = OwnHandle;
#endif
  }
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
  inline static cudnnHandle_t GetDnnHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->dnn_handle_ownership_, NoHandle) << "No handle exist in source stream";
      return stream->dnn_handle_;
    }
  }
#endif
  inline void DestroyDnnHandle() {
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    if (dnn_handle_ownership_ == OwnHandle) {
      cudnnStatus_t err = cudnnDestroy(dnn_handle_);
      this->dnn_handle_ownership_ = NoHandle;
      CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    }
#endif
  }
  inline void CreateDnnHandle() {
// #if MSHADOW_USE_CUDNN == 1 && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    this->DestroyDnnHandle();
    cudnnStatus_t err = cudnnCreate(&dnn_handle_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    // At this point, we have the resource which may need to be freed
    this->dnn_handle_ownership_ = OwnHandle;
    err = cudnnSetStream(dnn_handle_, stream_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
#endif
  }
  inline void DestroyCuTensorHandle() {
#if MSHADOW_USE_CUTENSOR == 1
    if (cutensor_handle_ownership_ == OwnHandle) {
      // not destroy method available
      if (cutensor_cachelines_ != nullptr) {
        cutensorStatus_t err;
        const char* cacheFilename = getenv("MXNET_CUTENSOR_CACHEFILE");
        if (cacheFilename != nullptr) {
          err = cutensorHandleWriteCacheToFile(&cutensor_handle_, cacheFilename);
          CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);
        }
        err = cutensorHandleDetachPlanCachelines(&cutensor_handle_);
        CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);
        free(cutensor_cachelines_);
        cutensor_cachelines_ = nullptr;
      }
      this->cutensor_handle_ownership_ = NoHandle;
    }
#endif
  }
  inline void CreateCuTensorHandle() {
#if MSHADOW_USE_CUTENSOR == 1
    this->DestroyCuTensorHandle();
    cutensorStatus_t err = cutensorInit(&cutensor_handle_);
    CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);
    const char* cacheFilename = getenv("MXNET_CUTENSOR_CACHEFILE");
    if (cacheFilename != nullptr) {
      constexpr int32_t numCachelines = 1024;
      size_t sizeCache = numCachelines * sizeof(cutensorPlanCacheline_t);
      cutensor_cachelines_ = malloc(sizeCache);
      err = cutensorHandleAttachPlanCachelines(&cutensor_handle_, (cutensorPlanCacheline_t*) cutensor_cachelines_, numCachelines);
      CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);

      uint32_t numCachelinesRead = 0;
      cutensorStatus_t status = cutensorHandleReadCacheFromFile(&cutensor_handle_, cacheFilename, &numCachelinesRead);
      if (status == CUTENSOR_STATUS_IO_ERROR) {
        printf("File (%s) doesn't seem to exist.\n", cacheFilename);
      } else if (status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE) {
        printf("Cannot read cache: Please attach at least %d cachelines to the handle.\n", numCachelinesRead);
      }
    }
    // At this point, we have the resource which may need to be freed
    this->cutensor_handle_ownership_ = OwnHandle;
#endif
  }
};
template<>
inline void DeleteStream<gpu>(Stream<gpu> *stream) {
  if (stream) {
    stream->DestroyCuTensorHandle();
    MSHADOW_CUDA_CALL(cudaStreamDestroy(stream->stream_));
    stream->DestroyBlasHandle();
    stream->DestroySolverHandle();
    stream->DestroyDnnHandle();
    delete stream;
  }
}
template<>
inline Stream<gpu> *NewStream<gpu>(bool create_blas_handle,
                                   bool create_dnn_handle,
                                   int dev_id) {
  // RAII on Cuda exception
  struct StreamDeleter { void operator()(Stream<gpu> *ptr) const { DeleteStream<gpu>(ptr); } };
  std::unique_ptr<Stream<gpu>, StreamDeleter> st(new Stream<gpu>());
  MSHADOW_CUDA_CALL(cudaStreamCreate(&st->stream_));
  if (create_blas_handle) {
    st->CreateBlasHandle();
    st->CreateSolverHandle();
  }
  if (create_dnn_handle) {
    st->CreateDnnHandle();
  }
#if MSHADOW_USE_CUTENSOR == 1
  st->CreateCuTensorHandle();
#endif
  st->dev_id = dev_id;
  if (dev_id != -1) {
    MSHADOW_CUDA_CALL(cudaGetDeviceProperties(&st->prop, dev_id));
  }
  return st.release();
}
#endif
}  // namespace mshadow
#endif  // MSHADOW_STREAM_GPU_INL_H_
