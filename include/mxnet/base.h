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
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief configuration of MXNet as well as basic data structure.
 */
#ifndef MXNET_BASE_H_
#define MXNET_BASE_H_

#include "dmlc/base.h"
#include <string>
#include "dmlc/io.h"
#include "dmlc/type_traits.h"
#include "dmlc/parameter.h"
#include "mshadow/tensor.h"
// nnvm headers for symbolic construction.
#include "nnvm/op.h"
#include "nnvm/symbolic.h"
#include "libinfo.h"
#include "tuple.h"


/*!
 * \brief define compatible keywords in g++
 *  Used to support g++-4.6 and g++4.7
 */
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
#error "Currently we need g++ 4.8 or higher to fully support c++11 features"
#define override
#define final
#endif
#endif

/*!
 * \brief define dllexport for Visual Studio
 */
#ifdef _MSC_VER
#ifdef MXNET_EXPORTS
#define MXNET_API __declspec(dllexport)
#else
#define MXNET_API __declspec(dllimport)
#endif
#else
#define MXNET_API
#endif

/*!
 * \brief define prediction only
 */
#ifndef MXNET_PREDICT_ONLY
#define MXNET_PREDICT_ONLY 0
#endif

/*! \brief major version */
#define MXNET_MAJOR 1
/*! \brief minor version */
#define MXNET_MINOR 5
/*! \brief patch version */
#define MXNET_PATCH 0
/*! \brief mxnet version */
#define MXNET_VERSION (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
/*! \brief helper for making version number */
#define MXNET_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + patch)
/*!
 * \brief define function name as profiler message
 */
#define PROFILER_MESSAGE_FUNCNAME (__FUNCTION__)

/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief mxnet cpu */
typedef mshadow::cpu cpu;
/*! \brief mxnet gpu */
typedef mshadow::gpu gpu;
/*! \brief index type usually use unsigned */
typedef mshadow::index_t index_t;
/*! \brief data type that will be used to store ndarray */
typedef mshadow::default_real_t real_t;
/*! \brief operator structure from NNVM */
using Op = nnvm::Op;

/*! \brief Context information about the execution environment */
struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = cpu::kDevMask,
    kGPU = gpu::kDevMask,
    kCPUPinned = 3,
    kCPUShared = 5,
  };
  /*! \brief the device type we run the op on */
  DeviceType dev_type;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
  /*! \brief default constructor */
  Context() : dev_type(kCPU), dev_id(0) {}
  /*!
   * \brief Get corresponding device mask
   * \return cpu::kDevMask or gpu::kDevMask
   */
  inline DeviceType dev_mask() const {
    if (dev_type == kCPUPinned || dev_type == kCPUShared) return kCPU;
    return dev_type;
  }
  /*!
   * \brief Returns dev_id for kGPU and kCPUPinned, 0 otherwise
   */
  inline int real_dev_id() const {
    if (dev_type == kCPUPinned || dev_type == kGPU) return dev_id;
    return 0;
  }
  /*!
   * \brief Comparator, used to enable Context as std::map key.
   * \param b another context to compare
   * \return compared result
   */
  inline bool operator<(const Context &b) const;
  /*!
   * \brief check if current context equals another one
   * \param b another context to compare
   * \return whether dev mask and id are same
   */
  inline bool operator==(const Context &b) const {
    return dev_type == b.dev_type && dev_id == b.dev_id;
  }
  /*!
   * \brief check if current context not equals another one
   * \param b another context to compare
   * \return whether they are not the same
   */
  inline bool operator!=(const Context &b) const {
    return !(*this == b);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  inline void Save(dmlc::Stream *strm) const {
    strm->Write(&dev_type, sizeof(dev_type));
    strm->Write(&dev_id, sizeof(dev_id));
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  inline bool Load(dmlc::Stream *strm) {
    if (strm->Read(&dev_type, sizeof(dev_type)) != sizeof(dev_type)) return false;
    if (strm->Read(&dev_id, sizeof(int32_t)) != sizeof(int32_t)) return false;
    return true;
  }
  /*! \brief the maximal device type */
  static const int32_t kMaxDevType = 6;
  /*! \brief the maximal device index */
  static const int32_t kMaxDevID = 16;
  /*!
   * \brief Create a new context.
   * \param dev_type device type.
   * \param dev_id device id. -1 for current device.
   */
  inline static Context Create(DeviceType dev_type, int32_t dev_id = -1);
  /*! \return CPU Context */
  inline static Context CPU(int32_t dev_id = 0);
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context. -1 for current GPU.
   */
  inline static Context GPU(int32_t dev_id = -1);
  /*!
   * Get the number of GPUs available.
   * \return The number of GPUs that are available.
   */
  inline static int32_t GetGPUCount();
  /*!
   * Is the cuda driver installed and visible to the system.
   * \return Whether the driver is present.
   */
  inline static bool GPUDriverPresent();
  /*!
   * Get the number of streams that a GPU Worker has available to operations.
   * \return The number of streams that are available.
   */
  inline static int32_t GetGPUStreamsPerWorker();
  /*!
   * \brief get the free and total available memory on a GPU
   * \param dev the GPU number to query
   * \param free_mem pointer to the uint64_t holding free GPU memory
   * \param total_mem pointer to the uint64_t holding total GPU memory
   * \return No return value
   */
  inline static void GetGPUMemoryInformation(int dev, uint64_t *free, uint64_t *total);
  /*!
   * Create a pinned CPU context.
   * \param dev_id the device id for corresponding GPU.
   * \return Pinned CPU context. -1 for current GPU.
   */
  inline static Context CPUPinned(int32_t dev_id = -1);
  /*!
   * Create a CPU shared memory context.
   * \param dev_id dummy device id.
   * \return CPU shared memory context.
   */
  inline static Context CPUShared(int32_t dev_id = 0);
  /*!
   * Create a context from string of the format [cpu|gpu|cpu_pinned](n)
   * \param str the string pattern
   * \return Context
   */
  inline static Context FromString(const std::string& str);

 private:
#if MXNET_USE_CUDA
    static void CudaLibChecks();
#endif
#if MXNET_USE_CUDNN
    static void CuDNNLibChecks();
#endif
};

#if MXNET_USE_CUDA
/*! \brief Holds an auxiliary mshadow gpu stream that can be synced with a primary stream. */
class GPUAuxStream {
 public:
  /*!
   * \brief constructor.
   * \param primary_stream gpu stream that is synced with the created auxiliary stream.
   */
  explicit GPUAuxStream(mshadow::Stream<gpu> *primary_stream) :
      primary_stream_(primary_stream),
      aux_stream_(primary_stream),
      gpu_stream_sync_event_(nullptr) {
    if (Context::GetGPUStreamsPerWorker() >= 2) {
      // Create auxiliary stream on the same device with the same properties as the primary stream
      bool primary_has_blas_handle =
          primary_stream->blas_handle_ownership_ == mshadow::Stream<gpu>::OwnHandle;
      bool primary_has_dnn_handle =
          primary_stream->dnn_handle_ownership_ == mshadow::Stream<gpu>::OwnHandle;
      aux_stream_ = mshadow::NewStream<gpu>(primary_has_blas_handle,
                                            primary_has_dnn_handle,
                                            primary_stream->dev_id);
      MSHADOW_CUDA_CALL(cudaEventCreateWithFlags(&gpu_stream_sync_event_, cudaEventDisableTiming));
    }
  }
  /*! \brief destructor */
  ~GPUAuxStream() {
    // If the aux_stream_ == primary_stream_, then we created no new streams to destroy.
    if (aux_stream_ != primary_stream_) {
      MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(aux_stream_));
      MSHADOW_CATCH_ERROR(cudaEventDestroy(gpu_stream_sync_event_));
    }
  }
  /*!
   * \brief Makes future aux stream work wait on the completion of existing primary stream work.
   */
  void PreAuxStreamUseSync() {
    // If the aux_stream_ == primary_stream_, then no synchronization is necessary.
    if (aux_stream_ != primary_stream_)
      StreamSync(primary_stream_, aux_stream_, gpu_stream_sync_event_);
  }
  /*!
   * \brief Makes future primary stream work wait on the completion of existing aux stream work.
   */
  void PostAuxStreamUseSync() {
    // If the aux_stream_ == primary_stream_, then no synchronization is necessary.
    if (aux_stream_ != primary_stream_)
      StreamSync(aux_stream_, primary_stream_, gpu_stream_sync_event_);
  }
  /*! \brief Getter for created auxiliary stream. */
  mshadow::Stream<gpu> *GetStream() { return aux_stream_; }
  /*!
   * \brief Make future work enqueued to `s2` wait on completion of current work enqueued to `s1`.
   * \param s1 stream with work that must be completed before future s2 work can begin.
   * \param s2 stream whose future work is made to wait on the completion of existing s1 work.
   * \param event used to pass s1 state to s2.
   */
  static void StreamSync(mshadow::Stream<gpu> *s1, mshadow::Stream<gpu> *s2, cudaEvent_t event) {
    MSHADOW_CUDA_CALL(cudaEventRecord(event, s1->stream_));
    MSHADOW_CUDA_CALL(cudaStreamWaitEvent(s2->stream_, event, 0));
  }

 private:
  mshadow::Stream<gpu> *primary_stream_;
  mshadow::Stream<gpu> *aux_stream_;
  cudaEvent_t gpu_stream_sync_event_;
};

/*!
 * \brief Provides automatic coordination of an auxilary stream with a primary one.
 * This object, upon construction, prepares an aux stream for use by syncing it with enqueued
 * primary-stream work.  Object destruction will sync again so future primary-stream work
 * will wait on enqueued aux-stream work.  If MXNET_GPU_WORKER_NSTREAMS == 1, then this defaults
 * simply: the primary stream will equal the aux stream and the syncs will be executed as nops.
 * See ./src/operator/cudnn/cudnn_convolution-inl.h for a usage example.
 */
class SyncedGPUAuxStream {
 public:
  /*!
   * \brief constructor.
   * \param gpu_aux_stream auxilary gpu stream that is managed by this RAII object.
   */
  explicit SyncedGPUAuxStream(GPUAuxStream *gpu_aux_stream) : gpu_aux_stream_(gpu_aux_stream) {
    gpu_aux_stream_->PreAuxStreamUseSync();
  }
  /*! \brief destructor */
  ~SyncedGPUAuxStream() {
    gpu_aux_stream_->PostAuxStreamUseSync();
  }
  /*! \brief copy constructor deleted to prevent unexpected synchronizations. */
  SyncedGPUAuxStream(const SyncedGPUAuxStream&) = delete;
  /*! \brief copy assignment operator deleted to prevent unexpected synchronizations. */
  void operator=(const SyncedGPUAuxStream&) = delete;
  /*! \brief move constructor permitted as alternative to copying. */
  SyncedGPUAuxStream(SyncedGPUAuxStream&&) = default;
  /*! \brief move assignment operator permitted as alternative to copy assignment. */
  SyncedGPUAuxStream& operator=(SyncedGPUAuxStream&&) = default;
  /*! \brief Getter for underlying mshadow::Stream<gpu>. */
  inline mshadow::Stream<gpu>* GetStream() const {
    return gpu_aux_stream_->GetStream();
  }

 private:
  GPUAuxStream *gpu_aux_stream_;
};
#endif  // MXNET_USE_CUDA

/*!
 * \brief execution time context.
 *  The information needed in runtime for actual execution.
 */
struct RunContext {
  /*! \brief base Context */
  Context ctx;
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
  /*!
   * \brief the auxiliary stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *aux_stream;
  /*!
   * \brief indicator of whether this execution is run in bulk mode
   */
  bool is_bulk;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return static_cast<mshadow::Stream<xpu>*>(stream);
  }
#if MXNET_USE_CUDA
  /*!
   * \brief get an RAII object that transparently handles the syncing of the auxiliary stream.
   * \return the aux stream auto-syncing object
   */
  inline SyncedGPUAuxStream get_gpu_aux_stream() const {
    return SyncedGPUAuxStream(static_cast<GPUAuxStream*>(aux_stream));
  }
#endif
  /*! \brief get the base Context from RunContext */
  inline const Context& get_ctx() const {
    return ctx;
  }
};
}  // namespace mxnet

//! \cond Doxygen_Suppress
namespace mxnet {
// implementing Context
inline bool Context::operator<(const Context &b) const {
  if (dev_type == b.dev_type) {
    return dev_id < b.dev_id;
  } else {
    return dev_type < b.dev_type;
  }
}
inline Context Context::Create(DeviceType dev_type, int32_t dev_id) {
  Context ctx;
  ctx.dev_type = dev_type;
  ctx.dev_id = dev_id < 0 ? 0 : dev_id;
  if (dev_type & kGPU) {
#if MXNET_USE_CUDA
    CudaLibChecks();
#endif
#if MXNET_USE_CUDNN
    CuDNNLibChecks();
#endif
    if (dev_id < 0) {
#if MXNET_USE_CUDA
      CHECK_EQ(cudaGetDevice(&ctx.dev_id), cudaSuccess);
#else
      LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif
    }
  }
  return ctx;
}
inline Context Context::CPU(int32_t dev_id) {
  return Create(kCPU, dev_id);
}

inline Context Context::CPUPinned(int32_t dev_id) {
  return Create(kCPUPinned, dev_id);
}

inline Context Context::CPUShared(int32_t dev_id) {
  return Create(kCPUShared, dev_id);
}

inline Context Context::GPU(int32_t dev_id) {
  return Create(kGPU, dev_id);
}

inline bool Context::GPUDriverPresent() {
#if MXNET_USE_CUDA
  int cuda_driver_version = 0;
  CHECK_EQ(cudaDriverGetVersion(&cuda_driver_version), cudaSuccess);
  return cuda_driver_version > 0;
#else
  return false;
#endif
}

inline int32_t Context::GetGPUCount() {
#if MXNET_USE_CUDA
  if (!GPUDriverPresent()) {
    return 0;
  }
  int32_t count;
  cudaError_t e = cudaGetDeviceCount(&count);
  // TODO(junwu): Remove e == 35
  // This is skipped for working around wheel build system with older CUDA driver.
  if (e == cudaErrorNoDevice || e == 35) {
    return 0;
  }
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);
  return count;
#else
  return 0;
#endif
}

inline int32_t Context::GetGPUStreamsPerWorker() {
  // The default number of streams available if the user has not set MXNET_GPU_WORKER_NSTREAMS.
  const int32_t default_num_streams = 1;
  // The get_aux_stream() interface can supply one additional stream beyond the standard one.
  static int32_t num_streams =
      dmlc::GetEnv("MXNET_GPU_WORKER_NSTREAMS", default_num_streams) >= 2 ? 2 : 1;
  return num_streams;
}

inline void Context::GetGPUMemoryInformation(int dev, uint64_t *free_mem,
                                             uint64_t *total_mem) {
#if MXNET_USE_CUDA

  size_t memF, memT;
  cudaError_t e;

  int curDevice;
  e = cudaGetDevice(&curDevice);
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);

  e = cudaSetDevice(dev);
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);

  e = cudaMemGetInfo(&memF, &memT);
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);

  e = cudaSetDevice(curDevice);
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);

  *free_mem = static_cast<uint64_t>(memF);
  *total_mem = static_cast<uint64_t>(memT);

#else
  LOG(FATAL)
      << "This call is only supported for MXNet built with CUDA support.";
#endif
}

inline Context Context::FromString(const std::string& str) {
  Context ret;
  try {
    const std::string::size_type l = str.find('(');
    CHECK_NE(l, std::string::npos);
    const std::string::size_type r = str.find(')');
    CHECK_EQ(r, str.length()-1);

    const std::string type = str.substr(0, l);
    int id = std::stoi(str.substr(l+1, r-l-1));
    if (type == "cpu") {
      ret = CPU(id);
    } else if (type == "gpu") {
      ret = GPU(id);
    } else if (type == "cpu_pinned") {
      ret = CPUPinned(id);
    } else if (type == "cpu_shared") {
      ret = CPUShared(id);
    } else {
      LOG(FATAL) << "Invalid context string " << str;
    }
  } catch (...) {
    LOG(FATAL) << "Invalid context string " << str;
  }
  return ret;
}

inline std::ostream& operator<<(std::ostream &out, const Context &ctx) {
  if (ctx.dev_type == Context::kCPU) {
    out << "cpu(";
  } else if (ctx.dev_type == Context::kGPU) {
    out << "gpu(";
  } else if (ctx.dev_type == Context::kCPUPinned) {
    out << "cpu_pinned(";
  } else if (ctx.dev_type == Context::kCPUShared) {
    out << "cpu_shared(";
  } else {
    out << "unknown(";
  }
  out << ctx.dev_id << ")";
  return out;
}

// describe op registration point
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define MXNET_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" STRINGIZE(__LINE__))
#define ADD_FILELINE "\n\nDefined in " __FILE__ ":L" STRINGIZE(__LINE__)


#if MXNET_USE_MKLDNN == 1
constexpr size_t kMKLDNNAlign = 64;
#endif

}  // namespace mxnet

namespace std {
template<> struct hash<mxnet::Context> {
  size_t operator()(const mxnet::Context& ctx) const {
    size_t res = 0;
    res = dmlc::HashCombine(res, static_cast<size_t>(ctx.dev_type));
    res = dmlc::HashCombine(res, static_cast<size_t>(ctx.dev_id));
    return res;
  }
};

#if __cplusplus < 201402L && !defined(_MSC_VER)
template<typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif
}  // namespace std

#include "./tensor_blob.h"
//! \endcond
#endif  // MXNET_BASE_H_
