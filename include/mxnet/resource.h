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
 * \file resource.h
 * \brief Global resource allocation handling.
 */
#ifndef MXNET_RESOURCE_H_
#define MXNET_RESOURCE_H_

#include <dmlc/logging.h>
#include "./base.h"
#include "./engine.h"
#include "./random_generator.h"

namespace mxnet {

/*!
 * \brief The resources that can be requested by Operator
 */
struct ResourceRequest {
  /*! \brief Resource type, indicating what the pointer type is */
  enum Type {
    /*! \brief mshadow::Random<xpu> object */
    kRandom,
    /*! \brief A dynamic temp space that can be arbitrary size */
    kTempSpace,
    /*! \brief common::RandGenerator<xpu> object, which can be used in GPU kernel functions */
    kParallelRandom
#if MXNET_USE_CUDNN == 1
    ,
    /*! \brief cudnnDropoutDescriptor_t object for GPU dropout kernel functions */
    kCuDNNDropoutDesc
#endif  // MXNET_USE_CUDNN == 1
  };
  /*! \brief type of resources */
  Type type;
  /*! \brief default constructor */
  ResourceRequest() {}
  /*!
   * \brief constructor, allow implicit conversion
   * \param type type of resources
   */
  ResourceRequest(Type type)  // NOLINT(*)
      : type(type) {}
};


/*!
 * \brief Resources used by mxnet operations.
 *  A resource is something special other than NDArray,
 *  but will still participate
 */
struct Resource {
  /*! \brief The original request */
  ResourceRequest req;
  /*! \brief engine variable */
  engine::VarHandle var;
  /*! \brief identifier of id information, used for debug purpose */
  int32_t id;
  /*!
   * \brief pointer to the resource, do not use directly,
   *  access using member functions
   */
  void *ptr_;
  /*! \brief default constructor */
  Resource() : id(0) {}
  /*!
   * \brief Get random number generator.
   * \param stream The stream to use in the random number generator.
   * \return the mshadow random number generator requested.
   * \tparam xpu the device type of random number generator.
   */
  template<typename xpu, typename DType>
  inline mshadow::Random<xpu, DType>* get_random(
      mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kRandom);
    mshadow::Random<xpu, DType> *ret =
        static_cast<mshadow::Random<xpu, DType>*>(ptr_);
    ret->set_stream(stream);
    return ret;
  }

  /*!
   * \brief Get parallel random number generator.
   * \tparam xpu the device type of random number generator.
   * \tparam DType the return type.
   * \return the parallel random number generator. for gpu, it is allocated on global memory.
   */
  template<typename xpu, typename DType>
  inline common::random::RandGenerator<xpu, DType>* get_parallel_random() const {
    CHECK_EQ(req.type, ResourceRequest::kParallelRandom);
    return static_cast<common::random::RandGenerator<xpu, DType>*>(ptr_);
  }

  /*!
   * \brief Get space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
   *
   *  This space can be shared with other calls to this->get_space.
   *  So the caller need to serialize the calls when using the conflicted space.
   *  The old space can get freed, however, this will incur a synchronization,
   *  when running on device, so the launched kernels that depend on the temp space
   *  can finish correctly.
   *
   * \param shape the Shape of returning tensor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   * \tparam xpu the device type of random number generator.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim>
  inline mshadow::Tensor<xpu, ndim, real_t> get_space(
      mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream) const {
    return get_space_typed<xpu, ndim, real_t>(shape, stream);
  }
  /*!
   * \brief Get cpu space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor.
   * \return the mshadow tensor requested.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<int ndim>
  inline mshadow::Tensor<cpu, ndim, real_t> get_host_space(
      mshadow::Shape<ndim> shape) const {
    return get_host_space_typed<cpu, ndim, real_t>(shape);
  }
  /*!
   * \brief Get space requested as mshadow Tensor in specified type.
   *  The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   * \tparam xpu the device type of random number generator.
   * \tparam ndim the number of dimension of the tensor requested.
   */
  template<typename xpu, int ndim, typename DType>
  inline mshadow::Tensor<xpu, ndim, DType> get_space_typed(
      mshadow::Shape<ndim> shape, mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kTempSpace);
    return mshadow::Tensor<xpu, ndim, DType>(
        reinterpret_cast<DType*>(get_space_internal(shape.Size() * sizeof(DType))),
        shape, shape[ndim - 1], stream);
  }
#if MXNET_USE_CUDNN == 1
  /*!
   * \brief Get cudnn dropout descriptor from shared state space.
   *
   * \param dropout_desc reference to previously created cudnn dropout descriptor.
   * \param stream the stream of retruning tensor.
   * \return the mshadow tensor requested.
   */
  void get_cudnn_dropout_desc(
      cudnnDropoutDescriptor_t* dropout_desc,
      mshadow::Stream<gpu> *stream,
      const float dropout,
      uint64_t seed) const;
#endif  // MXNET_USE_CUDNN == 1

  /*!
   * \brief Get CPU space as mshadow Tensor in specified type.
   * The caller can request arbitrary size.
   *
   * \param shape the Shape of returning tensor
   * \return the mshadow tensor requested
   * \tparam ndim the number of dimnesion of tensor requested
   * \tparam DType request data type
   */
  template<int ndim, typename DType>
  inline mshadow::Tensor<cpu, ndim, DType> get_host_space_typed(
    mshadow::Shape<ndim> shape) const {
      return mshadow::Tensor<cpu, ndim, DType>(
        reinterpret_cast<DType*>(get_host_space_internal(shape.Size() * sizeof(DType))),
        shape, shape[ndim - 1], nullptr);
  }
  /*!
   * \brief internal function to get space from resources.
   * \param size The size of the space.
   * \return The allocated space.
   */
  void* get_space_internal(size_t size) const;
  /*!
   * \brief internal function to get cpu space from resources.
   * \param size The size of space.
   * \return The allocated space
   */
  void *get_host_space_internal(size_t size) const;
};

/*! \brief Global resource manager */
class ResourceManager {
 public:
  /*!
   * \brief Get resource of requested type.
   * \param ctx the context of the request.
   * \param req the resource request.
   * \return the requested resource.
   * \note The returned resource's ownership is
   *       still hold by the manager singleton.
   */
  virtual Resource Request(Context ctx, const ResourceRequest &req) = 0;
  /*!
   * \brief Seed all the allocated random number generators.
   * \param seed the seed to the random number generators on all devices.
   */
  virtual void SeedRandom(uint32_t seed) = 0;
  /*!
   * \brief Seed the random number generators of the given context.
   * \param seed the seed to the random number generators.
   */
  virtual void SeedRandom(Context ctx, uint32_t seed) = 0;
  /*! \brief virtual destructor */
  virtual ~ResourceManager() DMLC_THROW_EXCEPTION {}
  /*!
   * \return Resource manager singleton.
   */
  static ResourceManager *Get();
};
}  // namespace mxnet
#endif  // MXNET_RESOURCE_H_
