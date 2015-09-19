/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.h
 * \brief Global resource allocation handling.
 */
#ifndef MXNET_RESOURCE_H_
#define MXNET_RESOURCE_H_

#include <dmlc/logging.h>
#include "./base.h"
#include "./engine.h"

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
    kTempSpace
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
   * \param The stream to use in the random number generator.
   * \return the mshadow random number generator requested.
   * \tparam xpu the device type of random number generator.
   */
  template<typename xpu>
  inline mshadow::Random<xpu>* get_random(
      mshadow::Stream<xpu> *stream) const {
    CHECK_EQ(req.type, ResourceRequest::kRandom);
    mshadow::Random<xpu> *ret =
        static_cast<mshadow::Random<xpu>*>(ptr_);
    ret->set_stream(stream);
    return ret;
  }
  /*!
   * \brief Get space requested as mshadow Tensor.
   *  The caller can request arbitrary size.
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
    CHECK_EQ(req.type, ResourceRequest::kTempSpace);
    mshadow::TensorContainer<xpu, 1, real_t> *space =
        static_cast<mshadow::TensorContainer<xpu, 1, real_t>*>(ptr_);
    space->Resize(mshadow::Shape1(shape.Size()));
    return mshadow::Tensor<xpu, ndim, real_t>(
        space->dptr_, shape, shape[ndim - 1], stream);
  }
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
   * \brief Seed all the allocated random numbers.
   * \param seed the seed to the random number generators on all devices.
   */
  virtual void SeedRandom(uint32_t seed) = 0;
  /*! \brief virtual destructor */
  virtual ~ResourceManager() DMLC_THROW_EXCEPTION {}
  /*!
   * \return Resource manager singleton.
   */
  static ResourceManager *Get();
};
}  // namespace mxnet
#endif  // MXNET_RESOURCE_H_
