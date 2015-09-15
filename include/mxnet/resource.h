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
    /*! \brief Temporal space */
    kTempSpace
  };
  /*! \brief type of resources */
  Type type;
  /*! \brief size of space requested, in terms of number of reals */
  size_t space_num_reals;
  /*! \brief default constructor */
  ResourceRequest() {}
  /*!
   * \brief constructor, allow implicit conversion
   * \param type type of resources
   */
  ResourceRequest(Type type, size_t space_num_reals = 0)  // NOLINT(*)
      : type(type), space_num_reals(space_num_reals) {}
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
  /*!
   * \brief pointer to the resource, do not use directly,
   *  access using member functions
   */
  void *ptr_;
  /*!
   * \brief Get random number generator.
   * \return the mshadow random number generator requested.
   * \tparam xpu the device type of random number generator.
   */
  template<typename xpu>
  inline mshadow::Random<xpu>* get_random() const {
    CHECK_EQ(req.type, ResourceRequest::kRandom);
    return static_cast<mshadow::Random<xpu>*>(ptr_);
  }
  /*!
   * \brief Get space requested as mshadow Tensor.
   *  The resulting tensor must fit in space requsted.
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
    CHECK_GE(req.space_num_reals, shape.Size());
    return mshadow::Tensor<xpu, ndim, real_t>(
        static_cast<real_t*>(ptr_), shape, shape[ndim - 1], stream);
  }
};
}  // namespace mxnet
#endif  // MXNET_RESOURCE_H_
