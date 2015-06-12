/*!
 *  Copyright (c) 2015 by Contributors
 * \file ref_counter.h
 * \brief reference counter that can be used to track
 *   lifeness of objects
 */
#ifndef MXNET_REF_COUNTER_H_
#define MXNET_REF_COUNTER_H_
#include <dmlc/base.h>

namespace mxnet {
/*!
 * \brief reference counter similiar to shared_ptr
 *  use this implementation to avoid c++11 issues
 */
class RefCounter {
 public:
  /*! \brief copy constructor */
  RefCounter() : value_(new int(1)) {
    value_ = new int(1);
  }
  /*! \brief copy constructor */
  RefCounter(const RefCounter &c)
      : value_(c.value_) {
    ++value_[0];
  }
#if DMLC_USE_CXX11
  /*! \brief move constructor */
  RefCounter(RefCounter &&c)
      : value_(c.value_) {
    c.value_ = nullptr;
  }
#endif
  /*! \brief destructor */
  ~RefCounter() {
    if (value_ != NULL) { 
      --value_[0];
      if (value_[0] == 0) delete value_;
    }
  }
  /*! \brief assign operator */
  inline RefCounter &operator=(const RefCounter &c) {
    --value_[0];
    if (value_[0] == 0) delete value_;
    value_ = c.value_;
    ++value_[0];
    return *this;
  }
  /*! \return value from the reference counter */
  inline int value() const {
    return value_[0];
  }
  
 private:
  int *value_;
};
}  // namespace mxnet
#endif  // MXNET_REF_COUNTER_H_
