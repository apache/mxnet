/*!
 * Copyright (c) 2016 by Contributors
 * \file nnpack_util.h
 * \brief
 * \author Carwin
*/
#ifndef MXNET_OPERATOR_NNPACK_NNPACK_UTIL_H_
#define MXNET_OPERATOR_NNPACK_NNPACK_UTIL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <nnpack.h>

namespace mxnet {
namespace op {

class NNPACKInitialize {
 public:
  pthreadpool_t threadpool;

 public:
  NNPACKInitialize() {
    nnp_status status = nnp_initialize();
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnp_initialize failed status=" << status;
    }
    int num_threads = dmlc::GetEnv("MXNET_CPU_NNPACK_NTHREADS", 4);
    this->threadpool = pthreadpool_create(num_threads);
  }
  virtual ~NNPACKInitialize() {
    nnp_status status = nnp_deinitialize();
    if (nnp_status_success != status) {
      LOG(FATAL) << "nnp_deinitialize failed status=" << status;
    }
    pthreadpool_destroy(threadpool);
  }
};

// nnpackinitialize will be used in all other nnpack op
extern NNPACKInitialize nnpackinitialize;

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NNPACK_NNPACK_UTIL_H_
