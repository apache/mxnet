/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_rabit-inl.h
 * \brief distributed version of PS using BSP
 *     synchronization in the backend
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_RABIT_INL_H_ // NOLINT(*)
#define MSHADOW_PS_RABIT_INL_H_ // NOLINT(*)
#include <vector>
#include "./mshadow_ps.h"
#include "./ps_local-inl.h"

#if MSHADOW_RABIT_PS
#include <rabit.h>
namespace mshadow {
namespace ps {
// multi-threaded implementation of
template<typename xpu, typename DType>
class RabitModel : public LocalModel<xpu, DType> {
 public:
  // parent type
  typedef LocalModel<xpu, DType> Parent;
  // constructor
  RabitModel() {
    // enforce usage of fifo queue
    this->use_fifo_push_queue = 1;
    destroy_reduce_thread_ = false;
    disable_allreduce_ = 0;
    this->init_reducer_ = 0;
  }
  virtual ~RabitModel(void) {
    Parent::Destroy();
    if (init_reducer_ != 0) {
      destroy_reduce_thread_ = true;
      reduce_queue_.Abort(1);
      thread_reduce_handler_.Join();
      reduce_queue_.Destroy();
    }
  }
  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    this->use_fifo_push_queue = 1;
    // use fifo
    reduce_queue_.Init(true);
    thread_reduce_handler_.Start(ReduceGlobalThread, this);
    init_reducer_ = 1;
    // initialize other things
    Parent::Init(devices);
  }
  // set parameters
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "msg:disable_allreduce")) {
      disable_allreduce_ = atoi(val);
    }
    Parent::SetParam(name, val);
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // summation the data fron all devices
    LocalModel<xpu, DType>::ReduceSum(data);
    CHECK_EQ(data[0].CheckContiguous(), true) << "data must be contiguous";
    ReduceTask tsk;
    tsk.data = data[0]; tsk.key = key;
    reduce_queue_.Push(tsk, 0);
  }

 private:
  // reduce task
  struct ReduceTask {
    int key;
    mshadow::Tensor<cpu, 2> data;
  };
  // destroy reduce
  bool destroy_reduce_thread_;
  // whether reducer is initialized
  int init_reducer_;
  // check disable_allreduce functionalities
  int disable_allreduce_;
  // reduce handler thread
  utils::Thread thread_reduce_handler_;
  // queue for allreduce task
  utils::ThreadPQueue<ReduceTask> reduce_queue_;
  // reduce handler
  inline void ReduceHandler(void) {
    while (!destroy_reduce_thread_) {
      ReduceTask tsk;
      if (reduce_queue_.Pop(&tsk)) {
        CHECK_EQ(disable_allreduce_, 0) << "Allreduce disabled error";
        int key = tsk.key;
        rabit::Allreduce<rabit::op::Max>(&key, 1);
        CHECK_EQ(key, tsk.key) << "Allreduce not concensus";
        rabit::Allreduce<rabit::op::Sum>
            (tsk.data.dptr_, tsk.data.MSize());
        tsk.data *= 1.0f / rabit::GetWorldSize();
        CHECK_EQ(disable_allreduce_, 0) << "Allreduce disabled error";
        this->HandleReduceFinish(tsk.data, tsk.key);
      } else {
        CHECK_EQ(destroy_reduce_thread_, true) << "abort but not destroy";
      }
    }
  }
  /*!\brief entry point of reduce thread */
  inline static MSHADOW_THREAD_PREFIX ReduceGlobalThread(void *pthread) {
    static_cast<RabitModel*>(pthread)->ReduceHandler();
    return NULL;
  }
};
}  // namespace ps
}  // namespace mshadow
#endif  // MSHADOW_RABIT_PS
#endif  // MSHADOW_PS_RABIT_INL_H_ // NOLINT(*)
