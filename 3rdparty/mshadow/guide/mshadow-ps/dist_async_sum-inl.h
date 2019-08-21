/**
 * @brief  Simple test of KVLayer
 */
#include "ps.h"
#include "parameter/kv_layer.h"
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <map>
#include <mshadow/tensor.h>
#include <mshadow-ps/mshadow_ps.h>
#include "dbstr.h"
#include "glog/logging.h"

namespace mshadow {
namespace ps {


template<typename DType>
class Updater : public IModelUpdater<DType> {
 protected:
  void InitModel_(int key, Tensor<cpu, 1, DType> data) {
    data = 0;
    data_[key] = data;
  }

  void Update_(int key, Tensor<cpu, 1, DType> data) {
    data_[key] += data;
    // LOG(ERROR) << dbstr(data_[key]);
  }
  std::map<int, Tensor<cpu, 1, DType> > data_;
};

template<typename DType>
IModelUpdater<DType> *CreateModelUpdater(void) {
  return new Updater<DType>();
}

}  // namespace ps
}  // namespace mshadow

// this function is runed by specific thread
template<typename xpu>
inline void RunWorkerThread(int devid,
                            mshadow::ps::ISharedModel<xpu, float> *ps) {
  // initialize tensor engine
  mshadow::InitTensorEngine<xpu>(devid);
  mshadow::Stream<xpu> *stream  = mshadow::NewStream<xpu>();
  // allocate tensor on xpu
  mshadow::TensorContainer<xpu, 2> data(mshadow::Shape2(2, 3));
  // set the computation stream to the new allocated stream
  // this will make subsequent computation whose target is data
  // to use the stream, stream is needed for async execution in GPU
  data.set_stream(stream);
  // intiaialize the key, register the shape on parameter server
  ps->InitKey(data[0].shape_, 0, devid);
  ps->InitKey(data[1].shape_, 1, devid);
  // first step, pull the data back from server
  ps->PullReq(data[0], 0, devid);
  ps->PullReq(data[1], 1, devid);

  // PullWait will block until these request finishes
  ps->PullWait(0, devid);
  ps->PullWait(1, devid);

  data[1] = devid + data[0];

  LOG(ERROR) << "node " << ::ps::MyNodeID() << ", dev " << devid << ": before sync\n"
             << dbstr(data);

  // push data[0] out, for update, or aggregation
  // 0 is the key of the data, devid is the current device id
  ps->Push(data[0], 0, devid);
  // pull request is used to request the data to be copied back
  // once computation is done
  ps->PullReq(data[0], 0, devid);
  // computation can be done here..
  // the pull request handler will be overlapped with
  // similar as previous call
  ps->PullWait(0, devid);

  ps->Push(data[1], 1, devid);
  ps->PullReq(data[1], 1, devid);
  // more computation can be done here...
  // the computation will be overlapped
  // PullWait will block until these request finishes
  ps->PullWait(1, devid);

  LOG(ERROR) << "node " << ::ps::MyNodeID() << ", dev " << devid
             << ": after sync\n" << dbstr(data);

  mshadow::DeleteStream(stream);
  mshadow::ShutdownTensorEngine<xpu>();
}

template<typename xpu>
inline int Run(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: device list\n"\
           "\tfor CPU the device list can be arbitrary\n"\
           "\tfor GPU the device list need to be actual device index\n");
    return 0;
  }
  // list of device ids
  std::vector<int> devs;
  // initialization
  for (int i = 1; i < argc; ++i) {
    // record the device id
    devs.push_back(atoi(argv[i]));
  }
  mshadow::ps::ISharedModel<xpu, float>
      *ps = mshadow::ps::CreateSharedModel<xpu, float>("dist");
  // intiaialize the ps
  ps->SetParam("update_on_server", "1");
  ps->Init(devs);
  // use openmp to launch #devs threads
  #pragma omp parallel num_threads(devs.size())
  {
    int tid = omp_get_thread_num();
    RunWorkerThread<xpu>(devs[tid], ps);
  }
  delete ps;
  return 0;
}
