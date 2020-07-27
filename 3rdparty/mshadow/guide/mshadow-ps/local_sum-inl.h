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

// This is an example demonstrating the usage of mshadow ps
#include <cstdio>
// use openmp to launch multiple threads
#include <omp.h>
#include <mshadow/tensor.h>
#include <mshadow-ps/mshadow_ps.h>

// simple util to print result
void Print_(mshadow::Tensor<mshadow::cpu, 2, float> ts) {
  for (mshadow::index_t i = 0; i < ts.size(0); ++i) {
    for (mshadow::index_t j = 0; j < ts.size(1); ++j) {
      printf("%g ", ts[i][j]);
    }
    printf("\n");
  }
}
template<typename xpu>
inline void Print(mshadow::Tensor<xpu, 2, float> ts) {
  mshadow::TensorContainer<mshadow::cpu, 2, float> tmp;
  tmp.Resize(ts.shape_);
  mshadow::Copy(tmp, ts);
  Print_(tmp);
}

// this function is runed by specific thread
template<typename xpu>
inline void RunWorkerThread(int devid,
                            mshadow::ps::ISharedModel<xpu, float> *ps) {
  // initialize tensor engine
  mshadow::InitTensorEngine<xpu>(devid);
  mshadow::Stream<xpu> *stream  = mshadow::NewStream<xpu>(devid);
  // allocate tensor on xpu
  mshadow::TensorContainer<xpu, 2> data(mshadow::Shape2(2, 3));
  // set the computation stream to the new allocated stream
  // this will make subsequent computation whose target is data
  // to use the stream, stream is needed for async execution in GPU
  data.set_stream(stream);
  // assume these operations sets the content of dataient
  data[0] = 1.0f;
  data[1] = devid + data[0];
  printf("dev%d: before sync, data:\n", devid);
  // use print to show result, do not call
  // print normally since Copy will block
  Print(data);
  printf("====================\n");
  // intiaialize the key, register the shape on parameter server
  ps->InitKey(data[0].shape_, 0, devid);
  ps->InitKey(data[1].shape_, 1, devid);
  // push data[0] out, for update, or aggregation
  // 0 is the key of the data, devid is the current device id
  ps->Push(data[0], 0, devid);
  // pull request is used to request the data to be copied back
  // once computation is done
  ps->PullReq(data[0], 0, devid);
  // computation can be done here..
  // the pull request handler will be overlapped with
  // similar as previous call
  ps->Push(data[1], 1, devid);
  ps->PullReq(data[1], 1, devid);
  // more computation can be done here...
  // the computation will be overlapped
  // PullWait will block until these request finishes
  ps->PullWait(0, devid);
  ps->PullWait(1, devid);
  printf("dev%d: after sync, data:\n", devid);
  // use print to show result, do not call
  // print normally since Copy will block
  Print(data);
  printf("====================\n");
  mshadow::DeleteStream(stream);
  mshadow::ShutdownTensorEngine<xpu>();
}

namespace mshadow {
namespace ps {
// model updater is used when update is happening on server side
// if we only use parameter server for sum aggregation
// this is not needed, but we must declare this function to return NULL
template<>
IModelUpdater<float> *CreateModelUpdater(void) {
  return NULL;
}
}
}

template<typename xpu>
inline int Run(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: device list\n"\
           "\tfor CPU the device list can be arbitrary\n"\
           "\tfor GPU the device list need to be actual device index\n");
    return 0;
  }
#if MSHADOW_RABIT_PS
  rabit::Init(argc, argv);
#endif
  // list of device ids
  std::vector<int> devs;
  // initialization
  for (int i = 1; i < argc; ++i) {
    // record the device id
    devs.push_back(atoi(argv[i]));
  }
  mshadow::ps::ISharedModel<xpu, float>
      *ps = mshadow::ps::CreateSharedModel<xpu, float>("local");
  // intiaialize the ps
  ps->Init(devs);
  // use openmp to launch #devs threads
  #pragma omp parallel num_threads(devs.size())
  {
    int tid = omp_get_thread_num();
    RunWorkerThread<xpu>(devs[tid], ps);
  }
  delete ps;
#if MSHADOW_RABIT_PS
  rabit::Finalize();
#endif
  return 0;
}
