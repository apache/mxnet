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

/**
 * Copyright (c) 2018 by Contributors
 */

#ifndef MXNET_MPI_COLLECTIVES_H_
#define MXNET_MPI_COLLECTIVES_H_

#if MXNET_USE_MPI_DIST_KVSTORE

#include <vector>
#include <string>
#include <mxnet/ndarray.h>

namespace mxnet {
namespace kvstore {

int MXMPIGetMpiSize(int *ret);

int MXMPIGetMpiRank(int *ret);

int MXMPIInit();

int MXMPIGetLocalRank(int *ret);

int MXMPIAllReduce(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &in_values,
                   const std::vector<mxnet::NDArray*> &out_values,
                   int priority);

int MXMPIAllReduceEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &in_values,
                     const std::vector<mxnet::NDArray*> &out_values,
                     int priority);

int MXMPIBroadcast(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int root_rank,
                   int priority);

int MXMPIBroadcastEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int root_rank,
                     int priority);

int MXMPIAllGather(const std::vector<int> &keys,
                   const std::vector<mxnet::NDArray*> &values,
                   int priority);

int MXMPIAllGatherEx(const std::vector<std::string> &keys,
                     const std::vector<mxnet::NDArray*> &values,
                     int priority);

}
}
#endif
#endif
