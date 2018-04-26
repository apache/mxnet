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

#ifndef MPI_WRAPPER_H_
#define MPI_WRAPPER_H_

#if MXNET_USE_MPI_DIST_KVSTORE

#include <mpi.h>

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mpi_message.pb.h"

template<typename DType>
MPI_Datatype MPI_Data_Type_Cast(void)
{
  LOG(FATAL) << "Need to template specialization to get mpi data type";
  return -1;
}

template<>
MPI_Datatype MPI_Data_Type_Cast<int>(void)
{
  return MPI_INT;
}

template<>
MPI_Datatype MPI_Data_Type_Cast<float>(void)
{
  return MPI_FLOAT;
}

template<>
MPI_Datatype MPI_Data_Type_Cast<double>(void)
{
  return MPI_DOUBLE;
}

template <class xpu, class DType>
struct MPI_Wrapper
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank)
  { return 0; };

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array)
  { return 0; };
};

//CPU Implementation
template <class DType>
struct MPI_Wrapper<mxnet::cpu, DType>
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank)
  {
    DType *buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret = MPI_Bcast(buf, count, MPI_Data_Type_Cast<DType>(), root_rank, MPI_COMM_WORLD);
    return ret;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array)
  {
    DType *send_buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    DType *recv_buf = reinterpret_cast<DType *>(output_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret;
    assert(input_array->data().Size() == output_array->data().Size());

    if (send_buf != recv_buf) {
      ret = MPI_Allreduce((const void *)send_buf, (void *)recv_buf,
                         count, MPI_Data_Type_Cast<DType>(), MPI_SUM, MPI_COMM_WORLD);
    } else {
      ret = MPI_Allreduce(MPI_IN_PLACE, (void *)recv_buf,
                         count, MPI_Data_Type_Cast<DType>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return ret;
  }
};

// TODO GPU Implementation
template <class DType>
struct MPI_Wrapper<mxnet::gpu, DType>
{
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank)
  {
    // TODO
    LOG(FATAL) << "MPI For GPU version has not been implemented.";
    return -1;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array)
  {
    // TODO
    LOG(FATAL) << "MPI For GPU version has not been implemented.";
    return -1;
  }
};

#endif
#endif
