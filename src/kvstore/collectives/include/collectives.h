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

#ifndef MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLLECTIVES_H_
#define MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLLECTIVES_H_

#if MXNET_USE_ALLREDUCE_DIST_KVSTORE

#include <mxnet/ndarray.h>

#include <vector>
#include <string>

#include "../../comm.h"

namespace mxnet {
namespace kvstore {

/*!
 * \brief Get total node number.
 * \param ret out param for total node number.
 * \return 0 when success, -1 when failure happens
 */
int MXGetMpiSize(int *ret);

/*!
 * \brief Get the rank of this node.
 * \param ret out param for rank.
 * \return 0 when success, -1 when failure happens
 */
int MXGetMpiRank(int *ret);

/*!
 * \brief Initialize collective library.
 * \param comm commDevice for reduce and broadcast
 *        within single node.
 * \return 0 when success, -1 when failure happens
 */
int MXCOLLIBInit(Comm *comm);

/*!
 * \brief Get the local rank.
 * \param ret out param for local rank.
 * \return 0 when success, -1 when failure happens
 */
int MXGetLocalRank(int *ret);

/*!
 * \brief Do Allreduce across the multi-node.
 * \param key key.
 * \param send_value value to be sent.
 * \param recv_value value to hold the result.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllReduce(int key,
                mxnet::NDArray* send_value,
                mxnet::NDArray* recv_value,
                int priority);

/*!
 * \brief Broadcast values in root rank to all other nodes.
 * \param key the key for the ndarray.
 * \param value the value to be broadcast.
 * \param root_rank the value in the rank to be broadcast.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXBroadcast(int key,
                mxnet::NDArray* value,
                int root_rank,
                int priority);

/*!
 * \brief All gather values in all nodes.
 * \param key the key for the value.
 * \param value the value to be gathered.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllGather(int key,
                mxnet::NDArray* value,
                int priority);

/*!
 * \brief Blocks until all rank reached this routine
 * \return - when success, -1 when failure happens
 */
int MXBarrier();

}  // namespace kvstore
}  // namespace mxnet
#endif
#endif  // MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLLECTIVES_H_
