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

namespace mxnet {
namespace kvstore {

/*!
 * \brief Get node number.
 * \param ret out param for node number.
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
 * \return 0 when success, -1 when failure happens
 */
int MXCOLLIBInit();

/*!
 * \brief Get the local rank.
 * \param ret out param for local rank.
 * \return 0 when success, -1 when failure happens
 */
int MXGetLocalRank(int *ret);

/*!
 * \brief Do Allreduce across the multi-node.
 * \param keys the list of keys.
 * \param in_values the list of input values.
 * \param out_values the list of output values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllReduce(const std::vector<int> &keys,
                const std::vector<mxnet::NDArray*> &in_values,
                const std::vector<mxnet::NDArray*> &out_values,
                int priority);

/*!
 * \brief Do Allreduce across the multi-node.
 * \param keys the list of keys.
 * \param in_values the list of input values.
 * \param out_values the list of output values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllReduceEx(const std::vector<std::string> &keys,
                  const std::vector<mxnet::NDArray*> &in_values,
                  const std::vector<mxnet::NDArray*> &out_values,
                  int priority);

/*!
 * \brief Broadcast values in root rank to all other nodes.
 * \param keys the list of keys.
 * \param values the list of values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXBroadcast(const std::vector<int> &keys,
                const std::vector<mxnet::NDArray*> &values,
                int root_rank,
                int priority);

/*!
 * \brief Broadcast values in root rank to all other nodes.
 * \param keys the list of keys.
 * \param values the list of values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXBroadcastEx(const std::vector<std::string> &keys,
                  const std::vector<mxnet::NDArray*> &values,
                  int root_rank,
                  int priority);

/*!
 * \brief All gather values in all nodes.
 * \param keys the list of keys.
 * \param values the list of values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllGather(const std::vector<int> &keys,
                const std::vector<mxnet::NDArray*> &values,
                int priority);

/*!
 * \brief All gather values in all nodes.
 * \param keys the list of keys.
 * \param values the list of values.
 * \param priority the priority of the action.
 * \return 0 when success, -1 when failure happens
 */
int MXAllGatherEx(const std::vector<std::string> &keys,
                  const std::vector<mxnet::NDArray*> &values,
                  int priority);

}  // namespace kvstore
}  // namespace mxnet
#endif
#endif  // MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLLECTIVES_H_
