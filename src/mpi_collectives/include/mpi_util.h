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

#ifndef MXNET_MPI_COLLECTIVES_INCLUDE_MPI_UTIL_H_
#define MXNET_MPI_COLLECTIVES_INCLUDE_MPI_UTIL_H_

#if MXNET_USE_MPI_DIST_KVSTORE


#include <stdio.h>
#include <vector>

#define DEBUG_ON 0

#if DEBUG_ON
#define MXMPI_DEBUG(rank, fmt, args...)  printf("rank[%d]:" fmt, rank, ## args)
#else
#define MXMPI_DEBUG(fmt, args...)
#endif

/****************************************************
 * The function is used to locate the index of the element
 * in the all equivalent elements.
 *
 * e.g.
 * vec = { 1,6,3,3,1,2,3 }
 *                     ^
 * countNth(vec, 3, 6) = 3
 * vec = { 1,6,3,3,1,2,3 }
 *                 ^
 * countNth(vec, 1, 4) = 2
 ***************************************************/
template <typename T>
size_t countNth(const std::vector<T> &vec,
                const T &key,
                size_t endIdx) {
  size_t curIdx = 0;
  size_t count = 0;
  for (auto &value : vec) {
    if (curIdx > endIdx) break;
    if (value == key) count++;
    curIdx++;
  }
  return count;
}

#endif
#endif  // MXNET_MPI_COLLECTIVES_INCLUDE_MPI_UTIL_H_
