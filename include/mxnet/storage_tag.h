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

/*!
 * Copyright (c) 2019 by Contributors
 * \file storage_tag.h
 * \author Bojian Zheng and Abhishek Tiwari
 * \brief default storage tag used for tracking storage allocations
 */

#ifndef MXNET_STORAGE_TAG_H_
#define MXNET_STORAGE_TAG_H_

#define MXNET_ENABLE_STORAGE_TAGGING 0

#if MXNET_ENABLE_STORAGE_TAGGING
#include <string>

namespace mxnet {

namespace {
/// \brief Given a path, extract the filename (i.e., remove directory).
///        This is equivalent of the following bash command:
/// \code{.sh}
/// basename ${path}
/// \endcode
inline std::string __extract_fname(const std::string& path) {
  std::size_t last_dir_pos = path.find_last_of("/\\");
  if (last_dir_pos == std::string::npos) {
    last_dir_pos = 0;
  }
  return path.substr(last_dir_pos + 1);
}
}  // namespace

/** Default Storage Tag 
 *
 *  If we are not using this default Storage tag, then for each operator
 *   that uses `Storage` and/or `Resource` allocations, we need to make
 *   **intrusive** changes by tagging those allocations inside every
 *   operator, which is apparently a very tedious and non-scalable solution.
 *
 *  Note that since the built-in functions `__builtin_*` are ONLY supported
 *    under GCC, the check on `__GNUG__`
 *    (which is equivalent to `__GNUC__ && __cplusplus`) is added.
 */
#ifdef __GNUG__  // If complied with GCC
#define MXNET_DEFAULT_STORAGE_TAG(tag) std::string(tag) \
    + ":" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__) \
    + ":" + __extract_fname(__builtin_FILE()) \
    + "+" +  std::to_string(__builtin_LINE()) \
    + ":" + __builtin_FUNCTION()
#else  // !__GNUG__
#define MXNET_DEFAULT_STORAGE_TAG(tag) std::string(tag) \
    + ":" + __extract_fname(__FILE__) \
    + "+" +  std::to_string(__LINE__)
#endif  // __GNUG__

}  // namespace mxnet

#endif  // MXNET_ENABLE_STORAGE_TAGGING
#endif  // MXNET_STORAGE_TAG_H_
