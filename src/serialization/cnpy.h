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
 *
 */

// File is based on https://github.com/leezu/cnpy/tree/libzip released under MIT License
// Copyright (C) 2011  Carl Rogers, 2018 Leonard Lausen

#ifndef MXNET_SERIALIZATION_CNPY_H_
#define MXNET_SERIALIZATION_CNPY_H_

#include <mxnet/ndarray.h>
#include <string>
#include <utility>
#include <vector>

#include "miniz.h"

namespace mxnet {

namespace npy {

void save_array(const std::string& fname, const NDArray& array);
NDArray load_array(const std::string& fname);

}

namespace npz {

void save_array(mz_zip_archive* archive, const std::string& array_name, const NDArray& array);

std::pair<std::vector<NDArray>, std::vector<std::string>>  load_arrays(const std::string& fname);

}
}  // namespace mxnet
#endif  // MXNET_SERIALIZATION_CNPY_H_
