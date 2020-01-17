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
 *  Copyright (c) 2015 by Contributors
 * \file dataloader.cc
 * \brief Pure c++ backed dataloader implementation
 */
#include <mxnet/io.h>

namespace mxnet {
namespace io {

template<typename DType = real_t>
class ThreadedDataLoader : public IIterator<DataBatch> {
 public:
  ThreadedDataLoader(){ }
  // destructor
  virtual ~ThreadedDataLoader(void) {
  }
  // constructor
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    // param_.InitAllowUnknown(kwargs);
  }
  // before first
  virtual void BeforeFirst(void) {
  }

  virtual bool Next(void) {
    
  }

  virtual const DataBatch &Value(void) const {
  }

 private:
  
};  // class ThreadedDataLoader
}  // namespace io
}  // namespace mxnet