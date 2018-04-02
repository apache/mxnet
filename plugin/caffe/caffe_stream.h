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
 * Copyright (c) 2016 by Contributors
 * \file caffe_stream.h
 * \brief define stream opertors >> and <<
 * \author Haoran Wang
*/
#ifndef PLUGIN_CAFFE_CAFFE_STREAM_H_
#define PLUGIN_CAFFE_CAFFE_STREAM_H_

#include<caffe/proto/caffe.pb.h>
#include<iostream>
namespace dmlc {
namespace parameter {
  std::istringstream &operator>>(std::istringstream &is, ::caffe::LayerParameter &para_);
  std::ostream &operator<<(std::ostream &os, ::caffe::LayerParameter &para_);
}
}

#endif  // PLUGIN_CAFFE_CAFFE_STREAM_H_
