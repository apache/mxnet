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
#ifndef MXNET_CYTHON_CYTHON_UTIL_H_
#define MXNET_CYTHON_CYTHON_UTIL_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

int CythonPrintFromCPP(const char *foo);
int Printf(const char *fmt, ...);
int TrivialCPPCall(int var);
uint64_t TimeInMilliseconds();

#ifdef __cplusplus
}
#endif  // __cplusplus

namespace shapes {

class Rectangle {
 public:
  int x0, y0, x1, y1;
  Rectangle();
  Rectangle(int x0, int y0, int x1, int y1);
  ~Rectangle();
  int getArea();
  void getSize(int* width, int* height);
  void move(int dx, int dy);
};

}  // namespace shapes

#endif  // MXNET_CYTHON_CYTHON_UTIL_H_
