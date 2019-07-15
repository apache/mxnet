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
 * Copyright (c) 2015 by Contributors
 * \file mxnet_acc.h
 * \brief Accelerator APIs to interact with accelerator libraries
 */
#ifndef MXNET_MXNET_ACC_H_
#define MXNET_MXNET_ACC_H_

#include <cstdlib>
#include "../dlpack/dlpack.h"

#define GETACCNAME_STR "getAccName"
typedef void (*getAccName_t)(char*);

extern "C" {
    /*
    Function: getAccName
    Parameters:
    - char* : short accelerator name given by acc library
    Returns:
    */
    void getAccName(char*);

    /*
    Function: getNumAcc
    Parameters:
    Returns: Number of accelerators in the system
    */
    int getNumAcc();

    /*
    Function: initialize
    Parameters:
    - int : MXNet version passed to the acc library
    Returns: Success/Failure code
             Failure code if library cannot be used with given MXNet version
    */
    int initialize(int);

    /*
    Function: supportedOps
    Parameters:
    - const char* : Graph json
    - const char*[] : Data names
    - const DLTensor* : Corresponding data
    - const int : Number of data elements
    - int* : Node/Operator IDs supported by acclerator
    Returns:
    */
    void supportedOps(const char*, const char*[], const DLTensor*,
                      const int, int*);

    /*
    Function: loadModel
    Parameters:
    - const char* : Model ID assigned to subgraph
    - const char* : Graph json
    - const char*[] : Data names
    - const DLTensor* : Corresponding data
    - const int : Number of data elements
    - const int : Accelerator ID in the system
    Returns: Success/Failure code
    */
    int loadModel(const char*, const char*, const char*[],
                  const DLTensor*, const int, const int);

    /*
    Function: unloadModel
    Parameters:
    - const char* : Model ID assigned to subgraph
    Returns:
    */
    void unloadModel(const char*);

    /*
    Function: infer
    Parameters:
    - const char* : Model ID assigned to subgraph
    - const char*[] : Input data names
    - const char*[] : Output data names
    - const DLTensor* : Corresponding Input data
    - DLTensor* : Output data given by accelerator
    - const int : Number of input data elements
    - const int : Number of output data elements
    Returns: Success/Failure code
    */
    int infer(const char*, const char*[], const char*[],
              const DLTensor*, DLTensor*, const int, const int);

    /*
    Function: configure
    Parameters:
    - const char*[] : Input keys
    - const char*[] : Input values
    - const int : Number of input pairs
    - char*[] : Output keys
    - char*[] : Output values
    - int* : Number of output pairs
    */
    int configure(const char*[], const char*[], const int,
                  char*[], char*[], int*);
}
#endif  // MXNET_MXNET_ACC_H_
