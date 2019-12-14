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
 * This file is used for testing LinearRegressionOutput can
 *   still bind if label is not provided
 */

#include <iostream>
#include <vector>
#include <string>
#include "dmlc/logging.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

int main() {
    LOG(INFO) << "Running LinearRegressionOutput symbol testing, "
                 "executor should be able to bind without label.";
    Symbol data = Symbol::Variable("data");
    Symbol label = Symbol::Variable("regress_label");
    Symbol symbol = LinearRegressionOutput(data, label);
    std::map<std::string, mxnet::cpp::OpReqType> opReqMap;
    for (const auto& iter : symbol.ListArguments()) {
        opReqMap[iter] = mxnet::cpp::OpReqType::kNullOp;
    }
    std::map<std::string, mxnet::cpp::NDArray> argMap({
        {"data", NDArray(Shape{1, 3}, Context::cpu(), true)}
    });

    try {
        symbol.SimpleBind(Context::cpu(),
                argMap,
                std::map<std::string, mxnet::cpp::NDArray>(),
                opReqMap,
                std::map<std::string, mxnet::cpp::NDArray>());
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error binding the symbol: " << MXGetLastError() << " " << e.what();
        throw;
    }
    return 0;
}
