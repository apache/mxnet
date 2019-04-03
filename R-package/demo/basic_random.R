# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require(mxnet)

mx.set.seed(10)

print(mx.runif(c(2,2), -10, 10))

# Test initialization module for neural nets.
uinit <- mx.init.uniform(0.1)
print(uinit("fc1_weight", c(2, 2), mx.cpu()))
print(uinit("fc1_gamma", c(2, 2), mx.cpu()))
