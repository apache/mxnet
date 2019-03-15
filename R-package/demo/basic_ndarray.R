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

x <- 1:3
mat <- mx.nd.array(x)

mat <- mat + 1.0
mat <- mat + mat
mat <- mat - 5
mat <- 10 / mat
mat <- 7 * mat
mat <- 1 - mat + (2 * mat) / (mat + 0.5)
as.array(mat)

x <- as.array(matrix(1:4, 2, 2))

mx.ctx.default(mx.cpu(1))
print(mx.ctx.default())
print(is.mx.context(mx.cpu()))
mat <- mx.nd.array(x)
mat <- (mat * 3 + 5) / 10
as.array(mat)
