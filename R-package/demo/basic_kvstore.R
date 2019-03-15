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

kv <- mx.kv.create()

dlist <- lapply(1:3, function(i) {
  x = as.array(c(i, i + 1))
  mat = mx.nd.array(x, mx.cpu(i))
  list(x = mat)
})
kv$init(c(0), dlist[[1]])
kv$push(c(0), dlist, 0)
kv$pull(c(0), dlist, 0)

print(as.array(dlist[[1]][[1]]))
